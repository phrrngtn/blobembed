INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable'
    AS pg (TYPE POSTGRES, SCHEMA 'socrata');

SET allow_unsigned_extensions = true;
LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';

SELECT be_load_hf_model('nomic',
    'nomic-ai/nomic-embed-text-v1.5-GGUF',
    'nomic-embed-text-v1.5.Q4_K_M.gguf');

-- ═══════════════════════════════════════════════════════════════════
-- Pick 10 diverse wide tables across all domains
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE TEMP TABLE target_tables AS
WITH RANKED AS (
    SELECT rc.domain, rc.resource_id,
           count(*) AS n_cols,
           count(*) FILTER (WHERE rc.data_type = 'Number') AS n_numeric,
           count(*) FILTER (WHERE rc.data_type = 'Text') AS n_text,
           ROW_NUMBER() OVER (PARTITION BY rc.domain ORDER BY count(*) DESC) AS rn
    FROM pg.resource_column AS rc
    WHERE rc.tt_end = '9999-12-31'
    GROUP BY rc.domain, rc.resource_id
    HAVING count(*) BETWEEN 30 AND 200
)
SELECT t.domain, t.resource_id, t.n_cols, t.n_numeric, t.n_text, r.name
FROM RANKED AS t
JOIN pg.resource AS r
    ON r.domain = t.domain AND r.resource_id = t.resource_id
    AND r.tt_end = '9999-12-31'
WHERE t.rn <= 3  -- top 3 widest per domain
ORDER BY t.n_cols DESC
LIMIT 12;

.print '=== Target tables ==='
SELECT domain, resource_id, name, n_cols, n_numeric, n_text FROM target_tables;

-- ═══════════════════════════════════════════════════════════════════
-- For each table: embed columns, classify, factor
-- ═══════════════════════════════════════════════════════════════════

-- Embed all columns from all target tables
CREATE OR REPLACE TEMP TABLE all_col_embeds AS
SELECT rc.domain, rc.resource_id, rc.field_name, rc.data_type,
       be_embed('nomic', rc.field_name)::FLOAT[768] AS vec
FROM pg.resource_column AS rc
WHERE rc.tt_end = '9999-12-31'
  AND (rc.domain, rc.resource_id) IN (
      SELECT domain, resource_id FROM target_tables
  );

.print ''
.print '=== Embedding complete ==='
SELECT count(*) AS total_columns_embedded FROM all_col_embeds;

-- Compute per-column avg similarity within each table
CREATE OR REPLACE TEMP TABLE col_classified AS
WITH AVG_SIMS AS (
    SELECT a.domain, a.resource_id, a.field_name, a.data_type,
           avg(array_cosine_similarity(a.vec, b.vec)) AS avg_sim
    FROM all_col_embeds AS a
    JOIN all_col_embeds AS b
        ON a.domain = b.domain AND a.resource_id = b.resource_id
        AND a.field_name != b.field_name
    GROUP BY a.domain, a.resource_id, a.field_name, a.data_type
)
SELECT *,
       CASE WHEN avg_sim < 0.55 THEN 'dimension'
            ELSE 'measure_candidate'
       END AS role
FROM AVG_SIMS;

-- ═══════════════════════════════════════════════════════════════════
-- Per-table summary: how many dimensions vs measures?
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Per-table dimension/measure split ==='
SELECT cc.domain, cc.resource_id, tt.name,
       tt.n_cols,
       count(*) FILTER (WHERE cc.role = 'dimension') AS n_dim,
       count(*) FILTER (WHERE cc.role = 'measure_candidate') AS n_meas,
       round(min(cc.avg_sim) FILTER (WHERE cc.role = 'dimension'), 3) AS dim_min_sim,
       round(max(cc.avg_sim) FILTER (WHERE cc.role = 'dimension'), 3) AS dim_max_sim,
       round(min(cc.avg_sim) FILTER (WHERE cc.role = 'measure_candidate'), 3) AS meas_min_sim,
       round(max(cc.avg_sim) FILTER (WHERE cc.role = 'measure_candidate'), 3) AS meas_max_sim
FROM col_classified AS cc
JOIN target_tables AS tt USING (domain, resource_id)
GROUP BY cc.domain, cc.resource_id, tt.name, tt.n_cols
ORDER BY tt.n_cols DESC;

-- ═══════════════════════════════════════════════════════════════════
-- String factoring: detect temporal patterns in column names
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE TEMP TABLE temporal_factors AS
SELECT domain, resource_id, field_name,
       -- cy_YYYY_ or fy_YYYY_ or just _YYYY_ prefix
       regexp_extract(field_name, '(?:^|_)((?:19|20)\d{2})(?:_|$)', 1) AS year_str,
       -- Everything except the year part = the measure template
       regexp_replace(field_name, '(?:^|_)((?:19|20)\d{2})(?:_|$)', '_YYYY_') AS template
FROM col_classified
WHERE role = 'measure_candidate';

-- Which tables have temporal patterns?
.print ''
.print '=== Temporal patterns detected ==='
SELECT tf.domain, tf.resource_id, tt.name,
       count(*) FILTER (WHERE tf.year_str IS NOT NULL AND tf.year_str != '') AS n_temporal,
       count(*) FILTER (WHERE tf.year_str IS NULL OR tf.year_str = '') AS n_non_temporal,
       count(DISTINCT tf.year_str) FILTER (WHERE tf.year_str IS NOT NULL AND tf.year_str != '') AS n_distinct_years,
       count(DISTINCT tf.template) FILTER (WHERE tf.year_str IS NOT NULL AND tf.year_str != '') AS n_distinct_measures
FROM temporal_factors AS tf
JOIN target_tables AS tt USING (domain, resource_id)
GROUP BY tf.domain, tf.resource_id, tt.name
ORDER BY n_temporal DESC;

-- Detail: what templates were found for tables with temporal patterns?
.print ''
.print '=== Temporal measure templates (tables with 5+ temporal columns) ==='
SELECT tf.domain, tt.name,
       regexp_replace(tf.template, '_YYYY_', '{year}') AS measure_template,
       count(*) AS n_year_columns,
       min(tf.year_str::INTEGER) AS from_year,
       max(tf.year_str::INTEGER) AS to_year,
       string_agg(DISTINCT tf.year_str, ', ' ORDER BY tf.year_str) AS years
FROM temporal_factors AS tf
JOIN target_tables AS tt USING (domain, resource_id)
WHERE tf.year_str IS NOT NULL AND tf.year_str != ''
GROUP BY tf.domain, tt.name, tf.template
HAVING count(*) >= 2
ORDER BY tf.domain, tt.name, n_year_columns DESC;

-- ═══════════════════════════════════════════════════════════════════
-- Semantic clustering: for non-temporal measure candidates,
-- find clusters of similar column names
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Semantic clusters in non-temporal columns ==='

-- For each table, find tight pairwise groups (sim > 0.80)
-- among measure candidates that weren't temporal
WITH NON_TEMPORAL AS (
    SELECT ce.domain, ce.resource_id, ce.field_name, ce.vec
    FROM all_col_embeds AS ce
    JOIN col_classified AS cc USING (domain, resource_id, field_name)
    JOIN temporal_factors AS tf USING (domain, resource_id, field_name)
    WHERE cc.role = 'measure_candidate'
      AND (tf.year_str IS NULL OR tf.year_str = '')
),
PAIRS AS (
    SELECT a.domain, a.resource_id,
           a.field_name AS col_a, b.field_name AS col_b,
           array_cosine_similarity(a.vec, b.vec) AS sim
    FROM NON_TEMPORAL AS a
    JOIN NON_TEMPORAL AS b
        ON a.domain = b.domain AND a.resource_id = b.resource_id
        AND a.field_name < b.field_name
    WHERE array_cosine_similarity(a.vec, b.vec) > 0.80
)
SELECT p.domain, tt.name, p.col_a, p.col_b, round(p.sim, 3) AS sim
FROM PAIRS AS p
JOIN target_tables AS tt USING (domain, resource_id)
ORDER BY p.sim DESC
LIMIT 30;

-- ═══════════════════════════════════════════════════════════════════
-- Show dimension columns per table (the interesting ones)
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Dimension columns per table ==='
SELECT cc.domain, tt.name, cc.field_name, cc.data_type,
       round(cc.avg_sim, 3) AS avg_sim
FROM col_classified AS cc
JOIN target_tables AS tt USING (domain, resource_id)
WHERE cc.role = 'dimension'
ORDER BY cc.domain, tt.name, cc.avg_sim;
