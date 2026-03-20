-- Full Socrata column classification pipeline:
--   1. Load WordNet taxonomy from YAML, embed each category
--   2. Pull columns from Socrata PG, embed them
--   3. Classify: topic (via taxonomy), role (dimension vs measure), data type
--
-- Prerequisites:
--   duckdb -unsigned

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable'
    AS pg (TYPE POSTGRES, SCHEMA 'socrata');

SET allow_unsigned_extensions = true;
LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';
LOAD '/Users/paulharrington/checkouts/blobtemplates/build/duckdb/blobtemplates.duckdb_extension';

SELECT be_load_hf_model('nomic',
    'nomic-ai/nomic-embed-text-v1.5-GGUF',
    'nomic-embed-text-v1.5.Q4_K_M.gguf');

-- ═══════════════════════════════════════════════════════════════════
-- STEP 1: Load and embed WordNet taxonomy
-- ═══════════════════════════════════════════════════════════════════

.print '=== Loading taxonomy from YAML ==='

CREATE OR REPLACE TABLE taxonomy_categories AS
WITH YAML_DOC AS (
    SELECT bt_yaml_to_json(content) AS doc
    FROM read_text('/Users/paulharrington/checkouts/blobembed/data/wordnet_categories.yaml')
),
CATEGORIES AS (
    SELECT unnest(from_json(doc::JSON->'categories', '["json"]')) AS cat
    FROM YAML_DOC
)
SELECT cat->>'synset_id'  AS synset_id,
       cat->>'category'   AS category,
       cat->>'hypernym'   AS hypernym,
       (cat->>'depth')::INTEGER AS depth,
       cat->>'gloss'      AS gloss
FROM CATEGORIES;

SELECT count(*) AS taxonomy_loaded FROM taxonomy_categories;

-- Embed taxonomy categories (category name + gloss for disambiguation)
-- This takes a few minutes for ~5k categories.
.print 'Embedding taxonomy categories...'

ALTER TABLE taxonomy_categories ADD COLUMN vec FLOAT[768];

UPDATE taxonomy_categories
SET vec = be_embed('nomic', category || ': ' || COALESCE(gloss, ''))::FLOAT[768];

.print 'Taxonomy embedded.'

-- ═══════════════════════════════════════════════════════════════════
-- STEP 2: Pick a diverse sample of Socrata tables
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE TEMP TABLE target_tables AS
WITH RANKED AS (
    SELECT rc.domain, rc.resource_id,
           count(*) AS n_cols,
           ROW_NUMBER() OVER (PARTITION BY rc.domain ORDER BY random()) AS rn
    FROM pg.resource_column AS rc
    WHERE rc.tt_end = '9999-12-31'
    GROUP BY rc.domain, rc.resource_id
    HAVING count(*) BETWEEN 8 AND 80
)
SELECT t.domain, t.resource_id, t.n_cols, r.name AS table_name
FROM RANKED AS t
JOIN pg.resource AS r
    ON r.domain = t.domain AND r.resource_id = t.resource_id
    AND r.tt_end = '9999-12-31'
WHERE t.rn <= 5
ORDER BY t.domain, t.n_cols DESC;

.print ''
.print '=== Target tables ==='
SELECT domain, resource_id, table_name, n_cols FROM target_tables;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 3: Embed all columns, classify roles
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Embedding columns ==='

CREATE OR REPLACE TEMP TABLE col_analysis AS
WITH COL_EMBEDS AS (
    SELECT rc.domain, rc.resource_id, rc.field_name,
           rc.data_type AS socrata_type,
           rc.description,
           be_embed('nomic', rc.field_name)::FLOAT[768] AS vec
    FROM pg.resource_column AS rc
    WHERE rc.tt_end = '9999-12-31'
      AND (rc.domain, rc.resource_id) IN (
          SELECT domain, resource_id FROM target_tables
      )
),
-- Avg similarity within each table for dimension/measure classification
AVG_SIMS AS (
    SELECT a.domain, a.resource_id, a.field_name,
           a.socrata_type, a.description, a.vec,
           avg(array_cosine_similarity(a.vec, b.vec)) AS avg_sim
    FROM COL_EMBEDS AS a
    JOIN COL_EMBEDS AS b
        ON a.domain = b.domain AND a.resource_id = b.resource_id
        AND a.field_name != b.field_name
    GROUP BY a.domain, a.resource_id, a.field_name,
             a.socrata_type, a.description, a.vec
)
SELECT *,
       -- Role classification
       CASE WHEN avg_sim < 0.55 THEN 'dimension'
            ELSE 'measure_candidate'
       END AS role,
       -- Medium-level data type from Socrata type
       CASE socrata_type
           WHEN 'Text' THEN 'string'
           WHEN 'Number' THEN 'number'
           WHEN 'Date' THEN 'date'
           WHEN 'Calendar date' THEN 'date'
           WHEN 'Checkbox' THEN 'boolean'
           WHEN 'Point' THEN 'geo'
           WHEN 'Location' THEN 'geo'
           WHEN 'MultiPolygon' THEN 'geo'
           WHEN 'Polygon' THEN 'geo'
           WHEN 'MultiLine' THEN 'geo'
           WHEN 'Line' THEN 'geo'
           WHEN 'MultiPoint' THEN 'geo'
           WHEN 'URL' THEN 'url'
           WHEN 'Email' THEN 'string'
           WHEN 'Phone' THEN 'string'
           WHEN 'Money' THEN 'number'
           WHEN 'Percent' THEN 'number'
           ELSE 'other'
       END AS medium_type
FROM AVG_SIMS;

SELECT count(*) AS columns_analyzed FROM col_analysis;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 4: Match each column against taxonomy for topic
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Matching columns to taxonomy topics ==='

CREATE OR REPLACE TEMP TABLE col_topics AS
SELECT ca.domain, ca.resource_id, ca.field_name,
       ca.socrata_type, ca.medium_type, ca.role,
       ca.description,
       round(ca.avg_sim, 3) AS avg_sim,
       tc.category AS topic,
       tc.hypernym AS topic_parent,
       round(array_cosine_similarity(tc.vec, ca.vec), 3) AS topic_score
FROM col_analysis AS ca,
     taxonomy_categories AS tc
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY ca.domain, ca.resource_id, ca.field_name
    ORDER BY array_cosine_distance(tc.vec, ca.vec)
) = 1;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 5: Per-table profile
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Table profiles ==='

SELECT ct.domain, tt.table_name, tt.n_cols,
       count(*) FILTER (WHERE ct.role = 'dimension') AS n_dim,
       count(*) FILTER (WHERE ct.role = 'measure_candidate') AS n_meas,
       -- Top 3 topics for dimensions
       (SELECT string_agg(DISTINCT t2.topic, ', ' ORDER BY t2.topic)
        FROM (SELECT topic FROM col_topics AS ct2
              WHERE ct2.domain = ct.domain AND ct2.resource_id = ct.resource_id
                AND ct2.role = 'dimension' AND ct2.topic_score > 0.5
              GROUP BY topic ORDER BY count(*) DESC LIMIT 3) AS t2
       ) AS dim_topics,
       -- Top 3 topics for measures
       (SELECT string_agg(DISTINCT t2.topic, ', ' ORDER BY t2.topic)
        FROM (SELECT topic FROM col_topics AS ct2
              WHERE ct2.domain = ct.domain AND ct2.resource_id = ct.resource_id
                AND ct2.role = 'measure_candidate' AND ct2.topic_score > 0.5
              GROUP BY topic ORDER BY count(*) DESC LIMIT 3) AS t2
       ) AS meas_topics,
       -- Data type breakdown
       count(*) FILTER (WHERE ct.medium_type = 'string') AS n_string,
       count(*) FILTER (WHERE ct.medium_type = 'number') AS n_number,
       count(*) FILTER (WHERE ct.medium_type = 'date') AS n_date,
       count(*) FILTER (WHERE ct.medium_type = 'geo') AS n_geo,
       count(*) FILTER (WHERE ct.medium_type = 'boolean') AS n_bool
FROM col_topics AS ct
JOIN target_tables AS tt USING (domain, resource_id)
GROUP BY ct.domain, ct.resource_id, tt.table_name, tt.n_cols
ORDER BY tt.n_cols DESC;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 6: Detailed column classification for a few tables
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Sample: detailed column classification ==='
.print '(showing first 3 tables)'

SELECT ct.domain, tt.table_name, ct.field_name,
       ct.role, ct.medium_type, ct.topic, ct.topic_parent,
       ct.topic_score
FROM col_topics AS ct
JOIN target_tables AS tt USING (domain, resource_id)
WHERE (ct.domain, ct.resource_id) IN (
    SELECT domain, resource_id FROM target_tables LIMIT 3
)
ORDER BY ct.domain, tt.table_name, ct.role, ct.topic_score DESC;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 7: Cross-table topic similarity
-- Which tables are about similar things?
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Cross-table similarity (by table-level embedding) ==='

CREATE OR REPLACE TEMP TABLE table_embeds AS
SELECT rc.domain, rc.resource_id, tt.table_name,
       be_embed('nomic',
           rc.domain || '.' || tt.table_name || ': '
           || string_agg(rc.field_name || ' (' || rc.data_type || ')',
                         ', ' ORDER BY rc.ordinal_position)
       )::FLOAT[768] AS vec
FROM pg.resource_column AS rc
JOIN target_tables AS tt USING (domain, resource_id)
WHERE rc.tt_end = '9999-12-31'
GROUP BY rc.domain, rc.resource_id, tt.table_name;

SELECT a.table_name AS table_a, b.table_name AS table_b,
       round(array_cosine_similarity(a.vec, b.vec), 3) AS sim
FROM table_embeds AS a, table_embeds AS b
WHERE a.domain || a.resource_id < b.domain || b.resource_id
ORDER BY sim DESC
LIMIT 15;
