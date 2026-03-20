INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable'
    AS pg (TYPE POSTGRES, SCHEMA 'socrata');

SET allow_unsigned_extensions = true;
LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';

SELECT be_load_hf_model('nomic',
    'nomic-ai/nomic-embed-text-v1.5-GGUF',
    'nomic-embed-text-v1.5.Q4_K_M.gguf');

-- Classify columns by embedding similarity
CREATE OR REPLACE TEMP TABLE ghg_classified AS
WITH COL_EMBEDS AS (
    SELECT rc.field_name, rc.data_type,
           be_embed('nomic', rc.field_name)::FLOAT[768] AS vec
    FROM pg.resource_column AS rc
    WHERE rc.domain = 'data.cityofnewyork.us'
      AND rc.resource_id = 'wq7q-htne'
      AND rc.tt_end = '9999-12-31'
),
AVG_SIMS AS (
    SELECT a.field_name, a.data_type,
           avg(array_cosine_similarity(a.vec, b.vec)) AS avg_sim
    FROM COL_EMBEDS AS a, COL_EMBEDS AS b
    WHERE a.field_name != b.field_name
    GROUP BY a.field_name, a.data_type
)
SELECT field_name, data_type, avg_sim,
       CASE WHEN avg_sim < 0.55 THEN 'dimension' ELSE 'measure_candidate' END AS role
FROM AVG_SIMS;

-- Factor the measure columns
CREATE OR REPLACE TEMP TABLE factored AS
SELECT field_name,
       regexp_extract(field_name, '^cy_(\d{4})_', 1) AS year,
       regexp_extract(field_name, '^cy_\d{4}_(.*)', 1) AS measure,
       regexp_extract(field_name, '^_(\d{4})_\d{4}_change_', 1) AS change_from,
       regexp_extract(field_name, '^_\d{4}_(\d{4})_change_', 1) AS change_to,
       regexp_extract(field_name, '^_\d{4}_\d{4}_change_(.*)', 1) AS change_measure
FROM ghg_classified
WHERE role = 'measure_candidate';

-- Summary: distinct measures with year counts
.print '=== Measures detected ==='
SELECT measure AS measure_name,
       min(year::INTEGER) AS from_year,
       max(year::INTEGER) AS to_year,
       count(*) AS n_years
FROM factored
WHERE measure IS NOT NULL AND measure != ''
GROUP BY measure
ORDER BY n_years DESC;

-- Change columns
.print ''
.print '=== Change/delta columns ==='
SELECT change_measure AS measure_name,
       change_from || ' to ' || change_to AS period,
       field_name
FROM factored
WHERE change_measure IS NOT NULL AND change_measure != ''
ORDER BY change_measure;

-- Dimensions
.print ''
.print '=== Dimensions ==='
SELECT field_name, data_type, round(avg_sim, 3) AS avg_sim
FROM ghg_classified WHERE role = 'dimension'
ORDER BY avg_sim;

-- Unmatched (neither pattern matched)
.print ''
.print '=== Unmatched measure candidates ==='
SELECT field_name
FROM factored
WHERE (measure IS NULL OR measure = '')
  AND (change_measure IS NULL OR change_measure = '');

-- Final recipe as structured output
.print ''
.print '=== UNPIVOT RECIPE ==='
.print 'Normalized schema:'
.print '  sector         TEXT    (dimension)'
.print '  category_label TEXT    (dimension)'
.print '  source_label   TEXT    (dimension)'
.print '  source_units   TEXT    (dimension)'
.print '  inventory_type TEXT    (dimension)'
.print '  year           INTEGER (unpivoted from cy_YYYY_ prefix)'
.print '  consumed       NUMBER  (measure)'
.print '  source_mmbtu   NUMBER  (measure)'
.print '  tco2e          NUMBER  (measure)'
.print '  tco2e_20_yr_gwp  NUMBER  (measure)'
.print '  tco2e_100_yr_gwp NUMBER  (measure)'
.print ''
.print 'SQL to unpivot:'
SELECT 'UNPIVOT ghg_data ON '
    || string_agg(DISTINCT 'cy_*_' || measure, ', ')
    || ' INTO NAME year VALUE '
    || string_agg(DISTINCT measure, ', ')
    || ';' AS unpivot_hint
FROM factored
WHERE measure IS NOT NULL AND measure != '';
