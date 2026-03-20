LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable' AS pg (TYPE POSTGRES);

SELECT be_load_hf_model('miniLM',
    'second-state/All-MiniLM-L6-v2-Embedding-GGUF', 'all-MiniLM-L6-v2-Q8_0.gguf');
SELECT be_load_hf_model('nomic',
    'nomic-ai/nomic-embed-text-v1.5-GGUF', 'nomic-embed-text-v1.5.Q4_K_M.gguf');

-- Embed with miniLM (384-dim) → use FLOAT[384] for array_cosine_similarity
CREATE TEMP TABLE ghg_mini AS
SELECT lower(rc.field_name) AS col,
       be_embed('miniLM', lower(rc.field_name))::FLOAT[384] AS vec
FROM pg.socrata.resource_column AS rc
WHERE rc.domain = 'data.cityofnewyork.us'
  AND rc.resource_id = 'wq7q-htne' AND rc.tt_end = '9999-12-31';

-- Embed with nomic (768-dim)
CREATE TEMP TABLE ghg_nomic AS
SELECT lower(rc.field_name) AS col,
       be_embed('nomic', lower(rc.field_name))::FLOAT[768] AS vec
FROM pg.socrata.resource_column AS rc
WHERE rc.domain = 'data.cityofnewyork.us'
  AND rc.resource_id = 'wq7q-htne' AND rc.tt_end = '9999-12-31';

-- Avg sim per column for each model
CREATE TEMP TABLE ghg_comparison AS
SELECT m.col, m.avg_sim_mini, n.avg_sim_nomic
FROM (
    SELECT a.col, avg(array_cosine_similarity(a.vec, b.vec)) AS avg_sim_mini
    FROM ghg_mini AS a, ghg_mini AS b WHERE a.col != b.col GROUP BY a.col
) AS m
JOIN (
    SELECT a.col, avg(array_cosine_similarity(a.vec, b.vec)) AS avg_sim_nomic
    FROM ghg_nomic AS a, ghg_nomic AS b WHERE a.col != b.col GROUP BY a.col
) AS n USING (col);

.print '=== Distribution comparison ==='
SELECT 'miniLM' AS model,
       round(min(avg_sim_mini), 3) AS p_min,
       round(percentile_cont(0.10) WITHIN GROUP (ORDER BY avg_sim_mini), 3) AS p10,
       round(percentile_cont(0.25) WITHIN GROUP (ORDER BY avg_sim_mini), 3) AS p25,
       round(percentile_cont(0.50) WITHIN GROUP (ORDER BY avg_sim_mini), 3) AS p50,
       round(percentile_cont(0.75) WITHIN GROUP (ORDER BY avg_sim_mini), 3) AS p75,
       round(max(avg_sim_mini), 3) AS p_max
FROM ghg_comparison
UNION ALL
SELECT 'nomic',
       round(min(avg_sim_nomic), 3),
       round(percentile_cont(0.10) WITHIN GROUP (ORDER BY avg_sim_nomic), 3),
       round(percentile_cont(0.25) WITHIN GROUP (ORDER BY avg_sim_nomic), 3),
       round(percentile_cont(0.50) WITHIN GROUP (ORDER BY avg_sim_nomic), 3),
       round(percentile_cont(0.75) WITHIN GROUP (ORDER BY avg_sim_nomic), 3),
       round(max(avg_sim_nomic), 3)
FROM ghg_comparison;

.print ''
.print '=== Side-by-side: lowest avg_sim columns ==='
SELECT col,
       round(avg_sim_nomic, 3) AS nomic,
       round(avg_sim_mini, 3) AS mini,
       CASE WHEN avg_sim_nomic < 0.55 THEN 'DIM' ELSE 'meas' END AS nomic_role
FROM ghg_comparison
ORDER BY avg_sim_nomic ASC
LIMIT 20;

.print ''
.print '=== Highest avg_sim columns ==='
SELECT col,
       round(avg_sim_nomic, 3) AS nomic,
       round(avg_sim_mini, 3) AS mini
FROM ghg_comparison
ORDER BY avg_sim_nomic DESC
LIMIT 10;
