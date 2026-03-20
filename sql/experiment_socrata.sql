-- Socrata embedding experiments: column matching, table matching, pivot detection.
--
-- Compares search quality across strategies:
--   A. Single column name alone
--   B. Column name with full path (domain.resource.column)
--   C. All columns of a table as a group
--   D. Semantically clustered sub-groups of columns
--
-- Prerequisites:
--   duckdb -unsigned
--   INSTALL postgres; LOAD postgres;
--   LOAD 'blobembed.duckdb_extension';
--   SELECT be_load_hf_model('nomic', ...);

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable'
    AS pg (TYPE POSTGRES, SCHEMA 'socrata');

SET allow_unsigned_extensions = true;
LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';

SELECT be_load_hf_model('nomic',
    'nomic-ai/nomic-embed-text-v1.5-GGUF',
    'nomic-embed-text-v1.5.Q4_K_M.gguf');

-- ═══════════════════════════════════════════════════════════════════
-- STEP 1: Build the embedding catalog from Socrata metadata
-- ═══════════════════════════════════════════════════════════════════

-- 1A: Column-level embeddings (just the column name)
CREATE OR REPLACE TABLE exp_col_embed_bare AS
SELECT rc.domain, rc.resource_id, rc.field_name,
       be_embed('nomic', rc.field_name)::FLOAT[768] AS vec
FROM pg.resource_column AS rc
WHERE rc.tt_end = '9999-12-31'
  AND rc.domain = 'data.cityofnewyork.us'
  -- sample: limit to a manageable subset
  AND rc.resource_id IN (
      SELECT resource_id
      FROM pg.resource_column
      WHERE tt_end = '9999-12-31' AND domain = 'data.cityofnewyork.us'
      GROUP BY resource_id
      HAVING count(*) BETWEEN 10 AND 60
      LIMIT 50
  );

-- 1B: Column-level embeddings with path context
CREATE OR REPLACE TABLE exp_col_embed_path AS
SELECT rc.domain, rc.resource_id, rc.field_name,
       be_embed('nomic',
           rc.domain || '.' || r.name || '.' || rc.field_name
           || CASE WHEN rc.description IS NOT NULL AND rc.description != ''
                   THEN ' (' || rc.description || ')'
                   ELSE '' END
       )::FLOAT[768] AS vec
FROM pg.resource_column AS rc
JOIN pg.resource AS r
    ON r.domain = rc.domain AND r.resource_id = rc.resource_id
    AND r.tt_end = '9999-12-31'
WHERE rc.tt_end = '9999-12-31'
  AND rc.domain = 'data.cityofnewyork.us'
  AND rc.resource_id IN (SELECT DISTINCT resource_id FROM exp_col_embed_bare);

-- 1C: Table-level embeddings (all column names as a group)
CREATE OR REPLACE TABLE exp_table_embed AS
SELECT rc.domain, rc.resource_id, r.name AS table_name,
       be_embed('nomic',
           rc.domain || '.' || r.name || ': '
           || string_agg(
               rc.field_name || ' (' || rc.data_type || ')',
               ', ' ORDER BY rc.ordinal_position)
       )::FLOAT[768] AS vec
FROM pg.resource_column AS rc
JOIN pg.resource AS r
    ON r.domain = rc.domain AND r.resource_id = rc.resource_id
    AND r.tt_end = '9999-12-31'
WHERE rc.tt_end = '9999-12-31'
  AND rc.domain = 'data.cityofnewyork.us'
  AND rc.resource_id IN (SELECT DISTINCT resource_id FROM exp_col_embed_bare)
GROUP BY rc.domain, rc.resource_id, r.name;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 2: Experiment A — Column name alone
-- Search: given a column name, find similar columns across all tables
-- ═══════════════════════════════════════════════════════════════════

-- Probe: "borough" — should match borough-like columns in NYC data
.print '=== Experiment A: bare column name search ==='
.print 'Probe: "borough"'
SELECT e.domain, e.resource_id, e.field_name,
       array_cosine_similarity(e.vec, be_embed('nomic', 'borough')::FLOAT[768]) AS score
FROM exp_col_embed_bare AS e
ORDER BY score DESC
LIMIT 10;

-- Probe: "latitude" — should find lat/lon/location columns
.print 'Probe: "latitude"'
SELECT e.domain, e.resource_id, e.field_name,
       array_cosine_similarity(e.vec, be_embed('nomic', 'latitude')::FLOAT[768]) AS score
FROM exp_col_embed_bare AS e
ORDER BY score DESC
LIMIT 10;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 3: Experiment B — Column name with path context
-- Same probes, but now the stored embeddings include path + description
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Experiment B: column with path context ==='
.print 'Probe: "borough" (enriched with dataset context)'
SELECT e.domain, e.resource_id, e.field_name,
       array_cosine_similarity(e.vec,
           be_embed('nomic', 'a NYC dataset about locations with a borough column')::FLOAT[768]
       ) AS score
FROM exp_col_embed_path AS e
ORDER BY score DESC
LIMIT 10;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 4: Experiment C — Table-level search
-- Given a set of column names, find the most similar table
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Experiment C: table-level search ==='
.print 'Probe: "permit_number, job_type, borough, block, lot, community_board, filing_date"'
SELECT e.domain, e.resource_id, e.table_name,
       array_cosine_similarity(e.vec,
           be_embed('nomic',
               'permit_number, job_type, borough, block, lot, community_board, filing_date'
           )::FLOAT[768]
       ) AS score
FROM exp_table_embed AS e
ORDER BY score DESC
LIMIT 10;

.print 'Probe: "first_name, last_name, agency, title, salary, start_date"'
SELECT e.domain, e.resource_id, e.table_name,
       array_cosine_similarity(e.vec,
           be_embed('nomic',
               'first_name, last_name, agency, title, salary, start_date'
           )::FLOAT[768]
       ) AS score
FROM exp_table_embed AS e
ORDER BY score DESC
LIMIT 10;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 5: Experiment D — Intra-table column clustering
-- For a specific table, cluster columns by embedding similarity
-- and identify potential pivot groups
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Experiment D: column clustering within NYC Greenhouse Gas table ==='

-- Pairwise similarity for the GHG table columns
CREATE OR REPLACE TEMP TABLE ghg_cols AS
SELECT rc.field_name,
       be_embed('nomic', rc.field_name)::FLOAT[768] AS vec
FROM pg.resource_column AS rc
WHERE rc.domain = 'data.cityofnewyork.us'
  AND rc.resource_id = 'wq7q-htne'
  AND rc.tt_end = '9999-12-31';

-- Show the top pairwise similarities (clusters will emerge)
.print 'Top 20 most similar column pairs within NYC GHG Emissions:'
SELECT a.field_name AS col_a, b.field_name AS col_b,
       array_cosine_similarity(a.vec, b.vec) AS sim
FROM ghg_cols AS a, ghg_cols AS b
WHERE a.field_name < b.field_name
ORDER BY sim DESC
LIMIT 20;

-- Dimension columns: which columns are LEAST similar to the majority?
-- (Low average similarity to all other columns = likely a dimension, not a pivoted measure)
.print ''
.print 'Columns by avg similarity to all others (low = likely dimension):'
SELECT a.field_name,
       avg(array_cosine_similarity(a.vec, b.vec)) AS avg_sim
FROM ghg_cols AS a, ghg_cols AS b
WHERE a.field_name != b.field_name
GROUP BY a.field_name
ORDER BY avg_sim ASC
LIMIT 15;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 6: Summary stats
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Embedding catalog stats ==='
SELECT 'bare_col_embeds' AS what, count(*) AS n FROM exp_col_embed_bare
UNION ALL
SELECT 'path_col_embeds', count(*) FROM exp_col_embed_path
UNION ALL
SELECT 'table_embeds', count(*) FROM exp_table_embed;
