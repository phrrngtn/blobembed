-- Load WordNet taxonomy categories, compute embeddings, and create HNSW index.
--
-- Prerequisites:
--   SET allow_unsigned_extensions=true;
--   LOAD 'path/to/blobembed.duckdb_extension';
--   LOAD 'path/to/blobtemplates.duckdb_extension';  -- for bt_yaml_to_json
--   INSTALL vss; LOAD vss;
--
--   SELECT be_load_hf_model('nomic',
--       'nomic-ai/nomic-embed-text-v1.5-GGUF',
--       'nomic-embed-text-v1.5.Q4_K_M.gguf');
--
-- Usage:
--   .read sql/load_taxonomy.sql

-- ── Step 1: Parse YAML into a table ────────────────────────────────
CREATE OR REPLACE TABLE taxonomy_raw AS
WITH YAML_DOC AS (
    SELECT bt_yaml_to_json(content) AS doc
    FROM read_text('data/wordnet_categories.yaml')
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

-- ── Step 2: Compute embeddings ─────────────────────────────────────
-- Embed the category name enriched with its gloss for better discrimination.
-- "fruit: the ripened reproductive body of a seed plant" embeds better than
-- just "fruit" because the gloss disambiguates polysemous words.
CREATE OR REPLACE TABLE taxonomy_categories AS
SELECT synset_id,
       category,
       hypernym,
       depth,
       gloss,
       be_embed('nomic', category || ': ' || gloss)::FLOAT[768] AS vec
FROM taxonomy_raw;

DROP TABLE taxonomy_raw;

-- ── Step 3: Create HNSW index for fast nearest-category lookup ─────
CREATE INDEX IF NOT EXISTS idx_taxonomy_cosine
    ON taxonomy_categories USING HNSW (vec)
    WITH (metric = 'cosine');

-- ── Verify ─────────────────────────────────────────────────────────
SELECT count(*) AS n_categories FROM taxonomy_categories;

-- ── Example: what category is "banana" closest to? ─────────────────
SELECT category, hypernym, depth,
       array_cosine_similarity(vec, be_embed('nomic', 'banana')::FLOAT[768]) AS score
FROM taxonomy_categories
ORDER BY array_cosine_distance(vec, be_embed('nomic', 'banana')::FLOAT[768])
LIMIT 5;
