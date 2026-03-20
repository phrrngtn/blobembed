-- Table topic decomposition: compress a table's columns into semantic topics.
--
-- Pipeline:
--   1. blobfilters strips known enumerations (months, days, etc.)
--   2. Remaining columns embedded and clustered by pairwise similarity
--   3. Each cluster labelled via taxonomy centroid matching
--   4. Output: table → topic vector (searchable, comparable)
--
-- Prerequisites:
--   duckdb -unsigned
--   PG attached with domain.enumeration populated
--   blobembed, blobfilters, blobtemplates extensions loaded
--   nomic model loaded

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable'
    AS pg (TYPE POSTGRES);

-- Convenience aliases for the two schemas we need
CREATE OR REPLACE VIEW socrata_resource AS SELECT * FROM pg.socrata.resource;
CREATE OR REPLACE VIEW socrata_resource_column AS SELECT * FROM pg.socrata.resource_column;

LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';
LOAD '/Users/paulharrington/checkouts/blobfilters/build/duckdb/blobfilters.duckdb_extension';
LOAD '/Users/paulharrington/checkouts/blobtemplates/build/duckdb/blobtemplates.duckdb_extension';

SELECT be_load_hf_model('nomic',
    'nomic-ai/nomic-embed-text-v1.5-GGUF',
    'nomic-embed-text-v1.5.Q4_K_M.gguf');

-- ── Prefetch domain filters from PG ────────────────────────────────
CREATE OR REPLACE TABLE domain_filters AS
SELECT domain_name, domain_label,
       bf_from_base64(filter_b64) AS filter
FROM pg.domain.enumeration
WHERE filter_b64 IS NOT NULL;

-- ── Load taxonomy embeddings (once per session) ────────────────────
CREATE OR REPLACE TABLE taxonomy_categories AS
WITH YAML_DOC AS (
    SELECT bt_yaml_to_json(content) AS doc
    FROM read_text('/Users/paulharrington/checkouts/blobembed/data/wordnet_categories.yaml')
),
CATEGORIES AS (
    SELECT unnest(from_json(doc::JSON->'categories', '["json"]')) AS cat
    FROM YAML_DOC
)
SELECT cat->>'category' AS category,
       cat->>'hypernym' AS hypernym,
       be_embed('nomic', (cat->>'category') || ': ' || COALESCE(cat->>'gloss', ''))::FLOAT[768] AS vec
FROM CATEGORIES;

.print '=== Setup complete ==='
SELECT count(*) AS n_domains FROM domain_filters;
SELECT count(*) AS n_taxonomy FROM taxonomy_categories;

-- ═══════════════════════════════════════════════════════════════════
-- Pick target tables — diverse set across domains
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE TEMP TABLE targets AS
WITH RANKED AS (
    SELECT rc.domain, rc.resource_id, count(*) AS n_cols,
           ROW_NUMBER() OVER (PARTITION BY rc.domain ORDER BY random()) AS rn
    FROM pg.socrata.resource_column AS rc
    WHERE rc.tt_end = '9999-12-31'
    GROUP BY rc.domain, rc.resource_id
    HAVING count(*) BETWEEN 10 AND 60
)
SELECT t.domain, t.resource_id, t.n_cols, r.name AS table_name
FROM RANKED AS t
JOIN pg.socrata.resource AS r
    ON r.domain = t.domain AND r.resource_id = t.resource_id
    AND r.tt_end = '9999-12-31'
WHERE t.rn <= 3;

.print ''
.print '=== Target tables ==='
SELECT domain, table_name, n_cols FROM targets ORDER BY domain, n_cols DESC;

-- ═══════════════════════════════════════════════════════════════════
-- LAYER 1: Strip known enumerations via blobfilters
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE TEMP TABLE col_enum_matches AS
WITH TABLE_COLS AS (
    SELECT rc.domain, rc.resource_id, lower(rc.field_name) AS col
    FROM pg.socrata.resource_column AS rc
    WHERE rc.tt_end = '9999-12-31'
      AND (rc.domain, rc.resource_id) IN (SELECT domain, resource_id FROM targets)
),
PER_COL_MATCH AS (
    SELECT tc.domain, tc.resource_id, tc.col,
           df.domain_label AS enum_label,
           bf_containment_json(json_array(tc.col), df.filter) AS is_match
    FROM TABLE_COLS AS tc
    CROSS JOIN domain_filters AS df
)
SELECT domain, resource_id, col, enum_label
FROM PER_COL_MATCH
WHERE is_match > 0;

-- Which columns were stripped?
CREATE OR REPLACE TEMP TABLE stripped_cols AS
SELECT DISTINCT domain, resource_id, col, enum_label
FROM col_enum_matches;

-- Remaining columns (not matched by any enumeration)
CREATE OR REPLACE TEMP TABLE remaining_cols AS
SELECT rc.domain, rc.resource_id, lower(rc.field_name) AS col, rc.data_type
FROM pg.socrata.resource_column AS rc
WHERE rc.tt_end = '9999-12-31'
  AND (rc.domain, rc.resource_id) IN (SELECT domain, resource_id FROM targets)
  AND lower(rc.field_name) NOT IN (
      SELECT col FROM stripped_cols AS sc
      WHERE sc.domain = rc.domain AND sc.resource_id = rc.resource_id
  );

.print ''
.print '=== Layer 1: Enumeration matches ==='
SELECT t.table_name, sc.enum_label, count(*) AS n_matched
FROM stripped_cols AS sc
JOIN targets AS t USING (domain, resource_id)
GROUP BY t.table_name, sc.enum_label
ORDER BY t.table_name, n_matched DESC;

-- ═══════════════════════════════════════════════════════════════════
-- LAYER 2: Embed remaining columns, compute pairwise similarity
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE TEMP TABLE col_embeds AS
SELECT domain, resource_id, col, data_type,
       be_embed('nomic', col)::FLOAT[768] AS vec
FROM remaining_cols;

.print ''
.print '=== Columns embedded ==='
SELECT count(*) AS n_embedded FROM col_embeds;

-- ═══════════════════════════════════════════════════════════════════
-- LAYER 3: Cluster columns within each table
-- Simple approach: greedy single-linkage with threshold 0.65
-- ═══════════════════════════════════════════════════════════════════

-- Clustering: each column's cluster = the column it's most similar to
-- (or itself if no neighbor exceeds threshold). Then propagate to the
-- lexicographically smallest member via min(). This gives star-shaped
-- clusters centered on the most "typical" member.

CREATE OR REPLACE TEMP TABLE col_clusters AS
WITH BEST_NEIGHBOR AS (
    SELECT a.domain, a.resource_id, a.col,
           b.col AS best_neighbor,
           array_cosine_similarity(a.vec, b.vec) AS sim
    FROM col_embeds AS a
    JOIN col_embeds AS b
        ON a.domain = b.domain AND a.resource_id = b.resource_id
        AND a.col != b.col
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY a.domain, a.resource_id, a.col
        ORDER BY array_cosine_similarity(a.vec, b.vec) DESC
    ) = 1
),
ASSIGNED AS (
    SELECT domain, resource_id, col,
           CASE WHEN sim >= 0.65
                THEN least(col, best_neighbor)
                ELSE col
           END AS cluster_id
    FROM BEST_NEIGHBOR
)
-- Propagate: if A→B and B→C, make A→min(A,B,C)
SELECT domain, resource_id, col,
       min(cluster_id) OVER (
           PARTITION BY domain, resource_id, cluster_id
       ) AS cluster_id
FROM ASSIGNED;

-- ═══════════════════════════════════════════════════════════════════
-- LAYER 4: Label each cluster via taxonomy centroid matching
-- ═══════════════════════════════════════════════════════════════════

-- For each cluster, pick the representative column (first member alphabetically)
-- and use its embedding as the cluster "centroid". This avoids the expensive
-- unnest/reaggregate for element-wise averaging — good enough for topic labelling
-- since cluster members are already similar (>0.65).

CREATE OR REPLACE TEMP TABLE cluster_reps AS
SELECT DISTINCT ON (domain, resource_id, cluster_id)
       cc.domain, cc.resource_id, cc.cluster_id, ce.vec
FROM col_clusters AS cc
JOIN col_embeds AS ce
    ON ce.domain = cc.domain AND ce.resource_id = cc.resource_id
    AND ce.col = cc.cluster_id;

CREATE OR REPLACE TEMP TABLE cluster_topics AS
SELECT cr.domain, cr.resource_id, cr.cluster_id,
       tc.category AS topic,
       tc.hypernym AS topic_parent,
       array_cosine_similarity(tc.vec, cr.vec) AS topic_score
FROM cluster_reps AS cr, taxonomy_categories AS tc
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY cr.domain, cr.resource_id, cr.cluster_id
    ORDER BY array_cosine_distance(tc.vec, cr.vec)
) = 1;

-- ═══════════════════════════════════════════════════════════════════
-- OUTPUT: Table topic profiles
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== TABLE TOPIC PROFILES ==='

WITH ENUM_TOPICS AS (
    -- Topics from Layer 1 (enumeration matches)
    SELECT t.table_name, sc.enum_label AS topic, 'enumeration' AS source,
           count(*) AS n_cols, 1.0 AS confidence
    FROM stripped_cols AS sc
    JOIN targets AS t USING (domain, resource_id)
    GROUP BY t.table_name, sc.enum_label
),
CLUSTER_TOPICS AS (
    -- Topics from Layer 3+4 (embedding clusters)
    SELECT t.table_name, ct.topic, 'embedding' AS source,
           count(*) AS n_cols, round(ct.topic_score, 3) AS confidence
    FROM col_clusters AS cc
    JOIN cluster_topics AS ct USING (domain, resource_id, cluster_id)
    JOIN targets AS t USING (domain, resource_id)
    GROUP BY t.table_name, ct.topic, ct.topic_score
),
ALL_TOPICS AS (
    SELECT * FROM ENUM_TOPICS
    UNION ALL
    SELECT * FROM CLUSTER_TOPICS
)
SELECT table_name,
       string_agg(
           topic || ' (' || n_cols || ')',
           ', ' ORDER BY n_cols DESC
       ) AS topic_profile
FROM ALL_TOPICS
WHERE confidence >= 0.5 OR source = 'enumeration'
GROUP BY table_name
ORDER BY table_name;

-- Detailed view: every cluster with its members and topic
.print ''
.print '=== DETAILED CLUSTER VIEW (first 5 tables) ==='

WITH DETAIL AS (
    SELECT t.table_name, ct.topic, ct.topic_parent,
           round(ct.topic_score, 3) AS score,
           string_agg(cc.col, ', ' ORDER BY cc.col) AS columns
    FROM col_clusters AS cc
    JOIN cluster_topics AS ct USING (domain, resource_id, cluster_id)
    JOIN targets AS t USING (domain, resource_id)
    GROUP BY t.table_name, ct.topic, ct.topic_parent, ct.topic_score, ct.cluster_id
),
ENUM_DETAIL AS (
    SELECT t.table_name, sc.enum_label AS topic, 'enumeration' AS topic_parent,
           1.0 AS score,
           string_agg(sc.col, ', ' ORDER BY sc.col) AS columns
    FROM stripped_cols AS sc
    JOIN targets AS t USING (domain, resource_id)
    GROUP BY t.table_name, sc.enum_label
)
SELECT * FROM (
    SELECT * FROM DETAIL
    UNION ALL
    SELECT * FROM ENUM_DETAIL
) AS combined
WHERE table_name IN (SELECT table_name FROM targets ORDER BY table_name LIMIT 5)
ORDER BY table_name, score DESC;
