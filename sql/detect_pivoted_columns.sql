-- Detect denormalized/pivoted columns in a table.
--
-- Given a table's column names, this pipeline:
--   1. Embeds each column name
--   2. Computes pairwise cosine similarity
--   3. Identifies clusters of semantically similar columns
--   4. For each cluster, finds the best category name from the taxonomy
--   5. Outputs an unpivot recipe
--
-- Prerequisites: taxonomy_categories table loaded (see load_taxonomy.sql)
--
-- Input: set :table_columns to a comma-separated list of column names, e.g.:
--   SET VARIABLE table_columns = 'date,customer,bananas,apples,kiwi,mango';

-- ── Step 1: Embed each column name ─────────────────────────────────
CREATE OR REPLACE TEMP TABLE col_embeddings AS
SELECT trim(col) AS col_name,
       be_embed('nomic', trim(col))::FLOAT[768] AS vec
FROM (
    SELECT unnest(string_split(getvariable('table_columns'), ',')) AS col
);

-- ── Step 2: Pairwise similarity matrix ─────────────────────────────
CREATE OR REPLACE TEMP TABLE col_pairs AS
SELECT a.col_name AS col_a,
       b.col_name AS col_b,
       array_cosine_similarity(a.vec, b.vec) AS sim
FROM col_embeddings AS a, col_embeddings AS b
WHERE a.col_name < b.col_name;

-- ── Step 3: Simple single-linkage clustering ───────────────────────
-- Columns with pairwise similarity > threshold form a cluster.
-- This is a greedy approach: start with the highest-similarity pair,
-- grow the cluster by adding any column with sim > threshold to any
-- cluster member.
--
-- Threshold 0.70 works well for "instances of a category" detection.
-- Columns not in any cluster are assumed to be pivot columns (dimensions).

CREATE OR REPLACE TEMP TABLE clusters AS
WITH RECURSIVE
SEED AS (
    -- All edges above threshold
    SELECT col_a, col_b, sim
    FROM col_pairs
    WHERE sim >= 0.70
),
-- Assign connected components via recursive self-join
EDGES AS (
    SELECT col_a AS node, col_b AS neighbor FROM SEED
    UNION ALL
    SELECT col_b AS node, col_a AS neighbor FROM SEED
),
COMPONENTS(node, component) AS (
    SELECT DISTINCT node, node AS component FROM EDGES
    UNION
    SELECT e.neighbor, c.component
    FROM COMPONENTS AS c
    JOIN EDGES AS e ON e.node = c.node
    WHERE e.neighbor != c.component
)
SELECT node AS col_name,
       min(component) AS cluster_id
FROM COMPONENTS
GROUP BY node;

-- ── Step 4: Compute cluster centroids ──────────────────────────────
CREATE OR REPLACE TEMP TABLE cluster_centroids AS
WITH FLAT AS (
    SELECT cl.cluster_id,
           unnest(ce.vec) AS val,
           generate_subscripts(ce.vec, 1) AS dim
    FROM clusters AS cl
    JOIN col_embeddings AS ce ON ce.col_name = cl.col_name
)
SELECT cluster_id,
       list(AVG(val) ORDER BY dim)::FLOAT[768] AS centroid
FROM FLAT
GROUP BY cluster_id;

-- ── Step 5: Find best taxonomy category for each cluster ───────────
CREATE OR REPLACE TEMP TABLE cluster_categories AS
SELECT cc.cluster_id,
       tc.category,
       tc.hypernym,
       array_cosine_similarity(tc.vec, cc.centroid) AS score
FROM cluster_centroids AS cc,
     taxonomy_categories AS tc
QUALIFY ROW_NUMBER() OVER (PARTITION BY cc.cluster_id ORDER BY
    array_cosine_distance(tc.vec, cc.centroid)) <= 1;

-- ── Step 6: Output the unpivot recipe ──────────────────────────────
WITH CLUSTERED_COLS AS (
    SELECT cl.cluster_id, cl.col_name, cat.category, cat.hypernym, cat.score
    FROM clusters AS cl
    JOIN cluster_categories AS cat USING (cluster_id)
),
PIVOT_COLS AS (
    -- Columns NOT in any cluster = pivot columns (dimensions)
    SELECT col_name
    FROM col_embeddings
    WHERE col_name NOT IN (SELECT col_name FROM clusters)
)
SELECT json_object(
    'category_name', cc.category,
    'category_hypernym', cc.hypernym,
    'confidence', round(cc.score, 4),
    'instance_columns', json_group_array(cc.col_name ORDER BY cc.col_name),
    'pivot_columns', (SELECT json_group_array(col_name ORDER BY col_name) FROM PIVOT_COLS),
    'suggested_schema', (
        SELECT json_group_array(col ORDER BY col)
        FROM (
            SELECT col_name AS col FROM PIVOT_COLS
            UNION ALL
            SELECT DISTINCT cc2.category FROM CLUSTERED_COLS AS cc2
                WHERE cc2.cluster_id = cc.cluster_id
            UNION ALL
            VALUES ('quantity')
        )
    )
) AS unpivot_recipe
FROM CLUSTERED_COLS AS cc
GROUP BY cc.cluster_id, cc.category, cc.hypernym, cc.score;
