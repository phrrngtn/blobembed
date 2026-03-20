-- Model benchmark: compare embedding quality across 3 GGUF models.
--
-- Models:
--   nomic-Q4: nomic-embed-text-v1.5 Q4_K_M (79MB, 768-dim)
--   nomic-Q8: nomic-embed-text-v1.5 Q8_0   (147MB, 768-dim)
--   snowflake: snowflake-arctic-embed-s F16  (67MB, 384-dim)
--   miniLM: all-MiniLM-L6-v2 Q8_0           (24MB, 384-dim)
--
-- Tests:
--   A. Column name similarity (same column, different name)
--   B. Table matching (column lists)
--   C. Category membership (instances → category)
--   D. Throughput (embeddings per second)

LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';

-- ── Load models ────────────────────────────────────────────────────
SELECT be_load_hf_model('nomic-q4',
    'nomic-ai/nomic-embed-text-v1.5-GGUF', 'nomic-embed-text-v1.5.Q4_K_M.gguf');
SELECT be_load_hf_model('nomic-q8',
    'nomic-ai/nomic-embed-text-v1.5-GGUF', 'nomic-embed-text-v1.5.Q8_0.gguf');
SELECT be_load_hf_model('snowflake',
    'ChristianAzinn/snowflake-arctic-embed-s-gguf', 'snowflake-arctic-embed-s-f16.GGUF');
SELECT be_load_hf_model('miniLM',
    'second-state/All-MiniLM-L6-v2-Embedding-GGUF', 'all-MiniLM-L6-v2-Q8_0.gguf');

.print '=== Models loaded ==='
SELECT 'nomic-q4' AS model, be_embed_dim('nomic-q4') AS dim
UNION ALL SELECT 'nomic-q8', be_embed_dim('nomic-q8')
UNION ALL SELECT 'snowflake', be_embed_dim('snowflake')
UNION ALL SELECT 'miniLM', be_embed_dim('miniLM');

-- ── Test data ──────────────────────────────────────────────────────
CREATE OR REPLACE TEMP TABLE models AS
SELECT unnest(['nomic-q4', 'nomic-q8', 'snowflake', 'miniLM']) AS model_name;

-- Cosine similarity for LIST(FLOAT) — works regardless of dimension
-- Cosine similarity for LIST(FLOAT) — works regardless of dimension
CREATE OR REPLACE MACRO list_cosine_sim(a, b) AS (
    list_dot_product(a, b)
    / (sqrt(list_dot_product(a, a)) * sqrt(list_dot_product(b, b)))
);

-- Helper: dot product of two float lists
CREATE OR REPLACE MACRO list_dot_product(a, b) AS (
    list_sum(list_transform(list_zip(a, b), p : p[1] * p[2]))
);

-- ═══════════════════════════════════════════════════════════════════
-- TEST A: Column name similarity
-- Same concept, different naming conventions. Higher = better.
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Test A: Column name similarity (higher = better) ==='

WITH PAIRS AS (
    SELECT * FROM (VALUES
        ('customer_id', 'cust_id', 'same column, abbreviated'),
        ('customer_id', 'total_revenue', 'different columns'),
        ('first_name', 'fname', 'name abbreviation'),
        ('first_name', 'last_name', 'related but different'),
        ('zip_code', 'postal_code', 'synonym'),
        ('zip_code', 'phone_number', 'unrelated'),
        ('created_at', 'creation_date', 'synonym'),
        ('created_at', 'total_amount', 'unrelated')
    ) AS t(col_a, col_b, relationship)
)
SELECT m.model_name, p.col_a, p.col_b, p.relationship,
       round(list_cosine_sim(
           be_embed(m.model_name, p.col_a),
           be_embed(m.model_name, p.col_b)
       ), 3) AS sim
FROM PAIRS AS p
CROSS JOIN models AS m
ORDER BY p.col_a, p.col_b, m.model_name;

-- ═══════════════════════════════════════════════════════════════════
-- TEST B: Table matching — column lists
-- Same table, different naming conventions. Higher = better.
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Test B: Table matching (higher = better) ==='

WITH TABLE_PAIRS AS (
    SELECT * FROM (VALUES
        ('SalesOrderHeader: OrderDate, CustomerID, SalesPersonID, TotalDue, SubTotal, TaxAmt',
         'order_header: order_date, customer_id, salesperson_id, total_amount, subtotal, tax',
         'same table, different names'),
        ('SalesOrderHeader: OrderDate, CustomerID, SalesPersonID, TotalDue, SubTotal, TaxAmt',
         'HumanResources.Employee: BusinessEntityID, NationalIDNumber, LoginID, JobTitle, BirthDate',
         'different tables')
    ) AS t(table_a, table_b, relationship)
)
SELECT m.model_name, tp.relationship,
       round(list_cosine_sim(
           be_embed(m.model_name, tp.table_a),
           be_embed(m.model_name, tp.table_b)
       ), 3) AS sim
FROM TABLE_PAIRS AS tp
CROSS JOIN models AS m
ORDER BY tp.relationship, m.model_name;

-- ═══════════════════════════════════════════════════════════════════
-- TEST C: Category membership — instances → category centroid
-- Banana/apple/kiwi should be closer to "fruit" than to "country".
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Test C: Category membership (fruit vs country) ==='

WITH INSTANCES AS (
    SELECT * FROM (VALUES
        ('banana', 'fruit', 'correct category'),
        ('banana', 'country', 'wrong category'),
        ('France', 'country', 'correct category'),
        ('France', 'fruit', 'wrong category')
    ) AS t(instance, category, expected)
)
SELECT m.model_name, i.instance, i.category, i.expected,
       round(list_cosine_sim(
           be_embed(m.model_name, i.instance),
           be_embed(m.model_name, i.category)
       ), 3) AS sim
FROM INSTANCES AS i
CROSS JOIN models AS m
ORDER BY i.instance, i.category, m.model_name;

-- ═══════════════════════════════════════════════════════════════════
-- TEST D: Throughput — embeddings per second
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Test D: Throughput (50 embeddings each) ==='

CREATE OR REPLACE TEMP TABLE test_texts AS
SELECT 'text_' || i AS txt FROM generate_series(1, 50) AS t(i);

SELECT m.model_name,
       count(*) AS n_embeds,
       round(epoch(max(current_timestamp) - min(current_timestamp)), 2) AS elapsed_sec
FROM test_texts AS tt
CROSS JOIN models AS m
WHERE be_embed(m.model_name, tt.txt) IS NOT NULL
GROUP BY m.model_name
ORDER BY m.model_name;
