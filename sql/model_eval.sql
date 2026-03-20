-- Model evaluation: which model best matches symbols to their known domains?
--
-- We have ground truth: domain.member tells us banana belongs to "countries"
-- wait no — banana doesn't. But "France" belongs to countries, "USD" belongs
-- to currencies, "January" belongs to months. We can sample members from
-- each domain and see if the model assigns them back to the correct domain.

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable' AS pg (TYPE POSTGRES);

LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';

SELECT be_load_hf_model('nomic',
    'nomic-ai/nomic-embed-text-v1.5-GGUF', 'nomic-embed-text-v1.5.Q4_K_M.gguf');
SELECT be_load_hf_model('miniLM',
    'second-state/All-MiniLM-L6-v2-Embedding-GGUF', 'all-MiniLM-L6-v2-Q8_0.gguf');
SELECT be_load_hf_model('snowflake',
    'ChristianAzinn/snowflake-arctic-embed-s-gguf', 'snowflake-arctic-embed-s-f16.GGUF');

-- ── Sample symbols from each domain (ground truth) ─────────────────
-- Take 5 random members from each domain as test symbols

CREATE TEMP TABLE test_symbols AS
SELECT domain_name, label AS symbol
FROM (
    SELECT domain_name, label,
           ROW_NUMBER() OVER (PARTITION BY domain_name ORDER BY random()) AS rn
    FROM pg.domain.member
)
WHERE rn <= 5;

.print '=== Test symbols (5 per domain) ==='
SELECT domain_name, string_agg(symbol, ', ' ORDER BY symbol) AS samples
FROM test_symbols
GROUP BY domain_name
ORDER BY domain_name;

-- ── Models to evaluate ─────────────────────────────────────────────
CREATE TEMP TABLE models AS
SELECT unnest(['nomic', 'miniLM', 'snowflake']) AS model;

-- ── Embed domain labels with each model ────────────────────────────
.print ''
.print '=== Embedding domain labels and symbols ==='

CREATE TEMP TABLE domain_embeds AS
SELECT m.model,
       e.domain_name,
       e.domain_label,
       be_embed(m.model, e.domain_label) AS vec
FROM (SELECT DISTINCT domain_name, domain_label FROM pg.domain.enumeration) AS e
CROSS JOIN models AS m;

-- ── Embed test symbols with each model ─────────────────────────────
CREATE TEMP TABLE symbol_embeds AS
SELECT m.model,
       ts.domain_name AS true_domain,
       ts.symbol,
       be_embed(m.model, lower(ts.symbol)) AS vec
FROM test_symbols AS ts
CROSS JOIN models AS m;

.print 'Embeddings complete.'
SELECT (SELECT count(*) FROM domain_embeds) AS domain_embeds,
       (SELECT count(*) FROM symbol_embeds) AS symbol_embeds;

-- ── Match each symbol to its best domain per model ─────────────────
CREATE TEMP TABLE matches AS
SELECT se.model, se.true_domain, se.symbol,
       de.domain_name AS predicted_domain,
       de.domain_label AS predicted_label,
       be_cosine_sim(se.vec, de.vec) AS score
FROM symbol_embeds AS se
JOIN domain_embeds AS de ON se.model = de.model
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY se.model, se.symbol
    ORDER BY be_cosine_sim(se.vec, de.vec) DESC
) <= 3;

-- ═══════════════════════════════════════════════════════════════════
-- RESULTS
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Top-1 accuracy per model ==='
WITH TOP1 AS (
    SELECT model, true_domain, symbol, predicted_domain, score
    FROM matches
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY model, symbol ORDER BY score DESC
    ) = 1
)
SELECT model,
       count(*) FILTER (WHERE true_domain = predicted_domain) AS correct,
       count(*) AS total,
       round(count(*) FILTER (WHERE true_domain = predicted_domain) * 100.0 / count(*), 1) AS accuracy_pct
FROM TOP1
GROUP BY model
ORDER BY accuracy_pct DESC;

.print ''
.print '=== Top-3 accuracy (correct domain in top 3 predictions) ==='
WITH TOP3_HIT AS (
    SELECT DISTINCT model, symbol, true_domain,
           bool_or(true_domain = predicted_domain) AS hit
    FROM matches
    GROUP BY model, symbol, true_domain
)
SELECT model,
       count(*) FILTER (WHERE hit) AS correct,
       count(*) AS total,
       round(count(*) FILTER (WHERE hit) * 100.0 / count(*), 1) AS accuracy_pct
FROM TOP3_HIT
GROUP BY model
ORDER BY accuracy_pct DESC;

.print ''
.print '=== Per-domain accuracy (top-1, all models) ==='
WITH TOP1 AS (
    SELECT model, true_domain, symbol, predicted_domain
    FROM matches
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY model, symbol ORDER BY score DESC
    ) = 1
)
SELECT true_domain,
       count(*) FILTER (WHERE model = 'nomic' AND true_domain = predicted_domain) AS nomic_correct,
       count(*) FILTER (WHERE model = 'miniLM' AND true_domain = predicted_domain) AS miniLM_correct,
       count(*) FILTER (WHERE model = 'snowflake' AND true_domain = predicted_domain) AS snowflake_correct,
       5 AS out_of
FROM TOP1
GROUP BY true_domain
ORDER BY true_domain;

.print ''
.print '=== Score gap: top-1 minus top-2 (confidence of prediction) ==='
WITH RANKED AS (
    SELECT model, symbol, true_domain, predicted_domain, score,
           ROW_NUMBER() OVER (PARTITION BY model, symbol ORDER BY score DESC) AS rank
    FROM matches
)
SELECT model,
       round(avg(score) FILTER (WHERE rank = 1), 3) AS avg_top1_score,
       round(avg(score) FILTER (WHERE rank = 2), 3) AS avg_top2_score,
       round(avg(score) FILTER (WHERE rank = 1) - avg(score) FILTER (WHERE rank = 2), 3) AS avg_gap
FROM RANKED
GROUP BY model
ORDER BY avg_gap DESC;

.print ''
.print '=== Misclassifications ==='
WITH TOP1 AS (
    SELECT model, true_domain, symbol, predicted_domain, round(score, 3) AS score
    FROM matches
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY model, symbol ORDER BY score DESC
    ) = 1
)
SELECT * FROM TOP1
WHERE true_domain != predicted_domain
ORDER BY model, true_domain, symbol;
