-- Build geonames_embedding table by reading source data from PG.
--
-- Prerequisites:
--   duckdb -unsigned
--   GeoNames data loaded in PG (gazetteer schema) via geonames_model.py

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable' AS pg (TYPE POSTGRES);

INSTALL icu; LOAD icu;
LOAD '/Users/paulharrington/checkouts/blobembed/build/duckdb/blobembed.duckdb_extension';

SELECT be_load_hf_model('nomic',
    'nomic-ai/nomic-embed-text-v1.5-GGUF',
    'nomic-embed-text-v1.5.Q4_K_M.gguf');

.timer on

-- ── Countries ──────────────────────────────────────────────────────
.print '=== Embedding countries ==='
CREATE OR REPLACE TABLE geonames_embedding AS
SELECT geonameid,
       'PCLI' AS feature_code,
       country AS feature_name,
       be_embed('nomic', strip_accents(country)) AS embedding
FROM pg.gazetteer.geonames_country
WHERE geonameid IS NOT NULL;

SELECT count(*) AS n FROM geonames_embedding;

-- ── Admin1 ─────────────────────────────────────────────────────────
.print '=== Embedding admin1 ==='
INSERT INTO geonames_embedding
SELECT a1.geonameid,
       'ADM1' AS feature_code,
       a1.name AS feature_name,
       be_embed('nomic', strip_accents(
           COALESCE(co.country, '') || ' > ' || a1.name
       )) AS embedding
FROM pg.gazetteer.geonames_admin1 AS a1
LEFT JOIN pg.gazetteer.geonames_country AS co
    ON co.iso = split_part(a1.code, '.', 1);

SELECT count(*) AS n FROM geonames_embedding;

-- ── Admin2 ─────────────────────────────────────────────────────────
.print '=== Embedding admin2 (~40 min) ==='
INSERT INTO geonames_embedding
SELECT a2.geonameid,
       'ADM2' AS feature_code,
       a2.name AS feature_name,
       be_embed('nomic', strip_accents(
           COALESCE(co.country, '') || ' > ' || COALESCE(a1.name, '') || ' > ' || a2.name
       )) AS embedding
FROM pg.gazetteer.geonames_admin2 AS a2
LEFT JOIN pg.gazetteer.geonames_country AS co
    ON co.iso = split_part(a2.code, '.', 1)
LEFT JOIN pg.gazetteer.geonames_admin1 AS a1
    ON a1.code = split_part(a2.code, '.', 1) || '.' || split_part(a2.code, '.', 2);

SELECT count(*) AS n FROM geonames_embedding;

-- ── Places ─────────────────────────────────────────────────────────
.print '=== Embedding places (~2 hours) ==='
INSERT INTO geonames_embedding
SELECT geonameid,
       feature_code,
       place_name AS feature_name,
       be_embed('nomic', strip_accents(full_path)) AS embedding
FROM pg.gazetteer.geonames_place
WHERE full_path IS NOT NULL;

.timer off

-- ── Stats ──────────────────────────────────────────────────────────
.print ''
.print '=== Final stats ==='
SELECT feature_code, count(*) AS n
FROM geonames_embedding
GROUP BY feature_code
ORDER BY feature_code;

-- ── Test queries ───────────────────────────────────────────────────
.print ''
.print '=== Test: Carrick-on-Shannon, Ireland ==='
SELECT geonameid, feature_code, feature_name,
       round(be_cosine_sim(embedding,
           be_embed('nomic', strip_accents('Ireland > Leitrim > Carrick-on-Shannon'))
       ), 4) AS score
FROM geonames_embedding
ORDER BY score DESC
LIMIT 10;

.print ''
.print '=== Test: San Francisco (ambiguous) ==='
SELECT geonameid, feature_code, feature_name,
       round(be_cosine_sim(embedding,
           be_embed('nomic', 'San Francisco')
       ), 4) AS score
FROM geonames_embedding
ORDER BY score DESC
LIMIT 10;

.print ''
.print '=== Test: United States > California > San Francisco (with context) ==='
SELECT geonameid, feature_code, feature_name,
       round(be_cosine_sim(embedding,
           be_embed('nomic', 'United States > California > San Francisco')
       ), 4) AS score
FROM geonames_embedding
ORDER BY score DESC
LIMIT 10;
