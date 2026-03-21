-- Full place resolution pipeline against world-cities dataset.
-- Layer 1: blobfilters (enriched with alt names)
-- Layer 2: embedding fallback for what filters miss

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable' AS pg (TYPE POSTGRES);
INSTALL icu; LOAD icu;
LOAD '/Users/paulharrington/checkouts/blobfilters/build/duckdb/blobfilters.duckdb_extension';

-- ── Load test data ─────────────────────────────────────────────────
.print '=== Loading world-cities test data ==='
CREATE TABLE test_cities AS
SELECT * FROM read_csv(
    'https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv',
    auto_detect = true);

SELECT count(*) AS total_rows FROM test_cities;

-- Sample: take 500 random rows for testing
CREATE TABLE test_sample AS
SELECT * FROM test_cities ORDER BY random() LIMIT 500;

.print ''
.print '=== Sample rows ==='
SELECT * FROM test_sample LIMIT 5;

-- ── Build enriched filters from PG ─────────────────────────────────
.print ''
.print '=== Building enriched filters from PG alt names ==='
.timer on

CREATE TABLE geo_filters AS
WITH COUNTRY_TERMS AS (
    SELECT lower(strip_accents(country)) AS term FROM pg.gazetteer.geonames_country
    UNION SELECT lower(iso) FROM pg.gazetteer.geonames_country
    UNION SELECT lower(iso3) FROM pg.gazetteer.geonames_country
    UNION SELECT DISTINCT lower(strip_accents(alt_name)) FROM pg.gazetteer.geonames_alt_name AS an
           JOIN pg.gazetteer.geonames_country AS c ON c.geonameid::INTEGER = an.geonameid
)
SELECT 'country' AS level, bf_build_json(json_group_array(term)) AS filter, count(*) AS n_terms
FROM COUNTRY_TERMS;

INSERT INTO geo_filters
WITH ADMIN1_TERMS AS (
    SELECT lower(strip_accents(name)) AS term FROM pg.gazetteer.geonames_admin1
    UNION SELECT DISTINCT lower(strip_accents(alt_name)) FROM pg.gazetteer.geonames_alt_name AS an
           JOIN pg.gazetteer.geonames_admin1 AS a1 ON a1.geonameid = an.geonameid
)
SELECT 'admin1', bf_build_json(json_group_array(term)), count(*) FROM ADMIN1_TERMS;

INSERT INTO geo_filters
WITH CITY_TERMS AS (
    SELECT lower(strip_accents(place_ascii)) AS term FROM pg.gazetteer.geonames_place
    UNION SELECT DISTINCT lower(strip_accents(alt_name)) FROM pg.gazetteer.geonames_alt_name AS an
           JOIN pg.gazetteer.geonames_place AS p ON p.geonameid = an.geonameid
)
SELECT 'city', bf_build_json(json_group_array(term)), count(*) FROM CITY_TERMS;

.timer off

SELECT level, n_terms, bf_cardinality(filter) AS card FROM geo_filters;

-- ── Layer 1: blobfilter resolution ─────────────────────────────────
.print ''
.print '=== Layer 1: blobfilter column role detection ==='
.timer on

-- For each column, what proportion of values match each level?
WITH COL_COUNTRY AS (
    SELECT json_group_array(DISTINCT lower(strip_accents(country))) AS arr FROM test_sample
),
COL_SUBCOUNTRY AS (
    SELECT json_group_array(DISTINCT lower(strip_accents(subcountry))) AS arr FROM test_sample
),
COL_CITY AS (
    SELECT json_group_array(DISTINCT lower(strip_accents(name))) AS arr FROM test_sample
)
SELECT 'country_col' AS column_name, gf.level,
       round(bf_containment_json((SELECT arr FROM COL_COUNTRY), gf.filter), 3) AS containment
FROM geo_filters AS gf
UNION ALL
SELECT 'subcountry_col', gf.level,
       round(bf_containment_json((SELECT arr FROM COL_SUBCOUNTRY), gf.filter), 3)
FROM geo_filters AS gf
UNION ALL
SELECT 'city_col', gf.level,
       round(bf_containment_json((SELECT arr FROM COL_CITY), gf.filter), 3)
FROM geo_filters AS gf
ORDER BY column_name, containment DESC;

.timer off

-- ── Layer 1: per-row resolution ────────────────────────────────────
.print ''
.print '=== Layer 1: per-row place matching ==='
.timer on

-- For each row, check if city name is in the city filter
CREATE TABLE resolution_results AS
SELECT ts.name AS input_city,
       ts.subcountry AS input_admin1,
       ts.country AS input_country,
       ts.geonameid AS expected_geonameid,
       bf_containment_json(json_array(lower(strip_accents(ts.name))),
           (SELECT filter FROM geo_filters WHERE level = 'city')
       ) AS city_match,
       bf_containment_json(json_array(lower(strip_accents(ts.country))),
           (SELECT filter FROM geo_filters WHERE level = 'country')
       ) AS country_match,
       bf_containment_json(json_array(lower(strip_accents(ts.subcountry))),
           (SELECT filter FROM geo_filters WHERE level = 'admin1')
       ) AS admin1_match
FROM test_sample AS ts;

.timer off

.print ''
.print '=== Resolution stats ==='
SELECT count(*) AS total,
       count(*) FILTER (WHERE city_match > 0) AS city_found,
       count(*) FILTER (WHERE country_match > 0) AS country_found,
       count(*) FILTER (WHERE admin1_match > 0) AS admin1_found,
       count(*) FILTER (WHERE city_match > 0 AND country_match > 0) AS both_found,
       count(*) FILTER (WHERE city_match = 0 AND country_match = 0 AND admin1_match = 0) AS nothing_found,
       round(count(*) FILTER (WHERE city_match > 0) * 100.0 / count(*), 1) AS city_pct,
       round(count(*) FILTER (WHERE country_match > 0) * 100.0 / count(*), 1) AS country_pct
FROM resolution_results;

.print ''
.print '=== Unresolved rows (need embedding fallback) ==='
SELECT input_city, input_admin1, input_country
FROM resolution_results
WHERE city_match = 0
ORDER BY input_country
LIMIT 20;
