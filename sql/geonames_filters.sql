-- Build blobfilters for place name pre-filtering.
--
-- One filter per feature_code level (PCLI, ADM1, ADM2, PPL*).
-- Uses normalized names (NFKD + lowercase) for fuzzy matching.
-- Catches abbreviations and accent variations that embeddings miss.
--
-- Prerequisites: gazetteer data in PG, blobfilters extension

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable' AS pg (TYPE POSTGRES);
LOAD '/Users/paulharrington/checkouts/blobfilters/build/duckdb/blobfilters.duckdb_extension';
INSTALL icu; LOAD icu;

-- ── Build filters per geographic level ─────────────────────────────

-- Countries: country name + iso + iso3
.print '=== Building country filter ==='
CREATE TABLE geo_filters AS
WITH COUNTRY_TERMS AS (
    SELECT lower(strip_accents(country)) AS term FROM pg.gazetteer.geonames_country
    UNION
    SELECT lower(iso) FROM pg.gazetteer.geonames_country
    UNION
    SELECT lower(iso3) FROM pg.gazetteer.geonames_country
)
SELECT 'country' AS level,
       bf_build_json(json_group_array(term)) AS filter,
       count(*) AS n_terms
FROM COUNTRY_TERMS;

-- Admin1: region/state/province names
.print '=== Building admin1 filter ==='
INSERT INTO geo_filters
WITH ADMIN1_TERMS AS (
    SELECT DISTINCT lower(strip_accents(name)) AS term
    FROM pg.gazetteer.geonames_admin1
)
SELECT 'admin1', bf_build_json(json_group_array(term)), count(*)
FROM ADMIN1_TERMS;

-- Admin2: county/department names
.print '=== Building admin2 filter ==='
INSERT INTO geo_filters
WITH ADMIN2_TERMS AS (
    SELECT DISTINCT lower(strip_accents(name)) AS term
    FROM pg.gazetteer.geonames_admin2
)
SELECT 'admin2', bf_build_json(json_group_array(term)), count(*)
FROM ADMIN2_TERMS;

-- Cities: place names (ascii version for broader matching)
.print '=== Building city filter ==='
INSERT INTO geo_filters
WITH CITY_TERMS AS (
    SELECT DISTINCT lower(strip_accents(place_ascii)) AS term
    FROM pg.gazetteer.geonames_place
    WHERE place_ascii IS NOT NULL
)
SELECT 'city', bf_build_json(json_group_array(term)), count(*)
FROM CITY_TERMS;

.print ''
.print '=== Filter stats ==='
SELECT level, n_terms, bf_cardinality(filter) AS filter_card
FROM geo_filters;

-- ── Test: column role detection ────────────────────────────────────
-- Given unlabeled columns, which geographic level does each represent?

.print ''
.print '=== Column role detection test ==='
.print 'Probe: ["Ireland", "Leitrim", "Carrick-on-Shannon"]'

SELECT gf.level,
       bf_containment_json('["ireland", "leitrim", "carrick-on-shannon"]', gf.filter) AS containment,
       bf_intersection_card(
           bf_build_json('["ireland", "leitrim", "carrick-on-shannon"]'),
           gf.filter
       ) AS n_matches
FROM geo_filters AS gf
ORDER BY containment DESC;

-- Per-value probe: which level does each value belong to?
.print ''
.print 'Per-value level detection:'
SELECT val,
       (SELECT level FROM geo_filters ORDER BY bf_containment_json(json_array(lower(val)), filter) DESC LIMIT 1) AS best_level,
       (SELECT round(bf_containment_json(json_array(lower(val)), filter), 3) FROM geo_filters ORDER BY bf_containment_json(json_array(lower(val)), filter) DESC LIMIT 1) AS score
FROM (VALUES ('Ireland'), ('Leitrim'), ('Carrick-on-Shannon'),
             ('France'), ('Ile-de-France'), ('Paris'),
             ('US'), ('California'), ('San Francisco'),
             ('Sao Paulo'), ('Brasil'), ('NYC'),
             ('Bejing'), ('Stokholm'), ('Moskva')
) AS t(val);

-- ── Store filters in PG for prefetch ───────────────────────────────
.print ''
.print '=== Storing filters in PG ==='

-- Reuse the domain.enumeration table pattern
INSERT INTO pg.domain.enumeration (domain_name, domain_label, source, member_count, filter_b64)
SELECT 'geo_' || level,
       level,
       'geonames',
       n_terms,
       bf_to_base64(filter)
FROM geo_filters
ON CONFLICT DO NOTHING;

.print 'Filters stored in PG domain.enumeration.'
