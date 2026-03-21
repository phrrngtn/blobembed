-- Build enriched geonames filters using alternate names.
-- Adds abbreviations, transliterations, and common aliases.

INSTALL icu; LOAD icu;
INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable' AS pg (TYPE POSTGRES);
LOAD '/Users/paulharrington/checkouts/blobfilters/build/duckdb/blobfilters.duckdb_extension';

-- Load alternate names (English + no-language entries only)
.print '=== Loading alternate names ==='
.timer on

CREATE TABLE alt_names AS
SELECT * FROM read_csv('/tmp/altnames_en.tsv',
    delim = '\t', header = false, quote = '',
    columns = {
        'alt_id': 'INTEGER',
        'geonameid': 'INTEGER',
        'lang': 'VARCHAR',
        'alt_name': 'VARCHAR',
        'is_preferred': 'VARCHAR',
        'is_short': 'VARCHAR',
        'is_colloquial': 'VARCHAR',
        'is_historic': 'VARCHAR',
        'from_date': 'VARCHAR',
        'to_date': 'VARCHAR'
    });

.timer off
SELECT count(*) AS n_alt_names FROM alt_names;
SELECT lang, count(*) AS n FROM alt_names GROUP BY lang ORDER BY n DESC;

-- Join with places to get feature codes
.print ''
.print '=== Joining with places ==='

CREATE TABLE alt_with_level AS
SELECT an.geonameid, an.alt_name, an.lang,
       COALESCE(p.feature_code, c.iso, a1.code, a2.code) AS source,
       CASE
           WHEN c.iso IS NOT NULL THEN 'country'
           WHEN a1.code IS NOT NULL THEN 'admin1'
           WHEN a2.code IS NOT NULL THEN 'admin2'
           WHEN p.geonameid IS NOT NULL THEN 'city'
       END AS level
FROM alt_names AS an
LEFT JOIN pg.gazetteer.geonames_place AS p ON p.geonameid = an.geonameid
LEFT JOIN pg.gazetteer.geonames_country AS c ON c.geonameid::INTEGER = an.geonameid
LEFT JOIN pg.gazetteer.geonames_admin1 AS a1 ON a1.geonameid = an.geonameid
LEFT JOIN pg.gazetteer.geonames_admin2 AS a2 ON a2.geonameid = an.geonameid
WHERE COALESCE(p.geonameid, c.geonameid::INTEGER, a1.geonameid, a2.geonameid) IS NOT NULL;

SELECT level, count(*) AS n FROM alt_with_level GROUP BY level ORDER BY level;

-- Build enriched filters per level
.print ''
.print '=== Building enriched filters ==='

-- Country: original names + iso codes + alt names
CREATE TABLE enriched_filters AS
WITH COUNTRY_TERMS AS (
    SELECT lower(strip_accents(country)) AS term FROM pg.gazetteer.geonames_country
    UNION SELECT lower(iso) FROM pg.gazetteer.geonames_country
    UNION SELECT lower(iso3) FROM pg.gazetteer.geonames_country
    UNION SELECT lower(strip_accents(alt_name)) FROM alt_with_level WHERE level = 'country'
)
SELECT 'country' AS level, bf_build_json(json_group_array(term)) AS filter, count(*) AS n_terms
FROM COUNTRY_TERMS;

INSERT INTO enriched_filters
WITH ADMIN1_TERMS AS (
    SELECT lower(strip_accents(name)) AS term FROM pg.gazetteer.geonames_admin1
    UNION SELECT lower(strip_accents(alt_name)) FROM alt_with_level WHERE level = 'admin1'
)
SELECT 'admin1', bf_build_json(json_group_array(term)), count(*) FROM ADMIN1_TERMS;

INSERT INTO enriched_filters
WITH ADMIN2_TERMS AS (
    SELECT lower(strip_accents(name)) AS term FROM pg.gazetteer.geonames_admin2
    UNION SELECT lower(strip_accents(alt_name)) FROM alt_with_level WHERE level = 'admin2'
)
SELECT 'admin2', bf_build_json(json_group_array(term)), count(*) FROM ADMIN2_TERMS;

INSERT INTO enriched_filters
WITH CITY_TERMS AS (
    SELECT lower(strip_accents(place_ascii)) AS term FROM pg.gazetteer.geonames_place
    UNION SELECT lower(strip_accents(alt_name)) FROM alt_with_level WHERE level = 'city'
)
SELECT 'city', bf_build_json(json_group_array(term)), count(*) FROM CITY_TERMS;

.print ''
.print '=== Enriched vs original filter stats ==='
SELECT e.level,
       e.n_terms AS enriched_terms,
       bf_cardinality(e.filter) AS enriched_card
FROM enriched_filters AS e
ORDER BY e.level;

-- Test the problem cases
.print ''
.print '=== Testing previously-failed lookups ==='
SELECT val,
       (SELECT level FROM enriched_filters
        ORDER BY bf_containment_json(json_array(lower(strip_accents(val))), filter) DESC LIMIT 1
       ) AS best_level,
       (SELECT round(bf_containment_json(json_array(lower(strip_accents(val))), filter), 3)
        FROM enriched_filters
        ORDER BY bf_containment_json(json_array(lower(strip_accents(val))), filter) DESC LIMIT 1
       ) AS score
FROM (VALUES
    ('Ireland'), ('Leitrim'), ('Carrick-on-Shannon'),
    ('NYC'), ('Bejing'), ('Stokholm'), ('Moskva'),
    ('Brasil'), ('Sao Paulo'), ('München'),
    ('Kairo'), ('Singapur'), ('St Petersburg'),
    ('USA'), ('UK'), ('PRC'),
    ('Calif'), ('Tex'), ('Fla')
) AS t(val);
