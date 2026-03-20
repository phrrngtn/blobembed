-- Load GeoNames hierarchy: countries → admin1 → admin2 → cities
-- Sources: official TSV files + HuggingFace parquet
-- Produces a single joined table: geonames_places

INSTALL httpfs; LOAD httpfs;

-- ── Country info ───────────────────────────────────────────────────
-- countryInfo.txt has # comment lines that break DuckDB's sniffer.
-- Pre-stripped to data/geonames/countryInfo.tsv (grep -v '^#')
CREATE OR REPLACE TABLE geonames_countries AS
SELECT * FROM read_csv(
    'data/geonames/countryInfo.tsv',
    delim = '\t', header = false, quote = '',
    columns = {
        'iso': 'VARCHAR', 'iso3': 'VARCHAR', 'iso_numeric': 'VARCHAR',
        'fips': 'VARCHAR', 'country': 'VARCHAR', 'capital': 'VARCHAR',
        'area_sq_km': 'VARCHAR', 'population': 'VARCHAR', 'continent': 'VARCHAR',
        'tld': 'VARCHAR', 'currency_code': 'VARCHAR', 'currency_name': 'VARCHAR',
        'phone': 'VARCHAR', 'postal_code_format': 'VARCHAR', 'postal_code_regex': 'VARCHAR',
        'languages': 'VARCHAR', 'geonameid': 'VARCHAR', 'neighbours': 'VARCHAR',
        'equivalent_fips_code': 'VARCHAR'
    }
);

.print '=== Countries ==='
SELECT count(*) AS n FROM geonames_countries;

-- ── Admin1 (states, provinces, regions) ────────────────────────────
CREATE OR REPLACE TABLE geonames_admin1 AS
SELECT * FROM read_csv(
    'https://download.geonames.org/export/dump/admin1CodesASCII.txt',
    delim = '\t', header = false, quote = '',
    columns = {
        'code': 'VARCHAR',
        'name': 'VARCHAR',
        'asciiname': 'VARCHAR',
        'geonameid': 'INTEGER'
    }
);

.print '=== Admin1 ==='
SELECT count(*) AS n FROM geonames_admin1;

-- ── Admin2 (counties, departments) ─────────────────────────────────
CREATE OR REPLACE TABLE geonames_admin2 AS
SELECT * FROM read_csv(
    'https://download.geonames.org/export/dump/admin2Codes.txt',
    delim = '\t', header = false, quote = '',
    columns = {
        'code': 'VARCHAR',
        'name': 'VARCHAR',
        'asciiname': 'VARCHAR',
        'geonameid': 'INTEGER'
    }
);

.print '=== Admin2 ==='
SELECT count(*) AS n FROM geonames_admin2;

-- ── Cities from HuggingFace parquet (numeric column names: 0..18) ──
CREATE OR REPLACE TABLE geonames_cities AS
SELECT "0"::INTEGER AS geonameid,
       "1" AS name,
       "2" AS asciiname,
       "3" AS alternatenames,
       "4"::DOUBLE AS latitude,
       "5"::DOUBLE AS longitude,
       "6" AS feature_class,
       "7" AS feature_code,
       "8" AS country_code,
       "10" AS admin1_code,
       "11" AS admin2_code,
       "12" AS admin3_code,
       "14"::BIGINT AS population,
       "17" AS timezone
FROM read_parquet('https://huggingface.co/api/datasets/do-me/Geonames/parquet/default/train/0.parquet')
WHERE "6" = 'P'          -- populated places
  AND "14"::BIGINT >= 1000;

.print '=== Cities (pop >= 1000) ==='
SELECT count(*) AS n FROM geonames_cities;

-- ── Joined hierarchy ───────────────────────────────────────────────
CREATE OR REPLACE TABLE geonames_places AS
SELECT c.geonameid,
       c.name AS place_name,
       c.asciiname AS place_ascii,
       c.alternatenames,
       c.latitude, c.longitude,
       c.feature_code,
       c.population,
       c.country_code,
       co.country AS country_name,
       co.iso3 AS country_iso3,
       co.continent,
       a1.name AS admin1_name,
       a2.name AS admin2_name,
       co.country
           || CASE WHEN a1.name IS NOT NULL THEN ' > ' || a1.name ELSE '' END
           || CASE WHEN a2.name IS NOT NULL THEN ' > ' || a2.name ELSE '' END
           || ' > ' || c.name
       AS full_path
FROM geonames_cities AS c
LEFT JOIN geonames_countries AS co ON co.iso = c.country_code
LEFT JOIN geonames_admin1 AS a1
    ON a1.code = c.country_code || '.' || c.admin1_code
LEFT JOIN geonames_admin2 AS a2
    ON a2.code = c.country_code || '.' || c.admin1_code || '.' || c.admin2_code;

.print '=== Joined places ==='
SELECT count(*) AS n FROM geonames_places;

.print ''
.print '=== Ireland ==='
SELECT full_path, population, latitude, longitude
FROM geonames_places
WHERE country_code = 'IE'
ORDER BY population DESC
LIMIT 10;

.print ''
.print '=== Hierarchy coverage ==='
SELECT count(*) AS total_places,
       count(country_name) AS have_country,
       count(admin1_name) AS have_admin1,
       count(admin2_name) AS have_admin2
FROM geonames_places;
