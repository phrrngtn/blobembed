-- Enumeration detection via blobfilters.
--
-- Detects when a table's column names match known enumerations
-- (months, days, quarters, US states, etc.) using bitmap containment.
--
-- Three probe strategies:
--   (a) whole column name
--   (b) individual tokens (split on _ and transitions)
--   (c) union of (a) and (b)
--
-- Prerequisites:
--   duckdb -unsigned

INSTALL postgres; LOAD postgres;
ATTACH 'dbname=rule4_test host=localhost gssencmode=disable'
    AS pg (TYPE POSTGRES, SCHEMA 'socrata');

SET allow_unsigned_extensions = true;
LOAD '/Users/paulharrington/checkouts/blobfilters/build/duckdb/blobfilters.duckdb_extension';

-- ═══════════════════════════════════════════════════════════════════
-- STEP 1: Build reference filters for known enumerations
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE TABLE enum_domains AS
SELECT domain_name, domain_label, bf_build_json(members_json) AS filter
FROM (VALUES
    ('months_long',
     'month',
     '["january","february","march","april","may","june","july","august","september","october","november","december"]'),
    ('months_short',
     'month',
     '["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]'),
    ('days_long',
     'day_of_week',
     '["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]'),
    ('days_short',
     'day_of_week',
     '["mon","tue","wed","thu","fri","sat","sun"]'),
    ('quarters',
     'quarter',
     '["q1","q2","q3","q4","quarter1","quarter2","quarter3","quarter4","qtr1","qtr2","qtr3","qtr4"]'),
    ('us_states_long',
     'us_state',
     '["alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan","minnesota","mississippi","missouri","montana","nebraska","nevada","new_hampshire","new_jersey","new_mexico","new_york","north_carolina","north_dakota","ohio","oklahoma","oregon","pennsylvania","rhode_island","south_carolina","south_dakota","tennessee","texas","utah","vermont","virginia","washington","west_virginia","wisconsin","wyoming"]'),
    ('us_states_abbrev',
     'us_state',
     '["al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia","ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv","wi","wy"]'),
    ('currencies',
     'currency',
     '["usd","eur","gbp","jpy","cny","cad","aud","chf","inr","brl","krw","mxn","sgd","hkd","nzd","sek","nok","dkk","zar","rub","try","pln","thb","idr","myr","php","czk","ils","clp","cop"]'),
    ('boolean_labels',
     'boolean',
     '["yes","no","true","false","y","n","t","f","on","off","active","inactive","enabled","disabled"]'),
    ('compass',
     'direction',
     '["north","south","east","west","ne","nw","se","sw","n","s","e","w","northeast","northwest","southeast","southwest"]'),
    ('age_groups',
     'age_group',
     '["infant","toddler","child","adolescent","teen","teenager","adult","senior","elderly","youth","minor","juvenile"]'),
    ('gender',
     'gender',
     '["male","female","man","woman","men","women","boy","girl","m","f","other","nonbinary","unknown"]')
) AS t(domain_name, domain_label, members_json);

SELECT domain_name, domain_label, bf_cardinality(filter) AS n_members
FROM enum_domains
ORDER BY domain_name;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 2: Test against the Chicago Libraries table (the month problem)
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Test: Chicago Libraries 2026 Visitors ==='

-- Get column names
CREATE OR REPLACE TEMP TABLE test_cols AS
SELECT rc.field_name, lower(rc.field_name) AS col_lower
FROM pg.resource_column AS rc
WHERE rc.domain = 'data.cityofchicago.org'
  AND rc.resource_id = 'gy5m-7w2w'
  AND rc.tt_end = '9999-12-31';

-- (a) Probe with whole column names
.print ''
.print 'Strategy (a): whole column names'
WITH COL_JSON AS (
    SELECT json_group_array(col_lower) AS arr FROM test_cols
)
SELECT ed.domain_name, ed.domain_label,
       bf_containment_json(cj.arr, ed.filter) AS containment,
       bf_intersection_card(bf_build_json(cj.arr), ed.filter) AS n_matches
FROM enum_domains AS ed, COL_JSON AS cj
WHERE bf_containment_json(cj.arr, ed.filter) > 0
ORDER BY containment DESC;

-- (b) Probe with individual tokens (split column names on _)
.print ''
.print 'Strategy (b): individual tokens'
WITH TOKENS AS (
    SELECT DISTINCT unnest(string_split(col_lower, '_')) AS token
    FROM test_cols
    WHERE length(unnest(string_split(col_lower, '_'))) > 0
),
TOK_JSON AS (
    SELECT json_group_array(token) AS arr FROM TOKENS
)
SELECT ed.domain_name, ed.domain_label,
       bf_containment_json(tj.arr, ed.filter) AS containment,
       bf_intersection_card(bf_build_json(tj.arr), ed.filter) AS n_matches
FROM enum_domains AS ed, TOK_JSON AS tj
WHERE bf_containment_json(tj.arr, ed.filter) > 0
ORDER BY containment DESC;

-- (c) Probe with union of whole names + tokens
.print ''
.print 'Strategy (c): whole names + tokens combined'
WITH ALL_TERMS AS (
    SELECT DISTINCT term FROM (
        SELECT col_lower AS term FROM test_cols
        UNION ALL
        SELECT unnest(string_split(col_lower, '_')) AS term
        FROM test_cols
    )
    WHERE length(term) > 0
),
ALL_JSON AS (
    SELECT json_group_array(term) AS arr FROM ALL_TERMS
)
SELECT ed.domain_name, ed.domain_label,
       bf_containment_json(aj.arr, ed.filter) AS containment,
       bf_intersection_card(bf_build_json(aj.arr), ed.filter) AS n_matches
FROM enum_domains AS ed, ALL_JSON AS aj
WHERE bf_containment_json(aj.arr, ed.filter) > 0
ORDER BY containment DESC;

-- Show which columns matched which enumeration
.print ''
.print 'Detail: which columns matched months?'
SELECT tc.field_name, tc.col_lower,
       bf_containment_json(json_array(tc.col_lower),
           (SELECT filter FROM enum_domains WHERE domain_name = 'months_long')
       ) AS is_month_name
FROM test_cols AS tc
ORDER BY is_month_name DESC, tc.field_name;

-- ═══════════════════════════════════════════════════════════════════
-- STEP 3: Batch test across multiple tables
-- ═══════════════════════════════════════════════════════════════════

.print ''
.print '=== Batch: probe all target tables against all enumerations ==='

CREATE OR REPLACE TEMP TABLE batch_targets AS
WITH RANKED AS (
    SELECT rc.domain, rc.resource_id,
           count(*) AS n_cols,
           ROW_NUMBER() OVER (PARTITION BY rc.domain ORDER BY count(*) DESC) AS rn
    FROM pg.resource_column AS rc
    WHERE rc.tt_end = '9999-12-31'
    GROUP BY rc.domain, rc.resource_id
    HAVING count(*) BETWEEN 10 AND 100
)
SELECT t.domain, t.resource_id, t.n_cols, r.name AS table_name
FROM RANKED AS t
JOIN pg.resource AS r
    ON r.domain = t.domain AND r.resource_id = t.resource_id
    AND r.tt_end = '9999-12-31'
WHERE t.rn <= 5;

-- Build per-table column JSON + token JSON
CREATE OR REPLACE TEMP TABLE table_probes AS
WITH COLS AS (
    SELECT rc.domain, rc.resource_id,
           lower(rc.field_name) AS col_lower
    FROM pg.resource_column AS rc
    WHERE rc.tt_end = '9999-12-31'
      AND (rc.domain, rc.resource_id) IN (
          SELECT domain, resource_id FROM batch_targets
      )
),
TOKENS AS (
    SELECT domain, resource_id,
           unnest(string_split(col_lower, '_')) AS token
    FROM COLS
),
ALL_TERMS AS (
    SELECT DISTINCT domain, resource_id, term FROM (
        SELECT domain, resource_id, col_lower AS term FROM COLS
        UNION ALL
        SELECT domain, resource_id, token FROM TOKENS WHERE length(token) > 1
    )
)
SELECT domain, resource_id,
       json_group_array(term) AS terms_json
FROM ALL_TERMS
GROUP BY domain, resource_id;

-- Probe each table against each enumeration
SELECT bt.table_name, ed.domain_label,
       bf_containment_json(tp.terms_json, ed.filter) AS containment,
       bf_intersection_card(bf_build_json(tp.terms_json), ed.filter) AS n_matches,
       bt.n_cols
FROM table_probes AS tp
JOIN batch_targets AS bt USING (domain, resource_id)
CROSS JOIN enum_domains AS ed
WHERE bf_containment_json(tp.terms_json, ed.filter) > 0.15
ORDER BY containment DESC
LIMIT 30;
