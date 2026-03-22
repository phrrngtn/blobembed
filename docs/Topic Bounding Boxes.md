# Topic Bounding Boxes

R-tree-like coarse-to-fine table matching using semantic topic compression. A table's columns are clustered into topics, producing a "minimum bounding rectangle" for its semantic content. Matching happens at the coarse topic level first, drilling into column-level comparison only for candidates that pass.

Full design doc: `~/checkouts/blobembed/doc/topic-bounding-boxes.md`

## The R-Tree Analogy

A table with 27 columns compresses to ~13 topics. Comparison becomes set intersection:
- **Bounding box overlap** (cheap): do the topic sets overlap? O(1) via blobfilters.
- **Topic overlap ratio** (moderate): 4/5 of A's topics appear in B.
- **Column-level similarity** (expensive): pairwise cosine similarity, only for top candidates.

## Worked Example

Chicago Grocery Stores (27 columns → 13 topics):
```
geolocation (2): latitude, longitude
address (2):     address, location_address
community (2):   community_area, community_area_name
postal_code (2): location_zip, zip_code
coordinate (2):  x_coordinate, y_coordinate
block (2):       census_block, census_tract
site (3):        location, location_city, location_state
ward (1), category (1), account (1), identifier (1), square (1)
```

## Why It Tolerates Messy Data

- **Extra columns**: 27 vs 43 columns — the 13-topic overlap still registers
- **Pivoted tables**: 91-column GHG emissions table → `{temporal (80), category (3), source (3), measure (5)}`
- **Naming variants**: `latitude` and `lat` cluster to the same `geolocation` topic
- **Missing columns**: partial overlap is a weaker signal, not a failure

## How Topics Are Built

1. [[Blobfilter Domain Probing]] strips known enumerations → topics like "month (12)"
2. [[Regex Domain Probing]] factors temporal patterns → "temporal (20)"
3. Blobembed clusters remaining columns → named topics via [[WordNet Taxonomy]]
4. The topic vector is the union of all three layers' outputs

## Connection to the Catalog-as-Taxonomy

Database catalogs are taxonomies: columns-in-tables = co-occurrence context, analogous to words-in-sentences. Embed `"Server.DB.Schema.Table: col1, col2, ..."` as a single string. Cosine similarity for table matching.

Full doc: `~/checkouts/blobembed/doc/catalog-as-taxonomy.md`

## Key SQL Files

- `~/checkouts/blobembed/sql/table_topics.sql` — topic extraction
- `~/checkouts/blobembed/sql/detect_pivoted_columns.sql` — pivot detection via embedding clustering
- `~/checkouts/blobembed/sql/experiment_socrata.sql` — experiments on Socrata datasets

## Reference Data in PG

| Table | Rows | Purpose |
|---|---|---|
| `domain.wordnet_category` | 5,315 | Category labels + glosses |
| `domain.wordnet_category_embedding` | 5,315 | nomic 768-dim vectors for category matching |
| `domain.member_embedding` | 13,217 | Domain member embeddings for fine-grained matching |

All in PG `rule4_test`, queryable from DuckDB via postgres scanner.

## Links
- [[Resolution Sieve Architecture]]
- [[WordNet Taxonomy]]
- [[Domain Registry]]
- [[Blobfilter Domain Probing]]
- [[Blobrule4 Project]]
