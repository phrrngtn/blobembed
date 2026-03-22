# WordNet Taxonomy

~5,000 mid-level noun categories + 67 manual business/database categories extracted from WordNet, stored in `blobembed/data/wordnet_categories.yaml` (26k lines).

## Pipeline

`load_taxonomy.sql` in blobembed:
1. Parses YAML via blobtemplates (`bt_yaml_to_json`)
2. Embeds each category with nomic model
3. Creates HNSW cosine index in DuckDB

## Use Cases

- **Column pivot detection**: cluster column names by embedding similarity, match clusters against taxonomy to find category names
- **Schema-level topic detection**: embed all column names in a schema, find closest WordNet categories
- **Domain inference**: when a column's values don't match any known domain, use the taxonomy to suggest what domain it might belong to

## Storage (as of 2026-03-21)

Loaded into PG (`rule4_test`, schema `domain`):

| Table | Rows | Description |
|---|---|---|
| `domain.wordnet_category` | 5,315 | synset_id, category, hypernym, depth, gloss, embed_text |
| `domain.wordnet_category_embedding` | 5,315 | category, synset_id, model_name=`nomic`, embedding (768-dim FLOAT[]) |

67 manual business/database categories (depth=0), 5,248 WordNet categories (depths 3-8).

Embedding model: `nomic-ai/nomic-embed-text-v1.5-GGUF/Q8_0`. Embedded as "category: gloss" for rich semantic signal. Generated in 74s via blobembed on M4 Metal.

## Query Example

```sql
-- From DuckDB attached to PG:
SELECT e.category, c.hypernym, c.depth,
       be_cosine_sim(be_embed('nomic', 'order shipment'), e.embedding::FLOAT[]) AS sim
FROM pg.domain.wordnet_category_embedding AS e
JOIN pg.domain.wordnet_category AS c
    ON c.category = e.category AND COALESCE(c.synset_id,'') = COALESCE(e.synset_id,'')
ORDER BY sim DESC LIMIT 5
-- Returns: shipment (0.77), order (0.61), shipper (0.57), deliverer (0.55), batch (0.54)
```

## Semantic Test Results

| Query | Top match | Score |
|---|---|---|
| "Chicago" | city (municipality) | 0.56 |
| "revenue" | revenue (amount) | 0.84 |
| "patient diagnosis" | patient (case) | 0.73 |
| "order shipment" | shipment (delivery) | 0.77 |
| "employee salary" | employee (worker) | 0.69 |

## Links
- [[Domain Registry]]
- [[Blobembed]]
- [[Resolution Sieve Architecture]]
