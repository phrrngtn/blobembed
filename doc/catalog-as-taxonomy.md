# The Database Catalog as a Semantic Taxonomy

## The observation

A well-maintained database catalog is a taxonomy that nobody thinks of as a taxonomy.

When a DBA organizes columns into tables, tables into schemas, and schemas into databases, they are performing a classification act: "these columns co-occur because they describe the same business entity." That structure — built up over years of careful schema design, code review, and migration — encodes deep domain knowledge about how concepts relate to each other.

This is different from a linguistic taxonomy like WordNet, which organizes words by semantic relationships ("banana is-a fruit"). A database catalog organizes *identifiers* by structural co-occurrence ("OrderDate appears alongside CustomerID and TotalDue in SalesOrderHeader"). Both are useful. They answer different questions:

- **WordNet**: "What kind of thing is this?" → banana is a fruit
- **Catalog**: "What does this appear with?" → OrderDate appears with CustomerID in a sales context

## The approach

Embed the catalog hierarchy at every level, with each level inheriting context from its parents:

```
embed("PROD.AdventureWorks.Sales.SalesOrderHeader: OrderDate, CustomerID, TotalDue, ...")
```

The full path provides namespace disambiguation. The column list provides the semantic fingerprint of the table. The embedding captures both.

When an unknown set of columns arrives — say, from a CSV, a Socrata dataset, or a partner's API — embed it the same way and search for the nearest known table. The cosine similarity score serves as a confidence measure: high similarity means the unknown data likely represents the same business entity as the matched table, even if the column names differ.

## Why this might work

The intuition comes from years of database work, not from ML training, but it aligns with established information retrieval concepts:

**Distributional semantics.** The core idea in NLP is that words which appear in similar contexts have similar meanings ("you shall know a word by the company it keeps" — Firth, 1957). Columns that appear in the same table are in the same context. A table *is* a context window for its columns, in exactly the way a sentence is a context window for its words.

**Transfer learning.** The embedding model (nomic-embed, trained on billions of words) already knows that "OrderDate" relates to "order" and "date," that "CustomerID" relates to "customer" and "identifier." When you embed a column list, the model brings this general knowledge to bear on your specific schema. You don't need to train anything — the model's pre-existing knowledge of English maps onto database naming conventions because database designers name things in English.

**Supervised structure, unsupervised extraction.** The catalog was built by humans making deliberate design decisions — that's the supervised part. Extracting it via Rule4's metadata queries is automated — that's the unsupervised part. The result is a richly structured dataset that nobody had to hand-label for ML purposes, because the structure *was the purpose*.

## What it enables

### Table matching

Given unknown columns `[order_date, cust_id, salesperson, total_amt]`, find the nearest known table. If `Sales.SalesOrderHeader` scores 0.86 and the next candidate scores 0.71, that's a strong match with clear daylight.

### Column pivot detection

Within a single table, cluster column names by embedding similarity. Columns that form a tight cluster (e.g., `bananas, apples, kiwi, mango` all score >0.80 against each other) are likely instances of a category. Columns outside the cluster (e.g., `date, customer`) are likely dimensions. The category name can be inferred by matching the cluster centroid against a vocabulary of known categories — either from WordNet or from column names already in the Rule4 catalog.

### Schema evolution tracking

Embed the same table at different points in time (from the catalog TTST). If the embedding drifts significantly between snapshots, the table's semantic role may have changed — a signal worth flagging.

### Cross-database discovery

Embed tables from different servers, databases, or organizations. Tables with high cosine similarity likely represent the same business entity, even if they come from completely different systems with different naming conventions. This is the catalog equivalent of entity resolution.

## The hierarchy as levels of abstraction

```
Level 1: Server     "PROD"
Level 2: Database   "PROD.AdventureWorks"
Level 3: Schema     "PROD.AdventureWorks.Sales"
Level 4: Table      "PROD.AdventureWorks.Sales.SalesOrderHeader: OrderDate, CustomerID, ..."
Level 5: Column     "PROD.AdventureWorks.Sales.SalesOrderHeader.OrderDate (datetime, dimension)"
```

Each level carries more specificity. The table level is the sweet spot for most matching tasks because it contains both the hierarchical context (where this table lives) and the content fingerprint (what columns it has). But the other levels are useful too:

- **Schema-level** embeddings can detect "this unknown dataset probably belongs in the Sales schema"
- **Column-level** embeddings enable the finest-grained matching for individual column mapping

## Complementary signals

Embeddings are not the only signal, and probably not the strongest one for every case. The existing classification hierarchy from CLAUDE.md still applies:

1. **Foreign key membership** — deterministic, highest confidence
2. **Primary key structure** — deterministic
3. **Histogram shape** — statistical, from `sys.dm_db_stats_histogram`
4. **Data type** — heuristic
5. **Embedding similarity** — semantic, the new addition

The embedding signal is most valuable where the deterministic signals are absent — which is exactly the "columns from the wild" scenario, where you have names but no FK relationships or statistics. It fills the gap between "I know exactly what this is" and "I have no idea."

## Open questions

- **What similarity threshold indicates a reliable match?** Early experiments suggest 0.80+ for same-table-different-names, but this needs calibration against more data.
- **How does the embedding approach degrade with table width?** Very wide tables (100+ columns) may produce embeddings dominated by a few strong signals. Does truncating to the top-N most distinctive columns help?
- **Can the approach detect denormalized temporal patterns** (e.g., `2024_sales, 2025_sales, 2026_sales`) purely from embeddings, or does that require string factoring as a separate step?
- **How should Rule4 catalog embeddings and WordNet category embeddings be combined?** Separate searches with merged results? A single index? Weighted scoring?
