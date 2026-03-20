# Reflective Architecture: Metadata About Metadata

## What it is

The blobembed/Rule4 classification system is **reflective** (also called **meta-circular**): it uses its own mechanisms to describe and extend itself.

The vocabulary that classifies columns is itself classified data. The domain enumerations that detect known patterns were themselves detected. The table topic profiles could themselves serve as features for classifying the next table. Every layer's output can become another layer's input.

## Why it feels familiar

Rule4 already does this with SQL Server's `sys.extended_properties`. When you store a `survey.classification = 'dimension'` property on a column, you're using the database to store metadata *about* the database. The metadata lives alongside the data it describes, queryable with the same SQL, joinable with the same engine.

The blobembed system extends this to semantic metadata:

| Layer | What it does | Where it lives | How it evolves |
|---|---|---|---|
| Extended properties | Classification labels on columns | `sys.extended_properties` | DBA writes them, Rule4 automates |
| Domain enumerations | Member lists for known categories | `domain.enumeration` in PG | Seeded from Wikidata, grows via LLM discovery |
| WordNet taxonomy | Category names for embedding clusters | YAML → DuckDB table | Static seed, could be extended from catalog |
| Table topic profiles | Compressed semantic fingerprints | Computed at query time | Each new table enriches the corpus |
| Embedding vectors | Numeric representations of meaning | FLOAT[768] in DuckDB/PG | Pre-trained model, fixed per model version |

Each layer uses the layers below it. The table topic profiles (layer 4) are built from embedding vectors (layer 5) matched against the WordNet taxonomy (layer 3), after domain enumerations (layer 2) have stripped known patterns. And the extended properties (layer 1) are the final consumer that stores the results.

## The self-bootstrapping loop

The most explicitly reflective part is the LLM-driven domain discovery:

```
System doesn't recognize columns
  → Asks LLM: "What Wikidata categories cover these?"
  → LLM says: "Q11344 (chemical elements)"
  → System fetches Q11344 from Wikidata
  → Builds a blobfilter from the member list
  → Stores filter in domain.enumeration
  → Next time: recognizes chemical elements without asking
```

The system teaches itself new recognition patterns using its own classification infrastructure. The LLM is consulted once; the result is permanent. This is **learning by extending the vocabulary**, not by retraining a model.

## The catalog-as-taxonomy connection

The deepest form of reflexivity is using the database catalog itself as a semantic taxonomy (see `catalog-as-taxonomy.md`). When Rule4 scrapes column names from SQL Server, those column names become the vocabulary for classifying column names from other databases. The metadata about Database A helps classify Database B.

This works because database designers encode domain knowledge in their naming conventions. The column list of `Sales.SalesOrderHeader` — `OrderDate, CustomerID, TotalDue, SubTotal, TaxAmt` — is a human-curated description of what a sales order looks like. By embedding that description, we make it searchable. By storing the embedding alongside the metadata, we make it joinable.

The system's knowledge grows with every database it encounters:
- More tables scraped → richer embedding corpus → better table matching
- More columns classified → richer domain enumerations → fewer LLM calls
- More topic profiles → better cross-database discovery → more connections

## Formal parallels

The computer science term for this is a **reflective system** — one that can inspect and modify its own structure. Classic examples:

- **Lisp**: programs are data structures that programs can manipulate
- **Smalltalk**: classes are objects that respond to messages
- **SQL Server**: `sys.objects` is a table that describes all tables, including itself

The blobembed system adds:
- **Embedding vectors**: meaning is data that queries can compute over
- **blobfilters**: set membership is data that queries can test
- **LLM via reify**: classification rules are data that queries can extend

The common thread: the distinction between "the system" and "data in the system" dissolves. Everything is a table. Everything is joinable. Everything is queryable. And everything can contribute to classifying everything else.

## Practical implications

1. **No separate training step.** The system doesn't need a training pipeline. It learns by accumulating data through its normal operation (scraping catalogs, classifying columns, fetching from Wikidata).

2. **Graceful degradation.** If the LLM is unavailable, Layers 1-3 still work. If the embedding model changes, the domain enumerations (blobfilters) still work. Each layer is independent.

3. **Auditability.** Every classification decision is traceable: which blobfilter matched, which taxonomy category scored highest, which LLM call suggested which Wikidata QID. The `domain.discovery_log` table records the provenance.

4. **No cold start problem.** The system ships with 18 domains (800 members from Wikidata + curated) and 5,315 WordNet categories. It can classify columns from day one. The self-bootstrapping loop makes it better over time, but it's useful immediately.
