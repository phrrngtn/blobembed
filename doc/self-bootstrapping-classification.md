# Self-Bootstrapping Column Classification

## The architecture

Column classification uses a three-layer detection stack where each layer handles what it's best at, and the most expensive layer (LLM) feeds back into the cheapest layer (blobfilters), so the system gets better over time without increasing cost.

```
                    ┌─────────────────────────────┐
                    │   New database arrives       │
                    │   (Rule4 / blobranges /      │
                    │    blobboxes scrape)          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────┐
   Layer 1    │  blobfilters: enumeration detection   │
   (instant,  │  Probe column names against known     │
   determini- │  domain filters (months, states,      │
   stic)      │  currencies, race categories, etc.)   │
              │  bf_containment_json → match score     │
              └────────────┬─────────────────────────┘
                           │
                    matched columns removed
                           │
                           ▼
              ┌──────────────────────────────────────┐
   Layer 2    │  Regex string factoring               │
   (instant,  │  Detect temporal pivots:              │
   determini- │  cy_2022_tco2e → {year, measure}      │
   stic)      │  Detect version suffixes: _1, _2      │
              └────────────┬─────────────────────────┘
                           │
                    factored columns removed
                           │
                           ▼
              ┌──────────────────────────────────────┐
   Layer 3    │  blobembed: semantic similarity        │
   (seconds,  │  Embed remaining columns, cluster by  │
   fuzzy)     │  cosine similarity, match clusters     │
              │  against WordNet taxonomy for category │
              │  naming. Classify dim vs measure.      │
              └────────────┬─────────────────────────┘
                           │
                   unrecognized columns remain
                           │
                           ▼
              ┌──────────────────────────────────────┐
   Layer 4    │  LLM via reify: domain discovery       │
   (slow,     │  Send unmatched column names to LLM.   │
   expensive, │  "What Wikidata domains should I       │
   bootstrap) │  fetch to classify these columns?"     │
              │  Returns: [{qid, domain_label}, ...]   │
              └────────────┬─────────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────────┐
   Feedback   │  blobhttp: fetch from Wikidata         │
   loop       │  SPARQL query for each new QID.        │
              │  Shred to relational form.             │
              │  Build blobfilters from member lists.   │
              │  Store in domain_enumerations table.    │
              └────────────┬─────────────────────────┘
                           │
                    new filters added to Layer 1
                           │
                           ▼
              ┌──────────────────────────────────────┐
              │  Next database that has these column  │
              │  patterns gets classified at Layer 1  │
              │  — deterministic, instant, no LLM.    │
              └──────────────────────────────────────┘
```

## Key principle: the LLM is the bootstrap, not the steady state

Each LLM call teaches the system a new domain permanently. The LLM identifies *which* Wikidata category to pull; Wikidata provides the authoritative member list; blobfilters makes it searchable. Over time:

- Layer 1 vocabulary grows (more domains recognized deterministically)
- Layer 4 gets called less (fewer unrecognized columns)
- Cost decreases as knowledge accumulates

The LLM is never trusted for the actual member lists — Wikidata is the source of truth. The LLM tells you where to look, not what the answer is.

## The reify adapter

The LLM call is a structured reify invocation with a JSON schema:

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "wikidata_qid": {"type": "string", "description": "Wikidata QID for the domain"},
      "domain_label": {"type": "string", "description": "Human-readable domain name"},
      "reason": {"type": "string", "description": "Why these columns match this domain"}
    },
    "required": ["wikidata_qid", "domain_label"]
  }
}
```

The prompt includes the unmatched column names (with frequency counts) and asks: "What Wikidata categories contain members that match these column names?" The LLM's training data includes knowledge of Wikidata's structure, so it can suggest QIDs directly.

## Data flow across the blob\* family

| Component | Role |
|---|---|
| **Rule4** | Scrapes database catalogs, extracts column names and metadata |
| **blobfilters** | Set-membership testing via bitmap containment — deterministic, O(1) per probe |
| **blobembed** | Semantic similarity via GGUF model embeddings — fuzzy, handles unknown patterns |
| **blobtemplates** | YAML parsing, JMESPath reshaping, JSON processing |
| **blobhttp** | Fetches Wikidata SPARQL results, HuggingFace models from MinIO |
| **blobapi/reify** | LLM calls via Bifrost for domain discovery (the bootstrap layer) |

## What makes this self-improving

1. **Every new database is training data.** Column names from Rule4 scrapes expand the frequency table, making the Socrata-style analysis ("what domains are most common?") more accurate over time.

2. **Every LLM call is permanent.** A new Wikidata domain fetched once serves all future databases. The ratio of LLM calls to databases classified decreases monotonically.

3. **The catalog IS the taxonomy.** Per the catalog-as-taxonomy design, columns-in-tables from Rule4 metadata are co-occurrence contexts. Tables that share column patterns cluster together in embedding space. New databases naturally find their neighbors.

4. **Wikidata evolves.** As Wikidata editors add and refine categories, re-fetching a QID updates the member list. The blobfilter can be rebuilt without any schema change.

## Current state

### Implemented and tested
- Layer 1: blobfilters enumeration detection (months, days, states, etc.) — scanned 11k Socrata tables, zero false positives
- Layer 2: Regex temporal factoring — decomposed NYC GHG Emissions 91 columns into 5 measures × 20 years
- Layer 3: blobembed semantic clustering — dimension/measure classification, cross-table similarity, taxonomy matching
- Domain enumerations: 18 domains (5 from Wikidata, 13 curated), 800 total members

### Not yet implemented
- Layer 4: reify adapter for LLM-driven domain discovery
- Automated Wikidata SPARQL fetch via blobhttp
- Feedback loop (new filters → Layer 1 vocabulary update)
- HNSW indexing of taxonomy embeddings for sub-linear search

## Calibration numbers

From experiments on Socrata metadata (319k columns, 11k resources, 5 domains):

| Signal | Score range | Interpretation |
|---|---|---|
| blobfilter containment ≥ 0.50 | Months in Libraries table | Definite enumeration match |
| Embedding avg_sim < 0.55 | Dimensions in GHG table | Column is not a pivoted measure |
| Embedding avg_sim > 0.55 | Measures in GHG table | Column is part of a repeated pattern |
| Pairwise sim > 0.95 | cy_2022_tco2e ↔ cy_2024_tco2e | Near-identical (temporal pivot) |
| Cross-table sim > 0.85 | Libraries 2014 ↔ Libraries 2026 | Same schema, different year |
| Cross-table sim 0.80–0.85 | CACFP ↔ Summer Meals | Related programs |
| Cross-table sim < 0.70 | Unrelated tables | No meaningful similarity |
