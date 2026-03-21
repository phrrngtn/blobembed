# The Resolution Sieve: Multi-Pass Classification Architecture

## The sieve

Each layer is cheaper and faster than the next. Data flows downward; only unresolved items reach the next layer. But critically, **results from lower layers feed back up** to refine the earlier passes.

```
         ┌─────────────────────────────────────────────┐
Pass 1   │  Normalization + Tokenization                │
         │  NFKD, casefold, split on _ and transitions  │
         │  Produces: whole tokens + sub-tokens          │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
Pass 2   │  blobfilters: known domain membership        │
         │  Probe tokens against per-level filters      │
         │  (countries, admin1, cities, months, etc.)    │
         │  Cost: O(1) per probe. Instant.               │
         │  Result: "95% of column X matches domain Y"   │
         └──────────────────────┬──────────────────────┘
                                │
              unresolved + domain confidence signals
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
Pass 3   │  Re-tokenize with domain context             │
         │  If 95% of a column is "country", re-tokenize │
         │  the remaining 5% more aggressively:          │
         │  try abbreviation expansion, prefix stripping, │
         │  language-specific normalization               │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
Pass 4   │  blobembed: semantic similarity               │
         │  Embed unresolved items, search against        │
         │  pre-computed index (HNSW, ~33ms/query)        │
         │  Uses hierarchical context from resolved       │
         │  columns: "? in Tamil Nadu, India" narrows     │
         │  the search space for the city column           │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
Pass 5   │  Semantic schema search: table matching       │
         │  Embed the table's column profile and search   │
         │  against known table schemas in Rule4          │
         │  "This looks like a table about locations +    │
         │  demographics" → match against similar tables  │
         │  that have already been classified              │
         └──────────────────────┬──────────────────────┘
                                │
              still unresolved (rare)
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
Pass 6   │  LLM via reify: ask for help                  │
         │  Send unresolved items with all context from   │
         │  passes 1-5. LLM sees what we've already       │
         │  figured out and fills the gaps.                │
         │  Results feed back permanently:                 │
         │  - New Wikidata domains → new blobfilters       │
         │  - New table schemas → Rule4 catalog            │
         └─────────────────────────────────────────────┘
```

## Multi-pass is essential

The sieve is not a single top-to-bottom waterfall. It's iterative:

**Pass 2 informs Pass 3.** If blobfilters determine that 95% of column A is a country name, the remaining 5% should be re-tokenized with country-specific heuristics: try ISO code expansion (PRC → China), try language-specific transliteration, try removing "Republic of" / "Kingdom of" prefixes.

**Pass 4 uses context from Pass 2.** When embedding the unresolved city "Thiruthuraipoondi", include the resolved context: "Thiruthuraipoondi, Tamil Nadu, India". The embedding for that string is much more specific than "Thiruthuraipoondi" alone. We saw this: adding "United States > California" to "San Francisco" jumped the score from 0.834 to 0.929.

**Pass 5 uses structure from all prior passes.** Once we know a table has columns for country, admin1, and city — and we know the countries are mostly South American — we can search Rule4 for similar table schemas: "tables with 3 geographic columns, predominantly South American data." That might find a previously-classified table with the same shape, giving us a complete classification template.

**Pass 2 filters evolve.** After Pass 6 discovers a new Wikidata domain, that domain becomes a new blobfilter in Pass 2. The sieve gets better with every table it processes.

## Re-tokenization

The first tokenization pass is generic: split on `_`, casefold, NFKD. But after we know the domain, we can re-tokenize more intelligently:

**For place names (after we know the column is geographic):**
- Strip common prefixes: "City of", "Republic of", "State of", "Province of"
- Expand abbreviations: "St" → "Saint", "Ft" → "Fort", "Mt" → "Mount"
- Try without hyphens: "Carrick-on-Shannon" → "Carrick on Shannon"
- Try without diacritics AND with: "São Paulo" → both "Sao Paulo" and "São Paulo"

**For column names (after we know the table domain):**
- If the table is about education, "sch_type" tokenizes differently than if it's about scheduling
- If we know adjacent columns, compound abbreviations resolve: "dept" next to "emp" suggests "department" not "departure"

**For data values (after we know the column's domain):**
- If column is "country", try ISO 3166 expansion on 2-letter and 3-letter codes
- If column is "currency", try ISO 4217 expansion
- If column is "language", try ISO 639 expansion

## The Rule4 catalog as a reference corpus

The most powerful feedback loop is Pass 5: matching against previously-classified tables.

Rule4 has scraped thousands of database schemas. Each table is a known, human-curated grouping of columns. When a new unknown table arrives, its column profile (the "topic bounding box" from the topic decomposition work) can be matched against the existing corpus:

```sql
-- New table arrives with columns: [country, region, city, population, area_sq_km]
-- Embed the column profile
-- Search against all known table profiles in Rule4

SELECT table_name, schema_name, database_name,
       be_cosine_sim(new_table_embedding, known_table_embedding) AS sim
FROM rule4_table_embeddings
ORDER BY sim DESC
LIMIT 5;

-- Top match: "gazetteer.geonames_place" with columns
-- [country_name, admin1_name, place_name, population, ...]
-- → We already know how to classify that table
-- → Copy the classification to the new table
```

This is the "unsupervised supervised" pattern: Rule4's structural metadata is the supervision, applied to new data without explicit labeling.

## Confidence cascading

Each pass produces a confidence score. The scores cascade:

| Pass | Signal | Confidence interpretation |
|---|---|---|
| blobfilter | containment = 1.0 | Exact match in the domain vocabulary |
| blobfilter | containment = 0.95 | 95% of column values are in the domain |
| embedding | cosine_sim > 0.90 | Very strong semantic match |
| embedding | cosine_sim 0.80-0.90 | Strong match, likely correct |
| embedding | cosine_sim 0.70-0.80 | Probable match, needs verification |
| schema match | cosine_sim > 0.85 | Table shape matches a known template |
| LLM | confidence field in response | Self-reported, needs calibration |

A high-confidence result from an early pass stops the sieve for that item. Low-confidence results propagate downward with the context of what was tried.

## Calibration from the GeoNames experiment

From testing 500 world cities against enriched blobfilters:

- **89-94% of cities** resolved at Pass 2 (instant, deterministic)
- **99% of countries** resolved at Pass 2
- **99% of admin1** resolved at Pass 2
- **100% of misses** resolved to the correct region at Pass 4 (embeddings, ~8ms each)
- **0% total failures** — every item resolved at some layer

The remaining optimization is reducing the 6-11% that reach Pass 4. Each new alternate name added to the blobfilters shrinks this percentage. The sieve gets tighter over time without any model retraining.
