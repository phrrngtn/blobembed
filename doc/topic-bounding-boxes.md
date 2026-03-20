# Topic Bounding Boxes: Coarse-to-Fine Table Matching

## The idea

A table with 27 columns is hard to compare directly against another table with 43 columns. But if you compress each table to its semantic topics — "this table is about locations, postal codes, and identifiers" — the comparison becomes a set intersection.

This is the R-tree analogy: the topic vector is the **minimum bounding rectangle** for a table's semantic content. You match at the coarse level first (do the bounding boxes overlap?), then drill into fine-grained column-level comparison only for candidates that pass.

## Worked example

The Chicago Grocery Stores table (27 columns) compresses to 13 topics:

```
Grocery Stores - 2011:
  geolocation (2): latitude, longitude
  address (2):     address, location_address
  community (2):   community_area, community_area_name
  postal_code (2): location_zip, zip_code
  coordinate (2):  x_coordinate, y_coordinate
  block (2):       census_block, census_tract
  region (4):      :@computed_region_* columns
  site (3):        location, location_city, location_state
  ward (1):        ward
  category (1):    size_category
  account (1):     account_number
  identifier (1):  license_id
  square (1):      square_feet
```

27 columns → 13 topics. The table is "about": **locations, postal codes, census geography, and a grocery store identifier with a size category.**

## The R-tree analogy

```
             Coarse level (topic bounding boxes)
             ┌──────────────────────────────────┐
             │  {geolocation, postal_code,       │
Table A ──── │   address, identifier, amount}    │
             └───────────────┬──────────────────┘
                             │
                     overlap? ──── yes → drill in
                             │
             ┌───────────────┴──────────────────┐
             │  {geolocation, postal_code,       │
Table B ──── │   address, identifier, status}    │
             └──────────────────────────────────┘

             Fine level (column embeddings)
             Table A.geolocation = {latitude, longitude}
             Table B.geolocation = {lat, lng, location_point}
             → cosine similarity confirms match
```

- **Bounding box intersection** is cheap: set overlap of topic labels. Could even be a blobfilter containment check.
- **Point-in-polygon** (column-level similarity) is expensive: requires embedding each column and computing pairwise cosine similarity.
- You only pay the expensive cost for candidate pairs that pass the cheap filter.

## Why this tolerates messy real-world data

**Extra columns:** Table A has 27 columns, Table B has 43. At the topic level, B might have the same 13 topics as A plus 5 more. The 13-topic overlap still registers as a strong match — the extra topics are noise, not disqualifiers.

**Pivoted tables:** The 91-column NYC GHG emissions table compresses to `{temporal (80), category (3), source (3), measure (5)}`. Another emissions dataset with different years, different column naming conventions, or even a different number of year columns would compress to the same topic set. The pivot structure is absorbed by the topic compression.

**Different naming conventions:** `latitude` and `lat` end up in the same `geolocation` topic because embedding similarity clusters them together before topic labelling. The topic label is the stable anchor; the column names are the noisy details.

**Missing columns:** If Table B lacks `census_block` and `census_tract`, it just won't have a `block` topic. The remaining topics still match. Partial overlap is a weaker signal, not a failure.

## The layered cost model

| Level | What's compared | Cost per pair | When to use |
|---|---|---|---|
| Topic set intersection | {geolocation, postal_code, ...} | O(1) via blobfilters | All pairs — this is the filter |
| Topic overlap ratio | 4/5 of A's topics in B | O(n_topics) | Candidate ranking |
| Column-level embedding sim | latitude ↔ lat = 0.89 | O(n_cols²) per pair | Verification of top candidates |

The first level eliminates most pairs. A table about `{crime, gender, school_type}` will have zero topic overlap with `{geolocation, postal_code, identifier}` and is immediately discarded.

## Connection to the classification stack

This fits into the existing architecture:

1. **blobfilters** strips known enumerations (months, states) → these become topics like "month (12)"
2. **Regex** factors temporal patterns → these become "temporal (20)"
3. **blobembed** clusters remaining columns → these become named topics via taxonomy
4. **The topic vector** is the union of all three layers' outputs

The topic vector is itself storable as a Rule4 extended property, searchable via blobfilters (build a filter from the topic labels), and embeddable (embed the topic list as a sentence for semantic search).

## Open questions

- What's the right topic granularity? Too coarse ("this table has locations") matches everything. Too fine ("this table has census_tract") is just column names again.
- Should the topic vector include the count per topic? `{geolocation: 2}` vs `{geolocation: 47}` are very different tables even if the topic labels match.
- Can the topic overlap ratio be calibrated into a probability? "4/5 topic overlap → 80% chance these tables are related" would make the filter tunable.
