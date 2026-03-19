# blobembed

> **Note:** This code is almost entirely AI-authored (Claude, Anthropic), albeit under close human supervision, and is for research and experimentation purposes. Successful experiments may be re-implemented in a more coordinated and curated manner.

In-database text embedding via GGUF models, exposed as scalar SQL functions for SQLite and DuckDB, with a shared C core. Part of the blob\* extension family (blobtemplates, blobboxes, blobfilters, blobhttp, blobodbc).

## What it does

Given a local GGUF model file, blobembed tokenizes input text, runs the encoder forward pass, and returns a float array (embedding vector) — all within a SQL query, no Python runtime required.

```sql
SELECT blob_embed('nomic-v1.5', 'quarterly revenue by region') AS embedding;
-- Returns: FLOAT[] of dimension 768
```

The core C library wraps [llama.cpp](https://github.com/ggerganov/llama.cpp)'s embedding API (~10 functions). Models are loaded once and reused across calls. The lifecycle is load-once / encode-many with KV cache clearing between calls.

## Architecture

blobembed is a **pure compute primitive**: text in, float array out. It does not manage model acquisition, storage, or versioning. That separation is deliberate:

### Model flow

1. **MinIO** — on-premises model registry. Source of truth for approved/vetted GGUF models. Nothing is fetched from the public internet at inference time.
2. **HF catalog TTST** — temporal table tracking HuggingFace Hub model catalog evolution (metadata, quantization variants, parameter counts, popularity). The delta between "exists on HF Hub" and "exists in MinIO" serves as an approval queue.
3. **blobhttp** — fetches GGUF files from MinIO into the local HuggingFace cache (`$HF_HUB_CACHE`, default `~/.cache/huggingface/hub/`), following the standard HF cache layout so that Python tooling (`hf cache ls`, `hf cache rm`) continues to work.
4. **blobembed** — resolves a model ID to a cache path and calls `llama_model_load_from_file`. mmap does the rest.

### Why the HF cache layout?

The [HuggingFace Hub cache](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) uses a simple, stable, content-addressed filesystem structure:

```
$HF_HUB_CACHE/
  models--nomic-ai--nomic-embed-text-v1.5-GGUF/
    refs/
      main                          # text file: commit hash
    blobs/
      <sha256>                      # actual file bytes
    snapshots/
      <commit-hash>/
        nomic-embed-text-v1.5.Q4_K_M.gguf  → ../../blobs/<sha256>
```

No database, no binary index — the filesystem is the API. Resolving a model: read `refs/main` → get commit hash → follow `snapshots/<hash>/<filename>` → symlink resolves to `blobs/<sha256>`.

By writing into this layout, blobhttp gives you:
- **Deduplication** — content-addressed by SHA256
- **No re-download** — models already fetched by Python tooling are reused
- **Integrity verification** — SHA256 from the catalog TTST can be checked against what MinIO serves
- **Air-gap support** — MinIO on the local network, nothing phones home
- **Standard management** — `hf cache ls` and `hf cache rm` work as expected

### Why MinIO instead of pulling from HF Hub directly?

Someone has to explicitly put a model into MinIO, which means it has been vetted. The catalog TTST tracks what is available on HF Hub; MinIO tracks what your organization has approved for on-premises use.

## llama.cpp embedding API

The interface blobembed wraps is compact:

```c
// Load model (mmap by default)
model = llama_model_load_from_file("model.gguf", params);
ctx   = llama_init_from_model(model, {.embeddings = true, .pooling_type = MEAN});

// Per-call: tokenize → encode → extract
tokens  = llama_tokenize(vocab, text, len, buf, max, true, true);
batch   = llama_batch_get_one(tokens, n_tokens);
llama_kv_cache_clear(ctx);
llama_encode(ctx, batch);
float *embd = llama_get_embeddings_seq(ctx, 0);
int dim     = llama_model_n_embd(model);  // e.g. 768
// L2-normalize, return
```

llama.cpp also exposes a callback-based loader (`llama_model_init_from_user`) that could load tensor data from any source (memory, network, database blob) via per-tensor callbacks. This is not needed for the embedding use case (small models, load-once) but exists if lazy-loading of large models from object storage ever becomes relevant.

## Target models

| Model | Params | Dimensions | Notes |
|-------|--------|------------|-------|
| nomic-embed-text-v1.5-GGUF | 137M | 768 | Sweet spot: small, CPU-friendly, multiple quantization levels |
| nomic-embed-text-v2-moe-GGUF | 475M | — | MoE architecture, newer |
| all-MiniLM-L12-v2-GGUF | 33M | 384 | Smallest/fastest |
| bge-m3-GGUF | 568M | 1024 | Multilingual |

## Prior art and acknowledgments

This project builds directly on the work of [Alex Garcia](https://github.com/asg017), whose SQLite extensions established the pattern blobembed follows:

- **[sqlite-lembed](https://github.com/asg017/sqlite-lembed)** — SQLite extension for local text embeddings via llama.cpp. Single-file C implementation that loads GGUF models, caches model+context, and exposes `lembed(model, text) → BLOB`. This is the direct reference implementation for blobembed's SQLite wrapper.
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)** — SQLite vector search extension. The natural complement to blobembed on the SQLite side (embed with blobembed, index/search with sqlite-vec).
- **[sqlite-http](https://github.com/asg017/sqlite-http)** — HTTP client as SQLite functions. Shares the philosophy of bringing capabilities into the database rather than shelling out to external processes.

## HuggingFace Hub API

The HF Hub publishes an [OpenAPI 3.1.0 spec](https://huggingface.co/.well-known/openapi.json) ([markdown](https://huggingface.co/.well-known/openapi.md), [playground](https://huggingface.co/spaces/huggingface/openapi)) covering webhooks, org management, tags, trending, and search endpoints. The main `/api/models` catalog endpoint is undocumented but stable — it is the backbone of the `huggingface_hub` Python client.

Key catalog endpoints:

| Endpoint | In OpenAPI spec | Purpose |
|----------|-----------------|---------|
| `GET /api/models?limit=N&full=true` | No (stable, undocumented) | List/search models with pagination |
| `GET /api/models/{namespace}/{repo}` | No (stable, undocumented) | Single model full metadata |
| `GET /api/models-tags-by-type` | Yes | Tag taxonomy (pipeline, library, license, language) |
| `GET /api/trending?type=model` | Yes | Trending models |

Pagination uses `Link` header with `rel="next"` (cursor-based). The `?full=true` variant returns `sha`, `lastModified`, `siblings` (file list), and `safetensors` metadata (parameter counts by dtype).

## Build

CMake + FetchContent, pulling llama.cpp as a dependency. Same pattern as the other blob\* extensions.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Status

Design phase. No code yet.
