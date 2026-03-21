/*
 * blobembed core: wraps llama.cpp's embedding API.
 *
 * Process-wide model registry with thread-safe access.
 * Each model entry holds a llama_model + llama_context pair.
 * The context is created with embeddings=true and mean pooling.
 */

#include "blobembed.h"

#include "llama.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

/* ── Thread-local error state ──────────────────────────────────────── */

static thread_local std::string g_errmsg;

static void set_error(const char *msg) {
    g_errmsg = msg ? msg : "unknown error";
}

static void set_error(const std::string &msg) {
    g_errmsg = msg;
}

extern "C" const char *blobembed_errmsg(void) {
    return g_errmsg.c_str();
}

/* ── Model registry ────────────────────────────────────────────────── */

/*
 * Context pool: multiple llama_context instances sharing one model.
 * Each DuckDB thread acquires a context, uses it for a chunk of rows,
 * then releases it. The model weights are shared (mmap'd, read-only).
 * Each context has its own KV cache (~72MB for nomic).
 */
static constexpr int MAX_POOL_SIZE = 16;

struct ContextPool {
    llama_context *contexts[MAX_POOL_SIZE];
    std::atomic<bool> in_use[MAX_POOL_SIZE];
    int size;

    ContextPool() : size(0) {
        for (int i = 0; i < MAX_POOL_SIZE; i++) {
            contexts[i] = nullptr;
            in_use[i].store(false);
        }
    }

    /* Acquire a free context. Spins briefly, then yields. */
    int acquire() {
        while (true) {
            for (int i = 0; i < size; i++) {
                bool expected = false;
                if (in_use[i].compare_exchange_weak(expected, true)) {
                    return i;
                }
            }
            std::this_thread::yield();
        }
    }

    void release(int idx) {
        llama_kv_self_clear(contexts[idx]);
        in_use[idx].store(false);
    }
};

struct ModelEntry {
    llama_model *model;
    ContextPool pool;
    int n_embd;
    int n_ctx;     /* context window size (max tokens per encode) */
};

static std::shared_mutex g_registry_mu;
static std::unordered_map<std::string, ModelEntry *> g_registry;

static ModelEntry *find_model(const char *name) {
    std::shared_lock lock(g_registry_mu);
    auto it = g_registry.find(name);
    if (it == g_registry.end()) return nullptr;
    return it->second;
}

/* ── HF cache resolution ──────────────────────────────────────────── */

/*
 * Resolve a model file path from the HuggingFace cache.
 *
 * Layout: $HF_HUB_CACHE/models--{org}--{repo}/snapshots/{commit}/{filename}
 * where {commit} is read from refs/{ref} (default "main").
 */
extern "C" char *blobembed_resolve_hf_path(const char *repo_id,
                                           const char *filename,
                                           const char *ref) {
    if (!repo_id || !filename) {
        set_error("repo_id and filename are required");
        return nullptr;
    }

    /* Determine cache root */
    const char *cache_dir = std::getenv("HF_HUB_CACHE");
    std::string cache_root;
    if (cache_dir) {
        cache_root = cache_dir;
    } else {
        const char *hf_home = std::getenv("HF_HOME");
        if (hf_home) {
            cache_root = std::string(hf_home) + "/hub";
        } else {
            const char *home = std::getenv("HOME");
            if (!home) {
                set_error("cannot determine home directory");
                return nullptr;
            }
            cache_root = std::string(home) + "/.cache/huggingface/hub";
        }
    }

    /* Convert repo_id to directory name: "org/repo" → "models--org--repo" */
    std::string repo_dir = "models--";
    for (const char *p = repo_id; *p; p++) {
        repo_dir += (*p == '/') ? '-' : *p;
    }
    /* repo_id "org/repo" → "models--org--repo" requires double dash */
    /* Redo: replace '/' with '--' */
    repo_dir = "models--";
    std::string rid(repo_id);
    for (size_t i = 0; i < rid.size(); i++) {
        if (rid[i] == '/') {
            repo_dir += "--";
        } else {
            repo_dir += rid[i];
        }
    }

    std::string base = cache_root + "/" + repo_dir;

    /* Read the ref file to get the commit hash */
    std::string ref_name = ref ? ref : "main";
    std::string ref_path = base + "/refs/" + ref_name;

    FILE *f = fopen(ref_path.c_str(), "r");
    if (!f) {
        set_error("ref not found in HF cache: " + ref_path +
                  " (have you downloaded the model? try: huggingface-cli download " +
                  std::string(repo_id) + " " + std::string(filename) + ")");
        return nullptr;
    }

    char commit[128] = {0};
    if (!fgets(commit, sizeof(commit), f)) {
        fclose(f);
        set_error("failed to read ref file: " + ref_path);
        return nullptr;
    }
    fclose(f);

    /* Strip trailing whitespace */
    size_t len = strlen(commit);
    while (len > 0 && (commit[len - 1] == '\n' || commit[len - 1] == '\r' ||
                       commit[len - 1] == ' '))
        commit[--len] = '\0';

    /* Construct snapshot path */
    std::string snapshot_path = base + "/snapshots/" +
                                std::string(commit) + "/" + filename;

    /* Verify the file exists */
    FILE *check = fopen(snapshot_path.c_str(), "r");
    if (!check) {
        set_error("model file not found in HF cache: " + snapshot_path);
        return nullptr;
    }
    fclose(check);

    char *result = (char *)malloc(snapshot_path.size() + 1);
    if (result) {
        memcpy(result, snapshot_path.c_str(), snapshot_path.size() + 1);
    }
    return result;
}

extern "C" void blobembed_free_string(char *s) {
    free(s);
}

/* ── Init / cleanup ────────────────────────────────────────────────── */

extern "C" void blobembed_init(void) {
    llama_backend_init();
}

extern "C" void blobembed_cleanup(void) {
    std::unique_lock lock(g_registry_mu);
    for (auto &kv : g_registry) {
        for (int i = 0; i < kv.second->pool.size; i++) {
            if (kv.second->pool.contexts[i]) llama_free(kv.second->pool.contexts[i]);
        }
        if (kv.second->model) llama_model_free(kv.second->model);
        delete kv.second;
    }
    g_registry.clear();
    lock.unlock();
    llama_backend_free();
}

/* ── Model loading ─────────────────────────────────────────────────── */

extern "C" int blobembed_load_model(const char *name, const char *path) {
    if (!name || !path) {
        set_error("name and path are required");
        return -1;
    }

    /* Check if already loaded */
    {
        std::shared_lock lock(g_registry_mu);
        if (g_registry.count(name)) {
            return 0; /* already loaded */
        }
    }

    /* Load model */
    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(path, mparams);
    if (!model) {
        set_error(std::string("failed to load model from: ") + path);
        return -1;
    }

    /* Read context length from model metadata, fall back to 2048 */
    int n_ctx_train = llama_model_n_ctx_train(model);
    if (n_ctx_train <= 0) n_ctx_train = 2048;

    /* Create a pool of contexts with embeddings enabled */
    int pool_size = (int)std::thread::hardware_concurrency();
    if (pool_size <= 0) pool_size = 4;
    /* Cap at 8 to avoid excessive memory use (each ctx ~72MB KV cache) */
    if (pool_size > 8) pool_size = 8;

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = (uint32_t)n_ctx_train;
    cparams.n_batch = (uint32_t)n_ctx_train;
    cparams.n_ubatch = (uint32_t)n_ctx_train;
    cparams.n_seq_max = 256; /* support batched multi-sequence embedding */

    auto *entry = new ModelEntry();
    entry->model = model;
    entry->n_embd = llama_model_n_embd(model);
    entry->n_ctx = n_ctx_train;
    entry->pool.size = pool_size;

    for (int i = 0; i < pool_size; i++) {
        entry->pool.contexts[i] = llama_init_from_model(model, cparams);
        if (!entry->pool.contexts[i]) {
            /* Clean up already-created contexts */
            for (int j = 0; j < i; j++) llama_free(entry->pool.contexts[j]);
            llama_model_free(model);
            delete entry;
            set_error("failed to create llama context pool");
            return -1;
        }
        entry->pool.in_use[i].store(false);
    }

    std::unique_lock lock(g_registry_mu);
    /* Double-check (another thread may have loaded it) */
    if (g_registry.count(name)) {
        for (int i = 0; i < pool_size; i++) llama_free(entry->pool.contexts[i]);
        llama_model_free(model);
        delete entry;
        return 0;
    }
    g_registry[name] = entry;
    return 0;
}

extern "C" int blobembed_load_hf_model(const char *name, const char *repo_id,
                                       const char *filename, const char *ref) {
    char *path = blobembed_resolve_hf_path(repo_id, filename, ref);
    if (!path) return -1;

    int rc = blobembed_load_model(name, path);
    free(path);
    return rc;
}

extern "C" void blobembed_unload_model(const char *name) {
    std::unique_lock lock(g_registry_mu);
    auto it = g_registry.find(name);
    if (it == g_registry.end()) return;

    ModelEntry *entry = it->second;
    g_registry.erase(it);
    lock.unlock();

    /* Wait for all in-flight encodes to finish, then free */
    for (int i = 0; i < entry->pool.size; i++) {
        while (entry->pool.in_use[i].load()) std::this_thread::yield();
        if (entry->pool.contexts[i]) llama_free(entry->pool.contexts[i]);
    }
    if (entry->model) llama_model_free(entry->model);
    delete entry;
}

/* ── Helpers ────────────────────────────────────────────────────────── */

/* Tokenize text into a malloc'd buffer. Returns token count, or -1 on error. */
static int tokenize_text(const llama_vocab *vocab,
                         const char *text, size_t text_len,
                         llama_token **out_tokens) {
    int max_tokens = (int)text_len + 128;
    auto *tokens = (llama_token *)malloc(max_tokens * sizeof(llama_token));
    if (!tokens) { set_error("out of memory"); return -1; }

    int n = llama_tokenize(vocab, text, (int)text_len,
                           tokens, max_tokens, true, true);
    if (n < 0) {
        max_tokens = -n;
        tokens = (llama_token *)realloc(tokens, max_tokens * sizeof(llama_token));
        n = llama_tokenize(vocab, text, (int)text_len,
                           tokens, max_tokens, true, true);
    }
    if (n < 0) { free(tokens); set_error("tokenization failed"); return -1; }

    *out_tokens = tokens;
    return n;
}

/* L2-normalize a vector in place. */
static void l2_normalize(float *vec, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += vec[i] * vec[i];
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < n; i++) vec[i] /= norm;
    }
}

/*
 * Encode a single window of tokens and extract the sequence-level embedding.
 * The embedding is ADDED to accum (for multi-window averaging).
 * Returns 0 on success.
 */
static int encode_window(ModelEntry *entry, llama_context *ctx,
                         const llama_token *tokens, int n_tokens,
                         float *accum) {
    llama_kv_self_clear(ctx);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = true;
    }
    batch.n_tokens = n_tokens;

    int rc;
    if (llama_model_has_encoder(entry->model) &&
        !llama_model_has_decoder(entry->model)) {
        rc = llama_encode(ctx, batch);
    } else {
        rc = llama_decode(ctx, batch);
    }
    llama_batch_free(batch);

    if (rc != 0) { set_error("encode failed"); return -1; }

    const float *embd = llama_get_embeddings_seq(ctx, 0);
    if (!embd) embd = llama_get_embeddings(ctx);
    if (!embd) { set_error("failed to extract embeddings"); return -1; }

    int n_embd = entry->n_embd;
    for (int i = 0; i < n_embd; i++) accum[i] += embd[i];
    return 0;
}

/* ── Embedding ─────────────────────────────────────────────────────── */

/*
 * Embed text of any length.
 *
 * If the text fits in the model's context window, encode in a single pass.
 * If it overflows, split into overlapping windows (50% overlap), encode
 * each window, and mean-pool the per-window embeddings. The overlap ensures
 * tokens near window boundaries get context from both sides across the two
 * windows they appear in.
 *
 * Returns a malloc'd L2-normalized float array. Caller frees via
 * blobembed_free_embedding().
 */
extern "C" float *blobembed_embed(const char *model_name,
                                  const char *text, size_t text_len,
                                  int *out_dim) {
    ModelEntry *entry = find_model(model_name);
    if (!entry) {
        set_error(std::string("model not loaded: ") + (model_name ? model_name : "(null)"));
        return nullptr;
    }

    const llama_vocab *vocab = llama_model_get_vocab(entry->model);
    int n_embd = entry->n_embd;
    int n_ctx = entry->n_ctx;

    /* Tokenize the full text */
    llama_token *tokens = nullptr;
    int n_tokens = tokenize_text(vocab, text, text_len, &tokens);
    if (n_tokens < 0) return nullptr;

    /* Allocate accumulator (zeroed) */
    float *accum = (float *)calloc(n_embd, sizeof(float));
    if (!accum) { free(tokens); set_error("out of memory"); return nullptr; }

    /* Acquire a context from the pool */
    int ctx_idx = entry->pool.acquire();
    llama_context *ctx = entry->pool.contexts[ctx_idx];

    if (n_tokens <= n_ctx) {
        /* ── Fast path: fits in one window ──────────────────────── */
        if (encode_window(entry, ctx, tokens, n_tokens, accum) != 0) {
            entry->pool.release(ctx_idx);
            free(tokens); free(accum);
            return nullptr;
        }
        free(tokens);
    } else {
        /* ── Overflow path: sliding window with 50% overlap ─────── */
        int stride = n_ctx / 2; /* 50% overlap */
        if (stride <= 0) stride = 1;
        int n_windows = 0;

        for (int offset = 0; offset < n_tokens; offset += stride) {
            int window_len = n_tokens - offset;
            if (window_len > n_ctx) window_len = n_ctx;

            if (encode_window(entry, ctx, tokens + offset, window_len, accum) != 0) {
                entry->pool.release(ctx_idx);
                free(tokens); free(accum);
                return nullptr;
            }
            n_windows++;

            /* Stop if this window reached the end */
            if (offset + window_len >= n_tokens) break;
        }

        free(tokens);

        /* Average across windows */
        if (n_windows > 1) {
            for (int i = 0; i < n_embd; i++) accum[i] /= (float)n_windows;
        }
    }

    /* Release context back to pool */
    entry->pool.release(ctx_idx);

    /* L2-normalize */
    l2_normalize(accum, n_embd);

    if (out_dim) *out_dim = n_embd;
    return accum;
}

extern "C" void blobembed_free_embedding(float *embd) {
    free(embd);
}

/* ── Batched embedding ─────────────────────────────────────────────── */

/*
 * Embed multiple texts in a single forward pass.
 *
 * Each text gets its own seq_id in the batch. The GPU loads the model
 * weights once and processes all sequences simultaneously — amortizing
 * the memory bandwidth cost across the batch.
 *
 * texts:     array of C strings
 * text_lens: array of lengths (parallel to texts)
 * n_texts:   number of texts
 * out_dim:   set to the embedding dimension
 *
 * Returns a malloc'd array of (n_texts * n_embd) floats, laid out as
 * [text0_emb0..text0_embN, text1_emb0..text1_embN, ...].
 * Each embedding is L2-normalized.
 * Caller must free with blobembed_free_embedding().
 */
extern "C" float *blobembed_embed_batch(const char *model_name,
                                        const char **texts,
                                        const size_t *text_lens,
                                        int n_texts,
                                        int *out_dim) {
    ModelEntry *entry = find_model(model_name);
    if (!entry) {
        set_error(std::string("model not loaded: ") + (model_name ? model_name : "(null)"));
        return nullptr;
    }

    const llama_vocab *vocab = llama_model_get_vocab(entry->model);
    int n_embd = entry->n_embd;
    int n_ctx = entry->n_ctx;

    /* Tokenize all texts, track per-text token counts */
    std::vector<llama_token> all_tokens;
    std::vector<int> token_counts(n_texts);
    std::vector<int> token_offsets(n_texts);

    int total_tokens = 0;
    for (int t = 0; t < n_texts; t++) {
        llama_token *toks = nullptr;
        int n = tokenize_text(vocab, texts[t], text_lens[t], &toks);
        if (n < 0) return nullptr;

        token_offsets[t] = total_tokens;
        token_counts[t] = n;

        all_tokens.insert(all_tokens.end(), toks, toks + n);
        total_tokens += n;
        free(toks);
    }

    /* Allocate result buffer */
    float *result = (float *)malloc(n_texts * n_embd * sizeof(float));
    if (!result) { set_error("out of memory"); return nullptr; }

    int ctx_idx = entry->pool.acquire();
    llama_context *ctx = entry->pool.contexts[ctx_idx];

    /* Process texts in sub-batches that fit within the context window.
     * Pack as many sequences as possible into each forward pass. */
    int t = 0;
    static int batch_call_count = 0;
    static double total_batch_time = 0.0;

    while (t < n_texts) {
        /* Greedily fill a batch up to n_ctx tokens */
        int batch_start = t;
        int batch_tokens = 0;
        while (t < n_texts && batch_tokens + token_counts[t] <= n_ctx) {
            batch_tokens += token_counts[t];
            t++;
        }
        int batch_size = t - batch_start;

        /* If a single text exceeds n_ctx, process it alone (overflow path) */
        if (batch_size == 0) {
            entry->pool.release(ctx_idx);
            int dim = 0;
            float *single = blobembed_embed(model_name, texts[t], text_lens[t], &dim);
            ctx_idx = entry->pool.acquire();
            ctx = entry->pool.contexts[ctx_idx];
            if (!single) {
                entry->pool.release(ctx_idx);
                free(result);
                return nullptr;
            }
            memcpy(result + t * n_embd, single, n_embd * sizeof(float));
            blobembed_free_embedding(single);
            t++;
            continue;
        }

        /* Build batch with batch_size sequences */
        llama_kv_self_clear(ctx);
        llama_batch batch = llama_batch_init(batch_tokens, 0, batch_size);

        for (int b = 0; b < batch_size; b++) {
            int src = batch_start + b;
            int offset = token_offsets[src];
            int count = token_counts[src];
            for (int i = 0; i < count; i++) {
                int pos = batch.n_tokens;
                batch.token[pos]    = all_tokens[offset + i];
                batch.pos[pos]      = i;
                batch.n_seq_id[pos] = 1;
                batch.seq_id[pos][0] = b; /* seq_id within this sub-batch */
                batch.logits[pos]   = true;
                batch.n_tokens++;
            }
        }

        /* Single forward pass for the sub-batch */
        int rc;
        if (llama_model_has_encoder(entry->model) &&
            !llama_model_has_decoder(entry->model)) {
            rc = llama_encode(ctx, batch);
        } else {
            rc = llama_decode(ctx, batch);
        }
        llama_batch_free(batch);

        if (rc != 0) {
            entry->pool.release(ctx_idx);
            free(result);
            set_error("batch encode failed");
            return nullptr;
        }

        /* Extract per-sequence embeddings from this sub-batch */
        for (int b = 0; b < batch_size; b++) {
            const float *embd = llama_get_embeddings_seq(ctx, b);
            if (!embd) {
                entry->pool.release(ctx_idx);
                free(result);
                set_error("failed to extract embeddings for sequence");
                return nullptr;
            }
            int dst = batch_start + b;
            memcpy(result + dst * n_embd, embd, n_embd * sizeof(float));
            l2_normalize(result + dst * n_embd, n_embd);
        }

        batch_call_count++;
        /* Progress: log every 20 sub-batches */
        if (batch_call_count % 20 == 0) {
            fprintf(stderr, "[blobembed] batch #%d: %d texts in sub-batch of %d tokens, "
                    "%d/%d texts done\n",
                    batch_call_count, batch_size, batch_tokens, t, n_texts);
        }
    }

    entry->pool.release(ctx_idx);
    if (out_dim) *out_dim = n_embd;
    return result;
}

/* ── Cosine similarity ─────────────────────────────────────────────── */

extern "C" double blobembed_cosine_sim(const float *a, int a_len,
                                       const float *b, int b_len) {
    if (!a || !b || a_len != b_len || a_len <= 0) {
        set_error("cosine_sim: dimension mismatch or null input");
        return NAN;
    }

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < a_len; i++) {
        dot    += (double)a[i] * (double)b[i];
        norm_a += (double)a[i] * (double)a[i];
        norm_b += (double)b[i] * (double)b[i];
    }

    double denom = sqrt(norm_a) * sqrt(norm_b);
    if (denom == 0.0) return 0.0;
    return dot / denom;
}

/* ── Info ──────────────────────────────────────────────────────────── */

extern "C" int blobembed_model_dim(const char *model_name) {
    ModelEntry *entry = find_model(model_name);
    if (!entry) return -1;
    return entry->n_embd;
}

extern "C" int blobembed_token_count(const char *model_name,
                                     const char *text, size_t text_len) {
    ModelEntry *entry = find_model(model_name);
    if (!entry) {
        set_error(std::string("model not loaded: ") + (model_name ? model_name : "(null)"));
        return -1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(entry->model);

    int max_tokens = (int)text_len + 128;
    auto *tokens = (llama_token *)malloc(max_tokens * sizeof(llama_token));
    if (!tokens) {
        set_error("out of memory");
        return -1;
    }

    int n_tokens = llama_tokenize(vocab, text, (int)text_len,
                                  tokens, max_tokens, true, true);
    if (n_tokens < 0) {
        max_tokens = -n_tokens;
        tokens = (llama_token *)realloc(tokens, max_tokens * sizeof(llama_token));
        n_tokens = llama_tokenize(vocab, text, (int)text_len,
                                  tokens, max_tokens, true, true);
    }

    free(tokens);

    if (n_tokens < 0) {
        set_error("tokenization failed");
        return -1;
    }
    return n_tokens;
}
