/*
 * blobembed core: wraps llama.cpp's embedding API.
 *
 * Process-wide model registry with thread-safe access.
 * Each model entry holds a llama_model + llama_context pair.
 * The context is created with embeddings=true and mean pooling.
 */

#include "blobembed.h"

#include "llama.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

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

struct ModelEntry {
    llama_model *model;
    llama_context *ctx;
    int n_embd;
    int n_ctx;     /* context window size (max tokens per encode) */
    std::mutex ctx_mu; /* serializes encode calls on this model's context */
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
        if (kv.second->ctx) llama_free(kv.second->ctx);
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

    /* Create context with embeddings enabled */
    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = (uint32_t)n_ctx_train;
    cparams.n_batch = (uint32_t)n_ctx_train;
    cparams.n_ubatch = (uint32_t)n_ctx_train; /* must equal n_batch for non-causal models */

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        set_error("failed to create llama context");
        return -1;
    }

    int n_embd = llama_model_n_embd(model);

    auto *entry = new ModelEntry();
    entry->model = model;
    entry->ctx = ctx;
    entry->n_embd = n_embd;
    entry->n_ctx = n_ctx_train;

    std::unique_lock lock(g_registry_mu);
    /* Double-check (another thread may have loaded it) */
    if (g_registry.count(name)) {
        llama_free(ctx);
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

    /* Wait for any in-flight encode to finish */
    std::lock_guard ctx_lock(entry->ctx_mu);
    if (entry->ctx) llama_free(entry->ctx);
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
static int encode_window(ModelEntry *entry,
                         const llama_token *tokens, int n_tokens,
                         float *accum) {
    llama_kv_self_clear(entry->ctx);

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
        rc = llama_encode(entry->ctx, batch);
    } else {
        rc = llama_decode(entry->ctx, batch);
    }
    llama_batch_free(batch);

    if (rc != 0) { set_error("encode failed"); return -1; }

    const float *embd = llama_get_embeddings_seq(entry->ctx, 0);
    if (!embd) embd = llama_get_embeddings(entry->ctx);
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

    /* Serialize context access for this model */
    std::lock_guard ctx_lock(entry->ctx_mu);

    if (n_tokens <= n_ctx) {
        /* ── Fast path: fits in one window ──────────────────────── */
        if (encode_window(entry, tokens, n_tokens, accum) != 0) {
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

            if (encode_window(entry, tokens + offset, window_len, accum) != 0) {
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

    /* L2-normalize */
    l2_normalize(accum, n_embd);

    if (out_dim) *out_dim = n_embd;
    return accum;
}

extern "C" void blobembed_free_embedding(float *embd) {
    free(embd);
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
