#ifndef BLOBEMBED_H
#define BLOBEMBED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize / shut down the llama.cpp backend.
 * Call blobembed_init() once before any other calls.
 * Call blobembed_cleanup() when done (frees all models).
 */
void blobembed_init(void);
void blobembed_cleanup(void);

/*
 * Load a GGUF model from a filesystem path and register it under `name`.
 * The model + context are cached for the lifetime of the process (or until
 * blobembed_unload_model is called).
 *
 * Returns 0 on success, non-zero on error (call blobembed_errmsg()).
 */
int blobembed_load_model(const char *name, const char *path);

/*
 * Load a GGUF model from the HuggingFace cache.
 *
 * Resolves the model path from the HF cache directory:
 *   $HF_HUB_CACHE/models--{repo_id with / replaced by --}/snapshots/{ref}/{filename}
 *
 * repo_id:  e.g. "nomic-ai/nomic-embed-text-v1.5-GGUF"
 * filename: e.g. "nomic-embed-text-v1.5.Q4_K_M.gguf"
 * ref:      git ref, pass NULL for "main"
 *
 * Returns 0 on success, non-zero on error (call blobembed_errmsg()).
 */
int blobembed_load_hf_model(const char *name, const char *repo_id,
                            const char *filename, const char *ref);

/*
 * Unload a previously loaded model, freeing its resources.
 */
void blobembed_unload_model(const char *name);

/*
 * Compute the embedding for a text string.
 *
 * Returns a malloc'd float array of length *out_dim.
 * The embedding is L2-normalized.
 * Caller must free with blobembed_free_embedding().
 * Returns NULL on error; call blobembed_errmsg() for details.
 */
float *blobembed_embed(const char *model_name,
                       const char *text, size_t text_len,
                       int *out_dim);

/*
 * Free an embedding returned by blobembed_embed().
 */
void blobembed_free_embedding(float *embd);

/*
 * Return the embedding dimension for a loaded model.
 * Returns -1 if the model is not loaded.
 */
int blobembed_model_dim(const char *model_name);

/*
 * Count the number of tokens in a text string for a loaded model.
 * Returns -1 on error; call blobembed_errmsg() for details.
 */
int blobembed_token_count(const char *model_name,
                          const char *text, size_t text_len);

/*
 * Resolve the filesystem path for a model file in the HF cache.
 * Returns a malloc'd string (caller must free with blobembed_free_string()),
 * or NULL if the file is not found in the cache.
 */
char *blobembed_resolve_hf_path(const char *repo_id,
                                const char *filename, const char *ref);

/*
 * Free a string returned by blobembed_resolve_hf_path().
 */
void blobembed_free_string(char *s);

/*
 * Return the last error message (thread-local).
 * Returns "" if no error has occurred.
 */
const char *blobembed_errmsg(void);

#ifdef __cplusplus
}
#endif

#endif
