/*
 * SQLite loadable extension for text embeddings via llama.cpp.
 *
 * Registers scalar functions with be_ prefix.
 * Embeddings are returned as BLOBs of packed float32 values,
 * compatible with sqlite-vec's vector format.
 */

#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include "blobembed.h"

#include <stdlib.h>
#include <string.h>

static int g_initialized = 0;

static void ensure_init(void) {
    if (!g_initialized) {
        blobembed_init();
        g_initialized = 1;
    }
}

/* ── be_load_model(name, path) → TEXT ─────────────────────────────── */

static void be_load_model_func(sqlite3_context *ctx, int argc,
                               sqlite3_value **argv) {
    (void)argc;
    ensure_init();

    if (sqlite3_value_type(argv[0]) == SQLITE_NULL ||
        sqlite3_value_type(argv[1]) == SQLITE_NULL) {
        sqlite3_result_error(ctx, "be_load_model: name and path are required", -1);
        return;
    }

    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *path = (const char *)sqlite3_value_text(argv[1]);

    if (blobembed_load_model(name, path) != 0) {
        sqlite3_result_error(ctx, blobembed_errmsg(), -1);
        return;
    }

    sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
}

/* ── be_load_hf_model(name, repo_id, filename [, ref]) → TEXT ─────── */

static void be_load_hf_model_func(sqlite3_context *ctx, int argc,
                                  sqlite3_value **argv) {
    ensure_init();

    if (sqlite3_value_type(argv[0]) == SQLITE_NULL ||
        sqlite3_value_type(argv[1]) == SQLITE_NULL ||
        sqlite3_value_type(argv[2]) == SQLITE_NULL) {
        sqlite3_result_error(ctx, "be_load_hf_model: name, repo_id, and filename are required", -1);
        return;
    }

    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *repo_id = (const char *)sqlite3_value_text(argv[1]);
    const char *filename = (const char *)sqlite3_value_text(argv[2]);
    const char *ref = NULL;
    if (argc >= 4 && sqlite3_value_type(argv[3]) != SQLITE_NULL) {
        ref = (const char *)sqlite3_value_text(argv[3]);
    }

    if (blobembed_load_hf_model(name, repo_id, filename, ref) != 0) {
        sqlite3_result_error(ctx, blobembed_errmsg(), -1);
        return;
    }

    sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
}

/* ── be_embed(model_name, text) → BLOB (packed float32) ───────────── */

static void be_embed_func(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) == SQLITE_NULL ||
        sqlite3_value_type(argv[1]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }

    const char *model_name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);
    int text_len = sqlite3_value_bytes(argv[1]);

    int dim = 0;
    float *embd = blobembed_embed(model_name, text, (size_t)text_len, &dim);
    if (!embd) {
        sqlite3_result_error(ctx, blobembed_errmsg(), -1);
        return;
    }

    /* Return as BLOB of packed float32 (compatible with sqlite-vec) */
    sqlite3_result_blob(ctx, embd, dim * (int)sizeof(float), SQLITE_TRANSIENT);
    blobembed_free_embedding(embd);
}

/* ── be_embed_dim(model_name) → INTEGER ───────────────────────────── */

static void be_embed_dim_func(sqlite3_context *ctx, int argc,
                              sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }

    const char *model_name = (const char *)sqlite3_value_text(argv[0]);
    int dim = blobembed_model_dim(model_name);
    if (dim < 0) {
        sqlite3_result_error(ctx, "model not loaded", -1);
        return;
    }
    sqlite3_result_int(ctx, dim);
}

/* ── be_token_count(model_name, text) → INTEGER ───────────────────── */

static void be_token_count_func(sqlite3_context *ctx, int argc,
                                sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) == SQLITE_NULL ||
        sqlite3_value_type(argv[1]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }

    const char *model_name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);
    int text_len = sqlite3_value_bytes(argv[1]);

    int count = blobembed_token_count(model_name, text, (size_t)text_len);
    if (count < 0) {
        sqlite3_result_error(ctx, blobembed_errmsg(), -1);
        return;
    }
    sqlite3_result_int(ctx, count);
}

/* ── be_unload_model(name) → TEXT ─────────────────────────────────── */

static void be_unload_model_func(sqlite3_context *ctx, int argc,
                                 sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        sqlite3_result_null(ctx);
        return;
    }

    const char *name = (const char *)sqlite3_value_text(argv[0]);
    blobembed_unload_model(name);
    sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
}

/* ── Extension init ──────────────────────────────────────────────── */

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_blobembed_init(sqlite3 *db, char **pzErrMsg,
                           const sqlite3_api_routines *pApi) {
    int rc;
    (void)pzErrMsg;
    SQLITE_EXTENSION_INIT2(pApi);

    rc = sqlite3_create_function(db, "be_load_model", 2,
                                 SQLITE_UTF8, NULL,
                                 be_load_model_func, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_function(db, "be_load_hf_model", 3,
                                 SQLITE_UTF8, NULL,
                                 be_load_hf_model_func, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_function(db, "be_load_hf_model", 4,
                                 SQLITE_UTF8, NULL,
                                 be_load_hf_model_func, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_function(db, "be_embed", 2,
                                 SQLITE_UTF8 | SQLITE_DETERMINISTIC, NULL,
                                 be_embed_func, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_function(db, "be_embed_dim", 1,
                                 SQLITE_UTF8 | SQLITE_DETERMINISTIC, NULL,
                                 be_embed_dim_func, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_function(db, "be_token_count", 2,
                                 SQLITE_UTF8 | SQLITE_DETERMINISTIC, NULL,
                                 be_token_count_func, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_function(db, "be_unload_model", 1,
                                 SQLITE_UTF8, NULL,
                                 be_unload_model_func, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    return SQLITE_OK;
}
