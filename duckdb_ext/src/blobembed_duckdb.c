/*
 * DuckDB C API extension for text embeddings via llama.cpp.
 *
 * Registers scalar functions with be_ prefix.
 * Embeddings are returned as LIST(FLOAT) — cast to FLOAT[N] for use
 * with the vss extension's HNSW index:
 *
 *   SELECT be_embed('nomic', text)::FLOAT[768] FROM documents;
 */

#define DUCKDB_EXTENSION_NAME blobembed
#include "duckdb_extension.h"

#include "blobembed.h"

#include <stdlib.h>
#include <string.h>

DUCKDB_EXTENSION_EXTERN

static int g_initialized = 0;

static void ensure_init(void) {
    if (!g_initialized) {
        blobembed_init();
        g_initialized = 1;
    }
}

/* ── String helpers ───────────────────────────────────────────────── */

static const char *str_ptr(duckdb_string_t *s, uint32_t *out_len) {
    uint32_t len = s->value.inlined.length;
    *out_len = len;
    if (len <= 12) {
        return s->value.inlined.inlined;
    }
    return s->value.pointer.ptr;
}

static char *str_dup_z(duckdb_string_t *s) {
    uint32_t len;
    const char *p = str_ptr(s, &len);
    char *z = (char *)malloc(len + 1);
    memcpy(z, p, len);
    z[len] = '\0';
    return z;
}

/* ── be_load_model(name, path) → VARCHAR ──────────────────────────── */

static void be_load_model_func(duckdb_function_info info,
                               duckdb_data_chunk input,
                               duckdb_vector output) {
    ensure_init();
    idx_t size = duckdb_data_chunk_get_size(input);
    duckdb_vector vec0 = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector vec1 = duckdb_data_chunk_get_vector(input, 1);

    duckdb_string_t *data0 = (duckdb_string_t *)duckdb_vector_get_data(vec0);
    duckdb_string_t *data1 = (duckdb_string_t *)duckdb_vector_get_data(vec1);
    uint64_t *val0 = duckdb_vector_get_validity(vec0);
    uint64_t *val1 = duckdb_vector_get_validity(vec1);

    for (idx_t row = 0; row < size; row++) {
        if ((val0 && !duckdb_validity_row_is_valid(val0, row)) ||
            (val1 && !duckdb_validity_row_is_valid(val1, row))) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
            continue;
        }

        char *name = str_dup_z(&data0[row]);
        char *path = str_dup_z(&data1[row]);

        if (blobembed_load_model(name, path) != 0) {
            free(name); free(path);
            duckdb_scalar_function_set_error(info, blobembed_errmsg());
            return;
        }

        free(name); free(path);
        duckdb_vector_assign_string_element(output, row, "ok");
    }
}

/* ── be_load_hf_model(name, repo_id, filename) → VARCHAR ──────────── */

static void be_load_hf_model_func(duckdb_function_info info,
                                  duckdb_data_chunk input,
                                  duckdb_vector output) {
    ensure_init();
    idx_t size = duckdb_data_chunk_get_size(input);
    duckdb_vector vec0 = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector vec1 = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector vec2 = duckdb_data_chunk_get_vector(input, 2);

    duckdb_string_t *data0 = (duckdb_string_t *)duckdb_vector_get_data(vec0);
    duckdb_string_t *data1 = (duckdb_string_t *)duckdb_vector_get_data(vec1);
    duckdb_string_t *data2 = (duckdb_string_t *)duckdb_vector_get_data(vec2);
    uint64_t *val0 = duckdb_vector_get_validity(vec0);
    uint64_t *val1 = duckdb_vector_get_validity(vec1);
    uint64_t *val2 = duckdb_vector_get_validity(vec2);

    for (idx_t row = 0; row < size; row++) {
        if ((val0 && !duckdb_validity_row_is_valid(val0, row)) ||
            (val1 && !duckdb_validity_row_is_valid(val1, row)) ||
            (val2 && !duckdb_validity_row_is_valid(val2, row))) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
            continue;
        }

        char *name = str_dup_z(&data0[row]);
        char *repo_id = str_dup_z(&data1[row]);
        char *filename = str_dup_z(&data2[row]);

        if (blobembed_load_hf_model(name, repo_id, filename, NULL) != 0) {
            free(name); free(repo_id); free(filename);
            duckdb_scalar_function_set_error(info, blobembed_errmsg());
            return;
        }

        free(name); free(repo_id); free(filename);
        duckdb_vector_assign_string_element(output, row, "ok");
    }
}

/* ── be_embed(model_name, text) → LIST(FLOAT) ────────────────────── */
/*
 * Single-row encoding. BERT-style models have O(n²) attention on total
 * tokens, so batching multiple sequences into one forward pass is slower
 * than encoding each text individually (2048×2048 vs 10×10 attention).
 * The batch API exists in the core for future decoder-style models.
 */

static void be_embed_func(duckdb_function_info info,
                          duckdb_data_chunk input,
                          duckdb_vector output) {
    idx_t size = duckdb_data_chunk_get_size(input);
    duckdb_vector vec0 = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector vec1 = duckdb_data_chunk_get_vector(input, 1);

    duckdb_string_t *data0 = (duckdb_string_t *)duckdb_vector_get_data(vec0);
    duckdb_string_t *data1 = (duckdb_string_t *)duckdb_vector_get_data(vec1);
    uint64_t *val0 = duckdb_vector_get_validity(vec0);
    uint64_t *val1 = duckdb_vector_get_validity(vec1);

    duckdb_list_entry *entries = (duckdb_list_entry *)duckdb_vector_get_data(output);
    duckdb_vector child = duckdb_list_vector_get_child(output);

    idx_t total_offset = 0;

    for (idx_t row = 0; row < size; row++) {
        if ((val0 && !duckdb_validity_row_is_valid(val0, row)) ||
            (val1 && !duckdb_validity_row_is_valid(val1, row))) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
            entries[row].offset = total_offset;
            entries[row].length = 0;
            continue;
        }

        char *model_name = str_dup_z(&data0[row]);
        uint32_t text_len;
        const char *text = str_ptr(&data1[row], &text_len);

        char *text_z = (char *)malloc(text_len + 1);
        memcpy(text_z, text, text_len);
        text_z[text_len] = '\0';

        int dim = 0;
        float *embd = blobembed_embed(model_name, text_z, (size_t)text_len, &dim);
        free(model_name);
        free(text_z);

        if (!embd) {
            duckdb_scalar_function_set_error(info, blobembed_errmsg());
            return;
        }

        duckdb_list_vector_reserve(output, total_offset + (idx_t)dim);

        entries[row].offset = total_offset;
        entries[row].length = (uint64_t)dim;

        float *child_data = (float *)duckdb_vector_get_data(child);
        memcpy(&child_data[total_offset], embd, (size_t)dim * sizeof(float));

        total_offset += (idx_t)dim;
        blobembed_free_embedding(embd);
    }

    duckdb_list_vector_set_size(output, total_offset);
}

/* ── be_embed_dim(model_name) → INTEGER ───────────────────────────── */

static void be_embed_dim_func(duckdb_function_info info,
                              duckdb_data_chunk input,
                              duckdb_vector output) {
    idx_t size = duckdb_data_chunk_get_size(input);
    duckdb_vector vec0 = duckdb_data_chunk_get_vector(input, 0);
    duckdb_string_t *data0 = (duckdb_string_t *)duckdb_vector_get_data(vec0);
    uint64_t *val0 = duckdb_vector_get_validity(vec0);
    int32_t *out_data = (int32_t *)duckdb_vector_get_data(output);

    for (idx_t row = 0; row < size; row++) {
        if (val0 && !duckdb_validity_row_is_valid(val0, row)) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
            continue;
        }

        char *name = str_dup_z(&data0[row]);
        int dim = blobembed_model_dim(name);
        free(name);

        if (dim < 0) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
        } else {
            out_data[row] = dim;
        }
    }
}

/* ── be_token_count(model_name, text) → INTEGER ───────────────────── */

static void be_token_count_func(duckdb_function_info info,
                                duckdb_data_chunk input,
                                duckdb_vector output) {
    idx_t size = duckdb_data_chunk_get_size(input);
    duckdb_vector vec0 = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector vec1 = duckdb_data_chunk_get_vector(input, 1);

    duckdb_string_t *data0 = (duckdb_string_t *)duckdb_vector_get_data(vec0);
    duckdb_string_t *data1 = (duckdb_string_t *)duckdb_vector_get_data(vec1);
    uint64_t *val0 = duckdb_vector_get_validity(vec0);
    uint64_t *val1 = duckdb_vector_get_validity(vec1);
    int32_t *out_data = (int32_t *)duckdb_vector_get_data(output);

    for (idx_t row = 0; row < size; row++) {
        if ((val0 && !duckdb_validity_row_is_valid(val0, row)) ||
            (val1 && !duckdb_validity_row_is_valid(val1, row))) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
            continue;
        }

        char *model_name = str_dup_z(&data0[row]);
        uint32_t text_len;
        const char *text = str_ptr(&data1[row], &text_len);

        char *text_z = (char *)malloc(text_len + 1);
        memcpy(text_z, text, text_len);
        text_z[text_len] = '\0';

        int count = blobembed_token_count(model_name, text_z, (size_t)text_len);
        free(model_name);
        free(text_z);

        if (count < 0) {
            duckdb_scalar_function_set_error(info, blobembed_errmsg());
            return;
        }
        out_data[row] = count;
    }
}

/* ── be_unload_model(name) → VARCHAR ──────────────────────────────── */

static void be_unload_model_func(duckdb_function_info info,
                                 duckdb_data_chunk input,
                                 duckdb_vector output) {
    idx_t size = duckdb_data_chunk_get_size(input);
    duckdb_vector vec0 = duckdb_data_chunk_get_vector(input, 0);
    duckdb_string_t *data0 = (duckdb_string_t *)duckdb_vector_get_data(vec0);
    uint64_t *val0 = duckdb_vector_get_validity(vec0);

    for (idx_t row = 0; row < size; row++) {
        if (val0 && !duckdb_validity_row_is_valid(val0, row)) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
            continue;
        }

        char *name = str_dup_z(&data0[row]);
        blobembed_unload_model(name);
        free(name);
        duckdb_vector_assign_string_element(output, row, "ok");
    }
}

/* ── be_cosine_sim(list_a, list_b) → DOUBLE ────────────────────────── */

static void be_cosine_sim_func(duckdb_function_info info,
                               duckdb_data_chunk input,
                               duckdb_vector output) {
    idx_t size = duckdb_data_chunk_get_size(input);
    duckdb_vector vec0 = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector vec1 = duckdb_data_chunk_get_vector(input, 1);
    uint64_t *val0 = duckdb_vector_get_validity(vec0);
    uint64_t *val1 = duckdb_vector_get_validity(vec1);
    double *out_data = (double *)duckdb_vector_get_data(output);

    /* List vectors: entries array + child vector */
    duckdb_list_entry *entries0 = (duckdb_list_entry *)duckdb_vector_get_data(vec0);
    duckdb_list_entry *entries1 = (duckdb_list_entry *)duckdb_vector_get_data(vec1);
    duckdb_vector child0 = duckdb_list_vector_get_child(vec0);
    duckdb_vector child1 = duckdb_list_vector_get_child(vec1);
    float *child_data0 = (float *)duckdb_vector_get_data(child0);
    float *child_data1 = (float *)duckdb_vector_get_data(child1);

    for (idx_t row = 0; row < size; row++) {
        if ((val0 && !duckdb_validity_row_is_valid(val0, row)) ||
            (val1 && !duckdb_validity_row_is_valid(val1, row))) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
            continue;
        }

        int len0 = (int)entries0[row].length;
        int len1 = (int)entries1[row].length;
        float *a = &child_data0[entries0[row].offset];
        float *b = &child_data1[entries1[row].offset];

        out_data[row] = blobembed_cosine_sim(a, len0, b, len1);
    }
}

/* ── Register functions ───────────────────────────────────────────── */

static void register_functions(duckdb_connection connection) {
    duckdb_logical_type varchar_type = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_logical_type int_type = duckdb_create_logical_type(DUCKDB_TYPE_INTEGER);
    duckdb_logical_type float_type = duckdb_create_logical_type(DUCKDB_TYPE_FLOAT);
    duckdb_logical_type double_type = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_logical_type list_float_type = duckdb_create_list_type(float_type);

    /* be_load_model(name, path) → VARCHAR */
    {
        duckdb_scalar_function func = duckdb_create_scalar_function();
        duckdb_scalar_function_set_name(func, "be_load_model");
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_set_return_type(func, varchar_type);
        duckdb_scalar_function_set_function(func, be_load_model_func);
        duckdb_register_scalar_function(connection, func);
        duckdb_destroy_scalar_function(&func);
    }

    /* be_load_hf_model(name, repo_id, filename) → VARCHAR */
    {
        duckdb_scalar_function func = duckdb_create_scalar_function();
        duckdb_scalar_function_set_name(func, "be_load_hf_model");
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_set_return_type(func, varchar_type);
        duckdb_scalar_function_set_function(func, be_load_hf_model_func);
        duckdb_register_scalar_function(connection, func);
        duckdb_destroy_scalar_function(&func);
    }

    /* be_embed(model_name, text) → LIST(FLOAT) */
    {
        duckdb_scalar_function func = duckdb_create_scalar_function();
        duckdb_scalar_function_set_name(func, "be_embed");
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_set_return_type(func, list_float_type);
        duckdb_scalar_function_set_function(func, be_embed_func);
        duckdb_register_scalar_function(connection, func);
        duckdb_destroy_scalar_function(&func);
    }

    /* be_embed_dim(model_name) → INTEGER */
    {
        duckdb_scalar_function func = duckdb_create_scalar_function();
        duckdb_scalar_function_set_name(func, "be_embed_dim");
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_set_return_type(func, int_type);
        duckdb_scalar_function_set_function(func, be_embed_dim_func);
        duckdb_register_scalar_function(connection, func);
        duckdb_destroy_scalar_function(&func);
    }

    /* be_token_count(model_name, text) → INTEGER */
    {
        duckdb_scalar_function func = duckdb_create_scalar_function();
        duckdb_scalar_function_set_name(func, "be_token_count");
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_set_return_type(func, int_type);
        duckdb_scalar_function_set_function(func, be_token_count_func);
        duckdb_register_scalar_function(connection, func);
        duckdb_destroy_scalar_function(&func);
    }

    /* be_unload_model(name) → VARCHAR */
    {
        duckdb_scalar_function func = duckdb_create_scalar_function();
        duckdb_scalar_function_set_name(func, "be_unload_model");
        duckdb_scalar_function_add_parameter(func, varchar_type);
        duckdb_scalar_function_set_return_type(func, varchar_type);
        duckdb_scalar_function_set_function(func, be_unload_model_func);
        duckdb_register_scalar_function(connection, func);
        duckdb_destroy_scalar_function(&func);
    }

    /* be_cosine_sim(list_a, list_b) → DOUBLE */
    {
        duckdb_scalar_function func = duckdb_create_scalar_function();
        duckdb_scalar_function_set_name(func, "be_cosine_sim");
        duckdb_scalar_function_add_parameter(func, list_float_type);
        duckdb_scalar_function_add_parameter(func, list_float_type);
        duckdb_scalar_function_set_return_type(func, double_type);
        duckdb_scalar_function_set_function(func, be_cosine_sim_func);
        duckdb_register_scalar_function(connection, func);
        duckdb_destroy_scalar_function(&func);
    }

    duckdb_destroy_logical_type(&list_float_type);
    duckdb_destroy_logical_type(&double_type);
    duckdb_destroy_logical_type(&float_type);
    duckdb_destroy_logical_type(&int_type);
    duckdb_destroy_logical_type(&varchar_type);
}

/* ── Extension entrypoint ────────────────────────────────────────── */

DUCKDB_EXTENSION_ENTRYPOINT(duckdb_connection connection,
                             duckdb_extension_info info,
                             struct duckdb_extension_access *access) {
    (void)info;
    (void)access;
    register_functions(connection);
    return true;
}
