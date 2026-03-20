#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "blobembed.h"

namespace nb = nanobind;

static bool g_initialized = false;

static void ensure_init() {
    if (!g_initialized) {
        blobembed_init();
        g_initialized = true;
    }
}

static nb::str py_load_model(const std::string &name, const std::string &path) {
    ensure_init();
    if (blobembed_load_model(name.c_str(), path.c_str()) != 0) {
        throw nb::value_error(blobembed_errmsg());
    }
    return nb::str("ok");
}

static nb::str py_load_hf_model(const std::string &name,
                                const std::string &repo_id,
                                const std::string &filename,
                                std::optional<std::string> ref) {
    ensure_init();
    const char *ref_c = ref ? ref->c_str() : nullptr;
    if (blobembed_load_hf_model(name.c_str(), repo_id.c_str(),
                                filename.c_str(), ref_c) != 0) {
        throw nb::value_error(blobembed_errmsg());
    }
    return nb::str("ok");
}

static std::vector<float> py_embed(const std::string &model_name,
                                   const std::string &text) {
    int dim = 0;
    float *embd = blobembed_embed(model_name.c_str(),
                                  text.c_str(), text.size(), &dim);
    if (!embd) {
        throw nb::value_error(blobembed_errmsg());
    }

    std::vector<float> result(embd, embd + dim);
    blobembed_free_embedding(embd);
    return result;
}

static int py_embed_dim(const std::string &model_name) {
    int dim = blobembed_model_dim(model_name.c_str());
    if (dim < 0) {
        throw nb::value_error("model not loaded");
    }
    return dim;
}

static int py_token_count(const std::string &model_name,
                          const std::string &text) {
    int count = blobembed_token_count(model_name.c_str(),
                                     text.c_str(), text.size());
    if (count < 0) {
        throw nb::value_error(blobembed_errmsg());
    }
    return count;
}

static void py_unload_model(const std::string &name) {
    blobembed_unload_model(name.c_str());
}

static double py_cosine_sim(const std::vector<float> &a,
                            const std::vector<float> &b) {
    double sim = blobembed_cosine_sim(a.data(), (int)a.size(),
                                     b.data(), (int)b.size());
    if (std::isnan(sim)) {
        throw nb::value_error("dimension mismatch");
    }
    return sim;
}

static nb::object py_resolve_hf_path(const std::string &repo_id,
                                     const std::string &filename,
                                     std::optional<std::string> ref) {
    const char *ref_c = ref ? ref->c_str() : nullptr;
    char *path = blobembed_resolve_hf_path(repo_id.c_str(),
                                           filename.c_str(), ref_c);
    if (!path) {
        return nb::none();
    }
    nb::str result(path);
    blobembed_free_string(path);
    return result;
}

NB_MODULE(blobembed_ext, m) {
    m.doc() = "Text embeddings via GGUF models (llama.cpp)";

    m.def("load_model", &py_load_model,
          nb::arg("name"), nb::arg("path"),
          "Load a GGUF model from a filesystem path");

    m.def("load_hf_model", &py_load_hf_model,
          nb::arg("name"), nb::arg("repo_id"), nb::arg("filename"),
          nb::arg("ref") = nb::none(),
          "Load a GGUF model from the HuggingFace cache");

    m.def("embed", &py_embed,
          nb::arg("model_name"), nb::arg("text"),
          "Compute the embedding for a text string (returns list of floats)");

    m.def("embed_dim", &py_embed_dim,
          nb::arg("model_name"),
          "Return the embedding dimension for a loaded model");

    m.def("token_count", &py_token_count,
          nb::arg("model_name"), nb::arg("text"),
          "Count the number of tokens in a text string");

    m.def("unload_model", &py_unload_model,
          nb::arg("name"),
          "Unload a previously loaded model");

    m.def("resolve_hf_path", &py_resolve_hf_path,
          nb::arg("repo_id"), nb::arg("filename"),
          nb::arg("ref") = nb::none(),
          "Resolve the filesystem path for a model in the HF cache");

    m.def("cosine_sim", &py_cosine_sim,
          nb::arg("a"), nb::arg("b"),
          "Compute cosine similarity between two embedding vectors");
}
