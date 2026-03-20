-- PG schema for storing embeddings with model provenance.
--
-- Run via: CALL postgres_execute('pg', '<sql>') or psql directly.

-- Model registry: intern model names
CREATE TABLE IF NOT EXISTS gazetteer.embedding_model (
    model_id        SERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL UNIQUE,    -- 'nomic-embed-text-v1.5-Q4_K_M'
    model_dim       INTEGER NOT NULL,        -- 768
    model_source    TEXT,                    -- 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    gguf_file       TEXT,                    -- 'nomic-embed-text-v1.5.Q4_K_M.gguf'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Embeddings: one row per (place, model) pair
-- Embedding stored as base64 for portability.
CREATE TABLE IF NOT EXISTS gazetteer.geonames_embedding (
    geonameid       INTEGER NOT NULL,
    model_id        INTEGER NOT NULL REFERENCES gazetteer.embedding_model(model_id),
    feature_code    TEXT,
    feature_name    TEXT,
    embedding_b64   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (geonameid, model_id)
);

CREATE INDEX IF NOT EXISTS idx_geo_emb_model
    ON gazetteer.geonames_embedding (model_id);
CREATE INDEX IF NOT EXISTS idx_geo_emb_feature
    ON gazetteer.geonames_embedding (feature_code);
