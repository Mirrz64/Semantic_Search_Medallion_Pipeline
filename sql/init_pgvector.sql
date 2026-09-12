-- Runs automatically on first container start via docker-compose's
-- /docker-entrypoint-initdb.d mount. Mirrors the schema that
-- src/gold/load_gold.py also creates idempotently (CREATE ... IF NOT
-- EXISTS) — the two are kept in sync deliberately so the schema is
-- reviewable here without reading Python.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS review_embeddings (
    review_id           TEXT PRIMARY KEY,
    asin                TEXT NOT NULL,
    parent_asin          TEXT,
    rating               DOUBLE PRECISION,
    review_text          TEXT NOT NULL,
    review_date          TIMESTAMP,
    verified_purchase    BOOLEAN,
    helpful_vote         BIGINT,
    embedding            VECTOR(384)
);

CREATE INDEX IF NOT EXISTS review_embeddings_embedding_idx
    ON review_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS review_embeddings_asin_idx
    ON review_embeddings (asin);
