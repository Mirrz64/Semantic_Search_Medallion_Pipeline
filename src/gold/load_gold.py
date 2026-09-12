"""
Gold layer: embed silver-tier reviews and load them into Postgres (pgvector).

Why this stage drops out of Spark:
Sentence-Transformers loads a model into a single process's memory and
batches inference itself; wrapping it in Spark UDFs adds serialization
overhead without adding parallelism that actually helps here (Spark's
strength is data-parallel *distributed* work, and a GPU/CPU embedding
model is better fed with plain, sequential batches). Pandas +
psycopg2's `execute_values` is the simpler, faster path from a few
hundred thousand rows of silver Parquet to indexed vectors in Postgres.

For genuinely large (100M+ row) datasets, replace the pandas read with
chunked Parquet reads (`pyarrow.parquet.ParquetFile.iter_batches`) so
memory stays bounded — the batching logic below already assumes
chunked input for exactly that reason.

Run:
    python -m src.gold.load_gold
"""

import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.common.config import (  # noqa: E402
    BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    GOLD_TABLE,
    POSTGRES,
    SILVER_PATH,
)

CREATE_TABLE_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {GOLD_TABLE} (
    review_id           TEXT PRIMARY KEY,
    asin                TEXT NOT NULL,
    parent_asin         TEXT,
    rating              DOUBLE PRECISION,
    review_text         TEXT NOT NULL,
    review_date         TIMESTAMP,
    verified_purchase   BOOLEAN,
    helpful_vote        BIGINT,
    embedding           VECTOR({EMBEDDING_DIM})
);

-- Approximate nearest-neighbour index. `lists` should be roughly
-- sqrt(row_count) per the pgvector docs; 100 is a reasonable default
-- for a few hundred thousand rows and can be tuned later with
-- `REINDEX` once the true row count is known.
CREATE INDEX IF NOT EXISTS {GOLD_TABLE}_embedding_idx
    ON {GOLD_TABLE}
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

UPSERT_SQL = f"""
INSERT INTO {GOLD_TABLE} (
    review_id, asin, parent_asin, rating, review_text,
    review_date, verified_purchase, helpful_vote, embedding
)
VALUES %s
ON CONFLICT (review_id) DO UPDATE SET
    rating = EXCLUDED.rating,
    review_text = EXCLUDED.review_text,
    embedding = EXCLUDED.embedding;
"""


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def iter_silver_batches(silver_path: str, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield the silver dataset in row-count-bounded chunks.

    Parquet is read with pandas here for simplicity; swap for
    `pyarrow.parquet.ParquetFile(...).iter_batches(batch_size=...)` if
    the silver dataset grows past what comfortably fits in memory.
    """
    full_df = pd.read_parquet(silver_path)
    for start in range(0, len(full_df), batch_size):
        yield full_df.iloc[start : start + batch_size]


def embed_and_load(conn, model: SentenceTransformer, silver_path: str) -> int:
    total_loaded = 0
    for batch in iter_silver_batches(silver_path, BATCH_SIZE):
        texts = batch["review_text"].tolist()
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        rows = [
            (
                row.review_id,
                row.asin,
                row.parent_asin,
                float(row.rating) if pd.notna(row.rating) else None,
                row.review_text,
                row.review_date,
                bool(row.verified_purchase) if pd.notna(row.verified_purchase) else None,
                int(row.helpful_vote) if pd.notna(row.helpful_vote) else None,
                embedding.tolist(),
            )
            for row, embedding in zip(batch.itertuples(index=False), embeddings)
        ]

        with conn.cursor() as cur:
            execute_values(cur, UPSERT_SQL, rows)
        conn.commit()

        total_loaded += len(rows)
        print(f"[gold] loaded {total_loaded} row(s) so far...")

    return total_loaded


def main() -> None:
    print(f"[gold] loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    conn = psycopg2.connect(POSTGRES.as_dsn())
    try:
        ensure_schema(conn)
        total = embed_and_load(conn, model, SILVER_PATH)
        print(f"[gold] done — {total} review(s) embedded and upserted into '{GOLD_TABLE}'")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
