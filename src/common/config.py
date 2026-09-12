"""
Central configuration for the Semantic Search Medallion Pipeline.

All stages (bronze, silver, gold, search) import from here instead of
reading environment variables directly, so there is a single source
of truth for paths, the embedding model name, and the database
connection details.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data lake paths (local disk for this project; swap for S3/ADLS/GCS paths in
# production by pointing these at "s3a://..." / "abfss://..." / "gs://..."
# URIs — Spark handles all three transparently as long as the matching
# hadoop connector jar is on the classpath).
# ---------------------------------------------------------------------------
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", str(PROJECT_ROOT / "data" / "raw"))
BRONZE_PATH = os.getenv("BRONZE_PATH", str(PROJECT_ROOT / "data" / "bronze"))
SILVER_PATH = os.getenv("SILVER_PATH", str(PROJECT_ROOT / "data" / "silver"))

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2 produces 384-dim vectors, is fast on CPU, and is a common
# default for semantic search demos. Swap for a larger model
# (e.g. all-mpnet-base-v2, 768-dim) if recall matters more than latency —
# just remember to update EMBEDDING_DIM and the pgvector column width to match.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# ---------------------------------------------------------------------------
# Postgres / pgvector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname: str = os.getenv("POSTGRES_DB", "semantic_search")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "postgres")

    def as_dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


POSTGRES = PostgresConfig()

GOLD_TABLE = os.getenv("GOLD_TABLE", "review_embeddings")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))
