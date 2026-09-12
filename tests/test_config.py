import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import config  # noqa: E402


def test_embedding_dim_matches_default_model():
    # all-MiniLM-L6-v2 produces 384-dim vectors; if EMBEDDING_MODEL_NAME
    # changes without updating EMBEDDING_DIM, the pgvector column width
    # in load_gold.py will silently mismatch the model's real output —
    # this test is a tripwire for that.
    assert config.EMBEDDING_DIM == 384
    assert config.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"


def test_postgres_dsn_contains_all_fields():
    dsn = config.POSTGRES.as_dsn()
    for field in ["host=", "port=", "dbname=", "user=", "password="]:
        assert field in dsn
