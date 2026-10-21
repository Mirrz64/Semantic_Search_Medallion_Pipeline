"""
Semantic search CLI — embeds a query and returns the closest reviews by
cosine similarity, using pgvector's `<=>` distance operator.

This is the piece the project's name promises but the pipeline didn't
actually have yet: everything upstream (bronze -> silver -> gold) only
gets data *into* pgvector. This is what gets a ranked answer back out.

Usage:
    python -m src.search.semantic_search "battery life disappointing" --top-k 5
    python -m src.search.semantic_search "great gift for a teenager" --min-rating 4
"""

import argparse
import sys
from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.common.config import EMBEDDING_MODEL_NAME, GOLD_TABLE, IVFFLAT_PROBES, POSTGRES  # noqa: E402

SEARCH_SQL = f"""
SELECT
    review_id,
    asin,
    rating,
    review_text,
    review_date,
    1 - (embedding <=> %(query_embedding)s::vector) AS similarity
FROM {GOLD_TABLE}
WHERE (%(min_rating)s::float IS NULL OR rating >= %(min_rating)s)
ORDER BY embedding <=> %(query_embedding)s::vector
LIMIT %(top_k)s;
"""


def run_search(query: str, top_k: int = 5, min_rating: float | None = None) -> list[dict]:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    conn = psycopg2.connect(POSTGRES.as_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET ivfflat.probes = {IVFFLAT_PROBES};")
            cur.execute(
                SEARCH_SQL,
                {
                    "query_embedding": query_embedding,
                    "min_rating": min_rating,
                    "top_k": top_k,
                },
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def truncate_snippet(text: str, max_len: int = 160) -> str:
    """Collapse newlines and cap length for display — split out as a pure
    function (no I/O) so it's straightforward to unit test."""
    flattened = text.replace("\n", " ")
    if len(flattened) <= max_len:
        return flattened
    return flattened[:max_len] + "..."


def print_results(query: str, results: list[dict]) -> None:
    print(f'\nTop {len(results)} result(s) for: "{query}"\n')
    for rank, row in enumerate(results, start=1):
        print(
            f"{rank}. similarity={row['similarity']:.3f}  "
            f"rating={row['rating']}  asin={row['asin']}\n"
            f"   {truncate_snippet(row['review_text'])}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over Amazon reviews.")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--min-rating", type=float, default=None, help="Optional minimum star rating filter"
    )
    args = parser.parse_args()

    results = run_search(args.query, top_k=args.top_k, min_rating=args.min_rating)
    print_results(args.query, results)


if __name__ == "__main__":
    main()
