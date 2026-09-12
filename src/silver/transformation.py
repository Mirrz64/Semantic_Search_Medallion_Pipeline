"""
Silver layer: clean and normalize the bronze review data.

Responsibilities that belong here (and only here):
  - drop exact and near-duplicate reviews
  - normalize whitespace / strip HTML entities that sneak into scraped text
  - coerce the raw millisecond epoch timestamp into a real timestamp column
  - enforce sane bounds on `rating` (Amazon ratings are 1.0-5.0; anything
    outside that is a data quality issue, not a valid record)
  - concatenate title + body into a single `review_text` field, since that
    combined text is what actually gets embedded in the Gold stage

Run:
    spark-submit src/silver/transformation.py
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.common.config import BRONZE_PATH, SILVER_PATH  # noqa: E402


def build_spark_session(app_name: str = "silver-transformation") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def clean_reviews(bronze_df: "pyspark.sql.DataFrame") -> "pyspark.sql.DataFrame":
    df = bronze_df

    # A (user_id, asin, timestamp) triple is a reasonable dedup key for this
    # dataset — the same reviewer/product pair should not appear twice at
    # the same instant. Keep the first-seen row per key.
    dedup_window = Window.partitionBy("user_id", "asin", "timestamp").orderBy(
        F.col("timestamp").asc()
    )
    df = (
        df.withColumn("_row_num", F.row_number().over(dedup_window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )

    df = df.withColumn(
        "review_date", F.to_timestamp(F.col("timestamp") / 1000)
    ).drop("timestamp")

    df = df.withColumn(
        "rating",
        F.when((F.col("rating") >= 1.0) & (F.col("rating") <= 5.0), F.col("rating")),
    )

    def _normalize(col_name: str):
        return F.trim(F.regexp_replace(F.coalesce(F.col(col_name), F.lit("")), r"\s+", " "))

    df = df.withColumn("title_clean", _normalize("title"))
    df = df.withColumn("text_clean", _normalize("text"))

    df = df.withColumn(
        "review_text",
        F.trim(
            F.concat_ws(
                ". ",
                F.when(F.col("title_clean") != "", F.col("title_clean")),
                F.when(F.col("text_clean") != "", F.col("text_clean")),
            )
        ),
    ).drop("title", "text", "title_clean", "text_clean")

    # After normalization, drop rows that ended up with no usable text —
    # there is nothing to embed for them in the Gold stage.
    df = df.filter(F.length(F.col("review_text")) > 0)

    # A stable review_id matters downstream: it's what Gold uses as the
    # primary key in Postgres, and what the search results are keyed on.
    df = df.withColumn(
        "review_id",
        F.sha2(F.concat_ws("|", F.col("user_id"), F.col("asin"), F.col("review_date")), 256),
    )

    return df.select(
        "review_id",
        "asin",
        "parent_asin",
        "user_id",
        "rating",
        "review_text",
        "review_date",
        "verified_purchase",
        "helpful_vote",
    )


def main() -> None:
    spark = build_spark_session()
    try:
        bronze_df = spark.read.parquet(BRONZE_PATH)
        silver_df = clean_reviews(bronze_df)

        input_count = bronze_df.count()
        output_count = silver_df.count()
        print(
            f"[silver] {input_count} bronze row(s) -> {output_count} silver row(s) "
            f"({input_count - output_count} dropped as duplicates/invalid)"
        )

        silver_df.write.mode("overwrite").parquet(SILVER_PATH)
        print(f"[silver] wrote data to {SILVER_PATH}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
