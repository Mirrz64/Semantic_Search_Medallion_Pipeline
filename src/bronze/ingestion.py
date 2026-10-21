"""
Bronze layer: ingest raw Amazon review data with PySpark.

Expected input: one or more newline-delimited JSON (JSONL) files under
RAW_DATA_PATH, in the shape used by the McAuley-Lab "Amazon Reviews"
datasets (https://amazon-reviews-2023.github.io/), e.g.:

    {"rating": 5.0, "title": "Great!", "text": "Works exactly as described.",
     "asin": "B000123456", "parent_asin": "B000123456",
     "user_id": "AF...", "timestamp": 1610000000000,
     "verified_purchase": true, "helpful_vote": 3}

The bronze stage does the absolute minimum: it lands the data close to
its raw shape (schema enforced so downstream stages don't fight
inferred types, obviously corrupt rows dropped) and partitions by
ingestion date so re-runs don't clobber history.

"""

import sys
from datetime import date
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.common.config import BRONZE_PATH, RAW_DATA_PATH  # noqa: E402

RAW_REVIEW_SCHEMA = StructType(
    [
        StructField("rating", DoubleType(), True),
        StructField("title", StringType(), True),
        StructField("text", StringType(), True),
        StructField("asin", StringType(), True),
        StructField("parent_asin", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("timestamp", LongType(), True),
        StructField("verified_purchase", BooleanType(), True),
        StructField("helpful_vote", LongType(), True),
    ]
)


def build_spark_session(app_name: str = "bronze-ingestion") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def ingest_raw_reviews(spark: SparkSession, raw_path: str) -> "pyspark.sql.DataFrame":
    """Read the raw JSONL review files with an enforced schema.

    Using ``mode="PERMISSIVE"`` with a corrupt-record column lets us
    count and inspect malformed rows instead of silently losing them,
    which matters once this runs against the full multi-GB dataset.
    """
    schema_with_corrupt = RAW_REVIEW_SCHEMA.add(
        StructField("_corrupt_record", StringType(), True)
    )

    df = (
        spark.read.schema(schema_with_corrupt)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .json(raw_path)
        .cache()
    )

    corrupt_count = df.filter(F.col("_corrupt_record").isNotNull()).count()
    if corrupt_count:
        print(f"[bronze] WARNING: dropping {corrupt_count} unparseable record(s)")

    clean_df = df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")

    # A row with no text and no title is not a review worth keeping —
    # everything else about "is this a valid record" is a Silver-layer
    # concern (dedup, trimming, type coercion belongs downstream).
    clean_df = clean_df.filter(
        F.col("text").isNotNull() | F.col("title").isNotNull()
    )

    return clean_df.withColumn("ingestion_date", F.lit(date.today().isoformat()))


def write_bronze(df: "pyspark.sql.DataFrame", bronze_path: str) -> None:
    (
        df.write.mode("append")
        .partitionBy("ingestion_date")
        .parquet(bronze_path)
    )


def main() -> None:
    spark = build_spark_session()
    try:
        raw_df = ingest_raw_reviews(spark, RAW_DATA_PATH)
        row_count = raw_df.count()
        print(f"[bronze] ingesting {row_count} review record(s) from {RAW_DATA_PATH}")
        write_bronze(raw_df, BRONZE_PATH)
        print(f"[bronze] wrote data to {BRONZE_PATH}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
