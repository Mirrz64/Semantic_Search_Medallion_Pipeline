from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, count, avg
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import pandas_udf
import pandas as pd
import os
import sys

# -------------------------------
# Python environment config
# -------------------------------
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# -------------------------------
# Spark config
# -------------------------------
print("configuration")

spark = (
    SparkSession.builder
    .appName("VectorEngine")
    .config("spark.sql.shuffle.partitions", "20")
    .config("spark.sql.caseSensitive", "true")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")  # ENABLE ARROW
    .master("local[*]")
    .getOrCreate()
)

print("gotten or created")

spark.sparkContext.setLogLevel("ERROR")

# -------------------------------
# Paths
# -------------------------------
SILVER_PATH = './data/silver/complete_reviews'
GOLD_PATH = './data/gold'
os.makedirs(GOLD_PATH, exist_ok=True)

# -------------------------------
# Read Silver Layer
# -------------------------------
silver_df = spark.read.parquet(SILVER_PATH)

# Cache to avoid recomputation
silver_df = silver_df.cache()
silver_df.count()

# -------------------------------
# Recursive Chunking Function
# -------------------------------
def recursive_chunker(text):
    try:
        if not text or len(str(text)) < 5:
            return []

        text = str(text)
        max_chars = 400
        overlap = 50
        separators = ["\n\n", "\n", ". ", " ", "", ","]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))

            if end < len(text):
                for sep in separators:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1 and last_sep > start:
                        end = last_sep + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end

        return [c for c in chunks if len(c) > 10]

    except Exception:
        return []

# -------------------------------
# Pandas UDF (VECTORISED)
# -------------------------------
@pandas_udf(ArrayType(StringType()))
def chunk_udf(text_series: pd.Series) -> pd.Series:
    return text_series.apply(recursive_chunker)

# -------------------------------
# GOLD TABLE 1: dim_products
# -------------------------------
print("Extract product metadata")

dim_products = silver_df.select(
    col('parent_asin').alias('product_id'),
    col('title').alias('product_name'),
    col('main_category'),
    col('price')
).distinct()

print(f"Added {dim_products.count()} product into dim_products")

dim_products.write.mode('overwrite').parquet(f'{GOLD_PATH}/dim_products')

# -------------------------------
# GOLD TABLE 2: dim_users
# -------------------------------
print("Extracting User Personalization Data...")

dim_users = silver_df.groupBy("user_id").agg(
    count("*").alias("review_count"),
    avg("rating").alias("avg_rating_given")
)

print(f"Added {dim_users.count()} unique users to dim_users table.")

dim_users.write.mode('overwrite').parquet(f"{GOLD_PATH}/dim_users")

# -------------------------------
# GOLD TABLE 3: fact_review_vectors
# -------------------------------
print("Applying Recursive Splitter for Vector Embeddings...")

fact_vectors = (
    silver_df
    .withColumn("review_chunk", explode(chunk_udf(col("text"))))
    .select(
        col("parent_asin").alias("product_id"),
        "user_id",
        "rating",
        "review_chunk"
    )
)

fact_vectors.write.mode('overwrite').parquet(f"{GOLD_PATH}/fact_vectors")

print("Gold Layer Finished: 3 Tables ready for Database Loading.")