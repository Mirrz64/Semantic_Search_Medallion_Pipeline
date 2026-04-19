# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit
import os

# spark config
spark = SparkSession.builder.appName("VectorEngine").config("spark.sql.shuffle.partitions", "20").config("spark.sql.caseSensitive", "true").config("spark.driver.memory", "4g").master("local[*]").getOrCreate()
# set the log level
spark.sparkContext.setLogLevel("ERROR")

# Ingest the metadata files
print("Reading the metadata file...")
meta_df = spark.read.json("./data/bronze/metadata/*.jsonl.gz")

# select the needed columns
meta_clean = meta_df.select(
    col('parent_asin').alias("metadata_parent_asin"),
    col('title'),
    col("main_category"),
    coalesce(col('price'), lit(0.0)).alias('price')
    ).drop_duplicates(["metadata_parent_asin"])

print(f"Processed {meta_clean.count()} data points")

# Ingest Reviews data
print("Reading Review files...")
reviews_df = spark.read.json("./data/bronze/review/*.jsonl.gz")

# select the needed columns
reviews_clean = reviews_df.select(
    col("parent_asin"),
    col("user_id"),
    col("rating"),
    col("text"),
    col('timestamp')
).filter("text IS NOT NULL")

print(f"Processed {reviews_clean.count()} data points")

# Join
silver_join = reviews_clean.join(meta_clean, reviews_clean.parent_asin == meta_clean.metadata_parent_asin, "inner").drop("metadata_parent_asin")

# write into the silver layer
SILVER_PATH = './data/silver'

os.makedirs(SILVER_PATH, exist_ok=True)

silver_join.coalesce(1).write.mode('overwrite').parquet(f"{SILVER_PATH}/complete_reviews")

print(f"Silver Layer is now completed, processed {silver_join.count()} rows")