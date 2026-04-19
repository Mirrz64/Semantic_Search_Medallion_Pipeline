from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, udf, explode, count, avg
from pyspark.sql.types import ArrayType, StringType
import os
import sys

# Python enviroment config
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# spark config
print("configuration")
spark = SparkSession.builder.appName("VectorEngine").config("spark.sql.shuffle.partitions", "20").config("spark.sql.caseSensitive", "true").config("spark.driver.memory", "4g").config("spark.sql.execution.pythonUDF.arrow.enabled", 'false').master("local[*]").getOrCreate()
print("gotten or created")
# set the log level
spark.sparkContext.setLogLevel("ERROR")

# Ingest from silver
SILVER_PATH = './data/silver/complete_reviews'
GOLD_PATH = './data/gold'
os.makedirs(GOLD_PATH, exist_ok=True)

silver_df = spark.read.parquet(SILVER_PATH)

# write the algorithm for semantic chunking
def recursive_chunker(text):
    try:
        if not text or len(str(text)) < 5: return []
            
        text = str(text) # to ensure 
        max_chars = 400
        overlap = 50
        separators = ["\n\n", "\n", ". ", " ", "", ","] # from largest to smallest order of preference in cutting
        
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
        return [] # In case of any unexpected error, return empty list to avoid crashing the pipeline
    
# Register the recursive chunker as a user defined function and its output
chunk_udf = udf(recursive_chunker, ArrayType(StringType()))

# Gold Table 1: dim Products
print("Extract product metadata")
dim_products = silver_df.select(
    col('parent_asin').alias('product_id'),
    col('title').alias('product_name'),
    col('main_category'),
    col('price')
).distinct()

print(f"Added {dim_products.count()} product into dim_products")
# write into the gold layer as dim_products
dim_products.coalesce(1).write.mode('overwrite').parquet(f'{GOLD_PATH}/dim_products')

# Gold Table 2: dim_users
print("Extracting User Personalization Data...")
dim_users = silver_df.groupBy("user_id").agg(
    count("*").alias("review_count"),
    avg("rating").alias("avg_rating_given")
)
print(f"Added {dim_users.count()} unique users to dim_users table.")
# write into the gold layer as dim_users
dim_users.coalesce(1).write.mode('overwrite').parquet(f"{GOLD_PATH}/dim_users")

#  GOLD TABLE 3: fact_review_vectors 
print("Applying Recursive Splitter for Vector Embeddings...")
fact_vectors = silver_df.repartition(20).withColumn("review_chunk", explode(chunk_udf(col("text")))) \
    .select(
        col("parent_asin").alias("product_id"),
        "user_id",
        "rating",
        "review_chunk"
    )
# write into gold layer as fact_vector
fact_vectors.coalesce(1).write.mode('overwrite').parquet(f"{GOLD_PATH}/fact_vectors")
print("Gold Layer Finished: 3 Tables ready for Database Loading.")