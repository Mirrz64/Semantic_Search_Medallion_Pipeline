from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType
from sentence_transformers import SentenceTransformer

# Spark Config
spark = SparkSession.builder \
    .appName("VectorEngine") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.sql.caseSensitive", "true") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load the Gold Chunks
chunks_df = spark.read.parquet("./data/gold/fact_vectors")

# Setup the AI Model as a UDF
model = SentenceTransformer('all-MiniLM-L6-v2')

@udf(returnType=ArrayType(FloatType()))
def get_embedding(text):
    if not text: return []
    # Turning the text into a list of 384 numbers
    return model.encode(text).tolist()

# 3. Generate the Vectors
print("Generating Embeddings...")
final_vectors_df = chunks_df.withColumn("embedding", get_embedding(col("review_chunk")))

# 4. Save to a final gold folder
final_vectors_df.write.mode("overwrite").parquet("./data/gold/final_vector_storage")
print("Done! You now have a Parquet file containing vector embeddings.")