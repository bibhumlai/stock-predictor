import time
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("RapidsBenchmark").getOrCreate()

# --- Configuration ---
NUM_ROWS = 10 * 1000 * 1000  # 10 Million Rows - adjusted to avoid OOM
SEED = 42

# --- 1. Generate two large DataFrames ---
print("Generating DataFrames...")
# DF1: ID (Long), Value (Float)
df1 = spark.range(0, NUM_ROWS, 1, 4).withColumnRenamed("id", "join_id")
df1 = df1.withColumn("value1", F.rand(SEED) * 1000)

# DF2: ID (Long), Key (String)
df2 = spark.range(0, NUM_ROWS, 1, 4).withColumnRenamed("id", "join_id")
df2 = df2.withColumn("key", F.lit("key_") + (F.rand(SEED + 1) * 10).cast("int"))

# --- 2. Define the GPU-Accelerated Operation (JOIN and AGGREGATE) ---
print("Starting GPU-Accelerated Operation...")
start_time = time.time()

result = df1.join(df2, on="join_id", how="inner") \
            .groupBy("key") \
            .agg(F.sum("value1").alias("sum_value")) \
            .orderBy(F.desc("sum_value")) \
            .collect() # Trigger the action and computation

end_time = time.time()

# --- 3. Print Results and Verification ---
print("\n" + "="*50)
print(f"Time taken for GPU-accelerated JOIN/AGG: {end_time - start_time:.2f} seconds")
print(f"Result sample (top 5): {result[:5]}")
print("="*50)

spark.stop()
