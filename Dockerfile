# 1. Base Image with CUDA support
# Use a robust base image that matches your host's CUDA driver capabilities.
# The `runtime` tag ensures GPU libraries are included.
FROM nvcr.io/nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set environment variables for Spark and RAPIDS versions
# Check the NVIDIA RAPIDS documentation for the latest compatible versions.
ENV SPARK_VERSION=3.5.0
ENV RAPIDS_VERSION=24.06.0
ENV SCALA_VERSION=2.12
ENV CUDF_CLASSIFIER=cuda12
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install Dependencies (Java, Python, Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    wget \
    python$PYTHON_VERSION \
    python3-pip \
    # Install necessary Python tools
    && pip install --no-cache-dir jupyterlab pandas pyspark==$SPARK_VERSION yfinance textblob torch transformers scikit-learn scipy streamlit plotly

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Download and extract Apache Spark
RUN wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.tgz && \
    tar -xzf spark-$SPARK_VERSION-bin-hadoop3.tgz && \
    mv spark-$SPARK_VERSION-bin-hadoop3 /opt/spark && \
    rm spark-$SPARK_VERSION-bin-hadoop3.tgz

# Set Spark Home
ENV SPARK_HOME=/opt/spark

# 3. Download RAPIDS Accelerator Jars and Discovery Script
# The jars are critical for GPU-acceleration.
# The discovery script is used by Spark to find and allocate GPU resources.
RUN wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/$RAPIDS_VERSION/rapids-4-spark_${SCALA_VERSION}-${RAPIDS_VERSION}.jar -O $SPARK_HOME/jars/rapids-4-spark.jar && \
    wget https://repo1.maven.org/maven2/ai/rapids/cudf/$RAPIDS_VERSION/cudf-${RAPIDS_VERSION}-${CUDF_CLASSIFIER}.jar -O $SPARK_HOME/jars/cudf.jar

# Copy GPU Discovery Script
COPY getGpusResources.sh $SPARK_HOME/bin/getGpusResources.sh
RUN chmod +x $SPARK_HOME/bin/getGpusResources.sh

# 4. Final Setup
WORKDIR /app
CMD ["/bin/bash"]
