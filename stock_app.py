from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType, LongType, DoubleType
import yfinance as yf
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta

# Initialize Spark Session
spark = SparkSession.builder.appName("StockAnalysis").getOrCreate()

TICKERS = ['AAPL', 'MSFT', 'AMZN', 'PLTR', 'ACN']

def fetch_historical_data(tickers):
    print(f"Fetching 5 years of historical data for: {tickers}")
    all_data = []
    
    for ticker in tickers:
        try:
            # Fetch 5 years of data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y")
            
            # Reset index to get Date as a column
            hist = hist.reset_index()
            
            # Select relevant columns and add Ticker
            hist['Ticker'] = ticker
            hist['Date'] = hist['Date'].dt.date
            
            # Keep only necessary columns
            hist = hist[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            all_data.append(hist)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if not all_data:
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_data)
    
    # Create Spark DataFrame
    # Define schema to ensure correct types
    schema = StructType([
        StructField("Date", DateType(), True),
        StructField("Ticker", StringType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Volume", LongType(), True)
    ])
    
    spark_df = spark.createDataFrame(combined_df, schema=schema)
    return spark_df

def predict_trend(tickers):
    print(f"\nFetching news and predicting trends for: {tickers}")
    predictions = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            sentiment_score = 0
            count = 0
            
            if news:
                for item in news:
                    title = item.get('title', '')
                    # Calculate sentiment
                    blob = TextBlob(title)
                    sentiment_score += blob.sentiment.polarity
                    count += 1
            
            avg_sentiment = sentiment_score / count if count > 0 else 0
            
            # Prediction Logic
            if avg_sentiment > 0.1:
                trend = "GAIN"
            elif avg_sentiment < -0.1:
                trend = "LOSS"
            else:
                trend = "NEUTRAL"
                
            predictions.append((ticker, avg_sentiment, trend))
            
        except Exception as e:
            print(f"Error analyzing news for {ticker}: {e}")
            predictions.append((ticker, 0.0, "ERROR"))

    # Create DataFrame for predictions
    schema = StructType([
        StructField("Ticker", StringType(), True),
        StructField("Sentiment_Score", FloatType(), True),
        StructField("Predicted_Trend", StringType(), True)
    ])
    
    return spark.createDataFrame(predictions, schema=schema)

# --- Main Execution ---

# 1. Historical Data
print("--- Historical Data (Last 5 Years) ---")
historical_df = fetch_historical_data(TICKERS)
if historical_df:
    historical_df.show(20)
    print(f"Total historical records: {historical_df.count()}")
    
    # Save to CSV
    print("Saving historical data to historical_data.csv...")
    historical_df.toPandas().to_csv("historical_data.csv", index=False)
    print("Saved.")
else:
    print("No historical data fetched.")

# 2. Trend Prediction
print("\n--- Future Trend Prediction (Next Week) ---")
prediction_df = predict_trend(TICKERS)
prediction_df.show()

# Save to CSV
print("Saving predictions to predictions.csv...")
prediction_df.toPandas().to_csv("predictions.csv", index=False)
print("Saved.")

spark.stop()
