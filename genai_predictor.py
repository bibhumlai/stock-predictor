import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

TICKERS = ['AAPL', 'MSFT', 'AMZN', 'PLTR', 'ACN']

try:
    from transformers import pipeline
    print("Loading AI Model (GPT-2)... this may take a moment...")
    # Initialize text generation pipeline
    # We use distilgpt2 for speed and smaller size, or gpt2
    generator = pipeline('text-generation', model='distilgpt2')
except ImportError:
    print("Transformers library not found. AI commentary will be disabled.")
    generator = None
except Exception as e:
    print(f"Error loading model: {e}")
    generator = None

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    # Get last 6 months
    hist = stock.history(period="6mo")
    return hist

def predict_future(ticker, hist):
    # Prepare data for Linear Regression
    hist = hist.reset_index()
    hist['DateOrdinal'] = pd.to_datetime(hist['Date']).map(datetime.toordinal)
    
    # Use last 60 days for trend
    recent_data = hist.tail(60)
    
    X = recent_data[['DateOrdinal']].values
    y = recent_data['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = recent_data['Date'].iloc[-1]
    last_ordinal = last_date.toordinal()
    current_price = recent_data['Close'].iloc[-1]
    
    # Predict Next Week (+7 days)
    next_week_ordinal = np.array([[last_ordinal + 7]])
    pred_week = model.predict(next_week_ordinal)[0]
    
    # Predict Next Month (+30 days)
    next_month_ordinal = np.array([[last_ordinal + 30]])
    pred_month = model.predict(next_month_ordinal)[0]
    
    return current_price, pred_week, pred_month

def generate_commentary(ticker, current, week, month):
    # Determine trend
    trend = "bullish" if week > current else "bearish"
    percent_change = ((week - current) / current) * 100
    
    prompt = f"The stock {ticker} is currently trading at ${current:.2f}. Analysis suggests a {trend} trend with a predicted move of {percent_change:.1f}% next week. Investors should"
    
    # Generate text
    try:
        if generator:
            output = generator(prompt, max_length=60, num_return_sequences=1, truncation=True)
            commentary = output[0]['generated_text']
            # Clean up unfinished sentences
            last_period = commentary.rfind('.')
            if last_period != -1:
                commentary = commentary[:last_period+1]
            return commentary
        else:
            return "AI Commentary unavailable (Model not loaded)."
    except Exception as e:
        return f"Could not generate commentary: {e}"

def get_ticker_analysis(ticker):
    # print(f"\n--- Analyzing {ticker} ---")
    hist = get_stock_data(ticker)
    
    if hist.empty:
        return None
        
    current, pred_week, pred_month = predict_future(ticker, hist)
    commentary = generate_commentary(ticker, current, pred_week, pred_month)
    
    return {
        "ticker": ticker,
        "current_price": current,
        "pred_week": pred_week,
        "pred_month": pred_month,
        "commentary": commentary,
        "history": hist
    }

if __name__ == "__main__":
    print("--- Local GenAI Stock Predictor ---")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    for ticker in TICKERS:
        result = get_ticker_analysis(ticker)
        if result:
            print(f"\n--- Analyzing {result['ticker']} ---")
            print(f"Current Price: ${result['current_price']:.2f}")
            print(f"Prediction (Next Week):  ${result['pred_week']:.2f}")
            print(f"Prediction (Next Month): ${result['pred_month']:.2f}")
            print(f"\nAI Commentary:\n{result['commentary']}")
