import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import plotly.express as px
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="GenAI Stock Predictor", layout="wide")

# --- Configuration ---
TICKERS = ['AAPL', 'MSFT', 'AMZN', 'PLTR', 'ACN']

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    try:
        from transformers import pipeline
        # Use distilgpt2 for speed on CPU
        return pipeline('text-generation', model='distilgpt2')
    except ImportError:
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

generator = load_model()

# --- Logic Functions ---
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
            # Limit max_length to avoid long wait times on CPU
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

# --- UI Layout ---
st.title("📈 GenAI Stock Predictor")
st.markdown("Powered by **Scikit-Learn** and **DistilGPT-2** (Running on Streamlit Cloud)")

# Sidebar
st.sidebar.header("Configuration")
selected_ticker = st.sidebar.selectbox("Select Ticker", TICKERS)

if st.sidebar.button("Analyze Stock"):
    with st.spinner(f"Analyzing {selected_ticker}... (AI inference may take a few seconds on CPU)"):
        data = get_ticker_analysis(selected_ticker)
        
        if not data:
            st.error(f"Could not fetch data for {selected_ticker}")
        else:
            # Layout
            col1, col2, col3 = st.columns(3)
            
            current = data['current_price']
            week = data['pred_week']
            month = data['pred_month']
            
            week_delta = ((week - current) / current) * 100
            month_delta = ((month - current) / current) * 100
            
            with col1:
                st.metric("Current Price", f"${current:.2f}")
            with col2:
                st.metric("Next Week Prediction", f"${week:.2f}", f"{week_delta:.2f}%")
            with col3:
                st.metric("Next Month Prediction", f"${month:.2f}", f"{month_delta:.2f}%")
            
            # Chart
            st.subheader("Price History & Trend")
            hist = data['history'].reset_index()
            fig = px.line(hist, x="Date", y="Close", title=f"{selected_ticker} - Last 6 Months")
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Commentary
            st.subheader("🤖 AI Market Commentary")
            st.info(data['commentary'])

else:
    st.info("Select a ticker and click 'Analyze Stock' to begin.")
