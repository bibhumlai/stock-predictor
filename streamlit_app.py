import streamlit as st
import pandas as pd
import plotly.express as px
from genai_predictor import get_ticker_analysis, TICKERS

st.set_page_config(page_title="GenAI Stock Predictor", layout="wide")

st.title("📈 GenAI Stock Predictor & Analyzer")
st.markdown("Powered by **Spark**, **Scikit-Learn**, and **Local LLM**")

# Sidebar
st.sidebar.header("Configuration")
selected_ticker = st.sidebar.selectbox("Select Ticker", TICKERS)

if st.sidebar.button("Analyze Stock"):
    with st.spinner(f"Analyzing {selected_ticker}... (This may take a moment for the LLM)"):
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
