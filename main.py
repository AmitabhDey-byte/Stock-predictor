import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="yo its me Amitabh",
    layout="centered"
)

model = joblib.load("xgboost_stock_model.pkl")

analyzer = SentimentIntensityAnalyzer()

st.title("AI Powered Stock Predictor")

ticker = st.text_input("Enter Stock Ticker", "RELIANCE.NS")

news_text = st.text_area(
    "Enter Latest News About the Company",
    "Reliance reports strong quarterly profits and expansion plans."
)

if st.button("Predict"):

    stock = yf.download(ticker, period="5d")

    latest = stock.iloc[-1]

    close = float(latest["Close"])
    high = float(latest["High"])
    low = float(latest["Low"])
    open_price = float(latest["Open"])
    volume = float(latest["Volume"])

    sentiment = analyzer.polarity_scores(news_text)["compound"]

    input_data = np.array([[
        close,high,low,open_price,volume,sentiment]], dtype=float)

    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction for {ticker}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Current Close", f"{close:.2f}")

    with col2:
        st.metric("Predicted Next Close", f"{prediction:.2f}")

    if prediction > close:
        st.success(" Predicted Trend: BULLISH")
    else:
        st.error(" Predicted Trend: BEARISH")

    st.subheader("Market Sentiment")

    if sentiment > 0:
        st.success(f"Positive Sentiment Score: {sentiment:.2f}")
    elif sentiment < 0:
        st.error(f"Negative Sentiment Score: {sentiment:.2f}")
    else:
        st.warning("Neutral Sentiment")

    st.subheader("Recent Stock Data")

    st.dataframe(stock.tail())