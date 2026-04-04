import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from model import fetch_stock_data, add_trend_label, add_features

st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")

st.title("📈 StockSense")
st.subheader("Smart Stock Trend Analyzer for Beginner Investors")

ticker = st.text_input("Enter NSE Stock Symbol", value="RELIANCE.NS")

if st.button("Analyze"):
    with st.spinner("Fetching data and training model..."):
        df = fetch_stock_data(ticker)
        df = add_trend_label(df)
        df = add_features(df)

        features = ["MA7", "MA21", "Momentum", "Volatility", "Volume_Change"]
        X = df[features]
        y = df["Trend"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        latest = df[features].iloc[-1:]
        prediction = model.predict(latest)[0]

    st.metric("Latest Close Price", f"₹{df['Close'].iloc[-1]:.2f}")
    st.metric("Trend Prediction", prediction)

    color = "green" if prediction == "Up" else "red" if prediction == "Down" else "orange"
    st.markdown(f"### Trend: :{color}[{prediction}] {'🟢' if prediction == 'Up' else '🔴' if prediction == 'Down' else '🟡'}")

    st.subheader("Recent Price Data")
    st.dataframe(df[["Close", "MA7", "MA21", "Trend"]].tail(10))

    st.subheader("Price Chart")
    st.line_chart(df["Close"])