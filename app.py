import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model import fetch_stock_data, add_trend_label, add_features

st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")

st.title("📈 StockSense")
st.subheader("Smart Stock Trend Analyzer for Beginner Investors")

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Wipro": "WIPRO.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Axis Bank": "AXISBANK.NS",
}

selected = st.selectbox("Select NSE Stock", list(stocks.keys()))
ticker = stocks[selected]

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

        # 7-day prediction
        future_predictions = []
        last_row = df[features].iloc[-1].copy()
        for i in range(7):
            pred = model.predict([last_row.values])[0]
            future_predictions.append(pred)
            last_row["Momentum"] = last_row["Momentum"] * 0.95
            last_row["Volatility"] = last_row["Volatility"] * 0.98

        # Buy/Sell/Hold signal
        up_count = future_predictions.count("Up")
        down_count = future_predictions.count("Down")
        if up_count >= 4:
            signal = "BUY 🟢"
            signal_color = "green"
        elif down_count >= 4:
            signal = "AVOID 🔴"
            signal_color = "red"
        else:
            signal = "HOLD 🟡"
            signal_color = "orange"

    # Current metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Close Price", f"₹{df['Close'].iloc[-1]:.2f}")
    col2.metric("Today's Trend", prediction)
    col3.metric("Recommendation", signal)

    color = "green" if prediction == "Up" else "red" if prediction == "Down" else "orange"
    st.markdown(f"### Today: :{color}[{prediction}] {'🟢' if prediction == 'Up' else '🔴' if prediction == 'Down' else '🟡'}")
    st.markdown(f"### Signal: :{signal_color}[{signal}]")

    # 7-day forecast table
    st.subheader("📅 7-Day Trend Forecast")
    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Predicted Trend": future_predictions
    })
    st.dataframe(forecast_df, use_container_width=True)

    # Recent data
    st.subheader("Recent Price Data")
    st.dataframe(df[["Close", "MA7", "MA21", "Trend"]].tail(10))

    st.subheader("Price Chart")
    st.line_chart(df["Close"])