import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestClassifier
from model import fetch_stock_data, add_trend_label, add_features, train_model, get_company_info, predict_future_prices

def get_news_sentiment(company_name, api_key):
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    positive_words = ["surge", "gain", "profit", "growth", "up", "rise", "high", "strong", "beat", "record"]
    negative_words = ["fall", "drop", "loss", "down", "weak", "crash", "risk", "cut", "miss", "decline"]

    results = []
    for article in articles:
        title = article.get("title", "").lower()
        score = 0
        for word in positive_words:
            if word in title:
                score += 1
        for word in negative_words:
            if word in title:
                score -= 1

        sentiment = "Positive 🟢" if score > 0 else "Negative 🔴" if score < 0 else "Neutral 🟡"
        results.append({
            "Headline": article.get("title", ""),
            "Sentiment": sentiment,
        })

    return results

st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")

st.title("📈 StockSense")
st.subheader("Smart Stock Trend Analyzer for Beginner Investors")

market = st.radio("Select Market", ["🇮🇳 Indian Stocks (NSE)", "🇺🇸 US Stocks"])

indian_stocks = {
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

us_stocks = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Meta": "META",
    "Netflix": "NFLX",
    "Nvidia": "NVDA",
    "AMD": "AMD",
    "Uber": "UBER",
}

if market == "🇮🇳 Indian Stocks (NSE)":
    selected = st.selectbox("Select NSE Stock", list(indian_stocks.keys()))
    ticker = indian_stocks[selected]
    currency = "₹"
else:
    selected = st.selectbox("Select US Stock", list(us_stocks.keys()))
    ticker = us_stocks[selected]
    currency = "$"

if st.button("Analyze"):
    with st.spinner("Fetching data and training model..."):
        df = fetch_stock_data(ticker)
        df = add_trend_label(df)
        df = add_features(df)
        model, accuracy = train_model(df)

        features = ["MA7", "MA21", "Momentum", "Volatility", "Volume_Change", "RSI"]
        latest = df[features].iloc[-1:]
        prediction = model.predict(latest)[0]

        # 7-day trend prediction
        future_predictions = []
        last_row = df[features].iloc[-1].copy()
        for i in range(7):
            pred = model.predict([last_row.values])[0]
            future_predictions.append(pred)
            last_row["Momentum"] = last_row["Momentum"] * 0.95
            last_row["Volatility"] = last_row["Volatility"] * 0.98

        # Price prediction
        real_prices, predicted_prices, future_prices, last_30_index = predict_future_prices(df)

        # Signal
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

        # Company info
        try:
            info = get_company_info(ticker)
        except:
            info = None

        # RSI value
        rsi_value = df["RSI"].iloc[-1]
        if rsi_value > 70:
            rsi_signal = "Overbought 🔴"
        elif rsi_value < 30:
            rsi_signal = "Oversold 🟢"
        else:
            rsi_signal = "Neutral 🟡"

    # Company info section
    if info:
        st.markdown("---")
        st.subheader(f"🏢 {info['name']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sector", info["sector"])
        c2.metric("52W High", f"{currency}{info['52w_high']}")
        c3.metric("52W Low", f"{currency}{info['52w_low']}")

    # Main metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", f"{currency}{df['Close'].iloc[-1]:.2f}")
    col2.metric("Today's Trend", prediction)
    col3.metric("Recommendation", signal)
    col4.metric("Model Accuracy", f"{accuracy*100:.1f}%")

    color = "green" if prediction == "Up" else "red" if prediction == "Down" else "orange"
    st.markdown(f"### Today: :{color}[{prediction}] {'🟢' if prediction == 'Up' else '🔴' if prediction == 'Down' else '🟡'}")
    st.markdown(f"### Signal: :{signal_color}[{signal}]")

    # RSI
    st.markdown("---")
    st.subheader("📊 RSI Indicator")
    col1, col2 = st.columns(2)
    col1.metric("RSI Value", f"{rsi_value:.1f}")
    col2.metric("RSI Signal", rsi_signal)
    st.caption("RSI > 70 = Overbought (price may fall) | RSI < 30 = Oversold (price may rise) | 30-70 = Neutral")

    # Candlestick chart
    st.markdown("---")
    st.subheader("🕯️ Candlestick Chart (Last 90 Days)")
    df_candle = df.tail(90)
    fig = go.Figure(data=[go.Candlestick(
        x=df_candle.index,
        open=df_candle["Open"],
        high=df_candle["High"],
        low=df_candle["Low"],
        close=df_candle["Close"]
    )])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Real vs Predicted + Future price graph
    st.markdown("---")
    st.subheader("🔮 Price Prediction — Real vs Predicted + Next 7 Days")

    future_dates = pd.date_range(
        start=last_30_index[-1] + pd.Timedelta(days=1), periods=7, freq="B"
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=last_30_index,
        y=real_prices,
        name="Real Price",
        line=dict(color="cyan", width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=last_30_index,
        y=predicted_prices,
        name="Predicted Price",
        line=dict(color="orange", width=2, dash="dash")
    ))
    fig3.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        name="Future Forecast",
        line=dict(color="lime", width=2, dash="dot")
    ))
    fig3.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 7-day forecast table
    st.markdown("---")
    st.subheader("📅 7-Day Trend Forecast")
    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Predicted Trend": future_predictions
    })
    st.dataframe(forecast_df, use_container_width=True)

    # Forecast bar chart
    colors = []
    for t in future_predictions:
        if t == "Up":
            colors.append("green")
        elif t == "Down":
            colors.append("red")
        else:
            colors.append("orange")

    fig2 = go.Figure(data=[
        go.Bar(
            x=[f"Day {i+1}" for i in range(7)],
            y=[1] * 7,
            marker_color=colors,
            text=future_predictions,
            textposition="inside",
        )
    ])
    fig2.update_layout(
        title="7-Day Trend Visual",
        template="plotly_dark",
        height=300,
        yaxis=dict(visible=False),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    # News Sentiment
    st.markdown("---")
    st.subheader("📰 Latest News Sentiment")
    API_KEY = "44704190ffeb4e34b77ffcd083b5be88"
    news = get_news_sentiment(selected, API_KEY)
    if news:
        news_df = pd.DataFrame(news)
        st.dataframe(news_df[["Headline", "Sentiment"]], use_container_width=True)
    else:
        st.info("No news found.")

    # Recent data
    st.markdown("---")
    st.subheader("📋 Recent Price Data")
    st.dataframe(df[["Close", "MA7", "MA21", "RSI", "Trend"]].tail(10))