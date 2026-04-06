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
else:
    selected = st.selectbox("Select US Stock", list(us_stocks.keys()))
    ticker = us_stocks[selected]