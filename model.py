import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def fetch_stock_data(ticker, period="10y"):
    import yfinance as yf
    yf.set_tz_cache_location("custom_cache_dir")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

def get_company_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
    }

def add_trend_label(df):
    df["Return"] = df["Close"].pct_change() * 100
    def label(ret):
        if ret > 1:
            return "Up"
        elif ret < -1:
            return "Down"
        else:
            return "Sideways"
    df["Trend"] = df["Return"].apply(label)
    df.dropna(inplace=True)
    return df

def add_features(df):
    df["MA7"]           = df["Close"].rolling(window=7).mean()
    df["MA21"]          = df["Close"].rolling(window=21).mean()
    df["Momentum"]      = df["Close"] - df["Close"].shift(7)
    df["Volatility"]    = df["Close"].rolling(window=7).std()
    df["Volume_Change"] = df["Volume"].pct_change() * 100

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.replace([float('inf'), float('-inf')], 0, inplace=True)
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ["MA7", "MA21", "Momentum", "Volatility", "Volume_Change", "RSI"]
    X = df[features]
    y = df["Trend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def predict_future_prices(df, days=30):
    df = df.copy()
    df["Days"] = np.arange(len(df))

    X = df[["Days"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    # Last 30 days real vs predicted
    last_30 = df.tail(30)
    real_prices = last_30["Close"].values
    predicted_prices = model.predict(last_30[["Days"]])

    # Next 7 days forecast
    last_day = df["Days"].iloc[-1]
    future_days = np.arange(last_day + 1, last_day + 8).reshape(-1, 1)
    future_prices = model.predict(future_days)

    return real_prices, predicted_prices, future_prices, last_30.index

if __name__ == "__main__":
    df = fetch_stock_data("RELIANCE.NS")
    df = add_trend_label(df)
    df = add_features(df)
    train_model(df)