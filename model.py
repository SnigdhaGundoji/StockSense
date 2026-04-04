import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def fetch_stock_data(ticker, period="1y"):
    import yfinance as yf
    yf.set_tz_cache_location("custom_cache_dir")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

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
    df.replace([float('inf'), float('-inf')], 0, inplace=True)
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ["MA7", "MA21", "Momentum", "Volatility", "Volume_Change"]
    X = df[features]
    y = df["Trend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("✅ Model trained successfully!\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/stock_model.pkl")
    print("✅ Model saved to models/stock_model.pkl")

    return model

if __name__ == "__main__":
    df = fetch_stock_data("RELIANCE.NS")
    df = add_trend_label(df)
    df = add_features(df)
    train_model(df)