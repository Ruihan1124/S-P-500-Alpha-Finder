import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from fetch_news.news_api import get_news
from datetime import datetime
import requests

# ========== Gemini LLM 设置 ==========
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL = st.secrets["GEMINI_URL"]

def ask_gemini_about_forecast(summary_text):
    prompt = (
        "以下是某支股票未来几天的价格预测结果，包括预测值区间。\n"
        "请基于这些数据，生成简洁、清晰的趋势解读：\n\n"
        f"{summary_text}"
    )

    response = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )

    if response.status_code == 200:
        try:
            reply = response.json()['candidates'][0]['content']['parts'][0]['text']
            print("\n🧠 Gemini 生成解读：\n" + reply)
        except Exception as e:
            print("❌ 解析错误：", e)
    else:
        print("❌ 请求失败：", response.text)

# ========== Step 1: Load Price Data from Alpha Vantage ==========
def fetch_stock_data(ticker: str, period="5y") -> pd.DataFrame:
    api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]

    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",  # full returns many years; compact returns 100 days
        "apikey": api_key
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()

        if "Time Series (Daily)" not in data:
            st.error(f"Alpha Vantage API 返回错误：{data.get('Note') or data.get('Error Message')}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df[["5. adjusted close"]].rename(columns={"5. adjusted close": "y"})
        df = df.reset_index().rename(columns={"index": "ds"})
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df.dropna(subset=["y", "ds"], inplace=True)

        # 只保留近几年（例如 5 年）
        if period == "5y":
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
            df = df[df["ds"] >= cutoff]

        return df

    except Exception as e:
        st.error(f"❌ 获取股票数据失败：{e}")
        return pd.DataFrame()

# ========== Step 2: Load and Process Sentiment Data ==========
def load_sentiment_data(ticker):
    sentiment_file = f"data/processed/{ticker}_news_7d_finnhub_sentiment.csv"
    if not os.path.exists(sentiment_file):
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_file}")
    df = pd.read_csv(sentiment_file)
    df['ds'] = pd.to_datetime(df['date'])
    daily_sentiment = df.groupby('ds')['weighted_score'].mean().reset_index()
    daily_sentiment.rename(columns={'weighted_score': 'sentiment_score'}, inplace=True)
    return daily_sentiment

# ========== Step 3: Merge and Add President Variable ==========
def merge_data(price_df, sentiment_df):
    df = pd.merge(price_df, sentiment_df, on='ds', how='left')
    df['sentiment_score'] = df['sentiment_score'].fillna(method='ffill').fillna(0)
    df['president'] = df['ds'].apply(lambda x: 1 if x >= pd.to_datetime('2021-01-20') else 0)
    return df

# ========== Step 4: Train Prophet with Regressors ==========
def forecast_with_regressors(df, days):
    model = Prophet()
    model.add_regressor('sentiment_score')
    model.add_regressor('president')
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    last_sentiment = df['sentiment_score'].iloc[-1]
    future['sentiment_score'] = last_sentiment
    last_president = df['president'].iloc[-1]
    future['president'] = last_president

    forecast = model.predict(future)
    return model, forecast

# ========== Step 5: Save Results ==========
def plot_forecast(model, forecast, ticker):
    fig = model.plot(forecast)
    plt.title(f"{ticker} Stock Price Forecast (with Sentiment & President)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{ticker}_combined_forecast_plot.png")
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path

def save_forecast_to_csv(forecast, ticker):
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{ticker}_combined_forecast.csv")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_path, index=False)
    return csv_path

# ========== Step 6: Main Pipeline ==========
def run_combined_forecast(ticker='AAPL', forecast_days=30):
    print(f"📈 Running combined forecast for {ticker}...")
    price_df = fetch_stock_data(ticker)
    sentiment_df = load_sentiment_data(ticker)
    merged_df = merge_data(price_df, sentiment_df)
    model, forecast = forecast_with_regressors(merged_df, forecast_days)
    fig_path = plot_forecast(model, forecast, ticker)
    csv_path = save_forecast_to_csv(forecast, ticker)
    print(f"✅ Forecast plot saved: {fig_path}")
    print(f"✅ Forecast data saved: {csv_path}")

    # LLM 解读模块
    summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_string(index=False)
    ask_gemini_about_forecast(summary)

if __name__ == "__main__":
    run_combined_forecast()
