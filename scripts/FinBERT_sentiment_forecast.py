import os
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from fetch_news.news_api import get_news
from datetime import datetime
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm
from ticker_resolver import get_sp500_tickers, resolve_ticker_local

# ========== Gemini LLM 设置 ==========
GEMINI_API_KEY = 'Your API key'
GEMINI_URL = "Your URL"

def ask_gemini_combined(summary_price, summary_sentiment):
    prompt = (
        "The following is the analysis result of a certain stock, including two parts:\n"
        "\n📉 Sentiment Analysis（weighted sentiment score, recent days）：\n"
        f"{summary_sentiment}\n"
        "\n📈 Stock price forecast (Prophet model, next few days):\n"
        f"{summary_price}\n"
        "\nPlease generate an overall trend interpretation based on these two parts and answer the questions raised by users."
    )

    print("\nYou can now ask questions about [Forecast + Sentiment] (type 'exit' to exit):")
    while True:
        user_input = input("You:")
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("The conversation ends.")
            break

        user_prompt = prompt + f"\n\nUser question:{user_input}"

        response = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": user_prompt}]}]}
        )

        if response.status_code == 200:
            try:
                reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                print("AI：" + reply)
            except Exception as e:
                print("❌ 解析错误：", e)
        else:
            print("❌ 请求失败：", response.text)

# ========== FinBERT 情绪分析模块 ==========
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[prediction.item()], round(confidence.item(), 4)

def fetch_and_analyze_sentiment(ticker, days=7, source='finnhub'):
    save_path = os.path.join("data", "processed")
    os.makedirs(save_path, exist_ok=True)
    df = get_news(ticker, days, source=source)
    if df is None or df.empty or 'headline' not in df.columns:
        raise ValueError("无法抓取新闻或缺少 headline 列")

    sentiments, confidences = [], []
    for text in tqdm(df['headline'], desc="情绪分析中"):
        label, conf = classify_sentiment(str(text))
        sentiments.append(label)
        confidences.append(conf)

    df['sentiment'] = sentiments
    df['confidence'] = confidences
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['score'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

    # 设定来源权重表
    source_weights = {
        'seekingalpha': 1.2, 'marketwatch': 1.0, 'bloomberg': 1.1, 'cnbc': 0.9,
        'wsj': 1.2, 'benzinga': 0.8, 'yahoo': 1.0, 'investorplace': 0.85,
        'reuters': 1.1, 'fool': 0.95, 'default': 1.0
    }

    # 获取最大权重（用于归一化）
    max_weight = max(source_weights.values())

    # 定义函数用于获取权重并归一化到 [0, 1]
    def get_weight(source_name):
        raw_weight = source_weights.get(str(source_name).lower(), source_weights['default'])
        return raw_weight / max_weight  # 归一化到 [0, 1]

    # 应用权重
    df['source_weight'] = df['source'].apply(get_weight)

    # 计算归一化的加权得分：-1 * weight ~ +1 * weight => 仍然在 [-1, 1]
    df['weighted_score'] = df['score'] * df['source_weight']
    return df

# ========== Stock Data ==========
def fetch_stock_data(ticker: str, period="5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period)
    df = df[["Close"]].copy()
    df.columns = ["y"]
    df = df.reset_index().rename(columns={"Date": "ds"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df.dropna(subset=["y", "ds"], inplace=True)
    return df

# ========== Daily Sentiment ==========
def prepare_sentiment_daily(sentiment_df):
    sentiment_df['ds'] = pd.to_datetime(sentiment_df['date'])
    daily_sentiment = sentiment_df.groupby('ds')['weighted_score'].mean().reset_index()
    daily_sentiment.rename(columns={'weighted_score': 'sentiment_score'}, inplace=True)
    return daily_sentiment

# ========== Sentiment Trend Plot ==========
def plot_sentiment_trend(sentiment_df, ticker):
    sentiment_df['ds'] = pd.to_datetime(sentiment_df['date'])
    daily_avg = sentiment_df.groupby('ds')['weighted_score'].mean()
    if daily_avg.empty:
        print("⚠️ 无有效情绪数据用于绘图")
        return

    plot_path = os.path.join("data", "plots")
    os.makedirs(plot_path, exist_ok=True)

    plt.figure(figsize=(10, 5))
    daily_avg.plot(kind='line', marker='o')
    plt.title(f"{ticker} - Daily Weighted Sentiment Score")
    plt.ylabel("Weighted Sentiment Score (-1 to 1)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"{ticker}_sentiment_trend.png"))
    plt.close()

# ========== Merge + President ==========
def merge_data(price_df, sentiment_df):
    df = pd.merge(price_df, sentiment_df, on='ds', how='left')
    df['sentiment_score'] = df['sentiment_score'].fillna(method='ffill').fillna(0)
    df['president'] = df['ds'].apply(lambda x: 1 if x >= pd.to_datetime('2021-01-20') else 0)
    return df

# ========== Forecast ==========
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

# ========== Save ==========
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

# ========== Run Main ==========
def run_combined_forecast(ticker):
    days = 7  # 固定为7天，因为免费版Finnhub只支持7天

    price_df = fetch_stock_data(ticker)
    sentiment_raw = fetch_and_analyze_sentiment(ticker, days)
    sentiment_df = prepare_sentiment_daily(sentiment_raw)
    merged_df = merge_data(price_df, sentiment_df)
    model, forecast = forecast_with_regressors(merged_df, days)

    # 📈 绘制情绪趋势图
    plot_sentiment_trend(sentiment_raw, ticker)

    fig_path = plot_forecast(model, forecast, ticker)
    csv_path = save_forecast_to_csv(forecast, ticker)
    print(f"✅ Forecast plot saved: {fig_path}")
    print(f"✅ Forecast data saved: {csv_path}")
    print(f"✅ Sentiment trend saved: data/plots/{ticker}_sentiment_trend.png")

    # 合并分析摘要并与 Gemini 交互
    summary_price = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_string(index=False)
    summary_sentiment = sentiment_raw.groupby('date')['weighted_score'].mean().reset_index()
    sentiment_lines = "\n".join(
        [f"{row['date']}: score = {row['weighted_score']:.3f}" for _, row in summary_sentiment.iterrows()]
    )

    ask_gemini_combined(summary_price, sentiment_lines)


if __name__ == "__main__":
    run_combined_forecast()