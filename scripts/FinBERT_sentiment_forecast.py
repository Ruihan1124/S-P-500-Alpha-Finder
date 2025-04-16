# ========== run_sentiment_pipeline.py ==========
"""
功能：
1️⃣ 输入股票代码和分析天数，抓取相关新闻
2️⃣ 使用 FinBERT 进行情绪分析
3️⃣ 生成并保存包含 weighted_score 的 CSV 文件
4️⃣ 可视化过去 N 天情绪趋势折线图
"""

import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm
from fetch_news.news_api import get_news

# 加载 FinBERT 模型
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

def fetch_news(ticker, days, source='finnhub'):
    if days > 7 and source == 'finnhub':
        print("⚠️ 免费版 Finnhub API 只能抓取过去 7 天的新闻")
        return None
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_path = os.path.join(project_root, "data", "raw")
    os.makedirs(save_path, exist_ok=True)
    df = get_news(ticker, days, source)
    if df is not None and not df.empty:
        output_file = f"{ticker}_news_{days}d_{source}.csv"
        df.to_csv(os.path.join(save_path, output_file), index=False)
        print(f"✅ News saved to {output_file}")
        return os.path.join(save_path, output_file)
    print("⚠️ No news found")
    return None

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[prediction.item()], round(confidence.item(), 4)

def plot_sentiment_trend(df, ticker):
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    trend = df.groupby('date')['weighted_score'].mean()
    plt.figure(figsize=(10, 5))
    trend.plot(kind='line', marker='o')
    plt.title(f"{ticker} - Past {len(trend)} Days Weighted Sentiment Trend")
    plt.ylabel("Weighted Sentiment Score")
    plt.xlabel("Date")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_sentiment(filepath, ticker, days, source):
    df = pd.read_csv(filepath)
    sentiments, confidences = [], []
    for text in tqdm(df['headline']):
        label, conf = classify_sentiment(str(text))
        sentiments.append(label)
        confidences.append(conf)

    df['sentiment'] = sentiments
    df['confidence'] = confidences
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['score'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

    source_weights = {
        'seekingalpha': 1.2, 'marketwatch': 1.0, 'bloomberg': 1.1, 'cnbc': 0.9, 'wsj': 1.2,
        'benzinga': 0.8, 'yahoo': 1.0, 'investorplace': 0.85, 'reuters': 1.1, 'fool': 0.95, 'default': 1.0
    }
    df['source_weight'] = df['source'].apply(lambda x: source_weights.get(str(x).lower(), 1.0))
    df['weighted_score'] = df['score'] * df['source_weight']

    processed_path = os.path.join(os.path.dirname(filepath), "..", "processed")
    os.makedirs(processed_path, exist_ok=True)
    output_file = os.path.join(processed_path, f"{ticker}_news_{days}d_{source}_sentiment.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Sentiment data saved to: {output_file}")
    plot_sentiment_trend(df, ticker)

def main():
    ticker = input("请输入股票代码（如 AAPL）: ").upper()
    days = int(input("请输入分析天数（建议 7）: "))
    source = 'finnhub'

    news_file = fetch_news(ticker, days, source)
    if news_file:
        analyze_sentiment(news_file, ticker, days, source)

if __name__ == "__main__":
    main()


# ========== stock_forecast_with_sentiment.py ==========
"""
功能：
1️⃣ 下载股票历史数据
2️⃣ 合并情绪得分
3️⃣ 使用 Prophet 进行预测并可视化
4️⃣ 可引入人为设定的特殊事件（如 Trump 发言）作为影响因子（后续扩展）
"""

import pandas as pd
import argparse
from prophet import Prophet
import matplotlib.pyplot as plt
import yfinance as yf
import os

def fetch_stock_data(ticker, period="5y"):
    df = yf.download(ticker, period=period)
    df = df[["Close"]].copy()
    df.columns = ["y"]
    df = df.reset_index().rename(columns={"Date": "ds"})
    df.dropna(inplace=True)
    return df

def merge_sentiment(df_stock, sentiment_path):
    df_sent = pd.read_csv(sentiment_path)
    df_sent['ds'] = pd.to_datetime(df_sent['date'])
    df_sent = df_sent.groupby('ds')['weighted_score'].mean().reset_index()
    df_sent.columns = ['ds', 'sentiment_score']
    df_merged = pd.merge(df_stock, df_sent, on='ds', how='left')
    df_merged['sentiment_score'].fillna(method='ffill', inplace=True)
    df_merged['sentiment_score'].fillna(0, inplace=True)
    return df_merged

def run_forecast(df, days, ticker):
    model = Prophet()
    model.add_regressor('sentiment_score')
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    future = pd.merge(future, df[['ds', 'sentiment_score']], on='ds', how='left')
    future['sentiment_score'].fillna(method='ffill', inplace=True)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.title(f"{ticker} Stock Forecast with Sentiment")
    plt.tight_layout()
    plt.show()


def main():
    ticker = input("请输入股票代码（如 AAPL）: ").upper()
    days = int(input("请输入预测天数（建议 7）: "))

    df_stock = fetch_stock_data(ticker)
    sentiment_path = f"data/processed/{ticker}_news_{days}d_finnhub_sentiment.csv"
    df_merged = merge_sentiment(df_stock, sentiment_path)
    run_forecast(df_merged, days, ticker)

if __name__ == "__main__":
    main()


# ========== main.py ==========
"""
功能：统一运行 情绪分析 + 股票预测 两个模块（基于用户手动输入）
"""

import subprocess

if __name__ == "__main__":
    ticker = input("请输入股票代码（如 AAPL）: ").upper()
    days = input("请输入分析天数（建议 7）: ")

    subprocess.run(["python", "run_sentiment_pipeline.py", "--ticker", ticker, "--days", days])
    subprocess.run(["python", "stock_forecast_with_sentiment.py", "--ticker", ticker, "--days", days])