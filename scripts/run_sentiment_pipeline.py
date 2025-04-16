"""
本脚本用于实现以下功能：

1️⃣ 自动抓取某只股票在过去 N 天内的相关新闻（支持来源：Finnhub 等）；
2️⃣ 利用 FinBERT 模型对每条新闻标题进行情绪分析，判断为 positive / neutral / negative，并计算置信度；
3️⃣ 根据不同新闻来源的可信度设定“加权值”，生成加权情绪得分（weighted_score）；
4️⃣ 可视化每日平均加权情绪得分，用于观察市场情绪趋势。

📌 新闻来源权重设定说明（source_weights）：
    为了让情绪分析结果更贴近市场实际情况，我们为不同的新闻来源赋予了不同的权重，依据如下：
    - 🧠 更专业的金融新闻媒体（如 WSJ, Bloomberg, SeekingAlpha）赋予更高权重（1.1～1.2）；
    - 📣 通用新闻网站或自动聚合类媒体（如 Yahoo, Fool）赋予中等权重（约 1.0）；
    - 📱 内容较短、简略或偏标题党的媒体（如 Benzinga）赋予较低权重（约 0.8）。

    权重范围大致为 0.8 到 1.2，用于调节情绪打分的影响力，增强重要来源的分析权重。

使用方式：
运行脚本后按提示输入股票代码（如 AAPL）和分析天数（建议不超过 7 天），程序会自动执行分析流程。

Note: 免费版 Finnhub API 仅支持抓取过去 7 天内的新闻数据。
"""

import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm
from fetch_news.news_api import get_news

# 加载 FinBERT 模型
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

# ========== 步骤1：抓取新闻 ==========
def fetch_news(ticker, days, source='finnhub'):
    if days > 7 and source == 'finnhub':
        print("⚠️ 免费版 Finnhub API 只能抓取过去 7 天的新闻，请调整天数或更换 API 来源")
        return None

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_path = os.path.join(project_root, "data", "raw")
    os.makedirs(save_path, exist_ok=True)

    try:
        df = get_news(ticker, days, source=source)
        if df is not None and not df.empty:
            output_file = f"{ticker}_news_{days}d_{source}.csv"
            df.to_csv(os.path.join(save_path, output_file), index=False)
            print(f"✅ News for {ticker} saved to {output_file}")
            return os.path.join(save_path, output_file)
        else:
            print(f"⚠️ No news found for {ticker}")
            return None
    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return None

# ========== 步骤2：情绪分析 ==========
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[prediction.item()], round(confidence.item(), 4)

def analyze_sentiment(filepath, ticker, days, source):
    df = pd.read_csv(filepath)
    if 'headline' not in df.columns:
        raise ValueError("Missing 'headline' column in news file")

    sentiments = []
    confidences = []
    print("🔍 Running FinBERT sentiment analysis...")

    for text in tqdm(df['headline']):
        label, conf = classify_sentiment(str(text))
        sentiments.append(label)
        confidences.append(conf)

    df['sentiment'] = sentiments
    df['confidence'] = confidences
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['score'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

    # 加权设置
    source_weights = {
        'seekingalpha': 1.2,
        'marketwatch': 1.0,
        'bloomberg': 1.1,
        'cnbc': 0.9,
        'wsj': 1.2,
        'benzinga': 0.8,
        'yahoo': 1.0,
        'investorplace': 0.85,
        'reuters': 1.1,
        'fool': 0.95,
        'default': 1.0
    }

    def get_weight(source_name):
        return source_weights.get(str(source_name).lower(), source_weights['default'])

    df['source_weight'] = df['source'].apply(get_weight)
    df['weighted_score'] = df['score'] * df['source_weight']

    # 保存处理结果
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_path = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_path, exist_ok=True)
    output_file = os.path.join(processed_path, f"{ticker}_news_{days}d_{source}_sentiment.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Processed data saved to: {output_file}")

    return df

# ========== 步骤3：画折线图 ==========
def plot_sentiment_trend(df, ticker):
    if 'weighted_score' not in df.columns or df['weighted_score'].dropna().empty:
        print("⚠️ 加权得分为空，请检查 weighted_score 是否正确计算")
        return

    daily_weighted_avg = df.groupby('date')['weighted_score'].mean()

    if daily_weighted_avg.empty:
        print("⚠️ 无有效的加权数据用于绘图")
        return

    plt.figure(figsize=(10, 5))
    daily_weighted_avg.plot(kind='line', marker='o')
    plt.title(f"{ticker} - Daily Weighted Sentiment Score")
    plt.ylabel("Weighted Sentiment Score (-1 to 1)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ========== 主流程 ==========
def run_pipeline():
    ticker = input("请输入股票代码 (如 AAPL): ").upper()
    if not ticker.isalpha() or len(ticker) > 5:
        print("❌ 股票代码格式错误")
        return

    try:
        days = int(input("请输入分析天数 (建议填 7): "))
    except:
        print("❌ 天数输入错误")
        return

    source = 'finnhub'
    news_file = fetch_news(ticker, days, source)
    if news_file:
        df = analyze_sentiment(news_file, ticker, days, source)
        plot_sentiment_trend(df, ticker)

if __name__ == "__main__":
    run_pipeline()
