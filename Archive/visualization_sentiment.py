'''
功能：可视化指定股票的情绪趋势（基于已完成的 FinBERT 情绪打分结果）
输入：data/processed 中的情绪分析结果 CSV 文件（如 AAPL_news_15d_finnhub_sentiment.csv）
输出：折线图：每天的情绪数量变化 / 饼图：总情绪占比
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 可视化情绪趋势（折线图）
def plot_sentiment_trend(df, ticker):
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    trend = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 5))
    trend.plot(kind='line', marker='o')
    plt.title(f"{ticker} - Daily News Sentiment Trend")
    plt.ylabel("Number of Articles")
    plt.xlabel("Date")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Sentiment')
    plt.show()

# 可视化情绪占比（饼图）
def plot_sentiment_pie(df, ticker):
    counts = df['sentiment'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#99ff99','#ff9999'])
    plt.title(f"{ticker} - Sentiment Distribution")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# 主函数：加载指定文件并可视化
def visualize_sentiment(ticker='AAPL', days=7, source='finnhub'):
    # 获取路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(project_root, "data", "processed", f"{ticker}_news_{days}d_{source}_sentiment.csv")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    if df.empty:
        print("❌ CSV is empty.")
        return

    plot_sentiment_trend(df, ticker)
    plot_sentiment_pie(df, ticker)

if __name__ == "__main__":
    visualize_sentiment(ticker='AAPL', days=15, source='finnhub')
# create different charts to visualize the data
