import requests
import pandas as pd
import os
from config import ALPHA_VANTAGE_API_KEY


def fetch_news(stock_symbol, days=7):
    """
    获取指定股票的相关新闻（最近 7/15/30 天）。
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching news:", response.text)
        return None

    news_data = response.json().get("feed", [])
    df = pd.DataFrame(news_data)

    if not df.empty:
        df = df[['time_published', 'title', 'summary', 'url', 'source']]  # 确保提取的是完整的新闻链接
        df.rename(columns={'time_published': 'datetime', 'title': 'headline'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%dT%H%M%S')

    return df


def get_today():
    return pd.Timestamp.today().strftime('%Y-%m-%d')


def get_past_date(days):
    return (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')


if __name__ == "__main__":
    stock_symbol = "AAPL"  # 示例：获取苹果公司新闻
    news_df = fetch_news(stock_symbol, days=7)
    if news_df is not None:
        pd.set_option("display.max_columns", None)  # 显示所有列
        pd.set_option("display.width", 1000)  # 设置宽度，避免换行
        print(news_df.head(10))  # 打印前10行
        os.makedirs("data/raw", exist_ok=True)
        news_df.to_csv(f"data/raw/{stock_symbol}_news.csv", index=False)

