import requests
import pandas as pd
import os
from config import NEWS_API_KEY


def fetch_news(stock_symbol, company_name, days=7):
    """
    使用 NewsAPI 获取指定公司的相关新闻。
    """
    url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching news:", response.text)
        return None

    news_data = response.json()

    if "articles" not in news_data or not news_data["articles"]:
        print("No news data available.")
        return None

    df = pd.DataFrame(news_data["articles"])

    if not df.empty:
        df = df[['publishedAt', 'title', 'description', 'url', 'source']]
        df.rename(columns={'publishedAt': 'datetime', 'title': 'headline', 'description': 'summary'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])  # 转换时间格式
        df['source'] = df['source'].apply(lambda x: x['name'] if isinstance(x, dict) else x)  # 提取来源

    return df


if __name__ == "__main__":
    stock_symbol = input("请输入股票代码（如 AAPL）：").strip().upper()
    company_name = input("请输入公司名称（如 Apple）：").strip()
    news_df = fetch_news(stock_symbol, company_name, days=7)
    if news_df is not None:
        pd.set_option("display.max_columns", None)  # 显示所有列
        pd.set_option("display.width", 1000)  # 避免换行
        print(news_df.head(10))  # 显示前10行
        os.makedirs("data/raw", exist_ok=True)
        csv_path = f"data/raw/{stock_symbol}_news.csv"
        news_df.to_csv(csv_path, index=False)
        print(f"CSV 文件已保存到: {os.path.abspath(csv_path)}")

