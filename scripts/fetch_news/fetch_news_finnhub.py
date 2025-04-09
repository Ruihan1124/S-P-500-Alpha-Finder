'''
功能：使用 Finnhub API 抓取指定股票的相关新闻数据（推荐主力使用）
说明：支持输入股票代码（如 AAPL），自动抓取对应公司的新闻，返回 DataFrame
'''

import requests
import pandas as pd
import os
from config import FINNHUB_API_KEY


def fetch_news_finnhub(stock_symbol, days):
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

    url = f"https://finnhub.io/api/v1/company-news?symbol={stock_symbol}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching news for {stock_symbol}: {response.text}")
        return None

    news_data = response.json()
    print(f"{stock_symbol}: returned {len(news_data)} articles")

    df = pd.DataFrame(news_data)

    if not df.empty:
        df = df[['datetime', 'headline', 'summary', 'url', 'source']]
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')  # unix timestamp

    return df


if __name__ == "__main__":
    symbol = input("Enter stock ticker: ").strip().upper()
    days = int(input("Enter number of days (7/15/30): ").strip())

    news_df = fetch_news_finnhub(symbol, days)
    if news_df is not None and not news_df.empty:
        print(news_df.head())
        # 保存逻辑同原先一致
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        raw_data_path = os.path.join(project_root, "data/raw")
        os.makedirs(raw_data_path, exist_ok=True)
        output_path = os.path.join(raw_data_path, f"{symbol}_news_{days}d_finnhub.csv")
        news_df.to_csv(output_path, index=False)
        print(f"✅ Saved to {output_path}")
    else:
        print("No news found.")
