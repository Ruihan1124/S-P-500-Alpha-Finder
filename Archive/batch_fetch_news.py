'''
功能：批量抓取多只股票的新闻数据并保存为 CSV 文件
说明：可选择使用 Finnhub 或 Alpha Vantage 为数据源。
默认开发阶段使用 Finnhub,抓取30支股票过去15天的数据。
可设置抓取天数、输出路径、新闻来源等。
'''

from fetch_news.news_api import get_news
from utils import get_sp500_tickers
import os
import pandas as pd
import time

def batch_fetch_all_news(days=15, source='finnhub'):
    tickers = ['AAPL']
    # tickers = get_sp500_tickers()
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
    os.makedirs(save_path, exist_ok=True)

    for ticker in tickers[:50]:  # 安全测试
        try:
            df = get_news(ticker, days, source=source)
            if df is not None and not df.empty:
                output_file = f"{ticker}_news_{days}d_{source}.csv"
                df.to_csv(os.path.join(save_path, output_file), index=False)
                print(f" {ticker} saved as {output_file}")
            else:
                print(f" No news for {ticker}")
            time.sleep(1.2)
        except Exception as e:
            print(f" Error processing {ticker}: {e}")

if __name__ == "__main__":
    batch_fetch_all_news(days=15, source='finnhub')

