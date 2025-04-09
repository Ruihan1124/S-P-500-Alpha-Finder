'''
功能：使用 Alpha Vantage API 抓取市场新闻数据（作为备用来源）
说明：Alpha Vantage 新闻为全市场新闻流，无法精确针对 ticker，因此推荐作为补充或回退方案使用
'''

import requests
import pandas as pd
import os
from config import ALPHA_VANTAGE_API_KEY


def fetch_news(stock_symbol, days):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching news:", response.text)
        return None

    news_data = response.json().get("feed", [])
    print(f"API returned {len(news_data)} articles")  # print the number of articles returned

    df = pd.DataFrame(news_data)

    if not df.empty:
        df = df[['time_published', 'title', 'summary', 'url', 'source']]  # make sure to keep only these columns
        df.rename(columns={'time_published': 'datetime', 'title': 'headline'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%dT%H%M%S')

        # filter 'Motley Fool'
        df = df[df['source'] != 'Motley Fool']

        # filter by days
        start_date = get_past_date(days)
        df = df[df['datetime'] >= start_date]

    return df


def get_today():
    return pd.Timestamp.today().strftime('%Y-%m-%d')


def get_past_date(days):
    return pd.Timestamp.today() - pd.Timedelta(days=days)


if __name__ == "__main__":
    stock_symbol = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip().upper()
    days = int(input("Enter number of days for news (7, 15, 30): ").strip())

    if days not in [7, 15, 30]:
        print("Invalid input. Please enter 7, 15, or 30.")
    else:
        news_df = fetch_news(stock_symbol, days)
        if news_df is not None and not news_df.empty:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)
            print(news_df.head(10))

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # get the project root directory
            raw_data_path = os.path.join(project_root, "data/raw")
            os.makedirs(raw_data_path, exist_ok=True)

            # generate the output path
            output_path = os.path.join(raw_data_path, f"{stock_symbol}_news_{days}d.csv")
            print(f"Saving news data to: {output_path}")

            news_df.to_csv(output_path, index=False)

            # make sure the file was saved
            if os.path.exists(output_path):
                print(f"File successfully saved at {output_path}")
            else:
                print(f"ERROR: File {output_path} was NOT saved.")
        else:
            print("No news data available, skipping file save.")
