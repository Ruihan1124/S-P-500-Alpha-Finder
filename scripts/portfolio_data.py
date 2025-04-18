# 获取"SPY": None, "BND": None, "GLD": None, 'DBC':None，数据，保存到csv里

import datetime
import requests
import pandas as pd

# import config  # 请确保 config.py 中包含 ALPHA_VANTAGE_API_KEY
ALPHA_VANTAGE_API_KEY = "5JCSBHR4UV9VS19K"


def get_asset_prices_alpha(ticker, days):
    """
    利用 Alpha Vantage API 获取指定 ticker 的历史日收盘价数据，
    并过滤出从 (今天 - days) 到今天的数据。
    """
    url = "https://www.alphavantage.co/query"
    # 当需要的数据天数较多时，使用 full 模式，否则 compact 模式即可
    outputsize = "full" if days > 100 else "compact"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": outputsize,
    }

    response = requests.get(url, params=params)
    data = response.json()

    # 判断数据是否正确返回
    if "Time Series (Daily)" not in data:
        print(f"无法获取 {ticker} 的数据，返回信息：", data)
        return None

    ts_data = data["Time Series (Daily)"]
    records = []
    for date_str, daily_data in ts_data.items():
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        records.append((date_obj, float(daily_data["4. close"])))
    # 生成 DataFrame 并按日期升序排列
    df = pd.DataFrame(records, columns=["date", "close"])
    df = df.sort_values("date").set_index("date")
    # 过滤出最近 days 天的数据
    start_date = datetime.datetime.now() - datetime.timedelta(days=days) - datetime.timedelta(days=365)
    # df = df[(df.index >= start_date)]
    df = df[(df.index >= start_date) & (df.index <= datetime.datetime.now() - datetime.timedelta(days=365))]
    return df


def fetch_price_data(days):
    """
    根据输入的天数获取各资产的历史收盘价数据。
    资产包括：SPY、BND、GLD，以及CASH（价格恒定为1）。
    返回一个 DataFrame，各列代表不同资产的收盘价。
    """
    assets = {"SPY": None, "BND": None, "GLD": None, 'DBC': None}
    data_dict = {}

    for ticker in assets:
        print(f"正在获取 {ticker} 的数据...")
        df = get_asset_prices_alpha(ticker, days)
        if df is not None:
            # 因为各标的交易日可能略有差异，这里取日期序列交集：以 SPY 为基准（如果SPY获取失败，则以任一成功标的）
            data_dict[ticker] = df["close"]

    # 为 CASH 生成价格数据。若 SPY 数据可用，则采用其日期；否则任选一标的的日期
    if "SPY" in data_dict and data_dict["SPY"] is not None:
        dates = data_dict["SPY"].index
    else:
        dates = list(data_dict.values())[0].index if data_dict else pd.date_range(end=datetime.datetime.now(),
                                                                                  periods=days)
    data_dict["CASH"] = pd.Series([1] * len(dates), index=dates)

    # 合并数据，取各资产日期的交集
    price_df = pd.DataFrame(data_dict)
    price_df = price_df.dropna()
    return price_df


def main():
    days = 1090


    print("正在获取历史价格数据，请稍候...")
    price_df = fetch_price_data(days)

    if price_df.empty:
        print("获取的数据为空，请检查 API key 或网络连接。")
        return

    print("\n获取到的价格数据（最近几天）：")
    print(price_df.tail())

    # 可选择保存数据到 CSV 文件，后续组合优化部分可直接读取该文件
    output_file = "prices.csv"
    price_df.to_csv(output_file)
    print(f"\n数据已保存到 {output_file}")


if __name__ == "__main__":
    main()
