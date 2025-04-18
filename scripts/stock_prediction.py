'''
功能：使用 Prophet 模型对抓取的股票历史数据进行未来 N 天的价格预测
输入：股票代码（Ticker，默认 AAPL）、预测天数（默认 30）
输出：预测图表（PNG）与预测数据（CSV），保存在 data/processed 目录
'''

import os
import sys
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import argparse

# 获取股票历史数据
def fetch_stock_data(ticker: str, period="5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df = df[[("Close", ticker)]].copy()
        df.columns = ["y"]
    else:
        df = df[["Close"]].copy()
        df.columns = ["y"]
    df = df.reset_index().rename(columns={"Date": "ds"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df.dropna(subset=["y", "ds"], inplace=True)
    return df

# 使用 Prophet 模型进行股价预测
def forecast_stock_price(df: pd.DataFrame, days: int):
    model = Prophet(seasonality_mode="additive",
                    changepoint_prior_scale=0.3,
                    daily_seasonality=False,
                    yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return model, forecast

# 生成并保存预测图表
def plot_forecast(model, forecast, ticker):
    fig = model.plot(forecast)
    plt.title(f"{ticker} Stock price forecast", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{ticker}_forecast_plot.png")
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path

# 保存预测结果为 CSV 文件
def save_forecast_to_csv(forecast, ticker):
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{ticker}_forecast_data.csv")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_path, index=False)
    return csv_path

# 运行主预测任务
def run_stock_prediction(ticker="AAPL", days=30):
    df = fetch_stock_data(ticker)
    model, forecast = forecast_stock_price(df, days)
    fig_path = plot_forecast(model, forecast, ticker)
    csv_path = save_forecast_to_csv(forecast, ticker)
    print(f"✅ saved in:: {fig_path}")
    print(f"✅ saved in:: {csv_path}")



from ticker_resolver import get_sp500_tickers, resolve_ticker_local

# CLI 模式支持（同时兼容 Jupyter 和命令行）
import argparse

def main():
    parser = argparse.ArgumentParser(description="📈 Prophet Stock Predictor")
    parser.add_argument("ticker", nargs="?", help="股票代码或公司名称（如 AAPL 或 Apple）")
    parser.add_argument("days", nargs="?", type=int, default=30, help="预测天数（默认30天）")
    args = parser.parse_args()

    # 加载 S&P500 数据（公司名 -> 股票代码）
    sp500_dict = get_sp500_tickers()
    if not sp500_dict:
        print("❌ 无法加载 S&P500 公司列表，请检查网络连接或重试。")
        return

    if args.ticker:
        # 用户传入公司名或代码
        keyword = args.ticker.strip().lower()
        matches = {name: symbol for name, symbol in sp500_dict.items() if keyword in name.lower() or keyword == symbol.lower()}
        if not matches:
            print("❌ 未找到对应的公司名称或股票代码，请检查拼写")
            return
        elif len(matches) == 1:
            name, resolved_ticker = list(matches.items())[0]
            print(f"✅ 找到：{name}（{resolved_ticker}）")
        else:
            print("🔍 找到多个匹配项，请选择：")
            for i, (name, symbol) in enumerate(matches.items()):
                print(f"{i+1}. {name} ({symbol})")
            try:
                choice = int(input("请输入对应序号："))
                if 1 <= choice <= len(matches):
                    resolved_ticker = list(matches.values())[choice - 1]
                else:
                    print("❌ 无效选择。")
                    return
            except:
                print("❌ 输入无效。")
                return
        run_stock_prediction(resolved_ticker, args.days)
    else:
        # 没传入公司名，进入交互模式
        resolved_ticker = resolve_ticker_local(sp500_dict)
        run_stock_prediction(resolved_ticker, args.days)


if __name__ == "__main__":
    if not any('ipykernel_launcher' in arg or 'jupyter' in arg for arg in sys.argv):
        main()



# 显示图像（可选）
from IPython.display import Image, display
display(Image(filename="data/processed/AAPL_forecast_plot.png"))

# 查看保存内容
import pandas as pd
pd.read_csv("data/processed/AAPL_forecast_data.csv").tail()

