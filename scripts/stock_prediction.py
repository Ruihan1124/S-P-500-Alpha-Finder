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

# CLI 模式支持（同时兼容 Jupyter 和命令行）
def main():
    parser = argparse.ArgumentParser(description="Prophet Stock predictor")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock code")
    parser.add_argument("days", nargs="?", type=int, default=30, help="Forecast days")
    args = parser.parse_args()
    run_stock_prediction(args.ticker, args.days)

if __name__ == "__main__":
    if not any('ipykernel_launcher' in arg or 'jupyter' in arg for arg in sys.argv):
        main()

# 运行主函数（AAPL为例，其他股票只需更改股票代码即可）（不会自动触发 main()，要手动）
run_stock_prediction("AAPL", 30)

# 显示图像（可选）
from IPython.display import Image, display
display(Image(filename="data/processed/AAPL_forecast_plot.png"))

# 查看保存内容
import pandas as pd
pd.read_csv("data/processed/AAPL_forecast_data.csv").tail()
