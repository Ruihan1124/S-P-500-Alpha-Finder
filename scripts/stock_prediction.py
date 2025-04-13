'''
功能：使用 Prophet 模型对历史股票价格数据进行未来 30 天的预测
输入：Yahoo Finance 抓取的历史收盘价数据
输出：绘制的预测图像（.png）与预测数据（.csv），保存在 data/processed 文件夹中
'''

import os
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

# 获取股票历史数据
def fetch_stock_data(ticker, period="5y"):
    df = yf.download(ticker, period=period)

    if df.empty or "Close" not in df.columns:
        raise ValueError(f"❌ 无法获取 {ticker} 的有效股价数据。")

    df = df.reset_index()[["Date", "Close"]].copy()
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

    # 转换为 float 并去除异常值
    df["y"] = pd.to_numeric(df["y"].astype(str), errors="coerce").astype(float)
    df.dropna(subset=["y", "ds"], inplace=True)

    if df.empty or df["y"].ndim != 1:
        raise TypeError("❌ y 列不是有效的 1 维 Series，请检查数据格式。")

    return df

# 执行 Prophet 模型预测
def forecast_stock_price(df, days=30):
    model = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.3,
        daily_seasonality=False,
        yearly_seasonality=True
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return model, forecast

# 绘图并保存图片
def plot_forecast(model, forecast, ticker):
    fig = model.plot(forecast)
    plt.title(f"{ticker} 股票价格预测", fontsize=14)
    plt.xlabel("日期")
    plt.ylabel("价格 (USD)")
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{ticker}_forecast_plot.png")
    fig.savefig(fig_path)
    print(f"✅ 图像已保存：{fig_path}")

# 保存预测结果为 CSV
def save_forecast_to_csv(forecast, ticker):
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{ticker}_forecast_data.csv")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_path, index=False)
    print(f"✅ 数据已保存：{csv_path}")

# 主执行函数
def run_stock_prediction():
    ticker = "AAPL"
    days = 30
    print(f"📈 正在处理 {ticker} 未来 {days} 天的股价预测...")

    df = fetch_stock_data(ticker)
    model, forecast = forecast_stock_price(df, days)
    plot_forecast(model, forecast, ticker)
    save_forecast_to_csv(forecast, ticker)

# 命令行执行
if __name__ == "__main__":
    run_stock_prediction()
