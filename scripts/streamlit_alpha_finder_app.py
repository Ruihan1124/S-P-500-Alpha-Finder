import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from FinBERT_sentiment_forecast import (
    fetch_stock_data, fetch_and_analyze_sentiment, prepare_sentiment_daily,
    merge_data, forecast_with_regressors, plot_forecast, save_forecast_to_csv,
    plot_sentiment_trend, GEMINI_API_KEY, GEMINI_URL
)
from portfolio_data import fetch_price_data
from Asset_allocation_gpt import generate_random_portfolios, summarize_top_n_portfolios
from ticker_resolver import get_sp500_tickers

st.set_page_config(page_title="S&P 500 Alpha Finder & Optimizer", layout="wide")

st.title("📈 S&P 500 Alpha Finder & Portfolio Optimizer")
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.radio("Choose Module:", ["📊 Stock Analysis", "💼 Portfolio Optimization"])

# ============================ Module 1: Stock Analysis ============================
if page == "📊 Stock Analysis":
    st.header("📊 Stock Sentiment & Forecast Analysis")

    user_input = st.text_input("Enter Company Name or Ticker (e.g., Apple or AAPL):", "AAPL")
    sp500_dict = get_sp500_tickers()
    matches = {name: symbol for name, symbol in sp500_dict.items() if user_input.lower() in name.lower() or user_input.upper() == symbol.upper()}

    if len(matches) == 1:
        ticker = list(matches.values())[0]
        st.success(f"✅ Found: {list(matches.keys())[0]} ({ticker})")
    elif len(matches) > 1:
        selected = st.selectbox("Multiple matches found. Please choose:", [f"{name} ({symbol})" for name, symbol in matches.items()])
        ticker = selected.split('(')[-1].replace(')', '')
    else:
        ticker = user_input.upper()
        st.warning("⚠️ No company match found. Will use raw input as ticker.")

    run_stock = st.button("Run Stock Forecast")

    if run_stock:
        try:
            days = 7
            price_df = fetch_stock_data(ticker)
            sentiment_raw = fetch_and_analyze_sentiment(ticker, days)
            sentiment_df = prepare_sentiment_daily(sentiment_raw)
            merged_df = merge_data(price_df, sentiment_df)
            model, forecast = forecast_with_regressors(merged_df, days)

            plot_sentiment_trend(sentiment_raw, ticker)
            plot_forecast(model, forecast, ticker)
            save_forecast_to_csv(forecast, ticker)

            st.session_state["forecast_result"] = forecast
            st.session_state["sentiment_raw"] = sentiment_raw
            st.session_state["ticker"] = ticker

            st.success("✅ Forecast complete. You can now ask questions below!")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    if "forecast_result" in st.session_state and "sentiment_raw" in st.session_state:
        ticker = st.session_state["ticker"]
        st.image(f"data/plots/{ticker}_sentiment_trend.png", caption="📉 Sentiment Trend", use_container_width=True)
        st.image(f"data/processed/{ticker}_combined_forecast_plot.png", caption="📈 Prophet Forecast", use_container_width=True)

        st.subheader("🧠 Ask AI about Forecast + Sentiment")
        user_q = st.text_input("Your question:", key="user_question")
        ask = st.button("Ask AI")

        if ask and user_q:
            forecast = st.session_state["forecast_result"]
            sentiment_raw = st.session_state["sentiment_raw"]

            summary_price = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_string(index=False)
            summary_sentiment = sentiment_raw.groupby('date')['weighted_score'].mean().reset_index()
            sentiment_lines = "\n".join(
                [f"{row['date']}: score = {row['weighted_score']:.3f}" for _, row in summary_sentiment.iterrows()]
            )
            prompt = (
                "以下是某支股票的分析结果，包括两部分内容：\n"
                "\n📉 市场情绪分析（weighted sentiment score，近几日）：\n"
                f"{sentiment_lines}\n"
                "\n📈 股价预测（Prophet 模型，未来几日）：\n"
                f"{summary_price}\n"
                f"\n请基于这两个部分回答用户的问题：{user_q}"
            )

            response = requests.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]}
            )
            if response.status_code == 200:
                reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(f"**AI Response:** {reply}")
            else:
                st.error("❌ Gemini API Error")

# ============================ Module 2: Portfolio Optimization ============================
elif page == "💼 Portfolio Optimization":
    st.header("💼 Portfolio Optimization with AI")

    days = st.slider("Lookback Days (max ~365):", 100, 400, 252)
    max_dd = st.slider("Max Acceptable Drawdown (%):", 5, 50, 20)
    capital = st.number_input("Total Investment ($):", min_value=1000, value=10000)
    run_portfolio = st.button("Generate Optimized Portfolios")

    if run_portfolio:
        try:
            st.info("Fetching price data from Alpha Vantage...")
            prices_df = fetch_price_data(days)
            prices_df.to_csv("prices.csv")

            st.success("Generating portfolios...")
            results_df, weights_list = generate_random_portfolios(prices_df, num_portfolios=3000)

            st.subheader("Efficient Frontier")
            fig, ax = plt.subplots()
            sc = ax.scatter(
                results_df['Volatility'],
                results_df['Annual Return'],
                c=results_df['Sharpe Ratio'],
                cmap='viridis', alpha=0.6
            )
            ax.set_xlabel("Volatility")
            ax.set_ylabel("Annual Return")
            ax.set_title("Efficient Frontier")
            st.pyplot(fig)

            threshold = -max_dd / 100
            valid_idx = results_df[results_df['Max Drawdown'] >= threshold].index
            valid_df = results_df.loc[valid_idx].sort_values("Sharpe Ratio", ascending=False)

            if valid_df.empty:
                st.warning("No portfolios match the drawdown limit. Try increasing tolerance.")
            else:
                top5 = valid_df.head(5)
                st.subheader("Top 5 Portfolios by Sharpe Ratio")
                top5['Sharpe Ratio'] = top5['Sharpe Ratio'].round(2)
                st.dataframe(top5)

                summary = summarize_top_n_portfolios(prices_df, top5, [weights_list[i] for i in top5.index])

                st.subheader("🧠 Ask AI about Portfolio Suggestions")
                question = st.text_input("Your question about the portfolio:", key="portfolio_q")
                if st.button("Ask AI (Portfolio)"):
                    prompt = f"以下是五个候选投资组合的信息，请基于此回答我的问题：\n\n{summary}\n\n问题：{question}"
                    response = requests.post(
                        GEMINI_URL,
                        params={"key": GEMINI_API_KEY},
                        headers={"Content-Type": "application/json"},
                        json={"contents": [{"parts": [{"text": prompt}]}]}
                    )
                    if response.status_code == 200:
                        reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                        st.markdown(f"**AI Response:** {reply}")
                    else:
                        st.error("❌ Gemini API Error")

        except Exception as e:
            st.error(f"❌ Error: {e}")
