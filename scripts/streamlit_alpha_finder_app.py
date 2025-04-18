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
# from portfolio_data import fetch_price_data
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
                "The following is an analysis of a specific stock, consisting of two parts:\n"
                "\n📉 Market sentiment analysis (weighted sentiment score, recent days):\n"
                f"{sentiment_lines}\n"
                "\n📈 Stock price forecast (Prophet model, next few days):\n"
                f"{summary_price}\n"
                f"\nPlease answer the user's question based on the two parts above: {user_q}"
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

    max_dd = st.slider("Max Acceptable Drawdown (%):", 5, 50, 20)
    capital = st.number_input("Total Investment ($):", min_value=1000, value=10000)

    if 'run_optimization' not in st.session_state:
        st.session_state.run_optimization = False

    if st.button("Generate Optimized Portfolios"):
        try:
            st.session_state.run_optimization = True
            st.info("Loading price data from prices.csv...")
            prices_df = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
            st.session_state.prices_df = prices_df

            st.success("Generating portfolios...")
            results_df, weights_list = generate_random_portfolios(prices_df, num_portfolios=3000)
            st.session_state.results_df = results_df
            st.session_state.weights_list = weights_list

            threshold = -max_dd / 100
            valid_idx = results_df[results_df['Max Drawdown'] >= threshold].index
            valid_df = results_df.loc[valid_idx].sort_values("Sharpe Ratio", ascending=False)

            if valid_df.empty:
                st.warning("No portfolios match the drawdown limit. Try increasing tolerance.")
                st.session_state.run_optimization = False
            else:
                top5 = valid_df.head(5).copy()
                top5['Sharpe Ratio'] = top5['Sharpe Ratio'].round(2)

                comp_list = []
                for idx in top5.index:
                    w = weights_list[idx]
                    comp_str = " | ".join(f"{col}={w_val * 100:.2f}%" for col, w_val in zip(prices_df.columns, w))
                    comp_list.append(comp_str)
                top5["Composition"] = comp_list

                st.session_state.top5 = top5
                st.session_state.valid_df = valid_df

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.run_optimization = False

    if st.session_state.run_optimization:
        prices_df = st.session_state.prices_df
        weights_list = st.session_state.weights_list
        top5 = st.session_state.top5

        st.subheader("Top 5 Portfolios by Sharpe Ratio + Allocation")
        st.dataframe(top5, use_container_width=True)

        best_idx = top5.index[0]
        best_portfolio = top5.loc[best_idx]
        best_weights = weights_list[best_idx]

        st.subheader("🌟 Recommended Portfolio (Highest Sharpe Ratio)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Annual Return:** {best_portfolio['Annual Return'] * 100:.2f}%")
            st.markdown(f"**Volatility:** {best_portfolio['Volatility'] * 100:.2f}%")
            st.markdown(f"**Sharpe Ratio:** {best_portfolio['Sharpe Ratio']:.2f}")
            st.markdown(f"**Max Drawdown:** {best_portfolio['Max Drawdown'] * 100:.2f}%")
        with col2:
            potential_loss = abs(best_portfolio["Max Drawdown"]) * capital
            st.markdown(f"**Capital:** ${capital:,.2f}")
            st.markdown(f"**Est. Max Loss:** ${potential_loss:,.2f}")

        st.markdown("**Asset Allocation:**")
        for asset, w in zip(prices_df.columns, best_weights):
            st.markdown(f"- {asset}: {w * 100:.2f}%")

        st.subheader("📈 Cumulative Return of Recommended Portfolio")
        port_val = (prices_df * best_weights).sum(axis=1)
        cumulative = port_val / port_val.iloc[0]
        fig2, ax2 = plt.subplots()
        ax2.plot(cumulative.index, cumulative.values)
        ax2.set_title("Cumulative Return")
        ax2.set_ylabel("Normalized Value")
        ax2.set_xlabel("Date")
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("🔍 Select Your Favorite Portfolio (1–5)")
        option = st.selectbox("Choose one portfolio to compare with the system recommendation:", [1, 2, 3, 4, 5])

        chosen_idx = top5.index[option - 1]
        chosen_portfolio = top5.loc[chosen_idx]
        chosen_weights = weights_list[chosen_idx]

        st.markdown(f"### 🧩 You Selected: Portfolio {option}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Annual Return:** {chosen_portfolio['Annual Return'] * 100:.2f}%")
            st.markdown(f"**Volatility:** {chosen_portfolio['Volatility'] * 100:.2f}%")
            st.markdown(f"**Sharpe Ratio:** {chosen_portfolio['Sharpe Ratio']:.2f}")
            st.markdown(f"**Max Drawdown:** {chosen_portfolio['Max Drawdown'] * 100:.2f}%")
        with col2:
            potential_loss2 = abs(chosen_portfolio["Max Drawdown"]) * capital
            st.markdown(f"**Est. Max Loss:** ${potential_loss2:,.2f}")

        st.markdown("**Your Portfolio Allocation:**")
        for asset, w in zip(prices_df.columns, chosen_weights):
            st.markdown(f"- {asset}: {w * 100:.2f}%")

        st.subheader("📉 Comparison of Recommended vs Your Selected Portfolio")
        cum_best = (prices_df * best_weights).sum(axis=1) / (prices_df * best_weights).sum(axis=1).iloc[0]
        cum_user = (prices_df * chosen_weights).sum(axis=1) / (prices_df * chosen_weights).sum(axis=1).iloc[0]
        fig3, ax3 = plt.subplots()
        ax3.plot(cum_best.index, cum_best.values, label="Recommended", linewidth=2)
        ax3.plot(cum_user.index, cum_user.values, label=f"Portfolio {option}", linestyle='--', linewidth=2)
        ax3.set_title("Cumulative Return Comparison")
        ax3.set_ylabel("Normalized Value")
        ax3.set_xlabel("Date")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

        diff_data = {
            "指标": ["Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
            "Recommended": [
                f"{best_portfolio['Annual Return'] * 100:.2f}%",
                f"{best_portfolio['Volatility'] * 100:.2f}%",
                f"{best_portfolio['Sharpe Ratio']:.2f}",
                f"{best_portfolio['Max Drawdown'] * 100:.2f}%"
            ],
            f"Portfolio {option}": [
                f"{chosen_portfolio['Annual Return'] * 100:.2f}%",
                f"{chosen_portfolio['Volatility'] * 100:.2f}%",
                f"{chosen_portfolio['Sharpe Ratio']:.2f}",
                f"{chosen_portfolio['Max Drawdown'] * 100:.2f}%"
            ],
            "Difference": [
                f"{(best_portfolio['Annual Return'] - chosen_portfolio['Annual Return']) * 100:.2f}%",
                f"{(best_portfolio['Volatility'] - chosen_portfolio['Volatility']) * 100:.2f}%",
                f"{best_portfolio['Sharpe Ratio'] - chosen_portfolio['Sharpe Ratio']:.2f}",
                f"{(best_portfolio['Max Drawdown'] - chosen_portfolio['Max Drawdown']) * 100:.2f}%"
            ]
        }
        st.subheader("📋 Performance Comparison Table")
        st.dataframe(pd.DataFrame(diff_data), use_container_width=True)

        # === AI 问答区 ===
        top5_weights = [weights_list[i] for i in top5.index]
        summary = summarize_top_n_portfolios(prices_df, top5, top5_weights)

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
                try:
                    reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                    st.markdown(f"**AI Response:** {reply}")
                except Exception as e:
                    st.error(f"Response parsing error: {e}")
            else:
                st.error("❌ Gemini API Error")
