import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scripts.fetch_news.fetch_news_finnhub import fetch_news_finnhub
from Archive.sentiment_analysis import analyze_sentiment_for_file
from Archive.utils import get_sp500_tickers

st.set_page_config(page_title="S&P 500 Alpha Finder & Portfolio Optimizer", layout="wide")

@st.cache_data(show_spinner=False)
def cached_fetch_news(ticker, days):
    try:
        df = fetch_news_finnhub(ticker, days)
        if df is not None and not df.empty:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            raw_data_path = os.path.join(project_root, "data", "raw")
            os.makedirs(raw_data_path, exist_ok=True)
            file_path = os.path.join(raw_data_path, f"{ticker}_news_{days}d_finnhub.csv")
            df.to_csv(file_path, index=False)
            return file_path
        return None
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return None

@st.cache_data(show_spinner=False)
def cached_analyze_sentiment(csv_path):
    try:
        df = analyze_sentiment_for_file(csv_path)
        if df is not None and not df.empty:
            processed_path = csv_path.replace("raw", "processed").replace(".csv", "_sentiment.csv")
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            df.to_csv(processed_path, index=False)
            return df
        return None
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return None

def plot_sentiment_charts(df, ticker):
    try:
        st.subheader("News Sentiment Trend")
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        trend = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        st.line_chart(trend)

        st.subheader("Sentiment Distribution")
        counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(4.2, 4.2))  # 缩小饼图大小至原来的0.7倍
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
               colors=['#66b3ff', '#99ff99', '#ff9999'])
        ax.axis('equal')
        ax.set_title(f"{ticker} - Sentiment Distribution")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting sentiment charts: {e}")

def main():
    st.title("S&P 500 Alpha Finder & Portfolio Optimizer")
    st.subheader("Find high-growth stocks and optimize your investment portfolio")

    option = st.sidebar.radio("Select Feature:", ["Stock Analysis", "Portfolio Optimization"])

    if option == "Stock Analysis":
        st.header("Stock Analysis")
        st.subheader("Analyze sentiment, financial indicators & price forecast")

        st.session_state.setdefault("run_analysis", False)

        with st.sidebar:
            st.header("Stock Selection")
            stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "TSLA")
            days = st.radio("Select News Period:", [7, 15, 30], index=1)
            if st.button("Analyze Stock"):
                st.session_state["run_analysis"] = True

        if st.session_state["run_analysis"]:
            ticker = stock_symbol.upper()
            valid_tickers = get_sp500_tickers()
            if ticker not in valid_tickers:
                st.warning(f"'{ticker}' is not a valid S&P 500 ticker. Please check your input.")
                return

            file_path = cached_fetch_news(ticker, days)
            if file_path:
                sentiment_df = cached_analyze_sentiment(file_path)
                if sentiment_df is not None and not sentiment_df.empty:
                    plot_sentiment_charts(sentiment_df, ticker)
                else:
                    st.warning("Sentiment analysis failed or returned no data.")
            else:
                st.warning("News fetch failed or returned no data.")

            st.subheader("(Upcoming) Financial Indicators")
            st.info("Financial data and forecast charts will appear here.")

            st.subheader("(Upcoming) AI-Generated Investment Summary")
            st.info("LLM-based stock summary will appear here.")

    elif option == "Portfolio Optimization":
        st.header("Portfolio Optimization")
        st.subheader("Optimize your asset allocation based on risk and return preferences")

        st.session_state.setdefault("run_optimization", False)

        with st.sidebar:
            st.header("Investment Preferences")
            investable_funds = st.number_input("Enter Investable Funds ($)", value=10000)
            max_drawdown = st.slider("Select Maximum Acceptable Drawdown (%)", 0, 50, 10)

            if st.button("Optimize Portfolio"):
                st.session_state["run_optimization"] = True

        if st.session_state["run_optimization"]:
            st.success(f"Optimizing portfolio for ${investable_funds} with max drawdown {max_drawdown}%... (Mockup Data)")

            portfolio_data = pd.DataFrame({
                "Asset": ["Stocks", "Bonds", "Gold", "Cash"],
                "Weight (%)": [50, 30, 15, 5]
            })
            st.subheader("Optimal Asset Allocation")
            st.table(portfolio_data)

            st.subheader("Efficient Frontier (Mockup)")
            frontier_data = pd.DataFrame({
                "Expected Return (%)": np.linspace(5, 12, 10),
                "Risk (%)": np.linspace(8, 25, 10)
            })
            st.line_chart(frontier_data.set_index("Risk (%)"))

            st.subheader("AI-Generated Portfolio Insights")
            st.markdown("""
            This portfolio is designed to maximize returns while maintaining a balanced risk exposure.

            **Risk Level:** Moderate  
            **Stocks:** Primary growth driver  
            **Bonds:** Stability & income  
            **Gold:** Hedge against market volatility  
            **Cash:** Liquidity buffer  

            **Suitable for:** Moderate risk investors aiming for long-term growth.
            """)

if __name__ == "__main__":
    main()
