import streamlit as st
import pandas as pd
import openai
import os


def main():
    st.set_page_config(page_title="S&P 500 Alpha Finder", layout="wide")

    st.title("S&P 500 Alpha Finder")
    st.subheader("Find high-growth stocks based on news sentiment, financial analysis & AI predictions")

    # Sidebar for user input
    with st.sidebar:
        st.header("Stock Selection")
        stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "TSLA")
        days = st.radio("Select News Period:", [7, 15, 30], index=0)
        if st.button("Analyze Stock"):
            st.session_state["run_analysis"] = True

    # Placeholder for mockup data
    if "run_analysis" in st.session_state:
        st.success(f"Fetching data for {stock_symbol} ({days} days)... (Mockup Data)")

        # Mockup sentiment data
        sentiment_data = pd.DataFrame({
            "Date": pd.date_range(start="2025-02-20", periods=10, freq="D"),
            "Sentiment Score": [0.2, -0.1, 0.3, 0.5, -0.2, 0.1, 0.4, -0.3, 0.2, 0.0]
        })

        st.subheader("News Sentiment Trend")
        st.line_chart(sentiment_data.set_index("Date"))

        # Mockup news table
        st.subheader("Latest News")
        mock_news = pd.DataFrame({
            "Date": pd.date_range(start="2025-02-20", periods=5, freq="D"),
            "Headline": [
                "Tesla announces new model",
                "Apple stock reaches new highs",
                "Market uncertainty impacts tech stocks",
                "AI stocks surge amid new regulations",
                "Inflation concerns hit Wall Street"
            ],
            "Source": ["CNBC", "Bloomberg", "Reuters", "Yahoo Finance", "WSJ"],
            "URL": ["#", "#", "#", "#", "#"]
        })
        st.dataframe(mock_news, hide_index=True)

        # Mockup financial indicators
        st.subheader("Key Financial Indicators")
        finance_data = pd.DataFrame({
            "Indicator": ["PE Ratio", "ROE", "EPS", "EV/EBITDA"],
            "Value": [15.2, 12.5, 3.4, 10.7]
        })
        st.table(finance_data)

        # Mockup LLM Summary
        st.subheader("AI-Generated Summary & Investment Analysis")
        llm_summary = """Based on the latest financial data and sentiment analysis, the stock shows a moderate growth potential.

            **Market Sentiment:** Positive
        **Stock Forecast:** Bullish trend expected with potential resistance at $320.
        **Risk Assessment:** Moderate risk due to recent market volatility.
        """
        st.write(llm_summary)

        # Interactive Q&A with AI (mockup only)
        st.subheader("Ask AI about the Stock")
        user_question = st.text_input("Ask a question about the stock (e.g., 'Is TSLA a good buy right now?')")
        if user_question:
            st.write(
                "AI Response: This is a placeholder response. In the full version, AI will analyze market data to provide an answer.")


if __name__ == "__main__":
    main()
