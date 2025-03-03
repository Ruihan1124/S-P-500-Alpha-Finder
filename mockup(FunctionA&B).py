import streamlit as st
import pandas as pd
import numpy as np
import openai
import os


st.set_page_config(page_title="S&P 500 Alpha Finder & Portfolio Optimizer", layout="wide")

def get_ai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a stock market expert."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def main():
    # Application Title
    st.title("S&P 500 Alpha Finder & Portfolio Optimizer")
    st.subheader("Find high-growth stocks and optimize your investment portfolio")

    # Sidebar selection for different features
    option = st.sidebar.radio("Select Feature:", ["Stock Analysis", "Portfolio Optimization"])

    if option == "Stock Analysis":
        # ===================== Function A: Stock Analysis =====================
        st.header("Stock Analysis")
        st.subheader("Find high-growth stocks based on news sentiment, financial analysis & AI predictions")

        
        st.session_state.setdefault("run_analysis", False)

        # Sidebar for user input
        with st.sidebar:
            st.header("Stock Selection")
            stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "TSLA")
            days = st.radio("Select News Period:", [7, 15, 30], index=0)
            if st.button("Analyze Stock"):
                st.session_state["run_analysis"] = True

        
        if st.session_state["run_analysis"]:
            st.success(f"Fetching data for {stock_symbol} ({days} days)... (Mockup Data)")

            # Sentiment Data
            sentiment_data = pd.DataFrame({
                "Date": pd.date_range(start="2025-02-20", periods=10, freq="D"),
                "Sentiment Score": np.random.uniform(-0.5, 0.5, 10)
            })

            st.subheader("News Sentiment Trend")
            st.line_chart(sentiment_data.set_index("Date"))

            # News Table
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
            st.dataframe(mock_news.set_index("Date"))

            # Financial Indicators
            st.subheader("Key Financial Indicators")
            finance_data = pd.DataFrame({
                "Indicator": ["PE Ratio", "ROE", "EPS", "EV/EBITDA"],
                "Value": [15.2, 12.5, 3.4, 10.7]
            })
            st.table(finance_data)

            # AI Summary
            st.subheader("AI-Generated Summary & Investment Analysis")
            llm_summary = """Based on the latest financial data and sentiment analysis, the stock shows a moderate growth potential.

                **Market Sentiment:** Positive  
                **Stock Forecast:** Bullish trend expected with potential resistance at $320.  
                **Risk Assessment:** Moderate risk due to recent market volatility.  
            """
            st.write(llm_summary)

            # Interactive Q&A with AI
            st.subheader("Ask AI about the Stock")
            user_question = st.text_input("Ask a question about the stock (e.g., 'Is TSLA a good buy right now?')")
            if user_question:
                response = get_ai_response(user_question)
                st.write(f"AI Response: {response}")

    elif option == "Portfolio Optimization":
        # ===================== Function B: Portfolio Optimization =====================
        st.header("Portfolio Optimization")
        st.subheader("Optimize your asset allocation based on risk and return preferences")


        st.session_state.setdefault("run_optimization", False)

        # Sidebar for investment preferences
        with st.sidebar:
            st.header("Investment Preferences")
            investable_funds = st.number_input("Enter Investable Funds ($)", value=10000)
            max_drawdown = st.slider("Select Maximum Acceptable Drawdown (%)", 0, 50, 10)

            if st.button("Optimize Portfolio"):
                st.session_state["run_optimization"] = True

        # Mockup portfolio optimization data
        if st.session_state["run_optimization"]:
            st.success(f"Optimizing portfolio for ${investable_funds} with max drawdown {max_drawdown}%... (Mockup Data)")

            # Optimal Portfolio Allocation
            portfolio_data = pd.DataFrame({
                "Asset": ["Stocks", "Bonds", "Gold", "Cash"],
                "Weight (%)": [50, 30, 15, 5]
            })
            st.subheader("Optimal Asset Allocation")
            st.table(portfolio_data)

            # Efficient Frontier
            st.subheader("Efficient Frontier (Mockup)")
            frontier_data = pd.DataFrame({
                "Expected Return (%)": np.linspace(5, 12, 10),
                "Risk (%)": np.linspace(8, 25, 10)
            })
            st.line_chart(frontier_data.set_index("Risk (%)"))

            # AI Portfolio Insights
            st.subheader("AI-Generated Portfolio Insights")
            ai_explanation = """This portfolio is designed to maximize returns while maintaining a balanced risk exposure.

            **Risk Level:** Moderate  
            **Stocks:** Primary growth driver  
            **Bonds:** Stability & income  
            **Gold:** Hedge against market volatility  
            **Cash:** Liquidity buffer  

            **Suitable for:** Moderate risk investors aiming for long-term growth.  
            """
            st.write(ai_explanation)

            # Interactive AI Q&A for portfolio optimization
            st.subheader("Ask AI about the Portfolio")
            user_question = st.text_input("Ask a question about the portfolio (e.g., 'Why is gold included?')")
            if user_question:
                response = get_ai_response(user_question)
                st.write(f"AI Response: {response}")

if __name__ == "__main__":
    main()
