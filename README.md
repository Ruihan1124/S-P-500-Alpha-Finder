# S&P 500 Alpha Finder and Portfolio Optimizer

This project is an AI-powered platform that integrates stock sentiment analysis, price forecasting, and portfolio optimization for S&P 500 companies. It uses FinBERT for natural language sentiment analysis, Prophet for time series forecasting, and Gemini (Google's large language model) to generate explanations and respond to user queries.

## Features

- Sentiment Analysis: Classifies news headlines as positive, neutral, or negative using FinBERT, and calculates weighted sentiment scores based on source reliability.
- Stock Forecasting: Predicts short-term stock price movements using Prophet with sentiment score and political factors as external regressors.
- Interactive Q&A: Uses Google Gemini to answer user questions about stock performance and portfolio selection.
- Portfolio Optimization: Simulates thousands of portfolios, filters by maximum drawdown, and ranks by Sharpe ratio to recommend optimal allocations.
- Web Interface: Built with Streamlit, supporting interactive visualizations and multi-module navigation.

## Project Structure

- `FinBERT_sentiment_forecast.py`: Handles sentiment classification, stock price forecasting, trend visualization, and Gemini interaction.
- `Asset_allocation_gpt.py`: Generates and evaluates investment portfolios based on historical price data and user constraints.
- `streamlit_alpha_finder_app.py`: Main user interface built with Streamlit.
- `fetch_news/`: Contains scripts to retrieve news data from APIs such as Finnhub.
- `ticker_resolver.py`: Maps company names to S&P 500 tickers.
- `data/`: Stores sentiment results, forecast plots, and exported CSVs.

## How to Run

1. Install dependencies:

2. Launch the application:

3. Setup your API keys:
- Get a Finnhub API key at https://finnhub.io for news sentiment analysis.
- Get a Gemini API key at https://makersuite.google.com for LLM-based answers.
- Replace `GEMINI_API_KEY` in relevant scripts with your actual key.

## Use Cases

- Retail investors exploring market sentiment and trends.
- Analysts evaluating data-driven portfolio strategies.
- Students and developers learning financial NLP and forecasting.

## Tech Stack

- FinBERT (Transformers) for sentiment classification
- Prophet for time series forecasting
- Gemini Flash for language-based response generation
- Yahoo Finance and Finnhub for data acquisition
- Streamlit for user interface
- Python libraries: pandas, numpy, matplotlib, torch, tqdm, requests

## Example Outputs

- Sentiment trend plot: daily average weighted sentiment scores.
- Forecast plot: 7-day stock price prediction with confidence intervals.
- Portfolio recommendation: asset allocation, performance stats, comparison charts.

## To Do

- Add user account system with saved preferences and history.
- Include financial indicators (e.g., P/E ratio, ROE) for stock selection.
- Automate daily updates and email alerts.

## License

This project is currently for educational and research purposes. No license is attached.

## Contact

For suggestions or collaboration opportunities, feel free to open an issue or pull request.





