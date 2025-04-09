# S&P 500 Alpha Finder

## Project Overview
S&P 500 Alpha Finder is a data-driven platform that integrates **Natural Language Processing (NLP)**, **Time Series Forecasting**, and **Financial Statement Analysis** to identify potential high-growth stocks from the S&P 500 index.

## Key Features

-  **News Sentiment Analysis** (FinBERT + Finnhub API)
-  **Stock Price Prediction** (Prophet + Technical Indicators)
-  **Financial Statement Analysis** (PE, ROE, EPS, EV/EBITDA)
-  **LLM-based Summary & Risk Report** (GPT-4)
- **Interactive Streamlit Interface** for easy exploration

---

## Module Overview

### `scripts/fetch_news/`

| File | Description |
|------|-------------|
| `fetch_news.py` | Fetches news from Alpha Vantage (backup source). |
| `fetch_news_finnhub.py` | Fetches company news from Finnhub API (main source). |
| `news_api.py` | Unified interface providing `get_news(ticker, days, source='finnhub')`. |

### `scripts/`

| File | Description |
|------|-------------|
| `utils.py` | Gets S&P 500 stock tickers from Wikipedia. |
| `batch_fetch_news.py` | Batch download news for selected tickers using the unified API. |

---

## Installation

### 1. Create and activate a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # on Unix or MacOS
venv\Scripts\activate     # on Windows

