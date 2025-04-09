'''
功能：统一封装新闻抓取接口，支持根据参数在 Alpha Vantage 和 Finnhub 之间切换
函数：get_news(ticker, days, source='finnhub') 返回指定股票新闻的 DataFrame
'''

from scripts.fetch_news.fetch_news import fetch_news as fetch_news_alpha
from scripts.fetch_news.fetch_news_finnhub import fetch_news_finnhub


def get_news(stock_symbol, days, source='finnhub'):
    """
    通用新闻抓取接口
    source: 'finnhub'（推荐） or 'alphavantage'
    """
    if source == 'finnhub':
        print(f"[Finnhub] Fetching news for {stock_symbol}...")
        return fetch_news_finnhub(stock_symbol, days)
    elif source == 'alphavantage':
        print(f"[Alpha Vantage] Fetching news for {stock_symbol}...")
        return fetch_news_alpha(stock_symbol, days)
    else:
        raise ValueError("Invalid source. Use 'finnhub' or 'alphavantage'.")
