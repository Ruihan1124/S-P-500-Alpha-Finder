'''
功能：从 Wikipedia 抓取 S&P 500 指数成分股列表（返回 ticker 列表）
说明：用于批量拉取数据时获取标准股票列表
'''

import yfinance as yf

def get_sp500_tickers():
    sp500 = yf.Ticker("^GSPC")
    table = yf.download("^GSPC", period="1d")  # 这只是获取指数价格
    # 真正的S&P 500成分股可以从下面网站获取：
    import pandas as pd
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    return table['Symbol'].tolist()
