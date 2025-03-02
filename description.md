# 1、A description of the final outcome of the project.
The goal of this project is to develop a comprehensive investment decision-support platform tailored for long-term investors and FinTech users. Its main features include:

## (1. News and Market Sentiment Analysis
Automated Data Retrieval and Sentiment Scoring: Automatically gather and analyze news data related to selected stocks. Utilize FinBERT to classify and score sentiments, and incorporate the VIX (Volatility Index) to provide investors with sentiment trends and relevant news links.
User Interface: Users can view sentiment trend line charts and click through to access related news directly from the platform interface.
## (2. Stock Price Forecasting (Prophet Model) and Financial Analysis
Prophet Model Forecasting: Employ the Prophet model to forecast individual stock prices for the next 30 days, coupled with historical data visualization. Users can also optimize parameters (e.g., seasonality, changepoint priors) to improve prediction accuracy.
Financial Statement Integration: Integrate financial statement data to automatically compute key metrics such as Price-to-Earnings (PE), Return on Equity (ROE), Earnings Per Share (EPS), and Enterprise Value to EBITDA (EV/EBITDA). Visualize a company’s fundamental health using tools like radar charts and data tables.
## (3. LLM-Generated Summaries and Risk Assessment
Professional Summaries: Based on the forecast results and sentiment analysis, a large language model (e.g., GPT-4 or GPT-3.5) generates professional summaries covering market overviews, sentiment insights, price trend interpretations, and investment recommendations along with risk warnings.
Customized Risk Alerts: In conjunction with user-defined risk preferences (maximum drawdown, volatility, etc.), the system provides corresponding risk alerts and mitigation strategies.
## (4. Portfolio Allocation and Robo-Advisory
Mean-Variance Optimization: Using the Markowitz mean-variance optimization model, select a portfolio (stocks, bonds, gold, cash) that meets the user’s investable capital and maximum drawdown requirements, and recommend an optimal allocation with the highest Sharpe ratio.
LLM Explanations: Once again, employing a large language model, the system explains the underlying risks, expected returns, and the role of each asset class within the portfolio.
## (5. Interactive Chat
Real-Time Q&A: Users can ask questions about the project’s analytical processes, technical details, or investment strategies. The system leverages collected data and analysis results, combined with the capabilities of a large language model, to provide real-time answers.

# 2、Why your project will be useful for an investment analyst, a trader or a fintech company or a fintech user.
Most people do not have a professional finance background, yet they still need to invest to preserve and grow their wealth. This project aims to bridge the knowledge gap by providing user-friendly features, automated analyses, and clear explanations that enable newcomers to make informed decisions. Here’s how:

## (1.Lowering the Barrier to Entry
The platform consolidates multiple data sources—such as news sentiment, stock price predictions, and financial statements—into a single interface.
Users no longer need to scour different websites or possess specialized data-processing skills. A straightforward, streamlined dashboard displays all the relevant information at once.

## (2.Automated Insights & Educational Explanations
Through the use of a Large Language Model (LLM), the system automatically generates concise, easy-to-understand summaries of complex financial metrics.
This approach makes learning about key concepts (e.g., market sentiment, price forecast, fundamental ratios) more accessible, helping beginners gradually build investment knowledge.

## (3.Personalized Risk Management & Asset Allocation
Many inexperienced investors struggle to gauge or manage risk effectively. This project incorporates Markowitz mean-variance optimization to recommend diversified portfolios (stocks, bonds, gold, cash) based on each user’s maximum allowable drawdown and total investable funds.
Users can explore different allocations—more conservative or more aggressive—depending on their risk tolerance, improving their sense of confidence in portfolio-building decisions.

## (4.Interactive Visualization & Guided Learning
Using an intuitive interface (e.g., Streamlit), users can access various charts (such as sentiment trend lines, stock price forecasts, and radar plots for key ratios) with just a few clicks.
The built-in Q&A or chat feature allows users to query the system—for example, “What does a higher VIX index mean?”—and receive detailed explanations in real time, reinforcing their understanding of investment principles.

## (5.Step-by-Step Knowledge Building
Over time, users become more familiar with how to interpret market sentiment, how stock forecasts are generated, and why certain financial ratios matter.
As their comprehension improves, they can shift from passively reading summaries to actively engaging with the platform’s data, ultimately growing into more confident and informed investors.
In essence, this project demystifies the complexities of investing for individuals who lack a strong financial background. By combining robust quantitative methods with an interactive, explanation-driven design, it empowers newcomers to learn, strategize, and invest in a more informed, responsible manner.

# 3、The datasets you will use.
All core data for this project will be retrieved from Alpha Vantage, including news, historical stock quotes, financial statements, and market indicators as fully as possible. Here are the specifics:

## (1.News Data (News & Sentiment)
Alpha Vantage provides a “News & Sentiment” interface that can retrieve recent to several weeks’ worth of news and sentiment scores based on the company’s ticker or a keyword search.
Note that the free tier limits both the number of API calls and the volume of news articles you can request. You’ll need to determine whether an upgraded plan is necessary based on your project scope.

## (2.Historical Stock Prices
Using Alpha Vantage’s Time Series endpoints, you can fetch daily, weekly, or monthly historical market data for a given stock, often spanning multiple years.
The data includes open, close, high, low, and volume, which can be used for Prophet modeling and technical analysis (e.g., moving averages, RSI, MACD).

## (3.Financial Statements (Fundamental Data)
Alpha Vantage offers a “Fundamental Data” API that covers a company’s balance sheet, income statement, cash flow statement, and certain derived metrics (e.g., PE ratio, EPS).
Generally, you can obtain at least 5 years of annual statements, and quarterly data is also available. These resources should fulfill your needs for financial analysis and ratio calculations.

## (4.VIX Index (Market Fear Index)
VIX is typically marked with the ticker symbol “^VIX” or something similar.
You should verify whether Alpha Vantage can successfully return historical VIX data. If it’s not available or incomplete, you may have to rely on other data sources such as Yahoo Finance to obtain the index.

# 4、A system and data flow diagram.

![Project Overview](https://github.com/user-attachments/assets/2bee3a84-5930-4a84-b57e-884d9d04df6e)

https://github.com/user-attachments/assets/c399649f-5b72-4d6f-afd3-53bef3266023
