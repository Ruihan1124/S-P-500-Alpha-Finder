import requests
import json

NEWS_API_KEY = "754cba392288404cb3909e256f979726"

url = f"https://newsapi.org/v2/everything?q=Apple&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
response = requests.get(url)

if response.status_code == 200:
    news_data = response.json()
    print(json.dumps(news_data, indent=4))  # 打印新闻数据
else:
    print("Error:", response.text)
