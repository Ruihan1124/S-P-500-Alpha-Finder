'''
功能：使用 FinBERT 模型对抓取的新闻进行情绪分析
输入：data/raw 目录下的新闻数据（如 AAPL_news_15d_finnhub.csv）
输出：添加情绪标签和置信度的 DataFrame，保存为 sentiment CSV 文件
'''

import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

# 加载 FinBERT 模型
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

# 预测单条文本的情绪
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[prediction.item()], round(confidence.item(), 4)

# 分析整个文件
def analyze_sentiment_for_file(filepath):
    df = pd.read_csv(filepath)

    if 'headline' not in df.columns:
        raise ValueError(f"'headline' column not found in {filepath}")

    sentiments = []
    confidences = []

    print(f"📊 Analyzing sentiment for: {os.path.basename(filepath)}")

    for text in tqdm(df['headline']):
        label, conf = classify_sentiment(str(text))
        sentiments.append(label)
        confidences.append(conf)

    df['sentiment'] = sentiments
    df['confidence'] = confidences

    return df

# 运行分析任务
def run_sentiment_analysis():
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_folder = os.path.join(project_root, "data", "raw")
    output_folder = os.path.join(project_root, "data", "processed")
    os.makedirs(output_folder, exist_ok=True)

    # 获取已处理文件名（去掉后缀）
    processed_files = {
        f.replace('_sentiment.csv', '') for f in os.listdir(output_folder) if f.endswith('_sentiment.csv')
    }

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv') and 'finnhub' in filename:
            base_name = filename.replace('.csv', '')
            if base_name in processed_files:
                print(f"⏭️ Skipping already processed: {filename}")
                continue

            filepath = os.path.join(input_folder, filename)
            df = analyze_sentiment_for_file(filepath)

            output_file = filename.replace('.csv', '_sentiment.csv')
            df.to_csv(os.path.join(output_folder, output_file), index=False)
            print(f"✅ Saved: {output_file}")


if __name__ == "__main__":
    run_sentiment_analysis()
