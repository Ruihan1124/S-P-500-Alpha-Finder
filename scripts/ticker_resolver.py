import requests
import pandas as pd

# Step 1: 爬取 S&P 500 列表
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]  # 第一个表格即为成分股
        sp500_dict = dict(zip(df['Security'], df['Symbol']))
        return sp500_dict
    except Exception as e:
        print(f"❌ 获取 S&P500 公司列表失败: {e}")
        return {}

# Step 2: 用户输入公司名，模糊匹配
def resolve_ticker_local(sp500_dict):
    while True:
        user_input = input("请输入公司名称（如 Apple Inc.）：").strip().lower()
        matches = {name: symbol for name, symbol in sp500_dict.items() if user_input in name.lower()}
        if not matches:
            print("❌ 未找到匹配的公司名称，请重新输入。\n")
        elif len(matches) == 1:
            name, symbol = list(matches.items())[0]
            print(f"✅ 找到：{name}（{symbol}）")
            return symbol
        else:
            print("🔍 找到多个匹配项，请选择：")
            for i, (name, symbol) in enumerate(matches.items()):
                print(f"{i+1}. {name} ({symbol})")
            try:
                choice = int(input("请输入对应序号："))
                if 1 <= choice <= len(matches):
                    selected = list(matches.items())[choice - 1]
                    print(f"✅ 选择成功：{selected[0]}（{selected[1]}）")
                    return selected[1]
                else:
                    print("❌ 无效选择，请重新输入。\n")
            except:
                print("❌ 输入无效，请输入数字。\n")

# 主程序入口
if __name__ == "__main__":
    sp500_dict = get_sp500_tickers()
    if sp500_dict:
        ticker = resolve_ticker_local(sp500_dict)
        print("📈 你选择的股票代码是：", ticker)
