# 随机生成资产组合，计算收益和最大回撤，提出不满足客户最大回撤需求的组合，建立有效前沿，找到夏普比率最大的五组投资组合供客户选择，并将五组投资组合给到ai，客户可以继续提问

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 设置中文字体（根据系统情况选择，如 Windows 下可用 SimHei）
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def compute_max_drawdown(series):
    """
    计算价格序列的最大回撤，返回值为负数，
    例如 -0.25 表示最大回撤 25%
    """
    cumulative_max = series.cummax()
    drawdown = (series / cumulative_max) - 1
    return drawdown.min()


def generate_random_portfolios(prices_df, num_portfolios=5000):
    """
    随机生成多个投资组合，计算每个组合的主要指标：
      - 年化收益率
      - 年化波动率
      - 夏普比率
      - 最大回撤
    返回 (results_df, weights_list)
    """
    results = []
    weights_list = []

    np.random.seed(42)  # 固定随机种子
    for _ in range(num_portfolios):
        weights = np.random.random(len(prices_df.columns))
        weights /= np.sum(weights)
        weights_list.append(weights)

        portfolio_value = (prices_df * weights).sum(axis=1)
        daily_returns = portfolio_value.pct_change().dropna()

        annual_return = daily_returns.mean() * 252
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        max_dd = compute_max_drawdown(portfolio_value)

        results.append([annual_return, volatility, sharpe_ratio, max_dd])

    results_df = pd.DataFrame(
        results,
        columns=["Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"]
    )
    return results_df, weights_list


def plot_efficient_frontier(results_df):
    """
    绘制有效前沿图：横轴为年化波动率，纵轴为年化收益率，颜色代表夏普比率。
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        results_df["Volatility"],
        results_df["Annual Return"],
        c=results_df["Sharpe Ratio"],
        cmap="viridis",
        alpha=0.5,
        s=10
    )
    plt.xlabel("年化波动率")
    plt.ylabel("年化收益率")
    plt.title("有效前沿图")
    plt.colorbar(scatter, label="夏普比率")
    plt.show()


def plot_cumulative_return(prices_df, weights, title="组合累计收益"):
    """
    根据指定组合的权重计算组合历史累计收益率，并绘制时间序列图。
    """
    portfolio_value = (prices_df * weights).sum(axis=1)
    cumulative_return = portfolio_value / portfolio_value.iloc[0]  # 归一化至1

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_return.index, cumulative_return.values, label="累计收益")
    plt.xlabel("日期")
    plt.ylabel("累计收益")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_two_cumulative_returns(prices_df, rec_weights, chosen_weights,
                                rec_label="推荐组合", chosen_label="您选择的组合"):
    """
    在同一图中绘制推荐组合与用户选择组合的累计收益曲线对比。
    """
    # 推荐组合累计收益
    port_val_rec = (prices_df * rec_weights).sum(axis=1)
    cum_rec = port_val_rec / port_val_rec.iloc[0]

    # 用户选择组合累计收益
    port_val_chosen = (prices_df * chosen_weights).sum(axis=1)
    cum_chosen = port_val_chosen / port_val_chosen.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(cum_rec.index, cum_rec.values, label=rec_label, linewidth=2)
    plt.plot(cum_chosen.index, cum_chosen.values, label=chosen_label, linestyle='--', linewidth=2)
    plt.xlabel("日期")
    plt.ylabel("累计收益(归一化)")
    plt.title("累计收益曲线对比")
    plt.legend()
    plt.show()


def print_portfolio_details(portfolio_series, weights, prices_df, capital, combo_name):
    """
    打印组合的详细指标和资金分配，并绘制该组合的累计收益图。
    """
    print(f"\n【{combo_name}】")
    print(f"  年化收益率：{portfolio_series['Annual Return'] * 100:.2f}%")
    print(f"  年化波动率：{portfolio_series['Volatility'] * 100:.2f}%")
    print(f"  夏普比率：{portfolio_series['Sharpe Ratio']:.2f}")
    print(f"  最大回撤：{portfolio_series['Max Drawdown'] * 100:.2f}%")

    potential_loss = abs(portfolio_series["Max Drawdown"]) * capital
    print(f"\n若全仓投资 {capital} 元，该组合预期最大亏损约为：￥{potential_loss:,.2f}")

    print("\n【建议的资产配置】")
    for asset, w in zip(prices_df.columns, weights):
        allocation = w * capital
        print(f"  {asset}: ￥{allocation:,.2f}")

    plot_cumulative_return(prices_df, weights, title=f"{combo_name} 累计收益")


def main():
    filename = "prices.csv"
    try:
        prices_df = pd.read_csv(filename, index_col=0, parse_dates=True)
    except Exception as e:
        print("读取 CSV 文件出错：", e)
        return

    try:
        capital = float(input("请输入可支配资金（例如 10000）："))
    except ValueError:
        print("请输入有效的资金数值！")
        return
    try:
        max_drawdown_pct = float(input("请输入可接受的最大回撤百分比（如20 表示20%）："))
    except ValueError:
        print("请输入有效数字！")
        return

    threshold = -max_drawdown_pct / 100

    print("\n生成随机组合，请稍候...")
    results_df, weights_list = generate_random_portfolios(prices_df, num_portfolios=5000)

    plot_efficient_frontier(results_df)

    # 筛选最大回撤不超过用户阈值的组合
    valid_idx = results_df[results_df["Max Drawdown"] >= threshold].index
    if len(valid_idx) == 0:
        print("没有组合符合该回撤要求，请放宽限制重试。")
        return

    valid_df = results_df.loc[valid_idx]
    valid_weights = [weights_list[i] for i in valid_idx]

    # 按夏普比率降序排序
    sorted_df = valid_df.sort_values("Sharpe Ratio", ascending=False)
    sorted_indices = sorted_df.index.tolist()
    sorted_weights = [valid_weights[valid_idx.get_loc(i)] for i in sorted_indices]

    # 取前 5 名
    top5 = sorted_df.head(5).copy()
    top5_idx = top5.index.tolist()

    # 拼接资产比例字符串，添加“Composition”列
    comp_list = []
    for idx in top5_idx:
        w = valid_weights[valid_idx.get_loc(idx)]
        comp_str = " | ".join(f"{col}={w_val * 100:.2f}%" for col, w_val in zip(prices_df.columns, w))
        comp_list.append(comp_str)
    top5["Composition"] = comp_list

    print("\n【候选组合排名前 5】（基于夏普比率排序）：")
    display_cols = ["Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown", "Composition"]
    print(top5[display_cols].to_string())
    print("-" * 60)

    # （1）输出推荐组合（即排序后第1名）
    best_idx = sorted_indices[0]
    best_portfolio = sorted_df.loc[best_idx]
    best_weight = sorted_weights[0]
    print_portfolio_details(best_portfolio, best_weight, prices_df, capital, combo_name="推荐组合")

    # （2）让用户从前 5 名中选择一个方案（可选地与推荐组合比较）
    print("\n您也可以从上述【前5】候选组合中选择一个方案查看详细信息，并与推荐组合进行对比：")
    while True:
        try:
            choice = int(input("请输入1~5的数字（0表示放弃选择）："))
            if 0 <= choice <= 5:
                break
            else:
                print("请输入有效数字（0~5）！")
        except ValueError:
            print("请输入数字！")

    if choice == 0:
        print("您选择不再查看其他方案，程序结束。")
        # return
    else:

        chosen_idx = top5_idx[choice - 1]
        chosen_portfolio = top5.loc[chosen_idx]
        chosen_weight = valid_weights[valid_idx.get_loc(chosen_idx)]
        print_portfolio_details(chosen_portfolio, chosen_weight, prices_df, capital,
                                combo_name=f"您选择的方案 {choice}")

        # 计算各指标差值（推荐组合 - 您选择的组合）
        diff_ann_ret = best_portfolio["Annual Return"] - chosen_portfolio["Annual Return"]
        diff_vol = best_portfolio["Volatility"] - chosen_portfolio["Volatility"]
        diff_sr = best_portfolio["Sharpe Ratio"] - chosen_portfolio["Sharpe Ratio"]
        diff_dd = best_portfolio["Max Drawdown"] - chosen_portfolio["Max Drawdown"]

        # 合并输出为一个表格
        comp_data = {
            "指标": ["年化收益率", "年化波动率", "夏普比率", "最大回撤"],
            "推荐组合": [
                f"{best_portfolio['Annual Return'] * 100:.2f}%",
                f"{best_portfolio['Volatility'] * 100:.2f}%",
                f"{best_portfolio['Sharpe Ratio']:.2f}",
                f"{best_portfolio['Max Drawdown'] * 100:.2f}%"
            ],
            "您选择的组合": [
                f"{chosen_portfolio['Annual Return'] * 100:.2f}%",
                f"{chosen_portfolio['Volatility'] * 100:.2f}%",
                f"{chosen_portfolio['Sharpe Ratio']:.2f}",
                f"{chosen_portfolio['Max Drawdown'] * 100:.2f}%"
            ],
            "差值（推荐-选择）": [
                f"{diff_ann_ret * 100:.2f}%",
                f"{diff_vol * 100:.2f}%",
                f"{diff_sr:.2f}",
                f"{diff_dd * 100:.2f}%"
            ]
        }
        compare_df = pd.DataFrame(comp_data)
        print("\n【推荐组合与您选择的组合对比】")
        print(compare_df.to_string(index=False))

        # chosen_idx = top5_idx[choice - 1]
        # chosen_portfolio = top5.loc[chosen_idx]
        # chosen_weight = valid_weights[valid_idx.get_loc(chosen_idx)]
        # print_portfolio_details(chosen_portfolio, chosen_weight, prices_df, capital,
        #                         combo_name=f"您选择的方案 {choice}")

        # # 输出各指标的差值（推荐组合 - 您选择的组合）
        # diff_ann_ret = best_portfolio["Annual Return"] - chosen_portfolio["Annual Return"]
        # diff_vol = best_portfolio["Volatility"] - chosen_portfolio["Volatility"]
        # diff_sr = best_portfolio["Sharpe Ratio"] - chosen_portfolio["Sharpe Ratio"]
        # diff_dd = best_portfolio["Max Drawdown"] - chosen_portfolio["Max Drawdown"]

        # print("\n【推荐组合与您选择的组合对比】")
        # print(f"  年化收益率差值: {diff_ann_ret*100:.2f}%")
        # print(f"  年化波动率差值: {diff_vol*100:.2f}%")
        # print(f"  夏普比率差值: {diff_sr:.2f}")
        # print(f"  最大回撤差值: {diff_dd*100:.2f}%")

    summary_text = summarize_top_n_portfolios(prices_df, top5, [valid_weights[valid_idx.get_loc(i)] for i in top5_idx])

    ask_gemini(summary_text)


def summarize_single_portfolio(prices_df, portfolio_series, weights, name="组合"):
    desc = [f"{name}："]
    desc.append(f"- 年化收益率：{portfolio_series['Annual Return'] * 100:.2f}%")
    desc.append(f"- 年化波动率：{portfolio_series['Volatility'] * 100:.2f}%")
    desc.append(f"- 夏普比率：{portfolio_series['Sharpe Ratio']:.2f}")
    desc.append(f"- 最大回撤：{portfolio_series['Max Drawdown'] * 100:.2f}%")
    desc.append("  资产配置：")
    for asset, w in zip(prices_df.columns, weights):
        desc.append(f"    - {asset}: {w * 100:.2f}%")
    return "\n".join(desc)


def summarize_top_n_portfolios(prices_df, top_n_df, top_n_weights, n=5):
    summary = [f"以下是基于夏普比率筛选出的前 {n} 个投资组合：\n"]
    for i in range(n):
        portfolio_series = top_n_df.iloc[i]
        weights = top_n_weights[i]
        summary.append(summarize_single_portfolio(prices_df, portfolio_series, weights, name=f"组合{i + 1}"))
        summary.append("\n" + "-" * 50 + "\n")
    return "\n".join(summary)


GEMINI_API_KEY = 'AIzaSyA_6-8P1nNtRrSniqW4TWAFM43veS7xaPM'
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def ask_gemini(summary_text):
    print("\n你现在可以问我关于投资组合的问题（输入 exit 退出）：")

    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "退出"]:
            print("对话结束。")
            break

        prompt = f"以下是一个投资组合的信息，我先获取过去一年的数据，随机生成不同的投资组合，剔除掉了最大回撤不符合要求的组合，再绘制有效前沿找到的夏普比率最好的前五组，请你基于它回答我的问题：\n\n{summary_text}\n\n我的问题是：{user_input}"

        response = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )

        if response.status_code == 200:
            try:
                reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                print("AI：" + reply)
            except Exception as e:
                print("解析错误：", e)
        else:
            print("出错了：", response.text)


if __name__ == "__main__":
    main()

