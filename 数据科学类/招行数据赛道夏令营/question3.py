import requests
import time
import json
import pandas as pd

# 模拟产品信息
products = {
    "招招1号": {"risk_level": 1, "features": "低风险，当日赎回", "suitable_for": "资金灵活性需求高，适合学生或短期投资者", "yield_score": 1},
    "招招2号": {"risk_level": 2, "features": "低风险，短期持有", "suitable_for": "短期投资者", "yield_score": 2},
    "招招3号": {"risk_level": 2, "features": "低风险，收益稳定", "suitable_for": "追求稳定收益的客户", "yield_score": 3},
    "招招4号": {"risk_level": 3, "features": "流动性好，额度紧张", "suitable_for": "需要定期流动资金的客户", "yield_score": 4},
    "招招5号": {"risk_level": 3, "features": "收益较高", "suitable_for": "追求收益和流动性平衡的客户", "yield_score": 5},
    "招招6号": {"risk_level": 5, "features": "往期业绩优秀", "suitable_for": "风险承受能力高，追求高收益的客户", "yield_score": 6},
    "招招7号": {"risk_level": 5, "features": "高收益", "min_investment": 500000, "suitable_for": "高净值客户", "yield_score": 7},
    "招招8号": {"risk_level": 5, "features": "高收益", "min_investment": 1000000, "suitable_for": "高净值客户", "yield_score": 8}
}

# API 配置
url = 'https://one-api-other.nowcoder.com/v1/chat/completions'
headers = {
    'Authorization': 'sk-wjuJxyDIEe8PEXty5353B4A3F46441DcB24cCb77CdEa515f',  
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

# 客户营销智能体
def marketing_agent(customer_info, use_api=False):
    # 提取客户信息
    name = customer_info["name"]
    age = customer_info["age"]
    risk_level = customer_info["risk_level"]
    profile = customer_info["profile"]
    funds = customer_info.get("funds", 0)
    needs = customer_info.get("needs", "").lower()

    # 提示词 1：客户背景描述
    customer_prompt = (
        f"客户姓名：{name}\n"
        f"客户年龄：{age}岁\n"
        f"客户职业：{profile}\n"
        f"客户风险评级：{risk_level}级（1级最低，5级最高）\n"
        f"客户可支配资金：{funds}元\n"
        f"客户需求：{needs if needs else '无特殊需求'}\n"
        "请根据以上客户信息，分析其背景和需求，推荐合适的理财产品。"
    )

    # 提示词 2：产品推荐规则
    recommendation_prompt = (
        "产品推荐规则：\n"
        "1. 不可推荐超出客户风险承受能力的产品（客户风险评级为N级，只能推荐风险等级≤N的产品）。\n"
        "2. 如果客户需求中包含‘流动性’或‘灵活性’，优先推荐流动性好的产品（如‘当日赎回’或‘流动性好’的产品）。\n"
        "3. 如果客户需求中包含‘收益’或‘高回报’，优先推荐收益较高的产品（根据 yield_score 排序）。\n"
        "4. 如果客户资金有限，需考虑产品的起购金额限制。\n"
        "5. 综合考虑客户的年龄、职业和需求，推荐最适合的产品组合（1-2个产品）。\n"
        "可用产品列表：\n"
    )
    for prod, info in products.items():
        recommendation_prompt += (
            f"- {prod}：风险等级{info['risk_level']}，特点：{info['features']}，"
            f"适合人群：{info['suitable_for']}，收益评分：{info['yield_score']}"
            f"{', 起购金额：' + str(info['min_investment']) if 'min_investment' in info else ''}\n"
        )
    recommendation_prompt += (
        "请根据客户信息和产品推荐规则，推荐适合的产品（1-2个），并简要说明推荐理由（不超过100字）。"
    )

    # 筛选符合风险等级和资金的产品
    suitable_products = [
        (prod, info) for prod, info in products.items()
        if info["risk_level"] <= risk_level
        and (info.get("min_investment", 0) <= funds if "min_investment" in info else True)
    ]

    # 如果没有合适产品
    if not suitable_products:
        return f"{name}您好！根据您的风险评级和资金情况，暂无适合的产品推荐。理财有风险，投资须谨慎！"

    # 按收益评分排序（降序）
    suitable_products = sorted(suitable_products, key=lambda x: x[1]["yield_score"], reverse=True)

    # 需求分析
    recommendation = []
    reasoning = ""
    needs_liquidity = any(word in needs for word in ["流动性", "灵活性", "流动资金"])
    needs_high_yield = any(word in needs for word in ["收益", "高回报", "收益最大化"])

    # 推荐逻辑
    if needs_liquidity and needs_high_yield:
        liquidity_products = [(p, info) for p, info in suitable_products if "流动性" in info["features"] or "当日赎回" in info["features"]]
        high_yield_products = [(p, info) for p, info in suitable_products if info["yield_score"] > 4]
        if liquidity_products and high_yield_products:
            recommendation = [liquidity_products[0][0], high_yield_products[0][0]] if liquidity_products[0][0] != high_yield_products[0][0] else [liquidity_products[0][0]]
            reasoning = (
                f"推荐‘{recommendation[0]}’，{products[recommendation[0]]['features']}，适合您的流动性需求；"
                f"搭配‘{recommendation[1]}’，{products[recommendation[1]]['features']}，可追求更高收益。"
            ) if len(recommendation) > 1 else f"推荐‘{recommendation[0]}’，{products[recommendation[0]]['features']}，满足您的需求。"
    elif needs_liquidity:
        liquidity_products = [(p, info) for p, info in suitable_products if "流动性" in info["features"] or "当日赎回" in info["features"]]
        if liquidity_products:
            recommendation = [liquidity_products[0][0]]
            reasoning = f"推荐‘{recommendation[0]}’，{products[recommendation[0]]['features']}，满足您的流动性需求。"
        else:
            recommendation = [suitable_products[0][0]]
            reasoning = f"推荐‘{recommendation[0]}’，{products[recommendation[0]]['features']}，适合您的需求。"
    elif needs_high_yield:
        recommendation = [suitable_products[0][0]]
        reasoning = f"推荐‘{recommendation[0]}’，{products[recommendation[0]]['features']}，可帮助您实现收益最大化。"
    else:
        recommendation = [suitable_products[0][0]]
        reasoning = f"推荐‘{recommendation[0]}’，{products[recommendation[0]]['features']}，适合您的需求。"

    # 确保推荐产品存在
    for prod in recommendation:
        if prod not in products:
            return f"错误：推荐产品‘{prod}’不存在于产品列表中！"

    # 提示词 3：话术生成规则
    spiel_prompt = (
        "话术生成规则：\n"
        "1. 话术需自然、亲切，体现对客户的尊重，使用礼貌用语（如‘您好’、‘欢迎咨询’）。\n"
        "2. 话术需简练，控制在1-2句话，突出产品特点和客户需求，不长篇大论。\n"
        "3. 不可保证收益，不可夸大营销（如‘稳赚不赔’），需提醒‘理财有风险，投资须谨慎’。\n"
        "4. 根据客户年龄和职业调整语气：\n"
        "   - 年轻人（年龄<30）：语气轻松、活泼，突出灵活性和低风险。\n"
        "   - 中年人（年龄30-50）：语气专业，突出收益和稳定性。\n"
        "   - 老年人（年龄>50）：语气稳重、温和，突出安全性和流动性。\n"
        "5. 话术需结合客户背景和推荐理由，结构为：\n"
        "   - 开场：'{客户姓名}您好！根据您的{背景描述}，我为您推荐{产品}。'\n"
        "   - 推荐理由：简要说明推荐原因，突出产品特点和客户需求匹配。\n"
        "   - 结尾：'理财有风险，投资须谨慎！欢迎随时咨询哦！'\n"
        "请根据以上规则，生成符合要求的营销话术。"
    )

    # 构造完整提示词
    full_prompt = f"{customer_prompt}\n\n{recommendation_prompt}\n\n推荐产品：{', '.join(recommendation)}\n推荐理由：{reasoning}\n\n{spiel_prompt}"

    # 如果使用 API 生成话术
    if use_api:
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": full_prompt}
            ]
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            response.raise_for_status()
            answer = response.json()['choices'][0]['message']['content']
            return answer
        except Exception as e:
            print(f"API 调用失败: {e}")
            return None

    # 手动生成话术
    background = background_info(profile, age, needs)
    product_str = "和".join(recommendation) if len(recommendation) > 1 else recommendation[0]

    # 根据年龄调整语气
    if age < 30:
        tone = "轻松"
    elif age <= 50:
        tone = "专业"
    else:
        tone = "稳重"

    spiel = (f"{name}您好！根据您的{background}，我为您推荐{product_str}。{reasoning}"
             "理财有风险，投资须谨慎！欢迎随时咨询哦！")
    return spiel

# 辅助函数：生成背景信息描述
def background_info(profile, age, needs):
    return f"{age}岁，{profile}" + (f"，{needs}" if needs else "")

# run 函数：结合客户信息和 marketing_agent 的推荐逻辑
def run(inf):
    # 构造客户信息
    customer = {
        "name": "张强",
        "age": 45,
        "risk_level": 4,
        "profile": "私营业主",
        "funds": 1000000,
        "needs": "短期高收益，但需保证本金安全"
    }

    # 使用 marketing_agent 生成推荐和话术
    spiel = marketing_agent(customer, use_api=False)

    # 构造 API 请求，附加 inf 信息
    prompt = (
        f"客户信息：{customer['name']}，{customer['age']}岁，风险评级{customer['risk_level']}级，"
        f"{customer['profile']}，可支配资金{customer['funds']}元，需求是“{customer['needs']}”\n"
        f"本地推荐结果：{spiel}\n"
        f"附加信息：{inf}\n"
        "请根据以上信息，进一步优化推荐话术，确保符合客户需求，话术自然、简练，并提醒‘理财有风险，投资须谨慎’。"
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
        response.raise_for_status()
        response_data = response.json()
        print("API 响应：", response.text)
        return response_data['choices'][0]['message']['content']
    except Exception as e:
        print(f"API 调用失败: {e}")
        # 如果 API 调用失败，返回本地生成的推荐
        return spiel

# 测试客户信息
customers = [
    {"name": "王路飞", "age": 25, "risk_level": 1, "profile": "在校大学生", "needs": "资金灵活性高"},
    {"name": "唐小舞", "age": 35, "risk_level": 5, "profile": "公司白领", "funds": 400000, "needs": "追求收益最大化"},
    {"name": "萧尘", "age": 58, "risk_level": 3, "profile": "企业职工", "needs": "每月需流动资金"},
    {"name": "李明", "age": 42, "risk_level": 4, "profile": "自由职业者", "funds": 300000, "needs": "收益和流动性平衡"},
    {"name": "张丽", "age": 65, "risk_level": 2, "profile": "退休人员", "needs": "低风险，资金安全"}
]

# 测试智能体
if __name__ == "__main__":
    # 打印本地推荐结果
    for customer in customers:
        print("推荐话术：")
        print(marketing_agent(customer, use_api=False))
        print("-" * 50)

    # 调用 run 函数
    print("\nAPI 推荐结果：")
    result = run("question3.py")
    print("最终推荐：", result)