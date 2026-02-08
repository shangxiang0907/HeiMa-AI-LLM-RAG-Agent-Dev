import json
import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    """
    彩票信息抽取任务 - Few-Shot 学习示例：
    1. 使用 Few-Shot 方式让模型理解彩票信息抽取任务
    2. 从彩票文本中提取：期数、中奖号码（红球+篮球）、一等奖
    3. 按照 JSON 格式输出
    """
    # 1. 加载环境变量
    load_dotenv()
    
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("未在环境变量或 .env 中找到 API_KEY，请先配置后再运行。")
    
    # 2. 初始化客户端（兼容阿里云 DashScope 的 OpenAI 模式）
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # 也可以使用本地模型：base_url="http://localhost:11434/v1"
    
    # 3. 定义需要抽取的信息字段（Schema）
    schema = ['期数', '中奖号码', '一等奖']
    
    # 4. 定义示例数据（Few-Shot 示例）
    examples_data = [
        {
            "content": "2025年第100期,开好红球22 21 06 01 03 11 篮球 07,一等奖中奖为2注。",
            "answers": {
                "期数": "2025100",
                "中奖号码": [1, 3, 6, 11, 21, 22, 7],
                "一等奖": "2注"
            }
        },
        {
            "content": "2025101期,有3注1等奖,10注2等奖,开号篮球11,中奖红球3、5、7、11、12、16。",
            "answers": {
                "期数": "2025101",
                "中奖号码": [3, 5, 7, 11, 12, 16, 11],
                "一等奖": "3注"
            }
        }
    ]
    
    # 5. 定义需要抽取信息的问题列表（根据作业要求，应该有5条文本，这里提供3条作为测试）
    questions = [
        "2025年第102期,开好红球05 12 18 23 28 33 篮球 09,一等奖中奖为5注。",
        "2025103期,有1注1等奖,20注2等奖,开号篮球15,中奖红球2、8、14、19、25、30。",
        "2025年第104期,开好红球01 07 13 20 26 31 篮球 04,一等奖中奖为0注。"
    ]
    
    # 6. 构建消息列表
    messages = []
    
    # 6.1 添加系统提示
    messages.append({
        "role": "system",
        "content": f"你是一个彩票信息抽取专家。请从彩票文本中提取以下信息：{schema}。\n\n"
                   f"提取规则：\n"
                   f"1. 期数：提取年份和期号，格式为YYYYNNN（如'2025年第100期'提取为'2025100'，'2025101期'提取为'2025101'）\n"
                   f"2. 中奖号码：提取所有红球号码（按升序排列）和篮球号码，篮球号码放在最后。如果篮球号码与红球号码重复，也要包含在列表中。\n"
                   f"3. 一等奖：提取一等奖的中奖注数，格式为'数字+注'（如'2注'、'3注'）\n\n"
                   f"请按照JSON格式输出，参考下面的示例。"
    })
    
    # 6.2 添加 Few-Shot 示例（让模型学习如何抽取信息）
    for example in examples_data:
        # 添加用户输入（待抽取的句子）
        messages.append({
            "role": "user",
            "content": example["content"]
        })
        # 添加助手回复（期望的抽取结果）
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["answers"], ensure_ascii=False)
        })
    
    # 7. 对每个问题进行信息抽取
    print("=" * 80)
    print("彩票信息抽取任务 - Few-Shot 学习示例")
    print("=" * 80)
    print(f"\n需要抽取的字段: {schema}")
    print(f"\nFew-Shot 示例数量: {len(examples_data)}")
    print(f"待抽取的问题数量: {len(questions)}")
    print("\n" + "-" * 80)
    
    for i, q in enumerate(questions, 1):
        print(f"\n【问题 {i}】")
        print(f"原文: {q}")
        print("\n抽取结果:")
        
        # 添加当前问题到消息列表
        current_messages = messages + [{
            "role": "user",
            "content": f"请按照上述示例，抽取以下彩票文本的信息：{q}"
        }]
        
        # 调用 API
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=current_messages
        )
        
        # 打印抽取结果
        result = response.choices[0].message.content
        print(result)
        
        # 尝试解析 JSON 并格式化输出
        try:
            result_dict = json.loads(result)
            print("\n格式化后的结果:")
            print(json.dumps(result_dict, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print("(注意: 返回结果不是有效的 JSON 格式)")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("任务完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
