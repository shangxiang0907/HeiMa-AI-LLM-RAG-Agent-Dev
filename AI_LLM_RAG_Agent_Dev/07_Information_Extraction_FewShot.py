import json
import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    """
    信息抽取任务 - Few-Shot 学习示例：
    1. 使用 Few-Shot 方式让模型理解信息抽取任务
    2. 从金融文本中提取：日期、股票名称、开盘价、收盘价、成交量
    3. 按照 JSON 格式输出，缺失信息用 '原文未提及' 表示
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
    schema = ['日期', '股票名称', '开盘价', '收盘价', '成交量']
    
    # 4. 定义示例数据（Few-Shot 示例）
    examples_data = [
        {
            "content": "2023-01-10,股市震荡。股票强大科技A股今日开盘价100人民币,一度飙升至105人民币,随后回落至98人民币,最终以102人民币收盘,成交量达到520000。",
            "answers": {
                "日期": "2023-01-10",
                "股票名称": "强大科技A股",
                "开盘价": "100人民币",
                "收盘价": "102人民币",
                "成交量": "520000"
            }
        },
        {
            "content": "2024-05-16,股市波动。股票英伟达美股今日开盘价105美元,一度飙升至109美元,随后回落至100美元,最终以116美元收盘,成交量达到3560000。",
            "answers": {
                "日期": "2024-05-16",
                "股票名称": "英伟达美股",
                "开盘价": "105美元",
                "收盘价": "116美元",
                "成交量": "3560000"
            }
        }
    ]
    
    # 5. 定义需要抽取信息的问题列表
    questions = [
        "2025-06-16,股市震荡。股票传智教育A股今日开盘价66人民币,一度飙升至70人民币,随后回落至65人民币,最终以68人民币收盘,成交量达到123000。",
        "2025-06-06,股市波动。股票黑马程序员A股今日开盘价200人民币,一度飙升至211人民币,随后回落至201人民币,最终以206人民币收盘。"
    ]
    
    # 6. 构建消息列表
    messages = []
    
    # 6.1 添加系统提示
    messages.append({
        "role": "system",
        "content": f"你帮我完成信息抽取,我给你句子,你抽取{schema}信息,按JSON字符串输出,如果某些信息不存在,用'原文未提及'来表示,参考下面的示例。"
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
    print("信息抽取任务 - Few-Shot 学习示例")
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
            "content": f"按照上述示例,现在抽取这个句子的信息:{q}"
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
