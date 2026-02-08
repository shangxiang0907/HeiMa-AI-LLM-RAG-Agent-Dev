import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    """
    文本匹配任务 - Few-Shot 学习示例：
    1. 使用 Few-Shot 方式让模型理解文本匹配任务
    2. 识别成对的句子中，2句话是否有关联
    3. 按照指定格式输出：'是' 或 '不是'
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
    
    # 3. 定义 Few-Shot 示例数据（让模型学习什么是文本匹配任务）
    examples_data = [
        {
            "sentence1": "公司ABC发布了季度财报,显示盈利增长。",
            "sentence2": "财报披露,公司ABC利润上升",
            "answer": "是"
        },
        {
            "sentence1": "黄金价格下跌,投资者抛售。",
            "sentence2": "外汇市场交易额创下新高",
            "answer": "不是"
        }
    ]
    
    # 4. 定义需要判断的文本对（测试数据）
    test_pairs = [
        {
            "sentence1": "股票市场今日大涨,投资者乐观。",
            "sentence2": "持续上涨的市场让投资者感到满意。"
        },
        {
            "sentence1": "油价大幅下跌,能源公司面临挑战。",
            "sentence2": "未来智能城市的建设趋势愈发明显。"
        },
        {
            "sentence1": "利率上升,影响房地产市场。",
            "sentence2": "高利率对房地产有一定冲击。"
        }
    ]
    
    # 5. 构建消息列表
    messages = []
    
    # 5.1 添加系统提示（解释什么是文本匹配任务）
    messages.append({
        "role": "system",
        "content": "你是一个文本匹配专家。你的任务是判断给定的两个句子是否有关联。\n\n"
                   "如果两个句子在语义上相关、讨论同一主题或存在逻辑关联，则回答'是'；\n"
                   "如果两个句子在语义上无关、讨论不同主题或不存在逻辑关联，则回答'不是'。\n\n"
                   "请按照以下格式输出：\n"
                   "- 如果有关联，输出：是\n"
                   "- 如果无关联，输出：不是"
    })
    
    # 5.2 添加 Few-Shot 示例（让模型学习如何判断文本匹配）
    for example in examples_data:
        # 添加用户输入（待判断的句子对）
        messages.append({
            "role": "user",
            "content": f"句子一:{example['sentence1']}\n句子二:{example['sentence2']}"
        })
        # 添加助手回复（期望的判断结果）
        messages.append({
            "role": "assistant",
            "content": example["answer"]
        })
    
    # 6. 对每个文本对进行匹配判断
    print("=" * 80)
    print("文本匹配任务 - Few-Shot 学习示例")
    print("=" * 80)
    print(f"\nFew-Shot 示例数量: {len(examples_data)}")
    print(f"待判断的文本对数量: {len(test_pairs)}")
    print("\n" + "-" * 80)
    
    # 显示 Few-Shot 示例
    print("\n【Few-Shot 示例】")
    for i, example in enumerate(examples_data, 1):
        print(f"\n示例 {i}:")
        print(f"  句子一: {example['sentence1']}")
        print(f"  句子二: {example['sentence2']}")
        print(f"  答案: {example['answer']}")
    
    print("\n" + "-" * 80)
    
    # 对每个测试文本对进行判断
    results = []
    for i, pair in enumerate(test_pairs, 1):
        print(f"\n【测试 {i}】")
        print(f"句子一: {pair['sentence1']}")
        print(f"句子二: {pair['sentence2']}")
        print("\n判断结果:")
        
        # 添加当前问题到消息列表
        current_messages = messages + [{
            "role": "user",
            "content": f"句子一:{pair['sentence1']}\n句子二:{pair['sentence2']}"
        }]
        
        # 调用 API
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=current_messages
        )
        
        # 获取判断结果
        result = response.choices[0].message.content.strip()
        print(result)
        
        results.append(result)
        print("-" * 80)
    
    # 7. 汇总结果
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    print(f"\n期望结果: ['是', '不是', '是']")
    print(f"实际结果: {results}")
    
    # 验证结果
    expected_results = ['是', '不是', '是']
    correct_count = sum(1 for i, (expected, actual) in enumerate(zip(expected_results, results)) 
                        if expected == actual)
    
    print(f"\n正确率: {correct_count}/{len(test_pairs)} ({correct_count/len(test_pairs)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("任务完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
