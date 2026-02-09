"""
使用 LangChain 访问阿里云通义大模型（Tongyi）示例

本示例对应课件中的代码片段：

from langchain_community.llms.tongyi import Tongyi
llm = Tongyi(model="qwen-max")
res = llm.invoke("帮我讲个笑话吧")
print(res)

在此基础上，我们做了以下增强：
- 使用 .env / 环境变量中读取 API Key
- 演示多轮调用与不同提示词
- 对返回结果增加简单的格式化输出
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.llms.tongyi import Tongyi


def init_llm() -> Tongyi:
    """
    初始化 Tongyi LLM 模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）
    """
    load_dotenv()

    # 兼容两种环境变量命名方式
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    # LangChain 的 Tongyi 封装会自动从环境变量中读取 key，
    # 这里设置一份到 DASHSCOPE_API_KEY，确保兼容性。
    os.environ["DASHSCOPE_API_KEY"] = api_key

    # 课件中的示例使用 qwen-max，这里保持一致
    llm = Tongyi(model="qwen-max")
    return llm


def single_call_demo(llm: Tongyi) -> None:
    """
    对应课件中的最小示例：调用一次模型，生成一个笑话。
    """
    print("=" * 80)
    print("【示例1】最小调用示例：让模型讲一个笑话")
    print("-" * 80)

    prompt = "帮我讲个轻松幽默、适合职场分享的中文笑话。"
    res = llm.invoke(prompt)

    print("提示词：", prompt)
    print("\n模型回复：")
    print(res)
    print()


def multi_call_demo(llm: Tongyi) -> None:
    """
    展示使用同一个 LLM 实例进行多次调用。
    """
    print("=" * 80)
    print("【示例2】多次调用同一模型，完成不同任务")
    print("-" * 80)

    prompts: List[str] = [
        "用 2 句话解释一下什么是大语言模型（LLM），面向零基础读者。",
        "把下面这句话润色成更正式的技术分享开场白：今天我们主要讲大模型在实际项目里的使用。",
        "请给出 3 条使用通义千问进行应用开发时的最佳实践要点，以列表形式回答。",
    ]

    for i, prompt in enumerate(prompts, start=1):
        print(f"\n--- 子任务 {i} ---")
        print("提示词：", prompt)
        res = llm.invoke(prompt)
        print("模型回复：")
        print(res)

    print()


def main() -> None:
    """
    主函数：演示如何使用 LangChain 调用阿里云通义大模型 Tongyi。
    """
    print("=" * 80)
    print("使用 LangChain 访问阿里云通义大模型（Tongyi）示例")
    print("=" * 80)

    llm = init_llm()

    # 对应 PPT 中的最小示例
    single_call_demo(llm)

    # 扩展示例：多次调用
    multi_call_demo(llm)

    print("=" * 80)
    print("演示结束")
    print("=" * 80)


if __name__ == "__main__":
    main()

