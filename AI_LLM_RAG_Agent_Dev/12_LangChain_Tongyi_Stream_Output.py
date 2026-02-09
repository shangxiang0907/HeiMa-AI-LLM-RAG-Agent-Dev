"""
使用 LangChain 实现通义大模型（Tongyi）的流式输出示例

本示例对应课件中的代码片段：

from langchain_community.llms.tongyi import Tongyi
model = Tongyi(model="qwen-max")
res = model.stream(input="你是谁呀能做什么?")
for chunk in res:
    print(chunk, end="", flush=True)

核心概念：
- invoke 方法：一次性返回完整结果
- stream 方法：逐段返回结果，实现流式输出

在此基础上，我们做了以下增强：
- 使用 .env / 环境变量中读取 API Key
- 对比 invoke 和 stream 两种方法的输出方式
- 演示流式输出的实时效果
"""

import os
import time

from dotenv import load_dotenv
from langchain_community.llms.tongyi import Tongyi


def init_llm() -> Tongyi:
    """
    初始化 Tongyi LLM 模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）

    注意：不使用 qwen3-max，因为 qwen3-max 是聊天模型，qwen-max 是大语言模型
    """
    # load_dotenv() 会从项目根目录的 .env 文件中读取环境变量，
    # 并将它们加载到当前进程的环境变量中（os.environ），
    # 这样后续就可以通过 os.getenv() 来访问这些变量了。
    # 如果项目根目录没有 .env 文件，则不会报错，只是不会加载任何变量。
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


def invoke_demo(llm: Tongyi) -> None:
    """
    演示使用 invoke 方法：一次性返回完整结果。
    """
    print("=" * 80)
    print("【示例1】使用 invoke 方法：一次性返回完整结果")
    print("-" * 80)

    prompt = "你是谁呀能做什么?"
    print(f"提示词：{prompt}")
    print("\n模型回复（一次性返回）：")

    start_time = time.time()
    res = llm.invoke(prompt)
    elapsed_time = time.time() - start_time

    print(res)
    print(f"\n总耗时：{elapsed_time:.2f} 秒")
    print()


def stream_demo(llm: Tongyi) -> None:
    """
    演示使用 stream 方法：逐段返回结果，实现流式输出。
    """
    print("=" * 80)
    print("【示例2】使用 stream 方法：逐段返回结果，流式输出")
    print("-" * 80)

    prompt = "你是谁呀能做什么?"
    print(f"提示词：{prompt}")
    print("\n模型回复（流式输出）：")

    start_time = time.time()
    res = llm.stream(input=prompt)

    # 流式打印每个 chunk
    full_response = []
    chunk_count = 0
    for chunk in res:
        chunk_count += 1
        full_response.append(chunk)
        # end="" 表示不换行，flush=True 表示立即刷新输出缓冲区
        print(chunk, end="", flush=True)

    elapsed_time = time.time() - start_time

    print(f"\n\n总耗时：{elapsed_time:.2f} 秒")
    print(f"共接收到 {chunk_count} 个数据块")
    print(f"完整回复长度：{len(''.join(full_response))} 字符")
    print()


def stream_comparison_demo(llm: Tongyi) -> None:
    """
    对比演示：展示流式输出和一次性输出的区别。
    """
    print("=" * 80)
    print("【示例3】对比演示：流式输出 vs 一次性输出")
    print("-" * 80)

    prompt = "请用中文详细介绍一下人工智能的发展历史，大约200字。"
    print(f"提示词：{prompt}\n")

    # 流式输出
    print("--- 流式输出（stream）---")
    print("开始时间：", time.strftime("%H:%M:%S", time.localtime()))
    print("输出内容：", end="", flush=True)

    start_time = time.time()
    res_stream = llm.stream(input=prompt)
    for chunk in res_stream:
        print(chunk, end="", flush=True)
    stream_time = time.time() - start_time

    print(f"\n流式输出耗时：{stream_time:.2f} 秒\n")

    # 一次性输出
    print("--- 一次性输出（invoke）---")
    print("开始时间：", time.strftime("%H:%M:%S", time.localtime()))
    print("输出内容：", end="", flush=True)

    start_time = time.time()
    res_invoke = llm.invoke(prompt)
    invoke_time = time.time() - start_time

    print(res_invoke)
    print(f"\n一次性输出耗时：{invoke_time:.2f} 秒")
    print()


def main() -> None:
    """
    主函数：演示如何使用 LangChain 实现通义大模型的流式输出。
    """
    print("=" * 80)
    print("LangChain 通义大模型流式输出示例")
    print("=" * 80)
    print()

    llm = init_llm()

    # 示例1：invoke 方法演示
    invoke_demo(llm)

    # 示例2：stream 方法演示
    stream_demo(llm)

    # 示例3：对比演示
    stream_comparison_demo(llm)

    print("=" * 80)
    print("演示结束")
    print("=" * 80)


if __name__ == "__main__":
    main()
