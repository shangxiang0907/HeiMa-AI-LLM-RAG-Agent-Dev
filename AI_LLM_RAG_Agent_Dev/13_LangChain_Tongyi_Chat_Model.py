"""
使用 LangChain 调用通义聊天模型（ChatTongyi）示例

本示例演示如何使用 LangChain 的 ChatTongyi 进行多轮对话，包括：
- SystemMessage：设置系统角色
- HumanMessage：用户消息
- AIMessage：AI 回复消息
- 流式输出：实时显示模型生成的内容

核心概念：
- ChatTongyi：聊天模型，与 Tongyi LLM 不同，专门用于对话场景
- SystemMessage：设置 AI 的角色和行为
- HumanMessage：用户输入的消息
- AIMessage：AI 的回复消息
- stream 方法：流式输出，逐段返回结果

在此基础上，我们做了以下增强：
- 使用 .env / 环境变量中读取 API Key
- 演示多轮对话场景
- 演示流式输出的实时效果
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def init_chat_model() -> ChatTongyi:
    """
    初始化 ChatTongyi 聊天模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）

    注意：使用 qwen3-max，这是聊天模型，适合对话场景
    """
    load_dotenv()

    # 兼容两种环境变量命名方式
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    # LangChain 的 ChatTongyi 封装会自动从环境变量中读取 key，
    # 这里设置一份到 DASHSCOPE_API_KEY，确保兼容性。
    os.environ["DASHSCOPE_API_KEY"] = api_key

    # 使用 qwen3-max 聊天模型
    chat = ChatTongyi(model="qwen3-max")
    return chat


def chat_with_system_message_demo(chat: ChatTongyi) -> None:
    """
    演示使用 SystemMessage + HumanMessage + AIMessage 进行多轮对话。

    这个示例展示了：
    1. 通过 SystemMessage 设置 AI 的角色（边塞诗人）
    2. 用户请求写一首唐诗
    3. AI 回复一首示例诗
    4. 用户要求按照上一首的格式再写一首
    5. 使用流式输出显示 AI 的回复
    """
    print("=" * 80)
    print("【示例1】演示 SystemMessage + HumanMessage + AIMessage 多轮对话")
    print("-" * 80)

    # 准备消息列表
    messages = [
        SystemMessage(content="你是一名来自边塞的诗人"),
        HumanMessage(content="给我写一首唐诗"),
        AIMessage(content="锄禾日当午,汗滴禾下土,谁知盘中餐,粒粒皆辛苦。"),
        HumanMessage(content="基于你上一首的格式,再来一首"),
    ]

    print("消息列表：")
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, SystemMessage):
            print(f"  {i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"  {i}. [用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"  {i}. [AI] {msg.content}")

    print("\n模型回复（流式输出）：")
    print("-" * 80)

    # 流式输出
    for chunk in chat.stream(input=messages):
        print(chunk.content, end="", flush=True)

    print("\n")
    print("-" * 80)
    print()


def simple_chat_demo(chat: ChatTongyi) -> None:
    """
    演示简单的对话场景：只有 SystemMessage 和 HumanMessage。
    """
    print("=" * 80)
    print("【示例2】简单对话：SystemMessage + HumanMessage")
    print("-" * 80)

    messages = [
        SystemMessage(content="你是一名专业的 Python 编程助手，擅长编写清晰、高效的代码。"),
        HumanMessage(content="请用 Python 写一个简单函数。"),
    ]

    print("消息列表：")
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, SystemMessage):
            print(f"  {i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"  {i}. [用户] {msg.content}")

    print("\n模型回复（流式输出）：")
    print("-" * 80)

    # 流式输出
    for chunk in chat.stream(input=messages):
        print(chunk.content, end="", flush=True)

    print("\n")
    print("-" * 80)
    print()


def multi_turn_conversation_demo(chat: ChatTongyi) -> None:
    """
    演示多轮对话：展示如何维护对话历史。
    """
    print("=" * 80)
    print("【示例3】多轮对话：维护对话历史")
    print("-" * 80)

    # 初始化对话历史
    messages = [
        SystemMessage(content="你是一个友好的聊天助手，喜欢用简洁、幽默的方式回答问题。"),
    ]

    # 第一轮对话
    print("--- 第一轮对话 ---")
    user_msg_1 = HumanMessage(content="你好，请简单介绍一下你自己。")
    messages.append(user_msg_1)
    print(f"[用户] {user_msg_1.content}")

    print("\n[AI] ", end="", flush=True)
    ai_response_1 = ""
    for chunk in chat.stream(input=messages):
        ai_response_1 += chunk.content
        print(chunk.content, end="", flush=True)
    messages.append(AIMessage(content=ai_response_1))
    print("\n")

    # 第二轮对话
    print("\n--- 第二轮对话 ---")
    user_msg_2 = HumanMessage(content="你能帮我做什么？简单回答")
    messages.append(user_msg_2)
    print(f"[用户] {user_msg_2.content}")

    print("\n[AI] ", end="", flush=True)
    for chunk in chat.stream(input=messages):
        print(chunk.content, end="", flush=True)
    print("\n")

    print("-" * 80)
    print()


def main() -> None:
    """
    主函数：演示如何使用 LangChain 调用通义聊天模型。
    """
    print("=" * 80)
    print("LangChain 通义聊天模型（ChatTongyi）示例")
    print("=" * 80)
    print()

    chat = init_chat_model()

    # 示例1：SystemMessage + HumanMessage + AIMessage 多轮对话
    chat_with_system_message_demo(chat)

    # 示例2：简单对话
    simple_chat_demo(chat)

    # 示例3：多轮对话维护历史
    multi_turn_conversation_demo(chat)

    print("=" * 80)
    print("演示结束")
    print("=" * 80)


if __name__ == "__main__":
    main()
