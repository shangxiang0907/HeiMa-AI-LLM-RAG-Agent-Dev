"""
LangChain 聊天提示词模板（ChatPromptTemplate）示例

本示例演示如何使用 ChatPromptTemplate 和 MessagesPlaceholder 来动态注入历史会话信息。

核心概念：
- 历史会话信息并不是静态的（固定的），而是随着对话的进行不停地积攒，即动态的
- 所以，历史会话信息需要支持动态注入
- MessagesPlaceholder 作为占位符，提供 history 作为占位的 key
- 基于 invoke 动态注入历史会话记录
- 必须是 invoke，format 无法注入

关键点：
1. ChatPromptTemplate：用于构建聊天提示词模板
2. MessagesPlaceholder：用于占位历史会话消息列表
3. invoke 方法：动态注入历史会话记录（format 方法不支持）
4. 历史会话数据：使用元组格式 (role, content) 或消息对象格式
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


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


def build_chat_prompt_template() -> ChatPromptTemplate:
    """
    构建一个包含 MessagesPlaceholder 的 ChatPromptTemplate。

    这个模板包含：
    - system：系统消息
    - ai：AI 消息（可选）
    - MessagesPlaceholder("history")：历史会话占位符（关键！）
    - human：用户消息

    注意：MessagesPlaceholder 必须使用 invoke 方法才能注入数据，format 方法不支持。
    """
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个友好的聊天助手，擅长用简洁、幽默的方式回答问题。"),
            ("ai", "你好！我是你的AI助手，很高兴为你服务。"),
            MessagesPlaceholder("history"),  # 这是关键：历史会话占位符
            ("human", "{input}"),  # 当前用户输入
        ]
    )
    return chat_template


def demo_basic_chat_prompt_template(chat: ChatTongyi) -> None:
    """
    演示基本的 ChatPromptTemplate 使用，不包含历史会话。
    """
    print("=" * 80)
    print("【示例1】基本的 ChatPromptTemplate 使用（无历史会话）")
    print("=" * 80)

    # 构建简单的聊天提示词模板（不包含 MessagesPlaceholder）
    simple_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个专业的 Python 编程助手。"),
            ("human", "{question}"),
        ]
    )

    # 使用 invoke 方法生成提示词
    prompt_value = simple_template.invoke({"question": "请用 Python 写一个简单的函数"})
    messages = prompt_value.to_messages()

    print("\n生成的提示词消息列表：")
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, SystemMessage):
            print(f"  {i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"  {i}. [用户] {msg.content}")

    print("\n模型回复（流式输出）：")
    print("-" * 80)
    for chunk in chat.stream(input=messages):
        print(chunk.content, end="", flush=True)
    print("\n")
    print("-" * 80)
    print()


def demo_messages_placeholder_with_invoke(chat: ChatTongyi) -> None:
    """
    演示使用 MessagesPlaceholder 和 invoke 方法动态注入历史会话记录。

    这是图片中展示的核心示例。
    """
    print("=" * 80)
    print("【示例2】使用 MessagesPlaceholder 和 invoke 动态注入历史会话")
    print("=" * 80)

    # 构建包含 MessagesPlaceholder 的聊天提示词模板
    chat_template = build_chat_prompt_template()

    # 准备历史会话数据（使用元组格式，与图片中的示例一致）
    history_data = [
        ("human", "你好，请介绍一下你自己。"),
        ("ai", "你好！我是你的AI助手，很高兴为你服务。我可以帮你回答问题、提供建议等。"),
        ("human", "你能帮我做什么？"),
        ("ai", "我可以帮你回答问题、提供信息、协助写作、分析问题等。有什么需要帮助的吗？"),
    ]

    print("\n历史会话数据：")
    for i, (role, content) in enumerate(history_data, 1):
        role_map = {"human": "用户", "ai": "AI"}
        print(f"  {i}. [{role_map.get(role, role)}] {content}")

    # 使用 invoke 方法动态注入历史会话记录
    # 注意：必须是 invoke，format 无法注入 MessagesPlaceholder
    print("\n使用 invoke 方法动态注入历史会话...")
    prompt_value = chat_template.invoke({"history": history_data, "input": "请总结一下我们刚才的对话"})

    # 将 PromptValue 转换为消息列表
    messages = prompt_value.to_messages()

    print("\n生成的完整提示词消息列表：")
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, SystemMessage):
            print(f"  {i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"  {i}. [用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"  {i}. [AI] {msg.content}")

    print("\n模型回复（流式输出）：")
    print("-" * 80)
    for chunk in chat.stream(input=messages):
        print(chunk.content, end="", flush=True)
    print("\n")
    print("-" * 80)
    print()


def demo_dynamic_history_injection(chat: ChatTongyi) -> None:
    """
    演示动态历史会话注入：展示历史会话如何随着对话进行而累积。

    这个示例模拟了多轮对话的场景，每次对话都会将新的消息添加到历史中。
    """
    print("=" * 80)
    print("【示例3】动态历史会话注入：模拟多轮对话")
    print("=" * 80)

    chat_template = build_chat_prompt_template()

    # 初始化历史会话（空列表）
    history_data = []

    # 第一轮对话
    print("\n--- 第一轮对话 ---")
    user_input_1 = "你好，请介绍一下你自己。"
    print(f"[用户] {user_input_1}")

    prompt_value_1 = chat_template.invoke({"history": history_data, "input": user_input_1})
    messages_1 = prompt_value_1.to_messages()

    print("\n[AI] ", end="", flush=True)
    ai_response_1 = ""
    for chunk in chat.stream(input=messages_1):
        ai_response_1 += chunk.content
        print(chunk.content, end="", flush=True)
    print("\n")

    # 将第一轮对话添加到历史中
    history_data.append(("human", user_input_1))
    history_data.append(("ai", ai_response_1))

    # 第二轮对话（历史会话已包含第一轮）
    print("\n--- 第二轮对话（历史会话已包含第一轮） ---")
    user_input_2 = "你能帮我做什么？"
    print(f"[用户] {user_input_2}")

    prompt_value_2 = chat_template.invoke({"history": history_data, "input": user_input_2})
    messages_2 = prompt_value_2.to_messages()

    print("\n[AI] ", end="", flush=True)
    ai_response_2 = ""
    for chunk in chat.stream(input=messages_2):
        ai_response_2 += chunk.content
        print(chunk.content, end="", flush=True)
    print("\n")

    # 将第二轮对话添加到历史中
    history_data.append(("human", user_input_2))
    history_data.append(("ai", ai_response_2))

    # 第三轮对话（历史会话已包含前两轮）
    print("\n--- 第三轮对话（历史会话已包含前两轮） ---")
    user_input_3 = "请总结一下我们刚才的对话"
    print(f"[用户] {user_input_3}")

    prompt_value_3 = chat_template.invoke({"history": history_data, "input": user_input_3})
    messages_3 = prompt_value_3.to_messages()

    print("\n[AI] ", end="", flush=True)
    for chunk in chat.stream(input=messages_3):
        print(chunk.content, end="", flush=True)
    print("\n")

    print("\n最终历史会话记录数：", len(history_data))
    print("-" * 80)
    print()


def demo_format_vs_invoke() -> None:
    """
    演示 format 与 invoke 的区别，强调 format 无法注入 MessagesPlaceholder。
    """
    print("=" * 80)
    print("【示例4】format vs invoke：为什么必须使用 invoke")
    print("=" * 80)

    chat_template = build_chat_prompt_template()
    history_data = [
        ("human", "你好"),
        ("ai", "你好！"),
    ]

    print("\n--- 使用 invoke 方法（正确方式） ---")
    try:
        prompt_value = chat_template.invoke({"history": history_data, "input": "测试"})
        messages = prompt_value.to_messages()
        print("✅ invoke 方法成功：生成了", len(messages), "条消息")
        print("   消息列表：")
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, SystemMessage):
                print(f"     {i}. [系统] {msg.content[:50]}...")
            elif isinstance(msg, HumanMessage):
                print(f"     {i}. [用户] {msg.content[:50]}...")
            elif isinstance(msg, AIMessage):
                print(f"     {i}. [AI] {msg.content[:50]}...")
    except Exception as e:
        print(f"❌ invoke 方法失败：{e}")

    print("\n--- 尝试使用 format 方法（错误方式） ---")
    try:
        # format 方法无法处理 MessagesPlaceholder
        result = chat_template.format(history=history_data, input="测试")
        print("✅ format 方法成功（但实际上不会正确处理 MessagesPlaceholder）")
        print("   结果类型：", type(result))
    except Exception as e:
        print(f"❌ format 方法失败：{e}")
        print("   原因：format 方法无法注入 MessagesPlaceholder 类型的数据")

    print("\n💡 关键要点：")
    print("   • MessagesPlaceholder 必须使用 invoke 方法才能注入数据")
    print("   • format 方法只能处理字符串占位符，无法处理 MessagesPlaceholder")
    print("   • 历史会话信息是动态的，需要动态注入，所以必须使用 invoke")
    print("-" * 80)
    print()


def demo_message_objects_vs_tuples(chat: ChatTongyi) -> None:
    """
    演示历史会话数据可以使用消息对象格式或元组格式。
    """
    print("=" * 80)
    print("【示例5】历史会话数据格式：消息对象 vs 元组")
    print("=" * 80)

    chat_template = build_chat_prompt_template()

    # 方式1：使用元组格式（与图片示例一致）
    print("\n--- 方式1：使用元组格式 (role, content) ---")
    history_tuples = [
        ("human", "你好"),
        ("ai", "你好！很高兴为你服务。"),
    ]

    prompt_value_1 = chat_template.invoke({"history": history_tuples, "input": "请介绍一下你自己"})
    messages_1 = prompt_value_1.to_messages()
    print("✅ 元组格式成功：生成了", len(messages_1), "条消息")

    # 方式2：使用消息对象格式
    print("\n--- 方式2：使用消息对象格式 ---")
    history_objects = [
        HumanMessage(content="你好"),
        AIMessage(content="你好！很高兴为你服务。"),
    ]

    prompt_value_2 = chat_template.invoke({"history": history_objects, "input": "请介绍一下你自己"})
    messages_2 = prompt_value_2.to_messages()
    print("✅ 消息对象格式成功：生成了", len(messages_2), "条消息")

    print("\n💡 两种格式都可以使用，功能等价：")
    print("   • 元组格式：更简洁，适合快速开发")
    print("   • 消息对象格式：类型安全，支持更多高级功能")
    print("-" * 80)
    print()


def main() -> None:
    """
    主函数：综合演示 ChatPromptTemplate 和 MessagesPlaceholder 的使用。
    """
    print("=" * 80)
    print("LangChain 聊天提示词模板（ChatPromptTemplate）示例")
    print("=" * 80)
    print()

    chat = init_chat_model()

    # 示例1：基本的 ChatPromptTemplate 使用
    demo_basic_chat_prompt_template(chat)

    # 示例2：使用 MessagesPlaceholder 和 invoke 动态注入历史会话（核心示例）
    demo_messages_placeholder_with_invoke(chat)

    # 示例3：动态历史会话注入：模拟多轮对话
    demo_dynamic_history_injection(chat)

    # 示例4：format vs invoke 的区别
    demo_format_vs_invoke()

    # 示例5：历史会话数据格式对比
    demo_message_objects_vs_tuples(chat)

    print("=" * 80)
    print("演示结束")
    print("=" * 80)
    print()
    print("📌 核心要点总结：")
    print("   1. 历史会话信息是动态的，需要动态注入")
    print("   2. MessagesPlaceholder 作为占位符，提供 history 作为占位的 key")
    print("   3. 基于 invoke 动态注入历史会话记录")
    print("   4. 必须是 invoke，format 无法注入 MessagesPlaceholder")
    print("=" * 80)


if __name__ == "__main__":
    main()
