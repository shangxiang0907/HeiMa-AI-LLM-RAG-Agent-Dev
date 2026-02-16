"""
LangChain 临时会话记忆示例

本示例对应课件中关于临时会话记忆的图片，重点演示：

1. 问题场景：如何封装历史记录
   - 除了自行维护历史消息外，也可以借助 LangChain 内置的历史记录附加功能
   - LangChain 提供了 History 功能，帮助模型在有历史记忆的情况下回答

2. 核心组件：
   - RunnableWithMessageHistory：在原有链的基础上创建带有历史记录功能的新链
   - InMemoryChatMessageHistory：为历史记录提供内存存储（临时用）
   - get_history 函数：获取指定会话ID的历史会话记录函数

3. 使用方式：
   - 基于 RunnableWithMessageHistory 在原有链的基础上创建带有历史记录功能的新链
   - 基于 InMemoryChatMessageHistory 为历史记录提供内存存储（临时用）
   - 通过 session_id 区分不同的会话，每个会话维护独立的历史记录

核心概念：
- RunnableWithMessageHistory：为链添加历史记录功能的包装器
- InMemoryChatMessageHistory：基于内存的临时历史记录存储
- session_id：会话标识符，用于区分不同的对话会话
- 临时会话记忆：存储在内存中，程序重启后丢失
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory


def init_chat_model() -> ChatTongyi:
    """
    初始化 ChatTongyi 聊天模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）

    与其他示例保持一致，使用 qwen3-max 作为聊天模型。
    """
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    os.environ["DASHSCOPE_API_KEY"] = api_key
    chat = ChatTongyi(model="qwen3-max")
    return chat


def print_prompt(full_prompt: Any) -> Any:
    """
    打印完整提示词，用于调试和可视化。

    参数:
        full_prompt: 完整的提示词对象

    返回:
        原样返回 full_prompt，方便在链中继续传递
    """
    print("=" * 20)
    print(str(full_prompt))
    print("=" * 20)
    return full_prompt


def demo_temporary_session_memory_introduction() -> None:
    """
    介绍临时会话记忆的基本概念和作用。

    说明：
    1. 什么是临时会话记忆
    2. 为什么需要临时会话记忆
    3. 如何使用临时会话记忆
    """
    print("=" * 80)
    print("【示例一】临时会话记忆基本介绍")
    print("=" * 80)

    print("\n1. 什么是临时会话记忆：")
    print("   - 如果想要封装历史记录，除了自行维护历史消息外，")
    print("     也可以借助 LangChain 内置的历史记录附加功能")
    print("   - LangChain 提供了 History 功能，帮助模型在有历史记忆的情况下回答")

    print("\n2. 为什么需要临时会话记忆：")
    print("   - 在多轮对话中，模型需要记住之前的对话内容")
    print("   - 例如：用户说'小明有一只猫'，然后说'小刚有两只狗'，")
    print("     最后问'共有几只宠物？'，模型需要记住前面的信息才能正确回答")

    print("\n3. 核心组件：")
    print("   - RunnableWithMessageHistory：在原有链的基础上创建带有历史记录功能的新链")
    print("   - InMemoryChatMessageHistory：为历史记录提供内存存储（临时用）")
    print("   - get_history 函数：获取指定会话ID的历史会话记录函数")

    print("\n4. 使用方式：")
    print("   - 基于 RunnableWithMessageHistory 创建带有历史记录功能的新链")
    print("   - 基于 InMemoryChatMessageHistory 为历史记录提供内存存储")
    print("   - 通过 session_id 区分不同的会话，每个会话维护独立的历史记录")

    print()


def demo_basic_chain_without_history() -> None:
    """
    演示没有历史记录的基础链。

    展示：
    1. 如何创建基础链
    2. 没有历史记录时的对话效果
    """
    print("=" * 80)
    print("【示例二】没有历史记录的基础链")
    print("=" * 80)

    # 创建模型实例
    model = init_chat_model()
    print("\n1. 创建模型实例：")
    print("   model = ChatTongyi(model='qwen3-max')")

    # 创建提示词模板
    prompt = PromptTemplate.from_template(
        "你需要根据对话历史回应用户问题。对话历史: {chat_history}。用户当前输入: {input}, 请给出回应"
    )

    print("\n2. 创建提示词模板：")
    print("   prompt = PromptTemplate.from_template(...)")
    print("   模板包含两个占位符：{chat_history} 和 {input}")

    # 创建基础链
    base_chain = prompt | print_prompt | model | StrOutputParser()

    print("\n3. 创建基础链：")
    print("   base_chain = prompt | print_prompt | model | StrOutputParser()")
    print("   注意：这个链没有历史记录功能，每次调用都是独立的")

    # 调用链（没有历史记录）
    print("\n4. 调用链（没有历史记录）：")
    print("   res = base_chain.invoke({'input': '小明有一只猫', 'chat_history': ''})")
    print("=" * 80)
    res1 = base_chain.invoke({"input": "小明有一只猫", "chat_history": ""})
    print(f"\n✅ 第一次调用结果：")
    print(res1)
    print("=" * 80)

    print("\n   注意：由于没有历史记录，每次调用都需要手动传入 chat_history")
    print("   而且模型无法记住之前的对话内容")

    print()


def demo_create_history_function() -> None:
    """
    演示如何创建 get_history 函数。

    展示：
    1. 历史记录存储字典的创建
    2. get_history 函数的实现
    3. InMemoryChatMessageHistory 的使用
    """
    print("=" * 80)
    print("【示例三】创建 get_history 函数")
    print("=" * 80)

    print("\n1. 历史记录存储字典：")
    print("   chat_history_store = {}")
    print("   作用：存放多个会话ID所对应的历史会话记录")
    chat_history_store: Dict[str, InMemoryChatMessageHistory] = {}

    print("\n2. get_history 函数的要求：")
    print("   - 函数传入为会话ID（字符串类型）")
    print("   - 函数要求返回 BaseChatMessageHistory 的子类")
    print("   - BaseChatMessageHistory 类专用于存放某个会话的历史记录")
    print("   - InMemoryChatMessageHistory 是官方自带的基于内存存放历史记录的类")

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        """
        获取指定会话ID的历史会话记录函数。

        参数:
            session_id: 会话ID（字符串类型）

        返回:
            InMemoryChatMessageHistory 实例，用于存储该会话的历史记录
        """
        if session_id not in chat_history_store:
            # 返回一个新的实例
            chat_history_store[session_id] = InMemoryChatMessageHistory()
        return chat_history_store[session_id]

    print("\n3. get_history 函数实现：")
    print("   def get_history(session_id: str):")
    print("       if session_id not in chat_history_store:")
    print("           chat_history_store[session_id] = InMemoryChatMessageHistory()")
    print("       return chat_history_store[session_id]")

    print("\n4. 测试 get_history 函数：")
    history1 = get_history("user_001")
    history2 = get_history("user_001")
    history3 = get_history("user_002")

    print(f"   get_history('user_001') 第一次调用：{id(history1)}")
    print(f"   get_history('user_001') 第二次调用：{id(history2)}")
    print(f"   结论：相同 session_id 返回同一个实例（id 相同）")

    print(f"\n   get_history('user_002') 调用：{id(history3)}")
    print(f"   结论：不同 session_id 返回不同的实例（id 不同）")

    print()


def demo_conversation_chain_with_history() -> None:
    """
    演示使用 RunnableWithMessageHistory 创建带历史记录的对话链。

    这是课件中展示的完整示例：
    1. 创建基础链
    2. 创建 get_history 函数
    3. 使用 RunnableWithMessageHistory 创建带历史记录的链
    4. 演示多轮对话
    """
    print("=" * 80)
    print("【示例四】使用 RunnableWithMessageHistory 创建带历史记录的对话链（完整示例）")
    print("=" * 80)

    # 创建模型实例
    model = init_chat_model()
    print("\n1. 创建模型实例：")
    print("   model = ChatTongyi(model='qwen3-max')")

    # 创建提示词模板
    prompt = PromptTemplate.from_template(
        "你需要根据对话历史回应用户问题。对话历史: {chat_history}。用户当前输入: {input}, 请给出回应"
    )

    print("\n2. 创建提示词模板：")
    print("   prompt = PromptTemplate.from_template(...)")
    print("   模板包含两个占位符：{chat_history} 和 {input}")

    # 创建基础链
    base_chain = prompt | print_prompt | model | StrOutputParser()

    print("\n3. 创建基础链：")
    print("   base_chain = prompt | print_prompt | model | StrOutputParser()")

    # 创建历史记录存储字典
    chat_history_store: Dict[str, InMemoryChatMessageHistory] = {}

    # 创建 get_history 函数
    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        """获取指定会话ID的历史会话记录函数。"""
        if session_id not in chat_history_store:
            chat_history_store[session_id] = InMemoryChatMessageHistory()
        return chat_history_store[session_id]

    print("\n4. 创建历史记录存储和 get_history 函数：")
    print("   chat_history_store = {}")
    print("   def get_history(session_id: str): ...")

    # 使用 RunnableWithMessageHistory 创建带历史记录的链
    conversation_chain = RunnableWithMessageHistory(
        base_chain,  # 被附加历史消息的 Runnable，通常是 chain
        get_history,  # 获取指定会话ID的历史会话的函数
        input_messages_key="input",  # 声明用户输入消息在模板中的占位符
        history_messages_key="chat_history",  # 声明历史消息在模板中的占位符
    )

    print("\n5. 使用 RunnableWithMessageHistory 创建带历史记录的链：")
    print("   conversation_chain = RunnableWithMessageHistory(")
    print("       base_chain,              # 被附加历史消息的 Runnable")
    print("       get_history,             # 获取历史会话的函数")
    print("       input_messages_key='input',        # 用户输入消息的占位符")
    print("       history_messages_key='chat_history'  # 历史消息的占位符")
    print("   )")

    # 配置会话ID
    session_config = {"configurable": {"session_id": "user_001"}}

    print("\n6. 配置会话ID：")
    print("   session_config = {'configurable': {'session_id': 'user_001'}}")
    print("   这是固定格式，用于配置当前会话的ID")

    # 演示多轮对话
    print("\n7. 演示多轮对话：")
    print("=" * 80)

    print("\n【第一轮对话】")
    print("输入：'小明有一只猫'")
    print("-" * 80)
    res1 = conversation_chain.invoke(
        {"input": "小明有一只猫"}, session_config
    )
    print(f"\n✅ 第一轮对话结果：")
    print(res1)
    print("-" * 80)

    print("\n【第二轮对话】")
    print("输入：'小刚有两只狗'")
    print("-" * 80)
    res2 = conversation_chain.invoke(
        {"input": "小刚有两只狗"}, session_config
    )
    print(f"\n✅ 第二轮对话结果：")
    print(res2)
    print("-" * 80)

    print("\n【第三轮对话】")
    print("输入：'共有几只宠物？'")
    print("-" * 80)
    res3 = conversation_chain.invoke(
        {"input": "共有几只宠物？"}, session_config
    )
    print(f"\n✅ 第三轮对话结果：")
    print(res3)
    print("=" * 80)

    print("\n结论：")
    print("- 模型能够记住之前的对话内容")
    print("- 在第三轮对话中，模型知道小明有一只猫，小刚有两只狗")
    print("- 因此能够正确回答'共有几只宠物？'的问题")

    print()


def demo_multiple_sessions() -> None:
    """
    演示多个会话的独立历史记录。

    展示：
    1. 不同 session_id 的会话有独立的历史记录
    2. 每个会话维护自己的对话上下文
    """
    print("=" * 80)
    print("【示例五】多个会话的独立历史记录")
    print("=" * 80)

    # 创建模型实例
    model = init_chat_model()

    # 创建提示词模板
    prompt = PromptTemplate.from_template(
        "你需要根据对话历史回应用户问题。对话历史: {chat_history}。用户当前输入: {input}, 请给出回应"
    )

    # 创建基础链
    base_chain = prompt | model | StrOutputParser()

    # 创建历史记录存储字典
    chat_history_store: Dict[str, InMemoryChatMessageHistory] = {}

    # 创建 get_history 函数
    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        """获取指定会话ID的历史会话记录函数。"""
        if session_id not in chat_history_store:
            chat_history_store[session_id] = InMemoryChatMessageHistory()
        return chat_history_store[session_id]

    # 使用 RunnableWithMessageHistory 创建带历史记录的链
    conversation_chain = RunnableWithMessageHistory(
        base_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("\n1. 说明：")
    print("   不同的 session_id 对应不同的会话，每个会话维护独立的历史记录")

    # 会话1：用户001
    print("\n2. 会话1（user_001）：")
    print("=" * 80)
    session_config_1 = {"configurable": {"session_id": "user_001"}}

    print("\n【第一轮】输入：'我喜欢吃苹果'")
    res1_1 = conversation_chain.invoke(
        {"input": "我喜欢吃苹果"}, session_config_1
    )
    print(f"结果：{res1_1[:100]}...")

    print("\n【第二轮】输入：'我还喜欢什么水果？'")
    res1_2 = conversation_chain.invoke(
        {"input": "我还喜欢什么水果？"}, session_config_1
    )
    print(f"结果：{res1_2[:100]}...")
    print("=" * 80)

    # 会话2：用户002
    print("\n3. 会话2（user_002）：")
    print("=" * 80)
    session_config_2 = {"configurable": {"session_id": "user_002"}}

    print("\n【第一轮】输入：'我喜欢打篮球'")
    res2_1 = conversation_chain.invoke(
        {"input": "我喜欢打篮球"}, session_config_2
    )
    print(f"结果：{res2_1[:100]}...")

    print("\n【第二轮】输入：'我还喜欢什么运动？'")
    res2_2 = conversation_chain.invoke(
        {"input": "我还喜欢什么运动？"}, session_config_2
    )
    print(f"结果：{res2_2[:100]}...")
    print("=" * 80)

    print("\n结论：")
    print("- 会话1（user_001）和会话2（user_002）有独立的历史记录")
    print("- 会话1的对话内容不会影响会话2")
    print("- 每个会话维护自己的对话上下文")

    print()


def demo_temporary_storage_limitation() -> None:
    """
    演示临时存储的限制。

    说明：
    1. InMemoryChatMessageHistory 是临时存储
    2. 程序重启后历史记录会丢失
    3. 适合开发和测试，不适合生产环境
    """
    print("=" * 80)
    print("【示例六】临时存储的限制")
    print("=" * 80)

    print("\n1. InMemoryChatMessageHistory 的特点：")
    print("   - 基于内存存储，速度快")
    print("   - 程序重启后历史记录会丢失")
    print("   - 适合开发和测试场景")
    print("   - 不适合生产环境（需要持久化存储）")

    print("\n2. 临时存储 vs 持久化存储：")
    print("   临时存储（InMemoryChatMessageHistory）：")
    print("   - 优点：简单、快速、无需配置")
    print("   - 缺点：程序重启后丢失，无法跨进程共享")

    print("\n   持久化存储（如 PostgreSQLChatMessageHistory）：")
    print("   - 优点：数据持久化，可跨进程共享")
    print("   - 缺点：需要配置数据库，速度相对较慢")

    print("\n3. 使用建议：")
    print("   - 开发和测试：使用 InMemoryChatMessageHistory")
    print("   - 生产环境：使用持久化存储（如数据库）")

    print()


def main() -> None:
    """
    入口函数：演示临时会话记忆的用法和重要性。

    本示例分为六个部分：
    1. 临时会话记忆基本介绍
    2. 没有历史记录的基础链
    3. 创建 get_history 函数
    4. 使用 RunnableWithMessageHistory 创建带历史记录的对话链（完整示例）
    5. 多个会话的独立历史记录
    6. 临时存储的限制
    """
    print("=" * 80)
    print("LangChain 临时会话记忆示例")
    print("=" * 80)
    print()

    # 示例一：临时会话记忆基本介绍
    demo_temporary_session_memory_introduction()

    # 示例二：没有历史记录的基础链
    demo_basic_chain_without_history()

    # 示例三：创建 get_history 函数
    demo_create_history_function()

    # 示例四：使用 RunnableWithMessageHistory 创建带历史记录的对话链（完整示例）
    demo_conversation_chain_with_history()

    # 示例五：多个会话的独立历史记录
    demo_multiple_sessions()

    # 示例六：临时存储的限制
    demo_temporary_storage_limitation()

    print("=" * 80)
    print("全部示例执行完毕。")
    print("=" * 80)
    print("\n总结：")
    print("1. RunnableWithMessageHistory 可以在原有链的基础上创建带有历史记录功能的新链")
    print("2. InMemoryChatMessageHistory 为历史记录提供内存存储（临时用）")
    print("3. get_history 函数用于获取指定会话ID的历史会话记录")
    print("4. 通过 session_id 区分不同的会话，每个会话维护独立的历史记录")
    print("5. 临时存储适合开发和测试，生产环境需要使用持久化存储")
    print("=" * 80)


if __name__ == "__main__":
    main()
