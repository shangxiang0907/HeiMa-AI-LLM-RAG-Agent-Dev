"""
LangChain 长期会话记忆示例

本示例对应课件中关于长期会话记忆的图片，重点演示：

1. 问题场景：如何实现持久化会话记录
   - InMemoryChatMessageHistory 是临时存储，程序重启后丢失
   - 生产环境需要持久化存储，可以使用文件存储或数据库存储
   - 本示例演示基于文件存储的长期会话记忆实现

2. 核心组件：
   - FileChatMessageHistory：自定义的基于文件存储的会话历史记录类
   - BaseChatMessageHistory：LangChain 提供的基类，需要继承并实现三个方法
   - messages_from_dict / message_to_dict：消息序列化和反序列化工具

3. 实现方法：
   - add_messages：同步模式，添加消息到文件
   - messages：同步模式，从文件读取消息
   - clear：同步模式，清除文件中的消息

核心概念：
- FileChatMessageHistory：基于文件存储的持久化历史记录存储
- BaseChatMessageHistory：会话历史记录的基类
- session_id：会话标识符，用作文件名
- 长期会话记忆：存储在文件中，程序重启后仍然保留
"""

import json
import os
from typing import Any, Dict, Sequence

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    基于文件存储的会话历史记录类。

    核心思路：
    - 基于文件存储会话记录，以 session_id 为文件名
    - 不同 session_id 有不同文件存储消息
    - 继承 BaseChatMessageHistory 实现三个方法：
      1. add_messages: 同步模式，添加消息
      2. messages: 同步模式，获取消息
      3. clear: 同步模式，清除消息

    属性:
        storage_path: 存储路径，会话文件将保存在此目录下
        session_id: 会话ID，用作文件名
    """

    def __init__(self, storage_path: str, session_id: str):
        """
        初始化 FileChatMessageHistory 实例。

        参数:
            storage_path: 存储路径，会话文件将保存在此目录下
            session_id: 会话ID，用作文件名
        """
        self.storage_path = storage_path
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        """
        从文件中读取消息列表。

        返回:
            消息列表，如果文件不存在则返回空列表
        """
        file_path = os.path.join(self.storage_path, self.session_id)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        添加消息到文件。

        参数:
            messages: 要添加的消息序列
        """
        # 获取现有消息
        all_messages = list(self.messages)
        # 添加新消息
        all_messages.extend(messages)
        # 序列化消息
        serialized = [message_to_dict(message) for message in all_messages]
        # 确保目录存在
        os.makedirs(self.storage_path, exist_ok=True)
        # 构建文件路径
        file_path = os.path.join(self.storage_path, self.session_id)
        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """
        清除文件中的消息。
        """
        # 确保目录存在
        os.makedirs(self.storage_path, exist_ok=True)
        # 构建文件路径
        file_path = os.path.join(self.storage_path, self.session_id)
        # 写入空列表
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


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


def demo_permanent_session_memory_introduction() -> None:
    """
    介绍长期会话记忆的基本概念和作用。

    说明：
    1. 什么是长期会话记忆
    2. 为什么需要长期会话记忆
    3. 如何实现长期会话记忆
    """
    print("=" * 80)
    print("【示例一】长期会话记忆基本介绍")
    print("=" * 80)

    print("\n1. 什么是长期会话记忆：")
    print("   - 临时会话记忆（InMemoryChatMessageHistory）存储在内存中，")
    print("     程序重启后历史记录会丢失")
    print("   - 长期会话记忆（FileChatMessageHistory）存储在文件中，")
    print("     程序重启后历史记录仍然保留")

    print("\n2. 为什么需要长期会话记忆：")
    print("   - 生产环境需要持久化存储会话记录")
    print("   - 用户希望在不同时间、不同设备上继续之前的对话")
    print("   - 需要跨进程、跨服务器共享会话记录")

    print("\n3. 核心组件：")
    print("   - BaseChatMessageHistory：LangChain 提供的基类")
    print("   - FileChatMessageHistory：自定义的基于文件存储的实现")
    print("   - messages_from_dict / message_to_dict：消息序列化工具")

    print("\n4. 实现方法：")
    print("   - 继承 BaseChatMessageHistory 实现三个方法：")
    print("     * add_messages: 同步模式，添加消息到文件")
    print("     * messages: 同步模式，从文件读取消息")
    print("     * clear: 同步模式，清除文件中的消息")
    print("   - 以 session_id 为文件名，不同会话有不同文件")

    print()


def demo_file_chat_message_history_class() -> None:
    """
    演示 FileChatMessageHistory 类的实现。

    展示：
    1. 类的继承关系
    2. 三个核心方法的实现
    3. 文件存储机制
    """
    print("=" * 80)
    print("【示例二】FileChatMessageHistory 类实现")
    print("=" * 80)

    print("\n1. 类的继承关系：")
    print("   class FileChatMessageHistory(BaseChatMessageHistory):")
    print("       - 继承自 BaseChatMessageHistory")
    print("       - BaseChatMessageHistory 是 LangChain 提供的基类")
    print("       - 需要实现三个方法：add_messages, messages, clear")

    print("\n2. 类的属性：")
    print("   - storage_path: 存储路径，会话文件将保存在此目录下")
    print("   - session_id: 会话ID，用作文件名")

    print("\n3. messages 方法（读取消息）：")
    print("   @property")
    print("   def messages(self) -> list[BaseMessage]:")
    print("       - 从文件中读取消息列表")
    print("       - 文件路径：os.path.join(self.storage_path, self.session_id)")
    print("       - 使用 json.load 读取文件内容")
    print("       - 使用 messages_from_dict 将字典转换为消息对象")
    print("       - 如果文件不存在，返回空列表")

    print("\n4. add_messages 方法（添加消息）：")
    print("   def add_messages(self, messages: Sequence[BaseMessage]):")
    print("       - 获取现有消息：all_messages = list(self.messages)")
    print("       - 添加新消息：all_messages.extend(messages)")
    print("       - 序列化消息：message_to_dict(message)")
    print("       - 确保目录存在：os.makedirs(os.path.dirname(file_path), exist_ok=True)")
    print("       - 写入文件：json.dump(serialized, f)")

    print("\n5. clear 方法（清除消息）：")
    print("   def clear(self):")
    print("       - 构建文件路径")
    print("       - 确保目录存在")
    print("       - 写入空列表：json.dump([], f)")

    print()


def demo_file_storage_mechanism() -> None:
    """
    演示文件存储机制。

    展示：
    1. 文件存储路径结构
    2. 不同 session_id 对应不同文件
    3. 文件内容的 JSON 格式
    """
    print("=" * 80)
    print("【示例三】文件存储机制")
    print("=" * 80)

    # 创建存储目录
    storage_path = "./chat_history"
    os.makedirs(storage_path, exist_ok=True)

    print("\n1. 存储路径结构：")
    print(f"   storage_path = '{storage_path}'")
    print("   会话文件将保存在此目录下，文件名就是 session_id")

    # 创建两个不同会话的历史记录
    history1 = FileChatMessageHistory(storage_path, "user_001")
    history2 = FileChatMessageHistory(storage_path, "user_002")

    print("\n2. 不同 session_id 对应不同文件：")
    print(f"   history1 = FileChatMessageHistory('{storage_path}', 'user_001')")
    print(f"   文件路径：{os.path.join(storage_path, 'user_001')}")
    print(f"   history2 = FileChatMessageHistory('{storage_path}', 'user_002')")
    print(f"   文件路径：{os.path.join(storage_path, 'user_002')}")

    # 添加一些消息
    from langchain_core.messages import HumanMessage, AIMessage

    history1.add_messages([HumanMessage(content="你好")])
    history2.add_messages([HumanMessage(content="Hello")])

    print("\n3. 文件内容格式：")
    print("   文件内容为 JSON 格式，包含消息的序列化数据")
    file_path_1 = os.path.join(storage_path, "user_001")
    if os.path.exists(file_path_1):
        with open(file_path_1, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"\n   文件 {file_path_1} 的内容：")
            print("   " + content[:200] + "..." if len(content) > 200 else "   " + content)

    print("\n4. 读取消息：")
    messages1 = history1.messages
    messages2 = history2.messages
    print(f"   history1.messages: {len(messages1)} 条消息")
    print(f"   history2.messages: {len(messages2)} 条消息")
    print("   结论：不同 session_id 有独立的文件存储，互不干扰")

    print()


def demo_create_get_history_with_file_storage() -> None:
    """
    演示如何创建使用文件存储的 get_history 函数。

    展示：
    1. 历史记录存储字典的创建
    2. get_history 函数的实现
    3. FileChatMessageHistory 的使用
    """
    print("=" * 80)
    print("【示例四】创建使用文件存储的 get_history 函数")
    print("=" * 80)

    storage_path = "./chat_history"
    os.makedirs(storage_path, exist_ok=True)

    print("\n1. 历史记录存储字典：")
    print("   chat_history_store = {}")
    print("   作用：存放多个会话ID所对应的历史会话记录")
    chat_history_store: Dict[str, FileChatMessageHistory] = {}

    print("\n2. get_history 函数的要求：")
    print("   - 函数传入为会话ID（字符串类型）")
    print("   - 函数要求返回 BaseChatMessageHistory 的子类")
    print("   - FileChatMessageHistory 是我们自定义的基于文件存储的类")

    def get_history(session_id: str) -> FileChatMessageHistory:
        """
        获取指定会话ID的历史会话记录函数。

        参数:
            session_id: 会话ID（字符串类型）

        返回:
            FileChatMessageHistory 实例，用于存储该会话的历史记录
        """
        if session_id not in chat_history_store:
            chat_history_store[session_id] = FileChatMessageHistory(
                storage_path, session_id
            )
        return chat_history_store[session_id]

    print("\n3. get_history 函数实现：")
    print("   def get_history(session_id: str):")
    print("       if session_id not in chat_history_store:")
    print("           chat_history_store[session_id] = FileChatMessageHistory(")
    print("               storage_path, session_id")
    print("           )")
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


def demo_conversation_chain_with_file_history() -> None:
    """
    演示使用 FileChatMessageHistory 创建带历史记录的对话链。

    这是完整的示例：
    1. 创建基础链
    2. 创建使用文件存储的 get_history 函数
    3. 使用 RunnableWithMessageHistory 创建带历史记录的链
    4. 演示多轮对话
    5. 演示程序重启后历史记录仍然保留
    """
    print("=" * 80)
    print("【示例五】使用 FileChatMessageHistory 创建带历史记录的对话链（完整示例）")
    print("=" * 80)

    # 创建存储目录
    storage_path = "./chat_history"
    os.makedirs(storage_path, exist_ok=True)

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
    chat_history_store: Dict[str, FileChatMessageHistory] = {}

    # 创建 get_history 函数
    def get_history(session_id: str) -> FileChatMessageHistory:
        """获取指定会话ID的历史会话记录函数。"""
        if session_id not in chat_history_store:
            chat_history_store[session_id] = FileChatMessageHistory(
                storage_path, session_id
            )
        return chat_history_store[session_id]

    print("\n4. 创建历史记录存储和 get_history 函数：")
    print(f"   storage_path = '{storage_path}'")
    print("   chat_history_store = {}")
    print("   def get_history(session_id: str): ...")
    print("   使用 FileChatMessageHistory 替代 InMemoryChatMessageHistory")

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
    res1 = conversation_chain.invoke({"input": "小明有一只猫"}, session_config)
    print(f"\n✅ 第一轮对话结果：")
    print(res1)
    print("-" * 80)

    print("\n【第二轮对话】")
    print("输入：'小刚有两只狗'")
    print("-" * 80)
    res2 = conversation_chain.invoke({"input": "小刚有两只狗"}, session_config)
    print(f"\n✅ 第二轮对话结果：")
    print(res2)
    print("-" * 80)

    print("\n【第三轮对话】")
    print("输入：'共有几只宠物？'")
    print("-" * 80)
    res3 = conversation_chain.invoke({"input": "共有几只宠物？"}, session_config)
    print(f"\n✅ 第三轮对话结果：")
    print(res3)
    print("=" * 80)

    print("\n结论：")
    print("- 模型能够记住之前的对话内容")
    print("- 在第三轮对话中，模型知道小明有一只猫，小刚有两只狗")
    print("- 因此能够正确回答'共有几只宠物？'的问题")
    print("- 历史记录已保存到文件中，程序重启后仍然保留")

    print()


def demo_persistence_after_restart() -> None:
    """
    演示程序重启后历史记录仍然保留。

    展示：
    1. 第一次运行：创建会话并添加消息
    2. 模拟程序重启：重新创建 FileChatMessageHistory 实例
    3. 验证历史记录仍然存在
    """
    print("=" * 80)
    print("【示例六】程序重启后历史记录仍然保留")
    print("=" * 80)

    storage_path = "./chat_history"
    os.makedirs(storage_path, exist_ok=True)
    session_id = "user_persistence_test"

    print("\n1. 第一次运行：创建会话并添加消息")
    print("=" * 80)

    # 第一次运行：创建历史记录并添加消息
    history1 = FileChatMessageHistory(storage_path, session_id)
    from langchain_core.messages import HumanMessage, AIMessage

    history1.add_messages(
        [
            HumanMessage(content="我的名字是张三"),
            AIMessage(content="你好，张三！很高兴认识你。"),
            HumanMessage(content="我喜欢编程"),
        ]
    )

    print(f"   创建 FileChatMessageHistory(storage_path, '{session_id}')")
    print("   添加了 3 条消息")
    print(f"   文件路径：{os.path.join(storage_path, session_id)}")
    print(f"   当前消息数量：{len(history1.messages)}")

    print("\n2. 模拟程序重启：重新创建 FileChatMessageHistory 实例")
    print("=" * 80)

    # 模拟程序重启：重新创建实例（不保留内存中的引用）
    history2 = FileChatMessageHistory(storage_path, session_id)

    print(f"   重新创建 FileChatMessageHistory(storage_path, '{session_id}')")
    print("   这是新的实例，内存中没有任何消息")

    print("\n3. 验证历史记录仍然存在")
    print("=" * 80)

    messages = history2.messages
    print(f"   从文件读取消息数量：{len(messages)}")
    print("   消息内容：")
    for i, msg in enumerate(messages, 1):
        print(f"   {i}. {msg.type}: {msg.content}")

    print("\n结论：")
    print("- 即使程序重启，历史记录仍然保存在文件中")
    print("- 重新创建 FileChatMessageHistory 实例后，可以从文件读取之前的消息")
    print("- 这就是长期会话记忆的核心优势：数据持久化")

    print()


def demo_clear_history() -> None:
    """
    演示清除历史记录功能。

    展示：
    1. 添加消息
    2. 清除消息
    3. 验证消息已被清除
    """
    print("=" * 80)
    print("【示例七】清除历史记录功能")
    print("=" * 80)

    storage_path = "./chat_history"
    os.makedirs(storage_path, exist_ok=True)
    session_id = "user_clear_test"

    history = FileChatMessageHistory(storage_path, session_id)

    print("\n1. 添加消息：")
    from langchain_core.messages import HumanMessage

    history.add_messages([HumanMessage(content="测试消息1")])
    history.add_messages([HumanMessage(content="测试消息2")])
    print(f"   添加了 2 条消息")
    print(f"   当前消息数量：{len(history.messages)}")

    print("\n2. 清除消息：")
    history.clear()
    print("   调用 history.clear()")

    print("\n3. 验证消息已被清除：")
    messages = history.messages
    print(f"   当前消息数量：{len(messages)}")
    print("   结论：clear() 方法成功清除了文件中的消息")

    print()


def demo_temporary_vs_permanent_storage() -> None:
    """
    演示临时存储 vs 长期存储的对比。

    说明：
    1. InMemoryChatMessageHistory 的特点
    2. FileChatMessageHistory 的特点
    3. 使用场景建议
    """
    print("=" * 80)
    print("【示例八】临时存储 vs 长期存储的对比")
    print("=" * 80)

    print("\n1. InMemoryChatMessageHistory（临时存储）：")
    print("   优点：")
    print("   - 简单、快速、无需配置")
    print("   - 无需管理文件系统")
    print("   缺点：")
    print("   - 程序重启后历史记录会丢失")
    print("   - 无法跨进程共享")
    print("   - 不适合生产环境")
    print("   适用场景：")
    print("   - 开发和测试")
    print("   - 临时会话，不需要持久化")

    print("\n2. FileChatMessageHistory（长期存储）：")
    print("   优点：")
    print("   - 数据持久化，程序重启后仍然保留")
    print("   - 可以跨进程共享（通过文件系统）")
    print("   - 适合生产环境")
    print("   - 实现简单，无需数据库")
    print("   缺点：")
    print("   - 需要管理文件系统")
    print("   - 大量会话时文件管理复杂")
    print("   - 不适合分布式系统（需要使用数据库）")
    print("   适用场景：")
    print("   - 生产环境（单机或小规模）")
    print("   - 需要持久化会话记录")
    print("   - 不需要复杂查询的场景")

    print("\n3. 其他持久化存储方案：")
    print("   - PostgreSQLChatMessageHistory：基于 PostgreSQL 数据库")
    print("   - RedisChatMessageHistory：基于 Redis 缓存")
    print("   - MongoDBChatMessageHistory：基于 MongoDB 数据库")
    print("   适用场景：")
    print("   - 大规模分布式系统")
    print("   - 需要复杂查询")
    print("   - 需要高可用性")

    print("\n4. 使用建议：")
    print("   - 开发和测试：使用 InMemoryChatMessageHistory")
    print("   - 生产环境（单机）：使用 FileChatMessageHistory")
    print("   - 生产环境（分布式）：使用数据库存储（PostgreSQL/Redis/MongoDB）")

    print()


def main() -> None:
    """
    入口函数：演示长期会话记忆的用法和重要性。

    本示例分为八个部分：
    1. 长期会话记忆基本介绍
    2. FileChatMessageHistory 类实现
    3. 文件存储机制
    4. 创建使用文件存储的 get_history 函数
    5. 使用 FileChatMessageHistory 创建带历史记录的对话链（完整示例）
    6. 程序重启后历史记录仍然保留
    7. 清除历史记录功能
    8. 临时存储 vs 长期存储的对比
    """
    print("=" * 80)
    print("LangChain 长期会话记忆示例")
    print("=" * 80)
    print()

    # 示例一：长期会话记忆基本介绍
    demo_permanent_session_memory_introduction()

    # 示例二：FileChatMessageHistory 类实现
    demo_file_chat_message_history_class()

    # 示例三：文件存储机制
    demo_file_storage_mechanism()

    # 示例四：创建使用文件存储的 get_history 函数
    demo_create_get_history_with_file_storage()

    # 示例五：使用 FileChatMessageHistory 创建带历史记录的对话链（完整示例）
    demo_conversation_chain_with_file_history()

    # 示例六：程序重启后历史记录仍然保留
    demo_persistence_after_restart()

    # 示例七：清除历史记录功能
    demo_clear_history()

    # 示例八：临时存储 vs 长期存储的对比
    demo_temporary_vs_permanent_storage()

    print("=" * 80)
    print("全部示例执行完毕。")
    print("=" * 80)
    print("\n总结：")
    print("1. FileChatMessageHistory 继承 BaseChatMessageHistory，实现三个方法")
    print("2. 基于文件存储会话记录，以 session_id 为文件名")
    print("3. 不同 session_id 有不同文件存储消息")
    print("4. 数据持久化，程序重启后历史记录仍然保留")
    print("5. 适合生产环境（单机或小规模），不适合大规模分布式系统")
    print("=" * 80)


if __name__ == "__main__":
    main()
