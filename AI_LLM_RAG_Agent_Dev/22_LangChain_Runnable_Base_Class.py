"""
LangChain Runnable 抽象基类与 __or__ 运算符重写示例

本示例对应课件中关于 Runnable 基类和 __or__ 方法的图片，重点演示：

1. LangChain 中的绝大多数核心组件都继承了 Runnable 抽象基类（位于 langchain_core.runnables.base）
2. 使用「|」运算符（如 chain = prompt | model）时，chain 变量是 RunnableSequence 类型
3. 这是因为 Runnable 基类内部对 __or__ 魔术方法进行了改写
4. 继续使用「|」添加新组件，依旧会得到 RunnableSequence，这就是链的基础架构

核心概念：
- Runnable：LangChain 中所有可运行组件的抽象基类
- RunnableSequence：通过 __or__ 方法创建的序列对象，用于链式调用
- __or__：Python 的位或运算符重写，在 LangChain 中用于组合组件
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.base import Runnable, RunnableSerializable


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


def demo_runnable_inheritance() -> None:
    """
    演示 LangChain 核心组件的继承关系。

    展示 ChatPromptTemplate 和 ChatTongyi 都继承自 Runnable 基类。
    """
    print("=" * 80)
    print("【示例一】Runnable 基类继承关系")
    print("=" * 80)

    # 创建 ChatPromptTemplate 实例
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个有用的 AI 助手。"),
            ("human", "{question}"),
        ]
    )

    # 创建 ChatTongyi 实例
    chat = init_chat_model()

    # 检查继承关系
    print("\n1. ChatPromptTemplate 的继承关系：")
    print(f"   - isinstance(prompt, Runnable): {isinstance(prompt, Runnable)}")
    print(f"   - isinstance(prompt, RunnableSerializable): {isinstance(prompt, RunnableSerializable)}")
    print(f"   - prompt 的 MRO（方法解析顺序）:")
    for i, cls in enumerate(prompt.__class__.__mro__[:5], 1):  # 只显示前5个
        print(f"      {i}. {cls.__name__}")

    print("\n2. ChatTongyi 的继承关系：")
    print(f"   - isinstance(chat, Runnable): {isinstance(chat, Runnable)}")
    print(f"   - isinstance(chat, RunnableSerializable): {isinstance(chat, RunnableSerializable)}")
    print(f"   - chat 的 MRO（方法解析顺序）:")
    for i, cls in enumerate(chat.__class__.__mro__[:5], 1):  # 只显示前5个
        print(f"      {i}. {cls.__name__}")

    print("\n结论：ChatPromptTemplate 和 ChatTongyi 都继承自 Runnable 基类，")
    print("因此它们都支持通过 __or__ 方法进行链式组合。")
    print()


def demo_or_operator_basic() -> None:
    """
    演示基本的「|」运算符使用，展示 chain = prompt | model 的工作原理。

    重点说明：
    - prompt | chat 会调用 prompt.__or__(chat)
    - 返回的是 RunnableSequence 类型
    - RunnableSequence 是 RunnableSerializable 的子类
    """
    print("=" * 80)
    print("【示例二】__or__ 运算符的基本使用：chain = prompt | model")
    print("=" * 80)

    # 创建组件
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个有用的 AI 助手。"),
            ("human", "请简单解释一下什么是 LangChain 中的 Runnable 基类。"),
        ]
    )
    chat = init_chat_model()

    # 使用「|」运算符组合组件
    chain: RunnableSerializable = prompt | chat

    print("\n1. chain 对象的类型：")
    print(f"   - type(chain): {type(chain)}")
    print(f"   - isinstance(chain, RunnableSequence): {isinstance(chain, RunnableSequence)}")
    print(f"   - isinstance(chain, RunnableSerializable): {isinstance(chain, RunnableSerializable)}")
    print(f"   - isinstance(chain, Runnable): {isinstance(chain, Runnable)}")

    print("\n2. RunnableSequence 的内部结构：")
    if isinstance(chain, RunnableSequence):
        print(f"   - chain.steps 的长度: {len(chain.steps)}")
        print(f"   - chain.steps[0] 的类型: {type(chain.steps[0])}")
        print(f"   - chain.steps[1] 的类型: {type(chain.steps[1])}")

    print("\n3. 执行 chain.invoke(...)：")
    result = chain.invoke({})
    print(f"   模型回复: {result.content[:100]}..." if len(result.content) > 100 else f"   模型回复: {result.content}")
    print()

    print("结论：prompt | chat 返回的是 RunnableSequence 类型，")
    print("这是因为 Runnable 基类内部对 __or__ 魔术方法进行了改写。")
    print()


def demo_or_operator_chaining() -> None:
    """
    演示链式使用「|」运算符，展示如何继续添加组件。

    重点说明：
    - (prompt | chat) | output_parser 会继续得到 RunnableSequence
    - 链式调用可以无限扩展
    - 每次使用「|」都会创建一个新的 RunnableSequence
    """
    print("=" * 80)
    print("【示例三】链式使用「|」运算符：继续添加组件")
    print("=" * 80)

    # 创建组件
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个有用的 AI 助手。"),
            ("human", "请用一句话回答：什么是 Runnable？"),
        ]
    )
    chat = init_chat_model()

    # 第一次组合
    chain1 = prompt | chat
    print("\n1. 第一次组合：chain1 = prompt | chat")
    print(f"   - chain1 的类型: {type(chain1)}")
    print(f"   - isinstance(chain1, RunnableSequence): {isinstance(chain1, RunnableSequence)}")

    # 继续添加组件（这里我们添加一个简单的输出处理）
    # 注意：在实际应用中，你可能会添加 output_parser 等组件
    # 这里为了演示，我们再次组合一个 prompt（仅作示例）
    prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个文本格式化助手。"),
            ("human", "请将以下内容格式化：{text}"),
        ]
    )

    # 第二次组合
    chain2 = chain1 | prompt2
    print("\n2. 第二次组合：chain2 = chain1 | prompt2")
    print(f"   - chain2 的类型: {type(chain2)}")
    print(f"   - isinstance(chain2, RunnableSequence): {isinstance(chain2, RunnableSequence)}")

    if isinstance(chain2, RunnableSequence):
        print(f"   - chain2.steps 的长度: {len(chain2.steps)}")
        print(f"   - chain2.steps 中各组件的类型:")
        for i, step in enumerate(chain2.steps, 1):
            print(f"      {i}. {type(step).__name__}")

    print("\n结论：继续使用「|」添加新组件，依旧会得到 RunnableSequence，")
    print("这就是链的基础架构。")
    print()


def demo_or_operator_implementation() -> None:
    """
    演示 __or__ 方法的实现原理（模拟）。

    展示 Runnable 基类中 __or__ 方法的工作原理：
    - 接受另一个 Runnable 或可转换为 Runnable 的对象
    - 返回 RunnableSequence(steps=[self, coerce_to_runnable(other)])
    """
    print("=" * 80)
    print("【示例四】__or__ 方法的实现原理（模拟）")
    print("=" * 80)

    print("\nRunnable 基类中的 __or__ 方法签名（简化版）：")
    print("""
    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Iterator[Any]], Iterator[Other]]
        | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
    ) -> RunnableSerializable[Input, Other]:
        \"\"\"
        Runnable "or" operator.

        Compose this `Runnable` with another object to create a
        `RunnableSequence`.

        Args:
            other: Another `Runnable` or a `Runnable`-like object.

        Returns:
            A new `Runnable`.
        \"\"\"
        return RunnableSequence(steps=[self, coerce_to_runnable(other)])
    """)

    print("\n关键点：")
    print("1. __or__ 方法接受多种类型的 other 参数（Runnable、Callable、Mapping 等）")
    print("2. 通过 coerce_to_runnable(other) 将 other 转换为 Runnable 对象")
    print("3. 返回 RunnableSequence(steps=[self, coerce_to_runnable(other)])")
    print("4. 这样，prompt | chat 就创建了一个包含 prompt 和 chat 的序列")

    # 实际演示
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个有用的 AI 助手。"),
            ("human", "测试"),
        ]
    )
    chat = init_chat_model()

    chain = prompt | chat
    if isinstance(chain, RunnableSequence):
        print("\n实际验证：")
        print(f"   - chain.steps[0] 是 prompt: {chain.steps[0] is prompt}")
        print(f"   - chain.steps[1] 是 chat: {chain.steps[1] is chat}")
        print(f"   - chain.steps 的长度: {len(chain.steps)}")

    print()


def main() -> None:
    """
    入口函数：演示 Runnable 基类和 __or__ 方法的工作原理。

    本示例分为四个部分：
    1. 展示核心组件的继承关系
    2. 展示基本的「|」运算符使用
    3. 展示链式使用「|」运算符
    4. 展示 __or__ 方法的实现原理
    """
    print("=" * 80)
    print("LangChain Runnable 抽象基类与 __or__ 运算符重写示例")
    print("=" * 80)
    print()

    # 示例一：Runnable 基类继承关系
    demo_runnable_inheritance()

    # 示例二：基本的「|」运算符使用
    demo_or_operator_basic()

    # 示例三：链式使用「|」运算符
    demo_or_operator_chaining()

    # 示例四：__or__ 方法的实现原理
    demo_or_operator_implementation()

    print("=" * 80)
    print("全部示例执行完毕。")
    print("=" * 80)
    print("\n总结：")
    print("1. LangChain 中的绝大多数核心组件都继承了 Runnable 抽象基类")
    print("2. chain = prompt | model 返回的是 RunnableSequence 类型")
    print("3. 这是因为 Runnable 基类内部对 __or__ 魔术方法进行了改写")
    print("4. 继续使用「|」添加新组件，依旧会得到 RunnableSequence，这就是链的基础架构")
    print("=" * 80)


if __name__ == "__main__":
    main()
