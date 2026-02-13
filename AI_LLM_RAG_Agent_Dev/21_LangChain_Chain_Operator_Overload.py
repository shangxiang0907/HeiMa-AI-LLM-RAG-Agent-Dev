"""
运算符重写与 LangChain 中「|」链式调用的关系示例

本示例对应课件中关于「chain = chat_prompt_template | model」背后原理的图片，
分为两部分进行讲解：

1. 纯 Python 示例：通过重写 __or__ 方法，让 a | b | c 代码返回一个自定义的「序列对象」
   - a | b 会调用 a.__or__(b)
   - (a | b) | c 会继续调用 MySequence.__or__(c)
   - 自定义类只要实现 __or__，就可以控制「|」运算的行为
2. LangChain 示例：说明 ChatPromptTemplate 和 ChatTongyi 也是通过重写「|」来实现链式调用
   - chat_prompt_template | chat 本质是创建了一个 RunnableSequence（可运行的链）
   - chain.invoke(...) / chain.stream(...) 就是对这个「链对象」发起调用
"""

from __future__ import annotations

import os
from typing import Any, Iterable, List

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSerializable


# =========================
# 一、纯 Python 运算符重写示例
# =========================


class Test(object):
    """
    课件中左上角的 Test 类示例：

    - 每个 Test 实例带有一个 name，方便打印区分
    - 重写 __str__，打印时更直观
    - 重写 __or__，让「a | b」得到一个自定义序列 MySequence
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:  # pragma: no cover - 简单打印逻辑
        return f"Test({self.name})"

    def __repr__(self) -> str:  # pragma: no cover - 简单打印逻辑
        return str(self)

    def __or__(self, other: Any) -> "MySequence":
        """
        关键点：重写「|」运算符对应的魔法方法 __or__。

        当执行 a | b 时，本质就是在调用：a.__or__(b)
        这里我们让它返回一个 MySequence 对象，把 a 和 b 都放进去。
        """
        return MySequence(self, other)


class MySequence(object):
    """
    课件中左下角的 MySequence 类示例：

    - __init__ 接受任意多个元素，统一存入内部列表 self.sequence
    - 再次重写 __or__，使得「(a | b) | c」时，会把 c 继续追加到同一个序列中
    - 提供 run 方法，按照顺序依次输出其中的元素
    """

    def __init__(self, *args: Any) -> None:
        self.sequence: List[Any] = []
        for arg in args:
            self.sequence.append(arg)

    def __or__(self, other: Any) -> MySequence:
        """
        关键点：让 MySequence 自己也支持「|」。

        如：
            d = a | b | c

        解析顺序为：
            tmp = a.__or__(b)           -> 得到 MySequence(a, b)
            d = tmp.__or__(c)           -> 在 sequence 中追加 c，并返回 self

        注意：这里返回类型不需要加引号，因为 MySequence 类已经在上面定义好了。
        但第50行的 Test.__or__ 必须加引号，因为 MySequence 在 Test 之后才定义（前向引用）。
        """
        self.sequence.append(other)
        # 返回 self 而不是新对象的原因：
        # 1. 性能：链式调用 a | b | c | d 时，返回 self 只修改一次，避免创建多个中间对象
        # 2. 内存：不产生临时对象，内存占用更小
        # 3. 语义：MySequence 是可变对象，sequence | other 表示"向序列追加元素"，修改自身更直观
        # 4. 一致性：类似 list.append() 的原地修改模式
        return self

    def run(self) -> None:
        """
        依次输出内部保存的元素，方便观察最终的执行顺序。
        """
        print("MySequence.run() 输出内容：")
        for item in self.sequence:
            print(item)


def demo_pure_operator_overload() -> None:
    """
    演示「a | b | c」是如何通过 __or__ 魔法方法一步步计算得到的。
    """
    print("=" * 80)
    print("【示例一】纯 Python 运算符重写：a | b | c")
    print("=" * 80)

    a = Test("a")
    b = Test("b")
    c = Test("c")

    d = a | b | c

    print("\n变量 d 的类型：", type(d))
    print("调用 d.run()，依次输出内部保存的元素：\n")
    d.run()
    print("\n可以看到，a、b、c 按照『管道』顺序被依次保存并输出。")
    print()


# =========================
# 二、LangChain 中的「|」示例
# =========================


def init_chat_model() -> ChatTongyi:
    """
    初始化 ChatTongyi 聊天模型实例。

    与 20 号示例保持一致：
    - 优先从 DASHSCOPE_API_KEY / API_KEY 环境变量中读取密钥
    - 模型使用 qwen3-max
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


def build_demo_chat_prompt_template() -> ChatPromptTemplate:
    """
    构建一个简单的 ChatPromptTemplate，便于演示「|」运算符。
    """
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个解释型 AI 助手，善于用浅显的语言解释技术概念。",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "请用通俗的方式解释一下：在 Python / LangChain 里重写「|」运算符的作用。",
            ),
        ]
    )
    return chat_prompt_template


def demo_langchain_chain_operator(chat: ChatTongyi) -> None:
    """
    演示在 LangChain 中，「chat_prompt_template | chat」是如何生成一个链对象的。
    """
    print("=" * 80)
    print("【示例二】LangChain 运算符重写：chat_prompt_template | chat")
    print("=" * 80)

    chat_prompt_template = build_demo_chat_prompt_template()

    # 准备一个简单的「历史会话」，让模型更有上下文。
    history_data: Iterable[tuple[str, str]] = [
        ("human", "什么是运算符重写？"),
        ("ai", "就是通过重写诸如 __add__、__or__ 等魔法方法，自定义运算符行为。"),
    ]

    # 关键代码：通过「|」把 PromptTemplate 与 模型连接起来
    chain: RunnableSerializable = chat_prompt_template | chat

    print("\nchain 对象类型：", type(chain))
    print("chain 是否 RunnableSerializable 子类：", isinstance(chain, RunnableSerializable))

    print("\n调用 chain.invoke(...)，观察模型的解释：\n")
    res = chain.invoke({"history": history_data})
    print(res.content)
    print()


def main() -> None:
    """
    本文件入口：

    1. 先通过纯 Python 示例，理解「a | b | c」是如何依赖 __or__ 实现的
    2. 再通过 LangChain 示例，理解「chat_prompt_template | chat」背后的同样思想
    """
    # 示例一：纯 Python 运算符重写
    demo_pure_operator_overload()

    # 示例二：LangChain 中的链式调用背后也依赖运算符重写
    chat = init_chat_model()
    demo_langchain_chain_operator(chat)

    print("=" * 80)
    print("全部示例执行完毕。你现在可以把这段逻辑与课件上的图片对照理解。")
    print("=" * 80)


if __name__ == "__main__":
    main()

