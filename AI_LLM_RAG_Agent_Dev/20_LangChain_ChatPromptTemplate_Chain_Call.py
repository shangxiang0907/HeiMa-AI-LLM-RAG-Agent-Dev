"""
LangChain 链式调用（ChatPromptTemplate | ChatTongyi 模型）示例

本示例对应课件中关于「链式调用」的图片，重点演示：

1. 使用 ChatPromptTemplate 构建提示词模板，并通过 MessagesPlaceholder 注入历史会话
2. 使用「|」运算符把提示词模板和聊天模型链接成一个 chain
3. chain 的类型为 RunnableSerializable，可通过 invoke / stream 触发执行
4. 上一个组件（ChatPromptTemplate）的输出，会作为下一个组件（模型）的输入
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSerializable


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

    # LangChain 的 ChatTongyi 封装会自动从环境变量中读取 key，
    # 这里设置一份到 DASHSCOPE_API_KEY，确保兼容性。
    os.environ["DASHSCOPE_API_KEY"] = api_key

    chat = ChatTongyi(model="qwen3-max")
    return chat


def build_poem_chat_prompt_template() -> ChatPromptTemplate:
    """
    构建一个与课件截图基本一致的 ChatPromptTemplate。

    - system：你是一个边塞诗人，可以作诗
    - MessagesPlaceholder("history")：历史会话占位符
    - human：请再来一首唐诗，无需额外输出
    """
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个边塞诗人，可以作诗。"),
            MessagesPlaceholder("history"),
            ("human", "请再来一首唐诗，无需额外输出"),
        ]
    )
    return chat_prompt_template


def demo_chain_invoke_and_stream(chat: ChatTongyi) -> None:
    """
    演示如何通过「|」把 ChatPromptTemplate 和模型链接成 chain，
    并分别使用 invoke / stream 触发执行。
    """
    print("=" * 80)
    print("【示例】ChatPromptTemplate | ChatTongyi 链式调用（invoke & stream）")
    print("=" * 80)

    # 1. 构建提示词模板
    chat_prompt_template = build_poem_chat_prompt_template()

    # 2. 准备历史会话数据（与课件中的诗歌示例类似）
    history_data = [
        ("human", "你来写一首唐诗"),
        ("ai", "床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
        ("human", "好诗，再来一个"),
        ("ai", "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
    ]

    print("\n历史会话数据：")
    for i, (role, content) in enumerate(history_data, 1):
        role_map = {"human": "用户", "ai": "AI"}
        print(f"  {i}. [{role_map.get(role, role)}] {content}")

    # 3. 通过「|」运算符把提示词模板和模型链接成一个 chain
    chain: RunnableSerializable = chat_prompt_template | chat
    print("\nchain 对象类型：", type(chain))
    print("chain 是否 RunnableSerializable 子类：", isinstance(chain, RunnableSerializable))

    # --------------------------- invoke 调用 ---------------------------
    print("\n" + "-" * 80)
    print("一、使用 chain.invoke(...) 触发链式执行（阻塞调用）")
    print("-" * 80)

    # 这里传入的字典会先喂给 ChatPromptTemplate：
    #  - ChatPromptTemplate 接收到 {"history": history_data}
    #  - 解析 MessagesPlaceholder，生成 PromptValue / 消息列表
    #  - 然后把生成的消息列表作为输入传给模型 chat
    res = chain.invoke({"history": history_data})
    # 对于 ChatTongyi，返回的是 AIMessage 对象，可以通过 .content 获取文本
    print("\n模型回复（invoke）：")
    print(res.content)

    # --------------------------- stream 调用 ---------------------------
    print("\n" + "-" * 80)
    print("二、使用 chain.stream(...) 触发链式执行（流式输出）")
    print("-" * 80)
    print("\n模型回复（stream）：")

    for chunk in chain.stream({"history": history_data}):
        # chunk 同样是 AIMessageChunk，文本在 .content 中
        print(chunk.content, end="", flush=True)
    print("\n")

    print("-" * 80)
    print("链式调用示例结束")
    print("-" * 80)
    print()


def main() -> None:
    """
    入口函数：演示 LangChain 中最基础的「提示词模板 | 模型」链式调用。
    """
    print("=" * 80)
    print("LangChain 链式调用：ChatPromptTemplate | ChatTongyi 示例")
    print("=" * 80)
    print()

    chat = init_chat_model()
    demo_chain_invoke_and_stream(chat)

    print("=" * 80)
    print("演示结束")
    print("=" * 80)


if __name__ == "__main__":
    main()

