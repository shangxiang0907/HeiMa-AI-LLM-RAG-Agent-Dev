"""
LangChain RAG：向量检索如何加入链（InMemoryVectorStore 版本）

本示例在 33_LangChain_RAG_Complete_Workflow 的基础上，重点演示：

1. retriever（检索器）概念：
   - 基于向量存储（Vector Store）封装出来的「检索组件」
   - 通过 vector_store.as_retriever() 得到
   - 输入：用户查询文本；输出：与查询最相似的文档列表（List[Document])

2. 向量检索加入链（Chain）的典型方式：
   - 使用 Runnable 组合：{"context": retriever | format_docs, "input": RunnablePassthrough()}
   - 将「检索」作为链中的一个步骤，而不是在链外手动调用 similarity_search
   - 整体链结构：输入问题 -> 检索器 -> 格式化文档 -> PromptTemplate -> ChatModel -> StrOutputParser

3. 本示例使用的组件：
   - ChatTongyi：大语言模型，用于生成回答
   - InMemoryVectorStore：内存向量存储，适合小规模 demo
   - DashScopeEmbeddings：嵌入模型，将文本转换为向量
   - ChatPromptTemplate：提示词模板，将 context 和 input 组合成提示词
   - Runnable 系列（RunnablePassthrough 等）：用来把「检索」和「LLM 调用」串成一条链
   - StrOutputParser：输出解析器，将模型输出解析为字符串

对比 33 号脚本：
------------------------------------
- 33 号脚本：检索（similarity_search）在链外手动调用，然后把结果拼成 reference_text 传给链
- 本脚本：检索通过 retriever.as_retriever() 直接成为链中的一个步骤，链从「问题」自动走到「最终回答」
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore


def init_chat_model() -> ChatTongyi:
    """
    初始化 ChatTongyi 聊天模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）

    使用 qwen3-max 作为聊天模型。
    """
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    os.environ["DASHSCOPE_API_KEY"] = api_key
    model = ChatTongyi(model="qwen3-max")
    return model


def init_vector_store() -> InMemoryVectorStore:
    """
    初始化 InMemoryVectorStore 向量存储实例。

    使用 DashScopeEmbeddings 作为嵌入模型，将文本转换为向量。
    """
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    os.environ["DASHSCOPE_API_KEY"] = api_key

    vector_store = InMemoryVectorStore(
        embedding=DashScopeEmbeddings(model="text-embedding-v4")
    )
    return vector_store


def create_knowledge_texts() -> List[str]:
    """
    构造一个简单的知识库文本列表，用于写入向量存储。

    Returns:
        List[str]: 知识文本列表
    """
    return [
        "减肥就是要少吃多练。",
        "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来。",
        "跑步是很好的运动哦。",
    ]


def format_docs(docs: List[Document]) -> str:
    """
    将检索到的文档列表格式化为一个字符串，用作 Prompt 中的 {context}。

    Args:
        docs: 检索得到的文档列表

    Returns:
        str: 连接后的上下文字符串
    """
    if not docs:
        return "当前没有检索到任何相关资料，请根据常识谨慎回答。"

    contents = [doc.page_content for doc in docs]
    return "[参考资料开始]\n" + "\n\n".join(contents) + "\n[参考资料结束]"


def debug_print_docs(docs: List[Document]) -> List[Document]:
    """
    调试用：打印 retriever 检索出来的文档内容，然后原样返回。

    这样可以插在链路中间，观察中间结果，而不改变后续逻辑。
    """
    print("\n===== 调试：retriever 检索到的文档 =====")
    if not docs:
        print("（检索结果为空）")
    else:
        for i, doc in enumerate(docs, start=1):
            print(f"【文档 {i}】")
            print(f"内容：{doc.page_content}")
            if doc.metadata:
                print(f"元数据：{doc.metadata}")
            print("-" * 40)
    print("===== 调试结束 =====\n")
    return docs


def build_rag_chain_with_retriever():
    """
    基于 InMemoryVectorStore 构建一个包含「向量检索」步骤的 RAG 链。

    链的整体结构：
        问题(str)
          │
          ├─> {"context": retriever | format_docs, "input": RunnablePassthrough()}
          │         └── retriever：从向量存储中检索相关文档
          │         └── format_docs：把文档列表转成字符串
          │
          ├─> ChatPromptTemplate（把 context 和 input 注入到提示词）
          ├─> ChatTongyi 模型
          └─> StrOutputParser（解析为字符串）
    """
    # 1. 初始化模型和向量存储
    model = init_chat_model()
    vector_store = init_vector_store()

    # 2. 准备知识库数据并写入向量存储
    knowledge_texts = create_knowledge_texts()
    vector_store.add_texts(knowledge_texts)

    # 3. 从向量存储构建 retriever（检索器）
    #    search_kwargs 中可以设置 k 等检索参数
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 4. 定义提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是专业的健身与营养顾问，请以我提供的参考资料为主，"
                "结合自己的知识，给出简洁、专业、可执行的建议。参考资料如下：\n{context}",
            ),
            ("user", "用户提问：{input}"),
        ]
    )

    # 5. 构建完整 RAG 链：
    #   - 输入是一个 str（用户问题）
    #   - 通过字典映射，把同一个输入分发给：
    #       * "input"：直接原样透传（RunnablePassthrough）
    #       * "context"：先经过 retriever，再经过 format_docs
    #   - 然后把 {"input": ..., "context": ...} 送入 prompt -> model -> parser
    rag_chain = {
        "input": RunnablePassthrough(),  # 直接把原始问题传给 Prompt 的 {input}
        # 在 retriever 和 format_docs 之间插入 debug_print_docs，打印中间检索结果
        "context": retriever | debug_print_docs | format_docs,
    } | prompt | model | StrOutputParser()

    return rag_chain


def rag_retriever_chain_demo() -> None:
    """
    演示：向量检索如何作为链中的一个环节（基于 InMemoryVectorStore）。

    与 33 号脚本的对比：
        - 33 号：similarity_search 在链外手动调用
        - 本示例：retriever 直接在链内完成检索，输入只需要给一个问题字符串即可
    """
    print("=" * 80)
    print("【示例】RAG：向量检索加入链（InMemoryVectorStore + retriever）")
    print("=" * 80)
    print()

    # 构建 RAG 链（内部已经初始化模型、构建向量存储并写入数据）
    rag_chain = build_rag_chain_with_retriever()

    # 准备几条不同的问题，展示同一个链如何复用
    questions = [
        "怎么减肥比较科学？",
        "减脂期间应该怎么吃？",
        "有什么运动适合刚开始减肥的人？",
    ]

    for i, question in enumerate(questions, start=1):
        print(f"【问题 {i}】{question}")
        print("-" * 80)

        # 这里直接把 str 问题传给链，链内部会自动：
        #   问题 -> retriever 检索 -> format_docs -> Prompt -> Model -> Parser
        answer = rag_chain.invoke(question)

        print("【AI 回答】")
        print(answer)
        print()

    print("=" * 80)
    print("示例说明")
    print("=" * 80)
    print(
        """
在这个示例中，向量检索不再是「链外的一次函数调用」，而是正式成为链中的一个步骤：

1. 通过 InMemoryVectorStore.as_retriever() 得到 retriever 对象；
2. 使用 {"context": retriever | format_docs, "input": RunnablePassthrough()} 把检索逻辑嵌入链图；
3. 对调用者来说，只需要传入一个问题字符串，链会自动完成「检索 + 构造提示词 + 调用 LLM + 解析输出」的全流程。

这种写法更符合 LangChain 推荐的 RAG 结构，也更方便在后续组合、并行、监控和部署。
        """
    )


def main() -> None:
    """
    入口函数：演示向量检索加入链的完整流程。
    """
    # 确保可以读取 .env 中的配置
    load_dotenv()

    rag_retriever_chain_demo()


if __name__ == "__main__":
    main()

