"""
LangChain RAG（检索增强生成）完整流程示例

本示例对应课件中关于 RAG 完整流程的内容，重点演示：

1. RAG 流程概述：
   - RAG（Retrieval-Augmented Generation）是结合向量检索和大语言模型的生成能力
   - 流程：用户提问 -> 向量库检索 -> 构建提示词（用户提问 + 检索到的参考资料）-> LLM 生成回答

2. 核心组件：
   - ChatTongyi：大语言模型，用于生成回答
   - InMemoryVectorStore：向量存储，用于存储和检索文档
   - DashScopeEmbeddings：嵌入模型，用于将文本转换为向量
   - ChatPromptTemplate：提示词模板，用于构建包含上下文的提示词
   - StrOutputParser：输出解析器，用于将模型输出解析为字符串

3. 完整流程步骤：
   a. 初始化模型和向量存储
   b. 准备资料（向量库的数据）
   c. 用户提问
   d. 检索向量库（找到与用户提问最相似的文档）
   e. 构建提示词（用户提问 + 检索到的参考资料）
   f. 通过链式调用生成回答

核心概念：
- RAG：检索增强生成，结合向量检索和 LLM 生成
- Vector Store：向量存储，用于存储和检索文档向量
- Similarity Search：相似性搜索，根据查询找到最相似的文档
- Prompt Template：提示词模板，用于构建包含上下文的提示词
- Chain：链式调用，将多个组件串联起来执行
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    初始化向量存储实例。

    使用 DashScopeEmbeddings 作为嵌入模型，将文本转换为向量。
    """
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    os.environ["DASHSCOPE_API_KEY"] = api_key

    # 初始化向量存储，使用 DashScopeEmbeddings 作为嵌入模型
    vector_store = InMemoryVectorStore(
        embedding=DashScopeEmbeddings(model="text-embedding-v4")
    )
    return vector_store


def print_prompt(prompt):
    """
    打印提示词的辅助函数，用于调试。

    这个函数可以在链中使用，用于查看传递给模型的提示词内容。

    Args:
        prompt: 提示词对象

    Returns:
        原样返回提示词对象，以便继续链式调用
    """
    print(prompt.to_string())
    print("=" * 20)
    return prompt


def rag_complete_workflow_demo() -> None:
    """
    演示完整的 RAG 工作流程。

    展示从向量存储初始化、数据准备、检索到生成回答的完整流程。
    """
    print("=" * 80)
    print("【示例】RAG 完整流程演示")
    print("=" * 80)
    print()

    # ==================== 步骤 1：初始化模型和向量存储 ====================
    print("【步骤 1】初始化模型和向量存储")
    print("-" * 80)

    # 初始化聊天模型
    model = init_chat_model()
    print("✓ ChatTongyi 模型初始化成功（model='qwen3-max'）")

    # 初始化向量存储
    vector_store = init_vector_store()
    print("✓ InMemoryVectorStore 初始化成功（使用 DashScopeEmbeddings）")
    print()

    # ==================== 步骤 2：定义提示词模板 ====================
    print("【步骤 2】定义提示词模板")
    print("-" * 80)
    print("提示词结构：")
    print("  - System: 以我提供的已知参考资料为主,简洁和专业的回答用户问题。参考资料:{context}")
    print("  - User: 用户提问:{input}")
    print()

    # 定义提示词模板
    # 提示词: 用户的提问 + 向量库中检索到的参考资料
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "以我提供的已知参考资料为主,简洁和专业的回答用户问题。参考资料:{context}. ",
            ),
            ("user", "用户提问:{input}"),
        ]
    )
    print("✓ 提示词模板创建成功")
    print()

    # ==================== 步骤 3：准备资料（向量库的数据） ====================
    print("【步骤 3】准备资料（向量库的数据）")
    print("-" * 80)
    print("add_texts 传入一个 list[str]")
    print()

    # 准备一下资料（向量库的数据）
    # add_texts 传入一个 list[str]
    knowledge_texts = [
        "减肥就是要少吃多练。",
        "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来。",
        "跑步是很好的运动哦。",
    ]

    print("准备添加的知识文本：")
    for i, text in enumerate(knowledge_texts, 1):
        print(f"  {i}. {text}")

    # 将文本添加到向量存储
    vector_store.add_texts(knowledge_texts)
    print(f"\n✓ 成功添加 {len(knowledge_texts)} 条文本到向量存储")
    print()

    # ==================== 步骤 4：用户提问 ====================
    print("【步骤 4】用户提问")
    print("-" * 80)
    input_text = "怎么减肥?"
    print(f"用户提问：{input_text}")
    print()

    # ==================== 步骤 5：检索向量库 ====================
    print("【步骤 5】检索向量库")
    print("-" * 80)
    print(f"查询文本：{input_text}")
    print("检索前 k=2 个最相似的文档：\n")

    # 检索向量库
    result = vector_store.similarity_search(input_text, k=2)

    print(f"检索结果数量：{len(result)} 个文档")
    for i, doc in enumerate(result, 1):
        print(f"【结果 {i}】")
        print(f"  内容：{doc.page_content}")
        print()

    # ==================== 步骤 6：构建参考文本 ====================
    print("【步骤 6】构建参考文本")
    print("-" * 80)

    # 将检索到的文档内容拼接成参考文本
    reference_text = "["
    for doc in result:
        reference_text += doc.page_content
    reference_text += "]"

    print(f"参考文本：{reference_text}")
    print()

    # ==================== 步骤 7：构建链并调用 ====================
    print("【步骤 7】构建链并调用")
    print("-" * 80)
    print("链结构：prompt | print_prompt | model | StrOutputParser()")
    print("说明：")
    print("  - prompt: 提示词模板，接收 input 和 context 参数")
    print("  - print_prompt: 调试函数，打印提示词内容（可选）")
    print("  - model: 聊天模型，生成回答")
    print("  - StrOutputParser(): 输出解析器，将模型输出解析为字符串")
    print()

    # chain
    chain = prompt | print_prompt | model | StrOutputParser()

    print("调用链：chain.invoke({'input': input_text, 'context': reference_text})")
    print("-" * 80)
    res = chain.invoke({"input": input_text, "context": reference_text})

    print("\n" + "=" * 80)
    print("【最终回答】")
    print("=" * 80)
    print(res)
    print()

    # ==================== 流程总结 ====================
    print("=" * 80)
    print("RAG 流程总结")
    print("=" * 80)
    print("""
完整的 RAG 流程包括以下步骤：

1. 初始化阶段：
   - 初始化聊天模型（ChatTongyi）
   - 初始化向量存储（InMemoryVectorStore）
   - 定义提示词模板（ChatPromptTemplate）

2. 数据准备阶段：
   - 将知识文本添加到向量存储（add_texts）
   - 文本会被自动转换为向量并存储

3. 查询阶段：
   - 用户提问
   - 从向量存储中检索最相似的文档（similarity_search）
   - 将检索到的文档内容拼接成参考文本

4. 生成阶段：
   - 构建提示词（用户提问 + 检索到的参考资料）
   - 通过链式调用将提示词传递给模型
   - 模型基于提示词生成回答
   - 使用输出解析器将结果解析为字符串

RAG 的优势：
- 结合了向量检索的精确性和 LLM 的生成能力
- 可以基于特定领域的知识库生成回答
- 回答更加准确和可靠，因为基于实际文档内容
    """)


def rag_workflow_without_debug_demo() -> None:
    """
    演示不带调试函数的 RAG 工作流程。

    这是更简洁的版本，不包含 print_prompt 调试函数。
    """
    print("=" * 80)
    print("【示例】RAG 完整流程演示（不带调试）")
    print("=" * 80)
    print()

    # 初始化
    model = init_chat_model()
    vector_store = init_vector_store()

    # 定义提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "以我提供的已知参考资料为主,简洁和专业的回答用户问题。参考资料:{context}. ",
            ),
            ("user", "用户提问:{input}"),
        ]
    )

    # 准备资料
    knowledge_texts = [
        "减肥就是要少吃多练。",
        "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来。",
        "跑步是很好的运动哦。",
    ]
    vector_store.add_texts(knowledge_texts)
    print(f"✓ 已添加 {len(knowledge_texts)} 条知识文本到向量存储\n")

    # 用户提问
    input_text = "怎么减肥?"

    # 检索向量库
    result = vector_store.similarity_search(input_text, k=2)

    # 构建参考文本
    reference_text = "[" + "".join([doc.page_content for doc in result]) + "]"

    # 构建链（不带调试函数）
    chain = prompt | model | StrOutputParser()

    # 调用链
    print("执行 RAG 流程...")
    print("-" * 80)
    res = chain.invoke({"input": input_text, "context": reference_text})

    print("\n【用户提问】")
    print(input_text)
    print("\n【AI 回答】")
    print(res)
    print()


def rag_workflow_multiple_queries_demo() -> None:
    """
    演示多个查询的 RAG 工作流程。

    展示如何对不同的用户提问进行检索和回答。
    """
    print("=" * 80)
    print("【示例】RAG 多查询演示")
    print("=" * 80)
    print()

    # 初始化
    model = init_chat_model()
    vector_store = init_vector_store()

    # 定义提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "以我提供的已知参考资料为主,简洁和专业的回答用户问题。参考资料:{context}. ",
            ),
            ("user", "用户提问:{input}"),
        ]
    )

    # 准备资料
    knowledge_texts = [
        "减肥就是要少吃多练。",
        "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来。",
        "跑步是很好的运动哦。",
    ]
    vector_store.add_texts(knowledge_texts)
    print(f"✓ 已添加 {len(knowledge_texts)} 条知识文本到向量存储\n")

    # 构建链
    chain = prompt | model | StrOutputParser()

    # 多个查询
    queries = [
        "怎么减肥?",
        "减脂期间应该怎么吃?",
        "推荐什么运动?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"【查询 {i}】{query}")
        print("-" * 80)

        # 检索向量库
        result = vector_store.similarity_search(query, k=2)

        # 构建参考文本
        reference_text = "[" + "".join([doc.page_content for doc in result]) + "]"

        # 调用链
        res = chain.invoke({"input": query, "context": reference_text})

        print(f"回答：{res}\n")


def main() -> None:
    """
    入口函数：演示 RAG 完整流程。

    本示例分为三个部分：
    1. 完整的 RAG 流程演示（带调试函数）
    2. 简洁的 RAG 流程演示（不带调试函数）
    3. 多查询演示
    """
    print("=" * 80)
    print("LangChain RAG（检索增强生成）完整流程示例")
    print("=" * 80)
    print()

    # 加载环境变量
    load_dotenv()

    # 示例1：完整的 RAG 流程演示（带调试函数）
    rag_complete_workflow_demo()

    print("\n" + "=" * 80)
    print("=" * 80)
    print()

    # 示例2：简洁的 RAG 流程演示（不带调试函数）
    rag_workflow_without_debug_demo()

    print("\n" + "=" * 80)
    print("=" * 80)
    print()

    # 示例3：多查询演示
    rag_workflow_multiple_queries_demo()

    print("=" * 80)
    print("演示结束")
    print("=" * 80)
    print("\n提示：")
    print("- RAG 是结合向量检索和 LLM 生成的技术")
    print("- 通过向量存储检索相关文档，然后基于文档内容生成回答")
    print("- 提示词模板中的 {context} 用于注入检索到的参考资料")
    print("- 提示词模板中的 {input} 用于注入用户提问")
    print("- 链式调用可以方便地将多个组件串联起来")


if __name__ == "__main__":
    main()
