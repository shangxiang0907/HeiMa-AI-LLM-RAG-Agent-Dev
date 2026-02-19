"""
LangChain 向量存储（Vector Store）使用示例

本示例对应课件中关于向量存储的内容，重点演示：

1. 向量存储简介：
   - 向量存储是典型的 RAG（Retrieval Augmented Generation）流程的核心组件
   - 用于存储文档的嵌入向量，并执行相似性搜索
   - 典型的向量存储应用包括两个阶段：
     * 索引阶段（存储）：文档 -> 嵌入模型 -> 嵌入向量 -> 向量存储
     * 查询阶段（检索）：查询文本 -> 嵌入模型 -> 查询向量 -> 相似性搜索 -> Top-k 结果

2. LangChain 向量存储统一接口：
   - add_documents：存入向量（将文档转换为向量并存储）
   - delete：删除向量（通过指定的 id 删除）
   - similarity_search：向量检索（根据查询文本找到最相似的文档）

3. 向量存储类型：
   - 内置向量存储：InMemoryVectorStore（内存向量存储，适合小规模数据）
   - 外部向量存储：Chroma、FAISS、Milvus 等（持久化存储，适合大规模数据）

核心概念：
- Vector Store：向量存储，用于存储和检索文档向量
- Embedding：嵌入向量，将文本转换为数值向量
- Similarity Search：相似性搜索，根据查询向量找到最相似的文档
- RAG：检索增强生成，结合向量检索和大语言模型的生成能力
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def init_embedding_model() -> DashScopeEmbeddings:
    """
    初始化 DashScopeEmbeddings 嵌入模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）
    """
    load_dotenv()

    # 兼容两种环境变量命名方式
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    # LangChain 的 DashScopeEmbeddings 会自动从环境变量中读取 key
    os.environ["DASHSCOPE_API_KEY"] = api_key

    embed = DashScopeEmbeddings()
    return embed


def create_sample_documents() -> List[Document]:
    """
    创建示例文档用于演示。

    Returns:
        List[Document]: 示例文档列表
    """
    documents = [
        Document(
            page_content="Python 是一种高级编程语言，由 Guido van Rossum 创建。",
            metadata={"source": "python_intro", "topic": "programming"},
        ),
        Document(
            page_content="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
            metadata={"source": "ml_intro", "topic": "ai"},
        ),
        Document(
            page_content="深度学习使用神经网络来模拟人脑的学习过程。",
            metadata={"source": "dl_intro", "topic": "ai"},
        ),
        Document(
            page_content="自然语言处理（NLP）是计算机科学和人工智能的一个分支，研究如何让计算机理解和生成人类语言。",
            metadata={"source": "nlp_intro", "topic": "ai"},
        ),
        Document(
            page_content="向量数据库专门用于存储和检索高维向量数据，常用于相似性搜索。",
            metadata={"source": "vector_db", "topic": "database"},
        ),
    ]
    return documents


def inmemory_vectorstore_basic_demo() -> None:
    """
    演示 InMemoryVectorStore（内置向量存储）的基本用法。

    展示如何使用内存向量存储进行文档的存储、删除和检索。
    """
    print("=" * 80)
    print("【示例1】InMemoryVectorStore 基本用法")
    print("-" * 80)

    # 初始化嵌入模型
    embedding = init_embedding_model()

    # 创建内存向量存储
    vector_store = InMemoryVectorStore(embedding=embedding)
    print("✓ 内存向量存储创建成功\n")

    # 创建示例文档
    documents = create_sample_documents()
    print(f"准备存储 {len(documents)} 个文档\n")

    # 1. 添加文档到向量存储，并指定 id
    print("1. 添加文档到向量存储（add_documents）")
    print("-" * 80)
    doc_ids = [f"doc_{i+1}" for i in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=doc_ids)
    print(f"✓ 成功添加 {len(documents)} 个文档到向量存储")
    print(f"  文档 ID：{doc_ids}\n")

    # 2. 相似性搜索
    print("2. 相似性搜索（similarity_search）")
    print("-" * 80)
    query = "什么是机器学习？"
    print(f"查询文本：{query}")
    print(f"搜索前 {3} 个最相似的文档：\n")

    similar_docs = vector_store.similarity_search(query, k=3)
    print(f"删除前搜索结果数量：{len(similar_docs)} 个文档\n")
    for i, doc in enumerate(similar_docs, start=1):
        print(f"【结果 {i}】")
        print(f"  内容：{doc.page_content}")
        print(f"  元数据：{doc.metadata}\n")

    # 3. 删除文档
    print("3. 删除文档（delete）")
    print("-" * 80)
    delete_ids = ["doc_1"]
    print(f"删除文档 ID：{delete_ids}")
    vector_store.delete(ids=delete_ids)
    print("✓ 文档删除成功\n")

    # 再次搜索，验证删除效果
    print("4. 验证删除效果（再次搜索）")
    print("-" * 80)
    similar_docs_after = vector_store.similarity_search(query, k=3)
    print(f"删除后搜索结果数量：{len(similar_docs_after)} 个文档")
    
    # 验证删除是否生效
    deleted_doc_found = any(
        doc.metadata.get("source") == "python_intro"  # doc_1 对应的文档
        for doc in similar_docs_after
    )
    
    if deleted_doc_found:
        print("⚠️  警告：被删除的文档仍在搜索结果中，删除可能未生效")
    else:
        print("✓ 验证通过：被删除的文档已从搜索结果中移除")
    
    if len(similar_docs_after) < len(similar_docs):
        print(f"✓ 验证通过：搜索结果数量从 {len(similar_docs)} 减少到 {len(similar_docs_after)}")
    elif len(similar_docs_after) == len(similar_docs):
        print(f"ℹ️  搜索结果数量未变化（可能因为 k={3} 的限制，或删除的文档不在前 k 个结果中）")
    print()


def inmemory_vectorstore_with_metadata_demo() -> None:
    """
    演示 InMemoryVectorStore 与元数据的使用。

    展示文档的元数据信息。
    """
    print("=" * 80)
    print("【示例2】InMemoryVectorStore - 文档元数据")
    print("-" * 80)

    # 初始化嵌入模型和向量存储
    embedding = init_embedding_model()
    vector_store = InMemoryVectorStore(embedding=embedding)

    # 添加文档
    documents = create_sample_documents()
    doc_ids = [f"doc_{i+1}" for i in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=doc_ids)
    print(f"✓ 已添加 {len(documents)} 个文档\n")

    # 搜索并展示元数据
    print("1. 相似性搜索（展示文档元数据）")
    print("-" * 80)
    query = "人工智能相关的内容"
    print(f"查询文本：{query}\n")

    similar_docs = vector_store.similarity_search(query, k=3)
    print(f"搜索结果数量：{len(similar_docs)} 个文档")
    for i, doc in enumerate(similar_docs, start=1):
        print(f"【结果 {i}】")
        print(f"  内容：{doc.page_content}")
        print(f"  元数据：{doc.metadata}\n")
    print()


def inmemory_vectorstore_similarity_search_with_score_demo() -> None:
    """
    演示 InMemoryVectorStore 的相似性搜索（带分数）。

    展示如何获取搜索结果的相似度分数。
    """
    print("=" * 80)
    print("【示例3】InMemoryVectorStore - 相似性搜索（带分数）")
    print("-" * 80)

    # 初始化嵌入模型和向量存储
    embedding = init_embedding_model()
    vector_store = InMemoryVectorStore(embedding=embedding)

    # 添加文档
    documents = create_sample_documents()
    doc_ids = [f"doc_{i+1}" for i in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=doc_ids)
    print(f"✓ 已添加 {len(documents)} 个文档\n")

    # 相似性搜索（带分数）
    print("1. 相似性搜索（带分数）")
    print("-" * 80)
    query = "什么是深度学习？"
    print(f"查询文本：{query}\n")

    # similarity_search_with_score 返回 (Document, score) 元组列表
    # 分数越小表示越相似（距离越小）
    results = vector_store.similarity_search_with_score(query, k=3)
    print("搜索结果（带相似度分数）：")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"【结果 {i}】相似度分数：{score:.4f}")
        print(f"  内容：{doc.page_content}")
        print(f"  元数据：{doc.metadata}\n")
    print()


def chroma_vectorstore_demo() -> None:
    """
    演示 Chroma 向量存储的使用。

    Chroma 是一个开源的向量数据库，支持持久化存储。
    """
    print("=" * 80)
    print("【示例4】Chroma 向量存储的使用")
    print("-" * 80)

    try:
        from langchain_chroma import Chroma
    except ImportError:
        print("✗ 未安装 langchain-chroma 库")
        print("请运行：pip install langchain-chroma")
        print("\n示例代码：")
        print("-" * 80)
        print("""
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

# 初始化嵌入模型
embedding = DashScopeEmbeddings()

# 创建 Chroma 向量存储
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db",  # 本地保存数据的目录，不需要持久化时可删除此参数
)

# 添加文档
documents = [Document(page_content="示例文档内容", metadata={"source": "test"})]
vector_store.add_documents(documents=documents, ids=["doc1"])

# 相似性搜索
results = vector_store.similarity_search("查询文本", k=3)

# 删除文档
vector_store.delete(ids=["doc1"])
        """)
        return

    # 初始化嵌入模型
    embedding = init_embedding_model()

    # 创建 Chroma 向量存储
    persist_directory = "./chroma_langchain_db"
    print(f"创建 Chroma 向量存储（持久化目录：{persist_directory}）")
    print("-" * 80)

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding,
        persist_directory=persist_directory,  # 本地保存数据的目录
    )
    print("✓ Chroma 向量存储创建成功\n")

    # 添加文档
    documents = create_sample_documents()
    doc_ids = [f"chroma_doc_{i+1}" for i in range(len(documents))]
    print(f"添加 {len(documents)} 个文档到 Chroma")
    vector_store.add_documents(documents=documents, ids=doc_ids)
    print("✓ 文档添加成功\n")

    # 相似性搜索（不带过滤）
    print("2. 相似性搜索（不带过滤条件）")
    print("-" * 80)
    query = "人工智能相关的内容"
    print(f"查询文本：{query}")
    print("过滤条件：无\n")

    similar_docs = vector_store.similarity_search(query, k=3)
    print(f"搜索结果数量：{len(similar_docs)} 个文档")
    for i, doc in enumerate(similar_docs, start=1):
        print(f"【结果 {i}】")
        print(f"  内容：{doc.page_content}")
        print(f"  元数据：{doc.metadata}\n")

    # 相似性搜索（带 topic 过滤）
    print("3. 相似性搜索（带元数据过滤 - topic）")
    print("-" * 80)
    print(f"查询文本：{query}")
    print("过滤条件：topic='ai'（只搜索 AI 相关主题的文档）\n")

    similar_docs_filtered = vector_store.similarity_search(
        query=query,
        k=3,
        filter={"topic": "ai"}  # 只返回 topic 为 "ai" 的文档
    )
    print(f"过滤后搜索结果数量：{len(similar_docs_filtered)} 个文档")
    for i, doc in enumerate(similar_docs_filtered, start=1):
        print(f"【结果 {i}】")
        print(f"  内容：{doc.page_content}")
        print(f"  元数据：{doc.metadata}\n")

    # 相似性搜索（带 source 过滤）
    print("4. 相似性搜索（带元数据过滤 - source）")
    print("-" * 80)
    print(f"查询文本：{query}")
    print("过滤条件：source='ml_intro'（只搜索特定来源的文档）\n")

    similar_docs_source_filtered = vector_store.similarity_search(
        query=query,
        k=3,
        filter={"source": "ml_intro"}  # 只返回 source 为 "ml_intro" 的文档
    )
    print(f"过滤后搜索结果数量：{len(similar_docs_source_filtered)} 个文档")
    for i, doc in enumerate(similar_docs_source_filtered, start=1):
        print(f"【结果 {i}】")
        print(f"  内容：{doc.page_content}")
        print(f"  元数据：{doc.metadata}\n")

    # 删除文档
    print("5. 删除文档")
    print("-" * 80)
    delete_ids = ["chroma_doc_1"]
    print(f"删除文档 ID：{delete_ids}")
    vector_store.delete(ids=delete_ids)
    print("✓ 文档删除成功\n")

    print(f"提示：Chroma 数据已持久化到 {persist_directory} 目录")
    print("下次运行时，如果使用相同的 collection_name 和 persist_directory，")
    print("会自动加载之前存储的数据。\n")
    print()


def vectorstore_rag_workflow_demo() -> None:
    """
    演示完整的 RAG 工作流程。

    展示从文档加载、向量化、存储到检索的完整流程。
    """
    print("=" * 80)
    print("【示例5】完整的 RAG 工作流程演示")
    print("-" * 80)

    # 初始化嵌入模型
    embedding = init_embedding_model()

    # 创建向量存储
    vector_store = InMemoryVectorStore(embedding=embedding)
    print("✓ 向量存储初始化完成\n")

    # 索引阶段：文档 -> 嵌入模型 -> 嵌入向量 -> 向量存储
    print("【索引阶段】文档存储流程")
    print("-" * 80)
    documents = create_sample_documents()
    print(f"1. 准备文档：{len(documents)} 个文档")
    print("2. 通过嵌入模型将文档转换为向量")
    print("3. 将向量存入向量存储\n")

    doc_ids = [f"rag_doc_{i+1}" for i in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=doc_ids)
    print("✓ 索引阶段完成：文档已转换为向量并存储\n")

    # 查询阶段：查询文本 -> 嵌入模型 -> 查询向量 -> 相似性搜索 -> Top-k 结果
    print("【查询阶段】文档检索流程")
    print("-" * 80)
    query = "请介绍一下人工智能的相关技术"
    print(f"1. 用户查询：{query}")
    print("2. 通过嵌入模型将查询转换为向量")
    print("3. 在向量存储中执行相似性搜索")
    print("4. 返回 Top-k 最相似的文档\n")

    similar_docs = vector_store.similarity_search(query, k=3)
    print("✓ 查询阶段完成：找到最相似的文档\n")

    print("检索结果：")
    print("-" * 80)
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n【结果 {i}】")
        print(f"  内容：{doc.page_content}")
        print(f"  元数据：{doc.metadata}")

    print("\n" + "=" * 80)
    print("RAG 工作流程说明：")
    print("=" * 80)
    print("""
典型的 RAG 流程包括两个阶段：

1. 索引阶段（存储）：
   文档 -> 嵌入模型 -> 嵌入向量 -> 向量存储
    
2. 查询阶段（检索）：
   查询文本 -> 嵌入模型 -> 查询向量 -> 相似性搜索 -> Top-k 结果

LangChain 为向量存储提供了统一接口：
- add_documents：存入向量
- delete：删除向量
- similarity_search：向量检索
    """)
    print()


def vectorstore_installation_demo() -> None:
    """
    演示向量存储相关的安装要求。

    展示如何安装所需的依赖库。
    """
    print("=" * 80)
    print("【示例6】向量存储安装要求")
    print("-" * 80)

    print("向量存储相关的库安装：")
    print("-" * 80)

    # 检查 InMemoryVectorStore
    try:
        from langchain_core.vectorstores import InMemoryVectorStore
        print("✓ InMemoryVectorStore 可用（langchain-core 内置）")
    except ImportError as e:
        print(f"✗ InMemoryVectorStore 不可用：{e}")

    # 检查 DashScopeEmbeddings
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        print("✓ DashScopeEmbeddings 可用（langchain-community）")
    except ImportError as e:
        print(f"✗ DashScopeEmbeddings 不可用：{e}")
        print("  请运行：pip install langchain-community")

    # 检查 Chroma
    try:
        from langchain_chroma import Chroma
        print("✓ Chroma 可用（langchain-chroma）")
    except ImportError:
        print("✗ Chroma 不可用")
        print("  请运行：pip install langchain-chroma")

    print("\n安装命令汇总：")
    print("-" * 80)
    print("  # 基础依赖")
    print("  pip install langchain-core langchain-community")
    print("  # Chroma 向量存储（可选）")
    print("  pip install langchain-chroma")
    print()


def main() -> None:
    """
    主函数：演示向量存储的各种使用方法。
    """
    print("=" * 80)
    print("LangChain 向量存储（Vector Store）使用示例")
    print("=" * 80)
    print()

    # 加载环境变量
    load_dotenv()

    # 示例6：安装要求（先运行，让用户知道需要安装什么）
    vectorstore_installation_demo()

    # 示例1：InMemoryVectorStore 基本用法
    inmemory_vectorstore_basic_demo()

    # 示例2：元数据过滤
    inmemory_vectorstore_with_metadata_demo()

    # 示例3：相似性搜索（带分数）
    inmemory_vectorstore_similarity_search_with_score_demo()

    # 示例4：Chroma 向量存储
    chroma_vectorstore_demo()

    # 示例5：完整的 RAG 工作流程
    vectorstore_rag_workflow_demo()

    print("=" * 80)
    print("演示结束")
    print("=" * 80)
    print("\n提示：")
    print("- 向量存储是 RAG 流程的核心组件，用于存储和检索文档向量")
    print("- InMemoryVectorStore 适合小规模数据，数据存储在内存中")
    print("- Chroma 等外部向量存储支持持久化，适合大规模数据")
    print("- LangChain 提供了统一的接口：add_documents、delete、similarity_search")
    print("- 更多向量存储请参考：https://docs.langchain.com/oss/python/integrations/vectorstores/")


if __name__ == "__main__":
    main()
