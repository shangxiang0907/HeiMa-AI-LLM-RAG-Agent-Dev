"""
LangChain 嵌入模型（Embeddings）调用示例

本示例演示如何在 LangChain 中使用阿里云 DashScope 的嵌入模型：
- 使用 `DashScopeEmbeddings` 创建嵌入模型实例
- 调用 `embed_query()` 对单条文本生成向量
- 调用 `embed_documents()` 对多条文本批量生成向量

核心概念：
- Embedding（向量化）：将一段文本转换成一个浮点数列表（向量），
  使得「相似的文本」在向量空间中的距离更近，用于相似度搜索、向量数据库、RAG 检索等。
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings


def init_embedding_model() -> DashScopeEmbeddings:
    """
    初始化 DashScopeEmbeddings 嵌入模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）

    默认使用 DashScope 的 text-embedding 模型（LangChain 内部有默认值），
    一般命名类似于：text-embedding-v1 / text-embedding-v2。
    """
    load_dotenv()

    # 兼容两种环境变量命名方式
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    # LangChain 的 DashScopeEmbeddings 会自动从环境变量中读取 key，
    # 这里设置一份到 DASHSCOPE_API_KEY，确保兼容性。
    os.environ["DASHSCOPE_API_KEY"] = api_key

    # 如果需要显式指定模型，可以传入 model="text-embedding-v1" 等参数：
    # embed = DashScopeEmbeddings(model="text-embedding-v1")
    embed = DashScopeEmbeddings()
    return embed


def demo_embed_query(embed: DashScopeEmbeddings) -> None:
    """
    演示 `embed_query`：对单条文本进行向量化。
    """
    print("=" * 80)
    print("【示例1】embed_query：对单条文本生成向量")
    print("-" * 80)

    text = "我喜欢你"
    print(f"原始文本：{text}")

    vector: List[float] = embed.embed_query(text)

    print(f"\n向量维度：{len(vector)}")
    # 只展示前几个维度，避免输出过长
    preview_dims = 8
    print(f"前 {preview_dims} 维示例：{vector[:preview_dims]}")
    print("-" * 80)
    print()


def demo_embed_documents(embed: DashScopeEmbeddings) -> None:
    """
    演示 `embed_documents`：对多条文本批量生成向量。
    """
    print("=" * 80)
    print("【示例2】embed_documents：对多条文本批量生成向量")
    print("-" * 80)

    docs = ["我喜欢你", "我稀饭你", "晚上吃啥"]
    print("原始文本列表：")
    for i, d in enumerate(docs, 1):
        print(f"  {i}. {d}")

    vectors: List[List[float]] = embed.embed_documents(docs)

    print(f"\n共生成 {len(vectors)} 个向量，每个向量维度：{len(vectors[0]) if vectors else 0}")
    print("\n每条文本的向量前几维示例：")
    preview_dims = 6

    # 使用 zip(docs, vectors) 将「原始文本」和「对应的向量」一一配对，
    # 再用 enumerate(..., 1) 给每一对 (文本, 向量) 编上从 1 开始的序号 i。
    # 这样在 for 循环里就可以同时拿到：序号 i、文本 d，以及该文本的向量 v。
    for i, (d, v) in enumerate(zip(docs, vectors), 1):
        print(f"  {i}. 文本：{d}")
        print(f"     向量前 {preview_dims} 维：{v[:preview_dims]}")
    print("-" * 80)
    print()


def intro_summary() -> None:
    """
    简要总结：什么时候使用嵌入模型？
    """
    print("=" * 80)
    print("【嵌入模型简介】")
    print("=" * 80)
    print()
    print("📌 嵌入模型（Embeddings）的典型应用场景：")
    print("- 相似度搜索：找到与查询文本语义最接近的文档")
    print("- 向量数据库：将文档向量化后存入 Milvus、Faiss、PGVector 等")
    print("- RAG 检索：根据用户问题，在知识库中检索相关文档再交给大模型回答")
    print("- 文本聚类 / 降维可视化：基于语义相似性对文本分组")
    print()
    print("一般流程是：文本 -> 嵌入向量 -> 相似度计算 / 向量索引 -> 返回最相似结果。")
    print()


def main() -> None:
    """
    主函数：演示 DashScope 嵌入模型在 LangChain 中的基本用法。
    """
    print("=" * 80)
    print("LangChain 嵌入模型（DashScopeEmbeddings）调用示例")
    print("=" * 80)
    print()

    embed = init_embedding_model()

    # 示例1：单条文本向量化
    demo_embed_query(embed)

    # 示例2：多条文本批量向量化
    demo_embed_documents(embed)

    # 嵌入模型使用说明
    intro_summary()

    print("=" * 80)
    print("示例结束")
    print("=" * 80)


if __name__ == "__main__":
    main()

