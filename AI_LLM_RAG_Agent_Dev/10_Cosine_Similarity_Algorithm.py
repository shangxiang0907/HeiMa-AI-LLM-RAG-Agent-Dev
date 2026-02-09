"""
余弦相似度算法介绍与实现

余弦相似度（Cosine Similarity）是一种常用的相似度度量方法，广泛应用于：
- 文本相似度计算
- 推荐系统
- 信息检索
- 向量空间模型
- RAG（检索增强生成）中的文档检索

原理：
余弦相似度通过计算两个向量的夹角余弦值来衡量它们的相似程度。
公式：cos(θ) = (A · B) / (||A|| × ||B||)
其中：
- A · B 表示向量A和B的点积
- ||A|| 和 ||B|| 分别表示向量A和B的模（长度）
- 结果范围：[-1, 1]，值越接近1表示越相似，越接近-1表示越不相似

优点：
1. 不受向量长度影响，只关注方向
2. 对高维稀疏向量效果好
3. 计算效率高
"""

import numpy as np
from typing import List, Tuple
import math


def cosine_similarity_manual(vec1: List[float], vec2: List[float]) -> float:
    """
    手动实现余弦相似度计算
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
    
    Returns:
        余弦相似度值，范围 [-1, 1]
    """
    if len(vec1) != len(vec2):
        raise ValueError("两个向量的维度必须相同")
    
    # 计算点积
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # 计算向量模长
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    # 避免除零错误
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # 计算余弦相似度
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return cosine_sim


def cosine_similarity_numpy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    使用 NumPy 实现余弦相似度计算（更高效）
    
    Args:
        vec1: 第一个向量（numpy数组）
        vec2: 第二个向量（numpy数组）
    
    Returns:
        余弦相似度值，范围 [-1, 1]
    """
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算向量模长
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # 避免除零错误
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # 计算余弦相似度
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return float(cosine_sim)


def text_to_vector(text: str, vocabulary: List[str]) -> List[float]:
    """
    将文本转换为词频向量（简单的文本向量化方法）
    
    Args:
        text: 输入文本
        vocabulary: 词汇表
    
    Returns:
        词频向量
    """
    # 对文本做简单预处理：全部转小写并按空格切分为词列表
    words = text.lower().split()
    # 构建词频向量：对于词表 vocabulary 中的每个词，统计它在当前文本中出现的次数
    # 例如 vocabulary = ["机器学习", "人工智能"]，若文本中分别出现 2 次和 1 次，则 vector = [2, 1]
    vector = [words.count(word) for word in vocabulary]
    return vector


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的余弦相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
    
    Returns:
        文本相似度值
    """
    # 构建词汇表（两个文本的所有唯一词）
    # 使用 set 去重：同一个词在同一文本中出现多次，只会保留一份
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    # 集合并集运算（words1 | words2）：获取“在 text1 或 text2 中出现过的所有不重复的词”
    # 因为是集合并集，所以这里已经没有重复词，vocabulary 是一个“去重后的词表”
    vocabulary = sorted(list(words1 | words2))
    
    # 转换为向量
    vec1 = text_to_vector(text1, vocabulary)
    vec2 = text_to_vector(text2, vocabulary)
    
    # 计算余弦相似度
    return cosine_similarity_manual(vec1, vec2)


def find_most_similar(query_vector: np.ndarray, document_vectors: List[np.ndarray]) -> Tuple[int, float]:
    """
    在文档向量集合中找到与查询向量最相似的文档
    
    Args:
        query_vector: 查询向量
        document_vectors: 文档向量列表
    
    Returns:
        (最相似文档的索引, 相似度分数)
    """
    similarities = []
    
    for doc_vec in document_vectors:
        sim = cosine_similarity_numpy(query_vector, doc_vec)
        similarities.append(sim)
    
    # 找到最大相似度的索引
    max_index = np.argmax(similarities)
    max_similarity = similarities[max_index]
    
    return max_index, max_similarity


def main():
    """
    主函数：演示余弦相似度的各种应用场景
    """
    print("=" * 80)
    print("余弦相似度算法介绍与演示")
    print("=" * 80)
    
    # 1. 基础示例：计算两个向量的余弦相似度
    print("\n【示例1：基础向量相似度计算】")
    print("-" * 80)
    
    vec_a = [1, 2, 3, 4, 5]
    vec_b = [2, 4, 6, 8, 10]  # vec_b = 2 * vec_a，方向相同，相似度为1
    
    similarity_manual = cosine_similarity_manual(vec_a, vec_b)
    similarity_numpy = cosine_similarity_numpy(np.array(vec_a), np.array(vec_b))
    
    print(f"向量A: {vec_a}")
    print(f"向量B: {vec_b}")
    print(f"手动实现余弦相似度: {similarity_manual:.4f}")
    print(f"NumPy实现余弦相似度: {similarity_numpy:.4f}")
    print(f"说明: 向量B是向量A的2倍，方向相同，相似度为1.0")
    
    # 2. 不同方向的向量
    print("\n【示例2：不同方向的向量】")
    print("-" * 80)
    
    vec_c = [1, 0, 0]
    vec_d = [0, 1, 0]  # 垂直向量，相似度为0
    
    similarity_cd = cosine_similarity_numpy(np.array(vec_c), np.array(vec_d))
    print(f"向量C: {vec_c}")
    print(f"向量D: {vec_d}")
    print(f"余弦相似度: {similarity_cd:.4f}")
    print(f"说明: 两个向量垂直，相似度为0")
    
    # 3. 文本相似度计算
    print("\n【示例3：文本相似度计算】")
    print("-" * 80)
    
    texts = [
        "Machine learning is a subfield of artificial intelligence",
        "Artificial intelligence includes various machine learning techniques",
        "The weather is nice today and it is a good day for a walk",
        "Deep learning is a specialized area within machine learning"
    ]
    
    # 使用英文示例文本，便于词频法（基于空格分词）更好工作
    query_text = "The relationship between machine learning and artificial intelligence"
    
    print(f"查询文本: {query_text}\n")
    print("计算与各文本的相似度:")
    
    for i, text in enumerate(texts, 1):
        similarity = calculate_text_similarity(query_text, text)
        print(f"  文本{i}: {text}")
        print(f"  相似度: {similarity:.4f}\n")
    
    # 4. 文档检索示例（模拟RAG场景）
    print("\n【示例4：文档检索示例（模拟RAG场景）】")
    print("-" * 80)
    
    # 模拟文档向量（实际应用中这些向量来自 embedding 模型）
    documents = [
        "Python is a high level programming language widely used for data science and machine learning",
        "Java is an object oriented programming language often used for enterprise application development",
        "Machine learning algorithms can learn patterns and rules from data",
        "Database management systems are used to store and manage large amounts of structured data"
    ]
    
    # 模拟文档向量（简化示例，实际应使用 embedding 模型）
    # 这里使用简单的英文词频向量作为演示
    query = "How to use Python for machine learning"
    
    print(f"查询: {query}\n")
    print("文档库:")
    for i, doc in enumerate(documents, 1):
        print(f"  文档{i}: {doc}")
    
    # 计算查询与每个文档的相似度
    print("\n相似度排序结果:")
    similarities = []
    for i, doc in enumerate(documents):
        sim = calculate_text_similarity(query, doc)
        similarities.append((i + 1, doc, sim))
    
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    for rank, (doc_id, doc_text, sim) in enumerate(similarities, 1):
        print(f"  排名{rank}: 文档{doc_id} (相似度: {sim:.4f})")
        print(f"    内容: {doc_text}\n")
    
    # 5. 向量归一化的重要性
    print("\n【示例5：向量归一化的影响】")
    print("-" * 80)
    
    vec_e = [1, 2, 3]
    vec_f = [10, 20, 30]  # 10倍缩放
    vec_g = [100, 200, 300]  # 100倍缩放
    
    sim_ef = cosine_similarity_numpy(np.array(vec_e), np.array(vec_f))
    sim_eg = cosine_similarity_numpy(np.array(vec_e), np.array(vec_g))
    
    print(f"向量E: {vec_e}")
    print(f"向量F: {vec_f} (E的10倍)")
    print(f"向量G: {vec_g} (E的100倍)")
    print(f"\nE与F的余弦相似度: {sim_ef:.4f}")
    print(f"E与G的余弦相似度: {sim_eg:.4f}")
    print(f"说明: 余弦相似度不受向量长度影响，只关注方向")
    
    # 6. 实际应用场景说明
    print("\n【应用场景】")
    print("-" * 80)
    print("1. RAG系统：")
    print("   - 将查询和文档转换为向量（embedding）")
    print("   - 使用余弦相似度找到最相关的文档")
    print("   - 将相关文档作为上下文提供给LLM")
    print()
    print("2. 推荐系统：")
    print("   - 计算用户向量和物品向量的相似度")
    print("   - 推荐相似度高的物品给用户")
    print()
    print("3. 文本去重：")
    print("   - 计算文本之间的相似度")
    print("   - 识别和去除重复或高度相似的文本")
    print()
    print("4. 搜索引擎：")
    print("   - 计算查询和文档的相似度")
    print("   - 按相似度排序返回搜索结果")
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
