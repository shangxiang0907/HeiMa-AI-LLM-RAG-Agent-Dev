"""
LangChain CSVLoader 文档加载器示例

本示例对应课件中关于文档加载器的内容，重点演示：

1. Document 类：
   - Document 是 LangChain 内文档的统一载体
   - 所有文档加载器最终返回此类的实例
   - 核心属性：page_content（文档内容）和 metadata（文档元数据）

2. 文档加载器接口：
   - load()：一次性加载全部文档
   - lazy_load()：延迟流式传输文档，对大型数据集很有用，避免内存溢出

3. CSVLoader 使用：
   - 基本使用：加载带表头的 CSV 文件
   - 自定义解析：使用 csv_args 参数自定义分隔符、引号字符、字段名等

核心概念：
- Document：LangChain 文档的统一载体类
- CSVLoader：用于加载 CSV 文件的文档加载器
- BaseLoader：所有文档加载器需要实现的基类接口
- load()：同步加载所有文档
- lazy_load()：延迟流式加载文档，返回生成器
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document


def document_class_demo() -> None:
    """
    演示 Document 类的基本使用。

    Document 类是 LangChain 内文档的统一载体，所有文档加载器最终返回此类的实例。
    核心属性：
    - page_content：文档内容
    - metadata：文档元数据（字典）
    """
    print("=" * 80)
    print("【示例1】Document 类的基本使用")
    print("-" * 80)

    # 创建一个基础的 Document 类实例
    document = Document(
        page_content="Hello, world!", metadata={"source": "https://example.com"}
    )

    print("Document 对象创建成功！")
    print(f"page_content: {document.page_content}")
    print(f"metadata: {document.metadata}")
    print()


def csvloader_basic_demo() -> None:
    """
    演示 CSVLoader 的基本使用。

    对应课件中的示例：
    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(file_path="./xxx.csv")
    data = loader.load()
    print(data)
    """
    print("=" * 80)
    print("【示例2】CSVLoader 基本使用 - 加载带表头的 CSV 文件")
    print("-" * 80)

    # 创建示例 CSV 文件（如果不存在）
    csv_file = "./stu.csv"
    if not os.path.exists(csv_file):
        print(f"创建示例 CSV 文件：{csv_file}")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("name,age,gender\n")
            f.write("王梓涵,25,男\n")
            f.write("刘若曦,22,女\n")
            f.write("陈俊宇,20,男\n")
            f.write("赵思瑶,28,女\n")
            f.write("黄浩然,15,男\n")
            f.write("林雨桐,20,女\n")
            f.write("周博文,20,男\n")
            f.write("吴诗琪,24,女\n")
            f.write("马子轩,22,男\n")
            f.write("孙悦然,27,女\n")
        print(f"示例 CSV 文件已创建：{csv_file}\n")

    # 使用 CSVLoader 加载 CSV 文件
    loader = CSVLoader(file_path=csv_file)
    documents = loader.load()

    print(f"成功加载 {len(documents)} 个文档")
    print("\n前 3 个文档内容：")
    for i, doc in enumerate(documents[:3], start=1):
        print(f"\n文档 {i}:")
        print(f"  page_content: {doc.page_content}")
        print(f"  metadata: {doc.metadata}")
    print()


def csvloader_lazy_load_demo() -> None:
    """
    演示 CSVLoader 的 lazy_load() 方法。

    lazy_load() 方法用于延迟流式传输文档，对大型数据集很有用，避免内存溢出。
    """
    print("=" * 80)
    print("【示例3】CSVLoader lazy_load() 方法 - 延迟流式加载")
    print("-" * 80)

    csv_file = "./stu.csv"
    if not os.path.exists(csv_file):
        print(f"错误：找不到文件 {csv_file}，请先运行示例2")
        return

    loader = CSVLoader(file_path=csv_file)

    print("使用 lazy_load() 方法逐个加载文档：")
    print("-" * 80)
    for i, document in enumerate(loader.lazy_load(), start=1):
        print(f"文档 {i}:")
        print(f"  page_content: {document.page_content}")
        print(f"  metadata: {document.metadata}")
        if i >= 3:  # 只显示前 3 个
            print("\n... (其余文档省略)")
            break
    print()


def csvloader_custom_parsing_demo() -> None:
    """
    演示 CSVLoader 的自定义解析参数。

    对应课件中的示例，使用 csv_args 参数自定义 CSV 解析：
    - delimiter：指定分隔符
    - quotechar：指定字符串的引号包裹
    - fieldnames：字段列表（无表头使用，有表头勿用会读取首行做为数据）
    """
    print("=" * 80)
    print("【示例4】CSVLoader 自定义解析参数")
    print("-" * 80)

    # 创建一个无表头的 CSV 文件用于演示
    csv_file_no_header = "./stu_no_header.csv"
    print(f"创建无表头的 CSV 文件：{csv_file_no_header}")
    with open(csv_file_no_header, "w", encoding="utf-8") as f:
        # 不写表头，直接写数据
        f.write("王梓涵,25,男\n")
        f.write("刘若曦,22,女\n")
        f.write("陈俊宇,20,男\n")
    print(f"示例 CSV 文件已创建：{csv_file_no_header}\n")

    # 使用自定义参数加载无表头的 CSV 文件
    loader = CSVLoader(
        file_path=csv_file_no_header,
        csv_args={
            "delimiter": ",",  # 指定分隔符
            "quotechar": '"',  # 指定字符串的引号包裹
            # 字段列表（无表头使用，有表头勿用会读取首行做为数据）
            "fieldnames": ["name", "age", "gender"],
        },
    )

    documents = loader.load()

    print(f"成功加载 {len(documents)} 个文档（使用自定义字段名）")
    print("\n所有文档内容：")
    for i, doc in enumerate(documents, start=1):
        print(f"\n文档 {i}:")
        print(f"  page_content: {doc.page_content}")
        print(f"  metadata: {doc.metadata}")
    print()

    # 清理临时文件
    if os.path.exists(csv_file_no_header):
        os.remove(csv_file_no_header)
        print(f"已清理临时文件：{csv_file_no_header}\n")


def csvloader_with_header_demo() -> None:
    """
    演示带表头的 CSV 文件加载（默认行为）。

    当 CSV 文件有表头时，CSVLoader 会自动识别，不需要指定 fieldnames。
    """
    print("=" * 80)
    print("【示例5】CSVLoader 加载带表头的 CSV 文件（默认行为）")
    print("-" * 80)

    csv_file = "./stu.csv"
    if not os.path.exists(csv_file):
        print(f"错误：找不到文件 {csv_file}，请先运行示例2")
        return

    # 默认情况下，CSVLoader 会自动识别表头
    loader = CSVLoader(file_path=csv_file)
    documents = loader.load()

    print(f"成功加载 {len(documents)} 个文档")
    print("\n文档结构说明：")
    print("- CSVLoader 会自动将每一行数据转换为一个 Document 对象")
    print("- page_content 包含该行的所有字段信息")
    print("- metadata 包含行号等元数据信息")
    print("\n示例文档：")
    if documents:
        print(f"  page_content: {documents[0].page_content}")
        print(f"  metadata: {documents[0].metadata}")
    print()


def main() -> None:
    """
    主函数：演示 CSVLoader 的各种使用方法。
    """
    print("=" * 80)
    print("LangChain CSVLoader 文档加载器示例")
    print("=" * 80)
    print()

    # 加载环境变量（虽然 CSVLoader 不需要 API Key，但保持一致性）
    load_dotenv()

    # 示例1：Document 类的基本使用
    document_class_demo()

    # 示例2：CSVLoader 基本使用
    csvloader_basic_demo()

    # 示例3：lazy_load() 方法
    csvloader_lazy_load_demo()

    # 示例4：自定义解析参数
    csvloader_custom_parsing_demo()

    # 示例5：带表头的 CSV 文件加载
    csvloader_with_header_demo()

    print("=" * 80)
    print("演示结束")
    print("=" * 80)
    print("\n提示：")
    print("- 示例 CSV 文件 stu.csv 已创建在当前目录")
    print("- 可以手动编辑该文件来测试不同的 CSV 格式")
    print("- 更多文档加载器请参考：https://docs.langchain.com/oss/python/integrations/document_loaders")


if __name__ == "__main__":
    main()
