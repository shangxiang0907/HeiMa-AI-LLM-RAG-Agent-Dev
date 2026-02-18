"""
LangChain TextLoader 和文档分割器示例

本示例对应课件中关于 TextLoader 和 RecursiveCharacterTextSplitter 的内容，重点演示：

1. TextLoader 简介：
   - TextLoader 用于将文本文件加载为 Document 类型对象
   - 支持指定文件编码（默认使用系统编码）
   - 可以加载各种纯文本格式的文件

2. RecursiveCharacterTextSplitter 简介：
   - RecursiveCharacterTextSplitter（递归字符文本分割器）主要用于按自然段落分割大文档
   - 是 LangChain 官方推荐的默认字符分割器
   - 它在保持上下文完整性和控制片段大小之间实现了良好平衡，开箱即用效果佳

3. RecursiveCharacterTextSplitter 参数：
   - chunk_size: 分段的最大字符数
   - chunk_overlap: 分段之间允许重叠的字符数（用于保持上下文连续性）
   - separators: 文本分段依据，按优先级排序的分隔符列表
   - length_function: 字符统计依据（函数），默认使用 len

核心概念：
- TextLoader：用于加载文本文件的文档加载器
- RecursiveCharacterTextSplitter：递归字符文本分割器，用于将大文档分割成小块
- Document：LangChain 文档的统一载体类
- chunk_size：每个文档块的最大字符数
- chunk_overlap：文档块之间的重叠字符数
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_sample_text_file() -> str:
    """
    创建示例文本文件用于演示。

    Returns:
        str: 创建的文本文件路径
    """
    text_file = "./data/Python_Basics.txt"
    
    # 创建 data 目录（如果不存在）
    os.makedirs("./data", exist_ok=True)
    
    # 创建示例文本内容（英文版本，便于观察分割效果）
    sample_content = """Python Basics

I. Variables and Data Types

Python is a dynamically typed language, meaning variables do not need to be declared with a type beforehand. Python supports various data types including integers, floating-point numbers, strings, and boolean values.

Integer type (int): Used to represent whole numbers, such as 10, -5, 0, etc.
Floating-point type (float): Used to represent decimal numbers, such as 3.14, -0.5, 2.0, etc.
String type (str): Used to represent text, which can be enclosed in single or double quotes, such as 'hello', "world", etc.
Boolean type (bool): Used to represent truth values, with only two values: True and False.

II. Lists and Dictionaries

A list is one of the most commonly used data structures in Python, used to store an ordered sequence of elements. Lists can contain elements of different types and can dynamically add, remove, and modify elements.

A dictionary (dict) is another important data structure used to store key-value pairs. Each element in a dictionary consists of a key and a value, and you can quickly access the corresponding value through the key.

III. Conditional Statements and Loops

Conditional statements (if-elif-else) are used to execute different code blocks based on conditions. Python uses indentation to represent code blocks, which is an important feature of Python.

Loop statements include for loops and while loops. A for loop is used to iterate over sequences (such as lists, strings, etc.), while a while loop is used to repeatedly execute a code block as long as a condition is true.

IV. Function Definitions

Functions are an important way to organize code, allowing you to encapsulate a piece of code and call it by function name. Python uses the def keyword to define functions, and functions can accept parameters and return values.

Basic syntax for function definition:
def function_name(parameters):
    # Function body
    return value

V. Classes and Objects

Python is an object-oriented programming language that supports the concepts of classes and objects. A class is an abstract data type used to define the attributes and methods of objects. An object is an instance of a class and has the attributes and methods defined by the class.

Classes are defined using the class keyword, and attributes and methods can be defined within a class. Methods are functions defined within a class used to manipulate the data of objects.
"""
    
    # 写入文件
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print(f"示例文本文件已创建：{text_file}")
    return text_file


def textloader_basic_demo() -> None:
    """
    演示 TextLoader 的基本用法。

    展示如何使用 TextLoader 加载文本文件。
    """
    print("=" * 80)
    print("【示例1】TextLoader 基本用法")
    print("-" * 80)
    
    # 创建示例文本文件
    text_file = create_sample_text_file()
    print()
    
    # 使用 TextLoader 加载文本文件
    print("使用 TextLoader 加载文本文件：")
    print("-" * 80)
    loader = TextLoader(
        text_file,
        encoding="utf-8",  # 指定文件编码
    )
    docs = loader.load()
    
    print(f"成功加载 {len(docs)} 个文档")
    print(f"\n文档内容预览（前 200 个字符）：")
    print("-" * 80)
    if docs:
        content_preview = docs[0].page_content[:200]
        print(content_preview + "...")
        print(f"\n文档元数据：{docs[0].metadata}")
        print(f"文档总长度：{len(docs[0].page_content)} 个字符")
    print()


def recursive_character_splitter_basic_demo() -> None:
    """
    演示 RecursiveCharacterTextSplitter 的基本用法。

    展示如何使用 RecursiveCharacterTextSplitter 分割文档。
    """
    print("=" * 80)
    print("【示例2】RecursiveCharacterTextSplitter 基本用法")
    print("-" * 80)
    
    # 加载文档
    text_file = "./data/Python_Basics.txt"
    if not os.path.exists(text_file):
        print(f"错误：找不到文件 {text_file}，请先运行示例1")
        return
    
    loader = TextLoader(
        text_file,
        encoding="utf-8",
    )
    docs = loader.load()
    print(f"原始文档长度：{len(docs[0].page_content)} 个字符\n")
    
    # 创建文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 分段的最大字符数
        chunk_overlap=50,  # 分段之间允许重叠的字符数
        # 文本分段依据，按优先级排序
        # 优先使用双换行符分割，然后是单换行符，然后是句号等
        separators=["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", " ", ""],
        # 字符统计依据（函数）
        length_function=len,
    )
    
    # 分割文档
    split_docs = splitter.split_documents(docs)
    
    print(f"分割后得到 {len(split_docs)} 个文档块\n")
    print("前 3 个文档块内容：")
    print("-" * 80)
    for i, doc in enumerate(split_docs[:3], start=1):
        print(f"\n【文档块 {i}】")
        print(f"长度：{len(doc.page_content)} 个字符")
        print(f"内容：{doc.page_content[:150]}...")
        print(f"元数据：{doc.metadata}")
    print()


def recursive_character_splitter_parameters_demo() -> None:
    """
    演示 RecursiveCharacterTextSplitter 不同参数的效果。

    展示 chunk_size、chunk_overlap 和 separators 参数的影响。
    """
    print("=" * 80)
    print("【示例3】RecursiveCharacterTextSplitter 参数对比")
    print("-" * 80)
    
    # 加载文档
    text_file = "./data/Python_Basics.txt"
    if not os.path.exists(text_file):
        print(f"错误：找不到文件 {text_file}，请先运行示例1")
        return
    
    loader = TextLoader(
        text_file,
        encoding="utf-8",
    )
    docs = loader.load()
    original_length = len(docs[0].page_content)
    print(f"原始文档长度：{original_length} 个字符\n")
    
    # 不同的参数配置
    configurations = [
        {
            "name": "小块，无重叠",
            "chunk_size": 200,
            "chunk_overlap": 0,
            "separators": ["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", " ", ""],
        },
        {
            "name": "小块，有重叠",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", " ", ""],
        },
        {
            "name": "大块，有重叠",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", " ", ""],
        },
        {
            "name": "仅按段落分割",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", ""],  # 只使用换行符分割
        },
    ]
    
    for config in configurations:
        print(f"\n配置：{config['name']}")
        print(f"  chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']}")
        print("-" * 80)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=config["separators"],
            length_function=len,
        )
        
        split_docs = splitter.split_documents(docs)
        print(f"分割后得到 {len(split_docs)} 个文档块")
        
        # 显示每个块的长度
        chunk_lengths = [len(doc.page_content) for doc in split_docs]
        print(f"块长度范围：{min(chunk_lengths)} - {max(chunk_lengths)} 个字符")
        print(f"平均块长度：{sum(chunk_lengths) / len(chunk_lengths):.1f} 个字符")
        
        # 显示第一个块的内容预览
        if split_docs:
            print(f"\n第一个块内容预览（前 100 个字符）：")
            print(f"{split_docs[0].page_content[:100]}...")
    print()


def recursive_character_splitter_separators_demo() -> None:
    """
    演示不同 separators 参数的效果。

    展示分隔符优先级对分割结果的影响。
    """
    print("=" * 80)
    print("【示例4】RecursiveCharacterTextSplitter - separators 参数影响")
    print("-" * 80)
    
    # 加载文档
    text_file = "./data/Python_Basics.txt"
    if not os.path.exists(text_file):
        print(f"错误：找不到文件 {text_file}，请先运行示例1")
        return
    
    loader = TextLoader(
        text_file,
        encoding="utf-8",
    )
    docs = loader.load()
    print(f"原始文档长度：{len(docs[0].page_content)} 个字符\n")
    
    # 不同的分隔符配置
    separator_configs = [
        {
            "name": "默认分隔符（推荐）",
            "separators": ["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", " ", ""],
        },
        {
            "name": "仅按段落分割",
            "separators": ["\n\n", "\n", ""],
        },
        {
            "name": "按句子分割",
            "separators": [". ", "! ", "? ", ".", "!", "?", "\n", " ", ""],
        },
        {
            "name": "按单词分割",
            "separators": [" ", "\n", ""],
        },
    ]
    
    for config in separator_configs:
        print(f"\n分隔符配置：{config['name']}")
        print(f"  separators: {config['separators']}")
        print("-" * 80)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=config["separators"],
            length_function=len,
        )
        
        split_docs = splitter.split_documents(docs)
        print(f"分割后得到 {len(split_docs)} 个文档块")
        
        # 显示前两个块的内容
        for i, doc in enumerate(split_docs[:2], start=1):
            print(f"\n块 {i}（长度：{len(doc.page_content)} 字符）：")
            print(f"{doc.page_content[:120]}...")
    print()


def textloader_encoding_demo() -> None:
    """
    演示 TextLoader 的 encoding 参数。

    展示不同编码对文件加载的影响。
    """
    print("=" * 80)
    print("【示例5】TextLoader - encoding 参数")
    print("-" * 80)
    
    # 创建不同编码的测试文件
    test_files = []
    
    # UTF-8 编码文件
    utf8_file = "./data/test_utf8.txt"
    os.makedirs("./data", exist_ok=True)
    with open(utf8_file, "w", encoding="utf-8") as f:
        f.write("这是 UTF-8 编码的测试文件\n包含中文：你好世界\n包含英文：Hello World")
    test_files.append(("UTF-8", utf8_file, "utf-8"))
    
    print("测试不同编码的文件加载：")
    print("-" * 80)
    
    for encoding_name, file_path, encoding in test_files:
        print(f"\n编码：{encoding_name}")
        print(f"文件：{file_path}")
        try:
            loader = TextLoader(
                file_path,
                encoding=encoding,
            )
            docs = loader.load()
            print(f"✓ 成功加载，文档内容：")
            print(f"  {docs[0].page_content[:100]}...")
        except Exception as e:
            print(f"✗ 加载失败：{e}")
    print()


def splitter_metadata_demo() -> None:
    """
    演示分割后文档的元数据。

    展示分割器如何保留和更新文档元数据。
    """
    print("=" * 80)
    print("【示例6】文档分割后的元数据")
    print("-" * 80)
    
    # 加载文档
    text_file = "./data/Python_Basics.txt"
    if not os.path.exists(text_file):
        print(f"错误：找不到文件 {text_file}，请先运行示例1")
        return
    
    loader = TextLoader(
        text_file,
        encoding="utf-8",
    )
    docs = loader.load()
    print(f"原始文档元数据：{docs[0].metadata}\n")
    
    # 创建分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", " ", ""],
        length_function=len,
    )
    
    # 分割文档
    split_docs = splitter.split_documents(docs)
    
    print(f"分割后得到 {len(split_docs)} 个文档块\n")
    print("前 3 个文档块的元数据：")
    print("-" * 80)
    for i, doc in enumerate(split_docs[:3], start=1):
        print(f"\n文档块 {i}：")
        print(f"  元数据：{doc.metadata}")
        print(f"  内容长度：{len(doc.page_content)} 字符")
        print(f"  内容预览：{doc.page_content[:80]}...")
    print()


def main() -> None:
    """
    主函数：演示 TextLoader 和 RecursiveCharacterTextSplitter 的各种使用方法。
    """
    print("=" * 80)
    print("LangChain TextLoader 和文档分割器示例")
    print("=" * 80)
    print()

    # 加载环境变量（虽然 TextLoader 不需要 API Key，但保持一致性）
    load_dotenv()

    # 示例1：TextLoader 基本用法
    textloader_basic_demo()

    # 示例2：RecursiveCharacterTextSplitter 基本用法
    recursive_character_splitter_basic_demo()

    # 示例3：参数对比
    recursive_character_splitter_parameters_demo()

    # 示例4：separators 参数影响
    recursive_character_splitter_separators_demo()

    # 示例5：encoding 参数
    textloader_encoding_demo()

    # 示例6：文档分割后的元数据
    splitter_metadata_demo()

    print("=" * 80)
    print("演示结束")
    print("=" * 80)
    print("\n提示：")
    print("- 示例文本文件已创建在 ./data/Python_Basics.txt")
    print("- RecursiveCharacterTextSplitter 是 LangChain 官方推荐的默认字符分割器")
    print("- 它在保持上下文完整性和控制片段大小之间实现了良好平衡")
    print("- 更多文档加载器请参考：https://docs.langchain.com/oss/python/integrations/document_loaders")
    print("- 更多文本分割器请参考：https://docs.langchain.com/oss/python/modules/data_connection/text_splitters/")


if __name__ == "__main__":
    main()
