"""
LangChain JSONLoader 文档加载器示例

本示例对应课件中关于 JSONLoader 的内容，重点演示：

1. JSONLoader 简介：
   - JSONLoader 用于将 JSON 数据加载为 Document 类型对象
   - 需要额外安装：pip install jq
   - jq 是跨平台的 JSON 解析工具，LangChain 底层使用 jq 进行 JSON 解析
   - 使用 jq_schema 语法来提取 JSON 中的信息

2. jq_schema 基本语法：
   - `.` 表示整个 JSON 对象（根）
   - `[]` 表示数组
   - `.name` 表示提取 name 字段
   - `.hobby` 表示提取 hobby 数组
   - `.hobby[1]` 或 `.hobby.[1]` 表示提取 hobby 数组的第二个元素
   - `.other.addr` 表示提取嵌套对象 other 中的 addr 字段
   - `.[]` 表示遍历数组中的每个元素
   - `.[].name` 表示提取数组中所有对象的 name 字段

3. JSONLoader 参数：
   - file_path: 文件路径
   - jq_schema: jq schema 语法，用于指定提取规则
   - text_content: 抽取的是否是字符串，默认 True（False 时提取为 Python 对象）
   - json_lines: 是否是 JsonLines 文件（每一行都是 JSON 的文件），默认 False

4. JsonLines 格式：
   - JsonLines 是一种格式，每行都是一个独立的 JSON 对象
   - 适合处理大型数据集，可以逐行读取而不需要一次性加载整个文件

核心概念：
- JSONLoader：用于加载 JSON 文件的文档加载器
- jq_schema：jq 语法，用于从 JSON 中提取特定数据
- JsonLines：每行一个 JSON 对象的文件格式
- Document：LangChain 文档的统一载体类
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document


def jq_schema_basic_demo() -> None:
    """
    演示 jq_schema 基本语法。

    对应课件中的第一个示例，展示如何从单个 JSON 对象中提取数据。
    """
    print("=" * 80)
    print("【示例1】jq_schema 基本语法 - 单个 JSON 对象")
    print("-" * 80)

    # 创建示例 JSON 文件
    json_file = "./stu.json"
    print(f"创建示例 JSON 文件：{json_file}")
    with open(json_file, "w", encoding="utf-8") as f:
        f.write('''{
  "name": "周杰轮",
  "age": 11,
  "hobby": ["唱", "跳", "RAP"],
  "other": {
    "addr": "深圳",
    "tel": "12332112321"
    }
}''')
    print(f"示例 JSON 文件已创建：{json_file}\n")

    # 不同的 jq_schema 示例
    schemas = [
        (".", "提取整个 JSON 对象"),
        (".name", "提取 name 字段（周杰轮）"),
        (".hobby", "提取 hobby 数组"),
        (".hobby[1]", "提取 hobby 数组的第二个元素（跳）"),
        (".other.addr", "提取嵌套对象 other 中的 addr 字段（深圳）"),
    ]

    for jq_schema, description in schemas:
        print(f"\n{jq_schema} - {description}")
        print("-" * 80)
        try:
            loader = JSONLoader(
                file_path=json_file,
                jq_schema=jq_schema,
                text_content=False,  # 设置为 False 以便查看原始对象
            )
            documents = loader.load()
            print(f"成功加载 {len(documents)} 个文档")
            for i, doc in enumerate(documents, start=1):
                print(f"文档 {i}:")
                print(f"  page_content: {doc.page_content}")
                print(f"  metadata: {doc.metadata}")
        except Exception as e:
            print(f"错误：{e}")
    print()


def jsonloader_array_demo() -> None:
    """
    演示从 JSON 数组中提取数据。

    对应课件中的第二个示例，展示如何从 JSON 数组中提取数据。
    """
    print("=" * 80)
    print("【示例2】JSONLoader - JSON 数组提取")
    print("-" * 80)

    # 创建示例 JSON 数组文件
    json_file = "./stus.json"
    print(f"创建示例 JSON 数组文件：{json_file}")
    with open(json_file, "w", encoding="utf-8") as f:
        f.write('''[
  {"name": "周杰轮", "age": 11, "gender": "男"},
  {"name": "蔡依临", "age": 12, "gender": "女"},
  {"name": "王力鸿", "age": 11, "gender": "男"}
]''')
    print(f"示例 JSON 数组文件已创建：{json_file}\n")

    # 不同的 jq_schema 示例
    schemas = [
        (".[]", "提取数组中的每个对象（得到 3 个字典）"),
        (".[].name", "提取数组中所有对象的 name 字段（得到 3 个名字）"),
        (".[].age", "提取数组中所有对象的 age 字段"),
    ]

    for jq_schema, description in schemas:
        print(f"\n{jq_schema} - {description}")
        print("-" * 80)
        try:
            loader = JSONLoader(
                file_path=json_file,
                jq_schema=jq_schema,
                text_content=False,
            )
            documents = loader.load()
            print(f"成功加载 {len(documents)} 个文档")
            for i, doc in enumerate(documents, start=1):
                print(f"文档 {i}:")
                print(f"  page_content: {doc.page_content}")
                print(f"  metadata: {doc.metadata}")
        except Exception as e:
            print(f"错误：{e}")
    print()


def jsonloader_jsonlines_demo() -> None:
    """
    演示 JsonLines 格式文件的加载。

    对应课件中的第三个示例，展示如何使用 JSONLoader 加载 JsonLines 格式的文件。
    """
    print("=" * 80)
    print("【示例3】JSONLoader - JsonLines 格式")
    print("-" * 80)

    # 创建示例 JsonLines 文件
    json_lines_file = "./stu_json_lines.json"
    print(f"创建示例 JsonLines 文件：{json_lines_file}")
    print("JsonLines 格式：每行都是一个独立的 JSON 对象")
    with open(json_lines_file, "w", encoding="utf-8") as f:
        f.write('{"name":"周杰轮","age": 11, "gender":"男"}\n')
        f.write('{"name":"蔡依临","age": 12, "gender":"女"}\n')
        f.write('{"name":"王力鸿","age": 11, "gender":"男"}\n')
    print(f"示例 JsonLines 文件已创建：{json_lines_file}\n")

    # 使用 JSONLoader 加载 JsonLines 文件
    print("使用 JSONLoader 加载 JsonLines 文件：")
    print("-" * 80)
    loader = JSONLoader(
        file_path=json_lines_file,
        jq_schema=".",  # 提取每行的整个 JSON 对象
        text_content=False,
        json_lines=True,  # 重要：指定这是 JsonLines 格式
    )

    documents = loader.load()
    print(f"成功加载 {len(documents)} 个文档")
    print("\n所有文档内容：")
    for i, doc in enumerate(documents, start=1):
        print(f"\n文档 {i}:")
        print(f"  page_content: {doc.page_content}")
        print(f"  metadata: {doc.metadata}")
    print()


def jsonloader_text_content_demo() -> None:
    """
    演示 text_content 参数的区别。

    text_content=True（默认）：将提取的内容转换为字符串
    text_content=False：保持提取的内容为 Python 对象（字典、列表等）
    
    注意：当 jq_schema 提取的是复杂对象（字典、数组）时，必须设置 text_content=False，
    否则会报错：ValueError: Expected page_content is string, got <class 'dict'> instead.
    """
    print("=" * 80)
    print("【示例4】JSONLoader - text_content 参数区别")
    print("-" * 80)

    json_file = "./stu.json"
    if not os.path.exists(json_file):
        print(f"错误：找不到文件 {json_file}，请先运行示例1")
        return

    # 对于提取简单字符串值的情况，可以使用 text_content=True
    print("\n1. text_content=True - 提取简单字符串值")
    print("-" * 80)
    print("使用 jq_schema='.name' 提取字符串字段")
    try:
        loader1 = JSONLoader(
            file_path=json_file,
            jq_schema=".name",  # 提取字符串值
            text_content=True,  # 可以设置为 True，因为提取的是字符串
        )
        documents1 = loader1.load()
        print(f"成功加载 {len(documents1)} 个文档")
        if documents1:
            print(f"page_content 类型: {type(documents1[0].page_content)}")
            print(f"page_content: {documents1[0].page_content}")
    except Exception as e:
        print(f"错误：{e}")

    # 对于提取复杂对象的情况，必须使用 text_content=False
    print("\n2. text_content=False - 提取复杂对象（字典、数组）")
    print("-" * 80)
    print("使用 jq_schema='.' 提取整个 JSON 对象")
    loader2 = JSONLoader(
        file_path=json_file,
        jq_schema=".",  # 提取整个对象
        text_content=False,  # 必须设置为 False，因为提取的是字典
    )
    documents2 = loader2.load()
    print(f"成功加载 {len(documents2)} 个文档")
    if documents2:
        print(f"page_content 类型: {type(documents2[0].page_content)}")
        print(f"page_content: {documents2[0].page_content}")

    # 演示错误情况：当提取字典对象但 text_content=True 时会报错
    print("\n3. 错误示例：提取字典对象但 text_content=True")
    print("-" * 80)
    print("使用 jq_schema='.' 提取整个对象，但 text_content=True（会报错）")
    try:
        loader3 = JSONLoader(
            file_path=json_file,
            jq_schema=".",  # 提取整个对象（字典）
            text_content=True,  # 错误：提取的是字典，不能设置为 True
        )
        documents3 = loader3.load()
        print(f"成功加载 {len(documents3)} 个文档")
    except ValueError as e:
        print(f"预期的错误：{e}")
        print("解决方案：当 jq_schema 提取的是字典或数组时，必须设置 text_content=False")
    print()


def jsonloader_lazy_load_demo() -> None:
    """
    演示 JSONLoader 的 lazy_load() 方法。

    lazy_load() 方法用于延迟流式传输文档，对大型数据集很有用，避免内存溢出。
    """
    print("=" * 80)
    print("【示例5】JSONLoader lazy_load() 方法 - 延迟流式加载")
    print("-" * 80)

    json_lines_file = "./stu_json_lines.json"
    if not os.path.exists(json_lines_file):
        print(f"错误：找不到文件 {json_lines_file}，请先运行示例3")
        return

    loader = JSONLoader(
        file_path=json_lines_file,
        jq_schema=".",
        text_content=False,
        json_lines=True,
    )

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


def jsonloader_complex_schema_demo() -> None:
    """
    演示复杂的 jq_schema 用法。

    展示如何提取嵌套字段、数组元素等复杂结构。
    """
    print("=" * 80)
    print("【示例6】JSONLoader - 复杂 jq_schema 用法")
    print("-" * 80)

    json_file = "./stu.json"
    if not os.path.exists(json_file):
        print(f"错误：找不到文件 {json_file}，请先运行示例1")
        return

    # 创建更复杂的 JSON 文件用于演示
    complex_json_file = "./complex_stu.json"
    print(f"创建复杂 JSON 文件：{complex_json_file}")
    with open(complex_json_file, "w", encoding="utf-8") as f:
        f.write('''[
  {
    "name": "周杰轮",
    "age": 11,
    "hobby": ["唱", "跳", "RAP"],
    "other": {
      "addr": "深圳",
      "tel": "12332112321"
    }
  },
  {
    "name": "蔡依临",
    "age": 12,
    "hobby": ["唱", "跳"],
    "other": {
      "addr": "北京",
      "tel": "98765432100"
    }
  }
]''')
    print(f"复杂 JSON 文件已创建：{complex_json_file}\n")

    # 复杂的 jq_schema 示例
    schemas = [
        (".[].name", "提取所有对象的 name 字段"),
        (".[].hobby", "提取所有对象的 hobby 数组"),
        (".[].hobby[]", "提取所有对象的所有 hobby 元素（扁平化）"),
        (".[].other.addr", "提取所有对象的地址"),
        (".[] | {name: .name, addr: .other.addr}", "提取并组合多个字段"),
    ]

    for jq_schema, description in schemas:
        print(f"\n{jq_schema} - {description}")
        print("-" * 80)
        try:
            loader = JSONLoader(
                file_path=complex_json_file,
                jq_schema=jq_schema,
                text_content=False,
            )
            documents = loader.load()
            print(f"成功加载 {len(documents)} 个文档")
            for i, doc in enumerate(documents[:3], start=1):  # 只显示前 3 个
                print(f"文档 {i}:")
                print(f"  page_content: {doc.page_content}")
                print(f"  metadata: {doc.metadata}")
            if len(documents) > 3:
                print(f"\n... (还有 {len(documents) - 3} 个文档)")
        except Exception as e:
            print(f"错误：{e}")
    print()

    # 清理临时文件
    if os.path.exists(complex_json_file):
        os.remove(complex_json_file)
        print(f"已清理临时文件：{complex_json_file}\n")


def main() -> None:
    """
    主函数：演示 JSONLoader 的各种使用方法。
    """
    print("=" * 80)
    print("LangChain JSONLoader 文档加载器示例")
    print("=" * 80)
    print()

    # 加载环境变量（虽然 JSONLoader 不需要 API Key，但保持一致性）
    load_dotenv()

    # 示例1：jq_schema 基本语法
    jq_schema_basic_demo()

    # 示例2：JSON 数组提取
    jsonloader_array_demo()

    # 示例3：JsonLines 格式
    jsonloader_jsonlines_demo()

    # 示例4：text_content 参数区别
    jsonloader_text_content_demo()

    # 示例5：lazy_load() 方法
    jsonloader_lazy_load_demo()

    # 示例6：复杂 jq_schema 用法
    jsonloader_complex_schema_demo()

    print("=" * 80)
    print("演示结束")
    print("=" * 80)
    print("\n提示：")
    print("- 示例 JSON 文件已创建在当前目录：")
    print("  - stu.json（单个对象）")
    print("  - stus.json（数组对象）")
    print("  - stu_json_lines.json（JsonLines 格式）")
    print("- 可以手动编辑这些文件来测试不同的 JSON 格式")
    print("- 更多 jq 语法请参考：https://stedolan.github.io/jq/manual/")
    print("- 更多文档加载器请参考：https://docs.langchain.com/oss/python/integrations/document_loaders")


if __name__ == "__main__":
    main()
