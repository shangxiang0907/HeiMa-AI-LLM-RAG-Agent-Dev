"""
LangChain RunnableLambda 自定义函数加入链示例

本示例对应课件中关于 RunnableLambda 的图片，重点演示：

1. RunnableLambda 类的作用：
   - RunnableLambda 是 LangChain 内置的类
   - 将普通函数或 lambda 匿名函数转换为 Runnable 接口实例
   - 方便自定义函数加入 chain，实现更灵活的数据转换

2. 使用 RunnableLambda 的方式：
   - 显式使用：my_func = RunnableLambda(lambda x: {...})
   - 隐式使用：直接在链中使用 lambda 函数，会自动转换为 RunnableLambda

3. 为什么可以直接使用函数：
   - Runnable 接口在实现 `__or__` 方法时，支持 Callable 接口的实例
   - 函数就是 Callable 接口的实例
   - 本质是将函数自动转换为 RunnableLambda

核心概念：
- RunnableLambda：将函数转换为 Runnable 的包装器
- Callable：Python 中的可调用对象接口
- 自定义数据转换：在链中插入自定义的数据处理逻辑
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


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

    os.environ["DASHSCOPE_API_KEY"] = api_key
    chat = ChatTongyi(model="qwen3-max")
    return chat


def demo_runnable_lambda_introduction() -> None:
    """
    介绍 RunnableLambda 的基本概念和作用。

    说明：
    1. RunnableLambda 是什么
    2. 为什么需要 RunnableLambda
    3. 如何使用 RunnableLambda
    """
    print("=" * 80)
    print("【示例一】RunnableLambda 基本介绍")
    print("=" * 80)

    print("\n1. RunnableLambda 是什么：")
    print("   - RunnableLambda 是 LangChain 内置的类")
    print("   - 将普通函数或 lambda 匿名函数转换为 Runnable 接口实例")
    print("   - 方便自定义函数加入 chain，实现更灵活的数据转换")

    print("\n2. 为什么需要 RunnableLambda：")
    print("   - 除了 JsonOutputParser 这类固定功能的解析器之外")
    print("   - 我们也可以自己编写 Lambda 匿名函数来完成自定义逻辑的数据转换")
    print("   - 想怎么转换就怎么转换，更自由")

    print("\n3. 语法：")
    print("   RunnableLambda(函数对象或 lambda 匿名函数)")

    print("\n4. 示例：")
    print("   my_func = RunnableLambda(lambda ai_msg: {'name': ai_msg.content})")
    print("   这个函数将 AIMessage 对象转换为包含 'name' 键的字典")

    print()


def demo_runnable_lambda_basic() -> None:
    """
    演示 RunnableLambda 的基本用法。

    展示：
    1. 如何创建 RunnableLambda 实例
    2. RunnableLambda 如何将函数转换为 Runnable
    3. RunnableLambda 的输入输出
    """
    print("=" * 80)
    print("【示例二】RunnableLambda 基本用法")
    print("=" * 80)

    # 创建一个简单的 lambda 函数
    # 这个函数接收 AIMessage，提取 content 并包装成字典
    my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

    print("\n1. 创建 RunnableLambda 实例：")
    print("   my_func = RunnableLambda(lambda ai_msg: {'name': ai_msg.content})")
    print(f"   my_func 的类型：{type(my_func)}")
    print(f"   my_func 是否是 Runnable 的子类：{hasattr(my_func, 'invoke')}")

    # 创建一个 AIMessage 对象（模拟模型的输出）
    ai_message = AIMessage(content="张雨萱")
    print("\n2. 模拟模型的输出（AIMessage 对象）：")
    print(f"   ai_message = AIMessage(content='张雨萱')")
    print(f"   ai_message 的类型：{type(ai_message)}")
    print(f"   ai_message.content：{ai_message.content}")

    # 使用 RunnableLambda 处理 AIMessage
    print("\n3. 使用 RunnableLambda 处理 AIMessage：")
    result = my_func.invoke(ai_message)
    print(f"   result = my_func.invoke(ai_message)")
    print(f"   result 的类型：{type(result)}")
    print(f"   result 的值：{result}")
    print("\n结论：RunnableLambda 可以将 AIMessage 转换为自定义格式的字典。")

    print()


def demo_multi_model_chain_with_runnable_lambda() -> None:
    """
    演示使用 RunnableLambda 构建多模型链。

    这是课件中展示的完整示例：
    1. 第一个提示词：要求模型起名，仅告知名字
    2. 第一个模型：生成名字
    3. RunnableLambda：提取名字并包装成字典
    4. 第二个提示词：使用字典中的 name 字段
    5. 第二个模型：解析名字的含义
    6. StrOutputParser：将最终结果解析为字符串
    """
    print("=" * 80)
    print("【示例三】使用 RunnableLambda 构建多模型链（完整示例）")
    print("=" * 80)

    # 创建解析器实例
    str_parser = StrOutputParser()

    print("\n1. 创建解析器实例：")
    print(f"   str_parser = StrOutputParser()")

    # 创建模型实例
    model = init_chat_model()
    print(f"\n2. 创建模型实例：")
    print(f"   model = ChatTongyi(model='qwen3-max')")

    # 创建 RunnableLambda 实例
    # 这个函数接收 AIMessage，提取 content 并包装成字典
    my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

    print("\n3. 创建 RunnableLambda 实例：")
    print("   my_func = RunnableLambda(lambda ai_msg: {'name': ai_msg.content})")
    print("   作用：将 AIMessage 转换为包含 'name' 键的字典")

    # 创建第一个提示词模板
    first_prompt = PromptTemplate.from_template(
        "我邻居姓:{lastname},刚生了{gender},请起名,仅告知我名字,不要额外信息"
    )

    print("\n4. 创建第一个提示词模板：")
    print("   first_prompt = PromptTemplate.from_template(...)")
    print("   要求：仅告知名字，不要额外信息")

    # 创建第二个提示词模板
    second_prompt = PromptTemplate.from_template("姓名{name},请帮我解析含义。")

    print("\n5. 创建第二个提示词模板：")
    print("   second_prompt = PromptTemplate.from_template('姓名{name},请帮我解析含义。')")
    print("   使用第一个模型输出的 name 字段")

    # 构建链：first_prompt | model | my_func | second_prompt | model | str_parser
    chain = first_prompt | model | my_func | second_prompt | model | str_parser

    print("\n6. 构建链：")
    print("   chain = first_prompt | model | my_func | second_prompt | model | str_parser")
    print("\n   链的执行流程：")
    print("   1. first_prompt: 接收变量字典，输出 PromptValue")
    print("   2. 第一个 model: 接收 PromptValue，输出 AIMessage（名字）")
    print("   3. my_func: 接收 AIMessage，提取 content，输出 dict（{'name': '...'}）")
    print("   4. second_prompt: 接收 dict，使用 name 字段构建新的 PromptValue")
    print("   5. 第二个 model: 接收 PromptValue，输出 AIMessage（名字含义解析）")
    print("   6. str_parser: 接收 AIMessage，输出 str（最终字符串结果）")

    # 调用链
    print("\n7. 调用链：")
    print("   res = chain.invoke({'lastname': '张', 'gender': '女儿'})")
    print("=" * 80)
    res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
    print(f"\n✅ 成功！最终结果：")
    print(res)
    print(f"\n结果类型：{type(res)}")
    print("=" * 80)
    print()


def demo_direct_lambda_in_chain() -> None:
    """
    演示直接在链中使用 lambda 函数（不需要显式使用 RunnableLambda）。

    说明：
    1. 可以直接在链中使用 lambda 函数
    2. 因为 Runnable 接口的 `__or__` 方法支持 Callable 接口
    3. 函数会自动转换为 RunnableLambda
    """
    print("=" * 80)
    print("【示例四】直接在链中使用 lambda 函数")
    print("=" * 80)

    print("\n1. 说明：")
    print("   跳过 RunnableLambda 类，直接让函数加入链也是可以的。")
    print("   因为 Runnable 接口类在实现 `__or__` 的时候，支持 Callable 接口的实例。")
    print("   函数就是 Callable 接口的实例。")

    # 创建组件
    str_parser = StrOutputParser()
    model = init_chat_model()
    first_prompt = PromptTemplate.from_template(
        "我邻居姓:{lastname},刚生了{gender},请起名,仅告知我名字,不要额外信息"
    )
    second_prompt = PromptTemplate.from_template("姓名{name},请帮我解析含义。")

    print("\n2. 创建组件：")
    print("   str_parser = StrOutputParser()")
    print("   model = ChatTongyi(model='qwen3-max')")
    print("   first_prompt = PromptTemplate.from_template(...)")
    print("   second_prompt = PromptTemplate.from_template(...)")

    # 直接在链中使用 lambda 函数
    print("\n3. 直接在链中使用 lambda 函数：")
    print("   chain = first_prompt | model | (lambda ai_msg: {'name': ai_msg.content}) | second_prompt | model | str_parser")
    print("\n   注意：lambda 函数直接放在链中，不需要显式使用 RunnableLambda")
    print("   其本质是将函数自动转换为 RunnableLambda")

    chain = (
        first_prompt
        | model
        | (lambda ai_msg: {"name": ai_msg.content})
        | second_prompt
        | model
        | str_parser
    )

    # 调用链
    print("\n4. 调用链：")
    print("   res = chain.invoke({'lastname': '张', 'gender': '女儿'})")
    print("=" * 80)
    res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
    print(f"\n✅ 成功！最终结果：")
    print(res)
    print(f"\n结果类型：{type(res)}")
    print("=" * 80)
    print()


def demo_comparison_runnable_lambda_vs_direct_lambda() -> None:
    """
    对比显式使用 RunnableLambda 和直接使用 lambda 函数的区别。

    展示两种方式的等价性。
    """
    print("=" * 80)
    print("【示例五】RunnableLambda vs 直接使用 lambda 函数")
    print("=" * 80)

    # 创建组件
    str_parser = StrOutputParser()
    model = init_chat_model()
    first_prompt = PromptTemplate.from_template(
        "我邻居姓:{lastname},刚生了{gender},请起名,仅告知我名字,不要额外信息"
    )
    second_prompt = PromptTemplate.from_template("姓名{name},请帮我解析含义。")

    # 方式1：显式使用 RunnableLambda
    print("\n方式1：显式使用 RunnableLambda")
    my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})
    chain1 = first_prompt | model | my_func | second_prompt | model | str_parser
    print("   my_func = RunnableLambda(lambda ai_msg: {'name': ai_msg.content})")
    print("   chain1 = first_prompt | model | my_func | second_prompt | model | str_parser")

    # 方式2：直接使用 lambda 函数
    print("\n方式2：直接使用 lambda 函数")
    chain2 = (
        first_prompt
        | model
        | (lambda ai_msg: {"name": ai_msg.content})
        | second_prompt
        | model
        | str_parser
    )
    print("   chain2 = first_prompt | model | (lambda ai_msg: {'name': ai_msg.content}) | second_prompt | model | str_parser")

    print("\n两种方式的结果对比：")
    input_data = {"lastname": "张", "gender": "女儿"}

    print("\n方式1 的结果：")
    print("=" * 80)
    res1 = chain1.invoke(input_data)
    print(res1)
    print("=" * 80)

    print("\n方式2 的结果：")
    print("=" * 80)
    res2 = chain2.invoke(input_data)
    print(res2)
    print("=" * 80)

    print("\n结论：")
    print("- 两种方式功能完全等价")
    print("- 显式使用 RunnableLambda 更清晰，适合复杂函数")
    print("- 直接使用 lambda 函数更简洁，适合简单转换")
    print("- 本质都是将函数转换为 RunnableLambda")
    print()


def demo_custom_transformation_examples() -> None:
    """
    演示更多自定义数据转换的示例。

    展示不同的转换场景：
    1. 提取并重命名字段
    2. 添加额外字段
    3. 格式化输出
    """
    print("=" * 80)
    print("【示例六】更多自定义数据转换示例")
    print("=" * 80)

    model = init_chat_model()
    str_parser = StrOutputParser()

    # 示例1：提取并重命名字段
    print("\n示例1：提取并重命名字段")
    print("   将 AIMessage 的 content 提取为 'generated_name' 字段")
    prompt1 = PromptTemplate.from_template(
        "我邻居姓:{lastname},刚生了{gender},请起名,仅告知我名字,不要额外信息"
    )
    transform1 = RunnableLambda(
        lambda ai_msg: {"generated_name": ai_msg.content}
    )
    prompt2 = PromptTemplate.from_template("名字{generated_name}的含义是什么？")
    chain1 = prompt1 | model | transform1 | prompt2 | model | str_parser
    res1 = chain1.invoke({"lastname": "李", "gender": "儿子"})
    print(f"   结果：{res1[:100]}...")

    # 示例2：添加额外字段
    print("\n示例2：添加额外字段")
    print("   在提取名字的同时，添加时间戳和来源信息")
    from datetime import datetime

    transform2 = RunnableLambda(
        lambda ai_msg: {
            "name": ai_msg.content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "AI生成",
        }
    )
    prompt3 = PromptTemplate.from_template(
        "名字{name}（生成时间：{timestamp}，来源：{source}）的含义是什么？"
    )
    chain2 = prompt1 | model | transform2 | prompt3 | model | str_parser
    res2 = chain2.invoke({"lastname": "王", "gender": "女儿"})
    print(f"   结果：{res2[:100]}...")

    # 示例3：格式化输出
    print("\n示例3：格式化输出")
    print("   将名字格式化为特定结构")
    transform3 = RunnableLambda(
        lambda ai_msg: {
            "full_name": f"张{ai_msg.content}",
            "first_name": ai_msg.content,
        }
    )
    prompt4 = PromptTemplate.from_template(
        "请分析全名{full_name}中名字{first_name}部分的含义。"
    )
    chain3 = prompt1 | model | transform3 | prompt4 | model | str_parser
    res3 = chain3.invoke({"lastname": "张", "gender": "女儿"})
    print(f"   结果：{res3[:100]}...")

    print("\n总结：")
    print("- RunnableLambda 提供了极大的灵活性")
    print("- 可以根据需求自定义任何数据转换逻辑")
    print("- 比固定功能的解析器（如 JsonOutputParser）更自由")
    print()


def main() -> None:
    """
    入口函数：演示 RunnableLambda 的用法和重要性。

    本示例分为六个部分：
    1. RunnableLambda 基本介绍
    2. RunnableLambda 基本用法
    3. 使用 RunnableLambda 构建多模型链（完整示例）
    4. 直接在链中使用 lambda 函数
    5. RunnableLambda vs 直接使用 lambda 函数
    6. 更多自定义数据转换示例
    """
    print("=" * 80)
    print("LangChain RunnableLambda 自定义函数加入链示例")
    print("=" * 80)
    print()

    # 示例一：RunnableLambda 基本介绍
    demo_runnable_lambda_introduction()

    # 示例二：RunnableLambda 基本用法
    demo_runnable_lambda_basic()

    # 示例三：使用 RunnableLambda 构建多模型链（完整示例）
    demo_multi_model_chain_with_runnable_lambda()

    # 示例四：直接在链中使用 lambda 函数
    demo_direct_lambda_in_chain()

    # 示例五：RunnableLambda vs 直接使用 lambda 函数
    demo_comparison_runnable_lambda_vs_direct_lambda()

    # 示例六：更多自定义数据转换示例
    demo_custom_transformation_examples()

    print("=" * 80)
    print("全部示例执行完毕。")
    print("=" * 80)
    print("\n总结：")
    print("1. RunnableLambda 是 LangChain 内置的类，将函数转换为 Runnable 接口实例")
    print("2. 除了固定功能的解析器（如 JsonOutputParser），")
    print("   我们也可以使用 RunnableLambda 完成自定义逻辑的数据转换")
    print("3. 可以直接在链中使用 lambda 函数，不需要显式使用 RunnableLambda")
    print("4. Runnable 接口的 `__or__` 方法支持 Callable 接口，")
    print("   函数会自动转换为 RunnableLambda")
    print("5. RunnableLambda 提供了极大的灵活性，可以根据需求自定义任何数据转换逻辑")
    print("=" * 80)


if __name__ == "__main__":
    main()
