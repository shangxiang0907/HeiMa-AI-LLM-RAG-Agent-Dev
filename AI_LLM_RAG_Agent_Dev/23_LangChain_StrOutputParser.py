"""
LangChain StrOutputParser 字符串输出解析器示例

本示例对应课件中关于 StrOutputParser 的图片，重点演示：

1. 问题场景：当尝试构建 `chain = prompt | model | model` 时，会遇到类型不匹配错误
   - prompt 的输出是 PromptValue 类型
   - model 的输出是 AIMessage 类型
   - 第二个 model 的 invoke 方法期望的输入类型是 LanguageModelInput
     (即 PromptValue | str | Sequence[MessageLikeRepresentation])
   - 但实际接收到的是 AIMessage 类型，导致 ValueError

2. 解决方案：使用 StrOutputParser 进行类型转换
   - StrOutputParser 是 LangChain 内置的简单字符串解析器
   - 可以将 AIMessage 解析为简单的字符串
   - 是 Runnable 接口的子类，可以加入链
   - 正确的链式写法：`chain = prompt | model | parser | model`

3. 实际应用场景：当需要将第一个模型的输出作为第二个模型的输入时

核心概念：
- StrOutputParser：将 AIMessage 转换为字符串的解析器
- LanguageModelInput：模型输入的类型约束
- 类型转换：在链式调用中处理不同组件之间的类型不匹配问题
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


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


def demo_error_scenario() -> None:
    """
    演示错误的链式调用场景：prompt | model | model

    这个示例会展示为什么直接连接两个 model 会报错。
    注意：为了演示错误，这里会捕获异常并展示错误信息。
    """
    print("=" * 80)
    print("【示例一】错误的链式调用：prompt | model | model")
    print("=" * 80)

    # 创建提示词模板（与课件中的示例一致）
    prompt = PromptTemplate.from_template(
        "我邻居姓:{lastname}, 刚生了{gender},请起名,仅告知名字无需其它内容"
    )

    # 创建模型实例
    model = init_chat_model()

    # 尝试构建错误的链：prompt | model | model
    print("\n尝试构建链：chain = prompt | model | model")
    try:
        chain = prompt | model | model
        print("链构建成功（这不应该发生）")
    except Exception as e:
        print(f"链构建时出错：{type(e).__name__}: {e}")
        print("\n注意：实际上链的构建可能不会立即报错，")
        print("但在调用 invoke 时会出现类型不匹配的错误。")

    # 尝试调用链（这里会报错）
    print("\n尝试调用链：chain.invoke({'lastname': '张', 'gender': '女儿'})")
    try:
        chain = prompt | model | model
        res = chain.invoke({"lastname": "张", "gender": "女儿"})
        print(f"结果：{res.content}")
    except ValueError as e:
        print(f"\n❌ 错误类型：{type(e).__name__}")
        print(f"❌ 错误信息：{e}")
        print("\n错误原因分析：")
        print("1. prompt 的输出是 PromptValue 类型，可以正常输入给第一个 model")
        print("2. 第一个 model 的输出是 AIMessage 类型")
        print("3. 第二个 model 的 invoke 方法期望的输入类型是：")
        print("   LanguageModelInput = PromptValue | str | Sequence[MessageLikeRepresentation]")
        print("4. 但实际接收到的是 AIMessage 类型，导致类型不匹配")
    except Exception as e:
        print(f"\n❌ 其他错误：{type(e).__name__}: {e}")

    print()


def demo_str_output_parser_basic() -> None:
    """
    演示 StrOutputParser 的基本用法。

    展示：
    1. StrOutputParser 的创建
    2. StrOutputParser 如何将 AIMessage 转换为字符串
    3. StrOutputParser 是 Runnable 的子类，可以加入链
    """
    print("=" * 80)
    print("【示例二】StrOutputParser 基本用法")
    print("=" * 80)

    # 创建 StrOutputParser 实例
    parser = StrOutputParser()
    print("\n1. 创建 StrOutputParser 实例：")
    print(f"   parser = StrOutputParser()")
    print(f"   parser 的类型：{type(parser)}")
    print(f"   parser 是否是 Runnable 的子类：{hasattr(parser, 'invoke')}")

    # 创建一个 AIMessage 对象（模拟模型的输出）
    ai_message = AIMessage(content="张雨萱")
    print("\n2. 模拟模型的输出（AIMessage 对象）：")
    print(f"   ai_message = AIMessage(content='张雨萱')")
    print(f"   ai_message 的类型：{type(ai_message)}")
    print(f"   ai_message.content：{ai_message.content}")

    # 使用 StrOutputParser 解析 AIMessage
    print("\n3. 使用 StrOutputParser 解析 AIMessage：")
    parsed_result = parser.invoke(ai_message)
    print(f"   parsed_result = parser.invoke(ai_message)")
    print(f"   parsed_result 的类型：{type(parsed_result)}")
    print(f"   parsed_result 的值：{parsed_result}")
    print("\n结论：StrOutputParser 可以将 AIMessage 解析为简单的字符串。")
    print()


def demo_correct_chain_with_parser() -> None:
    """
    演示正确的链式调用：prompt | model | parser | model

    这是课件中展示的解决方案，使用 StrOutputParser 进行类型转换。
    """
    print("=" * 80)
    print("【示例三】正确的链式调用：prompt | model | parser | model")
    print("=" * 80)

    # 创建提示词模板（与课件中的示例一致）
    prompt = PromptTemplate.from_template(
        "我邻居姓:{lastname}, 刚生了{gender},请起名,仅告知名字无需其它内容"
    )

    # 创建模型实例
    model = init_chat_model()

    # 创建 StrOutputParser 实例
    parser = StrOutputParser()

    # 构建正确的链：prompt | model | parser | model
    print("\n构建链：chain = prompt | model | parser | model")
    chain = prompt | model | parser | model
    print(f"链构建成功！chain 的类型：{type(chain)}")

    print("\n链的执行流程：")
    print("1. prompt 接收变量字典，输出 PromptValue")
    print("2. 第一个 model 接收 PromptValue，输出 AIMessage")
    print("3. parser 接收 AIMessage，输出 str（字符串）")
    print("4. 第二个 model 接收 str，输出 AIMessage")

    # 先获取第一个模型的输出，以便展示完整流程
    # 注意：这里单独执行前半部分链仅用于演示，实际应用中应该直接执行完整链
    print("\n先执行前半部分链，获取第一个模型的输出（仅供演示，实际不是同一条调用链）：")
    first_chain = prompt | model
    first_result = first_chain.invoke({"lastname": "张", "gender": "女儿"})
    print(f"第一个模型的输出：{first_result.content}")

    # 调用完整链
    print("\n调用完整链：chain.invoke({'lastname': '张', 'gender': '女儿'})")
    print("=" * 80)
    res = chain.invoke({"lastname": "张", "gender": "女儿"})
    print(f"\n✅ 成功！第二个模型的输出：")
    print(res.content)
    print("\n" + "=" * 80)
    print()


def demo_chain_components_analysis() -> None:
    """
    分析链中各组件的输入输出类型。

    帮助理解为什么需要 StrOutputParser 进行类型转换。
    """
    print("=" * 80)
    print("【示例四】链中各组件的输入输出类型分析")
    print("=" * 80)

    prompt = PromptTemplate.from_template("测试：{text}")
    model = init_chat_model()
    parser = StrOutputParser()

    # 分析 prompt 的输入输出
    print("\n1. PromptTemplate 组件：")
    print("   - 输入类型：Dict[str, Any]（变量字典）")
    print("   - 输出类型：PromptValue")
    prompt_result = prompt.invoke({"text": "测试"})
    print(f"   - 实际输出类型：{type(prompt_result)}")

    # 分析 model 的输入输出
    print("\n2. ChatTongyi 模型组件：")
    print("   - 输入类型：LanguageModelInput")
    print("     (即 PromptValue | str | Sequence[MessageLikeRepresentation])")
    print("   - 输出类型：AIMessage")
    model_result = model.invoke(prompt_result)
    print(f"   - 实际输出类型：{type(model_result)}")
    print(f"   - 实际输出内容：{model_result.content[:50]}...")

    # 分析 parser 的输入输出
    print("\n3. StrOutputParser 组件：")
    print("   - 输入类型：AIMessage（或其他可解析的消息类型）")
    print("   - 输出类型：str（字符串）")
    parser_result = parser.invoke(model_result)
    print(f"   - 实际输出类型：{type(parser_result)}")
    print(f"   - 实际输出内容：{parser_result}")

    # 分析为什么需要 parser
    print("\n4. 为什么需要 StrOutputParser：")
    print("   - 第一个 model 输出 AIMessage 类型")
    print("   - 第二个 model 需要 LanguageModelInput 类型（可以是 str）")
    print("   - StrOutputParser 将 AIMessage 转换为 str，满足第二个 model 的要求")
    print("   - 因此：prompt | model | parser | model 可以正常工作")

    print()


def demo_practical_use_case() -> None:
    """
    演示实际应用场景：使用第一个模型的输出作为第二个模型的输入。

    场景：第一个模型生成名字，第二个模型对名字进行评价。
    """
    print("=" * 80)
    print("【示例五】实际应用场景：两阶段模型调用")
    print("=" * 80)

    # 第一个提示词：生成名字
    name_prompt = PromptTemplate.from_template(
        "我邻居姓:{lastname}, 刚生了{gender},请起名,仅告知名字无需其它内容"
    )

    # 第二个提示词：评价名字
    review_prompt = PromptTemplate.from_template(
        "请对以下名字进行简短评价，说明其寓意和特点：{name}"
    )

    model = init_chat_model()
    parser = StrOutputParser()

    # 构建两阶段链
    print("\n构建两阶段链：")
    print("阶段1：生成名字")
    print("阶段2：评价名字")
    print()

    # 阶段1：生成名字
    name_chain = name_prompt | model | parser
    print("阶段1 - 生成名字：")
    name = name_chain.invoke({"lastname": "张", "gender": "女儿"})
    print(f"   生成的名字：{name}")

    # 阶段2：评价名字
    review_chain = review_prompt | model
    print("\n阶段2 - 评价名字：")
    review = review_chain.invoke({"name": name})
    print(f"   评价结果：{review.content}")

    # 或者使用一个完整的链
    print("\n" + "-" * 80)
    print("或者，使用一个完整的链：")
    print("-" * 80)
    full_chain = name_prompt | model | parser | review_prompt | model
    print("\n完整链：name_prompt | model | parser | review_prompt | model")
    print("执行完整链：")
    final_result = full_chain.invoke({"lastname": "张", "gender": "女儿"})
    print(f"最终结果：{final_result.content}")
    print()


def main() -> None:
    """
    入口函数：演示 StrOutputParser 的用法和重要性。

    本示例分为五个部分：
    1. 展示错误的链式调用场景
    2. 展示 StrOutputParser 的基本用法
    3. 展示正确的链式调用（使用 StrOutputParser）
    4. 分析链中各组件的输入输出类型
    5. 演示实际应用场景
    """
    print("=" * 80)
    print("LangChain StrOutputParser 字符串输出解析器示例")
    print("=" * 80)
    print()

    # 示例一：错误的链式调用场景
    demo_error_scenario()

    # 示例二：StrOutputParser 基本用法
    demo_str_output_parser_basic()

    # 示例三：正确的链式调用
    demo_correct_chain_with_parser()

    # 示例四：链中各组件的输入输出类型分析
    demo_chain_components_analysis()

    # 示例五：实际应用场景
    demo_practical_use_case()

    print("=" * 80)
    print("全部示例执行完毕。")
    print("=" * 80)
    print("\n总结：")
    print("1. 当需要将第一个模型的输出作为第二个模型的输入时，")
    print("   不能直接使用 prompt | model | model，因为类型不匹配")
    print("2. StrOutputParser 是 LangChain 内置的字符串解析器，")
    print("   可以将 AIMessage 解析为简单的字符串")
    print("3. StrOutputParser 是 Runnable 的子类，可以加入链中")
    print("4. 正确的写法：chain = prompt | model | parser | model")
    print("=" * 80)


if __name__ == "__main__":
    main()
