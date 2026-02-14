"""
LangChain JsonOutputParser JSON输出解析器示例

本示例对应课件中关于 JsonOutputParser 的图片，重点演示：

1. 问题场景：构建多模型链时的标准处理逻辑
   - 非标准做法：chain = prompt | model | parser | model | parser
     上一个模型的输出没有被处理就输入下一个模型
   - 标准做法：invoke | stream 初始输入 → 提示词模板 → 模型 → 数据处理 → 
     提示词模板 → 模型 → 解析器 → 结果
     上一个模型的输出结果应该作为提示词模板的输入，构建下一个提示词，用来二次调用模型

2. 类型转换需求：
   - 模型的输出为：AIMessage类对象
   - 提示词模板要求输入为：dict类型（如右侧代码所示）
   - 所以需要完成：将模型输出的AIMessage → 转为字典 → 注入第二个提示词模板中，
     形成新的提示词(PromptValue对象)

3. 解决方案：使用 JsonOutputParser
   - StrOutputParser不满足 (AIMessage → Str)
   - 更换JsonOutputParser (AIMessage → Dict(JSON))

核心概念：
- JsonOutputParser：将 AIMessage 转换为字典（JSON格式）的解析器
- 多模型链：第一个模型生成JSON格式输出，第二个模型基于JSON数据进行处理
- 数据处理：在链式调用中，需要将第一个模型的输出转换为第二个提示词模板所需的格式
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
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


def demo_non_standard_chaining() -> None:
    """
    演示非标准的链式构建方式。

    展示为什么直接连接模型和解析器不是标准做法。
    """
    print("=" * 80)
    print("【示例一】非标准的链式构建方式")
    print("=" * 80)

    print("\n非标准做法：chain = prompt | model | parser | model | parser")
    print("问题：上一个模型的输出，没有被处理就输入下一个模型。")
    print("\n标准做法应该是：")
    print("invoke | stream 初始输入 → 提示词模板 → 模型 → 数据处理 → ")
    print("提示词模板 → 模型 → 解析器 → 结果")
    print("\n即：")
    print("上一个模型的输出结果，应该作为提示词模板的输入，")
    print("构建下一个提示词，用来二次调用模型。")
    print()


def demo_type_requirements() -> None:
    """
    演示类型转换需求。

    展示：
    1. 模型的输出为 AIMessage 类对象
    2. 提示词模板要求输入为 dict 类型
    3. 需要将 AIMessage 转换为字典
    """
    print("=" * 80)
    print("【示例二】类型转换需求分析")
    print("=" * 80)

    # 创建提示词模板
    prompt = PromptTemplate.from_template("测试：{name}")
    
    print("\n1. 模型的输出类型：")
    model = init_chat_model()
    test_result = model.invoke("测试")
    print(f"   - 模型输出类型：{type(test_result)}")
    print(f"   - 模型输出内容：{test_result.content[:50]}...")
    print(f"   - 模型输出是 AIMessage 类对象")

    print("\n2. 提示词模板的输入要求：")
    print("   - 提示词模板的 invoke 方法签名：")
    print("     def invoke(self, input: dict, config: RunnableConfig | None = None, **kwargs: Any) -> PromptValue")
    print("   - 要求输入类型：dict（字典）")
    print("   - 返回类型：PromptValue")

    print("\n3. 问题：")
    print("   模型的输出是 AIMessage 类型，但提示词模板需要 dict 类型")
    print("   所以需要完成：")
    print("   将模型输出的AIMessage → 转为字典 → 注入第二个提示词模板中，")
    print("   形成新的提示词(PromptValue对象)")

    print("\n4. 解决方案：")
    print("   - StrOutputParser不满足 (AIMessage → Str)")
    print("   - 更换JsonOutputParser (AIMessage → Dict(JSON))")
    print()


def demo_json_output_parser_basic() -> None:
    """
    演示 JsonOutputParser 的基本用法。

    展示：
    1. JsonOutputParser 的创建
    2. JsonOutputParser 如何将 AIMessage 转换为字典
    3. JsonOutputParser 与 StrOutputParser 的区别
    """
    print("=" * 80)
    print("【示例三】JsonOutputParser 基本用法")
    print("=" * 80)

    # 创建解析器实例
    str_parser = StrOutputParser()
    json_parser = JsonOutputParser()

    print("\n1. 创建解析器实例：")
    print(f"   str_parser = StrOutputParser()")
    print(f"   json_parser = JsonOutputParser()")
    print(f"   str_parser 的类型：{type(str_parser)}")
    print(f"   json_parser 的类型：{type(json_parser)}")

    # 创建一个包含JSON内容的AIMessage对象（模拟模型的输出）
    # 注意：实际使用中，模型需要返回有效的JSON格式
    ai_message_json = AIMessage(content='{"name": "张雨萱"}')
    ai_message_text = AIMessage(content="张雨萱")

    print("\n2. 模拟模型的输出（AIMessage 对象）：")
    print(f"   ai_message_json = AIMessage(content='{{\"name\": \"张雨萱\"}}')")
    print(f"   ai_message_text = AIMessage(content='张雨萱')")

    # 使用 StrOutputParser 解析
    print("\n3. 使用 StrOutputParser 解析：")
    str_result = str_parser.invoke(ai_message_json)
    print(f"   str_result = str_parser.invoke(ai_message_json)")
    print(f"   str_result 的类型：{type(str_result)}")
    print(f"   str_result 的值：{str_result}")
    print("   问题：结果是字符串，无法直接作为提示词模板的字典输入")

    # 使用 JsonOutputParser 解析
    print("\n4. 使用 JsonOutputParser 解析：")
    try:
        json_result = json_parser.invoke(ai_message_json)
        print(f"   json_result = json_parser.invoke(ai_message_json)")
        print(f"   json_result 的类型：{type(json_result)}")
        print(f"   json_result 的值：{json_result}")
        print("   优势：结果是字典，可以直接作为提示词模板的输入")
    except Exception as e:
        print(f"   解析失败：{type(e).__name__}: {e}")
        print("   注意：JsonOutputParser 要求模型输出必须是有效的JSON格式")

    print("\n结论：")
    print("- StrOutputParser：AIMessage → str（字符串）")
    print("- JsonOutputParser：AIMessage → dict（字典）")
    print("- 当需要将模型输出作为提示词模板的字典输入时，应使用 JsonOutputParser")
    print()


def demo_multi_model_chain_with_json_parser() -> None:
    """
    演示使用 JsonOutputParser 构建多模型链。

    这是课件中展示的完整示例：
    1. 第一个提示词：要求模型返回JSON格式的名字
    2. 第一个模型：生成JSON格式的名字
    3. JsonOutputParser：将AIMessage解析为字典
    4. 第二个提示词：使用字典中的name字段
    5. 第二个模型：解析名字的含义
    6. StrOutputParser：将最终结果解析为字符串
    """
    print("=" * 80)
    print("【示例四】使用 JsonOutputParser 构建多模型链（完整示例）")
    print("=" * 80)

    # 创建解析器实例
    str_parser = StrOutputParser()
    json_parser = JsonOutputParser()

    print("\n1. 创建解析器实例：")
    print(f"   str_parser = StrOutputParser()")
    print(f"   json_parser = JsonOutputParser()")

    # 创建模型实例
    model = init_chat_model()
    print(f"\n2. 创建模型实例：")
    print(f"   model = ChatTongyi(model='qwen3-max')")

    # 创建第一个提示词模板
    # 要求模型返回JSON格式，key是name，value是起的名字
    first_prompt = PromptTemplate.from_template(
        "我邻居姓:{lastname},刚生了{gender},请起名,并封装到JSON格式返回给我,"
        "要求key是name,value就是起的名字。请严格遵守格式要求"
    )

    print("\n3. 创建第一个提示词模板：")
    print("   first_prompt = PromptTemplate.from_template(...)")
    print("   要求：返回JSON格式，key是name，value是起的名字")

    # 创建第二个提示词模板
    # 使用第一个模型输出的name字段
    second_prompt = PromptTemplate.from_template("姓名{name},请帮我解析含义。")

    print("\n4. 创建第二个提示词模板：")
    print("   second_prompt = PromptTemplate.from_template('姓名{name},请帮我解析含义。')")
    print("   使用第一个模型输出的name字段")

    # 构建链：first_prompt | model | json_parser | second_prompt | model | str_parser
    chain = first_prompt | model | json_parser | second_prompt | model | str_parser

    print("\n5. 构建链：")
    print("   chain = first_prompt | model | json_parser | second_prompt | model | str_parser")
    print("\n   链的执行流程：")
    print("   1. first_prompt: 接收变量字典，输出 PromptValue")
    print("   2. 第一个 model: 接收 PromptValue，输出 AIMessage（包含JSON格式的名字）")
    print("   3. json_parser: 接收 AIMessage，输出 dict（字典，包含name字段）")
    print("   4. second_prompt: 接收 dict，使用name字段构建新的 PromptValue")
    print("   5. 第二个 model: 接收 PromptValue，输出 AIMessage（名字含义解析）")
    print("   6. str_parser: 接收 AIMessage，输出 str（最终字符串结果）")

    # 调用链
    print("\n6. 调用链：")
    print("   res = chain.invoke({'lastname': '张', 'gender': '女儿'})")
    print("=" * 80)
    res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
    print(f"\n✅ 成功！最终结果：")
    print(res)
    print(f"\n结果类型：{type(res)}")
    print("=" * 80)
    print()


def demo_chain_step_by_step() -> None:
    """
    演示链式调用的逐步执行过程。

    帮助理解每个步骤的输入输出类型。
    """
    print("=" * 80)
    print("【示例五】链式调用的逐步执行过程")
    print("=" * 80)

    # 创建组件
    first_prompt = PromptTemplate.from_template(
        "我邻居姓:{lastname},刚生了{gender},请起名,并封装到JSON格式返回给我,"
        "要求key是name,value就是起的名字。请严格遵守格式要求"
    )
    second_prompt = PromptTemplate.from_template("姓名{name},请帮我解析含义。")
    model = init_chat_model()
    json_parser = JsonOutputParser()
    str_parser = StrOutputParser()

    input_data = {"lastname": "张", "gender": "女儿"}

    print("\n输入数据：")
    print(f"   input_data = {input_data}")

    # 步骤1：第一个提示词模板
    print("\n步骤1：第一个提示词模板")
    step1_result = first_prompt.invoke(input_data)
    print(f"   输入类型：{type(input_data)}")
    print(f"   输出类型：{type(step1_result)}")
    print(f"   输出内容：{step1_result.to_string()[:100]}...")

    # 步骤2：第一个模型
    print("\n步骤2：第一个模型")
    step2_result = model.invoke(step1_result)
    print(f"   输入类型：{type(step1_result)}")
    print(f"   输出类型：{type(step2_result)}")
    print(f"   输出内容：{step2_result.content}")

    # 步骤3：JsonOutputParser
    print("\n步骤3：JsonOutputParser（关键步骤）")
    step3_result = json_parser.invoke(step2_result)
    print(f"   输入类型：{type(step2_result)}")
    print(f"   输出类型：{type(step3_result)}")
    print(f"   输出内容：{step3_result}")
    print("   注意：这里将 AIMessage 转换为了 dict，可以用于第二个提示词模板")

    # 步骤4：第二个提示词模板
    print("\n步骤4：第二个提示词模板")
    step4_result = second_prompt.invoke(step3_result)
    print(f"   输入类型：{type(step3_result)}（dict）")
    print(f"   输出类型：{type(step4_result)}")
    print(f"   输出内容：{step4_result.to_string()}")

    # 步骤5：第二个模型
    print("\n步骤5：第二个模型")
    step5_result = model.invoke(step4_result)
    print(f"   输入类型：{type(step4_result)}")
    print(f"   输出类型：{type(step5_result)}")
    print(f"   输出内容：{step5_result.content[:100]}...")

    # 步骤6：StrOutputParser
    print("\n步骤6：StrOutputParser")
    step6_result = str_parser.invoke(step5_result)
    print(f"   输入类型：{type(step5_result)}")
    print(f"   输出类型：{type(step6_result)}")
    print(f"   输出内容：{step6_result}")

    print("\n总结：")
    print("JsonOutputParser 在步骤3中起到了关键作用，")
    print("它将 AIMessage 转换为 dict，使得第二个提示词模板可以正确接收输入。")
    print()


def demo_str_vs_json_parser() -> None:
    """
    对比 StrOutputParser 和 JsonOutputParser 的区别。

    展示为什么在多模型链中需要使用 JsonOutputParser。
    """
    print("=" * 80)
    print("【示例六】StrOutputParser vs JsonOutputParser")
    print("=" * 80)

    model = init_chat_model()
    str_parser = StrOutputParser()
    json_parser = JsonOutputParser()

    # 创建一个要求返回JSON的提示词
    json_prompt = PromptTemplate.from_template(
        "请返回JSON格式，包含name字段，值为'测试名字'。格式：{{\"name\": \"测试名字\"}}"
    )

    print("\n1. 使用模型生成JSON格式输出：")
    json_result = (json_prompt | model).invoke({})
    print(f"   模型输出：{json_result.content}")

    # 使用 StrOutputParser
    print("\n2. 使用 StrOutputParser 解析：")
    str_parsed = str_parser.invoke(json_result)
    print(f"   解析结果类型：{type(str_parsed)}")
    print(f"   解析结果：{str_parsed}")
    print("   问题：结果是字符串，无法直接作为提示词模板的字典输入")

    # 使用 JsonOutputParser
    print("\n3. 使用 JsonOutputParser 解析：")
    try:
        json_parsed = json_parser.invoke(json_result)
        print(f"   解析结果类型：{type(json_parsed)}")
        print(f"   解析结果：{json_parsed}")
        print("   优势：结果是字典，可以直接作为提示词模板的输入")

        # 演示如何使用解析后的字典
        print("\n4. 使用解析后的字典作为提示词模板的输入：")
        name_prompt = PromptTemplate.from_template("名字是：{name}")
        final_result = name_prompt.invoke(json_parsed)
        print(f"   提示词模板输出：{final_result.to_string()}")
    except Exception as e:
        print(f"   解析失败：{type(e).__name__}: {e}")
        print("   注意：JsonOutputParser 要求模型输出必须是有效的JSON格式")

    print("\n结论：")
    print("- 当需要将模型输出作为字符串使用时，使用 StrOutputParser")
    print("- 当需要将模型输出作为字典（用于提示词模板）时，使用 JsonOutputParser")
    print()


def main() -> None:
    """
    入口函数：演示 JsonOutputParser 的用法和重要性。

    本示例分为六个部分：
    1. 展示非标准的链式构建方式
    2. 展示类型转换需求
    3. 展示 JsonOutputParser 的基本用法
    4. 展示使用 JsonOutputParser 构建多模型链（完整示例）
    5. 展示链式调用的逐步执行过程
    6. 对比 StrOutputParser 和 JsonOutputParser 的区别
    """
    print("=" * 80)
    print("LangChain JsonOutputParser JSON输出解析器示例")
    print("=" * 80)
    print()

    # 示例一：非标准的链式构建方式
    demo_non_standard_chaining()

    # 示例二：类型转换需求
    demo_type_requirements()

    # 示例三：JsonOutputParser 基本用法
    demo_json_output_parser_basic()

    # 示例四：使用 JsonOutputParser 构建多模型链（完整示例）
    demo_multi_model_chain_with_json_parser()

    # 示例五：链式调用的逐步执行过程
    demo_chain_step_by_step()

    # 示例六：StrOutputParser vs JsonOutputParser
    demo_str_vs_json_parser()

    print("=" * 80)
    print("全部示例执行完毕。")
    print("=" * 80)
    print("\n总结：")
    print("1. 构建多模型链时，应该遵循标准处理逻辑：")
    print("   初始输入 → 提示词模板 → 模型 → 数据处理 → 提示词模板 → 模型 → 解析器 → 结果")
    print("2. 模型的输出是 AIMessage 类型，提示词模板需要 dict 类型")
    print("3. JsonOutputParser 可以将 AIMessage 解析为字典（JSON格式）")
    print("4. 当需要将第一个模型的输出作为第二个提示词模板的输入时，")
    print("   应使用 JsonOutputParser 而不是 StrOutputParser")
    print("5. 正确的链式写法：")
    print("   chain = first_prompt | model | json_parser | second_prompt | model | str_parser")
    print("=" * 80)


if __name__ == "__main__":
    main()
