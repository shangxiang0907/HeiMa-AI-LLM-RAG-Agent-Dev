"""
PromptTemplate / FewShotPromptTemplate 中 format 与 invoke 的对比示例

本示例对应课件图片中的内容，重点演示两件事：

1. PromptTemplate 的两种生成提示词方式：
   - format(...)：纯字符串替换，返回 str
   - invoke({...})：Runnable 标准接口，返回 PromptValue，再通过 .to_string() 得到 str

2. FewShotPromptTemplate 中常用的 invoke(...) 用法：
   - 传入 input 字典，返回 PromptValue，再 .to_string()

并通过一个小表格形式的打印，帮助你理解二者在：
   - 功能
   - 返回值类型
   - 传参方式
   - 支持的占位符
上的差异。
"""

import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain_community.llms.tongyi import Tongyi


def init_llm() -> Tongyi:
    """
    初始化 Tongyi LLM 模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）
    """
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    # LangChain 的 Tongyi 封装会自动从环境变量中读取 key，
    # 这里设置一份到 DASHSCOPE_API_KEY，确保兼容性。
    os.environ["DASHSCOPE_API_KEY"] = api_key

    # 与课件及其他示例保持一致，使用 qwen-max 模型
    llm = Tongyi(model="qwen-max")
    return llm


def build_simple_prompt_template() -> PromptTemplate:
    """
    构建一个最接近课件截图的 PromptTemplate：

    我的邻居姓{lastname}，刚生了{gender}，你帮我起个名字，简单回答。
    """
    template = "我的邻居姓{lastname}，刚生了{gender}，你帮我起个名字，简单回答。"
    return PromptTemplate.from_template(template)


def demo_prompttemplate_format_vs_invoke(llm: Tongyi) -> None:
    """
    使用 PromptTemplate 演示 format 与 invoke 的区别。
    """
    print("=" * 80)
    print("【部分一】PromptTemplate 中 format 与 invoke 的对比")
    print("=" * 80)

    prompt_template = build_simple_prompt_template()

    # --- 写法 1：使用 format（纯字符串替换） ---
    print("\n[1] 使用 format(...) 生成提示词（返回 str）")
    prompt_text_by_format = prompt_template.format(
        lastname="张",
        gender="女儿",
    )
    print("生成的提示词文本：")
    print(prompt_text_by_format)

    res1 = llm.invoke(prompt_text_by_format)
    print("\n模型回复（format 方式）：")
    print(res1)

    # --- 写法 2：使用 invoke（Runnable 标准方法） ---
    print("\n" + "-" * 80)
    print("[2] 使用 invoke({...}) 生成提示词（返回 PromptValue）")

    # PromptTemplate 本身是 Runnable，invoke 返回 PromptValue
    prompt_value = prompt_template.invoke(
        input={"lastname": "李", "gender": "儿子"}
    )
    prompt_text_by_invoke = prompt_value.to_string()

    print("PromptTemplate.invoke(...) 返回的 PromptValue.to_string()：")
    print(prompt_text_by_invoke)

    res2 = llm.invoke(prompt_text_by_invoke)
    print("\n模型回复（invoke 方式）：")
    print(res2)


def build_antonym_fewshot_prompt() -> FewShotPromptTemplate:
    """
    构建一个与课件中类似的反义词 FewShotPromptTemplate。
    """
    example_prompt = PromptTemplate.from_template("单词:{word},反义词:{antonym}")
    examples: List[Dict[str, str]] = [
        {"word": "大", "antonym": "小"},
        {"word": "上", "antonym": "下"},
    ]

    few_shot_template = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        prefix="给出给定词的反义词,有如下示例:",
        suffix="基于示例告诉我:{input_word}的反义词是?",
        input_variables=["input_word"],
    )
    return few_shot_template


def demo_fewshot_invoke(llm: Tongyi) -> None:
    """
    使用 FewShotPromptTemplate 演示 invoke(...) 的用法。
    """
    print("\n" + "=" * 80)
    print("【部分二】FewShotPromptTemplate 中 invoke 的用法")
    print("=" * 80)

    few_shot_template = build_antonym_fewshot_prompt()

    # FewShotPromptTemplate 通常使用 invoke(...)，返回 PromptValue
    prompt_value = few_shot_template.invoke(input={"input_word": "左"})
    prompt_text = prompt_value.to_string()

    print("\nFewShotPromptTemplate.invoke(...).to_string() 生成的提示词：")
    print(prompt_text)

    res = llm.invoke(prompt_text)
    print("\n模型回复：")
    print(res)


def print_format_invoke_diff_table() -> None:
    """
    在终端打印一份对照表，呼应课件中的表格。
    """
    print("\n" + "=" * 80)
    print("【部分三】format 与 invoke 的对比总结")
    print("=" * 80)

    # 简易文本表格，仅用于说明。
    headers = ["维度", "format", "invoke"]
    rows = [
        (
            "功能",
            "纯字符串替换，解析 {} 占位符生成提示词",
            "Runnable 标准方法，解析占位符生成 PromptValue",
        ),
        (
            "返回值",
            "str（普通字符串）",
            "PromptValue（可进一步转换为 str 或消息结构）",
        ),
        (
            "传参方式",
            ".format(k=v, k=v, ...)",
            '.invoke(input={"k": v, "k": v, ...})',
        ),
        (
            "占位符类型",
            "支持 {} 占位符",
            "支持 {} 占位符，也支持 MessagesPlaceholder 等结构化占位符",
        ),
    ]

    col_widths = [8, 32, 40]

    def fmt(text: str, width: int) -> str:
        if len(text) <= width:
            return text + " " * (width - len(text))
        return text[: width - 3] + "..."

    header_line = " | ".join(
        fmt(h, w) for h, w in zip(headers, col_widths)
    )
    print(header_line)
    print("-" * len(header_line))

    for dim, f_val, i_val in rows:
        line = " | ".join(
            [
                fmt(dim, col_widths[0]),
                fmt(f_val, col_widths[1]),
                fmt(i_val, col_widths[2]),
            ]
        )
        print(line)

    print("\n提示：")
    print("1）在只需要一个纯文本提示词时，用 format 更直观；")
    print("2）在要与 Runnable 生态（链、流水线等）配合时，推荐使用 invoke。")


def main() -> None:
    """
    入口函数：综合演示 format 与 invoke 的差异。
    """
    print("=" * 80)
    print("LangChain PromptTemplate / FewShotPromptTemplate：format vs invoke 示例")
    print("=" * 80)

    llm = init_llm()

    # 部分一：PromptTemplate 中的 format 与 invoke
    demo_prompttemplate_format_vs_invoke(llm)

    # 部分二：FewShotPromptTemplate 中常见的 invoke 用法
    demo_fewshot_invoke(llm)

    # 部分三：总结对比表
    print_format_invoke_diff_table()

    print("\n" + "=" * 80)
    print("演示结束")
    print("=" * 80)


if __name__ == "__main__":
    main()

