"""
通用提示词模板（PromptTemplate）用法示例

本示例对应课件中的两段代码：

1）标准写法（先生成提示词文本，再喂给模型）：

from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，帮忙起名字，请简略回答。"
)

prompt_text = prompt_template.format(lastname="张", gender="女儿")
model = Tongyi(model="qwen-max")
res = model.invoke(input=prompt_text)

2）基于 chain 的写法（把 PromptTemplate 和 模型 链起来）：

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，帮忙起名字，请简略回答。"
)

model = Tongyi(model="qwen-max")
chain = prompt_template | model
res = chain.invoke(input={"lastname": "曹", "gender": "女儿"})

在此基础上，我们封装成可直接运行的脚本，并增加了说明性输出，便于学习。
"""

import os
from typing import Dict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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


def build_name_prompt_template() -> PromptTemplate:
    """
    构建一个通用的起名提示词模板。

    模板中预留了两个变量：
    - {lastname}：姓氏
    - {gender}：新生儿性别（如“女儿”“儿子”等）
    """
    template = (
        "你是一位经验丰富的中文起名顾问。\n"
        "我的邻居姓{lastname}，刚生了{gender}。\n"
        "请根据中文命名习惯，给出 2~3 个合适的名字备选，"
        "每个名字附上 1 句话的含义说明，答案尽量简短。"
    )
    return PromptTemplate.from_template(template)


def demo_standard_usage(llm: Tongyi, prompt_template: PromptTemplate) -> None:
    """
    演示「标准写法」：
    1. 先使用 PromptTemplate.format(...) 把变量注入，得到最终提示词文本
    2. 再把提示词文本传给 llm.invoke(...)
    """
    print("=" * 80)
    print("【示例1】标准写法：先生成提示词文本，再调用模型")
    print("-" * 80)

    # Step 1：通过 format 注入变量，生成最终提示词文本
    prompt_text = prompt_template.format(lastname="张", gender="女儿")

    print("生成的提示词（发送给大模型的最终文本）：\n")
    print(prompt_text)
    print("\n模型回复：\n")

    # Step 2：把最终提示词文本喂给模型
    res = llm.invoke(prompt_text)
    print(res)
    print()


def demo_chain_usage(llm: Tongyi, prompt_template: PromptTemplate) -> None:
    """
    演示「基于 chain 的写法」：
    1. 使用 `prompt_template | llm` 把提示词模板与模型连接成一个链
    2. 调用 chain.invoke(...) 时，直接传入变量字典，由链自动完成：
       - 把变量注入到模板中
       - 调用模型并返回结果
    """
    print("=" * 80)
    print("【示例2】基于 chain 的写法：PromptTemplate | LLM")
    print("-" * 80)

    # Step 1：把 PromptTemplate 和 LLM 通过管道符 `|` 链接起来
    chain = prompt_template | llm

    # Step 2：构造变量输入（不再手动拼接提示词）
    variables: Dict[str, str] = {"lastname": "曹", "gender": "女儿"}

    print("传给 chain 的变量字典：", variables)
    print("\n模型回复：\n")

    # 注意：这里传入的仍然是 input=...，但内容是变量字典
    res = chain.invoke(input=variables)
    print(res)
    print()


def main() -> None:
    """
    主函数：演示 LangChain 中 PromptTemplate 的两种常用用法。
    """
    print("=" * 80)
    print("LangChain 通用提示词模板（PromptTemplate）用法示例")
    print("=" * 80)

    llm = init_llm()
    prompt_template = build_name_prompt_template()

    # 示例 1：标准写法
    demo_standard_usage(llm, prompt_template)

    # 示例 2：基于 chain 的写法
    demo_chain_usage(llm, prompt_template)

    print("=" * 80)
    print("演示结束")
    print("=" * 80)


if __name__ == "__main__":
    main()

