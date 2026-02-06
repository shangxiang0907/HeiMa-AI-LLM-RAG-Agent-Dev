import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    """
    使用 OpenAI 官方库的基础示例：
    1. 从 .env 读取 API_KEY
    2. 初始化客户端（兼容阿里云 DashScope 的 OpenAI 模式）
    3. 发送一条简单对话消息并打印完整回复
    """
    # 1. 加载环境变量
    load_dotenv()

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("未在环境变量或 .env 中找到 API_KEY，请先配置后再运行。")

    # 2. 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 3. 发送一次基础对话请求（非流式）
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "用一句话介绍一下你自己。"},
        ],
        stream=False,
    )

    # 4. 打印模型回复
    reply = completion.choices[0].message.content
    print("模型回复：")
    print(reply)


if __name__ == "__main__":
    main()

