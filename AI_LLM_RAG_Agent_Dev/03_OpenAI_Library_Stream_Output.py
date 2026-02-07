import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    """
    使用 OpenAI 官方库的流式输出示例：
    1. 从 .env 读取 API_KEY
    2. 初始化客户端（兼容阿里云 DashScope 的 OpenAI 模式）
    3. 使用 stream=True 进行流式对话，并边接收边打印
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

    # 3. 发送一次基础对话请求（流式）
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "详细介绍一下你自己"},
        ],
        stream=True,
    )

    print("模型流式回复：", flush=True)

    # 4. 流式打印模型回复内容
    full_reply = []
    for chunk in completion:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            full_reply.append(content)
            print(content, end="     ", flush=True)

    # 换行并打印完整结果（可选）
    print("\n\n----- 完整回复（汇总） -----")
    print("".join(full_reply))


if __name__ == "__main__":
    main()