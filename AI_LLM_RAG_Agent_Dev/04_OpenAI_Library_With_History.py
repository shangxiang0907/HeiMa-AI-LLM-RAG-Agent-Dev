import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    """
    使用 OpenAI 官方库附带历史消息调用模型示例：
    1. 从 .env 读取 API_KEY
    2. 初始化客户端（兼容阿里云 DashScope 的 OpenAI 模式）
    3. 构建包含多轮对话历史的 messages 列表
    4. 发送请求并获取模型回复
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

    # 3. 构建包含历史消息的对话列表
    messages = [
        {"role": "system", "content": "你是AI助理,回答很简洁"},
        {"role": "user", "content": "小明有2条宠物狗"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "小红有3只宠物猫"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "总共有几个宠物?"},
    ]

    # 4. 发送包含历史消息的对话请求
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=messages,
        stream=False,
    )

    # 5. 打印模型回复
    reply = completion.choices[0].message.content
    print("模型回复：")
    print(reply)

    # 6. 可选：展示如何将新回复添加到历史消息中
    print("\n----- 历史消息（包含新回复） -----")
    messages.append({"role": "assistant", "content": reply})
    for i, msg in enumerate(messages, 1):
        print(f"{i}. {msg['role']}: {msg['content']}")


if __name__ == "__main__":
    main()
