import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

client = OpenAI(
    # 从环境变量中读取 API Key
    api_key=os.getenv("API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=True
)
for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)
