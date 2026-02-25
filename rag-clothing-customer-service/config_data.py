md5_path = "./md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# 会话历史存储路径
chat_history_path = "./chat_history"

# splitter
chunk_size = 1000
chunk_overlap = 100
separators = [
    "\n\n",    # 段落分隔
    "\n",      # 行分隔
    "。",      # 中文句号
    "！",      # 中文感叹号
    "？",      # 中文问号
    ". ",      # 英文句号+空格
    "! ",      # 英文感叹号+空格
    "? ",      # 英文问号+空格
    ".",       # 英文句号
    "!",       # 英文感叹号
    "?",       # 英文问号
    " ",       # 空格
    ""         # 字符
]

# 检索返回匹配的文档数量
similarity_threshold = 2

# 模型配置
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"
