"""
知识库服务类
用于管理文件上传、MD5校验和向量数据库存储
"""
import hashlib
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config


class KnowledgeBaseService(object):
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        # 兼容两种环境变量命名方式
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
            )
        
        # LangChain 的 DashScopeEmbeddings 会自动从环境变量中读取 key
        os.environ["DASHSCOPE_API_KEY"] = api_key
        
        # 如果文件夹不存在则创建,如果存在则跳过
        os.makedirs(config.persist_directory, exist_ok=True)
        
        # 向量存储的实例 Chroma向量库对象
        self.chroma = Chroma(
            collection_name=config.collection_name,  # 数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,  # 数据库本地存储文件夹
        )
        
        # 文本分割器的对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  # 分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap,  # 连续文本段之间的字符重叠数量
            separators=config.separators,  # 自然段落划分的符号
            length_function=len,  # 使用Python自带的len函数做长度统计的依据
        )

    def check_md5(self, md5_str):
        """检查传入的md5字符串是否已经被处理过了"""
        if not os.path.exists(config.md5_path):
            return False
        
        with open(config.md5_path, 'r', encoding='utf-8') as f:
            existing_md5s = f.read().splitlines()
        
        return md5_str in existing_md5s

    def save_md5(self, md5_str):
        """将传入的md5字符串,记录到文件内保存"""
        # 确保目录存在
        os.makedirs(os.path.dirname(config.md5_path) if os.path.dirname(config.md5_path) else '.', exist_ok=True)
        
        with open(config.md5_path, 'a', encoding='utf-8') as f:
            f.write(md5_str + '\n')

    def get_string_md5(self, str_data):
        """将传入的字符串转换为md5字符串"""
        md5_hash = hashlib.md5()
        md5_hash.update(str_data.encode('utf-8'))
        return md5_hash.hexdigest()

    def upload_by_str(self, data: str, filename):
        """将传入的字符串,进行向量化,存入向量数据库中"""
        # 先得到传入字符串的md5值
        md5_hex = self.get_string_md5(data)
        if self.check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"
        
        # 使用文本分割器分割文本（分割器会根据 chunk_size 自动决定是否分割）
        knowledge_chunks: list[str] = self.spliter.split_text(data)
        
        # 构建元数据
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "小曹",
        }
        
        # 内容就加载到向量库中了
        # iterable -> list \ tuple
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )
        
        # 保存MD5值
        self.save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"
