"""
知识库服务类
用于管理文件上传、MD5校验和向量数据库存储
"""
import hashlib
import os
from config_data import md5_path


class KnowledgeBaseService(object):
    def __init__(self):
        # 向量存储的实例 Chroma向量库对象
        self.chroma = None
        # 文本分割器的对象
        self.spliter = None

    def check_md5(self, md5_str):
        """检查传入的md5字符串是否已经被处理过了"""
        if not os.path.exists(md5_path):
            return False
        
        with open(md5_path, 'r', encoding='utf-8') as f:
            existing_md5s = f.read().splitlines()
        
        return md5_str in existing_md5s

    def save_md5(self, md5_str):
        """将传入的md5字符串,记录到文件内保存"""
        # 确保目录存在
        os.makedirs(os.path.dirname(md5_path) if os.path.dirname(md5_path) else '.', exist_ok=True)
        
        with open(md5_path, 'a', encoding='utf-8') as f:
            f.write(md5_str + '\n')

    def get_string_md5(self, str_data):
        """将传入的字符串转换为md5字符串"""
        md5_hash = hashlib.md5()
        md5_hash.update(str_data.encode('utf-8'))
        return md5_hash.hexdigest()

    def upload_by_str(self, data, filename):
        """将传入的字符串,进行向量化,存入向量数据库中"""
        # TODO: 实现向量化和存储逻辑
        # 1. 使用 self.spliter 分割文本
        # 2. 使用 self.chroma 存储向量数据
        pass
