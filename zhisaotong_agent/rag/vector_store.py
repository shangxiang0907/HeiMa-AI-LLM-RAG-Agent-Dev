from __future__ import annotations

import os
from typing import Iterable, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from zhisaotong_agent.model.factory import get_embed_model
from zhisaotong_agent.utils.config_handler import ChromaConfig, load_chroma_config
from zhisaotong_agent.utils.api_key import init_dashscope_api_key
from zhisaotong_agent.utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from zhisaotong_agent.utils.logger_handler import get_logger
from zhisaotong_agent.utils.path_tool import get_abs_path


_chroma_conf: ChromaConfig = load_chroma_config()
logger = get_logger(__name__)


class VectorStoreService:
    """向量库服务封装。

    对外暴露：
    - `get_retriever()`：获取检索器
    - `load_document()`：扫描数据目录并将新文档导入向量库（按文件 MD5 去重）
    """

    def __init__(self) -> None:
        # 初始化向量库
        self.vector_store = Chroma(
            collection_name=_chroma_conf.collection.name,
            embedding_function=get_embed_model(),
            persist_directory=get_abs_path(_chroma_conf.client.persist_directory),
        )

        # 文本切分器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=_chroma_conf.processing.chunk_size,
            chunk_overlap=_chroma_conf.processing.chunk_overlap,
            separators=_chroma_conf.processing.separators,
            length_function=len,
        )

        # 缓存已处理文件的 MD5，避免每个文件都从磁盘重复读取
        self._processed_md5_hex = self._load_processed_md5_hex()

    # === 对外接口 ===

    def get_retriever(self):
        """获取检索器，保持对外行为与原设计一致。"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": _chroma_conf.retrieval.k}
        )

    def load_document(self) -> None:
        """从数据目录加载文档并写入向量库（按文件 MD5 去重）。"""

        def get_file_documents(read_path: str) -> List[Document]:
            """根据文件类型选择合适的 loader。"""
            lower_path = read_path.lower()
            if lower_path.endswith("txt"):
                return txt_loader(read_path)
            if lower_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: List[str] = listdir_with_allowed_type(
            get_abs_path(_chroma_conf.storage.data_path),
            tuple(_chroma_conf.knowledge.allow_knowledge_file_type),
        )

        for path in allowed_files_path:
            # 计算文件 MD5，用于去重
            md5_hex = get_file_md5_hex(path)

            if self._check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path} 内容已经存在知识库内，跳过")
                continue

            try:
                documents: List[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path} 内没有有效文本内容，跳过")
                    continue

                split_documents: List[Document] = self.splitter.split_documents(documents)

                if not split_documents:
                    logger.warning(f"[加载知识库]{path} 分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_documents)

                # 记录这个已经处理好的文件的 md5，避免下次重复加载
                self._save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功，分片数：{len(split_documents)}")
            except Exception as e:  # noqa: BLE001
                # exc_info 为 True 会记录详细的报错堆栈，如果为 False 仅记录报错信息本身
                logger.error(f"[加载知识库]{path} 加载失败：{str(e)}", exc_info=True)
                continue

    # === 内部工具方法 ===

    @staticmethod
    def _md5_store_path() -> str:
        return get_abs_path(_chroma_conf.storage.md5_hex_store)

    def _load_processed_md5_hex(self) -> set[str]:
        """一次性从磁盘加载已处理文件的 MD5，避免逐行重复扫描。"""
        md5_path = self._md5_store_path()
        if not os.path.exists(md5_path):
            # 如果文件不存在，提前创建空文件，方便后续追加
            open(md5_path, "w", encoding="utf-8").close()
            return set()

        processed: set[str] = set()
        with open(md5_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(line)
        return processed

    def _check_md5_hex(self, md5_for_check: str) -> bool:
        """检查 MD5 是否已经处理过（仅查内存缓存）。"""
        return md5_for_check in self._processed_md5_hex

    def _save_md5_hex(self, md5_for_save: str) -> None:
        """将新的 MD5 写入缓存和磁盘文件。"""
        if md5_for_save in self._processed_md5_hex:
            return

        self._processed_md5_hex.add(md5_for_save)
        with open(self._md5_store_path(), "a", encoding="utf-8") as f:
            f.write(md5_for_save + "\n")


if __name__ == "__main__":
    # 本模块主要通过 python -m 方式在项目根目录下运行，示例：
    #   cd /home/devbox/project
    #   python -m zhisaotong_agent.rag.vector_store
    #
    # 在运行前会尝试通过 .env / 环境变量初始化 DASHSCOPE_API_KEY，
    # 逻辑与 rag-clothing-customer-service/app_qa.py 中的 init_api_key 基本一致。
    #
    # 也可以在安装为包（pip install -e .）后，在任意目录运行同样的命令。
    # 下面代码仅用于本地简单验证向量库加载与检索是否正常。
    ok = init_dashscope_api_key()
    if not ok:
        raise SystemExit("DASHSCOPE_API_KEY 未正确配置，无法运行向量库测试。")

    vs = VectorStoreService()
    vs.load_document()

    retriever = vs.get_retriever()
    res: Iterable[Document] = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)

