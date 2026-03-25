"""
文件加载与遍历工具模块。

功能：
- 计算文件 MD5（十六进制字符串）；
- 按后缀过滤列出目录下的文件；
- 使用 LangChain 加载 PDF / TXT 文本为 Document 列表。

安全优先：
- 所有对外函数都做参数与路径的基本校验；
- 捕获常见异常并使用统一日志工具记录，而不是直接抛出到最上层；
- 对外接口在失败时返回“空结果”，由上层根据日志排查原因。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from zhisaotong_agent.utils.logger_handler import get_logger
from zhisaotong_agent.utils.path_tool import get_abs_path


logger = get_logger(__name__)


def _safe_to_abs_path(filepath: str) -> str:
    """
    将“项目内相对路径 or 已是绝对路径”的文件/目录转为绝对路径。

    说明：
    - 业务侧使用时有可能直接传入绝对路径，这里做宽松处理：
      - 绝对路径：直接返回；
      - 相对路径：通过 get_abs_path 转为基于项目根目录的绝对路径；
    """
    p = Path(filepath)
    if p.is_absolute():
        return str(p)
    try:
        return get_abs_path(str(p))
    except Exception as e:  # 极少走到这里，兜底记录日志
        logger.error("转换路径为绝对路径失败: filepath=%r, error=%s", filepath, e)
        raise


def get_file_md5_hex(filepath: str) -> str:
    """
    获取文件内容的 MD5 十六进制字符串。

    - filepath: 文件路径，支持项目内相对路径或绝对路径；
    - 如文件不存在或无法读取，记录错误日志并返回空字符串 ""。
    """
    try:
        abs_path = _safe_to_abs_path(filepath)
    except Exception:
        # 上层已经记录过日志
        return ""

    if not os.path.isfile(abs_path):
        logger.warning("计算 MD5 失败：目标不是文件或不存在: %r", abs_path)
        return ""

    md5 = hashlib.md5()
    try:
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        value = md5.hexdigest()
        logger.debug("文件 MD5 计算完成: path=%r, md5=%s", abs_path, value)
        return value
    except Exception as e:
        logger.error("计算文件 MD5 失败: path=%r, error=%s", abs_path, e)
        return ""


def listdir_with_allowed_type(
    path: str, allowed_types: Tuple[str, ...]
) -> Tuple[str, ...]:
    """
    列出给定目录下，后缀在 allowed_types 中的文件（非递归）。

    参数：
    - path: 目录路径，支持相对 / 绝对；
    - allowed_types: 允许的文件后缀元组，例如：(".txt", ".pdf")。

    返回：
    - 返回一个包含“绝对路径字符串”的 tuple；
    - 如果目录不存在、非目录或出错，返回空 tuple。
    """
    try:
        abs_dir = _safe_to_abs_path(path)
    except Exception:
        return tuple()

    if not os.path.isdir(abs_dir):
        logger.warning("列目录失败：目标不是目录或不存在: %r", abs_dir)
        return tuple()

    # 统一将后缀转为小写，便于比较
    normalized_suffixes = tuple(s.lower() for s in allowed_types)
    results: list[str] = []

    try:
        for name in os.listdir(abs_dir):
            full_path = os.path.join(abs_dir, name)
            if not os.path.isfile(full_path):
                continue
            suffix = Path(name).suffix.lower()
            if suffix in normalized_suffixes:
                results.append(str(Path(full_path).resolve()))
        logger.info(
            "目录扫描完成: dir=%r, allowed=%s, matched_count=%d",
            abs_dir,
            normalized_suffixes,
            len(results),
        )
        return tuple(results)
    except Exception as e:
        logger.error("遍历目录失败: dir=%r, error=%s", abs_dir, e)
        return tuple()


def pdf_loader(filepath: str, passwd: str | None = None) -> List[Document]:
    """
    使用 LangChain PyPDFLoader 加载 PDF 文件为 Document 列表。

    - filepath: PDF 文件路径，相对（项目内）或绝对；
    - passwd: PDF 打开密码，如不需要则传 None。

    异常处理：
    - 遇到 IO / 解密 / 解析错误时，记录错误日志并返回空列表。
    """
    try:
        abs_path = _safe_to_abs_path(filepath)
    except Exception:
        return []

    if not os.path.isfile(abs_path):
        logger.warning("PDF 加载失败：文件不存在: %r", abs_path)
        return []

    try:
        # PyPDFLoader 在部分版本中支持 pdf_password；如不支持，该参数会被忽略。
        loader = PyPDFLoader(abs_path, password=passwd)  # type: ignore[call-arg]
    except TypeError:
        # 向后兼容：某些版本没有 password 参数
        logger.info("当前 PyPDFLoader 版本不支持 password 参数，尝试不带密码初始化: %r", abs_path)
        loader = PyPDFLoader(abs_path)  # type: ignore[call-arg]
    except Exception as e:
        logger.error("初始化 PyPDFLoader 失败: path=%r, error=%s", abs_path, e)
        return []

    try:
        docs = loader.load()
        logger.info("PDF 加载成功: path=%r, pages=%d", abs_path, len(docs))
        return docs
    except Exception as e:
        logger.error("读取 PDF 失败: path=%r, error=%s", abs_path, e)
        return []


def txt_loader(filepath: str) -> List[Document]:
    """
    使用 LangChain TextLoader 加载 TXT 文本为 Document 列表。

    - filepath: 文本文件路径，相对（项目内）或绝对；
    - 默认假定编码为 utf-8，如不兼容则退化为 errors='ignore'。
    """
    try:
        abs_path = _safe_to_abs_path(filepath)
    except Exception:
        return []

    if not os.path.isfile(abs_path):
        logger.warning("TXT 加载失败：文件不存在: %r", abs_path)
        return []

    # 第一次尝试严格 utf-8，失败则降级忽略错误
    try:
        loader = TextLoader(abs_path, encoding="utf-8")
        docs = loader.load()
        logger.info("TXT 加载成功(utf-8): path=%r, docs=%d", abs_path, len(docs))
        return docs
    except UnicodeDecodeError:
        logger.warning(
            "TXT utf-8 解码失败，降级为 errors='ignore' 尝试加载: %r", abs_path
        )
        try:
            loader = TextLoader(abs_path, encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
            logger.info(
                "TXT 加载成功(autodetect_encoding): path=%r, docs=%d",
                abs_path,
                len(docs),
            )
            return docs
        except Exception as e:
            logger.error("TXT 加载失败(autodetect_encoding): path=%r, error=%s", abs_path, e)
            return []
    except Exception as e:
        logger.error("TXT 加载失败: path=%r, error=%s", abs_path, e)
        return []


__all__ = [
    "get_file_md5_hex",
    "listdir_with_allowed_type",
    "pdf_loader",
    "txt_loader",
]


if __name__ == "__main__":
    """
    简单自测代码：
    - 使用 data 目录中的示例文件测试目录扫描、MD5 与加载函数；
    - 运行方式（在项目根目录）：
        python -m zhisaotong_agent.utils.file_handler
    """

    # 假定 data 目录位于项目根目录下
    try:
        data_dir = get_abs_path("data")
    except Exception as e:
        logger.error("定位 data 目录失败: %s", e)
        raise SystemExit(1)

    logger.info("开始自测 file_handler 模块，data_dir=%r", data_dir)

    # 1. 扫描 TXT / PDF 文件
    txt_pdf_files = listdir_with_allowed_type(data_dir, (".txt", ".pdf"))
    logger.info("在 data 目录下发现 txt/pdf 文件数量: %d", len(txt_pdf_files))

    for f in txt_pdf_files:
        logger.info("发现文件: %s", f)
        md5_value = get_file_md5_hex(f)
        logger.info("文件 MD5: %s -> %s", f, md5_value or "<empty>")

        suffix = Path(f).suffix.lower()
        if suffix == ".pdf":
            docs = pdf_loader(f)
        else:
            docs = txt_loader(f)

        if docs:
            logger.info("加载成功: %s, 文档数=%d, 示例内容片段=%r", f, len(docs), docs[0].page_content[:80])
        else:
            logger.warning("加载失败或结果为空: %s", f)

