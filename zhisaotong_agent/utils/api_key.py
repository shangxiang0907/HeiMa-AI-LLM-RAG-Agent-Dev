from __future__ import annotations

"""
DashScope API Key 初始化工具。

用法示例（在应用入口或测试脚本中）：

    from zhisaotong_agent.utils.api_key import init_dashscope_api_key

    if not init_dashscope_api_key():
        # 可以选择直接退出，避免后续调用模型时报错
        raise SystemExit("DASHSCOPE_API_KEY 未正确配置")
"""

import os
from typing import Final

from dotenv import load_dotenv

from zhisaotong_agent.utils.logger_handler import get_logger


logger = get_logger(__name__)
_ENV_KEY_PRIMARY: Final[str] = "DASHSCOPE_API_KEY"
_ENV_KEY_FALLBACK: Final[str] = "API_KEY"


def init_dashscope_api_key() -> bool:
    """初始化并校验 DashScope API Key。

    行为与 `rag-clothing-customer-service/app_qa.py` 中的 init_api_key 基本一致：
    - 优先从 `.env` 文件加载环境变量；
    - 尝试读取 `DASHSCOPE_API_KEY`，如果没有，则退回读取 `API_KEY`；
    - 如果最终仍未获取到 key，则返回 False 并记录错误日志；
    - 如果获取成功，则把值写回 `os.environ["DASHSCOPE_API_KEY"]`，返回 True。
    """
    # 加载 .env 文件中的环境变量（如果存在）
    load_dotenv()

    api_key = os.getenv(_ENV_KEY_PRIMARY) or os.getenv(_ENV_KEY_FALLBACK)
    if not api_key:
        logger.error(
            f"未找到 {_ENV_KEY_PRIMARY} 或 {_ENV_KEY_FALLBACK} 环境变量，请在 .env 或系统环境中配置后再运行。"
        )
        return False

    # 统一写入 DASHSCOPE_API_KEY，供 DashScope / ChatTongyi / DashScopeEmbeddings 使用
    os.environ[_ENV_KEY_PRIMARY] = api_key
    logger.info("DashScope API Key 已成功加载并设置到环境变量。")
    return True

