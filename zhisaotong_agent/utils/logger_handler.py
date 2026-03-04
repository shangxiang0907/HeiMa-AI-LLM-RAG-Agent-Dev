"""
项目统一日志工具。

设计目标：
- 所有模块通过 `get_logger(__name__)` 获取 logger，保证格式与输出行为一致；
- 默认输出到控制台 + 按天轮转的日志文件（保留最近若干天）；
- 日志文件统一存放在项目根目录下的 `logs/` 目录。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from .path_tool import get_abs_path

# 日志根目录：项目根目录下的 logs 目录
LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

# 默认日志格式：时间 - logger 名称 - 级别 - 文件名:行号 - 消息
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def _build_log_file_path(name: str) -> str:
    """
    根据 logger 名称和当前日期生成日志文件路径。

    例如：logs/agent_20260304.log
    """
    safe_name = name.replace(":", "_").replace("/", "_")
    filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d')}.log"
    return os.path.join(LOG_ROOT, filename)


def get_logger(
    name: str = __name__,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: Optional[str] = None,
    when: str = "D",
    interval: int = 1,       # 多久轮转一次日志文件（配合 when 使用），默认 1 天
    backup_count: int = 7,   # 最多保留多少个历史日志文件，超过会自动删除旧文件
) -> logging.Logger:
    """
    获取一个已经按项目标准配置好的 logger。

    参数说明：
    - name: logger 名称，推荐在各模块里使用 `__name__` 传入；
    - console_level: 控制台输出日志级别（默认 INFO）；
    - file_level: 文件日志级别（默认 DEBUG）；
    - log_file: 指定日志文件的绝对路径；如果为 None，则自动根据 name+日期生成；
    - when / interval / backup_count: TimedRotatingFileHandler 的轮转策略：
        - when: "S", "M", "H", "D", "W0"-"W6", "midnight" 等；
        - interval: 间隔单位数量；
        - backup_count: 保留的历史文件数量。

    说明：
    - 为避免重复添加 handler，如果 logger 已经配置过 handler，则直接返回。
    """
    logger = logging.getLogger(name)
    # 顶层 logger 级别设置为 DEBUG，具体输出由各 handler 控制
    logger.setLevel(logging.DEBUG)

    # 如果已经配置过 handler，直接返回，避免重复添加
    if logger.handlers:
        return logger

    # 1. 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMATTER)
    logger.addHandler(console_handler)

    # 2. 文件 Handler（按时间轮转）
    if log_file is None:
        log_file = _build_log_file_path(name)

    # 如果传入了相对路径，自动转为基于项目根目录的绝对路径
    if not os.path.isabs(log_file):
        log_file = get_abs_path(log_file)

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMATTER)
    logger.addHandler(file_handler)

    return logger


__all__ = ["get_logger", "LOG_ROOT"]

