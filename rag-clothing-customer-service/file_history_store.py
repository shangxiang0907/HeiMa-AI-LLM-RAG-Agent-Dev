"""
基于文件存储的会话历史记录模块

本模块提供基于文件存储的持久化会话历史记录功能，用于实现长期会话记忆。
核心组件：
- FileChatMessageHistory：基于文件存储的会话历史记录类
- get_history：获取指定会话ID的历史会话记录函数
"""

import json
import os
from typing import Dict, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

import config_data as config


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    基于文件存储的会话历史记录类。

    核心思路：
    - 基于文件存储会话记录，以 session_id 为文件名
    - 不同 session_id 有不同文件存储消息
    - 继承 BaseChatMessageHistory 实现三个方法：
      1. add_messages: 同步模式，添加消息
      2. messages: 同步模式，获取消息
      3. clear: 同步模式，清除消息

    属性:
        storage_path: 存储路径，会话文件将保存在此目录下
        session_id: 会话ID，用作文件名
    """

    def __init__(self, storage_path: str, session_id: str):
        """
        初始化 FileChatMessageHistory 实例。

        参数:
            storage_path: 存储路径，会话文件将保存在此目录下
            session_id: 会话ID，用作文件名
        """
        self.storage_path = storage_path
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        """
        从文件中读取消息列表。

        返回:
            消息列表，如果文件不存在则返回空列表
        """
        file_path = os.path.join(self.storage_path, self.session_id)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        添加消息到文件。

        参数:
            messages: 要添加的消息序列
        """
        # 获取现有消息
        all_messages = list(self.messages)
        # 添加新消息
        all_messages.extend(messages)
        # 序列化消息
        serialized = [message_to_dict(message) for message in all_messages]
        # 确保目录存在
        os.makedirs(self.storage_path, exist_ok=True)
        # 构建文件路径
        file_path = os.path.join(self.storage_path, self.session_id)
        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """
        清除文件中的消息。
        """
        # 确保目录存在
        os.makedirs(self.storage_path, exist_ok=True)
        # 构建文件路径
        file_path = os.path.join(self.storage_path, self.session_id)
        # 写入空列表
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


# 历史记录存储字典，存放多个会话ID所对应的历史会话记录
chat_history_store: Dict[str, FileChatMessageHistory] = {}


def get_history(session_id: str, storage_path: str = None) -> FileChatMessageHistory:
    """
    获取指定会话ID的历史会话记录函数。

    参数:
        session_id: 会话ID（字符串类型）
        storage_path: 存储路径，默认为 config.chat_history_path

    返回:
        FileChatMessageHistory 实例，用于存储该会话的历史记录
    """
    if storage_path is None:
        storage_path = config.chat_history_path
    if session_id not in chat_history_store:
        chat_history_store[session_id] = FileChatMessageHistory(
            storage_path, session_id
        )
    return chat_history_store[session_id]
