from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from zhisaotong_agent.utils.config_handler import RagConfig, load_rag_config
from zhisaotong_agent.utils.api_key import init_dashscope_api_key


ModelT = TypeVar("ModelT")


class BaseModelFactory(ABC, Generic[ModelT]):
    """模型工厂抽象基类。

    目前只做一个非常轻量的封装，保持接口简单：
    - 不做缓存控制
    - 不做重试/降级
    方便后续在不改调用方的情况下渐进增强。
    """

    @abstractmethod
    def generator(self) -> Optional[ModelT]:
        """生成一个模型实例。当前实现始终返回非 None。"""
        raise NotImplementedError


class ChatModelFactory(BaseModelFactory[BaseChatModel]):
    """聊天模型工厂。"""

    def generator(self) -> Optional[BaseChatModel]:
        # 这里直接根据配置名称实例化模型，后续如果需要支持多模型切换，
        # 可以在不改调用方的前提下在工厂内部增加分支和缓存。
        return ChatTongyi(model=_rag_conf.model.chat_model_name)


class EmbeddingsFactory(BaseModelFactory[Embeddings]):
    """Embedding 模型工厂。"""

    def generator(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model=_rag_conf.model.embedding_model_name)


# === 懒加载配置与模型实例 ===

_rag_conf: RagConfig = load_rag_config()
_chat_model: Optional[BaseChatModel] = None
_embed_model: Optional[Embeddings] = None


def get_chat_model() -> BaseChatModel:
    """懒加载并返回全局聊天模型实例。"""
    global _chat_model
    if _chat_model is None:
        model = ChatModelFactory().generator()
        if model is None:
            raise RuntimeError("Chat model 初始化失败，请检查 rag 配置与环境变量。")
        _chat_model = model
    return _chat_model


def get_embed_model() -> Embeddings:
    """懒加载并返回全局 Embedding 模型实例。"""
    global _embed_model
    if _embed_model is None:
        model = EmbeddingsFactory().generator()
        if model is None:
            raise RuntimeError("Embedding model 初始化失败，请检查 rag 配置与环境变量。")
        _embed_model = model
    return _embed_model


if __name__ == "__main__":
    """
    简单自测：
    - 在项目根目录运行：
        python -m zhisaotong_agent.model.factory
    - 用于快速验证 rag 配置与 DashScope 相关环境变量是否正确，
      以及聊天模型 / Embedding 模型是否能成功初始化。
    """
    ok = init_dashscope_api_key()
    if not ok:
        raise SystemExit("DASHSCOPE_API_KEY 未正确配置，无法初始化模型。")

    chat_model = get_chat_model()
    embed_model = get_embed_model()

    print("Chat model instance:", chat_model)
    print("Embedding model instance:", embed_model)

