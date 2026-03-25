"""
配置文件读写与合并工具（分组结构 + 文件内环境区分）。

设计要点：
- 所有配置文件统一放在项目根目录下的 `config/` 目录；
- 每个配置文件内部使用“分组结构 + envs 环境区分”，例如（rag.yml）：

    envs:
      default:
        model:
          chat_model_name: qwen3-max
          embedding_model_name: text-embedding-v4
        retrieval:
          top_k: 5
          score_threshold: 0.3
      dev:
        retrieval:
          top_k: 3
      prod:
        retrieval:
          top_k: 8

- 环境选择规则：
    1. 优先使用函数参数 env；
    2. 否则读取环境变量 APP_ENV；
    3. 否则回落到 "default" 环境。

对外主要接口（示例）：

    from utils.config_handler import (
        load_rag_config,
        load_chroma_config,
        load_prompts_config,
        load_agent_config,
    )

    rag_conf = load_rag_config()              # 使用 APP_ENV 或 default
    rag_conf_dev = load_rag_config(env="dev") # 强制使用 dev 配置
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from zhisaotong_agent.utils.path_tool import get_abs_path


ENV_VAR_NAME = "APP_ENV"
DEFAULT_ENV_NAME = "default"


# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------


def _load_yaml_dict(relative_path: str, *, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    从 `config/` 目录加载 YAML 文件，并确保返回字典。

    :param relative_path: 相对于项目根目录的相对路径，例如 "config/rag.yml"
    """
    abs_path = Path(get_abs_path(relative_path))
    if not abs_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {abs_path}")

    with abs_path.open("r", encoding=encoding) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, Mapping):
        raise ValueError(f"配置文件应解析为字典（mapping），实际为 {type(data)!r}: {abs_path}")

    # 转为普通 dict，方便后续递归处理
    return dict(data)


def _deep_merge_dict(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """
    递归深度合并：将 override 合并到 base 中（原地修改 base 并返回）。

    合并规则：
    - 如果同名 key 在 base 与 override 中都是 dict，则递归合并；
    - 否则，override 的值直接覆盖 base 中的值。
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _select_env_config(
    raw_data: Mapping[str, Any],
    *,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从带有 envs 分组的配置中，选出指定环境并与 default 合并。

    约定结构：

        # 顶层可包含一些“全局字段”，对所有环境生效
        some_global_key: ...

        envs:
          default:
            ...
          dev:
            ...
          prod:
            ...
    """
    # 1. 拆分“全局字段”和 envs 字段
    global_part: Dict[str, Any] = {
        k: v for k, v in raw_data.items() if k != "envs"
    }
    envs = raw_data.get("envs", {})

    if not isinstance(envs, Mapping):
        raise ValueError("配置文件的 'envs' 字段必须是一个字典（mapping）。")

    # 2. 决定当前环境名
    env_name = env or os.getenv(ENV_VAR_NAME) or DEFAULT_ENV_NAME

    default_cfg = envs.get(DEFAULT_ENV_NAME, {}) or {}
    if not isinstance(default_cfg, Mapping):
        raise ValueError(f"'envs.default' 必须是一个字典，实际为 {type(default_cfg)!r}")

    result: Dict[str, Any] = dict(global_part)
    _deep_merge_dict(result, dict(default_cfg))

    if env_name != DEFAULT_ENV_NAME:
        env_cfg = envs.get(env_name)
        if env_cfg is None:
            # 对于不存在的环境，保留 default 配置，同时给出明确错误提示
            raise KeyError(
                f"在配置中未找到环境 {env_name!r}，"
                f"请检查 'envs' 字段或设置 {ENV_VAR_NAME} 环境变量。"
            )
        if not isinstance(env_cfg, Mapping):
            raise ValueError(f"'envs.{env_name}' 必须是一个字典，实际为 {type(env_cfg)!r}")
        _deep_merge_dict(result, dict(env_cfg))

    return result


def _load_grouped_env_config(
    relative_path: str,
    *,
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    加载带有 envs 分组的配置文件，并根据环境名返回合并后的配置字典。
    """
    raw = _load_yaml_dict(relative_path, encoding=encoding)
    return _select_env_config(raw, env=env)


# ---------------------------------------------------------------------------
# 结构化配置模型（可以根据实际需求持续扩展）
# ---------------------------------------------------------------------------


@dataclass
class RagModelConfig:
    chat_model_name: str = "qwen3-max"
    embedding_model_name: str = "text-embedding-v4"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RagModelConfig":
        return cls(
            chat_model_name=data.get("chat_model_name", cls.chat_model_name),
            embedding_model_name=data.get(
                "embedding_model_name", cls.embedding_model_name
            ),
        )


@dataclass
class RagRetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.3

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RagRetrievalConfig":
        return cls(
            top_k=int(data.get("top_k", cls.top_k)),
            score_threshold=float(data.get("score_threshold", cls.score_threshold)),
        )


@dataclass
class RagConfig:
    """
    RAG 相关配置：
    - model: 大模型名称配置；
    - retrieval: 检索参数配置。
    """

    model: RagModelConfig = field(default_factory=RagModelConfig)
    retrieval: RagRetrievalConfig = field(default_factory=RagRetrievalConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RagConfig":
        model_cfg = RagModelConfig.from_dict(data.get("model", {}) or {})
        retrieval_cfg = RagRetrievalConfig.from_dict(data.get("retrieval", {}) or {})
        return cls(model=model_cfg, retrieval=retrieval_cfg)


@dataclass
class ChromaClientConfig:
    type: str = "chromadb"
    persist_directory: str = "chroma_db"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChromaClientConfig":
        return cls(
            type=data.get("type", cls.type),
            persist_directory=data.get("persist_directory", cls.persist_directory),
        )


@dataclass
class ChromaCollectionConfig:
    name: str = "agent"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChromaCollectionConfig":
        return cls(
            name=data.get("name", cls.name),
        )


@dataclass
class ChromaRetrievalConfig:
    k: int = 3

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChromaRetrievalConfig":
        return cls(
            k=int(data.get("k", cls.k)),
        )


@dataclass
class ChromaStorageConfig:
    data_path: str = "data"
    md5_hex_store: str = "md5.text"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChromaStorageConfig":
        return cls(
            data_path=data.get("data_path", cls.data_path),
            md5_hex_store=data.get("md5_hex_store", cls.md5_hex_store),
        )


@dataclass
class ChromaKnowledgeConfig:
    allow_knowledge_file_type: list[str] = field(default_factory=lambda: ["txt", "pdf"])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChromaKnowledgeConfig":
        default_types = ["txt", "pdf"]
        return cls(
            allow_knowledge_file_type=list(
                data.get("allow_knowledge_file_type", default_types)
            ),
        )


@dataclass
class ChromaProcessingConfig:
    chunk_size: int = 200
    chunk_overlap: int = 20
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", "! ", "?", ""]
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChromaProcessingConfig":
        default_separators = ["\n\n", "\n", ". ", "! ", "?", ""]
        return cls(
            chunk_size=int(data.get("chunk_size", cls.chunk_size)),
            chunk_overlap=int(data.get("chunk_overlap", cls.chunk_overlap)),
            separators=list(data.get("separators", default_separators)),
        )


@dataclass
class ChromaConfig:
    client: ChromaClientConfig = field(default_factory=ChromaClientConfig)
    collection: ChromaCollectionConfig = field(default_factory=ChromaCollectionConfig)
    retrieval: ChromaRetrievalConfig = field(default_factory=ChromaRetrievalConfig)
    storage: ChromaStorageConfig = field(default_factory=ChromaStorageConfig)
    knowledge: ChromaKnowledgeConfig = field(default_factory=ChromaKnowledgeConfig)
    processing: ChromaProcessingConfig = field(default_factory=ChromaProcessingConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChromaConfig":
        client_cfg = ChromaClientConfig.from_dict(data.get("client", {}) or {})
        collection_cfg = ChromaCollectionConfig.from_dict(
            data.get("collection", {}) or {}
        )
        retrieval_cfg = ChromaRetrievalConfig.from_dict(data.get("retrieval", {}) or {})
        storage_cfg = ChromaStorageConfig.from_dict(data.get("storage", {}) or {})
        knowledge_cfg = ChromaKnowledgeConfig.from_dict(data.get("knowledge", {}) or {})
        processing_cfg = ChromaProcessingConfig.from_dict(
            data.get("processing", {}) or {}
        )
        return cls(
            client=client_cfg,
            collection=collection_cfg,
            retrieval=retrieval_cfg,
            storage=storage_cfg,
            knowledge=knowledge_cfg,
            processing=processing_cfg,
        )


@dataclass
class PromptsConfig:
    """
    提示词配置，使用文件路径方式存储提示词位置。

    设计说明：
    - prompt_paths: 存储提示词文件的相对路径（相对于项目根目录）
    - debug: 调试相关配置
    - 提示词内容通过路径加载，便于版本控制和统一管理
    """

    prompt_paths: Dict[str, str] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptsConfig":
        return cls(
            prompt_paths=dict(data.get("prompt_paths", {}) or {}),
            debug=dict(data.get("debug", {}) or {}),
        )

    def get_prompt_path(self, key: str) -> Optional[str]:
        """
        获取指定 key 的提示词文件路径。

        :param key: 提示词配置的 key，例如 "main_prompt_path"、"rag_summarize_prompt_path"、"report_prompt_path"
        :return: 提示词文件的相对路径，如果不存在则返回 None
        """
        return self.prompt_paths.get(key)

    def get_prompt_abs_path(self, key: str) -> Optional[str]:
        """
        获取指定 key 的提示词文件绝对路径。

        :param key: 提示词配置的 key，例如 "main_prompt_path"、"rag_summarize_prompt_path"、"report_prompt_path"
        :return: 提示词文件的绝对路径，如果不存在则返回 None
        """
        rel_path = self.get_prompt_path(key)
        if rel_path is None:
            return None
        try:
            return get_abs_path(rel_path)
        except Exception:
            return None


@dataclass
class AgentToolConfig:
    name: str
    description: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentToolConfig":
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
        )


@dataclass
class AgentConfig:
    """
    智能体相关配置：
    - external_data_path: 外部数据文件路径（相对于项目根），供 usage_report 等工具使用；
    - agent: 类型、迭代次数、可用工具等；
    - conversation: 对话相关参数（最大轮数、语言等）。
    """

    external_data_path: str = "data/external/records.csv"
    agent_type: str = "react"
    max_iterations: int = 10
    tools: Dict[str, AgentToolConfig] = field(default_factory=dict)
    conversation: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        agent_section = data.get("agent", {}) or {}
        conv_section = data.get("conversation", {}) or {}

        tools_list = agent_section.get("tools", []) or []
        tools: Dict[str, AgentToolConfig] = {}
        for item in tools_list:
            if not isinstance(item, Mapping):
                continue
            tool_cfg = AgentToolConfig.from_dict(item)
            if tool_cfg.name:
                tools[tool_cfg.name] = tool_cfg

        return cls(
            external_data_path=str(
                data.get("external_data_path", cls.external_data_path)
            ),
            agent_type=agent_section.get("type", cls.agent_type),
            max_iterations=int(agent_section.get("max_iterations", cls.max_iterations)),
            tools=tools,
            conversation=dict(conv_section),
        )

    def get_external_data_abs_path(self) -> str:
        """
        获取外部数据文件的绝对路径。

        :return: external_data_path 对应的绝对路径，便于直接用于文件读写。
        """
        return get_abs_path(self.external_data_path)


@dataclass
class AppConfig:
    """
    应用整体配置聚合结构，方便在应用启动时一次性加载所有配置。
    """

    rag: RagConfig
    chroma: ChromaConfig
    prompts: PromptsConfig
    agent: AgentConfig


# ---------------------------------------------------------------------------
# 对外暴露的加载函数
# ---------------------------------------------------------------------------


def load_rag_config(
    *,  # 这个星号表示后面的参数都是“仅限关键字参数”（keyword-only），调用时必须写成 env=...、encoding=...
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> RagConfig:
    """
    加载 RAG 配置，内部根据 env / APP_ENV / default 选择环境并合并。
    """
    data = _load_grouped_env_config("config/rag.yml", env=env, encoding=encoding)
    return RagConfig.from_dict(data)


def load_chroma_config(
    *,  # 仅限关键字参数，调用时需写成 env=...、encoding=...
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> ChromaConfig:
    """
    加载向量库（Chroma）配置。
    """
    data = _load_grouped_env_config("config/chroma.yml", env=env, encoding=encoding)
    return ChromaConfig.from_dict(data)


def load_prompts_config(
    *,  # 仅限关键字参数，调用时需写成 env=...、encoding=...
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> PromptsConfig:
    """
    加载提示词配置。
    """
    data = _load_grouped_env_config("config/prompts.yml", env=env, encoding=encoding)
    return PromptsConfig.from_dict(data)


def load_agent_config(
    *,  # 仅限关键字参数，调用时需写成 env=...、encoding=...
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> AgentConfig:
    """
    加载智能体配置。
    """
    data = _load_grouped_env_config("config/agent.yml", env=env, encoding=encoding)
    return AgentConfig.from_dict(data)


def load_all_configs(
    *,  # 仅限关键字参数，调用时需写成 env=...、encoding=...
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> AppConfig:
    """
    一次性加载所有核心配置，建议在应用启动时调用。
    """
    rag = load_rag_config(env=env, encoding=encoding)
    chroma = load_chroma_config(env=env, encoding=encoding)
    prompts = load_prompts_config(env=env, encoding=encoding)
    agent = load_agent_config(env=env, encoding=encoding)
    return AppConfig(rag=rag, chroma=chroma, prompts=prompts, agent=agent)


__all__ = [
    "ENV_VAR_NAME",
    "DEFAULT_ENV_NAME",
    "RagModelConfig",
    "RagRetrievalConfig",
    "RagConfig",
    "ChromaClientConfig",
    "ChromaCollectionConfig",
    "ChromaRetrievalConfig",
    "ChromaStorageConfig",
    "ChromaKnowledgeConfig",
    "ChromaProcessingConfig",
    "ChromaConfig",
    "PromptsConfig",
    "AgentToolConfig",
    "AgentConfig",
    "AppConfig",
    "load_rag_config",
    "load_chroma_config",
    "load_prompts_config",
    "load_agent_config",
    "load_all_configs",
]

if __name__ == "__main__":
    """
    简单自测：
    - 在不同 APP_ENV 下运行本模块，观察配置合并是否符合预期；
    - 也可以通过显式传入 env 参数测试指定环境。

    运行示例：
        # 在项目根目录下
        python -m zhisaotong_agent.utils.config_handler
    """
    current_env = os.getenv(ENV_VAR_NAME, DEFAULT_ENV_NAME)
    print(f"当前 APP_ENV = {current_env!r}")

    rag = load_rag_config()
    print("[RAG]", rag)

    chroma = load_chroma_config()
    print("[Chroma]", chroma)

    prompts = load_prompts_config()
    print("[Prompts]", prompts)

    agent = load_agent_config()
    print("[Agent]", agent)