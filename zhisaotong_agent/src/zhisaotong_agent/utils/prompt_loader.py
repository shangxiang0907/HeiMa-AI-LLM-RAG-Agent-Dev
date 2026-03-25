"""
提示词文件加载工具模块。

功能：
- 从配置文件读取提示词文件路径，并加载文件内容
- 支持不同环境（default/dev/prod）的提示词配置
- 提供便捷的接口函数：load_system_prompts、load_rag_prompts、load_report_prompts

使用示例：
    from zhisaotong_agent.utils.prompt_loader import (
        load_system_prompts,
        load_rag_prompts,
        load_report_prompts,
    )

    system_prompt = load_system_prompts()
    rag_prompt = load_rag_prompts(env="dev")
    report_prompt = load_report_prompts()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from zhisaotong_agent.utils.config_handler import load_prompts_config
from zhisaotong_agent.utils.logger_handler import get_logger

logger = get_logger(__name__)


def _load_prompt_file(
    key: str,
    *,
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> str:
    """
    内部辅助函数：根据配置中的 key 加载对应的提示词文件内容。

    :param key: 提示词配置的 key，例如 "main_prompt_path"、"rag_summarize_prompt_path"、"report_prompt_path"
    :param env: 环境名称，默认使用 APP_ENV 或 "default"
    :param encoding: 文件编码，默认为 "utf-8"
    :return: 提示词文件的内容（字符串）
    :raises FileNotFoundError: 如果提示词文件不存在
    :raises KeyError: 如果配置中不存在指定的 key
    """
    config = load_prompts_config(env=env, encoding=encoding)
    abs_path = config.get_prompt_abs_path(key)
    
    if abs_path is None:
        error_msg = f"提示词配置中不存在 key: {key!r}"
        logger.error(error_msg)
        raise KeyError(error_msg)

    path_obj = Path(abs_path)
    if not path_obj.exists():
        error_msg = f"提示词文件不存在: {abs_path} (key: {key!r})"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with path_obj.open("r", encoding=encoding) as f:
            content = f.read()
        logger.debug(f"成功加载提示词文件: key={key!r}, path={abs_path}, 长度={len(content)}")
        return content
    except Exception as e:
        error_msg = f"读取提示词文件失败: path={abs_path}, key={key!r}, error={e}"
        logger.error(error_msg)
        raise IOError(error_msg) from e


def load_system_prompts(
    *,
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> str:
    """
    加载系统提示词（主提示词）内容。

    对应配置中的 `main_prompt_path`，通常用于智能体的系统级提示词。

    :param env: 环境名称，默认使用 APP_ENV 或 "default"
    :param encoding: 文件编码，默认为 "utf-8"
    :return: 系统提示词文件的内容（字符串）
    :raises FileNotFoundError: 如果提示词文件不存在
    :raises KeyError: 如果配置中不存在 "main_prompt_path"
    """
    return _load_prompt_file("main_prompt_path", env=env, encoding=encoding)


def load_rag_prompts(
    *,
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> str:
    """
    加载 RAG 相关提示词内容。

    对应配置中的 `rag_summarize_prompt_path`，通常用于 RAG 检索后的总结与回答生成。

    :param env: 环境名称，默认使用 APP_ENV 或 "default"
    :param encoding: 文件编码，默认为 "utf-8"
    :return: RAG 提示词文件的内容（字符串）
    :raises FileNotFoundError: 如果提示词文件不存在
    :raises KeyError: 如果配置中不存在 "rag_summarize_prompt_path"
    """
    return _load_prompt_file("rag_summarize_prompt_path", env=env, encoding=encoding)


def load_report_prompts(
    *,
    env: Optional[str] = None,
    encoding: str = "utf-8",
) -> str:
    """
    加载报告生成相关提示词内容。

    对应配置中的 `report_prompt_path`，通常用于生成用户使用报告与优化建议。

    :param env: 环境名称，默认使用 APP_ENV 或 "default"
    :param encoding: 文件编码，默认为 "utf-8"
    :return: 报告提示词文件的内容（字符串）
    :raises FileNotFoundError: 如果提示词文件不存在
    :raises KeyError: 如果配置中不存在 "report_prompt_path"
    """
    return _load_prompt_file("report_prompt_path", env=env, encoding=encoding)


__all__ = [
    "load_system_prompts",
    "load_rag_prompts",
    "load_report_prompts",
]


if __name__ == "__main__":
    """
    简单自测代码：
    - 测试三个加载函数是否能正常工作
    - 运行方式（在项目根目录）：
        python -m zhisaotong_agent.utils.prompt_loader
    """
    import os
    from zhisaotong_agent.utils.config_handler import (
        ENV_VAR_NAME,
        DEFAULT_ENV_NAME,
    )

    current_env = os.getenv(ENV_VAR_NAME, DEFAULT_ENV_NAME)
    print(f"当前 APP_ENV = {current_env!r}")
    print()

    try:
        print("=" * 80)
        print("测试 load_system_prompts()")
        print("-" * 80)
        system_prompt = load_system_prompts()
        print(f"成功加载，内容长度: {len(system_prompt)} 字符")
        print(f"内容预览（前200字符）: {system_prompt[:200]}...")
        print()
    except Exception as e:
        print(f"加载失败: {e}")
        print()

    try:
        print("=" * 80)
        print("测试 load_rag_prompts()")
        print("-" * 80)
        rag_prompt = load_rag_prompts()
        print(f"成功加载，内容长度: {len(rag_prompt)} 字符")
        print(f"内容预览（前200字符）: {rag_prompt[:200]}...")
        print()
    except Exception as e:
        print(f"加载失败: {e}")
        print()

    try:
        print("=" * 80)
        print("测试 load_report_prompts()")
        print("-" * 80)
        report_prompt = load_report_prompts()
        print(f"成功加载，内容长度: {len(report_prompt)} 字符")
        print(f"内容预览（前200字符）: {report_prompt[:200]}...")
        print()
    except Exception as e:
        print(f"加载失败: {e}")
        print()
