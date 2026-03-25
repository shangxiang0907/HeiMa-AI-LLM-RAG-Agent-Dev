"""
ReAct 智能体封装：组合模型、系统提示、业务工具与中间件，并提供流式执行入口。

对外保持稳定：
- ``ReactAgent()``：无必填构造参数；
- ``execute_stream(query: str)``：入参仅为用户问题字符串，按块产出字符串（与原有 strip + 换行语义一致）。
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain.agents import create_agent

from zhisaotong_agent.agent.tools.agent_tools import TOOLS_LIST
from zhisaotong_agent.agent.tools.middleware import (
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from zhisaotong_agent.model.factory import get_chat_model
from zhisaotong_agent.utils.api_key import init_dashscope_api_key
from zhisaotong_agent.utils.prompt_loader import load_system_prompts

__all__ = ["ReactAgent"]


class ReactAgent:
    """基于 LangChain ``create_agent`` 的 ReAct 封装，供应用入口（如 app.py）调用。"""

    def __init__(self) -> None:
        self.agent = create_agent(
            model=get_chat_model(),
            system_prompt=load_system_prompts(),
            tools=TOOLS_LIST,
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str) -> Iterator[str]:
        input_dict: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }
        # 与原先一致：context 承载 runtime 标记；每次请求新建 dict，避免并发下 report 状态串扰
        runtime_context: dict[str, Any] = {"report": False}

        for chunk in self.agent.stream(
            input_dict,
            stream_mode="values",
            context=runtime_context,
        ):
            messages = chunk.get("messages")
            if not messages:
                continue
            latest_message = messages[-1]
            content = getattr(latest_message, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                text = content.strip()
                if not text:
                    continue
                yield text + "\n"
            else:
                # 非字符串 content（如部分模型的块列表）时避免 .strip() 抛错，仍保持「有则输出」
                text = str(content).strip()
                if text:
                    yield text + "\n"


if __name__ == "__main__":
    """
    自测：在项目根目录运行 ``python -m zhisaotong_agent.agent.react_agent``。
    与 ``model.factory``、``rag.rag_service`` 等模块一致：仅在脚本入口加载 .env 并校验 Key。

    覆盖两条业务路径：
    - 非报告：常规问答，``runtime.context['report']`` 保持 False，``report_prompt_switch`` 使用系统提示词；
    - 生成报告：触发 ``fill_context_for_report`` 后切换报告提示词并完成报告链路。
    """
    _ok = init_dashscope_api_key()
    if not _ok:
        raise SystemExit("DASHSCOPE_API_KEY 未正确配置，无法运行 ReactAgent 自测。")

    _agent = ReactAgent()

    def _run_demo(_title: str, _query: str) -> None:
        print(f"\n{'=' * 60}\n{_title}\n{'=' * 60}\n", flush=True)
        for _chunk in _agent.execute_stream(_query):
            print(_chunk, end="", flush=True)
        print("\n", flush=True)

    _run_demo(
        "自测 1：非报告场景（常规问答；不应依赖「使用报告」专用提示词链路）",
        "你好，扫地机器人尘盒一般多久清理一次？请简短回答。",
    )
    _run_demo(
        "自测 2：报告场景（应走 fill_context_for_report / 外部数据等报告生成逻辑）",
        "给我生成我的使用报告",
    )
