"""
工具调用与模型调用相关的中间件（监控、日志、动态提示词切换等）。

约束：
- 对外交互保持稳定：函数名、装饰器、入参与返回值类型不变；
- 仅增强健壮性与生产可用性（日志安全、异常保留栈、边界条件处理）。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from zhisaotong_agent.utils.logger_handler import get_logger
from zhisaotong_agent.utils.prompt_loader import load_report_prompts, load_system_prompts

logger = get_logger(__name__)


def _redact_mapping(obj: Mapping[str, Any]) -> dict[str, Any]:
    """
    对常见敏感字段做脱敏，避免把密钥/令牌/隐私直接写入日志。
    仅用于日志展示，不影响工具真实入参。
    """

    sensitive_keys = {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "session",
        "session_id",
        "phone",
        "mobile",
        "email",
        "id_card",
    }

    redacted: dict[str, Any] = {}
    for k, v in obj.items():
        key_lower = str(k).lower()
        if key_lower in sensitive_keys:
            redacted[k] = "***REDACTED***"
        else:
            redacted[k] = v
    return redacted


def _safe_preview(value: Any, *, max_len: int = 2000) -> str:
    """
    将任意对象转为可安全打印的短字符串，防止日志爆炸/序列化异常。
    """

    try:
        if isinstance(value, Mapping):
            value = _redact_mapping(value)  # type: ignore[assignment]
        text = repr(value)
    except Exception:
        text = "<unreprable>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    工具调用监控中间件。

    关于 @wrap_tool_call 装饰器（简化理解）：
    - 在 LangGraph/Agent 执行链中，每当「需要调用某个工具」时，框架会把原本
      直接执行工具的步骤“包裹”成一个中间件调用；
    - 这个中间件的签名约定为 monitor_tool(request, handler)：
        - request: ToolCallRequest，对应“这一次工具调用”的上下文（包括工具名、
          参数、runtime 等信息）；
        - handler: 真正执行工具的函数（可以理解成“next” 或 “inner handler”）。
    - monitor_tool 里可以：
        - 在调用前后打日志；
        - 修改 request 或 runtime.context；
        - 决定是否继续调用 handler，或者短路返回。

    本函数的核心逻辑是：
    - 从 request 中取出工具名称与参数，用安全方式打印日志（避免泄露敏感信息）；
    - 调用 handler(request) 让真正的工具执行；
    - 如果是特定工具 fill_context_for_report，就在 runtime.context 中打一个标记，
      供后续 @dynamic_prompt 中间件判断是否切换到“报告场景”的提示词。
    """

    tool_call = getattr(request, "tool_call", None) or {}
    tool_name = tool_call.get("name", "<unknown>")
    tool_args = tool_call.get("args", None)

    # 生产最佳实践：避免直接打印完整参数（可能含敏感信息/超长文本）
    logger.info("[tool monitor]执行工具：%s", tool_name)
    logger.info("[tool monitor]传入参数：%s", _safe_preview(tool_args))

    try:
        result = handler(request)
        logger.info("[tool monitor]工具%s调用成功", tool_name)

        # 保持外部交互一致：仍然以同样 key 打标记
        if tool_name == "fill_context_for_report":
            # Runtime.context 通常是跨节点共享的 dict；这里仅设置布尔标志位
            request.runtime.context["report"] = True

        return result
    except Exception:
        # 使用 exception 记录堆栈，且用 bare raise 保留原始 traceback
        logger.exception("工具%s调用失败", tool_name)
        raise


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """
    在模型执行前输出日志（消息条数与最新一条消息概览）的中间件。

    关于 @before_model 装饰器（简化理解）：
    - 在一次 Agent / Graph 的循环里，当“上一个节点”已经准备好了 messages，
      即将进入「调用大模型」这一步时，框架会先依次执行所有 @before_model
      注册的中间件；
    - 这些中间件的签名固定为 log_before_model(state, runtime)：
        - state: 当前 AgentState，里边最关键的是 state["messages"]，包含了
          系统/用户/AI/工具消息等；
        - runtime: LangGraph Runtime，包含运行上下文、配置、链路信息等。

    本函数只做只读日志：
    - 统计消息条数，方便排查“重复追加历史消息”等问题；
    - 打印最后一条消息的类型与内容（经过 _safe_preview 截断），方便观测当前
      发给模型的最后一条输入长什么样；
    - 不修改 state 与 runtime，保证是“零副作用”的观测中间件。
    """

    messages = state.get("messages") if isinstance(state, dict) else None
    msg_count = len(messages) if isinstance(messages, list) else 0
    logger.info("[log_before_model]即将调用模型，带有%d条消息。", msg_count)

    if not messages:
        return None

    last = messages[-1]
    try:
        content = getattr(last, "content", None)
        content_text = content.strip() if isinstance(content, str) else _safe_preview(content)
        logger.debug("[log_before_model]%s | %s", type(last).__name__, content_text)
    except Exception:
        logger.debug("[log_before_model]%s | <unloggable message>", type(last).__name__)

    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """
    动态切换提示词：当 runtime.context['report'] 为 True 时切换为报告提示词。

    关于 @dynamic_prompt 装饰器（简化理解）：
    - 每次框架要“构造给模型的系统提示词 / 主提示词”时，会调用此函数来获取
      本次调用应该使用的 prompt 内容；
    - 传入的 request: ModelRequest，里面包含本次模型调用相关的上下文信息，
      包括 runtime.context（可在工具中间件里写入标志位）。

    本函数的核心逻辑：
    - 读取 runtime.context["report"] 这个布尔标记；
        - 该标记由 monitor_tool 在特定工具（fill_context_for_report）执行后设置；
    - 如果是报告场景（report=True），则加载报告专用提示词；
    - 否则回退到系统默认提示词。

    这样就把「是否进入报告生成场景」这个业务决策，从 prompt 模板里解耦出来，
    统一用 runtime.context 这个“执行上下文状态”进行传递。
    """

    is_report = bool(request.runtime.context.get("report", False))
    if is_report:
        return load_report_prompts()
    return load_system_prompts()


if __name__ == "__main__":
    """
    简单自测代码（smoke test）：
    - 验证脱敏与日志预览函数不会抛异常；
    - 用最小 dummy 对象验证 report 标记与日志函数的健壮性；
    - 提示词加载依赖 config/prompts 配置文件，若缺失会打印友好错误。

    运行方式（在项目根目录）：
        python -m zhisaotong_agent.agent.tools.middleware
    """

    class _DummyRuntime:
        def __init__(self):
            self.context: dict[str, Any] = {}

    class _DummyToolRequest:
        def __init__(self, name: str, args: Any):
            self.tool_call = {"name": name, "args": args}
            self.runtime = _DummyRuntime()

    class _DummyMsg:
        def __init__(self, content: Any):
            self.content = content

    print("== middleware smoke test ==")
    print("safe_preview:", _safe_preview({"token": "abc", "k": "v", "text": "x" * 10}))

    dummy_req = _DummyToolRequest(
        "fill_context_for_report",
        {"user_id": "1001", "token": "should-not-leak", "note": "x" * 5000},
    )

    def _dummy_handler(_req: Any):
        return ToolMessage(content="ok", tool_call_id="smoke_test")

    try:
        # 这些函数上方有 langchain/langgraph 的装饰器。
        # 装饰器可能会把“函数”替换成“中间件对象”，导致它在运行时不再是可直接调用的普通函数。
        # 一种常见做法是通过 functools.wraps 保留原函数在 __wrapped__ 里；
        # 因此这里尽量尝试调用 __wrapped__，但前提是它确实存在且可调用。
        _monitor = getattr(monitor_tool, "__wrapped__", None)
        if callable(_monitor):
            _monitor(dummy_req, _dummy_handler)  # type: ignore[misc]
            print("monitor_tool(__wrapped__): ok; runtime.context['report'] =", dummy_req.runtime.context.get("report"))
        else:
            # 如果没有 __wrapped__，说明装饰器没有保留原函数，或装饰后对象不是可调用函数。
            # 这种情况下跳过调用测试，避免自测依赖框架内部实现细节而误报。
            print("monitor_tool: decorated object is not callable; skip call smoke test")
    except Exception as e:
        print("monitor_tool: failed:", e)

    try:
        _before = getattr(log_before_model, "__wrapped__", None)
        # 同上：尽量调用被装饰器包装前的原函数进行 smoke test。
        if callable(_before):
            _before({"messages": [_DummyMsg(" hello ")]}, _DummyRuntime())  # type: ignore[misc]
            print("log_before_model(__wrapped__): ok")
        else:
            # 保持自测稳健性：装饰器实现变化时不会导致自测崩溃。
            print("log_before_model: decorated object is not callable; skip call smoke test")
    except Exception as e:
        print("log_before_model: failed:", e)

    class _DummyModelRequest:
        def __init__(self, report_flag: bool):
            self.runtime = _DummyRuntime()
            self.runtime.context["report"] = report_flag

    for flag in (False, True):
        try:
            # report_prompt_switch 同样可能被框架装饰器包装为不可直接调用对象。
            _switch = getattr(report_prompt_switch, "__wrapped__", None)
            if callable(_switch):
                prompt = _switch(_DummyModelRequest(flag))  # type: ignore[misc]
                print(f"report_prompt_switch(__wrapped__)(report={flag}): ok; prompt_len={len(prompt)}")
            else:
                # 依赖最少：没有 __wrapped__ 就不调用，只提示跳过。
                print(f"report_prompt_switch: decorated object is not callable; skip call smoke test (report={flag})")
        except Exception as e:
            print(f"report_prompt_switch(report={flag}): failed:", e)

