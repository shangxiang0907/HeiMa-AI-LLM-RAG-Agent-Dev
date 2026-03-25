"""
智扫通 Streamlit 入口：会话展示、流式回复与基础生产化防护。

运行（在项目根目录）::

    streamlit run zhisaotong_agent/app.py
"""

from __future__ import annotations

import streamlit as st

from zhisaotong_agent.agent.react_agent import ReactAgent
from zhisaotong_agent.utils.api_key import init_dashscope_api_key
from zhisaotong_agent.utils.logger_handler import get_logger

logger = get_logger(__name__)

# 会话消息条数上限（user/assistant 各算一条），防止长时间对话撑爆 session 与首屏渲染
_MAX_SESSION_MESSAGES = 200


def _trim_messages(messages: list[dict[str, str]], max_items: int) -> None:
    while len(messages) > max_items:
        messages.pop(0)


st.set_page_config(page_title="智扫通机器人智能客服", layout="centered")
st.title("智扫通机器人智能客服")
st.divider()

if "_dashscope_ok" not in st.session_state:
    st.session_state["_dashscope_ok"] = init_dashscope_api_key()

if not st.session_state["_dashscope_ok"]:
    st.error(
        "未检测到有效的 DASHSCOPE_API_KEY（或 API_KEY）。"
        "请在环境变量或项目根目录 .env 中配置后刷新页面。"
    )
    st.stop()

if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = ReactAgent()
    except Exception:
        logger.exception("ReactAgent 初始化失败")
        st.error("智能体初始化失败，请检查配置文件与依赖服务是否正常。")
        st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("请输入您的问题…")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    _trim_messages(st.session_state["messages"], _MAX_SESSION_MESSAGES)

    with st.chat_message("user"):
        st.markdown(prompt)

    collected: list[str] = []

    def _stream() -> str:
        for chunk in st.session_state["agent"].execute_stream(prompt):
            collected.append(chunk)
            yield chunk

    assistant_text: str
    with st.chat_message("assistant"):
        with st.spinner("智能客服思考中…"):
            try:
                st.write_stream(_stream())
            except Exception:
                logger.exception("execute_stream 失败, prompt_len=%s", len(prompt))
                st.error("抱歉，当前请求处理失败，请稍后重试。")
                assistant_text = "（本次回复生成失败，请重试。）"
            else:
                assistant_text = "".join(collected).strip()
                if not assistant_text:
                    assistant_text = "（未返回有效内容，请换种方式提问或稍后重试。）"

    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
    _trim_messages(st.session_state["messages"], _MAX_SESSION_MESSAGES)
