"""
基于 Streamlit 的问答聊天页面

功能：
- 使用项目中的 RagService 进行基于知识库的问答
- 支持多会话：新建会话、切换会话、删除会话
- 对话为流式传输，实时展示模型回答
- 会话历史在页面中展示（会话内容在后端文件中持久化）

运行方式：
    streamlit run app_qa.py
"""

import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

import config_data as config
from rag import RagService
from file_history_store import chat_history_store, get_history


def init_api_key():
    """初始化并校验 DashScope API Key"""
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        st.error("未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请在 .env 或系统环境中配置后再运行。")
        return False
    os.environ["DASHSCOPE_API_KEY"] = api_key
    return True


def init_services():
    """初始化 RagService 与会话链"""
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RagService()
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = st.session_state.rag_service.get_conversation_chain()


def get_storage_path() -> str:
    """获取会话历史存储路径"""
    if "rag_service" in st.session_state:
        return st.session_state.rag_service.storage_path
    # 与 RagService 默认保持一致
    return config.chat_history_path


def list_sessions() -> list[str]:
    """列出已有的会话 ID（基于存储目录的文件名）"""
    storage_path = get_storage_path()
    if not os.path.exists(storage_path):
        return []
    return sorted(
        [
            fname
            for fname in os.listdir(storage_path)
            if os.path.isfile(os.path.join(storage_path, fname))
        ]
    )


def ensure_session_state_for_session(session_id: str):
    """为指定会话 ID 初始化 Streamlit 内存中的聊天记录

    优先从文件历史记录中恢复（如果存在），
    这样刷新页面或切换会话时可以看到完整历史。
    """
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if session_id not in st.session_state.chat_sessions:
        # 尝试从持久化历史中加载
        history_messages = []
        try:
            storage_path = get_storage_path()
            history = get_history(session_id, storage_path)
            for msg in history.messages:
                # LangChain 消息类型一般为 "human" / "ai" / "system"
                role = "assistant"
                msg_type = getattr(msg, "type", "")
                if msg_type == "human":
                    role = "user"
                elif msg_type == "ai":
                    role = "assistant"
                elif msg_type == "system":
                    # Streamlit 只支持 user/assistant，这里也作为 assistant 展示
                    role = "assistant"
                content = getattr(msg, "content", "")
                history_messages.append({"role": role, "content": content})
        except Exception as e:
            # 历史加载失败不影响新会话，只做提示
            st.warning(f"加载会话历史失败：{e}")
            history_messages = []

        st.session_state.chat_sessions[session_id] = history_messages


def new_session_id() -> str:
    """生成新的会话 ID"""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def delete_session(session_id: str):
    """删除指定会话（文件 + 内存）"""
    storage_path = get_storage_path()
    file_path = os.path.join(storage_path, session_id)
    # 删除会话文件
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            st.warning(f"删除会话文件失败: {e}")
    # 删除内存中的历史缓存
    chat_history_store.pop(session_id, None)
    if "chat_sessions" in st.session_state:
        st.session_state.chat_sessions.pop(session_id, None)


def sidebar_session_manager():
    """侧边栏会话管理 UI：选择会话 / 新建会话 / 删除会话"""
    st.sidebar.header("会话管理")

    existing_sessions = list_sessions()

    # 当前会话 ID
    if "current_session_id" not in st.session_state:
        if existing_sessions:
            st.session_state.current_session_id = existing_sessions[0]
        else:
            st.session_state.current_session_id = new_session_id()

    # 新建会话按钮
    if st.sidebar.button("➕ 新建会话"):
        st.session_state.current_session_id = new_session_id()
        ensure_session_state_for_session(st.session_state.current_session_id)
        st.sidebar.success(f"已创建新会话：{st.session_state.current_session_id}")

    # 会话选择下拉框（包含当前会话，即使它还没有对应文件）
    display_sessions = existing_sessions.copy()
    if st.session_state.current_session_id not in display_sessions:
        display_sessions.insert(0, st.session_state.current_session_id)

    if display_sessions:
        try:
            index = display_sessions.index(st.session_state.current_session_id)
        except ValueError:
            index = 0

        # 使用单选列表而不是下拉框，让会话列表一目了然
        selected = st.sidebar.radio(
            "选择会话",
            display_sessions,
            index=index,
        )
        st.session_state.current_session_id = selected

    st.sidebar.caption(f"当前会话 ID：`{st.session_state.current_session_id}`")

    # 删除当前会话
    if st.sidebar.button("🗑 删除当前会话"):
        to_delete = st.session_state.current_session_id
        delete_session(to_delete)
        # 删除后自动切换到其他会话或新建一个
        remaining = list_sessions()
        if remaining:
            st.session_state.current_session_id = remaining[0]
        else:
            st.session_state.current_session_id = new_session_id()
        st.sidebar.success(f"已删除会话：{to_delete}")


def render_chat_messages(session_id: str):
    """渲染当前会话的聊天记录"""
    ensure_session_state_for_session(session_id)
    for msg in st.session_state.chat_sessions[session_id]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main():
    st.set_page_config(page_title="RAG 服装客服 - 问答助手", page_icon="👕", layout="wide")
    st.title("👕 RAG 服装客服问答助手")
    st.caption("基于知识库的智能客服，支持多会话与流式回答。")

    if not init_api_key():
        return

    init_services()
    sidebar_session_manager()

    current_session_id = st.session_state.current_session_id
    ensure_session_state_for_session(current_session_id)

    # 展示历史消息
    render_chat_messages(current_session_id)

    # 底部输入框
    user_input = st.chat_input("请输入你的问题（例如：我身高180cm，140斤，适合穿多大尺码？）")

    if user_input:
        # 先在页面和内存中记录用户提问
        st.session_state.chat_sessions[current_session_id].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # 准备对话配置（用于后端历史记录）
        session_config = {"configurable": {"session_id": current_session_id}}

        # 展示助手流式回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            def stream_answer():
                full_text = ""
                try:
                    for chunk in st.session_state.conversation_chain.stream(
                        {"input": user_input},
                        config=session_config,
                    ):
                        # 这里 chunk 已经是字符串（StrOutputParser 输出）
                        full_text_inner = full_text + chunk
                        full_text = full_text_inner
                        yield chunk

                except Exception as e:
                    error_msg = f"调用模型出错：{e}"
                    st.error(error_msg)
                    yield "\n\n[错误] 调用模型失败，请检查日志与配置。"
                    return

                # 将完整回答保存到会话历史
                st.session_state.chat_sessions[current_session_id].append(
                    {"role": "assistant", "content": full_text}
                )

            # 使用 Streamlit 原生流式输出
            message_placeholder.write_stream(stream_answer)


if __name__ == "__main__":
    main()

