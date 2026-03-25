from __future__ import annotations

"""
RAG 总结服务模块。

在不改变对外接口的前提下，对原始 Demo 版实现进行“生产化”改造：
- 使用统一配置与日志；
- 增强异常处理与空结果处理；
- 控制上下文长度，避免超长 prompt；
- 对元数据做安全过滤，避免直接暴露内部字段；
- 保持 `RagSummarizeService` 的公开方法签名不变：
    - `__init__(self)`
    - `retriever_docs(self, query: str) -> list[Document]`
    - `rag_summarize(self, query: str) -> str`
"""

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from zhisaotong_agent.model.factory import get_chat_model
from zhisaotong_agent.rag.vector_store import VectorStoreService
from zhisaotong_agent.utils.config_handler import RagConfig, load_rag_config
from zhisaotong_agent.utils.logger_handler import get_logger
from zhisaotong_agent.utils.prompt_loader import load_rag_prompts
from zhisaotong_agent.utils.api_key import init_dashscope_api_key


logger = get_logger(__name__)


def _safe_debug_enabled() -> bool:
    """
    根据配置或环境变量决定是否打印/记录完整 prompt 等调试信息。

    当前简化实现：根据 logger 级别判断，后续可以接入专门的 debug 配置段。
    """
    return logger.isEnabledFor(10)  # logging.DEBUG


def _print_or_log_prompt(prompt: PromptTemplate | Any) -> PromptTemplate | Any:
    """
    链路中的调试节点：
    - 在 DEBUG 级别下，将最终渲染后的 prompt 记录到日志；
    - 避免直接使用 print，统一走项目日志体系。
    """
    try:
        if _safe_debug_enabled() and hasattr(prompt, "to_string"):
            # LangChain PromptTemplate 的调试输出
            prompt_str = prompt.to_string()
            logger.debug("RAG Prompt 模板内容：\n%s", prompt_str)
    except Exception as e:  # noqa: BLE001
        # 调试日志失败不应影响正常业务
        logger.warning("记录 RAG Prompt 调试信息失败：%s", e)
    return prompt


def _build_context_from_docs(
    docs: Iterable[Document],
    *,
    max_docs: int = 5,
    max_chars: int = 4000,
) -> str:
    """
    将检索到的 Document 列表拼接为给模型的上下文字符串。

    - 仅使用前 max_docs 条文档；
    - 控制总字符数不超过 max_chars（粗粒度控制，避免 prompt 过长）；
    - 对 metadata 做字段白名单过滤，仅暴露用户可读的关键信息。
    """

    def _filter_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        # 简单白名单：常见标题 / 来源字段；其他字段在生产中可按需扩展
        allow_keys = {"title", "source", "category", "file_name"}
        return {k: v for k, v in meta.items() if k in allow_keys}

    lines: List[str] = []
    total_chars = 0

    for idx, doc in enumerate(docs, start=1):
        if idx > max_docs:
            break

        meta_filtered = _filter_metadata(doc.metadata or {})
        meta_str = f" | 参考元信息：{meta_filtered}" if meta_filtered else ""
        snippet = f"【参考资料{idx}】内容：{doc.page_content}{meta_str}\n"

        # 粗略长度控制，避免超出 max_chars 太多
        if total_chars + len(snippet) > max_chars:
            # 尝试截断 snippet
            allowed = max_chars - total_chars
            if allowed <= 0:
                break
            snippet = snippet[:allowed]

        lines.append(snippet)
        total_chars += len(snippet)

        if total_chars >= max_chars:
            break

    return "".join(lines)


class RagSummarizeService:
    """
    RAG 总结服务：
    - 对外接口保持简单：传入 query，内部完成检索与模型生成，返回最终回答字符串；
    - 内部实现接入项目统一配置与日志体系，具备一定的生产可用性。
    """

    def __init__(self) -> None:
        # 加载 RAG 相关配置（检索参数等），失败时向上抛出，让上层在启动阶段感知问题。
        self._rag_conf: RagConfig = load_rag_config()

        # 初始化向量库服务与检索器
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

        # 加载 RAG 提示词模板
        prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(prompt_text)

        # 获取聊天模型实例（懒加载封装在工厂函数内）
        self.model = get_chat_model()

        # 初始化 LCEL 链：PromptTemplate -> （可选调试输出）-> 模型 -> 输出解析
        self.chain = self._init_chain()

        logger.info("RagSummarizeService 初始化完成。")

    def _init_chain(self):
        """
        构建 LangChain Expression Language (LCEL) 链。

        保持结构简单，后续如果需要增加重试 / 监控中间件，可在此集中调整。
        """
        return self.prompt_template | _print_or_log_prompt | self.model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        """
        对外暴露的检索函数，保持原方法签名不变。

        在生产环境中：
        - 会记录检索前后的基本信息；
        - 对异常进行捕获并记录错误日志，然后向上抛出，让上层决定兜底策略。
        """
        try:
            logger.debug("开始向量检索，query=%r", query)
            docs: List[Document] = list(self.retriever.invoke(query))
            logger.info("向量检索完成，query=%r, 命中文档数=%d", query, len(docs))
            return docs
        except Exception as e:  # noqa: BLE001
            logger.error("向量检索失败，query=%r, error=%s", query, e, exc_info=True)
            # 保持接口语义：如果检索失败，向上抛出异常，由调用方选择是兜底还是直接报错
            raise

    def rag_summarize(self, query: str) -> str:
        """
        核心对外接口：给定用户提问，返回基于知识库检索 + 大模型总结的回答。

        - 对外仍然返回纯字符串，以兼容已有调用方；
        - 内部增加异常处理和空检索兜底逻辑；
        - 对上下文长度和参考文档数量做基础控制。
        """
        # 1. 检索阶段
        try:
            context_docs = self.retriever_docs(query)
        except Exception:
            # 检索阶段如果直接失败，给出温和的报错提示
            logger.warning("RAG 检索阶段失败，fallback 到纯模型回答模式。query=%r", query)
            context = ""
        else:
            # 2. 处理“无相关文档”情况
            if not context_docs:
                logger.info("RAG 检索结果为空，query=%r", query)
                context = ""
            else:
                # 3. 构建受控长度的上下文字符串
                # max_docs / max_chars 可以后续接入配置
                context = _build_context_from_docs(
                    context_docs,
                    max_docs=self._rag_conf.retrieval.top_k,
                    max_chars=4000,
                )

        # 4. 调用模型生成回答
        try:
            result: str = self.chain.invoke(
                {
                    "input": query,
                    "context": context,
                }
            )
            logger.info("RAG 总结调用成功，query=%r, answer_len=%d", query, len(result))
            return result
        except Exception as e:  # noqa: BLE001
            # 统一捕捉生成阶段异常，避免直接把底层异常信息暴露给前端
            logger.error("RAG 总结调用失败，query=%r, error=%s", query, e, exc_info=True)
            # 为了保持接口简单，这里抛出一个通用异常，也可以按需返回固定文案
            raise RuntimeError("RAG 总结服务暂时不可用，请稍后重试。") from e


if __name__ == "__main__":
    """
    简单自测：
    - 需在项目根目录运行：
        python -m zhisaotong_agent.rag.rag_service
    - 用于快速验证 RAG 服务是否能正常跑通。
    """
    ok = init_dashscope_api_key()
    if not ok:
        raise SystemExit("DASHSCOPE_API_KEY 未正确配置，无法运行 RAG 总结服务自测。")

    service = RagSummarizeService()
    question = "小户型适合哪些扫地机器人？"
    print(f"Q: {question}")
    answer = service.rag_summarize(question)
    print("A:", answer)

