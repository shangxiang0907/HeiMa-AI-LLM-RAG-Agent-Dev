from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from file_history_store import get_history
import json


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


def debug_runnable(name: str, pretty: bool = False):
    def _convert_to_serializable(obj):
        """将对象转换为可 JSON 序列化的格式，递归处理嵌套结构"""
        # 处理 LangChain 消息对象（HumanMessage, AIMessage, SystemMessage 等）
        if hasattr(obj, 'content') and hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            if 'Message' in class_name:
                result = {
                    "type": class_name,
                    "content": obj.content
                }
                # 添加其他可能的属性
                if hasattr(obj, 'additional_kwargs') and obj.additional_kwargs:
                    result["additional_kwargs"] = _convert_to_serializable(obj.additional_kwargs)
                if hasattr(obj, 'response_metadata') and obj.response_metadata:
                    result["response_metadata"] = _convert_to_serializable(obj.response_metadata)
                return result
        
        # 处理 Document 对象
        if isinstance(obj, Document):
            return {
                "page_content": obj.page_content,
                "metadata": obj.metadata
            }
        
        # 处理列表
        if isinstance(obj, list):
            return [_convert_to_serializable(item) for item in obj]
        
        # 处理字典
        if isinstance(obj, dict):
            return {key: _convert_to_serializable(value) for key, value in obj.items()}
        
        # 其他类型，尝试直接返回（如果可序列化）
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # 如果无法序列化，返回字符串表示
            return str(obj)
    
    def _inner(x):
        print(f"\n[DEBUG][{name}]")
        
        if pretty:
            # 处理 Document 对象（单个）
            if isinstance(x, Document):
                doc_dict = {
                    "page_content": x.page_content,
                    "metadata": x.metadata
                }
                print(json.dumps(doc_dict, indent=2, ensure_ascii=False))
            # 处理 Document 列表
            elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], Document):
                docs_list = [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in x
                ]
                print(json.dumps(docs_list, indent=2, ensure_ascii=False))
            # 处理普通的 dict 或 list（可能包含消息对象）
            elif isinstance(x, (dict, list)):
                try:
                    serializable_x = _convert_to_serializable(x)
                    print(json.dumps(serializable_x, indent=2, ensure_ascii=False))
                except Exception as e:
                    # 如果序列化失败，回退到直接打印
                    print(f"序列化失败，使用直接打印: {e}")
                    print(x)
            else:
                print(x)
        else:
            print(x)
        
        print(f"[DEBUG][{name}] 结束\n")
        return x

    return RunnableLambda(_inner)


def extract_input_field(x):
    """
    从 RunnableWithMessageHistory 传入的字典中提取真正的用户查询字符串。
    - 如果 x 是形如 {"input": "...", "history": [...]} 的字典，则返回 x["input"]
    - 否则原样返回（兼容直接传字符串的情况）
    """
    if isinstance(x, dict) and "input" in x:
        return x["input"]
    return x


class RagService(object):
    def __init__(self, storage_path: str = None):
        """
        初始化 RagService。

        参数:
            storage_path: 会话历史记录的存储路径，默认为 config.chat_history_path
        """
        if storage_path is None:
            storage_path = config.chat_history_path
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录,如下:"),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问:{input}")
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.storage_path = storage_path

    def _get_chain(self):
        """获取最终的执行链（不带历史记录）

        整体上，这个链的输入是「用户问题字符串」，输出是「LLM 的回答字符串」。
        为了让数据流更清晰，这里在构造链的时候对中间字段做了一些约定：
        - user_input: 用户的原始提问（str）
        - retrieved_context: 根据用户提问从向量库检索出的参考资料文本（str）
        这两个字段最终会作为 prompt 模板中的 {input} 和 {context}。
        """
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            # 将检索到的多个文档片段拼接成一个字符串，作为 LLM 的「参考资料上下文」
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"
            return formatted_str

        # 1. inputs_mapping 负责把「同一个用户输入」拆成三个键：
        #    - "input": 提取用户问题字符串，用于 {input}
        #    - "context": 先走 retriever 再走 format_document，得到检索到的参考资料字符串，用于 {context}
        #    - "history": 保留历史消息，用于 {history}（由 RunnableWithMessageHistory 自动注入）
        def extract_history(x):
            """从 RunnableWithMessageHistory 传入的字典中提取 history"""
            if isinstance(x, dict) and "history" in x:
                return x["history"]
            return []
        
        inputs_mapping = {
            # 提取用户问题字符串，用于 prompt 模板的 {input}
            "input": RunnableLambda(extract_input_field) \
                    | debug_runnable("inputs_mapping.input", pretty=True),
            # retriever | format_document: 先用向量检索器查相关文档，再格式化成一段文本
            # 在 retriever 前后、format_document 后分别加上 debug_runnable
            "context": debug_runnable("inputs_mapping.context.before_retriever", pretty=True) \
                       | RunnableLambda(extract_input_field) \
                       | debug_runnable("inputs_mapping.context.after_extract_input") \
                       | retriever \
                       | debug_runnable("inputs_mapping.context.after_retriever", pretty=True) \
                       | format_document \
                       | debug_runnable("inputs_mapping.context.after_format_document"),
            # 保留 history 字段，传递给 prompt 模板
            "history": RunnableLambda(extract_history) \
                      | debug_runnable("inputs_mapping.history", pretty=True),
        }

        # 2. 完整链：
        #    用户问题(str)
        #      -> inputs_mapping 生成 {"input": 用户问题, "context": 检索出的参考资料文本}
        #      -> prompt_template 组装成 ChatPrompt
        #      -> print_prompt 打印完整提示词（调试用）
        #      -> chat_model 调用大模型
        #      -> StrOutputParser() 把大模型输出转成纯字符串
        rag_chain = (
            debug_runnable("chain.input", pretty=True)               # 整个链最开始的原始输入
            | inputs_mapping
            | debug_runnable("chain.after_inputs_mapping", pretty=True)  # dict: {"input": ..., "context": ...}
            | self.prompt_template
            | debug_runnable("chain.after_prompt_template")  # ChatPrompt
            | print_prompt                                  # 已有的 prompt 打印
            | debug_runnable("chain.after_print_prompt")    # 打印后的 prompt（同上）
            | self.chat_model
            | debug_runnable("chain.after_chat_model")      # LLM 输出（通常是 Message/ChatResult）
            | StrOutputParser()
            | debug_runnable("chain.after_output_parser")   # 最终字符串输出
        )
        return rag_chain

    def get_conversation_chain(self):
        """获取带历史记录的对话链"""
        base_chain = self._get_chain()
        
        # 创建 get_history 函数，使用当前实例的 storage_path
        def get_history_func(session_id: str):
            return get_history(session_id, self.storage_path)
        
        # 使用 RunnableWithMessageHistory 创建带历史记录的链
        conversation_chain = RunnableWithMessageHistory(
            base_chain,  # 被附加历史消息的 Runnable，通常是 chain
            get_history_func,  # 获取指定会话ID的历史会话的函数
            input_messages_key="input",  # 声明用户输入消息在模板中的占位符
            history_messages_key="history",  # 声明历史消息在模板中的占位符
        )
        return conversation_chain


if __name__ == "__main__":
    """
    简单的测试代码
    测试 RagService 的初始化、链获取和查询功能（带历史记录）
    无论从哪个路径执行本文件，都会先将当前工作目录切换为脚本所在目录。
    """
    import os
    from dotenv import load_dotenv

    # 将当前工作目录切换为脚本所在目录，保证相对路径（如 ./chroma_db、./md5.text）始终指向项目目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"[info] 已将当前工作目录切换为脚本所在目录: {script_dir}")

    # 加载环境变量
    load_dotenv()
    
    # 获取 API key
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("警告: 未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，测试可能失败")
        print("请先在 .env 或系统环境中配置 API key")
    else:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    
    try:
        print("=" * 50)
        print("开始测试 RagService（带历史记录功能）")
        print("=" * 50)
        
        # 1. 初始化 RagService
        print("\n[1] 初始化 RagService...")
        rag_service = RagService()
        print("✓ RagService 初始化成功")
        print(f"  - Embedding model: {config.embedding_model_name}")
        print(f"  - Chat model: {config.chat_model_name}")
        print(f"  - 会话历史存储路径: {rag_service.storage_path}")
        
        # 2. 获取带历史记录的对话链
        print("\n[2] 获取带历史记录的对话链...")
        conversation_chain = rag_service.get_conversation_chain()
        print("✓ 对话链获取成功")
        
        # 3. 配置会话ID
        session_id = "test_user_001"
        session_config = {"configurable": {"session_id": session_id}}
        print(f"\n[3] 配置会话ID: {session_id}")
        print(f"  session_config = {session_config}")
        
        # 4. 测试多轮对话功能
        print("\n[4] 测试多轮对话功能...")
        print("=" * 50)
        
        # 第一轮对话
        print("\n【第一轮对话】")
        test_query_1 = "我身高180cm，140kg，我应该穿什么尺码的衣服？"
        print(f"  用户提问: {test_query_1}")
        print("-" * 50)
        try:
            result_1 = conversation_chain.invoke(
                {"input": test_query_1},
                config=session_config
            )
            print(f"✓ 第一轮对话成功")
            print(f"  回答: {result_1}")
        except Exception as e:
            print(f"  注意: 查询时出现异常（可能是向量库为空或 API 配置问题）: {str(e)}")
            print("  这是正常的（如果还没有上传文档或 API key 未配置）")
        print("-" * 50)
        
        # 第二轮对话（测试历史记忆）
        print("\n【第二轮对话】")
        test_query_2 = "我刚才问的身高和体重是多少？"
        print(f"  用户提问: {test_query_2}")
        print("-" * 50)
        try:
            result_2 = conversation_chain.invoke(
                {"input": test_query_2},
                config=session_config
            )
            print(f"✓ 第二轮对话成功")
            print(f"  回答: {result_2}")
            print("  说明: 模型应该能够记住第一轮对话中的身高和体重信息")
        except Exception as e:
            print(f"  注意: 查询时出现异常: {str(e)}")
        print("-" * 50)
        
        # 第三轮对话（继续测试历史记忆）
        print("\n【第三轮对话】")
        test_query_3 = "那我适合什么颜色的衣服？"
        print(f"  用户提问: {test_query_3}")
        print("-" * 50)
        try:
            result_3 = conversation_chain.invoke(
                {"input": test_query_3},
                config=session_config
            )
            print(f"✓ 第三轮对话成功")
            print(f"  回答: {result_3}")
            print("  说明: 模型应该能够结合之前的身高体重信息给出建议")
        except Exception as e:
            print(f"  注意: 查询时出现异常: {str(e)}")
        print("-" * 50)
        
        print("\n" + "=" * 50)
        print("测试完成！")
        print("=" * 50)
        print("\n说明:")
        print("- 会话历史记录已保存到文件中，程序重启后仍然保留")
        print(f"- 会话文件路径: {os.path.join(rag_service.storage_path, session_id)}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
