"""
智能体业务工具：RAG 查询、用户信息、外部数据、报告上下文等。

设计要点：
- 与 config/agent.yml 中 tools 的 name 对应关系由调用方（如 react_agent）按 name 绑定；
- 配置通过 config_handler 加载，不使用全局 dict；
- 外部数据使用 CSV 标准库解析、懒加载 + 线程安全缓存；
- 所有工具对外入参、返回值与描述保持不变，便于与 ReAct 等框架对接。
"""

from __future__ import annotations

import csv
import random
import threading
from typing import Any, Dict

from langchain_core.tools import tool

from zhisaotong_agent.rag.rag_service import RagSummarizeService
from zhisaotong_agent.utils.api_key import init_dashscope_api_key
from zhisaotong_agent.utils.config_handler import load_agent_config
from zhisaotong_agent.utils.logger_handler import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 服务与配置（与项目其他模块一致，使用统一配置入口）
# ---------------------------------------------------------------------------

_rag_service: RagSummarizeService | None = None


def _get_rag_service() -> RagSummarizeService:
    """懒加载 RAG 服务，避免在导入时强依赖向量库与模型初始化。"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


# 模拟数据：实际场景可由会话/登录态提供
USER_IDS = [
    "1001", "1002", "1003", "1004", "1005",
    "1006", "1007", "1008", "1009", "1010",
]
MONTH_ARR = [
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
]

# ---------------------------------------------------------------------------
# 外部数据：懒加载 + 线程安全缓存，使用标准 csv 解析与配置路径
# 数据含义简要说明：
# - 每一行是“某个用户在某个月的扫地机器人使用概况”；
# - 列顺序为：
#   1) 用户ID：唯一标识一个家庭/设备（如 1001～1010）；
#   2) 特征：用户及住家画像（户型面积、家庭结构、地面材质等）；
#   3) 清洁效率：覆盖率、日均清扫面积、漏扫情况、回充成功率等表现指标；
#   4) 耗材：主刷/边刷/滤网寿命、集尘袋、水箱等维护与消耗状态；
#   5) 对比：与同类用户的对比结论（如优于/低于多少同类用户）；
#   6) 时间：年-月字符串（如 2025-01），用于做纵向趋势分析。
# 最终在内存中的结构为：
# { user_id: { month: { "特征": ..., "效率": ..., "耗材": ..., "对比": ... } } }
# ---------------------------------------------------------------------------

_external_data_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
_external_data_lock = threading.Lock()


def _load_external_data_from_file(file_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    从 CSV 文件加载外部使用记录，返回结构：
    { user_id: { month: { "特征": xxx, "效率": xxx, "耗材": xxx, "对比": xxx } } }
    约定：首行为表头，列顺序为 user_id, 特征, 效率, 耗材, 对比, 月份。
    """
    result: Dict[str, Dict[str, Dict[str, str]]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) < 6:
                logger.warning("外部数据行列数不足，已跳过。raw_row=%r", row)
                continue
            user_id = row[0].strip().strip('"')
            feature = row[1].strip().strip('"')
            efficiency = row[2].strip().strip('"')
            consumables = row[3].strip().strip('"')
            comparison = row[4].strip().strip('"')
            time_key = row[5].strip().strip('"')
            if user_id not in result:
                result[user_id] = {}
            result[user_id][time_key] = {
                "特征": feature,
                "效率": efficiency,
                "耗材": consumables,
                "对比": comparison,
            }
    return result


def _get_external_data() -> Dict[str, Dict[str, Dict[str, str]]]:
    """获取外部数据（懒加载、线程安全、仅加载一次）。"""
    global _external_data_cache
    # 第一层快速路径：在无锁情况下直接返回，避免每次读取都进入加锁开销
    if _external_data_cache:
        return _external_data_cache
    with _external_data_lock:
        # 第二层检查：防止并发场景下多个线程同时通过第一层判断而重复加载（double-checked locking）
        if _external_data_cache:
            return _external_data_cache
        agent_config = load_agent_config()
        file_path = agent_config.get_external_data_abs_path()
        try:
            loaded = _load_external_data_from_file(file_path)
            # 这里使用 clear() + update() 而不是直接重新赋值：
            # - 当前语义下缓存首次加载时本身就是空，但保留 clear() 便于未来支持“强制刷新”场景；
            # - 保证始终复用同一个 dict 实例，避免其他模块持有的引用因替换对象而失效。
            _external_data_cache.clear()
            _external_data_cache.update(loaded)
            logger.info("外部数据已加载，路径=%s，用户数=%d", file_path, len(_external_data_cache))
        except FileNotFoundError:
            logger.warning("外部数据文件不存在，路径=%s，将返回空数据", file_path)
        except Exception as e:
            logger.error("加载外部数据失败，路径=%s，error=%s", file_path, e, exc_info=True)
            raise
    return _external_data_cache


def _format_record_as_string(record: Dict[str, str]) -> str:
    """将单条使用记录格式化为纯字符串，便于工具返回。"""
    return "\n".join(f"{k}: {v}" for k, v in record.items())


# ---------------------------------------------------------------------------
# 工具定义（对外接口与描述保持不变）
# ---------------------------------------------------------------------------


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return _get_rag_service().rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(USER_IDS)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(MONTH_ARR)


@tool(
    description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串"
)
def fetch_external_data(user_id: str, month: str) -> str:
    data = _get_external_data()
    try:
        record = data[user_id][month]
        return _format_record_as_string(record)
    except KeyError:
        logger.warning(
            "[fetch_external_data] 未能检索到用户：%s 在 %s 的使用记录数据",
            user_id,
            month,
        )
        return ""


@tool(
    description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息"
)
def fill_context_for_report() -> str:
    return "fill_context_for_report已调用"


# ---------------------------------------------------------------------------
# 对外集合（供 react_agent 等按 name 绑定，如 rag_search -> rag_summarize）
# ---------------------------------------------------------------------------

TOOLS_LIST = [
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
]

# 配置中 name 与工具函数的映射，便于按 agent.yml 的 tools.name 绑定
TOOLS_BY_NAME: Dict[str, Any] = {
    "rag_summarize": rag_summarize,
    "get_weather": get_weather,
    "get_user_location": get_user_location,
    "get_user_id": get_user_id,
    "get_current_month": get_current_month,
    "fetch_external_data": fetch_external_data,
    "fill_context_for_report": fill_context_for_report,
}

# 若 agent.yml 中工具名与函数名不一致，可在此做别名（如 rag_search -> rag_summarize）
TOOLS_BY_NAME.setdefault("rag_search", rag_summarize)
TOOLS_BY_NAME.setdefault("usage_report", fetch_external_data)

__all__ = [
    "rag_summarize",
    "get_weather",
    "get_user_location",
    "get_user_id",
    "get_current_month",
    "fetch_external_data",
    "fill_context_for_report",
    "TOOLS_LIST",
    "TOOLS_BY_NAME",
]

if __name__ == "__main__":
    """
    自测：在项目根目录下运行
        python -m zhisaotong_agent.agent.tools.agent_tools
    仅验证工具可调用与返回格式，不强制要求 DASHSCOPE_API_KEY 或外部数据文件存在。
    """
    # 先手动加载 API Key 到环境变量，供 RAG 等依赖使用
    init_dashscope_api_key()

    print("=== agent_tools 自测 ===\n")

    # 1. 无外部依赖的工具
    print("[1] get_weather('北京') ->", get_weather.invoke({"city": "北京"}))
    print("[2] get_user_location() ->", get_user_location.invoke({}))
    print("[3] get_user_id() ->", get_user_id.invoke({}))
    print("[4] get_current_month() ->", get_current_month.invoke({}))
    print("[5] fill_context_for_report() ->", fill_context_for_report.invoke({}))

    # 2. 外部数据（文件不存在时返回空字符串）
    out = fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})
    print("[6] fetch_external_data('1001', '2025-01') ->", repr(out)[:80] + ("..." if len(repr(out)) > 80 else ""))

    # 3. RAG 工具（需配置 DASHSCOPE_API_KEY 与知识库）
    print("\n[7] rag_summarize (可选，依赖 API 与向量库)")
    if not init_dashscope_api_key():
        print("    未配置 DASHSCOPE_API_KEY，已跳过")
    else:
        try:
            ans = rag_summarize.invoke({"query": "扫地机器人如何保养？"})
            print("    ->", ans[:120] + "..." if len(ans) > 120 else ans)
        except Exception as e:
            print("    失败:", e)

    print("\n=== 自测结束 ===")
