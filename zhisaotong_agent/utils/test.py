"""
用于配合 logger_handler 测试导入与运行行为的示例模块。
"""

from .logger_handler import get_logger


def run() -> None:
    """
    作为被其他模块导入并调用的入口函数。
    """
    logger = get_logger(__name__)
    logger.info(f"test.run() 被调用，当前模块 __name__ = {__name__!r}，logger.name = {logger.name!r}")


if __name__ == "__main__":
    # 直接以脚本方式运行：python -m zhisaotong_agent.utils.test
    run()
