"""
用于测试“导入并调用 utils.test.run”时的行为。

你可以通过两种方式运行本文件：

1. 在项目根目录下：

    python -m zhisaotong_agent.utils.test_import_runner

2. 在包根目录 zhisaotong_agent/ 下：

    python -m utils.test_import_runner
"""

from . import test


def main() -> None:
    # 打印当前模块和被导入模块的 __name__，便于理解包结构
    print(f"[runner] __name__ = {__name__!r}")
    print(f"[runner] imported test.__name__ = {test.__name__!r}")

    # 调用 test.run()，其内部会使用 logger 输出一条带有 __name__ 和 logger.name 的日志
    test.run()


if __name__ == "__main__":
    main()

