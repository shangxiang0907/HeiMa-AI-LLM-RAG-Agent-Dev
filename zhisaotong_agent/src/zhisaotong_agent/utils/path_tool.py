from pathlib import Path


def get_project_root() -> str:
    """
    获取项目根目录的绝对路径。

    约定：本文件位于包内 `zhisaotong_agent/utils/path_tool.py`，
    「项目根」指包安装目录（含 config、data 等），即本文件向上两级目录。
    """
    return str(Path(__file__).resolve().parent.parent)


def get_abs_path(relative_path: str) -> str:
    """
    将“项目内相对路径”转换为基于项目根目录的绝对路径。

    要求：
    - 只接受相对路径；
    - 如果传入的是绝对路径，则抛出 ValueError，避免误用掩盖问题。
    """
    p = Path(relative_path)
    if p.is_absolute():
        raise ValueError(f"get_abs_path 仅接受相对路径，实际收到绝对路径: {relative_path!r}")

    root = Path(get_project_root())
    return str((root / p).resolve())


if __name__ == "__main__":
    # 简单自测代码
    print("项目根目录：", get_project_root())

    # 1. 测试相对路径转换
    rel_examples = [
        "PROJECT_OVERVIEW.md",          # 根目录下已存在的文件（根据当前项目结构）
        "data",                         # 根目录下已有的 data 目录
        "utils/path_tool.py",           # 本文件自身
        "./data/../PROJECT_OVERVIEW.md" # 含有 . 和 .. 的相对路径
    ]
    for rel in rel_examples:
        # 这里的 !r 表示使用 repr() 的形式显示变量值（带引号、便于调试区分空字符串等）
        print(f"relative: {rel!r} -> abs: {get_abs_path(rel)!r}")

    # 2. 测试绝对路径误用场景（应抛出异常）
    current_file_abs = str(Path(__file__).resolve())
    try:
        print("absolute input ->", get_abs_path(current_file_abs))
    except ValueError as e:
        print("absolute input error:", e)

