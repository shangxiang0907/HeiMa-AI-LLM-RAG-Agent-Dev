#!/bin/bash
# 在已激活环境中执行 pip install -e . 后启动 Streamlit（不创建 venv）。监听 0.0.0.0 便于 devbox 访问。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 使用当前环境（如已激活的 venv）里的 Python；python3 -m 避免可执行文件未在 PATH
PY=python3
"$PY" -m pip install -e .
"$PY" -m streamlit run src/zhisaotong_agent/app.py --server.address 0.0.0.0 --server.port 8501
