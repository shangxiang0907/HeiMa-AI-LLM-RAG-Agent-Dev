#!/bin/bash
# 启动Streamlit问答应用，监听0.0.0.0以便在sealos devbox中访问

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 使用 python3 -m，避免 streamlit 可执行文件未加入 PATH（如 pip --user 安装）
python3 -m streamlit run app_qa.py --server.address 0.0.0.0 --server.port 8501
