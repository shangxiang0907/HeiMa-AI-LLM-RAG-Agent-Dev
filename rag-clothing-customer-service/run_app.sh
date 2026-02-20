#!/bin/bash
# 启动Streamlit应用，监听0.0.0.0以便在sealos devbox中访问

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 启动Streamlit应用，监听0.0.0.0
streamlit run app_file_uploader.py --server.address 0.0.0.0 --server.port 8501
