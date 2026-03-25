@echo off
REM 已激活 venv 后运行：pip install -e . 再启动 Streamlit。与 run_app.sh 行为一致。

cd /d "%~dp0"
set PY=python
"%PY%" -m pip install -e .
"%PY%" -m streamlit run src\zhisaotong_agent\app.py --server.address 0.0.0.0 --server.port 8501
