## 智扫通Agent 项目说明

本项目是一个面向消费者（toC）的**扫地机器人智能客服系统**，目标是在用户购买前、使用中与售后阶段，提供贯穿全生命周期的问答与决策支持服务。系统基于 RAG（Retrieval-Augmented Generation）技术，从扫地机器人领域知识库中检索准确信息，并生成自然语言回答或建议。

### 一、核心功能

- **智能问答服务**
  - 处理购买前咨询：如功能差异、价格、型号对比、适用户型等。
  - 解决使用与售后问题：如操作指引、故障排查、维护保养建议等。
  - 依托 RAG 技术，从知识库中检索高置信度文档片段，结合大模型生成可靠回复。

- **使用报告与优化建议**
  - 针对已购用户，自动分析扫地机器人的使用数据（如清洁频率、耗材状态、错误日志等）。
  - 生成个性化使用报告，总结使用情况并提出优化建议（如清扫计划、禁区设置、耗材更换提醒等）。
  - 支持用户主动查询报告或系统定期推送，帮助用户最大化产品价值。

### 二、安装与启动

**环境要求**：Python **3.10+**。建议使用虚拟环境，**先激活**后再安装与运行。

**安装**（在子项目根目录 `zhisaotong_agent/` 下执行）：

```bash
pip install -e .
```

**API Key**：在 `zhisaotong_agent/` 目录放置 `.env`，或预先导出环境变量，配置 **`DASHSCOPE_API_KEY`**（亦可使用 **`API_KEY`**，应用会兼容读取）。

**启动 Streamlit**（任选其一）：

- **脚本**（会先执行 `pip install -e .`，再启动应用；默认监听 `0.0.0.0:8501`，便于远程或容器内访问）  
  - Linux / macOS：`./run_app.sh`  
  - Windows：`run_app.bat`
- **命令行**：

```bash
python3 -m streamlit run src/zhisaotong_agent/app.py --server.address 0.0.0.0 --server.port 8501
```

### 三、代码目录结构

> 外层目录名 `zhisaotong_agent/` 为仓库中的子项目根；在其中有 `pyproject.toml`，可执行 `pip install -e .` 做可编辑安装。Python 包源码在 `src/zhisaotong_agent/`（import 名 `zhisaotong_agent`）。本说明文件 `PROJECT_OVERVIEW.md` 位于子项目根目录，与 `pyproject.toml` 同级。

```text
zhisaotong_agent/                 # 子项目根（含构建配置）
  ├─ pyproject.toml               # 包元数据、依赖、setuptools 配置
  ├─ MANIFEST.in                  # 发行包时包含的非 .py 资源
  ├─ PROJECT_OVERVIEW.md          # 项目说明（本文件）
  ├─ run_app.sh                   # Linux/macOS 启动脚本
  ├─ run_app.bat                  # Windows 启动脚本
  └─ src/
       └─ zhisaotong_agent/       # 可安装包根目录（import 名同此）
            ├─ agent/             # 智能体相关逻辑
            │  ├─ tools/          # 智能体可调用的工具集合
            │  │  ├─ agent_tools.py
            │  │  └─ middleware.py
            │  └─ react_agent.py
            ├─ config/            # 配置文件（模型、RAG、数据源等）
            ├─ data/              # 原始数据与中间数据
            ├─ model/
            │  └─ factory.py
            ├─ prompts/           # 提示词模板
            ├─ rag/
            │  ├─ rag_service.py
            │  └─ vector_store.py
            ├─ utils/             # 通用工具（config_handler、path_tool 等）
            └─ app.py             # Streamlit 入口
```

### 四、模块间逻辑架构（Mermaid）

> 下图以 Mermaid 描述主要模块间的调用与依赖关系，仅为架构设计示意，不包含任何实现细节。

```mermaid
flowchart LR
    %% 前端入口
    UI[前端页面 / Streamlit 应用] -->|用户问题/指令| APP[app.py 应用入口]

    %% 智能体与工具
    APP --> RA[ReactAgent\nreact_agent.py]
    RA -->|调用工具| AT[Agent Tools\nagent_tools.py]
    RA -->|监控/埋点| MW[middleware.py]

    %% RAG 服务
    AT -->|RAG 查询| RAG[rageSummarizeService\nrag_service.py]
    RAG -->|检索文档| VS[VectorStoreService\nvector_store.py]

    %% 模型层
    RAG -->|调用大模型| CM[Chat Model]
    RAG -->|生成向量| EM[Embedding Model]
    CM & EM --> MF[Model Factory\nfactory.py]

    %% 知识库与数据
    VS -->|加载/构建索引| KB[(RAG 知识库\n文本/向量存储)]
    KB --- DATA[data 目录\n原始文档]

    %% 配置与工具
    APP --> CFG[配置管理\nconfig_handler.py]
    CFG --> CONF[配置文件\nconfig 目录]

    APP --> ULOG[日志工具\nlogger_handler.py]
    AT  --> FH[file_handler.py]
    AT  --> PL[prompt_loader.py]

    %% 用户数据分析与报告
    APP -->|上报使用数据| UD[(用户使用数据)]
    UD --> AT
    AT -->|生成报告/建议| UI
```

### 五、后续开发建议

- **第一阶段**：搭建基础工程（虚拟环境、依赖管理、基础配置与日志），补充 `README.md` 与本说明文件的关联。
- **第二阶段**：实现向量库构建与 RAG 服务（`rag_service.py`、`vector_store.py`），并导入扫地机器人知识文档。
- **第三阶段**：实现智能体逻辑（`ReactAgent` 及工具函数），打通从前端到 RAG 的完整闭环。
- **第四阶段**：接入真实用户数据与报表生成能力，迭代优化对话体验与报告模板。
