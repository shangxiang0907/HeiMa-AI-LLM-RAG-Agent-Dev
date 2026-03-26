## 项目说明 Project Overview

本仓库主要用于记录和实现 B 站黑马程序员课程  
**《黑马程序员大模型 RAG 与 Agent 智能体项目实战教程，基于主流的 LangChain 技术从大模型提示词到实战项目》**  
对应视频链接（bilibili）：[`https://www.bilibili.com/video/BV1yjz5BLEoY`](https://www.bilibili.com/video/BV1yjz5BLEoY)  
所有代码与示例均为本人学习过程中的实践与笔记整理，不是官方代码仓库。

This repository is a personal learning project based on the Bilibili course by HeiMa:  
**“Large Language Model RAG & Agent Practical Projects with LangChain – From Prompt Engineering to Real-World Applications”**  
Video link (Bilibili): [`https://www.bilibili.com/video/BV1yjz5BLEoY`](https://www.bilibili.com/video/BV1yjz5BLEoY).  
All source code and examples are my own notes and practice code, not the official course repo.

## 目录结构 Directory Structure

## `AI_LLM_RAG_Agent_Dev/`：课程相关代码与练习（个人笔记）
  - **`01_TestApiKey.py`**：测试大模型 / 平台 API Key 是否可用（Test script to verify LLM / platform API keys.）
  - **`02_OpenAI_Library_Basic_Usage.py`**：OpenAI / 通义等大模型基础调用示例（Basic usage examples for LLM SDKs.）
  - **`03_OpenAI_Library_Stream_Output.py`**：流式输出（streaming）示例（Streaming response examples.）
  - **`04_OpenAI_Library_With_History.py`**：带历史记忆的对话示例（Chat with conversation history.）
  - **`05_Financial_Text_Classification.py`**：金融文本分类案例（Financial text classification demo.）
  - **`06_JSON_Usage_Demo.py`**：使用大模型生成结构化 JSON 数据示例（Structured JSON output from LLMs.）
  - **`07_Information_Extraction_FewShot.py`**：少样本信息抽取（Few-shot information extraction.）
  - **`08_Lottery_Information_Extraction.py`**：彩票信息抽取实战案例（Lottery information extraction demo.）
  - **`09_Text_Matching_FewShot.py`**：文本匹配 / 相似度少样本示例（Few-shot text matching examples.）
  - **`10_Cosine_Similarity_Algorithm.py`**：余弦相似度算法与向量检索基础（Cosine similarity and basic vector search.）
  - **`11_LangChain_Tongyi_Basic_Usage.py`**：LangChain + 通义千问 基础用法（Basic LangChain usage with Tongyi Qianwen.）
  - **`12_LangChain_Tongyi_Stream_Output.py`**：LangChain 流式输出示例（Streaming with LangChain and Tongyi.）
  - **`13_LangChain_Tongyi_Chat_Model.py`**：LangChain ChatModel 配置与调用（Using LangChain chat models.）
  - **`14_LangChain_Message_Shorthand.py`**：LangChain 消息对象与简写语法（LangChain message classes and shorthand syntax.）
  - **`15_LangChain_Embeddings_DashScope.py`**：向量化与 DashScope Embeddings 示例，为后续 RAG 做准备（Embeddings with DashScope for RAG.）
  - **`16_LangChain_PromptTemplate_Tongyi.py`**：通用提示词模板（PromptTemplate）在通义模型上的用法示例（PromptTemplate examples with Tongyi.）
  - **`17_LangChain_FewShot_PromptTemplate.py`**：FewShot 提示词模板示例，展示如何通过多个示例构造少样本提示词（Few-shot prompt template examples using FewShotPromptTemplate.）
  - **`18_LangChain_PromptTemplate_Format_vs_Invoke.py`**：PromptTemplate 中 format 与 invoke 方法的对比示例（Comparison between format and invoke methods in PromptTemplate.）
  - **`19_LangChain_ChatPromptTemplate.py`**：聊天提示词模板示例，演示如何使用 ChatPromptTemplate 和 MessagesPlaceholder 动态注入历史会话信息（ChatPromptTemplate examples with dynamic history injection using MessagesPlaceholder.）
  - **`20_LangChain_ChatPromptTemplate_Chain_Call.py`**：链式调用示例，演示使用「|」运算符将提示词模板和模型链接成 chain（Chain call examples using the pipe operator to connect prompt templates and models.）
  - **`21_LangChain_Chain_Operator_Overload.py`**：运算符重写示例，说明 LangChain 中「|」链式调用的底层原理（Operator overloading examples explaining how the pipe operator works in LangChain chains.）
  - **`22_LangChain_Runnable_Base_Class.py`**：Runnable 抽象基类示例，演示 LangChain 核心组件的继承关系和链式调用机制（Runnable base class examples demonstrating the inheritance structure and chain mechanism.）
  - **`23_LangChain_StrOutputParser.py`**：字符串输出解析器示例，演示如何将 AIMessage 转换为字符串以解决链式调用中的类型不匹配问题（StrOutputParser examples for converting AIMessage to string in chains.）
  - **`24_LangChain_JsonOutputParser.py`**：JSON 输出解析器示例，演示如何将 AIMessage 转换为字典（JSON 格式）用于多模型链式调用（JsonOutputParser examples for converting AIMessage to JSON/dict in multi-model chains.）
  - **`25_LangChain_RunnableLambda.py`**：RunnableLambda 示例，演示如何将自定义函数加入链中实现灵活的数据转换（RunnableLambda examples for adding custom functions to chains.）
  - **`26_LangChain_Temporary_Session_Memory.py`**：临时会话记忆示例，演示如何使用 RunnableWithMessageHistory 和 InMemoryChatMessageHistory 实现带历史记录的对话（Temporary session memory examples using RunnableWithMessageHistory and InMemoryChatMessageHistory.）
  - **`27_LangChain_Permanent_Session_Memory.py`**：持久化会话记忆示例，对比临时记忆，展示如何跨会话保存历史对话（Persistent session memory examples showing how to persist chat history across sessions.）
  - **`28_LangChain_CSVLoader.py`**：CSVLoader 示例，演示如何从 CSV 文件加载结构化数据用于 RAG（CSVLoader examples for loading structured data from CSV into RAG pipelines.）
  - **`29_LangChain_JSONLoader.py`**：JSONLoader 示例，演示如何从 JSON 文件中加载文档与元数据（JSONLoader examples for loading documents and metadata from JSON files.）
  - **`30_LangChain_TextLoader.py`**：TextLoader 示例，演示如何从纯文本文件中加载长文档并做拆分（TextLoader examples for loading and splitting long plain-text documents.）
  - **`31_LangChain_PyPDFLoader.py`**：PyPDFLoader 示例，演示如何加载 PDF 文档、分段并为后续向量化做准备（PyPDFLoader examples for loading and chunking PDF documents for embeddings.）
  - **`32_LangChain_VectorStore.py`**：向量存储（Vector Store）综合示例，包含 InMemoryVectorStore、Chroma 等的增删查与 RAG 基础流程（Vector store examples with InMemoryVectorStore, Chroma, and basic RAG indexing/querying.）
  - **`33_LangChain_RAG_Complete_Workflow.py`**：RAG（检索增强生成）完整流程示例，从向量库构建到提示词注入与回答生成的端到端演示（End‑to‑end RAG workflow demo from vector store to prompt construction and answer generation.）
  - **`34_LangChain_RAG_Retriever_Chain_InMemory.py`**：基于 InMemory 向量存储的 RAG 检索链示例，演示如何使用 retriever.as_retriever() 将向量检索步骤直接加入 LangChain 链中（RAG retriever‑chain demo with InMemoryVectorStore using retriever.as_retriever() inside a LangChain Runnable graph.）
  - **`35_LangChain_Agent_First_Experience.py`**：LangChain Agent 智能体初体验示例，演示如何定义工具、创建 Agent 并调用，重点展示 Agent 的输入/输出结构（First experience with LangChain Agent, demonstrating tool definition, agent creation, and invocation, focusing on input/output structure.）
  - **`36_LangChain_Agent_Stream_Output.py`**：Agent 智能体流式输出示例，演示如何使用 `agent.stream(..., stream_mode="values")` 持续接收增量消息，实时观察 Agent 的思考过程和工具调用（Agent streaming output examples using `agent.stream()` to receive incremental messages and observe the agent's thinking process and tool calls in real-time.）
  - **`37_LangChain_Agent_ReAct_Framework.py`**：ReAct 思考-行动-观察框架示例，演示如何在 system_prompt 中约束 Agent 按照「思考 → 行动 → 观察 → 再思考」的流程解决问题，并通过流式输出观察完整的 ReAct 过程（ReAct framework examples showing how to constrain agents to follow the "Thought → Action → Observation → Re-thought" flow via system_prompt and observe the complete ReAct process through streaming.）
  - **`38_LangChain_Agent_Middleware.py`**：LangChain Agent 中间件示例，演示节点式钩子（before_agent, after_agent, before_model, after_model）和包装式钩子（wrap_model_call, wrap_tool_call）的使用，包含日志记录、重试逻辑、工具监控等完整示例（LangChain Agent middleware examples demonstrating node-style hooks and wrapper-style hooks for logging, retry logic, tool monitoring, etc.）
  - **`stu.csv`**：用于 CSVLoader 示例的简单学生信息数据集（A small student info CSV dataset used by the CSVLoader examples.）

> 后续若继续跟随课程实现更复杂的 RAG 检索增强问答、Agent 智能体、多工具编排等内容，会在该目录下持续补充脚本与说明。  
> As I progress through the course (more advanced RAG pipelines, Agents, tool orchestration, etc.), more scripts and notes will be added under this directory.

## `rag-clothing-customer-service/`：RAG 项目 - 服装商品智能客服（RAG Project - Intelligent Customer Service for Clothing E-commerce）
  - 基于 RAG（检索增强生成）技术的服装电商智能客服系统知识库，提供尺码推荐、洗涤养护、颜色选择等问答能力  
    A RAG-based knowledge base and QA system for clothing e-commerce, including size recommendations, washing care, and color selection.
  - **`rag.py`**：核心 `RagService` 实现，封装向量检索、提示词模板、多轮对话与会话历史管理逻辑  
    Core `RagService` implementation that wires together the vector retriever, prompt template, multi-turn dialog, and history management.
  - **`knowledge_base.py`**：知识库构建服务，负责文件内容分割、向量化、写入 Chroma 向量库以及 MD5 去重  
    Knowledge base service for splitting documents, generating embeddings, writing to the Chroma vector store, and avoiding duplicates via MD5.
  - **`vector_stores.py`**：向量存储封装，基于 `Chroma` 创建检索器（`as_retriever`）供 RAG 链使用  
    Vector store wrapper based on `Chroma`, exposing a retriever (`as_retriever`) for use in the RAG chain.
  - **`file_history_store.py`**：基于文件的会话历史持久化，实现 `FileChatMessageHistory`，支持按 `session_id` 长期保存对话记录  
    File-based chat history persistence (`FileChatMessageHistory`) that stores conversations per `session_id` on disk.
  - **`config_data.py`**：项目配置，包括向量库路径（如 `./chroma_db`）、会话历史目录、分词参数与模型名称等  
    Central configuration for paths (e.g. `./chroma_db`, `./chat_history`), splitter parameters, and model names.
  - **`app_qa.py`**：基于 Streamlit 的 RAG 问答页面，支持多会话管理、流式回答展示以及从文件中恢复历史对话  
    Streamlit-based RAG QA UI with multi-session management, streaming answers, and restoration of chat history from files.
  - **`app_file_uploader.py`**：知识库文件上传与入库页面，用于将本地文档切分后写入 Chroma 向量库  
    Streamlit app for uploading documents, chunking them, and inserting them into the Chroma vector store.
  - **`run_qa.sh`**：一键启动问答页面（`app_qa.py`），在 Devbox 中监听 `0.0.0.0:8501`  
    Helper script to start the QA Streamlit app (`app_qa.py`) on `0.0.0.0:8501` inside Devbox.
  - **`run_app.sh`**：一键启动知识库上传页面（`app_file_uploader.py`），同样监听 `0.0.0.0:8501`  
    Helper script to start the knowledge-base uploader app (`app_file_uploader.py`) on `0.0.0.0:8501`.
  - **`data/`**：原始知识库文档存放目录（如尺码推荐、洗涤养护说明等）  
    Directory for original knowledge base documents (size guides, washing instructions, etc.).
  - **`chroma_db/`**：Chroma 向量库持久化目录，用于存放已向量化后的文档数据  
    Persisted Chroma vector store files.
  - **`chat_history/`**：会话历史文件目录，每个用户会话对应一个以 `session_id` 命名的 JSON 文件  
    Directory containing per-session JSON chat history files identified by `session_id`.
    <img width="1509" height="1383" alt="image" src="https://github.com/user-attachments/assets/7aa885bc-5c6a-4f73-80e4-ecf1ef194491" />
    <img width="2136" height="1371" alt="image" src="https://github.com/user-attachments/assets/03ba3566-d21c-430e-8d2a-4de34476a8be" />

## 智扫通Agent（zhisaotong_agent）

`zhisaotong_agent/` 是一个面向消费者（toC）的**扫地机器人智能客服系统**，基于 RAG（Retrieval-Augmented Generation）技术，从扫地机器人领域知识库中检索信息，并生成自然语言回答或建议。

<img width="1938" height="1230" alt="image" src="https://github.com/user-attachments/assets/4118bab6-62fd-44ba-8957-827072a7bf76" />
<img width="1902" height="1287" alt="image" src="https://github.com/user-attachments/assets/dbac5827-21cb-4244-b0e0-72a56ace0acd" />


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
            ├─ config/            # YAML 配置
            ├─ data/              # 原始数据与中间数据
            ├─ model/             # 模型工厂
            ├─ prompts/           # 提示词模板
            ├─ rag/                # RAG 与向量存储服务
            └─ utils/             # 通用工具
                └─ app.py         # Streamlit 入口
```

### 四、模块间逻辑架构（Mermaid）

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

## 环境与运行 Environment & How to Run

- **运行环境 Environment**
  - 已在 **Linux（Debian 12）** 和 **Windows 10/11（原生环境）** 上实际跑通本仓库中的示例与 RAG 子项目。  
    All examples and the RAG sub‑project have been tested on **Linux (Debian 12)** and **Windows 10/11 (native)**.
  - 推荐使用 **Python 3.10–3.12** 版本运行（开发与主要测试环境为 Python 3.12）。  
    It is recommended to use **Python 3.10–3.12** (developed and mainly tested with Python 3.12).
  - 需要自行配置对应的大模型平台 API Key（目前示例多基于阿里云通义 / DashScope，部分通用示例脚本也支持 OpenAI 等）。  
    You need to configure your own API keys for LLM providers (most examples focus on Alibaba Cloud Tongyi/DashScope; some generic demo scripts also support OpenAI, etc.).

- **配置环境变量 Configure Environment Variables**
  - 项目使用 `.env` 文件来管理 API Key 等敏感配置信息。  
    The project uses `.env` file to manage sensitive configuration like API keys.
  - **步骤 Steps:**
    1. 复制示例配置文件：`cp .env.example .env`  
       Copy the example config file: `cp .env.example .env`
    2. 编辑 `.env` 文件，将占位符替换为你的真实 API Key：  
       Edit `.env` file and replace placeholders with your actual API keys:
       ```bash
       # 阿里云 DashScope API Key（推荐）
       # Alibaba Cloud DashScope API Key (Recommended)
       DASHSCOPE_API_KEY=your_dashscope_api_key_here
       
       # 或使用通用 API_KEY（兼容性选项）
       # Or use generic API_KEY (compatibility option)
       API_KEY=your_api_key_here
       ```
    3. **获取阿里云 DashScope API Key：**  
       **How to get Alibaba Cloud DashScope API Key:**
       - 登录 [阿里云控制台](https://home.console.aliyun.com/)  
         Login to [Alibaba Cloud Console](https://home.console.aliyun.com/)
       - 开通 DashScope 服务并创建 API Key  
         Enable DashScope service and create an API Key
       - 详细文档：[DashScope API Key 创建指南](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)  
         Documentation: [DashScope API Key Creation Guide](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)
  - **注意 Note:**
    - `.env` 文件已添加到 `.gitignore`，不会被提交到版本库，请放心填写真实密钥。  
      `.env` file is in `.gitignore` and will not be committed to the repository, so you can safely add your real API keys.
    - **重要：本项目仅支持阿里云 DashScope API Key**  
      **Important: This project only supports Alibaba Cloud DashScope API Key**
    - 项目代码会优先读取 `DASHSCOPE_API_KEY`，如果未设置则回退到 `API_KEY`。  
      The code will first try to read `DASHSCOPE_API_KEY`, and fall back to `API_KEY` if not set.
      - **即使使用 `API_KEY` 作为变量名，也必须填入阿里云 DashScope 的 API Key，而不是 OpenAI 的 API Key。**  
        **Even if you use `API_KEY` as the variable name, you must fill in Alibaba Cloud DashScope API Key, NOT OpenAI API Key.**
      - 本项目使用 `DashScopeEmbeddings` 和 `ChatTongyi`（通义千问），不支持 OpenAI 模型。  
        This project uses `DashScopeEmbeddings` and `ChatTongyi` (Tongyi Qianwen), and does not support OpenAI models.

- **安装依赖 Install Dependencies**
  - 首次使用前，请先安装项目所需的 Python 依赖包：  
    Before running any scripts, please install the required Python packages:
  
```bash
pip install -r requirements.txt
```

- **运行方式 How to Run**
  - 克隆本仓库后，在项目根目录下运行任意示例脚本，例如：  
    After cloning this repo, from the project root you can run any example script, e.g.:

```bash
cd /path/to/your/HeiMa-AI-LLM-RAG-Agent-Dev
python AI_LLM_RAG_Agent_Dev/11_LangChain_Tongyi_Basic_Usage.py
```

  - 如果你使用的是本仓库同款 Devbox 环境（路径通常为 `/home/devbox/project`），可以使用下面的命令：  
    If you are using the same Devbox environment as this repo (usually at `/home/devbox/project`), you can run:

```bash
cd /home/devbox/project/AI_LLM_RAG_Agent_Dev
python 11_LangChain_Tongyi_Basic_Usage.py
```

## 声明与目的 Disclaimer & Purpose

- **学习用途**：本仓库仅用于个人学习与笔记整理，无任何商业用途。  
  **For learning only**: This repository is for personal study and note-taking, not for commercial use.
- **非官方代码**：本项目与黑马程序员、课程官方无直接关联，仅参考其公开课程内容进行实践。  
  **Not official**: This is not an official repository of the HeiMa course; it is only inspired by and based on the public videos.
- **欢迎扩展**：你可以在此基础上继续扩展自己的 RAG / Agent 实战项目与实验。  
  **Feel free to extend**: You are welcome to build your own RAG and Agent experiments on top of this repo.

