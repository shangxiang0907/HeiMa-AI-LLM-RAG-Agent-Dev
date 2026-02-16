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

- **`AI_LLM_RAG_Agent_Dev/`**：课程相关代码与练习（个人笔记）
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
  - **`11_LangChain_Tongyi_Basic_Usage.py`**：LangChain + 通义 千问 基础用法（Basic LangChain usage with Tongyi Qianwen.）
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

> 后续若继续跟随课程实现 RAG 检索增强问答、Agent 智能体、多工具编排等内容，会在该目录下持续补充脚本与说明。  
> As I progress through the course (RAG pipelines, Agents, tool calling, etc.), more scripts and notes will be added under this directory.

## 环境与运行 Environment & How to Run

- **运行环境 Environment**
  - 基于 Devbox 提供的 Debian 12 + Python 开发环境。  
    Based on Devbox environment (Debian 12 with Python pre-configured).
  - 需要自行配置对应的大模型平台 API Key（如 OpenAI、阿里云通义等）。  
    You need to configure your own API keys for LLM providers (OpenAI, Tongyi, etc.).

- **安装依赖 Install Dependencies**
  - 首次使用前，请先安装项目所需的 Python 依赖包：  
    Before running any scripts, please install the required Python packages:
  
```bash
pip install -r requirements.txt
```

- **运行方式 How to Run**
  - 进入 Devbox 开发环境后，可直接运行单个示例脚本，例如：  
    After entering the Devbox environment, you can run any script directly, e.g.:

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

