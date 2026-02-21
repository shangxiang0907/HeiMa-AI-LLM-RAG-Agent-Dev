"""
基于Streamlit完成WEB网页上传服务
pip install streamlit
"""

import streamlit as st
import time
from knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库更新服务")

# 初始化知识库服务
if 'kb_service' not in st.session_state:
    st.session_state.kb_service = KnowledgeBaseService()

# file_uploader
uploader_file = st.file_uploader(
    label="请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False,
    # False表示仅接受一个文件的上传
)

if uploader_file is not None:
    # 提取文件的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024  # KB
    
    st.subheader(f"文件名: {file_name}")
    st.write(f"格式: {file_type} | 大小: {file_size:.2f} KB")
    
    # get_value -> bytes -> decode('utf-8')
    text = uploader_file.getvalue().decode("utf-8")
    
    # 显示文件内容预览
    with st.expander("文件内容预览"):
        st.text(text[:1000] + "..." if len(text) > 1000 else text)
    
    # 上传按钮
    if st.button("上传到知识库"):
        # 在spinner内的代码执行过程中,会有一个转圈动画
        with st.spinner("正在处理文件..."):
            time.sleep(1)  # 模拟等待效果
            result = st.session_state.kb_service.upload_by_str(text, file_name)
            st.success(result)
