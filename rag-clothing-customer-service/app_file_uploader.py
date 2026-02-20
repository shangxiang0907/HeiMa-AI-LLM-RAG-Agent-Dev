"""
基于Streamlit完成WEB网页上传服务
pip install streamlit
"""

import streamlit as st

# 添加网页标题
st.title("知识库更新服务")

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
    st.write(text)
