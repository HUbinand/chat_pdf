import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
import faiss
from functools import lru_cache
from load_and_split_pdf import load_and_split_pdf 
from openai import OpenAI
# 加载环境变量
load_dotenv()


@lru_cache(maxsize=1024)  # 缓存函数返回值，避免重复计算
def get_embedding(text):
    """获取单个文本的嵌入向量"""
    client = OpenAI(
        api_key=os.getenv("Qwen_Long_API_KEY"),  # 确保环境变量已设置
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    try:
        completion = client.embeddings.create(
            model="text-embedding-v3",
            input=text, 
            dimensions=1024, # 设置嵌入向量大小
            encoding_format="float"
        )
        
        # 将响应转换为字典
        completion_dict = completion.model_dump()
        
        # 提取并返回嵌入向量
        embedding_vector = completion_dict['data'][0]['embedding']
        return embedding_vector
    except Exception as e:
        print(f"请求过程中出现错误: {e}")
        return None

def get_embeddings_for_texts(texts):
    """为一个文本列表中的每个文本获取嵌入向量"""
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        if embedding is not None:  # 如果成功获取到嵌入向量，则添加到列表中
            embeddings.append(embedding)
        else:
            print(f"无法获取文本 '{text}' 的嵌入向量")
    return embeddings

# # 加载分割好的PDF内容
# pdf_path = r"C:\AIGC学习课程\RAG代码规范问答系统\data\java开发手册.pdf"
# split_texts = load_and_split_pdf(pdf_path)
# # 提取文本内容而非 Document 对象
# text_list = [doc.page_content for doc in split_texts]
# print(type(text_list))
# print(len(text_list))

# # 获取所有文本列表的嵌入向量
# embeddings = get_embeddings_for_texts(text_list)

# print("嵌入文本向量的个数",len(embeddings))
# print("文本嵌入向量维度：", len(embeddings[0]))