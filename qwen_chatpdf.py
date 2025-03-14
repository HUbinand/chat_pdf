# 任务：pdf文档阅读功能
# 输入问题，输出PDF相关内容

from load_and_split_pdf import load_and_split_pdf
from qwen_embedding import get_embeddings_for_texts,get_embedding
from qwen_llm import Qwen_LLM
import faiss
import numpy as np

# 读取PDF并切割
pdf_path = r"C:\AIGC学习课程\RAG代码规范问答系统\data\java开发手册.pdf"
split_texts = load_and_split_pdf(pdf_path)
# 提取文本内容而非 Document 对象
text_list = [doc.page_content for doc in split_texts]

# 获取所有文本列表的嵌入向量
embeddings = get_embeddings_for_texts(text_list)

# 创建 FAISS 欧氏距离索引
hit_vd = faiss.IndexFlatL2(1024)

# 将生成的嵌入向量列表添加到 FAISS 索引中
# hit_vd只接受数组类型的向量
hit_vd.add(np.array(embeddings).astype("float32"))

# 实例化
# Qwen_bot = Qwen_LLM("qwen-turbo")

# # chatpdf
# while True:
#     query_text = input("请输入查询文本：")
#     query_embedding = get_embedding(query_text)

#     distances, indices = hit_vd.search(np.array([query_embedding]).astype('float32'), 3)

#     print("最相似的 3 个向量的索引：", indices)
#     print("它们与查询向量的距离：", distances)

#     idx = indices[0][0]  # 获取最相似向量的索引
#     match_txt = text_list[idx] # 获取最相似向量的文本内容

#     bot_recive_prompt = (
#     """
#     背景：你是一个智能文档助手，擅长从PDF文档中提取和整合信息，以回答用户的问题。  
#     任务：请根据用户的问题和PDF文档内容，生成合理的回答，并标明信息来源。  
#     原则：  
#     1. 你的回答应基于提供的PDF内容，不可凭空编造信息。  
#     2. 如果PDF内容无法回答用户的问题，请礼貌地说明未找到相关信息。  
#     3. 回答应简洁明了，如有必要，可提供额外解释或建议进一步查阅文档的相关部分。    

#     示例：  

#     用户的问题：什么是深度学习？  
#     PDF内容：  
#     1. 深度学习是一种机器学习方法，它使用多层神经网络来建模复杂模式。

#     回复：深度学习是一种机器学习方法，利用多层神经网络来建模复杂模式。如果需要更详细的信息，建议查阅相关章节。  

#     用户的问题：{query_text}  
#     PDF内容：{match_txt}  
#     回复：
#     """
#     )
#     bot_recive_prompt = bot_recive_prompt.format(match_txt=match_txt, query_text=query_text)

#     if query_text.lower() in ["quit", "exit", "q"]:
#         print("再见")
#         break
#     response = Qwen_bot.chat(bot_recive_prompt)
#     print("Qwen_bot:", response)  # 在控制台中美化输出

def get_Qwen_PDFprompt(query_text):
    '''
    接收一个问题，返回机器人整合过的prompt
    '''
    query_embedding = get_embedding(query_text)

    distances, indices = hit_vd.search(np.array([query_embedding]).astype('float32'), 3)

    # print("最相似的 3 个向量的索引：", indices)
    # print("它们与查询向量的距离：", distances)

    idx = indices[0][0]  # 获取最相似向量的索引
    match_txt = text_list[idx] # 获取最相似向量的文本内容

    bot_recive_prompt = (
    """
    背景：你是一个智能文档助手，擅长从PDF文档中提取和整合信息，以回答用户的问题。  
    任务：请根据用户的问题和PDF文档内容，生成合理的回答，并标明信息来源。  
    原则：  
    1. 你的回答应基于提供的PDF内容，不可凭空编造信息。  
    2. 如果PDF内容无法回答用户的问题，请礼貌地说明未找到相关信息。  
    3. 回答应简洁明了，如有必要，可提供额外解释或建议进一步查阅文档的相关部分。    

    示例：  

    用户的问题：什么是深度学习？  
    PDF内容：  
    1. 深度学习是一种机器学习方法，它使用多层神经网络来建模复杂模式。

    回复：深度学习是一种机器学习方法，利用多层神经网络来建模复杂模式。如果需要更详细的信息，建议查阅相关章节。  

    用户的问题：{query_text}  
    PDF内容：{match_txt}  
    回复：
    """
    )
    Qwen_PDFprompt = bot_recive_prompt.format(match_txt=match_txt, query_text=query_text)

    return Qwen_PDFprompt
