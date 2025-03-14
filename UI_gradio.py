# 需求：使用gradio稳定的调用阿里云的qwenLLM大模型，然后结合本地读取PDF的内容生成输出内容。
# 需要实现不同用户访问对外开放的网站时候，可以独立对话，互不影响，实现多用户同时对话。
# 需要实现流式输出，即机器人回复的时候，可以逐字符输出，而不是一次性输出。
# 需要在本地保存用户的聊天记录。
import random
import gradio as gr
import time
from qwen_chatpdf import get_Qwen_PDFprompt
import threading
from qwen_llm import Qwen_LLM

# 一个字典用于存储每个会话的独立Qwen机器人实例
user_robots = {}

# 实例化Qwen机器人并为每个会话创建一个新的实例
def get_qwen_bot(session_hash):
    if session_hash not in user_robots:
        # 使用session_hash为每个用户创建独立的机器人
        user_robots[session_hash] = Qwen_LLM("qwen-turbo")
    return user_robots[session_hash]

# qwen函数
def qwen_PDFbot(message, history, request: gr.Request):
    session_hash = request.session_hash  # 获取当前用户会话的唯一标识
    Qwen_bot = get_qwen_bot(session_hash)  # 获取当前会话的独立Qwen机器人
    Qwen_PDFprompt = get_Qwen_PDFprompt(message)
    print(Qwen_PDFprompt)
    # 这里模拟流式输出
    respon = Qwen_bot.chat(Qwen_PDFprompt)
    # 你可以根据需要逐字符返回
    for i in range(1, len(respon) + 1):  # 循环逐步返回
        yield "Qwen_bot: " + str(respon)[:i]
        time.sleep(0.02)  # 模拟延迟0.02秒

# 自定义gr.ChatInterface
gr.ChatInterface(
    qwen_PDFbot,  # 指定处理逻辑的函数
    type="messages",  # 指定消息类型为“消息”格式，界面显示为对话框的样式
    title="基于RAG的代码规范智能问答系统",
    description="这是一个专门查询《阿里巴巴Java 开发手册》内容的智能问答系统",
    theme="soft",  # 选择软色主题，界面的配色会比较柔和
    cache_examples=True,  # 启用示例缓存，以提高加载速度
).launch(share=True)
