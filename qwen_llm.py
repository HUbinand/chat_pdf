#API秘钥：sk-f93b1dfaf71041c9a23e5c24d1d01247
#API请求URL:https://dashscope.aliyuncs.com/compatible-mode/v1
#API调用模型：qwen-turbo qwen-long qwen-plus

# windows下设置环境变量保存API秘钥
#会话范围：使用 $env:DEEPSEEK_API_KEY = "<your-api-key-here>" 
#设置的环境变量只在当前 PowerShell 会话中有效。一旦关闭 PowerShell 窗口，变量将消失。
# $env:Qwen_Long_API_KEY = "sk-f93b1dfaf71041c9a23e5c24d1d01247"
# $env:Qwen_Long_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 查看windows下设置的所有环境变量
# Get-ChildItem Env: 
# windows下查看是否配置成功
# echo $env:Qwen_Long_API_KEY
# echo $env:Qwen_Long_API_URL

#本地记忆原理：
# 历史记录：每次用户发送消息时，程序都会将这条消息添加到 conversation_history 列表中。
# 这个列表用于存储用户和模型的交互。连续上下文：在后续的对话中，
# 调用模型时会将整个历史记录传递给模型，这样模型就能够理解当前上下文，而不仅仅是单条消息。

#缓存原理：
# 使用 functools 模块的 lru_cache 装饰器来缓存 chat 方法的返回值。
# lru_cache 是一个缓存装饰器，它会将函数的结果缓存起来，以便在后续的调用中直接返回缓存的结果，
# 而不需要再次执行函数。这可以显著提高性能，特别是在处理大量重复输入时。

import os
from openai import OpenAI
from dotenv import load_dotenv
import datetime
from functools import lru_cache

# 加载环境变量
load_dotenv()


# 基类LLM，用于实现通用的接口方法
class BaseLLM:
    # 使用 lru_cache 装饰器，设置缓存大小为 1024，缓存 chat 方法的返回值
    @lru_cache(maxsize=1024)
    def chat(self, text):
        return self._chat(text)

    def _chat(self, text):
        raise NotImplementedError


# DeepSeek LLM 类，具体实现了与 DeepSeek API 的交互
class Qwen_LLM(BaseLLM):
    def __init__(self, model_name, log_dir=r'C:\AIGC学习课程\RAG代码规范问答系统\qwen_logs', log_file=None):
        # 使用环境变量获取 API 密钥和 URL
        self.client = OpenAI(api_key=os.environ.get("Qwen_Long_API_KEY"), base_url=os.environ.get("Qwen_Long_API_URL"))
        self.model_name = model_name
        self.conversation_history = []

        # 设置日志文件路径
        if log_file is None:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            log_file = os.path.join(log_dir, f"qwen_logs_{current_date}.txt")

        # 如果日志文件夹不存在，创建文件夹
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = log_file  # 保存聊天记录的文件

    def chat(self, text, temperature=0.2, max_tokens=2048):
        try:
            # 将当前文本添加到对话历史中
            self.conversation_history.append({"role": "user", "content": text})

            # 创建 API 请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False  # 关闭流式传输
            )

            # 获取模型的回答
            answer = response.choices[0].message.content

            # 将模型的回答添加到对话历史中
            self.conversation_history.append({"role": "assistant", "content": answer})

            # 自动保存聊天记录到文件
            self.save_to_file(text, answer)

            return answer

        except Exception as e:
            print(f"发生错误: {e}")
            return "对不起，发生了一个错误，请稍后再试。"

    def save_to_file(self, user_input, bot_response):
        """将问答内容保存到文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"用户: {user_input}\n")
                f.write(f"qwen_bot: {bot_response}\n")
                f.write("-" * 40 + "\n")  # 分隔线
        except Exception as e:
            print(f"日志保存失败: {e}")

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []


# # # 使用示例
# Qwen_bot = Qwen_LLM("qwen-turbo")

# # 没有自定义的提示词的简单多轮回答
# while True:
#     user_input = input("请输入你的问题: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         break
#     response = Qwen_bot.chat(user_input)
#     console.print("Qwen_bot:", response)  # 在控制台中美化输出