from langchain_community.document_loaders import PyMuPDFLoader  # 确保使用正确的 PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter  # 导入文本切割器

# 加载PDF并拆分文本
# chunk_size=500, chunk_overlap=0是指定每个文本片段的大小为50个字符，每个片段并且重叠字符为5确保上下文不丢失
def load_and_split_pdf(pdf_path):
    # 使用 PyMuPDFLoader 加载 PDF 文件
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load() #documents是列表,每个元素是一个Document对象，包含page_content和metadata属性
    
    # total_word_count = sum(len(doc.page_content.split()) for doc in documents)
    # print(f"文档总字数：{total_word_count}")

    # # 查看每一页的内容
    # for i, doc in enumerate(documents):
    #     print(f"文档第{i+1}页内容：")
    #     print(doc.page_content[:100])  # 只显示前500个字符，避免内容过多
    #     print("-" * 50)  # 分割线

    # 使用 CharacterTextSplitter 对文档进行拆分
    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    # print(type(texts))
    return texts

# # 示例使用
# pdf_path = r"C:\AIGC学习课程\RAG代码规范问答系统\data\java开发手册.pdf"  # 替换为你的PDF路径
# split_texts = load_and_split_pdf(pdf_path)

# # 输出拆分后的文本数量和部分内容
# print(f"共拆分为 {len(split_texts)} 段文本。")
# print(f"示例段落内容：{split_texts[0].page_content[:200]}")  # 打印第一个文本片段的前200个字符
