from pathlib import Path

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ai_demo_testing.service.VSFactory import VSFactory

persist_path = Path("src/ai_demo_testing/chroma_store/chroma.sqlite3")
@tool
def search_test_knowledge(query: str) -> list[Document]:
    """模拟或真实查询测试理论知识"""
    vector_stores = VSFactory.load_vectorstore(persist_path)
    docs = vector_stores.similarity_search(query, k=4)
    return docs

# # tools/self_correction.py
# @tool
# def fix_json_output(broken_json: str, error_msg: str) -> str:
#     """调用另一个模型修复 JSON 格式"""
#     # 这里可以再调用一次 LLM 修复
#     return "修复后的 JSON..."

tools = [search_test_knowledge]  # 统一导出