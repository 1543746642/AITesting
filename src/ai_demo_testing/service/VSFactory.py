import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

from langchain_openai import OpenAIEmbeddings


class VSFactory:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
        )

    @classmethod
    def create_vectorstore(cls, name):
        if name == 'chroma':
            db = Chroma(
                embedding_function=OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    base_url=os.getenv("OPENAI_API_BASE"),
                    api_key=os.getenv("OPENAI_API_KEY")
                ),
                persist_directory=str(Path(__file__).parent.parent / "chroma_store")
            )
            return db
        else:
            return None

    @classmethod
    def load_vectorstore(cls, name) -> Chroma:
        if name == 'chroma':
            # 必须使用与创建时完全相同的嵌入函数和存储路径
            embedding_func = OpenAIEmbeddings(
                model="text-embedding-3-small",
                base_url=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            persist_dir = str(Path(__file__).parent.parent / "chroma_store")

            # 使用 persist_directory 和 embedding_function 进行加载
            db = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_func
            )
            if db is None:
                db = VSFactory.create_vectorstore("chroma")
            return db
        else:
            return None
