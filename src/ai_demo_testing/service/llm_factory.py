import os

from langchain_ollama import OllamaLLM
from langchain_openai import OpenAI

from ai_demo_testing.service.testcase_sevice import callback_handler


class LLMFactory:
    def __init__(self):
        self.llm = None

    @classmethod
    def create_llm(cls, name):
        if name == 'llama':
            # 最新写法：使用 OllamaLLM
            cls.llm = OllamaLLM(
                model="llama3.1:8b",
                base_url="http://localhost:11434",
                callbacks=[callback_handler],
                verbose=True
            )
        elif name == 'openai':
            cls.llm = OpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                base_url=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        return cls.llm

    def get_llm(self):
        return self.llm
