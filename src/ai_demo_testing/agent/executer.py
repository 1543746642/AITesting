# agent/executor.py
from .builder import build_agent
from config.settings import Config


class TestingAgent:
    def __init__(self):
        self.current_role = Config.DEFAULT_ROLE
        self.executor = build_agent(self.current_role)
        self.history = []

    def switch_role(self, role_name: str):
        if role_name in ["test_case_generator", "test_case_reviewer", "api_test_specialist"]:
            self.current_role = role_name
            self.executor = build_agent(role_name)
            return f"已切换到角色：{role_name}"
        return "角色不存在"

    def chat(self, user_input: str):
        response = self.executor.invoke({
            "input": user_input,
            "chat_history": self.history[-Config.MAX_HISTORY:]
        })
        self.history.extend([("human", user_input), ("assistant", response["output"])])
        return response["output"]