# agent/builder.py
import yaml
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from config.settings import Config
from tools import tools  # 导入所有工具

# 加载角色配置
with open("prompts/roles.yaml", encoding="utf-8") as f:
    ROLES_CONFIG = yaml.safe_load(f)["roles"]


def get_llm():
    return ChatOllama(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)


def build_agent(role_name: str = Config.DEFAULT_ROLE):
    role_config = ROLES_CONFIG[role_name]
    system_prompt = role_config["system_prompt"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.format(tool_names=[t.name for t in tools])),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    llm = get_llm()
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor