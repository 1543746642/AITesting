from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的测试开发 Agent。
你的任务：根据用户需求文档，生成结构化测试用例。
规则：
- 先分析需求，提取等价类、边界值、异常场景。
- 如需运行代码或搜索知识，使用工具。
- 输出必须是 JSON 格式：{{"cases": [...]}}
- 覆盖全面，准确率>80%。
可用工具：{tools}"""),
    MessagesPlaceholder("chat_history"),  # 自动管理历史
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # 工具调用中间记录
])