from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain import hub
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

def create_agent(retriever, memory):
    from app.agents.rag_tool import create_rag_tool
    from app.agents.calculator_tool import create_calculator_tool
    from app.agents.summarizer_tool import create_summarizer_tool

    llm = Ollama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0
    )
    tools = [
        create_rag_tool(retriever),
        create_calculator_tool(),
        create_summarizer_tool()
    ]

    prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )