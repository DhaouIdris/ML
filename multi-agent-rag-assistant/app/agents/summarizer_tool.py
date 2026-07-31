from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from app.core.config import settings

def create_summarizer_tool():
    llm = Ollama(base_url=settings.ollama_base_url, model=settings.ollama_model)
    prompt = PromptTemplate.from_template(
        "Résume le texte suivant de manière concise en 3-5 points clés:\n\n{text}"
    )
    chain = prompt | llm
    return Tool(
        name="Summarizer",
        func=lambda text: chain.invoke({"text": text}),
        description="Utile pour résumer un texte long. Input: texte à résumer."
    )