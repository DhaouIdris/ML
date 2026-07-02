from langchain.tools import Tool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import Ollama
from app.core.config import settings

def create_rag_tool(retriever):
    llm = Ollama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1
    )
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return Tool(
        name="DocumentSearch",
        func=lambda q: chain({"question": q}),
        description="Utile pour répondre à des questions sur les documents PDF indexés. Input: question en langage naturel."
    )