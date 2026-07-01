from langchain_community.vectorstores import FAISS
from app.core.config import settings

def get_retriever(store: FAISS):
    return store.as_retriever(
        search_type="mmr",       # Maximum Marginal Relevance
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": 10
        }
    )