from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embeddings model: {settings.embedding_model}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings