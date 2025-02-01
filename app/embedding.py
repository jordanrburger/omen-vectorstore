import logging
from typing import Optional, Dict
from app.vectorizer import OpenAIProvider, SentenceTransformerProvider, EmbeddingProvider

logger = logging.getLogger(__name__)

def get_embedding_provider(config: Dict) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider based on configuration.
    """
    openai_api_key = config.get("OPENAI_API_KEY")
    if openai_api_key:
        logger.info("Using OpenAI embedding provider")
        return OpenAIProvider(
            api_key=openai_api_key,
            model=config.get("OPENAI_MODEL", "text-embedding-ada-002")
        )
    else:
        logger.info("Using SentenceTransformer embedding provider")
        return SentenceTransformerProvider(
            model_name=config.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
            device="cpu"
        ) 