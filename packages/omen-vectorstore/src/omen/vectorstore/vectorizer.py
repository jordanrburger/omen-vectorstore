"""
Document vectorization for the OMEN platform.
"""

from typing import List, Optional
from omen.vectorstore.models import MetadataDocument
from omen.vectorstore.embedding import EmbeddingProvider


class Vectorizer:
    """
    Handles vectorization of metadata documents using an embedding provider.
    """

    def __init__(self, embedding_provider: EmbeddingProvider):
        """
        Initialize the vectorizer.

        Args:
            embedding_provider: Provider to use for generating embeddings
        """
        self.embedding_provider = embedding_provider

    def vectorize_batch(
        self, 
        documents: List[MetadataDocument], 
        batch_size: Optional[int] = None
    ) -> List[MetadataDocument]:
        """
        Vectorize a batch of documents.

        Args:
            documents: List of documents to vectorize
            batch_size: Optional batch size for processing

        Returns:
            List of documents with vectors added
        """
        if not documents:
            return []

        # Process in batches if batch_size is specified
        if batch_size and batch_size > 0:
            result = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result.extend(self._process_batch(batch))
            return result
        else:
            return self._process_batch(documents)

    def _process_batch(self, documents: List[MetadataDocument]) -> List[MetadataDocument]:
        """
        Process a batch of documents.

        Args:
            documents: List of documents to process

        Returns:
            List of documents with vectors added
        """
        # Extract text content for embedding
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_provider.get_embeddings(texts)
        
        # Add vectors to documents
        for doc, vector in zip(documents, embeddings):
            doc.vector = vector
            
        return documents 