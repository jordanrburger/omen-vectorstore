"""
Metadata document processing for the OMEN platform.
"""

from typing import List, Optional
from omen.vectorstore.models import MetadataDocument
from omen.vectorstore.vectorizer import Vectorizer
from omen.vectorstore.indexer import QdrantIndexer


class MetadataProcessor:
    """
    Handles processing and indexing of metadata documents.
    """

    def __init__(
        self,
        vectorizer: Vectorizer,
        indexer: QdrantIndexer,
        batch_size: Optional[int] = None
    ):
        """
        Initialize the processor.

        Args:
            vectorizer: Vectorizer to use for document vectorization
            indexer: Indexer to use for storing documents
            batch_size: Optional batch size for processing
        """
        self.vectorizer = vectorizer
        self.indexer = indexer
        self.batch_size = batch_size

    def process_batch(
        self,
        documents: List[MetadataDocument],
        batch_size: Optional[int] = None
    ) -> None:
        """
        Process and index a batch of documents.

        Args:
            documents: List of documents to process
            batch_size: Optional batch size override
        """
        if not documents:
            return

        # Use provided batch size or default
        batch_size = batch_size or self.batch_size

        # Process in batches if batch_size is specified
        if batch_size and batch_size > 0:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self._process_batch(batch)
        else:
            self._process_batch(documents)

    def _process_batch(self, documents: List[MetadataDocument]) -> None:
        """
        Process a batch of documents.

        Args:
            documents: List of documents to process
        """
        # Vectorize documents
        vectorized = self.vectorizer.vectorize_batch(documents)
        
        # Index documents
        self.indexer.index_documents(vectorized) 