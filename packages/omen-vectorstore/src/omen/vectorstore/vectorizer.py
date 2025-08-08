"""
Document vectorization for the OMEN platform.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Union

from omen.core import get_logger
from omen.vectorstore.models import MetadataDocument
from omen.vectorstore.embedding import EmbeddingProvider

logger = get_logger(__name__)

# Max content length (in characters) to avoid token limit issues
# Text-embedding-3-large has a limit of 8192 tokens
# We'll use a much more conservative character limit based on experience
# that shows some documents tokenize at a higher rate than expected
MAX_CONTENT_LENGTH = 10000  # Approx 8192 tokens with safety margin

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

    def _truncate_content(self, content: str) -> str:
        """
        Truncate content to a safe size for embedding.
        
        Args:
            content: Content to truncate
            
        Returns:
            Truncated content
        """
        if len(content) <= MAX_CONTENT_LENGTH:
            return content
            
        logger.warning(f"Content length ({len(content)}) exceeds safe limit, truncating to {MAX_CONTENT_LENGTH} chars")
        
        # If it's JSON, handle more intelligently by keeping key structure
        if content.startswith('{') and content.endswith('}'):
            try:
                data = json.loads(content)
                # Truncate each string value in the JSON
                truncated_data = self._truncate_json_values(data)
                truncated = json.dumps(truncated_data)
                
                # If still too long, fall back to simple truncation
                if len(truncated) > MAX_CONTENT_LENGTH:
                    return content[:MAX_CONTENT_LENGTH] + "..."
                return truncated
            except json.JSONDecodeError:
                # Not valid JSON, use simple truncation
                return content[:MAX_CONTENT_LENGTH] + "..."
        
        # Simple truncation for text
        return content[:MAX_CONTENT_LENGTH] + "..."
        
    def _truncate_json_values(self, obj: Any) -> Any:
        """
        Recursively truncate string values in a JSON object.
        
        Args:
            obj: JSON object to truncate
            
        Returns:
            Truncated JSON object
        """
        # Limit the max number of items in lists and dictionaries
        MAX_ITEMS = 20
        MAX_STRING_LENGTH = 500  # More aggressive truncation of string values
        
        if isinstance(obj, dict):
            # If dictionary is too large, keep only the most important keys
            if len(obj) > MAX_ITEMS:
                logger.warning(f"Truncating large dictionary with {len(obj)} keys to {MAX_ITEMS} keys")
                # Try to keep the most important keys - id, name, type, etc.
                important_keys = ["id", "name", "type", "description", "source"]
                # First add important keys
                result = {}
                for key in important_keys:
                    if key in obj:
                        result[key] = self._truncate_json_values(obj[key])
                
                # Then add other keys up to MAX_ITEMS
                remaining_slots = MAX_ITEMS - len(result)
                if remaining_slots > 0:
                    for key, value in obj.items():
                        if key not in result and len(result) < MAX_ITEMS:
                            result[key] = self._truncate_json_values(value)
                
                # Add a note about truncation
                if len(obj) > MAX_ITEMS:
                    result["_truncated"] = f"Truncated {len(obj) - MAX_ITEMS} additional keys"
                
                return result
            else:
                return {k: self._truncate_json_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # If list is too large, keep only a subset
            if len(obj) > MAX_ITEMS:
                logger.warning(f"Truncating large list with {len(obj)} items to {MAX_ITEMS} items")
                truncated = [self._truncate_json_values(item) for item in obj[:MAX_ITEMS-1]]
                truncated.append({"_truncated": f"Truncated {len(obj) - (MAX_ITEMS-1)} additional items"})
                return truncated
            else:
                return [self._truncate_json_values(item) for item in obj]
        elif isinstance(obj, str):
            if len(obj) > MAX_STRING_LENGTH:
                return obj[:MAX_STRING_LENGTH] + "..."
            else:
                return obj
        else:
            return obj

    def _process_batch(self, documents: List[MetadataDocument]) -> List[MetadataDocument]:
        """
        Process a batch of documents.

        Args:
            documents: List of documents to process

        Returns:
            List of documents with vectors added
        """
        # Extract text content for embedding with truncation
        texts = []
        skipped_docs = []
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                truncated_content = self._truncate_content(doc.content)
                texts.append(truncated_content)
                processed_docs.append(doc)
            except Exception as e:
                logger.error(f"Error processing document {doc.id}: {e}")
                skipped_docs.append(doc.id)
        
        if skipped_docs:
            logger.warning(f"Skipped {len(skipped_docs)} documents due to processing errors")
            
        if not texts:
            logger.warning("No documents to vectorize after preprocessing")
            return []
        
        try:
            # Generate embeddings
            embeddings = self.embedding_provider.get_embeddings(texts)
            
            # Add vectors to documents
            for doc, vector in zip(processed_docs, embeddings):
                doc.vector = vector
                
            return processed_docs
        except Exception as e:
            logger.error(f"Error during vectorization: {e}")
            # Return documents without vectors if embedding fails
            return processed_docs 