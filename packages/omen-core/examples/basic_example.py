"""
Basic example demonstrating core OMEN functionality.
"""

from pathlib import Path
import uuid
from omen.core.config import AppSettings
from omen.core.llm import LLMClient
from omen.vectorstore.store import VectorStore
from qdrant_client import QdrantClient

def main():
    # Load configuration
    config = AppSettings()
    
    # Delete existing collection if it exists
    client = QdrantClient(
        host=config.qdrant.host,
        port=config.qdrant.port
    )
    try:
        client.delete_collection(config.qdrant.collection_name)
    except:
        pass
    
    # Initialize vector store
    store = VectorStore(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection_name=config.qdrant.collection_name,
        api_key=config.openai.api_key,
    )
    
    # Example text to embed
    texts = [
        "Customer transaction data from e-commerce system",
        "Product inventory levels across warehouses",
        "Marketing campaign performance metrics",
        "Customer support tickets and resolution times"
    ]
    
    # Store documents
    print("Storing documents...")
    for i, text in enumerate(texts):
        store.add_document(
            id=str(uuid.uuid4()),
            text=text,
            metadata={"source": "example", "type": "business_data"}
        )
    
    # Perform a semantic search
    print("\nPerforming semantic search...")
    query = "Find information about customer data"
    results = store.search(
        query=query,
        limit=2
    )
    
    print(f"\nSearch results for query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Text: {result.document.content}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Metadata: {result.document.metadata}")

if __name__ == "__main__":
    main() 