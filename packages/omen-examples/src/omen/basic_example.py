"""
Basic example demonstrating core OMEN functionality.
"""

from pathlib import Path
from omen.core.config import AppSettings
from omen.core.llm import LLMClient
from omen.vectorstore.store import VectorStore

def main():
    # Load configuration
    config = AppSettings()
    
    # Initialize LLM client
    llm = LLMClient(api_key=config.openai.api_key)
    
    # Initialize vector store
    store = VectorStore(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection_name=config.qdrant.collection_name
    )
    
    # Example text to embed
    texts = [
        "Customer transaction data from e-commerce system",
        "Product inventory levels across warehouses",
        "Marketing campaign performance metrics",
        "Customer support tickets and resolution times"
    ]
    
    # Create embeddings and store them
    print("Creating and storing embeddings...")
    for i, text in enumerate(texts):
        embedding = llm.create_embedding(text)
        store.add_document(
            id=f"doc_{i}",
            text=text,
            embedding=embedding,
            metadata={"source": "example", "type": "business_data"}
        )
    
    # Perform a semantic search
    print("\nPerforming semantic search...")
    query = "Find information about customer data"
    query_embedding = llm.create_embedding(query)
    
    results = store.search(
        query_embedding=query_embedding,
        limit=2
    )
    
    print(f"\nSearch results for query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Text: {result.text}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Metadata: {result.metadata}")

if __name__ == "__main__":
    main() 