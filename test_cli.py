#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A test script to directly import omen modules and run commands.
"""

import os
import sys
import importlib.util

def import_from_path(module_name, file_path):
    """Import a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

base_dir = os.path.dirname(os.path.abspath(__file__))
packages_dir = os.path.join(base_dir, "packages")

# Import core modules directly
print("Importing modules...")
core_config = import_from_path("omen.core.config", os.path.join(packages_dir, "omen-core/src/omen/core/config.py"))
core_logging = import_from_path("omen.core.logging", os.path.join(packages_dir, "omen-core/src/omen/core/logging.py"))

# Set up logging
logger = core_logging.get_logger(__name__)
core_logging.configure_logging("INFO")

# Print configuration
settings = core_config.load_settings()
print("\nSettings loaded successfully!")
print("Available settings:")
for key, value in settings.model_dump().items():
    if isinstance(value, str) and key.endswith('_key'):
        value = "***" if value else "Not set"
    print(f"  {key}: {value}")

# Import vectorstore modules
print("\nLoading vector search modules...")
vectorstore_init = import_from_path("omen.vectorstore.__init__", os.path.join(packages_dir, "omen-vectorstore/src/omen/vectorstore/__init__.py"))
vectorstore_search = import_from_path("omen.vectorstore.search", os.path.join(packages_dir, "omen-vectorstore/src/omen/vectorstore/search.py"))
vectorstore_embedding = import_from_path("omen.vectorstore.embedding", os.path.join(packages_dir, "omen-vectorstore/src/omen/vectorstore/embedding.py"))

# Testing vector search if a query is provided
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
    print(f"\nPerforming vector search for: '{query}'")
    
    # Get embedding provider and search engine
    embedding_provider = vectorstore_embedding.OpenAIEmbedding(
        model=settings.openai.embedding_model,
        api_key=settings.openai.api_key,
        dimension=settings.openai.embedding_dimension
    )
    
    try:
        # Test embedding generation
        vector = embedding_provider.embed(query)
        print(f"Generated embedding vector with dimension: {len(vector)}")
        
        # Test vector search (if Qdrant is running)
        from qdrant_client import QdrantClient
        
        qdrant_client = QdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port,
        )
        
        print("\nChecking Qdrant connection...")
        collections = qdrant_client.get_collections()
        print(f"Found {len(collections.collections)} collections in Qdrant")
        
        if settings.qdrant.collection_name in [c.name for c in collections.collections]:
            print(f"Collection '{settings.qdrant.collection_name}' exists")
            
            # Get collection info
            collection_info = qdrant_client.get_collection(settings.qdrant.collection_name)
            print(f"Collection has {collection_info.points_count} points")
            
            # Perform search
            if collection_info.points_count > 0:
                print(f"\nSearching for '{query}'...")
                search_results = qdrant_client.search(
                    collection_name=settings.qdrant.collection_name,
                    query_vector=vector,
                    limit=5
                )
                
                # Display results
                if search_results:
                    print(f"Found {len(search_results)} results:")
                    for i, result in enumerate(search_results):
                        print(f"{i+1}. Score: {result.score:.4f}, ID: {result.id}")
                        if hasattr(result.payload, 'content'):
                            content = result.payload.content
                            print(f"   Content: {content[:100]}..." if len(content) > 100 else content)
                else:
                    print("No search results found")
            else:
                print("Collection is empty, no search possible")
        else:
            print(f"Collection '{settings.qdrant.collection_name}' does not exist yet. Need to index data first.")
    
    except Exception as e:
        print(f"Error during search: {e}")
else:
    print("\nTo perform a search, run the script with a query: ./test_cli.py your search query")

print("\nScript completed successfully!") 