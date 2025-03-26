#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A standalone script that tests OpenAI API directly without relying on the omen modules.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("dotenv not installed, skipping .env loading")

# OpenAI API settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# Qdrant settings
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "55000"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "keboola_metadata")

# Print current configuration
print("\nCurrent configuration:")
print(f"  OpenAI API Key: {'Set' if OPENAI_API_KEY else 'Not set'}")
print(f"  OpenAI Embedding Model: {OPENAI_EMBEDDING_MODEL}")
print(f"  Qdrant Host: {QDRANT_HOST}")
print(f"  Qdrant Port: {QDRANT_PORT}")
print(f"  Qdrant Collection: {QDRANT_COLLECTION}")

# Test OpenAI API if enabled
if len(sys.argv) > 1 and OPENAI_API_KEY:
    query = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
    print(f"\nTesting OpenAI embedding with: '{query}'")
    
    try:
        import openai
        
        # Set the API key
        openai.api_key = OPENAI_API_KEY
        
        # Create embedding
        response = openai.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=query
        )
        
        # Extract the embedding vector
        vector = response.data[0].embedding
        print(f"Successfully generated embedding with dimension: {len(vector)}")
        
        # Try to use Qdrant if requested
        if "--search" in sys.argv:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Filter
                
                print(f"\nConnecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                
                # Check if collection exists
                collections = client.get_collections()
                if QDRANT_COLLECTION not in [c.name for c in collections.collections]:
                    print(f"Collection '{QDRANT_COLLECTION}' does not exist yet")
                else:
                    print(f"Collection '{QDRANT_COLLECTION}' exists")
                    
                    # Get collection info
                    collection_info = client.get_collection(QDRANT_COLLECTION)
                    print(f"Collection has {collection_info.points_count} points")
                    print(f"Vector dimension: {collection_info.config.params.vectors.size}")
                    
                    if collection_info.points_count > 0:
                        # Perform search
                        print("\nPerforming search...")
                        search_results = client.search(
                            collection_name=QDRANT_COLLECTION,
                            query_vector=vector,
                            limit=5
                        )
                        
                        # Display results
                        if search_results:
                            print(f"Found {len(search_results)} results:")
                            for i, result in enumerate(search_results):
                                print(f"{i+1}. Score: {result.score:.4f}, ID: {result.id}")
                                print(f"   Payload: {json.dumps(result.payload, indent=2)[:200]}...")
                        else:
                            print("No search results found")
            except ImportError:
                print("qdrant_client not installed, skipping vector search")
            except Exception as e:
                print(f"Error in Qdrant search: {e}")
    except ImportError:
        print("OpenAI package not installed. Install with: pip3 install openai")
    except Exception as e:
        print(f"Error using OpenAI API: {e}")
elif not OPENAI_API_KEY:
    print("\nOpenAI API key not set, skipping test")
    print("Set the OPENAI_API_KEY environment variable or add it to a .env file")
else:
    print("\nTo test OpenAI embeddings, run: ./direct_test.py your query here")
    print("Add --search to also test Qdrant search")

print("\nScript completed.") 