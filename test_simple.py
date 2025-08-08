#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simplified test script that directly loads modules without relying on package imports.
"""

import os
import sys
import importlib.util

# Set up paths
base_dir = os.path.dirname(os.path.abspath(__file__))
packages_dir = os.path.join(base_dir, "packages")

# Directly load core configuration
config_path = os.path.join(packages_dir, "omen-core/src/omen/core/config.py")
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Load embedding provider
embedding_path = os.path.join(packages_dir, "omen-vectorstore/src/omen/vectorstore/embedding.py")
spec = importlib.util.spec_from_file_location("embedding", embedding_path)
embedding = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(embedding)
    embedding_loaded = True
except ImportError as e:
    print(f"Could not load embedding module: {e}")
    embedding_loaded = False

# Load vector search
search_path = os.path.join(packages_dir, "omen-vectorstore/src/omen/vectorstore/search.py")
spec = importlib.util.spec_from_file_location("search", search_path)
search = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(search)
    search_loaded = True
except ImportError as e:
    print(f"Could not load search module: {e}")
    search_loaded = False

# Print settings
settings = config.load_settings()
print("\nSettings loaded successfully!")
print("Available settings:")
for key, value in settings.model_dump().items():
    if isinstance(value, str) and key.endswith('_key'):
        value = "***" if value else "Not set"
    print(f"  {key}: {value}")

# Try to use OpenAI embedding if available
if embedding_loaded and len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
    print(f"\nGenerating embedding for: '{query}'")
    try:
        # Create embedding provider
        embedding_provider = embedding.OpenAIEmbedding(
            model=settings.openai.embedding_model,
            api_key=settings.openai.api_key,
            dimension=settings.openai.embedding_dimension
        )
        
        # Generate embedding
        vector = embedding_provider.embed(query)
        print(f"Successfully generated embedding vector with dimension: {len(vector)}")
        
        print("\nTest successful: OpenAI embedding generation works!")
    except Exception as e:
        print(f"Error generating embedding: {e}")
else:
    if not embedding_loaded:
        print("\nSkipped embedding test: Module could not be loaded")
    elif len(sys.argv) <= 1:
        print("\nTo test embedding generation, run: ./test_simple.py your query here")

print("\nScript completed.") 