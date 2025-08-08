#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A minimal test script that only tests core configuration.
"""

import os
import sys
import importlib.util

# Set up paths
base_dir = os.path.dirname(os.path.abspath(__file__))
packages_dir = os.path.join(base_dir, "packages")

# Directly load core configuration
print("Loading core configuration module...")
config_path = os.path.join(packages_dir, "omen-core/src/omen/core/config.py")
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Load core logging
print("Loading core logging module...")
logging_path = os.path.join(packages_dir, "omen-core/src/omen/core/logging.py")
spec = importlib.util.spec_from_file_location("logging", logging_path)
logging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging)

# Print settings
settings = config.load_settings()
print("\nSettings loaded successfully!")
print("Available settings:")
for key, value in settings.model_dump().items():
    if isinstance(value, str) and key.endswith('_key'):
        value = "***" if value else "Not set"
    print(f"  {key}: {value}")

# Test direct OpenAI API access if a query is provided
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
    print(f"\nTesting OpenAI API with query: '{query}'")
    
    try:
        import openai
        
        # Set up OpenAI client
        openai.api_key = settings.openai.api_key
        
        # Generate embedding directly
        response = openai.embeddings.create(
            model=settings.openai.embedding_model,
            input=query
        )
        
        # Extract the embedding vector
        vector = response.data[0].embedding
        print(f"Successfully generated embedding vector with dimension: {len(vector)}")
        
        print(f"\nTest successful: OpenAI API is working!")
    except Exception as e:
        print(f"Error using OpenAI API: {e}")
else:
    print("\nTo test OpenAI API, run: ./test_core.py your query here")

print("\nScript completed.") 