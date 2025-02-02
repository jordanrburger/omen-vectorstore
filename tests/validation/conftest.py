"""
Common test configurations and fixtures for validation tests.
"""

import pytest
import os
from typing import Dict
import yaml

@pytest.fixture(scope="session")
def validation_config() -> Dict:
    """Load validation test configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "validation_config.yaml")
    
    # Default configuration
    default_config = {
        "performance": {
            "query_latency_p95_ms": 100,
            "batch_index_time_per_item_s": 1,
            "concurrent_searches": 50,
            "concurrent_search_latency_p95_ms": 200,
            "max_latency_ms": 500
        },
        "memory": {
            "max_index_memory_mb": 1000,
            "max_search_memory_mb": 500
        },
        "test_data": {
            "small_batch_size": 100,
            "large_batch_size": 1000,
            "concurrent_test_size": 100
        }
    }
    
    # Load custom configuration if exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
            # Merge custom config with defaults
            for category, values in custom_config.items():
                if category in default_config:
                    default_config[category].update(values)
                else:
                    default_config[category] = values
    
    return default_config

@pytest.fixture(scope="session")
def cleanup_collections():
    """Cleanup fixture to remove test collections after validation tests."""
    collections_to_cleanup = []
    
    def _register_collection(collection_name: str):
        collections_to_cleanup.append(collection_name)
    
    yield _register_collection
    
    # Cleanup registered collections
    from app.config import Config
    from qdrant_client import QdrantClient
    
    config = Config()
    client = QdrantClient(url=config.qdrant_url, port=config.qdrant_port)
    
    for collection_name in collections_to_cleanup:
        try:
            client.delete_collection(collection_name)
        except Exception as e:
            print(f"Failed to delete collection {collection_name}: {e}")
    
    client.close() 