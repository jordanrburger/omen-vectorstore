"""
Validation test suite for Qdrant production readiness assessment.
These tests evaluate performance, scalability, and reliability metrics.
"""

import pytest
import asyncio
import time
import os
import json
from datetime import datetime
from typing import List, Dict
import numpy as np
from qdrant_client import QdrantClient, models
from app.config import Config
from app.validation.indexer import ValidationIndexer
from app.vectorizer import OpenAIProvider
from app.vectorizer import MetadataVectorizer
import uuid

# Global dictionary to store validation metrics
validation_metrics = {
    "timestamp": "",
    "query_latency": {},
    "concurrent_operations": {},
    "batch_indexing": {},
    "memory_usage": {}
}

def save_validation_report():
    """Save validation metrics to a JSON file."""
    validation_metrics["timestamp"] = datetime.now().isoformat()
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Save report with timestamp
    filename = f"reports/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(validation_metrics, f, indent=2)
    
    print(f"\nValidation report saved to: {filename}")
    print("\nValidation Results Summary:")
    print("=" * 50)
    print(f"Query Latency (p95): {validation_metrics['query_latency'].get('p95', 'N/A'):.2f}ms")
    print(f"Query Latency (p99): {validation_metrics['query_latency'].get('p99', 'N/A'):.2f}ms")
    print(f"Batch Indexing Time: {validation_metrics['batch_indexing'].get('total_time', 'N/A'):.2f}s")
    print(f"Concurrent Operations Success: {validation_metrics['concurrent_operations'].get('success', 'N/A')}")
    print(f"Peak Memory Usage: {validation_metrics['memory_usage'].get('peak_mb', 'N/A'):.2f}MB")
    print("=" * 50)

class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    def __init__(self, vector_size: int = 1536):
        self.vector_size = vector_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for testing."""
        return [np.random.rand(self.vector_size).tolist() for _ in texts]

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup after all tests complete."""
    def save_report():
        save_validation_report()
    request.addfinalizer(save_report)

@pytest.fixture
def setup_environment():
    """Set up environment variables for testing."""
    os.environ["QDRANT_HOST"] = "localhost"
    os.environ["QDRANT_PORT"] = "6333"
    os.environ["OPENAI_API_KEY"] = "dummy-key"
    os.environ["OPENAI_MODEL"] = "text-embedding-3-small"

@pytest.fixture
def config(setup_environment):
    """Get configuration from environment."""
    return Config.from_env()

@pytest.fixture
def qdrant_client(config):
    """Create Qdrant client for testing."""
    client = QdrantClient(
        host=config.qdrant_host,
        port=config.qdrant_port
    )
    return client

@pytest.fixture
def embedding_provider():
    """Create mock embedding provider for testing."""
    return MockEmbeddingProvider()

@pytest.fixture
def vectorizer(embedding_provider):
    """Create metadata vectorizer with mock provider."""
    return MetadataVectorizer(embedding_provider)

@pytest.fixture
def indexer(qdrant_client, vectorizer):
    """Create indexer for testing."""
    return ValidationIndexer(qdrant_client, vectorizer)

async def generate_test_metadata(count: int) -> List[Dict]:
    """Generate test metadata."""
    return [
        {
            "id": i,
            "description": f"Test metadata {i}",
            "type": "table",
            "created": "2024-01-01",
            "size": 1000,
            "columns": ["id", "name", "value"]
        }
        for i in range(count)
    ]

@pytest.mark.asyncio
async def test_query_latency(indexer):
    """Test search query latency."""
    collection_name = f"test_latency_{uuid.uuid4()}"
    await indexer.create_collection(collection_name)

    # Generate test data
    test_data = await generate_test_metadata(1000)
    
    # Index test data
    await indexer.index_metadata(test_data, collection_name)
    
    # Measure search latencies
    latencies = []
    for _ in range(100):
        start_time = time.time()
        await indexer.search("test query", collection_name)
        latency = (time.time() - start_time) * 1000  # Convert to ms
        latencies.append(latency)
    
    # Calculate percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    # Store metrics
    validation_metrics["query_latency"] = {
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "min": float(min(latencies)),
        "max": float(max(latencies)),
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "sample_size": len(latencies)
    }
    
    # More realistic thresholds for test environment
    assert p95 < 2000, f"95th percentile latency {p95:.2f}ms exceeds 2000ms target"
    assert p99 < 3000, f"99th percentile latency {p99:.2f}ms exceeds 3000ms target"
    
    await indexer.delete_collection(collection_name)

@pytest.mark.asyncio
async def test_concurrent_operations(indexer):
    """Test concurrent indexing and searching."""
    collection_name = f"test_concurrent_{uuid.uuid4()}"
    await indexer.create_collection(collection_name)
    
    # Generate test data
    test_data = await generate_test_metadata(100)
    
    # Test concurrent operations
    async def index_and_search():
        start_time = time.time()
        await indexer.index_metadata(test_data[:50], collection_name)
        results = await indexer.search("test", collection_name)
        await indexer.index_metadata(test_data[50:], collection_name)
        return time.time() - start_time, results
    
    tasks = [index_and_search() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    # Calculate metrics
    operation_times = [r[0] for r in results]
    validation_metrics["concurrent_operations"] = {
        "success": True,
        "total_operations": len(tasks),
        "mean_operation_time": float(np.mean(operation_times)),
        "max_operation_time": float(max(operation_times)),
        "min_operation_time": float(min(operation_times))
    }
    
    # Verify results
    assert all(isinstance(r[1], list) for r in results)
    
    await indexer.delete_collection(collection_name)

@pytest.mark.asyncio
async def test_batch_indexing_performance(indexer):
    """Test batch indexing performance."""
    collection_name = f"test_batch_{uuid.uuid4()}"
    await indexer.create_collection(collection_name)
    
    # Generate large test dataset
    test_data = await generate_test_metadata(1000)
    
    # Measure indexing time
    start_time = time.time()
    await indexer.index_metadata(test_data, collection_name)
    indexing_time = time.time() - start_time
    
    # Store metrics
    validation_metrics["batch_indexing"] = {
        "total_time": float(indexing_time),
        "records_per_second": float(len(test_data) / indexing_time),
        "total_records": len(test_data)
    }
    
    # More realistic threshold for test environment
    assert indexing_time < 60, f"Batch indexing took {indexing_time:.2f}s, exceeding 60s target"
    
    await indexer.delete_collection(collection_name)

@pytest.mark.asyncio
async def test_memory_usage(indexer):
    """Test memory usage during operations."""
    import psutil
    import os

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    collection_name = f"test_memory_{uuid.uuid4()}"
    await indexer.create_collection(collection_name)
    
    # Track memory usage
    initial_memory = get_memory_usage()
    peak_memory = initial_memory
    memory_samples = []
    
    # Generate test data with larger payloads
    test_data = [
        {
            "id": i,
            "description": f"Test metadata {i}",
            "large_field": "x" * 1000  # 1KB of data per record
        } 
        for i in range(1000)
    ]
    
    # Index data and perform searches while monitoring memory
    await indexer.index_metadata(test_data, collection_name)
    for _ in range(100):
        await indexer.search("test", collection_name)
        current_memory = get_memory_usage()
        memory_samples.append(current_memory)
        peak_memory = max(peak_memory, current_memory)
    
    # Store metrics
    validation_metrics["memory_usage"] = {
        "initial_mb": float(initial_memory),
        "peak_mb": float(peak_memory),
        "mean_mb": float(np.mean(memory_samples)),
        "std_mb": float(np.std(memory_samples)),
        "total_samples": len(memory_samples)
    }
    
    await indexer.delete_collection(collection_name) 