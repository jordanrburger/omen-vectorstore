"""
Tests for batch processing functionality.
"""

import asyncio
from typing import List, Dict, Any
import pytest

from omen.core.batch import BatchProcessor

@pytest.fixture
def test_items() -> List[Dict[str, Any]]:
    """Create test items for batch processing."""
    return [
        {"id": 1, "data": "item1"},
        {"id": 2, "data": "item2"},
        {"id": 3, "data": "item3"},
        {"id": 4, "data": "item4"},
        {"id": 5, "data": "item5"}
    ]

@pytest.fixture
def test_processor():
    """Create a test batch processor."""
    return BatchProcessor(
        batch_size=2,
        max_workers=2,
        max_retries=3
    )

async def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Test processing function."""
    # Simulate some processing time
    await asyncio.sleep(0.1)
    return {
        "id": item["id"],
        "processed": True,
        "data": f"processed_{item['data']}"
    }

async def failing_process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Test failing processing function."""
    if item["id"] == 3:
        raise ValueError("Processing failed")
    return await process_item(item)

def test_batch_processor_creation():
    """Test batch processor creation."""
    processor = BatchProcessor()
    assert processor.batch_size == 100
    assert processor.max_workers == 4
    assert processor.max_retries == 3

def test_batch_processor_custom_settings():
    """Test batch processor with custom settings."""
    processor = BatchProcessor(
        batch_size=10,
        max_workers=2,
        max_retries=5
    )
    assert processor.batch_size == 10
    assert processor.max_workers == 2
    assert processor.max_retries == 5

@pytest.mark.asyncio
async def test_batch_processor_success(test_processor, test_items):
    """Test successful batch processing."""
    results = await test_processor.process_batch(
        items=test_items,
        process_fn=process_item
    )
    
    assert len(results) == len(test_items)
    for result in results:
        assert result["processed"]
        assert result["data"].startswith("processed_")

@pytest.mark.asyncio
async def test_batch_processor_partial_failure(test_processor, test_items):
    """Test batch processing with partial failures."""
    results = await test_processor.process_batch(
        items=test_items,
        process_fn=failing_process_item
    )
    
    # Should process all items except id=3
    assert len(results) == len(test_items) - 1
    for result in results:
        assert result["id"] != 3
        assert result["processed"]
        assert result["data"].startswith("processed_")

@pytest.mark.asyncio
async def test_batch_processor_empty_batch(test_processor):
    """Test processing empty batch."""
    results = await test_processor.process_batch(
        items=[],
        process_fn=process_item
    )
    assert len(results) == 0

@pytest.mark.asyncio
async def test_batch_processor_single_item(test_processor):
    """Test processing single item."""
    item = {"id": 1, "data": "single"}
    results = await test_processor.process_batch(
        items=[item],
        process_fn=process_item
    )
    
    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["processed"]
    assert results[0]["data"] == "processed_single"

@pytest.mark.asyncio
async def test_batch_processor_concurrent_limit(test_processor, test_items):
    """Test concurrent processing limit."""
    processed_count = 0
    
    async def counting_process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal processed_count
        processed_count += 1
        return await process_item(item)
    
    await test_processor.process_batch(
        items=test_items,
        process_fn=counting_process_item
    )
    
    # Should not exceed max_workers
    assert processed_count <= test_processor.max_workers

@pytest.mark.asyncio
async def test_batch_processor_retry(test_processor):
    """Test retry mechanism."""
    retry_count = 0
    
    async def retrying_process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal retry_count
        retry_count += 1
        if retry_count <= 2:
            raise ValueError("Temporary failure")
        return await process_item(item)
    
    results = await test_processor.process_batch(
        items=[{"id": 1, "data": "retry"}],
        process_fn=retrying_process_item
    )
    
    assert len(results) == 1
    assert retry_count == 3  # Initial try + 2 retries 