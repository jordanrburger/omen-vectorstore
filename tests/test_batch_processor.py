"""Tests for batch processing functionality."""

import pytest
from unittest.mock import Mock, patch
import time
from typing import List

from app.batch_processor import BatchProcessor, BatchConfig


@pytest.fixture
def mock_process_fn():
    """Create a mock processing function."""
    return Mock()

@pytest.fixture
def batch_processor():
    """Create a BatchProcessor with default config."""
    return BatchProcessor()

@pytest.fixture
def items():
    """Create test items."""
    return list(range(25))  # 25 items to test batching


def test_batch_processor_init():
    """Test BatchProcessor initialization."""
    # Test with default config
    processor = BatchProcessor()
    assert processor.config.batch_size == 10
    assert processor.config.max_retries == 3
    
    # Test with custom config
    custom_config = BatchConfig(batch_size=5, max_retries=2)
    processor = BatchProcessor(custom_config)
    assert processor.config.batch_size == 5
    assert processor.config.max_retries == 2


def test_process_batches_basic(batch_processor, mock_process_fn, items):
    """Test basic batch processing functionality."""
    batch_processor.process_batches(items, mock_process_fn)
    
    # Should have called process_fn 3 times (25 items with batch size 10)
    assert mock_process_fn.call_count == 3
    
    # Verify batch sizes
    assert len(mock_process_fn.call_args_list[0][0][0]) == 10  # First batch
    assert len(mock_process_fn.call_args_list[1][0][0]) == 10  # Second batch
    assert len(mock_process_fn.call_args_list[2][0][0]) == 5   # Last batch


def test_process_batches_empty(batch_processor, mock_process_fn):
    """Test processing empty list of items."""
    batch_processor.process_batches([], mock_process_fn)
    mock_process_fn.assert_not_called()


def test_process_batches_single_item(batch_processor, mock_process_fn):
    """Test processing single item."""
    batch_processor.process_batches([1], mock_process_fn)
    mock_process_fn.assert_called_once()
    assert len(mock_process_fn.call_args[0][0]) == 1


def test_process_batches_retry_success(batch_processor, items):
    """Test successful retry after failures."""
    fail_count = 0
    
    def process_with_failures(batch: List[int]):
        nonlocal fail_count
        if fail_count < 2:
            fail_count += 1
            raise ValueError("Simulated failure")
    
    # Should succeed on third try
    mock_fn = Mock(side_effect=process_with_failures)
    batch_processor.config.initial_retry_delay = 0.01  # Speed up test
    
    batch_processor.process_batches(items[:10], mock_fn)
    assert mock_fn.call_count == 3
    assert fail_count == 2


def test_process_batches_retry_failure(batch_processor, items):
    """Test retry exhaustion."""
    mock_fn = Mock(side_effect=ValueError("Simulated failure"))
    batch_processor.config.initial_retry_delay = 0.01  # Speed up test
    
    with pytest.raises(ValueError, match="Simulated failure"):
        batch_processor.process_batches(items[:10], mock_fn)
    
    assert mock_fn.call_count == batch_processor.config.max_retries + 1


def test_process_batches_backoff(batch_processor, items):
    """Test exponential backoff timing."""
    mock_fn = Mock(side_effect=ValueError("Simulated failure"))
    batch_processor.config.initial_retry_delay = 0.01
    batch_processor.config.backoff_factor = 2
    batch_processor.config.max_retry_delay = 1
    
    start_time = time.time()
    with pytest.raises(ValueError):
        batch_processor.process_batches(items[:10], mock_fn)
    elapsed_time = time.time() - start_time
    
    # With initial_delay=0.01, backoff=2, we expect:
    # 1st retry: 0.01s
    # 2nd retry: 0.02s
    # 3rd retry: 0.04s
    # Total >= 0.07s
    assert elapsed_time >= 0.07


@patch('app.batch_processor.tqdm')
def test_process_batches_progress(mock_tqdm, batch_processor, mock_process_fn, items):
    """Test progress bar functionality."""
    mock_progress = Mock()
    mock_tqdm.return_value.__enter__.return_value = mock_progress
    
    batch_processor.process_batches(items, mock_process_fn, description="Testing")
    
    # Verify progress bar was updated for each batch
    assert mock_progress.update.call_count == 3  # 25 items / 10 batch_size = 3 batches


def test_process_batches_custom_batch_size(items):
    """Test processing with custom batch size."""
    config = BatchConfig(batch_size=5)
    processor = BatchProcessor(config)
    mock_fn = Mock()
    
    processor.process_batches(items, mock_fn)
    
    assert mock_fn.call_count == 5  # 25 items / 5 batch_size = 5 batches
    for call in mock_fn.call_args_list[:-1]:  # All but last batch
        assert len(call[0][0]) == 5 