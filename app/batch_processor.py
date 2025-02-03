"""Batch processing functionality."""
import logging
import time
from dataclasses import dataclass
from typing import List, Callable, Any, Optional

from tqdm import tqdm

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    batch_size: int = 100
    max_retries: int = 3
    initial_retry_delay: float = 1.0

class BatchProcessor:
    """Handles batch processing with retries."""
    
    def __init__(self, config: BatchConfig):
        """Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config
        
    def process_batches(
        self,
        items: List[Any],
        process_fn: Callable[[List[Any]], None],
        batch_size: Optional[int] = None,
    ) -> None:
        """Process items in batches with retries.
        
        Args:
            items: List of items to process
            process_fn: Function to process a batch of items
            batch_size: Optional override for batch size
        """
        if not items:
            return
            
        # Use provided batch size or default from config
        batch_size = batch_size or self.config.batch_size
        
        # Process items in batches with progress bar
        with tqdm(total=len(items), desc="Processing items") as pbar:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                self._process_batch_with_retries(batch, process_fn)
                pbar.update(len(batch))
                
    def _process_batch_with_retries(
        self,
        batch: List[Any],
        process_fn: Callable[[List[Any]], None],
    ) -> None:
        """Process a batch with retries on failure.
        
        Args:
            batch: Batch of items to process
            process_fn: Function to process the batch
        """
        retry_count = 0
        delay = self.config.initial_retry_delay
        
        while True:
            try:
                process_fn(batch)
                break
            except Exception as e:
                retry_count += 1
                if retry_count > self.config.max_retries:
                    logging.error(
                        f"Failed to process batch after {retry_count} retries: {e}"
                    )
                    raise
                    
                logging.warning(
                    f"Error processing batch (attempt {retry_count}): {e}"
                )
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff 