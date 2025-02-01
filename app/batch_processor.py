"""Batch processing utilities with retry logic and progress tracking."""

import logging
import time
from typing import TypeVar, Generic, List, Callable, Optional
from dataclasses import dataclass
from tqdm import tqdm

T = TypeVar('T')  # Generic type for batch items

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 10
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    backoff_factor: float = 2.0
    show_progress: bool = True

class BatchProcessor(Generic[T]):
    """Handles batch processing with retries and progress tracking."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize the batch processor with configuration."""
        self.config = config or BatchConfig()
        self.logger = logging.getLogger(__name__)

    def process_batches(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], None],
        description: str = "Processing"
    ) -> None:
        """
        Process items in batches with retry logic and progress tracking.
        
        Args:
            items: List of items to process
            process_fn: Function that processes a batch of items
            description: Description for the progress bar
        """
        total_batches = (len(items) + self.config.batch_size - 1) // self.config.batch_size
        
        with tqdm(total=total_batches, desc=description, disable=not self.config.show_progress) as pbar:
            for i in range(0, len(items), self.config.batch_size):
                batch = items[i:i + self.config.batch_size]
                self._process_batch_with_retry(batch, process_fn)
                pbar.update(1)

    def _process_batch_with_retry(
        self,
        batch: List[T],
        process_fn: Callable[[List[T]], None]
    ) -> None:
        """Process a single batch with retries and exponential backoff."""
        retry_count = 0
        delay = self.config.initial_retry_delay

        while True:
            try:
                process_fn(batch)
                return
            except Exception as e:
                retry_count += 1
                if retry_count > self.config.max_retries:
                    self.logger.error(
                        f"Failed to process batch after {self.config.max_retries} retries: {str(e)}"
                    )
                    raise

                self.logger.warning(
                    f"Batch processing failed (attempt {retry_count}/{self.config.max_retries}): {str(e)}"
                    f"\nRetrying in {delay} seconds..."
                )
                
                time.sleep(delay)
                delay = min(
                    delay * self.config.backoff_factor,
                    self.config.max_retry_delay
                ) 