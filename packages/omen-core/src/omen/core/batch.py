"""
Batch processing utilities for OMEN platform.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from tqdm import tqdm

from omen.core.config import settings
from omen.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """Process items in batches, with retry support and progress tracking."""

    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 1.0,
    ):
        """Initialize the batch processor.
        
        Args:
            batch_size: Size of each batch. If None, uses value from settings.
            max_workers: Maximum number of worker threads. If None, uses value from settings.
            max_retries: Maximum number of retries for each item. If None, uses value from settings.
            retry_delay: Delay between retries in seconds.
        """
        self.batch_size = batch_size or settings.batch_size
        self.max_workers = max_workers or settings.max_workers
        self.max_retries = max_retries or settings.max_retries
        self.retry_delay = retry_delay

    def process(
        self,
        items: List[T],
        process_fn: Callable[[T], R],
        desc: str = "Processing",
        show_progress: bool = True,
        parallel: bool = True,
    ) -> List[Tuple[T, Optional[R], Optional[Exception]]]:
        """Process items in batches, optionally in parallel.
        
        Args:
            items: List of items to process
            process_fn: Function to process each item
            desc: Description for progress bar
            show_progress: Whether to show a progress bar
            parallel: Whether to process in parallel
            
        Returns:
            List of tuples with (item, result, exception)
        """
        if not items:
            logger.warning("No items to process")
            return []

        results: List[Tuple[T, Optional[R], Optional[Exception]]] = []
        
        # Create batches
        batches = [
            items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)
        ]
        
        # Setup progress bar
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=len(items), desc=desc)

        # Process each batch
        for batch in batches:
            batch_results = self._process_batch(
                batch, process_fn, progress_bar, parallel
            )
            results.extend(batch_results)

        if progress_bar:
            progress_bar.close()

        # Log summary
        success_count = sum(1 for _, result, error in results if error is None)
        logger.info(
            f"Processed {len(items)} items: {success_count} succeeded, "
            f"{len(items) - success_count} failed"
        )

        return results

    def _process_batch(
        self,
        batch: List[T],
        process_fn: Callable[[T], R],
        progress_bar: Optional[tqdm] = None,
        parallel: bool = True,
    ) -> List[Tuple[T, Optional[R], Optional[Exception]]]:
        """Process a single batch of items.
        
        Args:
            batch: Batch of items to process
            process_fn: Function to process each item
            progress_bar: Optional progress bar to update
            parallel: Whether to process in parallel
            
        Returns:
            List of tuples with (item, result, exception)
        """
        results: List[Tuple[T, Optional[R], Optional[Exception]]] = []

        if parallel and self.max_workers > 1 and len(batch) > 1:
            # Process in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_item = {
                    executor.submit(self._process_item, item, process_fn): item
                    for item in batch
                }
                
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result, error = future.result()
                        results.append((item, result, error))
                    except Exception as exc:
                        logger.error(f"Worker thread failed: {exc}")
                        results.append((item, None, exc))
                    
                    if progress_bar:
                        progress_bar.update(1)
        else:
            # Process sequentially
            for item in batch:
                result, error = self._process_item(item, process_fn)
                results.append((item, result, error))
                
                if progress_bar:
                    progress_bar.update(1)

        return results

    def _process_item(
        self, item: T, process_fn: Callable[[T], R]
    ) -> Tuple[Optional[R], Optional[Exception]]:
        """Process a single item with retry support.
        
        Args:
            item: Item to process
            process_fn: Function to process the item
            
        Returns:
            Tuple of (result, exception)
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                result = process_fn(item)
                return result, None
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.max_retries:
                    logger.warning(
                        f"Error processing item (attempt {retries}/{self.max_retries}): {e}"
                    )
                    time.sleep(self.retry_delay * retries)  # Exponential backoff
                else:
                    logger.error(f"Failed to process item after {retries} attempts: {e}")
                    return None, last_error

        return None, last_error


# Default batch processor instance
batch_processor = BatchProcessor() 