"""
Utility functions for the OMEN platform.
"""
import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from omen.core.logging import get_logger

logger = get_logger(__name__)


def generate_uuid() -> str:
    """Generate a UUID for use as an identifier."""
    return str(uuid.uuid4())


def generate_hash(data: Union[str, bytes, Dict[str, Any], List[Any]]) -> str:
    """Generate a hash for the given data."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)
    
    if isinstance(data, str):
        data = data.encode("utf-8")
    
    return hashlib.sha256(data).hexdigest()


def timestamp_now() -> float:
    """Get the current timestamp in seconds since epoch."""
    return datetime.now().timestamp()


def format_timestamp(timestamp: float) -> str:
    """Format a timestamp for display."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """Split a list into chunks of size n."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def flatten(lst: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in lst for item in sublist]


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, recursively merging nested dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if (
            key in result 
            and isinstance(result[key], dict) 
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result 