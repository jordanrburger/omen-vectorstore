import msgpack
import zlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

class StateManager:
    def __init__(self, state_dir: str = "state", shard_size: int = 1000):
        """Initialize the state manager with a directory for state files.
        
        Args:
            state_dir: Directory to store state files
            shard_size: Number of items per shard for large collections
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.shard_size = shard_size
        
        # Create subdirectories for different types of data
        self.metadata_dir = self.state_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "extraction_state.msgpack"
        
    def _compress_data(self, data: Any) -> bytes:
        """Compress data using MessagePack and zlib."""
        return zlib.compress(msgpack.packb(data, use_bin_type=True))
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data using zlib and MessagePack."""
        return msgpack.unpackb(zlib.decompress(compressed_data), raw=False)
    
    def _get_shard_path(self, metadata_type: str, shard_num: int) -> Path:
        """Get the path for a specific metadata shard."""
        return self.metadata_dir / f"{metadata_type}_shard_{shard_num}.msgpack"
    
    def save_metadata(self, metadata: Dict) -> None:
        """Save the extracted metadata in sharded files using MessagePack."""
        # Save different types of metadata in separate shards
        for metadata_type, items in metadata.items():
            if isinstance(items, list):
                # Shard list-type metadata
                for i in range(0, len(items), self.shard_size):
                    shard = items[i:i + self.shard_size]
                    shard_num = i // self.shard_size
                    shard_path = self._get_shard_path(metadata_type, shard_num)
                    with open(shard_path, 'wb') as f:
                        f.write(self._compress_data(shard))
            elif isinstance(items, dict):
                # Shard dictionary-type metadata
                items_list = list(items.items())
                for i in range(0, len(items_list), self.shard_size):
                    shard = dict(items_list[i:i + self.shard_size])
                    shard_num = i // self.shard_size
                    shard_path = self._get_shard_path(metadata_type, shard_num)
                    with open(shard_path, 'wb') as f:
                        f.write(self._compress_data(shard))
        
        # Save metadata index
        index = {
            metadata_type: {
                'total_items': len(items) if isinstance(items, list) else len(items.keys()),
                'shard_count': (len(items) + self.shard_size - 1) // self.shard_size 
                    if isinstance(items, (list, dict)) else 1
            }
            for metadata_type, items in metadata.items()
        }
        with open(self.metadata_dir / "index.msgpack", 'wb') as f:
            f.write(self._compress_data(index))
            
    def load_metadata(self) -> Optional[Dict]:
        """Load the previously saved metadata, reading shards as needed."""
        if not (self.metadata_dir / "index.msgpack").exists():
            return None
            
        try:
            # Load metadata index
            with open(self.metadata_dir / "index.msgpack", 'rb') as f:
                index = self._decompress_data(f.read())
            
            metadata = {}
            for metadata_type, info in index.items():
                if info['shard_count'] == 1:
                    # Single shard case
                    shard_path = self._get_shard_path(metadata_type, 0)
                    if shard_path.exists():
                        with open(shard_path, 'rb') as f:
                            metadata[metadata_type] = self._decompress_data(f.read())
                else:
                    # Multiple shards case
                    items = []
                    is_dict = False
                    for shard_num in range(info['shard_count']):
                        shard_path = self._get_shard_path(metadata_type, shard_num)
                        if shard_path.exists():
                            with open(shard_path, 'rb') as f:
                                shard_data = self._decompress_data(f.read())
                                if isinstance(shard_data, dict):
                                    is_dict = True
                                    items.extend(shard_data.items())
                                else:
                                    items.extend(shard_data)
                    
                    metadata[metadata_type] = dict(items) if is_dict else items
            
            return metadata
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            return None
            
    def save_extraction_state(self, state: Dict) -> None:
        """Save the extraction state using MessagePack."""
        state['last_extraction'] = datetime.utcnow().isoformat()
        with open(self.state_file, 'wb') as f:
            f.write(self._compress_data(state))
            
    def load_extraction_state(self) -> Dict:
        """Load the previous extraction state or return empty state."""
        if not self.state_file.exists():
            return {
                'last_extraction': None,
                'bucket_hashes': {},
                'table_hashes': {},
                'config_hashes': {},
                'config_row_hashes': {}
            }
        with open(self.state_file, 'rb') as f:
            return self._decompress_data(f.read())
            
    def compute_hash(self, data: Dict) -> str:
        """Compute a hash of the metadata to detect changes."""
        import hashlib
        # Convert the data to a stable string representation by sorting keys manually
        if isinstance(data, dict):
            # Sort dictionary items and convert to a list of tuples
            items = sorted(data.items(), key=lambda x: str(x[0]))
            # Recursively handle nested dictionaries
            data = [(k, self.compute_hash(v) if isinstance(v, dict) else v) for k, v in items]
        # Convert to bytes using msgpack
        data_str = msgpack.packb(data, use_bin_type=True)
        return hashlib.sha256(data_str).hexdigest() 