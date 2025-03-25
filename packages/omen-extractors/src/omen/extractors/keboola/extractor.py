"""
Implementation of the Keboola Storage API metadata extractor.
"""

from typing import Dict, List, Any, Optional, Set
import os
import json
from datetime import datetime
import logging

from omen.core import get_logger
from omen.core.models import MetadataItem, MetadataType, MetadataSource

# Only import kbcstorage when needed
try:
    from kbcstorage.client import Client
except ImportError:
    Client = None

logger = get_logger(__name__)


class KeboolaExtractor:
    """
    Extractor for Keboola Storage API metadata.

    Extracts metadata about buckets, tables, configurations and other entities
    from the Keboola Connection platform via the Storage API.
    """

    def __init__(self, token: str, url: str = "https://connection.keboola.com"):
        """
        Initialize the Keboola extractor.

        Args:
            token: Storage API token
            url: Storage API URL, defaults to https://connection.keboola.com
        """
        if Client is None:
            raise ImportError(
                "kbcstorage package is required. Install with 'pip install omen-extractors[keboola]'"
            )

        self.token = token
        self.url = url
        self.client = Client(token, url)
        self.state_file = os.path.expanduser("~/.omen/keboola_state.json")
        
        # Ensure state directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        logger.info(f"Initialized Keboola extractor for {url}")

    def _load_state(self) -> Dict[str, Any]:
        """
        Load state from the state file.

        Returns:
            Dict containing the state
        """
        if not os.path.exists(self.state_file):
            return {"last_run": None, "processed_tables": set(), "processed_buckets": set()}

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                
            # Convert lists back to sets
            if "processed_tables" in state:
                state["processed_tables"] = set(state["processed_tables"])
            if "processed_buckets" in state:
                state["processed_buckets"] = set(state["processed_buckets"])
                
            return state
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return {"last_run": None, "processed_tables": set(), "processed_buckets": set()}

    def _save_state(self, state: Dict[str, Any]) -> None:
        """
        Save state to the state file.

        Args:
            state: Dict containing the state
        """
        try:
            # Convert sets to lists for JSON serialization
            serializable_state = state.copy()
            if "processed_tables" in serializable_state:
                serializable_state["processed_tables"] = list(serializable_state["processed_tables"])
            if "processed_buckets" in serializable_state:
                serializable_state["processed_buckets"] = list(serializable_state["processed_buckets"])
                
            with open(self.state_file, "w") as f:
                json.dump(serializable_state, f)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def extract(self, incremental: bool = True) -> List[MetadataItem]:
        """
        Extract metadata from Keboola.

        Args:
            incremental: If True, only extract metadata that has changed since last run

        Returns:
            List of MetadataItem objects
        """
        logger.info(f"Starting {'incremental' if incremental else 'full'} extraction")
        
        state = self._load_state() if incremental else {
            "last_run": None, 
            "processed_tables": set(), 
            "processed_buckets": set()
        }
        
        # Set current time as last_run for the next state
        current_time = datetime.utcnow().isoformat()
        
        result: List[MetadataItem] = []
        
        # Extract bucket metadata
        buckets = self.extract_buckets(state.get("processed_buckets", set()) if incremental else set())
        result.extend(buckets)
        
        # Extract table metadata
        tables = self.extract_tables(state.get("processed_tables", set()) if incremental else set())
        result.extend(tables)
        
        # TODO: Extract component configurations when needed
        
        # Update state
        state["last_run"] = current_time
        state["processed_buckets"] = set(item.id for item in buckets)
        state["processed_tables"] = set(item.id for item in tables)
        
        self._save_state(state)
        
        logger.info(f"Extraction complete, {len(result)} items extracted")
        return result

    def extract_buckets(self, processed_buckets: Set[str]) -> List[MetadataItem]:
        """
        Extract bucket metadata.

        Args:
            processed_buckets: Set of bucket IDs that have already been processed

        Returns:
            List of MetadataItem objects for buckets
        """
        logger.info("Extracting bucket metadata")
        result = []
        
        buckets = self.client.buckets.list()
        
        for bucket in buckets:
            bucket_id = bucket["id"]
            
            # Skip if already processed in incremental mode
            if bucket_id in processed_buckets and bucket["uri"] in processed_buckets:
                continue
                
            # Get bucket detail
            try:
                detail = self.client.buckets.detail(bucket_id)
                metadata = {
                    "id": bucket_id,
                    "name": bucket["name"],
                    "stage": bucket["stage"],
                    "description": bucket.get("description", ""),
                    "created": bucket["created"],
                    "last_change_date": bucket.get("lastChangeDate"),
                    "attributes": detail.get("attributes", {}),
                    "backend": detail.get("backend"),
                    "sharing": detail.get("sharing"),
                    "uri": bucket["uri"],
                }
                
                # Create metadata item
                item = MetadataItem(
                    id=bucket_id,
                    name=bucket["name"],
                    content=json.dumps(metadata),
                    type=MetadataType.BUCKET,
                    source=MetadataSource(
                        type=MetadataType.BUCKET,
                        url=bucket["uri"],
                        created=bucket["created"],
                        updated=bucket.get("lastChangeDate"),
                    ),
                    attributes={
                        "stage": bucket["stage"],
                        "backend": detail.get("backend"),
                        "sharing": detail.get("sharing"),
                    },
                )
                
                result.append(item)
            except Exception as e:
                logger.error(f"Failed to extract bucket {bucket_id}: {e}")
        
        logger.info(f"Extracted {len(result)} buckets")
        return result

    def extract_tables(self, processed_tables: Set[str]) -> List[MetadataItem]:
        """
        Extract table metadata.

        Args:
            processed_tables: Set of table IDs that have already been processed

        Returns:
            List of MetadataItem objects for tables
        """
        logger.info("Extracting table metadata")
        result = []
        
        tables = self.client.tables.list()
        
        for table in tables:
            table_id = table["id"]
            
            # Skip if already processed in incremental mode
            if table_id in processed_tables and table["uri"] in processed_tables:
                continue
                
            # Get table detail
            try:
                # Get complete table detail using the client method
                detail = self.client.tables.detail(table_id)
                
                # Extract column metadata
                columns = []
                for col_name, col_info in detail.get("columnMetadata", {}).items():
                    column = {
                        "name": col_name,
                        "type": col_info.get("type"),
                        "nullable": col_info.get("nullable"),
                        "length": col_info.get("length"),
                        "default": col_info.get("default"),
                        "format": col_info.get("format"),
                        "description": col_info.get("description", ""),
                        "basetype": col_info.get("basetype"),
                    }
                    columns.append(column)
                
                # Create structured metadata
                metadata = {
                    "id": table_id,
                    "name": table["name"],
                    "bucket": {
                        "id": table["bucket"]["id"],
                        "name": table["bucket"]["name"],
                        "stage": table["bucket"]["stage"],
                        "uri": table["bucket"]["uri"],
                    },
                    "primaryKey": detail.get("primaryKey", []),
                    "created": table["created"],
                    "lastImportDate": table.get("lastImportDate"),
                    "lastChangeDate": table.get("lastChangeDate"),
                    "rowsCount": detail.get("rowsCount", 0),
                    "dataSizeBytes": detail.get("dataSizeBytes", 0),
                    "columns": columns,
                    "attributes": detail.get("attributes", {}),
                    "uri": table["uri"],
                    "definition": {
                        "backend": detail.get("definition", {}).get("backend", ""),
                        "dataStorage": detail.get("dataStorage", {}).get("backend", ""),
                        "type": detail.get("definition", {}).get("type", ""),
                    },
                }
                
                # Create metadata item
                item = MetadataItem(
                    id=table_id,
                    name=table["name"],
                    content=json.dumps(metadata),
                    type=MetadataType.TABLE,
                    source=MetadataSource(
                        type=MetadataType.TABLE,
                        url=table["uri"],
                        created=table["created"],
                        updated=table.get("lastChangeDate") or table.get("lastImportDate"),
                    ),
                    attributes={
                        "bucket_id": table["bucket"]["id"],
                        "bucket_name": table["bucket"]["name"],
                        "rows_count": str(detail.get("rowsCount", 0)),
                        "columns_count": str(len(columns)),
                        "data_size_bytes": str(detail.get("dataSizeBytes", 0)),
                    },
                    parent_id=table["bucket"]["id"],
                )
                
                result.append(item)
            except Exception as e:
                logger.error(f"Failed to extract table {table_id}: {e}")
        
        logger.info(f"Extracted {len(result)} tables")
        return result 