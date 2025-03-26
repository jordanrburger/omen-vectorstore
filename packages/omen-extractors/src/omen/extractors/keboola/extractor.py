"""
Implementation of the Keboola Storage API metadata extractor.
"""

from typing import Dict, List, Any, Optional, Set
import os
import json
from datetime import datetime, timezone
import logging
import requests
import uuid

from omen.core import get_logger
from omen.vectorstore.models import MetadataDocument, MetadataType, MetadataSource

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
        # Remove trailing slash from URL
        self.url = url.rstrip("/")
        self.headers = {"X-StorageApi-Token": self.token}
        self.state_file = os.path.expanduser("~/.omen/keboola_state.json")
        
        # Ensure state directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        logger.info(f"Initialized Keboola extractor for {self.url}")

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
        """Save state to the state file.
        
        Args:
            state: Dict containing the state
        """
        try:
            # Convert sets to lists for JSON serialization
            serializable_state = state.copy()
            if "processed_tables" in serializable_state:
                serializable_state["processed_tables"] = sorted(list(serializable_state["processed_tables"]))
            if "processed_buckets" in serializable_state:
                serializable_state["processed_buckets"] = sorted(list(serializable_state["processed_buckets"]))
            
            with open(self.state_file, "w") as f:
                json.dump(serializable_state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def extract(self, incremental: bool = False) -> List[MetadataDocument]:
        """Extract metadata from Keboola Storage API.
        
        Args:
            incremental: If True, only extract metadata for items that have changed since last run.
            
        Returns:
            List of metadata documents.
        """
        state = self._load_state() if incremental else {
            "last_run": None,
            "processed_tables": set(),
            "processed_buckets": set(),
        }
        
        documents = []
        
        # Extract bucket metadata
        buckets = self.extract_buckets(state["processed_buckets"])
        documents.extend(buckets)
        
        # Extract table metadata
        tables = self.extract_tables(state["processed_tables"])
        documents.extend(tables)
        
        # Update state
        if incremental:
            state["last_run"] = datetime.now(timezone.utc).isoformat()
            state["processed_buckets"].update(bucket.source.id for bucket in buckets)
            state["processed_tables"].update(table.source.id for table in tables)
            self._save_state(state)
        
        return documents

    def extract_buckets(self, processed_buckets: Set[str]) -> List[MetadataDocument]:
        """Extract metadata for all buckets.
        
        Args:
            processed_buckets: Set of bucket IDs that have already been processed.
            
        Returns:
            List of metadata documents for buckets.
        """
        documents = []
        try:
            response = requests.get(f"{self.url}/v2/storage/buckets", headers=self.headers)
            response.raise_for_status()
            buckets = response.json()
            
            for bucket in buckets:
                bucket_id = bucket["id"]
                if bucket_id in processed_buckets:
                    continue
                    
                try:
                    detail_response = requests.get(
                        f"{self.url}/v2/storage/buckets/{bucket_id}",
                        headers=self.headers
                    )
                    detail_response.raise_for_status()
                    detail = detail_response.json()
                    
                    documents.append(MetadataDocument(
                        id=str(uuid.uuid4()),
                        source=MetadataSource(
                            id=bucket_id,
                            type=MetadataType.BUCKET,
                            url=bucket["uri"],
                            updated_at=datetime.fromisoformat(bucket["lastChangeDate"].replace("Z", "+00:00")),
                            created_at=datetime.fromisoformat(bucket["created"].replace("Z", "+00:00")),
                        ),
                        content=json.dumps(detail),
                        metadata={
                            "stage": bucket["stage"],
                            "backend": detail.get("backend"),
                            "sharing": detail.get("sharing", {}),
                        }
                    ))
                except Exception as e:
                    logger.error(f"Failed to extract bucket {bucket_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to extract buckets: {e}")
            
        return documents

    def _extract_column_metadata(self, column_metadata: Any) -> List[Dict[str, Any]]:
        """
        Extract column metadata from various response formats.

        Args:
            column_metadata: Column metadata from the API response

        Returns:
            List of column metadata dictionaries
        """
        columns = []
        
        if isinstance(column_metadata, list):
            # Handle list format
            for col in column_metadata:
                if isinstance(col, dict):
                    column = {
                        "name": col.get("name", ""),
                        "type": col.get("type"),
                        "nullable": col.get("nullable"),
                        "length": col.get("length"),
                        "default": col.get("default"),
                        "format": col.get("format"),
                        "description": col.get("description", ""),
                        "basetype": col.get("basetype"),
                    }
                    columns.append(column)
        elif isinstance(column_metadata, dict):
            # Handle dictionary format
            for col_name, col_info in column_metadata.items():
                if isinstance(col_info, dict):
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
        
        return columns

    def extract_tables(self, processed_tables: Set[str]) -> List[MetadataDocument]:
        """Extract metadata for all tables.
        
        Args:
            processed_tables: Set of table IDs that have already been processed.
            
        Returns:
            List of metadata documents for tables.
        """
        documents = []
        try:
            response = requests.get(f"{self.url}/v2/storage/tables", headers=self.headers)
            response.raise_for_status()
            tables = response.json()
            
            for table in tables:
                table_id = table["id"]
                if table_id in processed_tables:
                    continue
                    
                try:
                    detail_response = requests.get(
                        f"{self.url}/v2/storage/tables/{table_id}",
                        headers=self.headers
                    )
                    detail_response.raise_for_status()
                    detail = detail_response.json()
                    
                    # Handle both list and dictionary responses
                    if isinstance(detail, list):
                        if not detail:
                            logger.warning(f"Empty detail response for table {table_id}")
                            continue
                        detail = detail[0]
                    elif not isinstance(detail, dict):
                        logger.warning(f"Unexpected response type for table {table_id}: {type(detail)}")
                        continue
                    
                    # Extract column metadata
                    columns = []
                    column_metadata = detail.get("columnMetadata", {})
                    if isinstance(column_metadata, dict):
                        for col_name, col_info in column_metadata.items():
                            columns.append({
                                "name": col_name,
                                "type": col_info.get("type"),
                                "nullable": col_info.get("nullable"),
                                "length": col_info.get("length"),
                                "default": col_info.get("default"),
                                "format": col_info.get("format"),
                                "description": col_info.get("description"),
                                "basetype": col_info.get("basetype"),
                            })
                    elif isinstance(column_metadata, list):
                        for col_info in column_metadata:
                            columns.append({
                                "name": col_info.get("name"),
                                "type": col_info.get("type"),
                                "nullable": col_info.get("nullable"),
                                "length": col_info.get("length"),
                                "default": col_info.get("default"),
                                "format": col_info.get("format"),
                                "description": col_info.get("description"),
                                "basetype": col_info.get("basetype"),
                            })
                    
                    # Create table document
                    documents.append(MetadataDocument(
                        id=str(uuid.uuid4()),
                        source=MetadataSource(
                            id=table_id,
                            type=MetadataType.TABLE,
                            url=table["uri"],
                            updated_at=datetime.fromisoformat(table["lastChangeDate"].replace("Z", "+00:00")),
                            created_at=datetime.fromisoformat(table["created"].replace("Z", "+00:00")),
                        ),
                        content=json.dumps({
                            **detail,
                            "columns": columns,
                        }),
                        metadata={
                            "bucket_id": table["bucket"]["id"],
                            "bucket_name": table["bucket"]["name"],
                            "rows_count": str(detail.get("rowsCount", 0)),
                            "columns_count": str(len(columns)),
                            "data_size_bytes": str(detail.get("dataSizeBytes", 0)),
                        }
                    ))
                except Exception as e:
                    logger.error(f"Failed to extract table {table_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to extract tables: {e}")
            
        return documents 