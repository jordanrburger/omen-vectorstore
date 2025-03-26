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

    def __init__(self, token: str, url: str = "https://connection.keboola.com", project_id: Optional[str] = None):
        """
        Initialize the Keboola extractor.

        Args:
            token: Storage API token
            url: Storage API URL, defaults to https://connection.keboola.com
            project_id: Optional project ID to associate with extracted metadata
                        (auto-detected from token if not provided)
        """
        if Client is None:
            raise ImportError(
                "kbcstorage package is required. Install with 'pip install omen-extractors[keboola]'"
            )

        self.token = token
        # Remove trailing slash from URL
        self.url = url.rstrip("/")
        self.headers = {"X-StorageApi-Token": self.token}
        self.project_id = project_id
        self.project_name = None
        
        # Auto-detect project ID and name from token info if not provided
        if not self.project_id:
            self._detect_project_info()
        
        # Use OMEN_STATE_DIR if available, otherwise use home directory
        state_dir = os.getenv("OMEN_STATE_DIR", os.path.expanduser("~/.omen"))
        
        # Include project ID in state file name to allow multiple projects
        project_suffix = f"_{self.project_id}" if self.project_id else ""
        self.state_file = os.path.join(state_dir, f"keboola_state{project_suffix}.json")
        
        # Ensure state directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        logger.info(f"Initialized Keboola extractor for {self.url}")
        logger.info(f"Working with project ID: {self.project_id}" + (f" ({self.project_name})" if self.project_name else ""))
        logger.info(f"Using state file: {self.state_file}")
    
    def _detect_project_info(self) -> None:
        """
        Auto-detect project ID and name from the token info.
        
        This method makes an API call to get the token information, which contains 
        the project ID that the token belongs to. This ensures that the correct
        project is always tracked regardless of which URL endpoint is used.
        
        Sets self.project_id and self.project_name.
        """
        try:
            logger.info("Auto-detecting project information from token...")
            
            # The Storage API /v2/storage/ endpoint returns token info including project (owner) details
            # This is the definitive source of project ID regardless of which endpoint URL is used
            response = requests.get(f"{self.url}/v2/storage/", headers=self.headers)
            response.raise_for_status()
            token_info = response.json()
            
            # Get project ID from token info - this is the authoritative source
            owner_info = token_info.get("owner", {})
            self.project_id = str(owner_info.get("id", "unknown"))
            
            # Get project name if available
            self.project_name = owner_info.get("name")
            
            # If project name wasn't in the token info, try to get it from projects endpoint
            if not self.project_name and self.project_id and self.project_id != "unknown":
                try:
                    # Try to get project details
                    project_response = requests.get(
                        f"{self.url}/v2/storage/projects/{self.project_id}",
                        headers=self.headers
                    )
                    if project_response.status_code == 200:
                        project_data = project_response.json()
                        self.project_name = project_data.get("name", None)
                except Exception:
                    # Ignore errors in getting project name
                    pass
            
            logger.info(f"Auto-detected project ID: {self.project_id}" + (f" ({self.project_name})" if self.project_name else ""))
        except Exception as e:
            logger.warning(f"Could not auto-detect project ID from token: {e}")
            self.project_id = "unknown"
            self.project_name = None

    def _load_state(self) -> Dict[str, Any]:
        """
        Load state from the state file.

        Returns:
            Dict containing the state
        """
        if not os.path.exists(self.state_file):
            return {"last_run": None, "processed_tables": set(), "processed_buckets": set(), "processed_configs": set()}

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                
            # Convert lists back to sets
            if "processed_tables" in state:
                state["processed_tables"] = set(state["processed_tables"])
            if "processed_buckets" in state:
                state["processed_buckets"] = set(state["processed_buckets"])
            if "processed_configs" in state:
                state["processed_configs"] = set(state["processed_configs"])
                
            return state
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return {"last_run": None, "processed_tables": set(), "processed_buckets": set(), "processed_configs": set()}

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
            if "processed_configs" in serializable_state:
                serializable_state["processed_configs"] = sorted(list(serializable_state["processed_configs"]))
            
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
        project_name_str = f" ({self.project_name})" if self.project_name else ""
        logger.info(f"Starting {'incremental' if incremental else 'full'} extraction for project {self.project_id}{project_name_str}")
        
        state = self._load_state() if incremental else {
            "last_run": None,
            "processed_tables": set(),
            "processed_buckets": set(),
            "processed_configs": set(),
        }
        
        # Initialize new sets to track items processed in this run
        newly_processed_buckets = set()
        newly_processed_tables = set()
        newly_processed_configs = set()
        last_run_timestamp = state.get("last_run")
        
        documents = []
        
        # Extract bucket metadata
        logger.info(f"Extracting buckets (last run: {last_run_timestamp})")
        buckets = self.extract_buckets(state["processed_buckets"] if incremental else set())
        documents.extend(buckets)
        logger.info(f"Extracted {len(buckets)} buckets")
        
        # Extract table metadata
        logger.info(f"Extracting tables (last run: {last_run_timestamp})")
        tables = self.extract_tables(state["processed_tables"] if incremental else set())
        documents.extend(tables)
        logger.info(f"Extracted {len(tables)} tables and their columns")
        
        # Extract configuration metadata
        logger.info(f"Extracting configurations (last run: {last_run_timestamp})")
        configs = self.extract_configurations(state["processed_configs"] if incremental else set())
        documents.extend(configs)
        logger.info(f"Extracted {len(configs)} configurations")
        
        # Update state with newly processed items
        if incremental:
            logger.info("Updating state for incremental extraction")
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Track all items processed in this run
            newly_processed_buckets = {bucket.source.id for bucket in buckets}
            newly_processed_tables = {table.source.id for table in tables if table.source.type == MetadataType.TABLE}
            newly_processed_configs = {config.source.id for config in configs}
            
            # Update sets in state with newly processed items
            state["processed_buckets"].update(newly_processed_buckets)
            state["processed_tables"].update(newly_processed_tables)
            state["processed_configs"].update(newly_processed_configs)
            state["last_run"] = current_time
            
            # Also store project name in state for reference
            state["project_name"] = self.project_name
            
            # Save updated state
            self._save_state(state)
            logger.info(f"Updated state with {len(newly_processed_buckets)} buckets, {len(newly_processed_tables)} tables, and {len(newly_processed_configs)} configurations")
        
        logger.info(f"Extraction complete, {len(documents)} items extracted")
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
                    logger.debug(f"Skipping already processed bucket: {bucket_id}")
                    continue
                    
                try:
                    detail_response = requests.get(
                        f"{self.url}/v2/storage/buckets/{bucket_id}",
                        headers=self.headers
                    )
                    detail_response.raise_for_status()
                    detail = detail_response.json()
                    
                    # Safe date parsing with default values if None
                    updated_at = None
                    created_at = None
                    
                    if bucket.get("lastChangeDate"):
                        try:
                            updated_at = datetime.fromisoformat(bucket["lastChangeDate"].replace("Z", "+00:00"))
                        except (AttributeError, ValueError) as e:
                            logger.warning(f"Failed to parse lastChangeDate for bucket {bucket_id}: {e}")
                            updated_at = datetime.now(timezone.utc)
                    else:
                        updated_at = datetime.now(timezone.utc)
                    
                    if bucket.get("created"):
                        try:
                            created_at = datetime.fromisoformat(bucket["created"].replace("Z", "+00:00"))
                        except (AttributeError, ValueError) as e:
                            logger.warning(f"Failed to parse created date for bucket {bucket_id}: {e}")
                            created_at = datetime.now(timezone.utc)
                    else:
                        created_at = datetime.now(timezone.utc)
                    
                    documents.append(MetadataDocument(
                        id=str(uuid.uuid4()),
                        source=MetadataSource(
                            id=bucket_id,
                            type=MetadataType.BUCKET,
                            url=bucket.get("uri", f"keboola://bucket/{bucket_id}"),
                            updated_at=updated_at,
                            created_at=created_at,
                            project_id=self.project_id,
                        ),
                        content=json.dumps(detail),
                        metadata={
                            "stage": bucket.get("stage", ""),
                            "backend": detail.get("backend", ""),
                            "sharing": detail.get("sharing", {}),
                            "project_id": self.project_id,
                            "project_name": self.project_name,
                            "bucket_name": bucket.get("name", bucket_id),
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
            # New API format: list of columns
            for col in column_metadata:
                if isinstance(col, dict):
                    columns.append(col)
        elif isinstance(column_metadata, dict):
            # Old API format: dict with column names as keys
            for name, details in column_metadata.items():
                if isinstance(details, dict):
                    details['name'] = name
                    columns.append(details)
                else:
                    columns.append({'name': name})
                    
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
                    logger.debug(f"Skipping already processed table: {table_id}")
                    continue
                    
                try:
                    detail_response = requests.get(
                        f"{self.url}/v2/storage/tables/{table_id}",
                        headers=self.headers
                    )
                    detail_response.raise_for_status()
                    detail = detail_response.json()
                    
                    # Get columns metadata
                    columns = self._extract_column_metadata(detail.get("columnMetadata", {}))
                    
                    # Parse dates
                    updated_at = None
                    created_at = None
                    
                    if table.get("lastChangeDate"):
                        try:
                            updated_at = datetime.fromisoformat(table["lastChangeDate"].replace("Z", "+00:00"))
                        except (AttributeError, ValueError) as e:
                            logger.warning(f"Failed to parse lastChangeDate for table {table_id}: {e}")
                            updated_at = datetime.now(timezone.utc)
                    else:
                        updated_at = datetime.now(timezone.utc)
                        
                    if table.get("created"):
                        try:
                            created_at = datetime.fromisoformat(table["created"].replace("Z", "+00:00"))
                        except (AttributeError, ValueError) as e:
                            logger.warning(f"Failed to parse created date for table {table_id}: {e}")
                            created_at = datetime.now(timezone.utc)
                    else:
                        created_at = datetime.now(timezone.utc)
                    
                    # Get bucket ID
                    bucket_id = table.get("bucket", {}).get("id") or table_id.split('.')[0]
                    bucket_name = table.get("bucket", {}).get("name", "")
                    
                    # Create table document
                    table_doc = MetadataDocument(
                        id=str(uuid.uuid4()),
                        source=MetadataSource(
                            id=table_id,
                            type=MetadataType.TABLE,
                            url=table.get("uri", f"keboola://table/{table_id}"),
                            updated_at=updated_at,
                            created_at=created_at,
                            project_id=self.project_id,
                        ),
                        content=json.dumps({
                            "id": table_id,
                            "name": table.get("name", ""),
                            "displayName": table.get("displayName", ""),
                            "columns": columns,
                            "rowsCount": detail.get("rowsCount", 0),
                            "dataSizeBytes": detail.get("dataSizeBytes", 0),
                        }),
                        metadata={
                            "bucket_id": bucket_id,
                            "bucket_name": bucket_name,
                            "rowsCount": detail.get("rowsCount", 0),
                            "dataSizeBytes": detail.get("dataSizeBytes", 0),
                            "columnCount": len(columns),
                            "project_id": self.project_id,
                            "project_name": self.project_name,
                            # Add relationship metadata
                            "relationships": [
                                {"type": "belongs_to", "target_type": "bucket", "target_id": bucket_id}
                            ]
                        }
                    )
                    documents.append(table_doc)
                    
                    # Create a document for each column
                    for column in columns:
                        column_name = column.get("name", "")
                        if not column_name:
                            continue
                            
                        column_id = f"{table_id}.{column_name}"
                        
                        column_doc = MetadataDocument(
                            id=str(uuid.uuid4()),
                            source=MetadataSource(
                                id=column_id,
                                type=MetadataType.COLUMN,
                                url=f"keboola://table/{table_id}/column/{column_name}",
                                updated_at=updated_at,
                                created_at=created_at,
                                project_id=self.project_id,
                            ),
                            content=json.dumps({
                                "id": column_id,
                                "name": column_name,
                                "table": table_id,
                                "definition": column,
                            }),
                            metadata={
                                "table_id": table_id,
                                "table_name": table.get("name", ""),
                                "bucket_id": bucket_id,
                                "bucket_name": bucket_name,
                                "dataType": column.get("type", ""),
                                "nullable": column.get("nullable", True),
                                "project_id": self.project_id,
                                "project_name": self.project_name,
                                # Add relationship metadata
                                "relationships": [
                                    {"type": "belongs_to", "target_type": "table", "target_id": table_id}
                                ]
                            }
                        )
                        documents.append(column_doc)
                        
                except Exception as e:
                    logger.error(f"Failed to extract table {table_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to extract tables: {e}")
            
        return documents
        
    def extract_configurations(self, processed_configs: Set[str]) -> List[MetadataDocument]:
        """Extract metadata for all configurations.
        
        Args:
            processed_configs: Set of configuration IDs that have already been processed.
            
        Returns:
            List of metadata documents for configurations.
        """
        documents = []
        
        try:
            # Get list of components first
            component_response = requests.get(f"{self.url}/v2/storage/components", headers=self.headers)
            component_response.raise_for_status()
            components = component_response.json()
            
            # Process each component
            for component in components:
                component_id = component.get("id")
                if not component_id:
                    continue
                    
                try:
                    # Get configurations for this component
                    config_response = requests.get(
                        f"{self.url}/v2/storage/components/{component_id}/configs", 
                        headers=self.headers
                    )
                    config_response.raise_for_status()
                    configurations = config_response.json()
                    
                    # Process each configuration
                    for config in configurations:
                        config_id = config.get("id")
                        if not config_id or config_id in processed_configs:
                            continue
                            
                        try:
                            # Get configuration details
                            detail_response = requests.get(
                                f"{self.url}/v2/storage/components/{component_id}/configs/{config_id}",
                                headers=self.headers
                            )
                            detail_response.raise_for_status()
                            detail = detail_response.json()
                            
                            # Parse dates
                            updated_at = None
                            created_at = None
                            
                            if detail.get("changeDescription"):
                                try:
                                    updated_at = datetime.fromisoformat(detail["changeDescription"].get("time", "").replace("Z", "+00:00"))
                                except (AttributeError, ValueError, KeyError) as e:
                                    logger.warning(f"Failed to parse changeDescription time for config {config_id}: {e}")
                                    updated_at = datetime.now(timezone.utc)
                            else:
                                updated_at = datetime.now(timezone.utc)
                                
                            if detail.get("created"):
                                try:
                                    created_at = datetime.fromisoformat(detail["created"].replace("Z", "+00:00"))
                                except (AttributeError, ValueError) as e:
                                    logger.warning(f"Failed to parse created date for config {config_id}: {e}")
                                    created_at = datetime.now(timezone.utc)
                            else:
                                created_at = datetime.now(timezone.utc)
                            
                            # Determine tables connected to this configuration
                            input_tables = []
                            output_tables = []
                            
                            # Extract input mapping if available
                            if "configuration" in detail and "storage" in detail["configuration"]:
                                storage_config = detail["configuration"]["storage"]
                                
                                # Input mapping
                                if "input" in storage_config:
                                    for input_table in storage_config["input"]:
                                        if "source" in input_table:
                                            input_tables.append(input_table["source"])
                                
                                # Output mapping
                                if "output" in storage_config:
                                    for output_table in storage_config["output"]:
                                        if "destination" in output_table:
                                            output_tables.append(output_table["destination"])
                            
                            # Create configuration document
                            config_doc = MetadataDocument(
                                id=str(uuid.uuid4()),
                                source=MetadataSource(
                                    id=config_id,
                                    type=MetadataType.CONFIGURATION,
                                    url=f"keboola://components/{component_id}/configs/{config_id}",
                                    updated_at=updated_at,
                                    created_at=created_at,
                                    project_id=self.project_id,
                                ),
                                content=json.dumps({
                                    "id": config_id,
                                    "name": detail.get("name", ""),
                                    "description": detail.get("description", ""),
                                    "component_id": component_id,
                                    "component_name": component.get("name", ""),
                                    "version": detail.get("version", ""),
                                    "input_tables": input_tables,
                                    "output_tables": output_tables,
                                }),
                                metadata={
                                    "name": detail.get("name", ""),
                                    "component_id": component_id,
                                    "component_name": component.get("name", ""),
                                    "project_id": self.project_id,
                                    "project_name": self.project_name,
                                    "configuration_version": detail.get("version", ""),
                                    "is_deleted": detail.get("isDeleted", False),
                                    "creator": detail.get("creator", {}).get("id", ""),
                                    "input_tables": input_tables,
                                    "output_tables": output_tables,
                                    # Add relationship metadata
                                    "relationships": [
                                        {"type": "belongs_to", "target_type": "project", "target_id": self.project_id}
                                    ] + [
                                        {"type": "uses", "target_type": "table", "target_id": table_id}
                                        for table_id in input_tables
                                    ] + [
                                        {"type": "produces", "target_type": "table", "target_id": table_id}
                                        for table_id in output_tables
                                    ]
                                }
                            )
                            documents.append(config_doc)
                            
                        except Exception as e:
                            logger.error(f"Failed to extract config {config_id}: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Failed to extract configs for component {component_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to extract configurations: {e}")
            
        return documents 