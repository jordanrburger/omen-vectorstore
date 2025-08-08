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
import re

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
        
        # If project_id is provided, ensure it's a string and clean it
        if project_id is not None:
            project_id = str(project_id).strip()
            # Check if valid
            if not project_id or project_id.lower() == "unknown" or project_id.lower() == "none":
                logger.warning(f"Provided project ID '{project_id}' appears invalid. Will attempt auto-detection.")
                project_id = None
        
        self.project_id = project_id
        self.project_name = None
        
        # Auto-detect project ID and name from token info if not provided
        if not self.project_id:
            logger.info("Project ID not specified, will auto-detect from token")
            self._detect_project_info()
            
            # Verify that a project ID was detected
            if self.project_id == "unknown":
                logger.warning("Project ID auto-detection failed. Falling back to alternative methods.")
                self._try_alternative_detection_methods()
        else:
            logger.info(f"Using provided project ID: {self.project_id}")
            # Try to get the project name if project ID is provided
            try:
                logger.info(f"Trying to get project name for provided project ID {self.project_id}")
                project_response = requests.get(
                    f"{self.url}/v2/storage/projects/{self.project_id}",
                    headers=self.headers
                )
                if project_response.status_code == 200:
                    project_data = project_response.json()
                    self.project_name = project_data.get("name", None)
                    logger.info(f"Retrieved project name: {self.project_name}")
            except Exception as e:
                logger.warning(f"Error getting project name for provided project ID: {str(e)}")
        
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
        
        # Display a notice about improved project ID detection
        if self.project_id != "unknown":
            logger.info("Project ID detection was successful! The extractor has been updated with improved project ID detection.")
    
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
            
            # Log token info structure for debugging (hiding sensitive parts)
            sanitized_info = {k: ('***' if k in ('token', 'description', 'refreshToken') else v) 
                             for k, v in token_info.items()}
            logger.debug(f"Token info response structure: {list(token_info.keys())}")
            
            # First try the most common paths for project ID
            project_id = None
            project_name = None
            
            # Check if token info contains 'owner' information (standard structure)
            if "owner" in token_info:
                owner_info = token_info["owner"]
                logger.debug(f"Found owner info with keys: {list(owner_info.keys())}")
                
                if "id" in owner_info:
                    project_id = str(owner_info["id"])
                    logger.info(f"Found project ID in 'owner.id': {project_id}")
                    project_name = owner_info.get("name")
            
            # Check alternative possible paths
            if project_id is None:
                # Check if project ID is at root level
                if "id" in token_info:
                    project_id = str(token_info["id"])
                    logger.info(f"Found project ID at root level: {project_id}")
                    project_name = token_info.get("name")
                
                # Check if projectId exists
                elif "projectId" in token_info:
                    project_id = str(token_info["projectId"])
                    logger.info(f"Found project ID in 'projectId': {project_id}")
                    project_name = token_info.get("projectName", token_info.get("name"))
                
                # Check if project exists
                elif "project" in token_info and isinstance(token_info["project"], dict):
                    project_dict = token_info["project"]
                    if "id" in project_dict:
                        project_id = str(project_dict["id"])
                        logger.info(f"Found project ID in 'project.id': {project_id}")
                        project_name = project_dict.get("name")
            
            # If still not found, try to parse from the URL as a last resort
            if project_id is None:
                # Last-ditch effort: check if we can get the project ID from any other field
                logger.warning("Could not find project ID in standard locations in the API response")
                
                # Try any field that might be the project ID
                for key in token_info.keys():
                    if key.lower().endswith('id') and key != 'id' and key != 'componentId':
                        possible_id = str(token_info[key])
                        logger.info(f"Trying alternative field '{key}' with value '{possible_id}' as project ID")
                        project_id = possible_id
                        break
                
                if project_id is None:
                    logger.warning("No project ID found in the API response")
                    project_id = "unknown"
            
            self.project_id = project_id
            self.project_name = project_name
            
            # If project name wasn't in the token info, try to get it from projects endpoint
            if not self.project_name and self.project_id and self.project_id != "unknown":
                try:
                    # Try to get project details
                    logger.info(f"Trying to get project name from project details endpoint")
                    project_response = requests.get(
                        f"{self.url}/v2/storage/projects/{self.project_id}",
                        headers=self.headers
                    )
                    if project_response.status_code == 200:
                        project_data = project_response.json()
                        self.project_name = project_data.get("name", None)
                        logger.info(f"Retrieved project name from projects endpoint: {self.project_name}")
                    else:
                        logger.warning(f"Failed to get project details: HTTP {project_response.status_code}")
                except Exception as e:
                    # Ignore errors in getting project name
                    logger.warning(f"Error getting project details: {str(e)}")
            
            logger.info(f"Auto-detected project ID: {self.project_id}" + (f" ({self.project_name})" if self.project_name else ""))
        except Exception as e:
            logger.warning(f"Could not auto-detect project ID from token: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
            self.project_id = "unknown"
            self.project_name = None

    def _try_alternative_detection_methods(self) -> None:
        """
        Try alternative methods to detect the project ID when the primary method fails.
        This is a fallback mechanism to improve reliability.
        """
        logger.info("Attempting alternative project ID detection methods")
        
        # Method 1: Try to get projects list and use the first one
        try:
            logger.info("Trying to get project ID from projects list")
            response = requests.get(f"{self.url}/v2/storage/projects", headers=self.headers)
            if response.status_code == 200:
                projects = response.json()
                if projects and len(projects) > 0 and "id" in projects[0]:
                    self.project_id = str(projects[0]["id"])
                    self.project_name = projects[0].get("name")
                    logger.info(f"Found project ID from projects list: {self.project_id}")
                    return
        except Exception as e:
            logger.warning(f"Failed to get projects list: {str(e)}")
        
        # Method 2: Try to get token verification and extract project ID
        try:
            logger.info("Trying to get project ID from token verification")
            response = requests.post(f"{self.url}/v2/storage/tokens/verify", headers=self.headers)
            if response.status_code == 200:
                verify_data = response.json()
                if "owner" in verify_data and "id" in verify_data["owner"]:
                    self.project_id = str(verify_data["owner"]["id"])
                    self.project_name = verify_data["owner"].get("name")
                    logger.info(f"Found project ID from token verification: {self.project_id}")
                    return
        except Exception as e:
            logger.warning(f"Failed to verify token: {str(e)}")
        
        # Method 3: Try to extract from URL if it contains project ID pattern
        try:
            logger.info("Trying to extract project ID from URL")
            
            # Look for patterns like: connection.keboola.com/admin/projects/123
            match = re.search(r'/projects/(\d+)', self.url)
            if match:
                self.project_id = match.group(1)
                logger.info(f"Extracted project ID from URL: {self.project_id}")
                return
        except Exception as e:
            logger.warning(f"Failed to extract project ID from URL: {str(e)}")
        
        logger.warning("All alternative methods failed to detect project ID")

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
                                    cd = detail.get("changeDescription")
                                    # Handle both string and dict calmly
                                    if isinstance(cd, str):
                                        # Attempt ISO parse if it looks like a timestamp, else ignore
                                        ts = cd.strip()
                                        if ts and ("-" in ts or ":" in ts):
                                            updated_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                        else:
                                            # Free text, fall back
                                            raise ValueError("changeDescription is free text")
                                    elif isinstance(cd, dict):
                                        time_val = cd.get("time") or cd.get("changedAt") or ""
                                        if time_val:
                                            updated_at = datetime.fromisoformat(str(time_val).replace("Z", "+00:00"))
                                        else:
                                            raise ValueError("No time field present in changeDescription")
                                    else:
                                        raise ValueError("Unsupported changeDescription type")
                                except Exception as e:
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
                            # Extract input/output mapping if available (robust to None/missing)
                            try:
                                cfg = detail.get("configuration")
                                if isinstance(cfg, dict):
                                    storage_config = cfg.get("storage") or {}
                                    # Input mapping
                                    inputs = storage_config.get("input") or []
                                    if isinstance(inputs, list):
                                        for input_table in inputs:
                                            if isinstance(input_table, dict):
                                                src = input_table.get("source")
                                                if src:
                                                    input_tables.append(src)
                                    # Output mapping
                                    outputs = storage_config.get("output") or []
                                    if isinstance(outputs, list):
                                        for output_table in outputs:
                                            if isinstance(output_table, dict):
                                                dst = output_table.get("destination")
                                                if dst:
                                                    output_tables.append(dst)
                            except Exception as e:
                                logger.warning(f"Failed to parse storage mappings for config {config_id}: {e}")
                            
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
                                "relationships": (
                                    ([{"type": "belongs_to", "target_type": "project", "target_id": self.project_id}] if self.project_id else [])
                                    + [{"type": "uses", "target_type": "table", "target_id": table_id} for table_id in input_tables]
                                    + [{"type": "produces", "target_type": "table", "target_id": table_id} for table_id in output_tables]
                                )
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