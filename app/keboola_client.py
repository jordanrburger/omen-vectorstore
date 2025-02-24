import logging
from typing import Dict, Generator, List, Optional

import requests
from kbcstorage.client import Client

from app.state_manager import StateManager


class KeboolaClient:
    """Client for interacting with the Keboola Storage API.
    
    This client handles metadata extraction, including tables, columns, configurations,
    and transformations. It supports incremental updates by tracking state changes
    and reusing unchanged metadata.
    
    Column Metadata Behavior:
    - When a column's metadata changes (detected by hash comparison), the new metadata
      completely replaces the old metadata
    - The new metadata takes precedence over any existing fields
    - This ensures that the metadata always reflects the current state of the column
      in Keboola Storage
    
    Attributes:
        client: The underlying Keboola Storage API client
        state_manager: Manager for handling state and metadata persistence
        token: The Keboola Storage API token
        url: The Keboola Storage API URL
        session: The requests session for API calls
    """

    def __init__(
        self,
        api_url: str,
        token: str,
        state_manager: Optional[StateManager] = None,
    ):
        """Initialize the Keboola Client with API URL and token."""
        self.client = Client(api_url, token)
        self.state_manager = state_manager or StateManager()
        self.token = token
        self.url = api_url.rstrip("/") + "/v2/storage"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-StorageApi-Token": self.token,
                "User-Agent": "omen-vectorstore/1.0.0",
            }
        )
        logging.info("Initialized KeboolaClient with API URL: %s", api_url)

    def extract_metadata(self, force_full: bool = False) -> Dict:
        """Extract metadata with support for incremental updates."""
        logging.info("Starting metadata extraction (force_full=%s)", force_full)

        # Load previous state and metadata
        state = self.state_manager.load_extraction_state()
        previous_metadata = (
            self.state_manager.load_metadata() if not force_full else None
        )

        # Initialize new state and metadata
        new_state = {
            "bucket_hashes": {},
            "table_hashes": {},
            "config_hashes": {},
            "config_row_hashes": {},
            "column_hashes": {},
        }
        metadata = {
            "buckets": [],
            "tables": {},
            "table_details": {},
            "configurations": {},
            "config_rows": {},
            "columns": {},
        }

        # Extract buckets and their metadata
        for bucket in self.list_buckets_paginated():
            bucket_id = bucket.get("id", "unknown")
            bucket_hash = self.state_manager.compute_hash(bucket)
            new_state["bucket_hashes"][bucket_id] = bucket_hash

            # Check if bucket has changed
            if (
                not force_full
                and previous_metadata
                and state["bucket_hashes"].get(bucket_id) == bucket_hash
            ):
                # Reuse previous bucket metadata
                metadata["buckets"].append(bucket)
                metadata["tables"][bucket_id] = previous_metadata["tables"].get(
                    bucket_id, []
                )
                metadata["table_details"].update(
                    previous_metadata["table_details"].get(bucket_id, {})
                )
                if bucket_id in previous_metadata.get("columns", {}):
                    metadata["columns"][bucket_id] = previous_metadata["columns"][bucket_id]
                continue

            metadata["buckets"].append(bucket)

            # Extract tables for this bucket
            metadata["tables"][bucket_id] = []
            metadata["columns"][bucket_id] = {}
            
            for table in self.list_tables_paginated(bucket_id):
                table_id = table.get("id", "unknown")
                table_hash = self.state_manager.compute_hash(table)
                new_state["table_hashes"][table_id] = table_hash

                # Check if table has changed
                table_changed = (
                    force_full
                    or not previous_metadata
                    or not state.get("table_hashes", {}).get(table_id)  # No previous hash
                    or state["table_hashes"].get(table_id) != table_hash  # Hash changed
                )

                # If table hasn't changed, reuse previous metadata
                if not table_changed:
                    # Reuse previous table metadata
                    metadata["tables"][bucket_id].append(table)
                    if table_id in previous_metadata["table_details"]:
                        metadata["table_details"][table_id] = previous_metadata[
                            "table_details"
                        ][table_id]
                    if table_id in previous_metadata.get("columns", {}).get(bucket_id, {}):
                        metadata["columns"][bucket_id][table_id] = previous_metadata["columns"][bucket_id][table_id]
                    continue

                # Table has changed, process all columns as new
                metadata["tables"][bucket_id].append(table)

                # Get fresh table details and process columns
                details = self.get_table_details(table_id)
                if details:
                    # Extract and process columns
                    if "columns" in details:
                        metadata["columns"][bucket_id][table_id] = []
                        enriched_columns = []
                        for column in details["columns"]:
                            column_id = f"{table_id}.{column.get('name', 'unknown')}"
                            # When table has changed, create fresh metadata
                            enriched_column = {}
                            enriched_column.update(column)  # Apply new fields
                            enriched_column["table_id"] = table_id
                            enriched_column["bucket_id"] = bucket_id
                            metadata["columns"][bucket_id][table_id].append(enriched_column)
                            enriched_columns.append(enriched_column)
                            logging.debug(f"Processed column {column_id} for table {table_id}")

                        # Update table details with enriched columns
                        details = details.copy()
                        details["columns"] = enriched_columns
                    metadata["table_details"][table_id] = details

        # Extract configurations and their rows
        try:
            components = self.client.components.list()
            for component in components:
                component_id = component.get("id")
                metadata["configurations"][component_id] = []

                try:
                    configs = list(self.list_configurations_paginated(component_id))
                    for config in configs:
                        config_id = config.get("id")
                        config_hash = self.state_manager.compute_hash(config)
                        new_state["config_hashes"][config_id] = config_hash

                        # Check if configuration has changed
                        if (
                            not force_full
                            and previous_metadata
                            and state["config_hashes"].get(config_id) == config_hash
                            and component_id
                            in previous_metadata.get("configurations", {})
                        ):
                            # Reuse previous configuration metadata
                            metadata["configurations"][component_id].append(config)
                            if config_id in previous_metadata.get("config_rows", {}):
                                metadata["config_rows"][config_id] = previous_metadata[
                                    "config_rows"
                                ][config_id]
                            continue

                        metadata["configurations"][component_id].append(config)

                        # Get configuration rows if supported
                        try:
                            metadata["config_rows"][config_id] = []
                            for row in self.list_config_rows_paginated(
                                component_id, config_id
                            ):
                                row_id = row.get("id")
                                row_hash = self.state_manager.compute_hash(row)
                                new_state["config_row_hashes"][row_id] = row_hash
                                metadata["config_rows"][config_id].append(row)
                        except Exception as e:
                            logging.debug(
                                f"Config {config_id} doesn't support rows: {e}"
                            )
                except Exception as e:
                    logging.error(
                        "Error fetching configurations for component "
                        f"{component_id}: {e}"
                    )
        except Exception as e:
            logging.error(f"Error fetching components: {e}")

        # Save new state and metadata
        self.state_manager.save_extraction_state(new_state)
        self.state_manager.save_metadata(metadata)

        return metadata

    def list_buckets_paginated(
        self, offset: int = 0, limit: int = 100
    ) -> Generator[Dict, None, None]:
        """Fetch and yield buckets."""
        try:
            buckets = self.client.buckets.list()
            for bucket in buckets:
                yield bucket
        except Exception as e:
            logging.error("Error fetching buckets: %s", e)

    def list_tables_paginated(
        self, bucket_id: str, offset: int = 0, limit: int = 100
    ) -> Generator[Dict, None, None]:
        """Fetch and yield tables."""
        try:
            tables = self.client.buckets.list_tables(bucket_id)
            for table in tables:
                yield table
        except Exception as e:
            logging.error(f"Error fetching tables for bucket {bucket_id}: {e}")

    def list_configurations_paginated(
        self, component_id: str, offset: int = 0, limit: int = 100
    ) -> Generator[Dict, None, None]:
        """Fetch and yield configurations."""
        try:
            configs = self.client.components.list_configs(component_id)
            for config in configs:
                yield config
        except Exception as e:
            logging.error(
                f"Error fetching configurations for component {component_id}: {e}"
            )

    def list_config_rows_paginated(
        self,
        component_id: str,
        config_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Generator[Dict, None, None]:
        """Fetch and yield configuration rows."""
        try:
            rows = self.client.components.list_config_rows(
                component_id,
                config_id,
            )
            for row in rows:
                yield row
        except Exception as e:
            logging.error(
                f"Error fetching configuration rows for config {config_id}: {e}"
            )

    def get_table_details(self, table_id: str) -> Optional[Dict]:
        """Fetch and return the details of a specific table.
        
        For linked tables, this will include both the source and target information.
        The source table details are included in the 'sourceTable' field.
        If the source table is inaccessible (e.g., in a different project), we'll
        still include the basic source table information without the details.
        """
        try:
            details = self.client.tables.detail(table_id)
            logging.info("Fetched details for table %s", table_id)
            
            # Handle linked tables
            if "sourceTable" in details:
                source_table_id = details["sourceTable"]["id"]
                try:
                    # Get source table details
                    source_details = self.client.tables.detail(source_table_id)
                    # Enrich the details with source information
                    details["isLinked"] = True
                    details["sourceTableDetails"] = source_details
                    details["columns"] = source_details.get("columns", [])  # Use source table columns
                    logging.info("Enriched linked table %s with source table %s details", 
                               table_id, source_table_id)
                except Exception as e:
                    # Source table is inaccessible (e.g., in a different project)
                    logging.warning("Could not fetch source table details for %s: %s", 
                                  source_table_id, e)
                    # Still mark as linked and include basic source info
                    details["isLinked"] = True
                    details["sourceTableDetails"] = {
                        "id": source_table_id,
                        "project": details["sourceTable"].get("project", {}),
                        "bucket": details["sourceTable"].get("bucket", {}),
                        "displayName": details["sourceTable"].get("displayName", ""),
                        "isAccessible": False
                    }
                    # Keep any existing columns or use empty list
                    if "columns" not in details:
                        details["columns"] = []
            
            return details
        except Exception as e:
            logging.error("Error fetching table details for table %s: %s", table_id, e)
            return None

    def list_buckets(self, include_deleted: bool = False) -> List[Dict]:
        """List all buckets in the project."""
        try:
            response = self.session.get(f"{self.url}/buckets")
            response.raise_for_status()
            buckets = response.json()
            if not include_deleted:
                buckets = [b for b in buckets if not b.get("isDeleted")]
            return buckets
        except requests.exceptions.RequestException as e:
            logging.error(f"Error listing buckets: {e}")
            raise

    def list_tables(
        self,
        bucket_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> Dict[str, List[Dict]]:
        """List tables in a bucket or all buckets."""
        try:
            # Get all tables
            response = self.session.get(f"{self.url}/tables")
            response.raise_for_status()
            tables = response.json()

            # Filter deleted tables if needed
            if not include_deleted:
                tables = [t for t in tables if not t.get("isDeleted")]

            # Group tables by bucket if no specific bucket requested
            if not bucket_id:
                result = {}
                for table in tables:
                    bucket = table["bucket"]["id"]
                    if bucket not in result:
                        result[bucket] = []
                    result[bucket].append(table)
                return result

            # Filter for specific bucket
            return {bucket_id: [t for t in tables if t["bucket"]["id"] == bucket_id]}

        except requests.exceptions.RequestException as e:
            logging.error(f"Error listing tables: {e}")
            raise

    def list_configurations(
        self,
        component_id: Optional[str] = None,
        include_versions: bool = False,
    ) -> List[Dict]:
        """List configurations, optionally filtered by component."""
        try:
            # Get all components using the correct endpoint
            response = self.session.get(f"{self.url}/components")
            response.raise_for_status()
            components = response.json()
            all_configs = []

            # For each component, get its configurations
            for component in components:
                component_id = component.get("id")
                if not component_id:
                    continue

                try:
                    # Get configurations for this component using the correct endpoint
                    config_response = self.session.get(
                        f"{self.url}/components/{component_id}/configs"
                    )
                    config_response.raise_for_status()
                    configs = config_response.json()
                    
                    # Enrich each config with component info and versions if requested
                    for config in configs:
                        config["componentId"] = component_id
                        config["component"] = component
                        
                        if include_versions:
                            try:
                                config_id = config.get("id")
                                if config_id:
                                    version_response = self.session.get(
                                        f"{self.url}/components/{component_id}/configs/{config_id}/versions"
                                    )
                                    version_response.raise_for_status()
                                    config["versions"] = version_response.json()
                            except Exception as e:
                                logging.warning(
                                    f"Error fetching versions for config {config_id}: {e}"
                                )
                                config["versions"] = []
                        
                        # Get configuration rows if supported
                        try:
                            config_id = config.get("id")
                            if config_id:
                                rows_response = self.session.get(
                                    f"{self.url}/components/{component_id}/configs/{config_id}/rows"
                                )
                                rows_response.raise_for_status()
                                config["rows"] = rows_response.json()
                        except Exception as e:
                            logging.debug(
                                f"Config {config_id} doesn't support rows: {e}"
                            )
                            config["rows"] = []
                            
                        all_configs.append(config)
                        
                except Exception as e:
                    logging.error(
                        f"Error fetching configurations for component {component_id}: {e}"
                    )

            return all_configs

        except Exception as e:
            logging.error(f"Error listing configurations: {e}")
            raise

    def get_transformation_details(self, transformation_id: str) -> dict:
        """
        Fetch detailed metadata for a specific transformation.
        
        Args:
            transformation_id: The ID of the transformation to fetch.
            
        Returns:
            dict: Detailed transformation metadata including code blocks, dependencies, etc.
            
        Raises:
            Exception: If there was an error fetching the transformation details.
        """
        try:
            response = self.client.transformations.get(transformation_id)
            return response
        except Exception as e:
            logging.error("Error fetching transformation details for %s: %s", transformation_id, e)
            raise

    def get_all_transformations(self) -> dict:
        """
        Fetch metadata for all transformations in the project.
        
        Returns:
            dict: Dictionary mapping transformation IDs to their metadata.
        """
        transformations = {}
        try:
            response = self.client.transformations.list()
            
            for transformation in response:
                try:
                    transformation_id = transformation["id"]
                    details = self.get_transformation_details(transformation_id)
                    transformations[transformation_id] = details
                except Exception as e:
                    logging.error("Error processing transformation %s: %s", transformation_id, e)
                    continue
        except Exception as e:
            logging.error("Error fetching transformations: %s", e)
        
        return {"transformations": transformations}
