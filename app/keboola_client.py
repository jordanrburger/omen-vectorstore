import logging
from typing import Dict, List, Optional, Tuple, Generator
from kbcstorage.client import Client
from app.state_manager import StateManager


class KeboolaClient:
    def __init__(self, api_url: str, token: str, state_manager: Optional[StateManager] = None):
        """Initialize the Keboola Client with API URL and token."""
        self.client = Client(api_url, token)
        self.state_manager = state_manager or StateManager()
        logging.info("Initialized KeboolaClient with API URL: %s", api_url)

    def extract_metadata(self, force_full: bool = False) -> Dict:
        """Extract metadata with support for incremental updates."""
        logging.info("Starting metadata extraction (force_full=%s)", force_full)
        
        # Load previous state and metadata
        state = self.state_manager.load_extraction_state()
        previous_metadata = self.state_manager.load_metadata() if not force_full else None
        
        # Initialize new state and metadata
        new_state = {
            'bucket_hashes': {},
            'table_hashes': {},
            'config_hashes': {},
            'config_row_hashes': {}
        }
        metadata = {
            'buckets': [],
            'tables': {},
            'table_details': {},
            'configurations': {},
            'config_rows': {}
        }
        
        # Extract buckets and their metadata
        for bucket in self.list_buckets_paginated():
            bucket_id = bucket.get('id', 'unknown')
            bucket_hash = self.state_manager.compute_hash(bucket)
            new_state['bucket_hashes'][bucket_id] = bucket_hash
            
            # Check if bucket has changed
            if (not force_full and 
                previous_metadata and 
                state['bucket_hashes'].get(bucket_id) == bucket_hash):
                # Reuse previous bucket metadata
                metadata['buckets'].append(bucket)
                metadata['tables'][bucket_id] = previous_metadata['tables'].get(bucket_id, [])
                metadata['table_details'].update(
                    previous_metadata['table_details'].get(bucket_id, {})
                )
                continue
            
            metadata['buckets'].append(bucket)
            
            # Extract tables for this bucket
            metadata['tables'][bucket_id] = []
            for table in self.list_tables_paginated(bucket_id):
                table_id = table.get('id', 'unknown')
                table_hash = self.state_manager.compute_hash(table)
                new_state['table_hashes'][table_id] = table_hash
                
                # Check if table has changed
                if (not force_full and 
                    previous_metadata and 
                    state['table_hashes'].get(table_id) == table_hash):
                    # Reuse previous table metadata
                    metadata['tables'][bucket_id].append(table)
                    if table_id in previous_metadata['table_details']:
                        metadata['table_details'][table_id] = (
                            previous_metadata['table_details'][table_id]
                        )
                    continue
                    
                metadata['tables'][bucket_id].append(table)
                
                # Get fresh table details
                details = self.get_table_details(table_id)
                if details:
                    metadata['table_details'][table_id] = details

        # Extract configurations and their rows
        try:
            components = self.client.components.list()
            for component in components:
                component_id = component.get('id')
                metadata['configurations'][component_id] = []
                
                try:
                    configs = list(self.list_configurations_paginated(component_id))
                    for config in configs:
                        config_id = config.get('id')
                        config_hash = self.state_manager.compute_hash(config)
                        new_state['config_hashes'][config_id] = config_hash
                        
                        # Check if configuration has changed
                        if (not force_full and 
                            previous_metadata and 
                            state['config_hashes'].get(config_id) == config_hash and
                            component_id in previous_metadata.get('configurations', {})):
                            # Reuse previous configuration metadata
                            metadata['configurations'][component_id].append(config)
                            if config_id in previous_metadata.get('config_rows', {}):
                                metadata['config_rows'][config_id] = (
                                    previous_metadata['config_rows'][config_id]
                                )
                            continue
                        
                        metadata['configurations'][component_id].append(config)
                        
                        # Get configuration rows if supported
                        try:
                            metadata['config_rows'][config_id] = []
                            for row in self.list_config_rows_paginated(component_id, config_id):
                                row_id = row.get('id')
                                row_hash = self.state_manager.compute_hash(row)
                                new_state['config_row_hashes'][row_id] = row_hash
                                metadata['config_rows'][config_id].append(row)
                        except Exception as e:
                            logging.debug(f"Config {config_id} doesn't support rows: {e}")
                except Exception as e:
                    logging.error(f"Error fetching configurations for component {component_id}: {e}")
        except Exception as e:
            logging.error(f"Error fetching components: {e}")
        
        # Save new state and metadata
        self.state_manager.save_extraction_state(new_state)
        self.state_manager.save_metadata(metadata)
        
        return metadata

    def list_buckets_paginated(self, offset: int = 0, limit: int = 100) -> Generator[Dict, None, None]:
        """Fetch and yield buckets."""
        try:
            buckets = self.client.buckets.list()
            for bucket in buckets:
                yield bucket
        except Exception as e:
            logging.error("Error fetching buckets: %s", e)

    def list_tables_paginated(self, bucket_id: str, offset: int = 0, limit: int = 100) -> Generator[Dict, None, None]:
        """Fetch and yield tables."""
        try:
            tables = self.client.buckets.list_tables(bucket_id)
            for table in tables:
                yield table
        except Exception as e:
            logging.error(f"Error fetching tables for bucket {bucket_id}: {e}")

    def list_configurations_paginated(self, component_id: str, offset: int = 0, limit: int = 100) -> Generator[Dict, None, None]:
        """Fetch and yield configurations."""
        try:
            configs = self.client.components.list_configs(component_id)
            for config in configs:
                yield config
        except Exception as e:
            logging.error(f"Error fetching configurations for component {component_id}: {e}")

    def list_config_rows_paginated(self, component_id: str, config_id: str, offset: int = 0, limit: int = 100) -> Generator[Dict, None, None]:
        """Fetch and yield configuration rows."""
        try:
            rows = self.client.components.list_config_rows(component_id, config_id)
            for row in rows:
                yield row
        except Exception as e:
            logging.error(f"Error fetching configuration rows for config {config_id}: {e}")

    def get_table_details(self, table_id: str) -> Optional[Dict]:
        """Fetch and return the details of a specific table."""
        try:
            details = self.client.tables.detail(table_id)
            logging.info("Fetched details for table %s", table_id)
            return details
        except Exception as e:
            logging.error("Error fetching table details for table %s: %s", table_id, e)
            return None
