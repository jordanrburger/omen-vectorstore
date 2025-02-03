import logging
import uuid
import os
from typing import Dict, List, Optional, Callable, Any, Union
from functools import partial
import json
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import PointStruct
from tqdm import tqdm

from app.vectorizer import EmbeddingProvider
from app.batch_processor import BatchProcessor, BatchConfig


class QdrantIndexer:
    """Indexes metadata into Qdrant with vector embeddings."""

    def __init__(
        self,
        tenant_id: str,
        collection_name: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """Initialize QdrantIndexer with tenant and collection configuration.
        
        Args:
            tenant_id: Unique identifier for the tenant
            collection_name: Optional name of the Qdrant collection (defaults to tenant_id_metadata)
            batch_config: Optional batch processing configuration
            host: Optional Qdrant host (defaults to env var QDRANT_HOST or "localhost")
            port: Optional Qdrant port (defaults to env var QDRANT_PORT or 6333)
            api_key: Optional Qdrant API key (defaults to env var QDRANT_API_KEY)
            url: Optional complete Qdrant URL (defaults to env var QDRANT_URL)
                If provided, overrides host and port settings
        """
        self.tenant_id = tenant_id
        self.collection_name = collection_name or f"{tenant_id}_metadata"
        self.vector_size = 1536  # OpenAI ada-002 embedding size
        self.batch_config = batch_config or BatchConfig(
            batch_size=100,
            max_retries=3,
            initial_retry_delay=1.0
        )
        
        # Initialize client based on provided configuration or environment variables
        if url or os.getenv("QDRANT_URL"):
            self.client = QdrantClient(
                url=url or os.getenv("QDRANT_URL"),
                api_key=api_key or os.getenv("QDRANT_API_KEY"),
                timeout=30  # Increased timeout for batch operations
            )
        else:
            self.client = QdrantClient(
                host=host or os.getenv("QDRANT_HOST", "localhost"),
                port=port or int(os.getenv("QDRANT_PORT", "6333")),
                api_key=api_key or os.getenv("QDRANT_API_KEY"),
                timeout=30  # Increased timeout for batch operations
            )
            
        # Initialize batch processor
        self.batch_processor = BatchProcessor(self.batch_config)
        
        # Ensure collection exists with correct schema
        self._ensure_collection()
        
    def _ensure_collection(self):
        """Ensure collection exists with correct schema."""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logging.info(f"Created collection {self.collection_name}")
                
        except Exception as e:
            logging.error(f"Error ensuring collection exists: {e}")
            raise
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(UnexpectedResponse)
    )
    def _upsert_batch(self, points: List[PointStruct]):
        """Upsert a batch of points with retry logic.
        
        Args:
            points: List of points to upsert
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # Ensure consistency
            )
        except Exception as e:
            logging.error(f"Error upserting batch: {e}")
            raise
            
    def index_metadata(self, metadata: Dict, embedding_provider: EmbeddingProvider):
        """Index metadata with vector embeddings.
        
        Args:
            metadata: Metadata dictionary to index
            embedding_provider: Provider for generating embeddings
        """
        try:
            # Process all items in parallel batches
            points = []
            
            # Process buckets
            logging.info("Processing buckets...")
            for bucket in metadata["buckets"]:
                point = self._create_point(
                    text=f"Bucket: {bucket.get('name', 'unknown')}",
                    metadata_type="bucket",
                    metadata=bucket,
                    embedding_provider=embedding_provider
                )
                points.append(point)
                
            # Process tables and their details
            logging.info("Processing tables and columns...")
            for bucket_id, tables in metadata["tables"].items():
                for table in tables:
                    table_id = table.get("id")
                    if table_id in metadata["table_details"]:
                        table_detail = metadata["table_details"][table_id]
                        
                        # Create table point with enriched metadata
                        point = self._create_point(
                            text=self._prepare_table_text(table, table_detail),
                            metadata_type="table",
                            metadata={**table, "details": table_detail},
                            embedding_provider=embedding_provider
                        )
                        points.append(point)
                        
                        # Process columns with enriched metadata
                        if table_id in metadata.get("columns", {}).get(bucket_id, {}):
                            for column in metadata["columns"][bucket_id][table_id]:
                                point = self._create_point(
                                    text=self._prepare_column_text(column, table_id, table_detail),
                                    metadata_type="column",
                                    metadata={**column, "table_id": table_id},
                                    embedding_provider=embedding_provider
                                )
                                points.append(point)
                                
            # Process configurations with enriched metadata
            logging.info("Processing configurations...")
            for component_id, configs in metadata["configurations"].items():
                for config in configs:
                    config_id = config.get("id")
                    point = self._create_point(
                        text=self._prepare_config_text(config, component_id),
                        metadata_type="configuration",
                        metadata={**config, "component_id": component_id},
                        embedding_provider=embedding_provider
                    )
                    points.append(point)
                    
                    # Process config rows
                    if config_id in metadata.get("config_rows", {}):
                        for row in metadata["config_rows"][config_id]:
                            point = self._create_point(
                                text=self._prepare_config_row_text(row, config_id, config),
                                metadata_type="config_row",
                                metadata={**row, "config_id": config_id},
                                embedding_provider=embedding_provider
                            )
                            points.append(point)
            
            # Process relationships
            logging.info("Processing relationships...")
            if "relationships" in metadata:
                for rel_type, relationships in metadata["relationships"].items():
                    for rel in relationships:
                        point = self._create_point(
                            text=self._prepare_relationship_text(rel, rel_type),
                            metadata_type=f"relationship_{rel_type}",
                            metadata={**rel, "relationship_type": rel_type},
                            embedding_provider=embedding_provider
                        )
                        points.append(point)
            
            # Process points in optimized batches
            logging.info(f"Processing {len(points)} points in batches...")
            self.batch_processor.process_batches(
                points,
                self._upsert_batch,
                batch_size=self.batch_config.batch_size
            )
            logging.info("Metadata indexing completed successfully")
            
        except Exception as e:
            logging.error(f"Error indexing metadata: {e}")
            raise
            
    def _create_point(
        self,
        text: str,
        metadata_type: str,
        metadata: Dict,
        embedding_provider: EmbeddingProvider
    ) -> PointStruct:
        """Create a point for indexing.
        
        Args:
            text: Text to embed
            metadata_type: Type of metadata
            metadata: Metadata dictionary
            embedding_provider: Provider for generating embeddings
            
        Returns:
            Point structure for indexing
        """
        try:
            # Enrich text with metadata context
            enriched_text = self._enrich_text_for_embedding(text, metadata_type, metadata)
            
            # Generate embedding
            embedding = embedding_provider.embed([enriched_text])[0]
            
            # Validate embedding dimensions
            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Embedding dimension mismatch: got {len(embedding)}, "
                    f"expected {self.vector_size}"
                )
                
            # Create point
            return PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": enriched_text,
                    "metadata_type": metadata_type,
                    "raw_metadata": metadata,  # Store original metadata
                    "tenant_id": self.tenant_id,
                    "indexed_at": datetime.datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logging.error(f"Error creating point: {e}")
            raise

    def _enrich_text_for_embedding(self, text: str, metadata_type: str, metadata: Dict) -> str:
        """Enrich text with metadata context for better embeddings.
        
        Args:
            text: Base text to enrich
            metadata_type: Type of metadata
            metadata: Metadata dictionary
            
        Returns:
            Enriched text for embedding
        """
        parts = [text]
        
        # Add type-specific context
        if metadata_type == "bucket":
            parts.extend([
                f"Description: {metadata.get('description', 'No description')}",
                f"Stage: {metadata.get('stage', 'unknown')}",
                f"Backend: {metadata.get('backend', 'unknown')}"
            ])
            
        elif metadata_type == "table":
            details = metadata.get("details", {})
            parts.extend([
                f"Description: {metadata.get('description', 'No description')}",
                f"Primary Key: {', '.join(details.get('primaryKey', []))}",
                f"Row Count: {details.get('statistics', {}).get('row_count', 0)}",
                f"Last Changed: {details.get('statistics', {}).get('last_change_date', 'unknown')}",
                "Columns: " + ", ".join(col.get("name", "") for col in details.get("columns", []))
            ])
            
        elif metadata_type == "column":
            stats = metadata.get("statistics", {})
            parts.extend([
                f"Description: {metadata.get('description', 'No description')}",
                f"Data Type: {metadata.get('type', 'unknown')}",
                f"Format: {metadata.get('format', 'unknown')}",
                f"Null Count: {stats.get('null_count', 0)}",
                f"Unique Count: {stats.get('unique_count', 0)}",
                f"Constraints: {', '.join(stats.get('constraints', []))}"
            ])
            
        elif metadata_type == "configuration":
            parts.extend([
                f"Description: {metadata.get('description', 'No description')}",
                f"Component: {metadata.get('component_id', 'unknown')}",
                f"Version: {metadata.get('version', 'unknown')}",
                f"Created By: {metadata.get('createdBy', {}).get('name', 'unknown')}",
                f"State: {metadata.get('state', 'unknown')}"
            ])
            
        elif metadata_type == "config_row":
            parts.extend([
                f"Description: {metadata.get('description', 'No description')}",
                f"Configuration ID: {metadata.get('config_id', 'unknown')}",
                f"Name: {metadata.get('name', 'unknown')}",
                f"State: {metadata.get('state', 'unknown')}"
            ])
            
        # Add any tags
        if "tags" in metadata:
            parts.append(f"Tags: {', '.join(metadata['tags'])}")
            
        # Add creation and modification info
        parts.extend([
            f"Created: {metadata.get('created', 'unknown')}",
            f"Last Modified: {metadata.get('lastModified', 'unknown')}"
        ])
        
        return "\n".join(part for part in parts if part and "unknown" not in part)

    def delete_tenant_collection(self) -> None:
        """Delete the entire collection for this tenant."""
        try:
            self.client.delete_collection(self.collection_name)
        except UnexpectedResponse as e:
            if "not found" not in str(e).lower():
                raise

    @classmethod
    def list_tenant_collections(cls, client: QdrantClient) -> List[Dict[str, Any]]:
        """List all tenant collections and their metadata."""
        collections = client.get_collections()
        tenant_collections = []
        for collection in collections.collections:
            if collection.metadata and "tenant_id" in collection.metadata:
                tenant_collections.append({
                    "tenant_id": collection.metadata["tenant_id"],
                    "collection_name": collection.name,
                    "created_at": collection.metadata.get("created_at"),
                    "vectors_count": collection.vectors_count,
                })
        return tenant_collections

    def search_metadata(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        metadata_type: Optional[str] = None,
        component_type: Optional[str] = None,
        table_id: Optional[str] = None,
        stage: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Search metadata using semantic search with enhanced filtering."""
        # Get query embedding
        query_embedding = embedding_provider.embed([query])[0]

        # Build filter conditions
        must_conditions = []
        
        if metadata_type:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata_type",
                    match=models.MatchValue(value=metadata_type)
                )
            )
        
        if component_type:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.component.type",
                    match=models.MatchValue(value=component_type)
                )
            )
        
        if table_id:
            must_conditions.append(
                models.FieldCondition(
                    key="table_id" if metadata_type == "columns" else "metadata.id",
                    match=models.MatchValue(value=table_id)
                )
            )
        
        if stage:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.stage",
                    match=models.MatchValue(value=stage)
                )
            )

        # Create filter if any conditions exist
        search_filter = None
        if must_conditions:
            search_filter = models.Filter(
                must=must_conditions
            )

        # Perform search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter
        )

        # Format results
        results = []
        for hit in search_results:
            result = {
                "score": hit.score,
                "metadata_type": hit.payload.get("metadata_type"),
                "metadata": hit.payload.get("raw_metadata", {}),
            }
            # Add additional metadata if present
            if "transformation_id" in hit.payload:
                result["transformation_id"] = hit.payload["transformation_id"]
            if "table_id" in hit.payload:
                result["table_id"] = hit.payload["table_id"]
            results.append(result)

        return results

    def find_table_columns(
        self,
        table_id: str,
        query: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Find all columns for a specific table, optionally filtered by a search query."""
        if query and not embedding_provider:
            raise ValueError("embedding_provider is required when query is provided")

        if query:
            return self.search_metadata(
                query=query,
                embedding_provider=embedding_provider,
                metadata_type="columns",
                table_id=table_id,
                limit=limit,
            )
        else:
            # Direct lookup without semantic search
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata_type",
                            match=models.MatchValue(value="columns"),
                        ),
                        models.FieldCondition(
                            key="table_id",
                            match=models.MatchValue(value=table_id),
                        ),
                    ]
                ),
                limit=limit,
            )[0]

            return [
                {
                    "metadata_type": "columns",
                    "metadata": point.payload["raw_metadata"],
                    "table_id": point.payload["table_id"],
                }
                for point in results
            ]

    def find_similar_columns(
        self,
        column_name: str,
        table_id: str,
        embedding_provider: EmbeddingProvider,
        limit: int = 10,
    ) -> List[Dict]:
        """Find columns similar to the specified column across all tables."""
        # First, find the specified column in the table
        base_column = None
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata_type",
                    match=models.MatchValue(value="columns"),
                ),
                models.FieldCondition(
                    key="raw_metadata.name",
                    match=models.MatchValue(value=column_name),
                ),
                models.FieldCondition(
                    key="table_id",
                    match=models.MatchValue(value=table_id),
                ),
            ]
        )
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=1,
        )
        
        if not results[0]:
            raise ValueError(f"Column {column_name} not found in table {table_id}")
            
        base_column = results[0][0]
        
        # Now search for similar columns using the embedding
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=base_column.vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata_type",
                        match=models.MatchValue(value="columns"),
                    ),
                ]
            ),
            limit=limit + 1,  # Add 1 to account for the input column
        )
        
        # Format results, excluding the input column
        results = []
        for hit in search_results:
            if hit.id != base_column.id:
                results.append({
                    "score": hit.score,
                    "metadata": hit.payload["raw_metadata"],
                    "table_id": hit.payload["table_id"],
                })
                if len(results) >= limit:
                    break
                    
        return results

    def _index_transformation_metadata(self, metadata: dict, embedding_provider: EmbeddingProvider) -> None:
        """
        Index transformation metadata into the vector store.
        
        Args:
            metadata: Dictionary containing transformation metadata.
            embedding_provider: Provider for generating embeddings.
        """
        if "transformations" not in metadata:
            return
            
        for transformation_id, transformation in metadata["transformations"].items():
            # Prepare texts for embedding
            texts = [self._prepare_transformation_text(transformation)]
            
            # Add texts for each block
            if "blocks" in transformation:
                for block in transformation["blocks"]:
                    texts.append(self._prepare_block_text(block))
                    
            # Get embeddings for all texts
            embeddings = embedding_provider.embed(texts)
            
            # Create points for transformation and blocks
            points = []
            
            # Add transformation point
            points.append(
                models.PointStruct(
                    id=self._generate_point_id("transformations", {"id": transformation_id}),
                    vector=embeddings[0],
                    payload={
                        "metadata_type": "transformations",
                        "raw_metadata": transformation,
                    }
                )
            )
            
            # Add points for each block
            for i, block in enumerate(transformation.get("blocks", []), 1):
                points.append(
                    models.PointStruct(
                        id=self._generate_point_id(
                            "transformation_blocks",
                            {"id": f"{transformation_id}_block_{i}"}
                        ),
                        vector=embeddings[i],
                        payload={
                            "metadata_type": "transformation_blocks",
                            "transformation_id": transformation_id,
                            "raw_metadata": block,
                        }
                    )
                )
                
            # Upsert all points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def find_related_transformations(self, table_id: str, embedding_provider: EmbeddingProvider, limit: int = 5) -> List[dict]:
        """
        Find transformations that are related to a specific table.
        
        Args:
            table_id: The ID of the table to find related transformations for.
            embedding_provider: Provider for generating embeddings.
            limit: Maximum number of results to return.
            
        Returns:
            List[dict]: List of related transformations.
        """
        # Search for transformations that use this table
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding_provider.embed([f"table {table_id}"])[0],
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata_type",
                        match=models.MatchValue(value="transformations")
                    )
                ]
            ),
            limit=limit
        )
        
        return [
            {
                "score": hit.score,
                "metadata": hit.payload["raw_metadata"],
            }
            for hit in results
        ]

    def _index_items(self, items: List[Dict], metadata_type: str, embedding_provider: EmbeddingProvider, batch_size: int = 10) -> None:
        """Index a list of items with retries and batch processing."""
        def process_batch(batch: List[Dict]) -> None:
            texts = [self._prepare_text_for_embedding(item, metadata_type) for item in batch]
            embeddings = embedding_provider.embed(texts)
            points = [
                models.PointStruct(
                    id=self._generate_point_id(metadata_type, item),
                    vector=embedding.tolist(),
                    payload={
                        "metadata_type": metadata_type,
                        "text": text,
                        **item
                    }
                )
                for item, embedding, text in zip(batch, embeddings, texts)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        
        self.batch_processor.process_batches(items, process_batch)

    def _prepare_table_text(self, table: Dict, table_detail: Dict) -> str:
        """Prepare rich text representation of a table."""
        parts = [f"Table: {table.get('name', 'unknown')}"]
        
        if "description" in table:
            parts.append(f"Description: {table['description']}")
            
        if "statistics" in table_detail:
            stats = table_detail["statistics"]
            parts.extend([
                f"Rows: {stats.get('row_count', 0)}",
                f"Size: {stats.get('size_bytes', 0)} bytes",
                f"Freshness Score: {stats.get('freshness_score', 0)}/100"
            ])
            
        if "columns" in table_detail:
            parts.append("Columns: " + ", ".join(
                col.get("name", "") for col in table_detail["columns"]
            ))
            
        if "dependencies" in table_detail:
            deps = table_detail["dependencies"]
            if deps.get("source_tables"):
                parts.append("Sources: " + ", ".join(deps["source_tables"]))
            if deps.get("target_tables"):
                parts.append("Targets: " + ", ".join(deps["target_tables"]))
                
        return "\n".join(parts)
        
    def _prepare_column_text(self, column: Dict, table_id: str, table_detail: Dict) -> str:
        """Prepare rich text representation of a column."""
        parts = [f"Column: {column.get('name', 'unknown')}"]
        
        if "description" in column:
            parts.append(f"Description: {column['description']}")
            
        if "statistics" in column:
            stats = column["statistics"]
            parts.extend([
                f"Type: {stats.get('data_type', 'unknown')}",
                f"Format: {stats.get('format', 'unknown')}",
                f"Quality Score: {stats.get('quality_score', 0)}/100",
                f"Null Rate: {stats.get('null_count', 0)}/{table_detail.get('statistics', {}).get('row_count', 0)}",
                f"Unique Values: {stats.get('unique_count', 0)}"
            ])
            
            if stats.get("sample_values"):
                parts.append("Samples: " + ", ".join(map(str, stats["sample_values"][:5])))
                
        return "\n".join(parts)
        
    def _prepare_config_text(self, config: Dict, component_id: str) -> str:
        """Prepare rich text representation of a configuration."""
        parts = [f"Configuration: {config.get('name', 'unknown')}"]
        
        if "description" in config:
            parts.append(f"Description: {config['description']}")
            
        parts.extend([
            f"Component: {component_id}",
            f"Version: {config.get('version', 'unknown')}",
            f"State: {config.get('state', 'unknown')}"
        ])
        
        if "createdBy" in config:
            parts.append(f"Created By: {config['createdBy'].get('name', 'unknown')}")
            
        return "\n".join(parts)
        
    def _prepare_config_row_text(self, row: Dict, config_id: str, config: Dict) -> str:
        """Prepare rich text representation of a configuration row."""
        parts = [
            f"Config Row: {row.get('name', 'unknown')}",
            f"Configuration: {config.get('name', 'unknown')} ({config_id})"
        ]
        
        if "description" in row:
            parts.append(f"Description: {row['description']}")
            
        parts.append(f"State: {row.get('state', 'unknown')}")
        
        return "\n".join(parts)
        
    def _prepare_relationship_text(self, rel: Dict, rel_type: str) -> str:
        """Prepare rich text representation of a relationship."""
        if rel_type == "column_to_column":
            return (
                f"Column Relationship: {rel['source_table']}.{rel['source_column']} → "
                f"{rel['target_table']}.{rel['target_column']}\n"
                f"Type: {rel['relationship_type']}\n"
                f"Confidence: {rel['confidence']:.2%}"
            )
        elif rel_type == "table_to_table":
            return (
                f"Table Relationship: {rel['source_table']} → {rel['target_table']}\n"
                f"Type: {rel['relationship_type']}\n"
                f"Confidence: {rel['confidence']:.2%}"
            )
        else:
            return f"Relationship ({rel_type}): {json.dumps(rel)}"