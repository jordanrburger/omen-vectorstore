import logging
import uuid
from typing import Dict, List, Optional, Callable, Any, Union
from functools import partial
import json

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
        collection_name: str = "keboola_metadata",
        batch_config: Optional[BatchConfig] = None,
    ):
        """Initialize QdrantIndexer with collection name and batch configuration."""
        self.collection_name = collection_name
        self.vector_size = 1536  # OpenAI ada-002 embedding size
        self.client = QdrantClient("localhost", port=55000)
        self.batch_processor = BatchProcessor(batch_config or BatchConfig())
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Ensure the collection exists with the correct configuration."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info is None:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
        except UnexpectedResponse:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def index_metadata(
        self,
        metadata: Dict,
        embedding_provider: EmbeddingProvider,
        batch_size: Optional[int] = None,
    ) -> None:
        """Index metadata into Qdrant with embeddings."""
        if not metadata:
            return
        
        # Create a function to process points
        def process_points(points):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        # Process transformations
        if "transformations" in metadata:
            for transformation_id, transformation in metadata["transformations"].items():
                logging.info(f"Processing transformation {transformation_id}")
                
                # Prepare texts for embedding
                texts = []
                
                # Add transformation metadata
                texts.append(self._prepare_transformation_text(transformation))
                
                # Add texts for each block
                if "blocks" in transformation:
                    for block in transformation["blocks"]:
                        block_text = self._prepare_block_text(block)
                        if block_text:
                            texts.append(block_text)
                
                # Get embeddings for all texts
                embeddings = embedding_provider.embed(texts)
                
                # Prepare points for indexing
                points = []
                for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                    point_id = self._generate_point_id(transformation_id, i)
                    points.append({
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "text": text,
                            "metadata_type": "transformation",
                            "raw_metadata": transformation if i == 0 else transformation["blocks"][i-1],
                            "transformation_id": transformation_id
                        }
                    })
                
                # Index points in batches
                self.batch_processor.process_batches(points, process_fn=process_points)
        
        # Process other metadata types
        for metadata_type, items in metadata.items():
            if metadata_type != "transformations":  # Skip transformations as they're already processed
                logging.info(f"Processing metadata type: {metadata_type}")
                
                if metadata_type == "buckets":
                    # Handle buckets (list)
                    logging.info(f"Found {len(items)} items of type {metadata_type}")
                    self._process_items(
                        items,
                        metadata_type,
                        embedding_provider,
                        batch_size=batch_size,
                    )
                
                elif metadata_type == "tables":
                    # Handle tables dictionary (bucket_id -> list of tables)
                    for bucket_id, tables in items.items():
                        logging.info(
                            f"Processing {len(tables)} tables for bucket {bucket_id}"
                        )
                        self._process_items(
                            tables,
                            metadata_type,
                            embedding_provider,
                            batch_size=batch_size,
                        )
                
                elif metadata_type == "table_details":
                    # Handle table details and their columns
                    for table_id, details in items.items():
                        if "columns" in details:
                            logging.info(
                                f"Processing {len(details['columns'])} columns for table {table_id}"
                            )
                            # Create new column objects with table_id
                            enriched_columns = []
                            for column in details["columns"]:
                                if isinstance(column, str):
                                    # Convert string to dictionary if needed
                                    enriched_column = {"name": column}
                                else:
                                    # Make a copy to avoid modifying the original
                                    enriched_column = dict(column)
                                
                                enriched_column["table_id"] = table_id
                                if "bucket_id" in details:
                                    enriched_column["bucket_id"] = details["bucket_id"]
                                enriched_columns.append(enriched_column)
                            
                            self._process_items(
                                enriched_columns,
                                "columns",
                                embedding_provider,
                                batch_size=batch_size,
                            )
                
                elif metadata_type == "configurations":
                    # Handle configurations as a list
                    logging.info(f"Found {len(items)} items of type {metadata_type}")
                    self._process_items(
                        items,
                        metadata_type,
                        embedding_provider,
                        batch_size=batch_size,
                    )

    def _process_items(
        self,
        items: List[Dict],
        metadata_type: str,
        embedding_provider: EmbeddingProvider,
        batch_size: Optional[int] = None,
    ) -> None:
        """Process a batch of items for indexing."""
        def process_batch(batch: List[Dict]) -> None:
            texts = [self._prepare_text_for_embedding(item, metadata_type) for item in batch]
            embeddings = embedding_provider.embed(texts)
            
            points = []
            for item, embedding in zip(batch, embeddings):
                point_id = self._generate_point_id(metadata_type, item)
                payload = {
                    "metadata_type": metadata_type,
                    "raw_metadata": item,
                    "text": self._prepare_text_for_embedding(item, metadata_type)
                }
                
                # Add table_id for columns
                if metadata_type == "columns" and "table_id" in item:
                    payload["table_id"] = item["table_id"]
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        
        if batch_size:
            self.batch_processor.batch_config.batch_size = batch_size
        self.batch_processor.process_batches(items, process_batch)

    def _prepare_text_for_embedding(self, item, metadata_type):
        """Prepare text for embedding based on metadata type."""
        if metadata_type == "columns":
            return self._prepare_column_text(item)
        elif metadata_type == "transformations":
            return self._prepare_transformation_text(item)
        elif metadata_type == "blocks":
            return self._prepare_block_text(item)
        elif metadata_type == "tables":
            return self._prepare_table_text(item)
        else:
            return self._prepare_default_text(item)

    def _prepare_default_text(self, item):
        """Prepare default text format for items."""
        parts = []
        if "name" in item:
            parts.append(f"Name: {item['name']}")
        if "description" in item:
            parts.append(f"Description: {item['description']}")
        if "type" in item:
            parts.append(f"Type: {item['type']}")
        if "tags" in item and item["tags"]:
            parts.append(f"Tags: {', '.join(item['tags'])}")
        return " | ".join(parts) if parts else "{}"

    def _prepare_table_text(self, table):
        """Prepare table metadata text."""
        parts = []
        if "name" in table:
            parts.append(f"Name: {table['name']}")
        if "description" in table:
            parts.append(f"Description: {table['description']}")
        if "type" in table:
            parts.append(f"Type: {table['type']}")
        if "tags" in table and table["tags"]:
            parts.append(f"Tags: {', '.join(table['tags'])}")
        return " | ".join(parts)

    def _prepare_column_text(self, column):
        """Prepare column metadata text."""
        if not column:
            return "{}"
        
        parts = []
        if "name" in column:
            parts.append(f"Name: {column['name']}")
        if "type" in column:
            parts.append(f"Type: {column['type']}")
        if "description" in column:
            parts.append(f"Description: {column['description']}")
        
        if "statistics" in column:
            stats = column["statistics"]
            stat_parts = []
            if "min" in stats:
                stat_parts.append(f"min={stats['min']}")
            if "max" in stats:
                stat_parts.append(f"max={stats['max']}")
            if "avg" in stats:
                stat_parts.append(f"avg={stats['avg']}")
            if "unique_count" in stats:
                stat_parts.append(f"unique={stats['unique_count']}")
            if "most_common" in stats:
                stat_parts.append(f"most_common={','.join(stats['most_common'])}")
            if stat_parts:
                parts.append(f"Statistics: {', '.join(stat_parts)}")
        
        if "quality_metrics" in column:
            quality = column["quality_metrics"]
            quality_parts = []
            if "completeness" in quality:
                quality_parts.append(f"{int(quality['completeness'] * 100)}% complete")
            if "validity" in quality:
                quality_parts.append(f"{int(quality['validity'] * 100)}% valid")
            if "range_check" in quality:
                quality_parts.append(f"Valid range: {quality['range_check']['min_valid']}-{quality['range_check']['max_valid']}")
            if "standardization" in quality:
                quality_parts.append(f"Standard: {quality['standardization']['standard']}")
            if "common_issues" in quality:
                quality_parts.append(f"Issues: {', '.join(quality['common_issues'])}")
            if quality_parts:
                parts.append(f"Quality: {', '.join(quality_parts)}")
        
        return " | ".join(parts)

    def _prepare_transformation_text(self, transformation: Dict) -> str:
        """Prepare a text representation of a transformation."""
        name = transformation.get("name", "Unnamed")
        transformation_type = transformation.get("type", "unknown")
        description = transformation.get("description", "")
        
        # Extract dependencies
        dependencies = []
        
        # Check direct dependencies first
        if "dependencies" in transformation:
            if "requires" in transformation["dependencies"]:
                dependencies.append(f"Requires {', '.join(transformation['dependencies']['requires'])}")
            if "produces" in transformation["dependencies"]:
                dependencies.append(f"Produces {', '.join(transformation['dependencies']['produces'])}")
        
        # If no direct dependencies, try to extract from blocks
        if not dependencies:
            inputs = set()
            outputs = set()
            for block in transformation.get("blocks", []):
                if "inputs" in block:
                    for input_item in block["inputs"]:
                        if isinstance(input_item, dict):
                            inputs.add(input_item.get("source", ""))
                        else:
                            inputs.add(input_item)
                if "outputs" in block:
                    for output_item in block["outputs"]:
                        if isinstance(output_item, dict):
                            outputs.add(output_item.get("destination", ""))
                        else:
                            outputs.add(output_item)
            
            inputs = {i for i in inputs if i}  # Remove empty strings
            outputs = {o for o in outputs if o}  # Remove empty strings
            
            if inputs:
                dependencies.append(f"Requires {', '.join(sorted(inputs))}")
            if outputs:
                dependencies.append(f"Produces {', '.join(sorted(outputs))}")
        
        dependencies_text = "; ".join(dependencies)
        
        # Extract runtime info
        runtime_parts = []
        if "runtime" in transformation:
            if "type" in transformation["runtime"]:
                runtime_parts.append(transformation["runtime"]["type"])
            if "memory" in transformation["runtime"]:
                runtime_parts.append(f"{transformation['runtime']['memory']} memory")
        else:
            # Fallback to top-level memory and default docker type
            if "memory" in transformation:
                runtime_parts.append(f"{transformation['memory']} memory")
            runtime_parts.insert(0, "docker")
        
        runtime_text = ", ".join(runtime_parts)
        
        return f"Name: {name} | Type: {transformation_type} | Description: {description} | Dependencies: {dependencies_text} | Runtime: {runtime_text}"

    def _prepare_block_text(self, block: Dict) -> str:
        """Prepare a text representation of a transformation block."""
        parts = []
        
        # Block name
        name = block.get("name", "Unnamed Block")
        parts.append(f"Block: {name}")
        
        # Inputs
        if "inputs" in block:
            input_details = []
            for input_item in block["inputs"]:
                if isinstance(input_item, dict):
                    if "file" in input_item and "source" in input_item:
                        input_details.append(f"{input_item['file']} from {input_item['source']}")
                else:
                    table_name = input_item.split(".")[-1]
                    input_details.append(f"{table_name} from {input_item}")
            if input_details:
                parts.append(f"Inputs: {', '.join(input_details)}")
        
        # Outputs
        if "outputs" in block:
            output_details = []
            for output_item in block["outputs"]:
                if isinstance(output_item, dict):
                    if "file" in output_item and "destination" in output_item:
                        output_details.append(f"{output_item['file']} to {output_item['destination']}")
                else:
                    table_name = output_item.split(".")[-1]
                    output_details.append(f"{table_name} to {output_item}")
            if output_details:
                parts.append(f"Outputs: {', '.join(output_details)}")
        
        # Code and operations
        code = block.get("code", "")
        operations = []
        
        # Extract description from first comment if available
        description = ""
        if code:
            lines = code.split("\n")
            for line in lines:
                if line.strip().startswith("#"):
                    description = line.strip("# ").strip()
                    break
        if description:
            operations.append(description)
        
        # Look for common operations
        operation_keywords = {
            "read_csv": "read_csv",
            "merge": "merge",
            "groupby": "groupby",
            "agg": "aggregate",
            "aggregate": "aggregate",
            "to_csv": "to_csv"
        }
        
        for keyword, operation in operation_keywords.items():
            if keyword in code.lower():
                operations.append(operation)
        
        # Add operations to parts
        if operations:
            parts.append(f"Code: {', '.join(operations)}")
        
        return " | ".join(parts)

    def _generate_point_id(self, metadata_type: str, item: Union[Dict, int]) -> str:
        """Generate a unique ID for a point."""
        # Create a deterministic UUID based on metadata type and item ID/name
        namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, 'keboola.metadata')
        
        # If item is a dictionary, use its ID or name
        if isinstance(item, dict):
            if "id" in item:
                item_id = str(item["id"])
            elif "name" in item:
                item_id = str(item["name"])
            else:
                item_id = str(uuid.uuid4())
        else:
            # If item is not a dictionary (e.g., an integer index), use its string representation
            item_id = str(item)
        
        # Generate a new UUID using the namespace UUID and the combined metadata type and item ID
        return str(uuid.uuid5(namespace_uuid, f"{metadata_type}_{item_id}"))

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