import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.vectorizer import EmbeddingProvider


class QdrantIndexer:
    """Indexes metadata into Qdrant with vector embeddings."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "keboola_metadata",
    ):
        """Initialize the indexer with Qdrant connection details."""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            if not any(c.name == self.collection_name for c in collections.collections):
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=384,  # Default size for all-MiniLM-L6-v2
                            distance=models.Distance.COSINE,
                        ),
                    )
                    logging.info(f"Created collection: {self.collection_name}")
                except UnexpectedResponse as e:
                    if "No space left on device" in str(e):
                        raise RuntimeError(
                            "Not enough disk space for Qdrant. Try:\n"
                            "1. Free up disk space\n"
                            "2. Reduce batch size\n"
                            "3. Restart Qdrant with lower WAL capacity: "
                            "QDRANT__STORAGE__WAL_CAPACITY_MB=512"
                        ) from e
                    raise
            else:
                logging.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logging.error(f"Error ensuring collection exists: {e}")
            raise

    def index_metadata(
        self,
        metadata: Dict,
        embedding_provider: EmbeddingProvider,
        batch_size: int = 10,
    ) -> None:
        """Index metadata into Qdrant with embeddings."""
        total_items = sum(
            len(items) if isinstance(items, list) else len(items.values())
            for items in metadata.values()
        )
        logging.info(
            f"Starting indexing of {total_items} total items "
            f"with batch size {batch_size}"
        )

        # Process each type of metadata
        for metadata_type, items in metadata.items():
            logging.info(f"Processing metadata type: {metadata_type}")
            
            if metadata_type == "table_details":
                # Handle table details and their columns
                for table_id, table_details in items.items():
                    # Index the table details
                    self._index_items(
                        [table_details],
                        "table_details",
                        embedding_provider,
                        batch_size,
                    )
                    
                    # Extract and index columns if present
                    if "columns" in table_details:
                        columns = table_details["columns"]
                        logging.info(
                            f"Processing {len(columns)} columns for table {table_id}"
                        )
                        self._index_items(
                            columns,
                            "columns",
                            embedding_provider,
                            batch_size,
                            {"table_id": table_id},
                        )
            elif isinstance(items, list):
                logging.info(f"Found {len(items)} items of type {metadata_type}")
                self._index_items(
                    items,
                    metadata_type,
                    embedding_provider,
                    batch_size,
                )
            elif isinstance(items, dict):
                if metadata_type == "tables":
                    # Handle tables dictionary (bucket_id -> list of tables)
                    for bucket_id, tables in items.items():
                        logging.info(
                            f"Processing {len(tables)} tables for bucket {bucket_id}"
                        )
                        self._index_items(
                            tables,
                            metadata_type,
                            embedding_provider,
                            batch_size,
                            {"bucket_id": bucket_id},
                        )
                else:
                    # Handle other dictionary metadata
                    item_list = list(items.values())
                    logging.info(
                        f"Found {len(item_list)} items of type {metadata_type}"
                    )
                    self._index_items(
                        item_list,
                        metadata_type,
                        embedding_provider,
                        batch_size,
                    )

    def _index_items(
        self,
        items: List[Dict],
        metadata_type: str,
        embedding_provider: EmbeddingProvider,
        batch_size: int,
        additional_payload: Optional[Dict] = None,
    ) -> None:
        """Index a batch of items with their embeddings."""
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            try:
                # Prepare texts for embedding
                texts = [self._prepare_text_for_embedding(item, metadata_type) for item in batch]

                # Get embeddings
                embeddings = embedding_provider.embed(texts)

                # Prepare points for Qdrant
                points = []
                for item, embedding in zip(batch, embeddings):
                    # Create payload
                    payload = {
                        "metadata_type": metadata_type,
                        "raw_metadata": item,
                    }
                    if additional_payload:
                        payload.update(additional_payload)

                    # Create point
                    point = models.PointStruct(
                        id=self._generate_point_id(metadata_type, item),
                        vector=embedding,
                        payload=payload,
                    )
                    points.append(point)

                # Upsert points to Qdrant with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=points,
                        )
                        break
                    except UnexpectedResponse as e:
                        if (
                            "No space left on device" in str(e)
                            and attempt < max_retries - 1
                        ):
                            logging.warning(
                                f"Storage issue, retrying with smaller batch... "
                                f"(attempt {attempt + 1})"
                            )
                            # Reduce batch size and retry
                            points = points[: len(points) // 2]
                            continue
                        raise

                logging.info(
                    f"Indexed {len(points)} {metadata_type} items "
                    f"({i + len(batch)}/{len(items)})"
                )

            except Exception as e:
                logging.error(f"Error indexing batch {i//batch_size + 1}: {e}")
                raise

    def _prepare_text_for_embedding(self, item: Dict, metadata_type: str) -> str:
        """Convert a metadata item into a text representation for embedding."""
        if metadata_type == "columns":
            # Special handling for column metadata
            fields = []
            if "name" in item:
                fields.append(f"Name: {item['name']}")
            if "type" in item:
                fields.append(f"Type: {item['type']}")
            if "description" in item:
                fields.append(f"Description: {item['description']}")
            
            # Add statistics if available
            if "statistics" in item:
                stats = item["statistics"]
                if item.get("type", "").upper() in ("INTEGER", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE"):
                    # Numeric column statistics
                    stat_parts = []
                    if "min" in stats:
                        stat_parts.append(f"min={stats['min']}")
                    if "max" in stats:
                        stat_parts.append(f"max={stats['max']}")
                    if "avg" in stats:
                        stat_parts.append(f"avg={stats['avg']}")
                    if "unique_count" in stats:
                        stat_parts.append(f"unique={stats['unique_count']}")
                    if stat_parts:
                        fields.append(f"Statistics: {', '.join(stat_parts)}")
                else:
                    # String/other column statistics
                    stat_parts = []
                    if "unique_count" in stats:
                        stat_parts.append(f"unique={stats['unique_count']}")
                    if "most_common" in stats and stats["most_common"]:
                        stat_parts.append(f"most_common={','.join(stats['most_common'][:3])}")
                    if stat_parts:
                        fields.append(f"Statistics: {', '.join(stat_parts)}")
            
            # Add quality metrics if available
            if "quality_metrics" in item:
                metrics = item["quality_metrics"]
                quality_parts = []
                
                # Add completeness and validity
                if "completeness" in metrics:
                    quality_parts.append(f"{int(metrics['completeness'] * 100)}% complete")
                if "validity" in metrics:
                    quality_parts.append(f"{int(metrics['validity'] * 100)}% valid")
                
                # Add range check for numeric columns
                if "range_check" in metrics:
                    range_check = metrics["range_check"]
                    quality_parts.append(
                        f"Valid range: {range_check['min_valid']}-{range_check['max_valid']}"
                    )
                
                # Add standardization info
                if "standardization" in metrics:
                    standard = metrics["standardization"]["standard"]
                    quality_parts.append(f"Standard: {standard}")
                
                # Add common issues
                if "common_issues" in metrics and metrics["common_issues"]:
                    quality_parts.append(f"Issues: {', '.join(metrics['common_issues'])}")
                
                if quality_parts:
                    fields.append(f"Quality: {', '.join(quality_parts)}")
            
            return " | ".join(fields)
        else:
            # Default handling for other metadata types
            fields = []
            if "name" in item:
                fields.append(f"Name: {item['name']}")
            elif "title" in item:
                fields.append(f"Title: {item['title']}")
            if "description" in item:
                fields.append(f"Description: {item['description']}")
            if "type" in item:
                fields.append(f"Type: {item['type']}")
            if "stage" in item:
                fields.append(f"Stage: {item['stage']}")
            if "tags" in item and item["tags"]:
                fields.append(f"Tags: {', '.join(item['tags'])}")
            return " | ".join(fields) or str(item)

    def _generate_point_id(self, metadata_type: str, item: Dict) -> int:
        """Generate a unique ID for a metadata item."""
        # Handle both dictionary and list items
        if isinstance(item, dict):
            if metadata_type == "columns":
                # For columns, include table_id in the hash to ensure uniqueness
                table_id = item.get("table_id", "unknown")
                item_id = f"{table_id}_{item.get('name', str(item))}"
            else:
                item_id = item.get("id", str(sorted(item.items())))
        else:
            item_id = str(item)
        # Generate a positive integer hash
        return abs(hash(f"{metadata_type}_{item_id}"))

    def search_metadata(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        metadata_type: Optional[str] = None,
        table_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Search for metadata similar to the query."""
        # Get query embedding
        query_embedding = embedding_provider.embed([query])[0]

        # Prepare search filter
        must_conditions = []
        if metadata_type:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata_type",
                    match=models.MatchValue(value=metadata_type),
                )
            )
        if table_id:
            must_conditions.append(
                models.FieldCondition(
                    key="table_id",
                    match=models.MatchValue(value=table_id),
                )
            )

        search_filter = models.Filter(must=must_conditions) if must_conditions else None

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter,
        )

        # Extract and return results
        return [
            {
                "score": hit.score,
                "metadata_type": hit.payload["metadata_type"],
                "metadata": hit.payload["raw_metadata"],
                **(
                    {"table_id": hit.payload["table_id"]}
                    if "table_id" in hit.payload
                    else {}
                ),
            }
            for hit in results
        ]

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
        exclude_same_table: bool = True,
    ) -> List[Dict]:
        """Find columns similar to the specified column across all tables."""
        # Get the source column's metadata
        source_columns = self.find_table_columns(table_id=table_id)
        source_column = next(
            (col for col in source_columns if col["metadata"]["name"] == column_name),
            None,
        )
        if not source_column:
            raise ValueError(f"Column {column_name} not found in table {table_id}")

        # Prepare search text from source column
        search_text = self._prepare_text_for_embedding(
            source_column["metadata"], "columns"
        )

        # Search for similar columns
        results = self.search_metadata(
            query=search_text,
            embedding_provider=embedding_provider,
            metadata_type="columns",
            limit=limit + (1 if not exclude_same_table else 0),
        )

        # Filter out the source column if requested
        if exclude_same_table:
            results = [
                r
                for r in results
                if r.get("table_id") != table_id
                or r["metadata"]["name"] != column_name
            ][:limit]

        return results
