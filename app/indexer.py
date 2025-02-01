import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.vectorizer import EmbeddingProvider


class QdrantIndexer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 55000,  # Updated default port
        collection_name: str = "keboola_metadata",
        vector_size: int = 1536,  # Updated for OpenAI embeddings
    ):
        """Initialize the Qdrant indexer."""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure the collection exists with the correct settings."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists:
                logging.info(f"Collection {self.collection_name} already exists")
                return

            # Create collection with optimized settings
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logging.info(f"Created collection {self.collection_name}")

            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
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

    def index_metadata(
        self,
        metadata: Dict,
        embedding_provider: EmbeddingProvider,
        batch_size: int = 10,  # Reduced from 50 to 10
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
            if isinstance(items, list):
                logging.info(f"Found {len(items)} items of type {metadata_type}")
                self._index_items(items, metadata_type, embedding_provider, batch_size)
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
                texts = [self._prepare_text_for_embedding(item) for item in batch]

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

    def _prepare_text_for_embedding(self, item: Dict) -> str:
        """Convert a metadata item into a text representation for embedding."""
        # Extract relevant fields based on common metadata fields
        fields = []

        # Add name/title if available
        if "name" in item:
            fields.append(f"Name: {item['name']}")
        elif "title" in item:
            fields.append(f"Title: {item['title']}")

        # Add description if available
        if "description" in item:
            fields.append(f"Description: {item['description']}")

        # Add type/stage if available
        if "type" in item:
            fields.append(f"Type: {item['type']}")
        if "stage" in item:
            fields.append(f"Stage: {item['stage']}")

        # Add any tags
        if "tags" in item and item["tags"]:
            fields.append(f"Tags: {', '.join(item['tags'])}")

        # Combine all fields
        return " | ".join(fields) or str(item)

    def _generate_point_id(self, metadata_type: str, item: Dict) -> int:
        """Generate a unique ID for a metadata item."""
        # Handle both dictionary and list items
        if isinstance(item, dict):
            item_id = item.get("id", str(sorted(item.items())))
        else:
            item_id = str(item)  # Convert list to string
        # Generate a positive integer hash
        return abs(hash(f"{metadata_type}_{item_id}"))

    def search_metadata(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        metadata_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Search for metadata similar to the query."""
        # Get query embedding
        query_embedding = embedding_provider.embed([query])[0]

        # Prepare search filter
        search_filter = None
        if metadata_type:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata_type",
                        match=models.MatchValue(value=metadata_type),
                    )
                ]
            )

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
            }
            for hit in results
        ]
