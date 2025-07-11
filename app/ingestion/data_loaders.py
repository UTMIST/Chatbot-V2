from app.ingestion.definitions import DataLoader, DataLoadConfig, QdrantDataLoadConfig  # [ADDED QdrantDataLoadConfig]
from dataclasses import dataclass
from llama_index.core.vector_stores.types import VectorStore
from pandas import DataFrame
from llama_index.core.schema import Node, MediaResource
from qdrant_client import QdrantClient  # [ADDED] Qdrant client
from qdrant_client.models import PointStruct, VectorParams, Distance  # [ADDED] Qdrant models


@dataclass
class VectorStoreDataLoaderConfig(DataLoadConfig):
    """Configuration for the VectorStoreDataLoader."""
    vector_store: VectorStore
    embeddings_colname: str = "embeddings"
    metadata_colname: str = "metadata"
    embeddings_text_colname: str = "embeddings_text"


class VectorStoreDataLoader(DataLoader):
    """
    Loads vectors into a llama-index VectorStore (local or in-memory).
    """
    def __init__(self, config: VectorStoreDataLoaderConfig):
        super().__init__(config)

    def load_data(self, data: DataFrame) -> None:
        """
        Convert rows into Nodes and add them to the configured VectorStore.
        """
        config: VectorStoreDataLoaderConfig = self.config

        nodes = []
        for _, row in data.iterrows():
            nodes.append(Node(
                text_resource=MediaResource(text=row[config.embeddings_text_colname]),
                embedding=row[config.embeddings_colname],
                metadata=row[config.metadata_colname]
            ))
        config.vector_store.add(nodes)


# [ADDED] QdrantDataLoader for loading data into a Qdrant collection
class QdrantDataLoader(DataLoader):
    """
    Loads embeddings and metadata into a Qdrant cloud collection.
    """

    def __init__(self, config: QdrantDataLoadConfig):
        """
        Instantiate Qdrant client and ensure the target collection exists.
        """
        super().__init__(config)
        self.config: QdrantDataLoadConfig = config

        # Build client kwargs from config
        client_kwargs = {"url": config.host, "port": config.port}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        self.client = QdrantClient(**client_kwargs, prefer_grpc=config.prefer_grpc)

        # Map distance string to Distance enum (defaulting to COSINE)
        try:
            distance_enum = Distance[config.distance.upper()]
        except KeyError:
            distance_enum = Distance.COSINE

        # Create the collection if it doesn't already exist
        existing = [c.name for c in self.client.get_collections().collections]
        if config.collection_name not in existing:
            self.client.create_collection(
                collection_name=config.collection_name,
                vectors_config=VectorParams(size=config.vector_size, distance=distance_enum)
            )

    def load_data(self, data: DataFrame) -> None:
        """
        Upsert rows into the Qdrant collection as points (id, vector, payload).
        """
        config: QdrantDataLoadConfig = self.config
        points: list[PointStruct] = []

        for _, row in data.iterrows():
            # 1) Try to read a top-level 'id' column…
            point_id = row.get("id")

            # 2) …otherwise pull it out of the metadata dict
            if point_id is None:
                metadata = row.get("metadata", {})
                point_id = metadata.get("id")

            # 3) Now build the PointStruct with a valid id
            points.append(PointStruct(
                id=point_id,
                vector=row.get("embeddings"),
                payload=row.get("metadata")
            ))

        # 4) Send all points to Qdrant in one upsert call
        self.client.upsert(
            collection_name=config.collection_name,
            points=points
        )

