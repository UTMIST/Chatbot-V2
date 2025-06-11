from ingestion.definitions import DataLoader, DataLoadConfig
from dataclasses import dataclass
from llama_index.core.vector_stores.types import VectorStore
from pandas import DataFrame
from llama_index.core.schema import Node, MediaResource


@dataclass
class VectorStoreDataLoaderConfig(DataLoadConfig):
    """Configuration for the VectorStoreDataLoader."""
    vector_store: VectorStore
    embeddings_colname: str = "embeddings"
    metadata_colname: str = "metadata"
    embeddings_text_colname: str = "embeddings_text"


class VectorStoreDataLoader(DataLoader):
    def __init__(self, config: VectorStoreDataLoaderConfig):
        super().__init__(config)

    def load_data(self, data: DataFrame) -> None:
        """
        Load the data into the vector store.
        """
        config: VectorStoreDataLoaderConfig = self.config
        
        nodes = []
        for _, row in data.iterrows():
            nodes.append(Node(
                text_resource=MediaResource(text=row[config.embeddings_text_colname]),
                embedding=row[config.embeddings_colname], 
                metadata=row[config.metadata_colname]))
        config.vector_store.add(nodes)
