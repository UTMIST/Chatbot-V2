from abc import ABC, abstractmethod
from pandas import DataFrame
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class DataSourceProcessStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class DataSourceConfig:
    """
    This class should contain the configuration for the data source.
    """
    pass


@dataclass
class DataTransformConfig:
    """
    This class should contain the configuration for the data transformation process.
    """
    pass


@dataclass
class DataLoadConfig:
    """
    This class should contain the configuration for the data loading process.
    """
    pass


# Qdrant-specific configurations

@dataclass
class QdrantDataLoadConfig(DataLoadConfig):
    """
    Configuration for loading embeddings into a Qdrant collection.
    """
    host: str                 # e.g. "abcd1234-xyz.qdrant.cloud"
    port: int                 # e.g. 443 or 6333/6334
    collection_name: str      # Name of the Qdrant collection
    vector_size: int          # Dimensionality of your embeddings

    # Defaults:
    prefer_grpc: bool = True
    api_key: Optional[str] = None
    distance: str = "Cosine"  # or "Euclid", etc.

@dataclass
class QdrantDataSourceConfig(DataSourceConfig):
    """
    Configuration for querying embeddings out of Qdrant.
    """
    host: str
    port: int
    collection_name: str

    # Defaults:
    prefer_grpc: bool = True
    api_key: Optional[str] = None


class DataSource(ABC):

    def __init__(self, config: DataSourceConfig):
        self.config: DataSourceConfig = config
        self.data: Optional[DataFrame] = None

    @abstractmethod
    def extract_data(self) -> None:
        pass

    @abstractmethod
    def get_raw_data(self) -> DataFrame:
        pass

    @abstractmethod
    def update_process_status(self, status: DataSourceProcessStatus) -> None:
        pass


class DataTransformer(ABC):

    def __init__(self, config: DataTransformConfig):
        self.config: DataTransformConfig = config

    @abstractmethod
    def apply_transformation(self, raw_data: DataFrame) -> DataFrame:
        pass


class DataLoader(ABC):

    def __init__(self, config: DataLoadConfig):
        self.config: DataLoadConfig = config

    @abstractmethod
    def load_data(self, data: DataFrame) -> None:
        pass
