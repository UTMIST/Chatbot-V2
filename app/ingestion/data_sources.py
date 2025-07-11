from app.ingestion.definitions import (
    DataSource,
    DataSourceProcessStatus,
    DataSourceConfig,
    QdrantDataSourceConfig,  # [MODIFIED] import Qdrant config
)
from dataclasses import dataclass
from pandas import DataFrame
import pandas as pd
from pathlib import Path
import os
import shutil
import datetime
from qdrant_client import QdrantClient  # [ADDED] to connect to Qdrant cloud


@dataclass
class LocalFileDataSourceConfig(DataSourceConfig):
    """
    Configuration for the local file data source.
    """
    source_dir: str
    target_dir: str
    file_names: list[str]


class LocalFileDataSource(DataSource):

    def __init__(self, config: LocalFileDataSourceConfig):
        super().__init__(config)

    def save_transformed_data(self, data: DataFrame) -> None:
        """
        Save the transformed data to a file in the finished folder specified in the config.
        """
        config: LocalFileDataSourceConfig = self.config
        target_dir = os.path.join(config.target_dir, "transformed")
        os.makedirs(target_dir, exist_ok=True)
        data.to_csv(os.path.join(target_dir, "transformed_data.csv"), index=False)

    def update_process_status(self, status: DataSourceProcessStatus) -> None:
        """
        Move source files to a status folder (succeeded/failed) based on process status.
        """
        config: LocalFileDataSourceConfig = self.config
        success_dir = os.path.join(config.target_dir, "succeeded")
        failed_dir = os.path.join(config.target_dir, "failed")
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(failed_dir, exist_ok=True)

        dest_dir = success_dir if status == DataSourceProcessStatus.SUCCESS else failed_dir
        for fname in config.file_names:
            src = os.path.join(config.source_dir, fname)
            if os.path.exists(src):
                shutil.move(src, os.path.join(dest_dir, fname))

    def extract_data(self) -> None:
        """
        Read all files from source_dir matching file_names, parse them into DataFrames,
        and concatenate into self.data.
        """
        config: LocalFileDataSourceConfig = self.config
        all_frames = []
        parsers = {
            ".csv": self._parse_csv,
            ".json": self._parse_json,
            ".xlsx": self._parse_excel,
            ".parquet": self._parse_parquet,
        }
        for fname in config.file_names:
            path = Path(config.source_dir) / fname
            ext = path.suffix.lower()
            if ext in parsers:
                df = parsers[ext](str(path))
                all_frames.append(df)
        self.data = pd.concat(all_frames, ignore_index=True, sort=False)

    def get_raw_data(self) -> DataFrame:
        """
        Return the loaded raw DataFrame.
        """
        return self.data

    # --- parsing helpers ---
    def _parse_csv(self, filepath: str) -> DataFrame:
        return pd.read_csv(filepath)

    def _parse_json(self, filepath: str) -> DataFrame:
        return pd.read_json(filepath)

    def _parse_excel(self, filepath: str) -> DataFrame:
        return pd.read_excel(filepath)

    def _parse_parquet(self, filepath: str) -> DataFrame:
        return pd.read_parquet(filepath)
    

# [ADDED] QdrantDataSource for querying stored vectors/metadata (In case we need batch analysis, reloading data for front-end widget, or reindexing)
class QdrantDataSource(DataSource):
    """
    DataSource that retrieves stored vectors/metadata from a Qdrant collection.
    """

    def __init__(self, config: QdrantDataSourceConfig):
        super().__init__(config)
        # [ADDED] instantiate Qdrant client using config
        client_kwargs = {"url": config.host, "port": config.port}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        self.client = QdrantClient(**client_kwargs, prefer_grpc=config.prefer_grpc)
        self.collection_name = config.collection_name

    def extract_data(self) -> None:
        """
        Ensure connection to Qdrant and collection accessibility.
        Mark process as successful if client can list collections.
        """
        try:
            cols = self.client.get_collections()
            if self.collection_name not in [c.name for c in cols.collections]:
                raise ValueError(f"Collection {self.collection_name} not found")
            self.update_process_status(DataSourceProcessStatus.SUCCESS)
        except Exception:
            self.update_process_status(DataSourceProcessStatus.FAILED)

    def get_raw_data(self) -> DataFrame:
        """
        Scroll through the Qdrant collection to retrieve all payloads (metadata),
        and return them as a pandas DataFrame.
        """
        records = []
        # [ADDED] fetch all points' payloads
        for point in self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vector=False
        ):
            payload = point.payload or {}
            payload["_id"] = point.id
            records.append(payload)
        df = pd.DataFrame(records)
        self.data = df
        return df

    def update_process_status(self, status: DataSourceProcessStatus) -> None:
        # [ADDED] Qdrant source has no filesystem moves; we log or ignore
        pass
