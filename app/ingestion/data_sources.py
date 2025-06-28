from ingestion.definitions import DataSource, DataSourceProcessStatus, DataSourceConfig
from dataclasses import dataclass
from pandas import DataFrame
import pandas as pd
from pathlib import Path
import os
import shutil
import datetime


@dataclass
class LocalFileDataSourceConfig(DataSourceConfig):
    """
    This class should contain the configuration for the local file data source.
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
        Update the process status of the data source and may modify the source data to reflect this.
        (Moves source files to finished folder specified in config based on process status)
        """
        
        config: LocalFileDataSourceConfig = self.config
        success_dir = os.path.join(config.target_dir, "succeeded")
        failed_dir = os.path.join(config.target_dir, "failed")
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(failed_dir, exist_ok=True)

        target_dir = success_dir if status == DataSourceProcessStatus.SUCCESS else failed_dir

        for file_name in config.file_names:
            filepath = os.path.join(config.source_dir, file_name)
            if os.path.exists(filepath):
                filename = os.path.basename(filepath)
                target_path = os.path.join(target_dir, filename)

                if os.path.exists(target_path):
                    base, ext = os.path.splitext(filename)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{base}_{timestamp}{ext}"
                    target_path = os.path.join(target_dir, filename)

                shutil.move(filepath, target_path)

    def get_raw_data(self) -> DataFrame:
        
        return self.data

    def extract_data(self) -> None:

        file_type_to_parser = {

            ".csv": self._parse_csv,
            ".json": self._parse_json,
            ".xlsx": self._parse_excel,
            ".parquet": self._parse_parquet

        }

        config: LocalFileDataSourceConfig = self.config

        all_dataframes = []

        for file_name in config.file_names:
            file_path = Path(config.source_dir) / file_name
            file_type = file_path.suffix
            parser = file_type_to_parser.get(file_type, self._parse_csv)
            data = parser(file_path)
            all_dataframes.append(data)

        self.data = pd.concat(all_dataframes, ignore_index=True, sort=False)

    def _parse_csv(self, filepath: str) -> DataFrame:
        return pd.read_csv(filepath)

    def _parse_json(self, filepath: str) -> DataFrame:
        return pd.read_json(filepath)

    def _parse_excel(self, filepath: str) -> DataFrame:
        return pd.read_excel(filepath)

    def _parse_parquet(self, filepath: str) -> DataFrame:
        return pd.read_parquet(filepath)
