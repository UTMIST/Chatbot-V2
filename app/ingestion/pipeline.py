from dataclasses import dataclass
from typing import Optional
from pandas import DataFrame

from ingestion.definitions import (
    DataSource,
    DataTransformer,
    DataLoader,
    DataSourceConfig,
    DataTransformConfig,
    DataLoadConfig,
    DataSourceProcessStatus
)


@dataclass
class PipelineConfig:
    source_config: DataSourceConfig
    transform_configs: list[DataTransformConfig]
    load_config: DataLoadConfig


class IngestionPipeline:
    def __init__(
        self,
        data_source: DataSource,
        transformers: list[DataTransformer],
        loader: DataLoader
    ):
        self.data_source = data_source
        self.transformers = transformers
        self.loader = loader
        self.processed_data: Optional[DataFrame] = None

    @classmethod
    def from_config(cls, config: PipelineConfig, 
                   source_class: type[DataSource],
                   transformer_classes: list[type[DataTransformer]],
                   loader_class: type[DataLoader]) -> 'IngestionPipeline':
        """
        Factory method to create a pipeline from configuration objects
        """
        data_source = source_class(config.source_config)
        transformers = [transformer_classes[i](config.transform_configs[i]) for i in range(len(config.transform_configs))]
        loader = loader_class(config.load_config)
        
        return cls(data_source, transformers, loader)

    def run(self) -> None:
        """
        Execute the full ingestion pipeline
        """
        try:
            self.data_source.extract_data()
            raw_data = self.data_source.get_raw_data()

            for transformer in self.transformers:
                raw_data = transformer.apply_transformation(raw_data)

            self.processed_data = raw_data

            self.loader.load_data(self.processed_data)

            self.data_source.update_process_status(DataSourceProcessStatus.SUCCESS)

        except Exception as e:
            self.data_source.update_process_status(DataSourceProcessStatus.FAILED)
            raise RuntimeError(f"Pipeline execution failed: {str(e)}") from e