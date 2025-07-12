# app/ingestion/pipeline.py

import os, logging
from typing import List
from pandas import DataFrame
from dotenv import load_dotenv

from .definitions      import DataSourceProcessStatus
from .data_sources     import LocalFileDataSource, LocalFileDataSourceConfig
from .data_transformers import DataTransformer, UniqueIDApplier, UniqueIDApplierConfig, DefaultVectorTransformer, VectorDataTransformConfig
from .data_loaders     import QdrantDataLoader, QdrantDataLoadConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class Pipeline:
    """
    Orchestrates the ETL pipeline: Extract → Transform → Load.
    """
    def __init__(
        self,
        data_source: LocalFileDataSource,
        transformers: List[DataTransformer],
        loader: QdrantDataLoader,
    ):
        self.data_source = data_source
        self.transformers = transformers
        self.loader       = loader

    def run(self) -> None:
        try:
            logger.info("▶️  Starting ETL pipeline")

            # 1) Extract
            logger.info("1) Extracting raw data…")
            self.data_source.extract_data()
            raw_df: DataFrame = self.data_source.get_raw_data()
            logger.info(f"   • Got {len(raw_df)} rows")
            # app/ingestion/pipeline.py, inside run(), right after extraction
            raw_df: DataFrame = self.data_source.get_raw_data()
            logger.info(f"   • Got {len(raw_df)} rows, columns: {raw_df.columns.tolist()}")


            # 2) Transform
            df = raw_df
            for tx in self.transformers:
                logger.info(f"2) Applying {tx.__class__.__name__}…")
                df = tx.apply_transformation(df)
                logger.info(f"   → {df.shape[0]} rows × {df.shape[1]} cols")

            # 3) Load
            coll = self.loader.config.collection_name
            logger.info(f"3) Upserting into Qdrant collection '{coll}'…")
            self.loader.load_data(df)
            logger.info(f"   • Upserted {len(df)} points")

            # 4) Mark success
            logger.info("4) Marking source as SUCCESS")
            self.data_source.update_process_status(DataSourceProcessStatus.SUCCESS)
            logger.info("✅ Pipeline executed successfully!")

        except Exception:
            logger.exception("❌ Pipeline failed — marking source as FAILED")
            self.data_source.update_process_status(DataSourceProcessStatus.FAILED)
            raise

if __name__ == "__main__":
    # Load .env for Qdrant credentials
    load_dotenv()

    # — 1) Configure source —
    source_cfg = LocalFileDataSourceConfig(
        source_dir="app/data",
        target_dir="app/data",
        file_names=["RelevanceDataLabelled.csv"],
    )
    source = LocalFileDataSource(source_cfg)

    # — 2) Configure transformers —
    id_applier = UniqueIDApplier(UniqueIDApplierConfig(id_column_name="id"))
    vectorizer = DefaultVectorTransformer(
        VectorDataTransformConfig(
            vectorize_columns=["Text"],
            metadata_columns=["id", "Relevance"],
        )
    )

    # — 3) Configure loader —
    load_cfg = QdrantDataLoadConfig(
        host=os.getenv("QDRANT_HOST"),
        port=int(os.getenv("QDRANT_PORT", "443")),
        collection_name="chatbot",
        vector_size=1536,
        api_key=os.getenv("QDRANT_API_KEY"),  # after the non-default fields
    )
    loader = QdrantDataLoader(load_cfg)

    # — 4) Build & run the pipeline —
    pipeline = Pipeline(
        data_source=source,
        transformers=[id_applier, vectorizer],
        loader=loader,
    )
    pipeline.run()
