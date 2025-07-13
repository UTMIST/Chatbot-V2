from ingestion.pipeline import IngestionPipeline, PipelineConfig
from ingestion.data_sources import LocalFileDataSource, LocalFileDataSourceConfig
from ingestion.data_transformers import VectorDataTransformConfig, DefaultVectorTransformer, UniqueIDApplier, UniqueIDApplierConfig
from ingestion.data_loaders import VectorStoreDataLoader, VectorStoreDataLoaderConfig
import os
from dotenv import load_dotenv
from chatbot_convrec.defaults import DATA_SOURCE_FOLDER, DATA_SOURCE_FINISHED_FOLDER, VEC_STORE

if __name__ == "__main__":


    data_source_config = LocalFileDataSourceConfig(
        source_dir=DATA_SOURCE_FOLDER,
        target_dir=DATA_SOURCE_FINISHED_FOLDER,
        file_names = [

            *os.listdir(DATA_SOURCE_FOLDER)

        ]
    )


    unique_id_config = UniqueIDApplierConfig(
        id_column_name="id"
    )

    transform_config = VectorDataTransformConfig(

        vectorize_columns = [

            "Description"

        ],

        metadata_columns = [

            "id",
            "Link"
        ],

        embeddings_output_colname="embeddings",
        metadata_output_colname="metadata",
        embeddings_text_output_colname="embeddings_text"
    )

    load_config = VectorStoreDataLoaderConfig(
        vector_store=VEC_STORE,
        embeddings_colname="embeddings",
        metadata_colname="metadata",
        embeddings_text_colname="embeddings_text"
    )

    pipeline_config = PipelineConfig(
        source_config=data_source_config,
        transform_configs=[unique_id_config, transform_config],
        load_config=load_config
    )

    pipeline : IngestionPipeline = IngestionPipeline.from_config(pipeline_config,
                                             source_class=LocalFileDataSource,
                                             transformer_classes=[UniqueIDApplier, DefaultVectorTransformer],
                                             loader_class=VectorStoreDataLoader)
    pipeline.run()

    data_source : LocalFileDataSource = pipeline.data_source

    data_source.save_transformed_data(pipeline.processed_data)




