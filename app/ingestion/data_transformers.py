from ingestion.definitions import DataTransformConfig, DataTransformer
from dataclasses import dataclass, field
import numpy as np
from pandas import DataFrame
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from uuid import uuid4


@dataclass
class VectorDataTransformConfig(DataTransformConfig):

    vectorize_columns : list[str]
    metadata_columns : list[str]
    embeddings_model : BaseEmbedding = field(default_factory=lambda: OpenAIEmbedding(model="text-embedding-ada-002"))
    embeddings_output_colname : str = "embeddings"
    metadata_output_colname : str = "metadata"
    embeddings_text_output_colname : str = "embeddings_text"

class DefaultVectorTransformer(DataTransformer):

    def __init__(self, config : VectorDataTransformConfig):
        super().__init__(config)

    def apply_transformation(self, raw_data : DataFrame) -> DataFrame:
        """
        Transform raw data into a DataFrame with embeddings and metadata.
        
        Returns:
            DataFrame: Transformed DataFrame with 'embeddings' and 'metadata' columns
        """
        config : VectorDataTransformConfig = self.config
        
        texts_to_embed = raw_data[config.vectorize_columns].astype(str).agg(' '.join, axis=1)
        embeddings = [
            config.embeddings_model.get_text_embedding(text)
            for text in texts_to_embed
        ]
        
        metadata_dicts = []
        for _, row in raw_data.iterrows():
            metadata_dict = {
                col: row[col] 
                for col in config.metadata_columns 
                if col in row.index
            }
            metadata_dicts.append(metadata_dict)
        
        transformed_df = DataFrame({
            config.embeddings_output_colname: embeddings,
            config.metadata_output_colname: metadata_dicts,
            config.embeddings_text_output_colname: texts_to_embed
        })
        
        return transformed_df
    
@dataclass
class UniqueIDApplierConfig(DataTransformConfig):
    id_column_name: str = "id"

        
class UniqueIDApplier(DataTransformer):
    
    def __init__(self, config : UniqueIDApplierConfig):
        super().__init__(config)

    def apply_transformation(self, raw_data : DataFrame) -> DataFrame:
        config : UniqueIDApplierConfig = self.config
        df_copy = raw_data.copy()
        df_copy[config.id_column_name] = [str(uuid4()) for _ in range(len(df_copy))]
        return df_copy
