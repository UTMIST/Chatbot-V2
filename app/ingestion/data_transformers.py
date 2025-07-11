from app.ingestion.definitions import DataTransformConfig, DataTransformer
from dataclasses import dataclass, field
from pandas import DataFrame
import numpy as np
import datetime
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from uuid import uuid4
from typing import Any, List, Dict


def _serialize_value(val: Any) -> Any:
    """
    Convert common non-JSON-serializable types to primitives:
      - numpy scalars -> Python scalars
      - datetime objects -> ISO strings
    """
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    return val

@dataclass
class VectorDataTransformConfig(DataTransformConfig):
    vectorize_columns: List[str]
    metadata_columns: List[str]
    embeddings_model: BaseEmbedding = field(
        default_factory=lambda: OpenAIEmbedding(model="text-embedding-ada-002")
    )
    embeddings_output_colname: str = "embeddings"
    metadata_output_colname: str = "metadata"
    embeddings_text_output_colname: str = "embeddings_text"

class DefaultVectorTransformer(DataTransformer):
    """
    Takes raw DataFrame rows, generates embedding lists, and attaches metadata.
    Ensures output vectors are Python lists and metadata values are JSON-serializable.
    """

    def __init__(self, config: VectorDataTransformConfig):
        super().__init__(config)

    def apply_transformation(self, raw_data: DataFrame) -> DataFrame:
        config: VectorDataTransformConfig = self.config

        # 1) Combine selected columns into single text per row
        texts_to_embed = raw_data[config.vectorize_columns] \
            .astype(str) \
            .agg(" ".join, axis=1)

        # 2) Generate embeddings and ensure they are Python lists
        embeddings: List[List[float]] = []
        for text in texts_to_embed:
            emb = config.embeddings_model.get_text_embedding(text)
            # If numpy array, convert to list; else assume it's already list-like
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            else:
                emb_list = list(emb)
            embeddings.append(emb_list)

        # 3) Build metadata dicts with JSON-serializable values
        metadata_list: List[Dict[str, Any]] = []
        for _, row in raw_data.iterrows():
            md: Dict[str, Any] = {}
            for col in config.metadata_columns:
                if col in row.index:
                    md[col] = _serialize_value(row[col])
            metadata_list.append(md)

        id_list = [md.get("id") for md in metadata_list]
        
        # 4) Assemble the transformed DataFrame
        transformed_df = DataFrame({
            "id": id_list,
            config.embeddings_output_colname: embeddings,
            config.metadata_output_colname: metadata_list,
            config.embeddings_text_output_colname: texts_to_embed
        })

        return transformed_df

@dataclass
class UniqueIDApplierConfig(DataTransformConfig):
    id_column_name: str = "id"

class UniqueIDApplier(DataTransformer):
    """
    Assigns a stable UUID string to each row in the DataFrame.
    Useful for setting the 'id' field when upserting into Qdrant.
    """

    def __init__(self, config: UniqueIDApplierConfig):
        super().__init__(config)

    def apply_transformation(self, raw_data: DataFrame) -> DataFrame:
        config: UniqueIDApplierConfig = self.config
        df_copy = raw_data.copy()
        df_copy[config.id_column_name] = [str(uuid4()) for _ in range(len(df_copy))]
        return df_copy
