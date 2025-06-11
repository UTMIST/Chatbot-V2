from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStore, VectorStoreQuery
from retrieval.query_transformers import QueryTransformer


@dataclass
class RetrievalConfig:
    """Configuration for the retriever."""
    top_k: int = 5
    score_threshold: Optional[float] = 0.7
    embedding_model: BaseEmbedding = field(default_factory=lambda: OpenAIEmbedding(model="text-embedding-ada-002"))

    def __post_init__(self):
        self.score_threshold = self.score_threshold or 0

class VectorStoreRetriever:
    """A class to retrieve relevant documents from Qdrant using OpenAI embeddings."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[RetrievalConfig] = None
    ) -> None:
        """
        Initialize the retriever with Qdrant client and configuration.
        
        Args:
            vector_store (VectorStore): The vector store to use
            config (Optional[RetrievalConfig]): Configuration for retrieval
        """
        self.config = config or RetrievalConfig()
        self.vector_store = vector_store

    async def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Retrieve the most relevant documents for a given query.
        
        Args:
            query (str): The search query
            
        Returns:
            List[NodeWithScore]: List of retrieved nodes with their similarity scores
        """ 

        query_embedding = self.config.embedding_model.get_text_embedding(query)
        vec_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.config.top_k,
        )
        query_result = self.vector_store.query(
            vec_store_query
        )

        nodes = query_result.nodes
        nodes_with_scores = [NodeWithScore(node=node, 
                                           score=score) for node, score in zip(nodes, query_result.similarities) if score >= self.config.score_threshold]

        return nodes_with_scores

    def get_metadata_from_nodes(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """
        Extract metadata from a list of NodeWithScore objects.
        
        Args:
            nodes (List[NodeWithScore]): List of retrieved nodes with scores
            
        Returns:
            List[Dict[str, Any]]: List of metadata dictionaries
        """
        return [node.node.metadata for node in nodes]

