from retrieval.retriever import RetrievalConfig, VectorStoreRetriever
from retrieval.query_transformers import LLMQueryTransformer
from llama_index.core.schema import NodeWithScore
from jinja2 import Template
from chatbot_convrec.defaults import DEFAULT_CONSTRAINTS_TO_QUERY_TRANSFORMER, DEFAULT_RECOMMENDATION_RETRIEVER


def retrieve_recommendation(constraints: dict,
                            user_query: str,
                            constraints_to_query_transformer: LLMQueryTransformer = DEFAULT_CONSTRAINTS_TO_QUERY_TRANSFORMER,
                            vec_retriever: VectorStoreRetriever = DEFAULT_RECOMMENDATION_RETRIEVER) -> list[NodeWithScore]:

    query = constraints_to_query_transformer.transform_query(

        {
            "constraints": constraints
        }
    )
    query = "User query: " + user_query + "Constraints: " + query
    return vec_retriever.retrieve(query)
