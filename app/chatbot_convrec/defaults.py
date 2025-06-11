
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from retrieval.query_transformers import LLMQueryTransformer
from jinja2 import Template
from retrieval.retriever import VectorStoreRetriever
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
from llama_index.core.vector_stores.simple import SimpleVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

# VEC_STORE = QdrantVectorStore(
#     client=QdrantClient(url=os.environ.get("QDRANT_URL"),
#                         api_key=os.environ.get("QDRANT_API_KEY")),
#     collection_name="test_collection_1",
# )
VEC_STORE = SimpleVectorStore.from_persist_path("storage/vector_store.json")

LLM = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
DATA_SOURCE_FOLDER = "/app/data/input"
DATA_SOURCE_FINISHED_FOLDER = "/app/data/finished"
CONSTRAINTS_TO_QUERY_PROMPT_PATH = "app/prompts/constraints_to_query.jinja"

DEFAULT_CONSTRAINTS_TO_QUERY_TRANSFORMER = LLMQueryTransformer(
    llm=LLM,
    prompt_template=Template(open(CONSTRAINTS_TO_QUERY_PROMPT_PATH).read())
)

DEFAULT_RECOMMENDATION_RETRIEVER = VectorStoreRetriever(
    vector_store=VEC_STORE
)
