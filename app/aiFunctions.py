# This takes the information from the vector database and an input 
# and returns the desired output
import os.path
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
os.environ["OPENAI_API_KEY"] = "Your Key"

# check if storage already exists
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Either way we can now query the index
def aiResponse(input):
    query_engine = index.as_query_engine()
    response = query_engine.query(input)
    return response
