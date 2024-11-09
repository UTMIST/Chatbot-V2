import os
from datetime import datetime
import tokenize
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from pathlib import Path
from openai import OpenAI as openai_client
from dotenv import load_dotenv
import faiss
import numpy as np
import torchtest
# Load environment variables
env_path = Path("..") / ".env"
load_dotenv(dotenv_path=env_path)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "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"

#add KEY before run!!

openai_client_instance = openai_client(api_key=os.environ.get("OPENAI_API_KEY"))

# Load documents
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
retriever = index.as_retriever()

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])

        if not nodes:  # If no nodes are found, log the query as unanswered
            with open("unanswered_queries.txt", "a") as f:
                f.write(f"{query_str}\n")

        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)

llm = OpenAI(model="gpt-3.5-turbo")
synthesizer = get_response_synthesizer(response_mode="compact")

rag_query_engine = RAGStringQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=qa_prompt,
)

# Temporary storage for new messages
new_messages = []

def ai_response(input):  
    response = rag_query_engine.query(str(input))
    # Store new query-response pair
    new_messages.append((input, response))
    return response

def update_vector_database():
    global new_messages
    index_path = 'path_to_your_faiss_index'
    document_store_path = 'path_to_your_document_store.npy'
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(768)  # Dimension of embeddings, adjust accordingly
    if os.path.exists(document_store_path):
        document_store = np.load(document_store_path, allow_pickle=True).tolist()
    else:
        document_store = []

    # Add new messages to the FAISS index and document store
    for message, response in new_messages:
        inputs = tokenize(message, return_tensors='pt', padding=True, truncation=True)
        with torchtest.no_grad():
            embeddings = model.encoder(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        index.add(embeddings)
        document_store.append((message, response))

    # Save updated FAISS index and document store
    faiss.write_index(index, index_path)
    np.save(document_store_path, np.array(document_store))

    # Clear new messages
    new_messages = []

async def save_unanswered_queries():
    now = datetime.now()
    unanswered_queries_path = "unanswered_queries.txt"
    if os.path.exists(unanswered_queries_path):
        with open(unanswered_queries_path, 'r') as f:
            queries = f.readlines()
        if queries:
            save_path = f"unanswered_queries_{now.strftime('%Y%m%d')}.txt"
            with open(save_path, 'w') as f:
                f.writelines(queries)
            os.remove(unanswered_queries_path)
