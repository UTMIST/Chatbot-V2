import os

os.environ["OPENAI_API_KEY"] = "your api key here"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=["D:/EngSci/utmistChatbot/test_data.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist()
