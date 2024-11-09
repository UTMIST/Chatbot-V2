import os
import openai
import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor
)

import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Set API key from environment
os.environ["OPENAI_API_KEY"] = "Your Key"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize the LLM with specific parameters
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)

#Split text into chunks. Need to adjust chunk size!!!
node_parser = TokenTextSplitter(
    separator=" ", chunk_size=500, chunk_overlap=250
)

#Define extractors



extractor_1 = [QuestionsAnsweredExtractor(questions = 3, llm =llm, metadata_mode = MetadataMode.EMBED)]
extractor_2 = [SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
    QuestionsAnsweredExtractor(
        questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
    ),
]
extractor_3 = [TitleExtractor(llm=llm, max_tokens=10, temperature=0.3, top_p=0.9)]
extractor_4 = [KeywordExtractor(llm=llm, max_keywords=10, threshold=0.2, include_scores=True)]

# check if storage already exists
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)








