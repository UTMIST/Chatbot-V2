from llama_index.llms.openai import OpenAI
from llama_index.core.llms.llm import LLM
from jinja2 import Template
from abc import ABC, abstractmethod


class QueryTransformer(ABC):
    @abstractmethod
    def transform_query(self, original_query: dict) -> str:
        pass

class LLMQueryTransformer(QueryTransformer):


    def __init__(self, 
                 llm: LLM,
                 prompt_template: Template):
        self.prompt_template = prompt_template
        self.llm = llm


    def transform_query(self, original_query: dict) -> str:
        prompt = self.prompt_template.render(**original_query)
        response = self.llm.complete(prompt)
        return response.text
    


