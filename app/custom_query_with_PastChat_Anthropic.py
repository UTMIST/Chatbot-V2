from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document
)
import os.path
import os
from enum import Enum
import torch
import re
import json

# Replace OpenAI with Anthropic
from llama_index.llms.anthropic import Anthropic
from llama_index.core import PromptTemplate
from pathlib import Path
from anthropic import Anthropic as AnthropicClient
from dotenv import load_dotenv
import sys

sys.path.append(os.path.abspath(r"app\Classifier Models"))

from get_constraint_classifier_outcome import get_constraint_prediction
from get_intent_classifier_outcome import get_binary_outcome

def strip_whole_str(input_str: str, substr: str) -> str:
    """
    Removes all occurrences of a specified substring from the input string.
    """
    return input_str.replace(substr, '')

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = "Your key"

# Create Anthropic client instance
anthropic_client = AnthropicClient(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# load existing index from storage
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Combine existing index with new index
retriever = index.as_retriever()

# Chat History management using a SimpleChatStore instance
chat_store = SimpleChatStore()

# Function to get the memory module for a given user.
def get_memory_module(userID: str):
    if userID not in chat_store.get_keys():
        return ChatMemoryBuffer.from_defaults(
            token_limit=1500,
            chat_store=chat_store,
            chat_store_key=userID,
        )
    else:
        chat_memory = ChatMemoryBuffer(token_limit=1500)
        chat_memory.set(chat_store.store.get(userID))
        return chat_memory

# Add new messages to the user's chat history.
def update_chat_history(userID: str, role: str, message: str) -> None:
    chat_memory = get_memory_module(userID)
    if role == 'user':
        chat_memory.put(ChatMessage(role=MessageRole.USER, content=message))
    elif role == 'bot':
        chat_memory.put(ChatMessage(role=MessageRole.CHATBOT, content=message))

# Retrieve the chat history for a specific user.
def get_chat_history(userID: str) -> list[ChatMessage]:
    if userID not in chat_store.get_keys():
        return []
    return chat_store.store.get(userID)

# NEW: Function to clear chat history for a specific user.
def clear_chat_history(userID: str) -> None:
    if userID in chat_store.store:
        chat_store.store.pop(userID)
        print(f"Chat history for user '{userID}' has been cleared.")

def embed_chat_history(chat_history):
    pass

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: Anthropic  # Changed from OpenAI to Anthropic
    qa_prompt: PromptTemplate

    # Custom query that incorporates chat history
    def custom_query(self, prompt_text):
        response = self.llm.complete(prompt_text)
        return str(response)

def determine_response_length(query: str) -> str:
    """
    Determines whether the user's query is general or specific.
    """
    CLASSIFICATION_PROMPT = """
You are an assistant that classifies user queries into "general" or "specific" for a chatbot that answers questions regarding a club, UTMIST.

- A "general" query asks for broad information or an overview of the club.
- A "specific" query asks for detailed information about a particular topic or they ask you to elaborate on a specific topic.

Given the following query, classify it accordingly.

Query: "{query}"

Classification (general/specific):
"""
    prompt = CLASSIFICATION_PROMPT.format(query=query)
    response = get_anthropic_response_content(
        # in anthropic, system prompt is separate from messages, we can choose to add system prompt here
        system_prompt="",
        messages=[{"role": "user", "content": prompt}]
    )
    classification = response.strip().lower()
    if "specific" in classification:
        return "specific"
    elif "general" in classification:
        return "general"
    else:
        # Default to "general" if the classification is unclear
        return "general"

# Change from OpenAI to Anthropic model
llm = Anthropic(model="claude-3-sonnet-20240229", max_tokens=500)
synthesizer = get_response_synthesizer(response_mode="compact")

# Intent Classification: Process one sentence at a time.
def classify_intent(sentence: str) -> str:
    intent_result = get_binary_outcome(sentence)
    return intent_result

# Constraints Classification: Process one sentence at a time.
def classify_constraints(sentence, intent):
    if intent['Provide_Preference'] == 1:
        constraint_result = get_constraint_prediction(sentence)
    else:
        constraint_result = "N/A"
    return constraint_result

# Update temporary JSON file using a user-specific filename (userID.json)
def update_temp_json(new_entry: dict, userID: str):
    temp_file = f"{userID}.json"
    if os.path.exists(temp_file):
        with open(temp_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(new_entry)
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)

# Classification Results: Process full user input (may be multiple sentences)
def get_classification_results(input_text, userID):
    """
    Splits the full user input into sentences, classifies each sentence using the intent and
    constraint classifiers, and combines the results. The combined result is stored in a 
    user-specific temporary JSON file.
    
    Returns:
      - results_list: A list of classification results for each sentence.
      - combined_results: A single string combining all sentence results.
    """
    # Split the full input into sentences (using a simple regex)
    sentences = re.split(r'(?<=[.!?])\s+', input_text.strip())
    
    results_list = []
    combined_results_lines = []
    intent_res = ""
    
    # Process each sentence individually.
    for sentence in sentences:
        if not sentence:
            continue
        intent_res = classify_intent(sentence)
        constraint_res = classify_constraints(sentence, intent_res)
        result_str = f"Sentence: {sentence}\nIntent: {intent_res}\nConstraints: {constraint_res}"
        results_list.append(result_str)
        combined_results_lines.append(result_str)
    
    combined_results = "\n\n".join(combined_results_lines)
    
    # Store the combined results in the user-specific temporary JSON file.
    new_entry = {
        "input_text": input_text,
        "classification_results": combined_results,
    }
    update_temp_json(new_entry, userID)
    
    return results_list, combined_results, intent_res

def generate_missing_info_prompt(combined_results, query):
    """
    Sends the combined classification details to Claude and asks it to identify any crucial details 
    that might be missing (such as device type, education level, language preference, etc.).
    """
    prompt = PromptTemplate(
        "Based on the following classification details with 1 being true and 0 being false\n"
        "{combined_results}\n"
        "This is the user query\n"
        "{query}\n\n"
        "Identify any crucial details that might be missing and would be of help to answer the user query and that are needed to provide an accurate recommendation. "
        "Consider details such as device type, education level, language preference, time availability, etc. "
        "If you find missing details, ask the user for them; if no additional information is needed, simply output 'None'."
    )
    try:
        prompt_formatted = prompt.format(combined_results=combined_results, query=query)
        local_prompt = PromptTemplate(prompt_formatted)
        query_engine = RAGStringQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=local_prompt,
        )
        missing_info = query_engine.custom_query(prompt_formatted)
        return missing_info
    except Exception as e:
        print(f"Error in generate_missing_info_prompt: {e}")
        return "Could not determine missing information. Please provide any relevant details that might be missing."

def Classify_Action(combined_results: str) -> str:
    prompt = PromptTemplate(
        "Below are the combined classification results derived from the user's query:\n"
        "{combined_results}\n\n"
        "Based on these results, determine which of the following actions the bot should take:\n"
        "1. Request More Information\n"
        "2. Generate Recommendation\n"
        "3. Answer a Question\n"
        "4. Other/Quick Response\n\n"

        "Consider the following:\n"
        "- When little is known about the user and their preferences (such as device type, education level, language preference, time availability, budget and other preferences), output 'Request More Information'.\n"
        "- If the user appears to be seeking personalized recommendation and only when sufficient information is present, output 'Generate Recommendation'.\n"
        "- If they want a recommendation, but not enough information is present, output 'Request More Information'.\n"
        "- If the query is straightforward and answerable, output 'Answer a Question'.\n"
        "- If the input does not fall into any of the categories above, classify as 'Other/Quick Response'.\n"
        "Please output exactly one of these phrases with exact capitalization."
    )
    prompt_formatted = prompt.format(combined_results=combined_results)
    local_prompt = PromptTemplate(prompt_formatted)
    query_engine = RAGStringQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        llm=llm,
        qa_prompt=local_prompt,
    )
    classification = query_engine.custom_query(prompt_formatted)
    return classification

# aiResponse combined with past chat history
def aiResponse(input, userID):
    # For debugging: print the current chat_store content.
    print("Current chat_store:", chat_store.store)
    
    # Optionally, clear the chat history for a new conversation.
    # Uncomment the next line if you want to clear previous history:
    # clear_chat_history(userID)

    # Load any existing temporary classification results.
    user_file = f"{userID}.json"
    if os.path.exists(user_file):
        with open(user_file, "r") as f:
            past_results = json.load(f)
    else:
        past_results = []

    # Get classification results for the current user input.
    results_list, combined_results, intent_res = get_classification_results(input, userID)
    # Optionally, you can combine past temporary results if desired:
    results_list.extend(past_results)

    classified_action = Classify_Action(combined_results)
    print(combined_results)
    print("Classified Action:", classified_action)

    # Retrieve past conversations from the chat_store.
    past_context_str = "\n\n".join([f"{msg.role.value}: {msg.content}" for msg in get_chat_history(userID=userID)])
    # Retrieve context from the vector database.
    nodes = retriever.retrieve(input)
    context_str = "\n\n".join([n.node.get_content() for n in nodes])
    print("Past Chat History:", past_context_str)
    print("Vector Context:", context_str)
    
    query_str = input
    past_context = past_context_str  # Use the actual past context
    
    if classified_action == "Answer a Question":
        print("intents", intent_res)
        if(intent_res["Club_Related_Inquiry"] == 1):
            qa_prompt = PromptTemplate(
                "Below are the combined classification results derived from the user's query:\n"
                "{combined_results}\n\n"
                "Below is the relevant context retrieved from the vector database:\n"
                "{context_str}\n\n"
                "The user's question is:\n"
                "{query_str}\n\n"
                "This is the user's Chat History:\n"
                "{past_context}\n\n"
                "Based on the above information, provide a clear, concise, and accurate answer to the question. "
                "Make sure to use the context provided from the vector database to enrich your answer. "
                "Keep your response simple and limited to 100 words."
            )
        else:
            qa_prompt = PromptTemplate(
                "Below are the combined classification results derived from the user's query:\n"
                "{combined_results}\n\n"
                "The user's question is:\n"
                "{query_str}\n\n"
                "This is the user's Chat History:\n"
                "{past_context}\n\n"
                "Based on the above information, provide a clear, concise, and accurate answer to the question. "
                "Make sure to use the context provided from the vector database to enrich your answer. "
                "Keep your response simple and limited to 100 words."
            )
    elif classified_action == "Generate Recommendation":
        qa_prompt = PromptTemplate(
            "Below are the combined classification results for a user's query:\n"
             "{combined_results}\n\n"
             "Below is the relevant context retrieved from the vector database:\n"
             "{context_str}\n\n"
             "The user's question is:\n"
             "{query_str}\n\n"
             "This is the user's Chat History:\n"
             "{past_context}\n\n"
            "Based on these classification results, generate a personalized recommendation. "
            "Your recommendation should consider the user's identified intent (e.g., Provide Preference, Inquire Resources, etc.) "
            "and any constraints such as education level, resource type, topic, language, budget, learning style, time commitment, "
            "level of depth, preferred topics, and format preferences. "
            "Please provide an actionable recommendation that is clear, concise, and tailored to the user's needs. "
            "Limit your response to 100 words maximum."
        )
    elif classified_action == "Other/Quick Response":
        qa_prompt = PromptTemplate(
            "Below is the user input, respond in a short and concise manner.\n"
            "{query_str}\n"
            "Limit your response to 50 words maximum."
        )
        qa_prompt_formatted = qa_prompt.format(query_str=input)
        qa_prompt_local = PromptTemplate(qa_prompt_formatted)
        query_engine = RAGStringQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt_local,
        )
        response = query_engine.custom_query(qa_prompt_formatted)
        return response
    else:
        # If classified action doesn't fall into the above categories, ask for missing info.
        missing_info_prompt = generate_missing_info_prompt(combined_results, input)
        return missing_info_prompt

    qa_prompt_formatted = qa_prompt.format(
        combined_results=combined_results, 
        past_context=past_context, 
        context_str=context_str, 
        query_str=query_str)
    
    qa_prompt_local = PromptTemplate(qa_prompt_formatted)
    query_engine = RAGStringQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        llm=llm,
        qa_prompt=qa_prompt_local,
    )
    print(qa_prompt_formatted, "Prompt")
    response = query_engine.custom_query(qa_prompt_formatted)
    return response

class Relevance(Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    IRRELEVANT = "irrelevant"

def classifyRelevance(input, retriever=retriever) -> Relevance:
    """
    Classifies how relevant a particular user input is to UTMIST.
    """
    RELEVANCE_PROMPT = """You are talking to a user as a representative of a club called the University of Toronto Machine Intelligence Team (UTMIST). 

Your job is to determine whether the user's query is relevant to any of the following, and output one of the responses according to the possible scenarios.

1. AI and machine learning related questions
2. UTMIST club information and events

Note that when the user refers to "you" or "your", they are referring to UTMIST.

<possible scenarios>
1. SCENARIO: If the query seems relevant to UTMIST (i.e. events or general info) or AI and the "context" explicitly contains information about the query; OUTPUT: "known"
2. SCENARIO: If the query is about GENERAL knowledge in AI/ML but NOT about UTMIST; OUTPUT: "known"
3. SCENARIO: If the query seems relevant to UTMIST or AI but the information is not in "context" AND it is NOT GENERAL knowledge about AI/ML; OUTPUT: "unknown"
4. SCENARIO: If the query is completely irrelevant to the criteria above; OUTPUT: "irrelevant"
</possible scenarios>

Example A:

<context>
UTMIST is a club to help students learn about AI
</context>
Query: When was UTMIST founded?
Output: unknown

Example B:

<context>
The GenAI conference will be on April 30, 2024
</context>
Query: What do you know about history?
Output: irrelevant

Example C:

<context>
The GenAI conference will help students learn about AI.
</context>
Query: What is the GenAI conference?
Output: relevant

### END OF EXAMPLES ###"""

    nodes = retriever.retrieve(input)
    context_str = "\n\n".join([n.node.get_content() for n in nodes])
    user_query = f"""<context>
{context_str}
</context>

Query: {input}

Output: """
    # Retry up to 3 times in case Claude returns an unexpected value.
    for i in range(3):
        response: str = get_anthropic_response_content(
            system_prompt=RELEVANCE_PROMPT, 
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": user_query}]
        )
        response = strip_whole_str(response, "Output:").strip()
        try:
            return Relevance(response)
        except ValueError:
            print("value error: " + response)
            pass
    return Relevance.UNKNOWN

def get_response_with_relevance(input: str, past_chat_history=[], retriever=retriever) -> str:
    relevance = classifyRelevance(input, retriever=retriever)
    print("relevance: " + str(relevance))
    if relevance == Relevance.KNOWN:
        return aiResponse(input, userID="default")
    elif relevance == Relevance.UNKNOWN:
        return get_unknown_response(input, past_chat_history, retriever)
    else:
        return "I'm sorry, I cannot answer that question as I am only here to provide information about UTMIST and AI/ML. If you think this is a mistake, please contact the UTMIST team."

def get_unknown_response(latest_user_input: str, past_chat_history=[], retriever=retriever) -> str:
    UNKNOWN_RESPONSE_PROMPT = """You are talking to a student as a representative of the University of Toronto Machine Intelligence Team (UTMIST), a student group dedicated to educating students about AI/ML through various events (conferences, workshops), academic programs, and other initiatives. 

Given the chat history, you must try to answer the user's latest inquiry using your knowledge of AI and nothing else. This means you must not use knowledge on any other topic other than UTMIST, AI, and/or machine learning.

If you cannot answer the user's query using your knowledge of AI/ML, you must tell the student that you don't know the answer.

<RULES>
1. DO NOT provide any information that is not related to AI/ML and/or computer science.
2. DO NOT make up information that you do not 100% know to be true.
</RULES>"""
    messages = list(past_chat_history)
    messages.append({"role": "user", "content": latest_user_input})
    return get_anthropic_response_content(
        system_prompt=UNKNOWN_RESPONSE_PROMPT, 
        model="claude-3-sonnet-20240229", 
        messages=messages
    )

def get_anthropic_response_content(system_prompt="", messages=[], model="claude-3-sonnet-20240229", **kwargs) -> str:
    assert messages or system_prompt, "prompt or messages must be provided"
    
    anthropic_messages = []
    for msg in messages:
        if msg["role"] != "system":
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        elif not system_prompt:  # If no system prompt was provided, use the one from messages
            system_prompt = msg["content"]
        
    response = _get_anthropic_response(
        messages=anthropic_messages,
        system=system_prompt, 
        model=model, 
        **kwargs
    )
    return _extract_anthropic_response_content(response)

def _extract_anthropic_response_content(response) -> str:
    # Anthropic's response structure has content as a list of message objects
    return response.content[0].text

def _get_anthropic_response(messages=[], system="", model="claude-3-sonnet-20240229", **kwargs):
    # Make API call with Anthropic's expected format
    try:
        response = anthropic_client.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=1000,  # Default max tokens
            **kwargs
        )
        return response
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        # Return a simple mock response in case of failure
        from types import SimpleNamespace
        content = SimpleNamespace()
        content.text = "Error processing your request"
        mock_response = SimpleNamespace()
        mock_response.content = [content]
        return mock_response