# rag_utils.py

import streamlit as st
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from transformers import pipeline

# Azure Search configuration from Streamlit secrets
AZURE_SEARCH_ENDPOINT = st.secrets["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY = st.secrets["AZURE_SEARCH_KEY"]
AZURE_SEARCH_INDEX = "med_chat_index"

# Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Hugging Face model pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Chat history storage
chat_history = []

# Document retrieval function
def retrieve_documents(query, top_k=3):
    results = search_client.search(query, top=top_k)
    docs = []
    for result in results:
        question = result.get('question', '')
        answer = result.get('answer', '')
        docs.append(f"Q: {question}\nA: {answer}")
    return "\n\n".join(docs)

# Main generation function
def generate_chat_response(user_query):
    context = retrieve_documents(user_query)
    history_text = "\n".join(chat_history[-4:])  # last 4 lines of conversation
    prompt = f"""You are a helpful medical assistant. Use the following history and knowledge base to answer the question.

Conversation so far:
{history_text}

Knowledge base:
{context}

User: {user_query}
Assistant:"""

    response = generator(prompt, max_new_tokens=256)[0]['generated_text']
    chat_history.append(f"User: {user_query}")
    chat_history.append(f"Assistant: {response}")
    return response
