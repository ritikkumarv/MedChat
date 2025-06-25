# rag_utils.py

import streamlit as st
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# Load Azure secrets
AZURE_SEARCH_ENDPOINT = st.secrets["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY = st.secrets["AZURE_SEARCH_KEY"]
AZURE_SEARCH_INDEX = "med_chat_index"

AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = st.secrets["AZURE_OPENAI_DEPLOYMENT"]

# Configure Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2025-01-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# History
chat_history = []

# Retrieve docs
def retrieve_documents(query, top_k=3):
    results = search_client.search(query, top=top_k)
    docs = []
    for result in results:
        question = result.get('question', '')
        answer = result.get('answer', '')
        docs.append(f"Q: {question}\nA: {answer}")
    return "\n\n".join(docs)

# Generate response using Azure OpenAI
def generate_chat_response(user_query):
    context = retrieve_documents(user_query)
    history_text = "\n".join(chat_history[-4:])  # last 4 lines of convo

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant. Use the provided knowledge base to answer user questions accurately."},
            {"role": "system", "content": f"Knowledge base:\n{context}"},
            {"role": "user", "content": user_query}
        ],
        temperature=0.5,
        max_tokens=2000
    )

    answer = response.choices[0].message.content.strip()
    chat_history.append(f"User: {user_query}")
    chat_history.append(f"Assistant: {answer}")
    return answer
