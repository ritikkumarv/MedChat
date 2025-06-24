import streamlit as st
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Get Azure credentials from Streamlit secrets
AZURE_SEARCH_ENDPOINT = st.secrets["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY = st.secrets["AZURE_SEARCH_KEY"]
AZURE_INDEX_NAME = "med_chat_index"  # Change if your index name is different

# Set up Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Load Hugging Face model (change model as needed)
@st.cache_resource
def load_model():
    model_name = "microsoft/BiomedLM"  # replace with your chosen model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

generator = load_model()

# Function to get relevant documents from Azure Search
def retrieve_docs(query: str, top_k=3):
    results = search_client.search(search_text=query, top=top_k)
    return [
        f"Q: {doc['question']}\nA: {doc['answer']}"
        for doc in results
        if 'question' in doc and 'answer' in doc
    ]

# Function to generate final answer using RAG logic
def generate_answer(user_question: str):
    context_docs = retrieve_docs(user_question)
    context = "\n".join(context_docs)

    prompt = (
        f"Based on the following medical FAQs, answer the user's question:\n\n"
        f"{context}\n\n"
        f"User Question: {user_question}\n\nAnswer:"
    )

    response = generator(prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]
    return response.split("Answer:")[-1].strip()
