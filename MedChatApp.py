# app.py

import streamlit as st
from rag_utils import generate_chat_response

st.set_page_config(page_title="MedChat Assistant", layout="centered")

st.title("🩺 MedChat - RAG-based Medical Assistant")
st.markdown("Ask any medical question. Powered by Azure Cognitive Search + Hugging Face.")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Input form
user_query = st.text_input("Ask a medical question:")

if user_query:
    response = generate_chat_response(user_query)
    st.session_state.chat_log.append(("You", user_query))
    st.session_state.chat_log.append(("Assistant", response))

# Display chat history
for role, message in st.session_state.chat_log:
    with st.chat_message(role):
        st.write(message)
