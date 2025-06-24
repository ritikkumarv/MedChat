import streamlit as st
from rag_utils import generate_answer

st.title("🩺 Medical RAG Chatbot")

user_question = st.text_input("Ask a medical question:")

if user_question:
    with st.spinner("Searching and generating answer..."):
        final_answer = generate_answer(user_question)
        st.success(final_answer)
