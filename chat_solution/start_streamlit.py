import streamlit as st

from embedding_db import EmbeddingDatabase
from embedding_model import EmbeddingModel
from llm import LargeLanguageModel
from rag import LearningAssistant

# Initialize models and RAG
embedding_model = EmbeddingModel()
embedding_db = EmbeddingDatabase()  # Remove the embedding_model argument
llm = LargeLanguageModel()
rag = LearningAssistant(embedding_db, llm)

# Streamlit app title
st.title("LLM Workshop Quiz")
st.write("""
Guide this chatbot to answer questions about LLMs that are available in the context.
Say something like "LLMs" or "rag", "ai ethics", 'role models', ,etc
""")

# Initialize session state for quiz
# add the messages to the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# append to the chat history and display the conversations  
if prompt:= st.chat_input("Ask for a quiz question"):
    st.write(f"User: {prompt}")
    answer = rag.call_llm(prompt)
    # remove any reference to the correct answer
    answer = answer.replace("(CORRECT)", "")
    st.write(answer)
