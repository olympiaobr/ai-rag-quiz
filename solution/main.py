from rag import QuestionAnsweringRAG
from embedding_db import EmbeddingDatabase
from embedding_model import EmbeddingModel
from llm import LargeLanguageModel
import streamlit as st

embedding_model = EmbeddingModel()
embedding_db = EmbeddingDatabase(embedding_model)
llm = LargeLanguageModel()
rag = QuestionAnsweringRAG(llm, embedding_db)

st.title("Q&A Food RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        msg = st.session_state.messages[-1]["content"]
        response = rag.query(msg)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
