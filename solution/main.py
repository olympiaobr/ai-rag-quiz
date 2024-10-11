from rag import QuestionAnsweringRAG
import streamlit as st


rag = QuestionAnsweringRAG()

st.title("Q&A RAG")

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
        response = rag.call(msg)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
