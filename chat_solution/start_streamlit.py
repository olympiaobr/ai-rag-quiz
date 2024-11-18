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

# Initialize session state for quiz
if "quiz" not in st.session_state:
    st.session_state.quiz = {
        "topic": "",
        "num_questions": 1,
        "questions": [],
        "current_question": None,
        "current_choices": [],
        "current_correct_answer": None,
        "current_explanation": None,
        "score": 0,
        "total_questions": 0
    }

# Function to generate a new question
def generate_question():
    response = rag.call_llm(st.session_state.quiz["topic"])
    st.session_state.quiz["current_question"] = response["question_text"]
    st.session_state.quiz["current_choices"] = response["answer_choices"]
    st.session_state.quiz["current_correct_answer"] = response["correct_answer"]
    st.session_state.quiz["current_explanation"] = response["explanation"]
    st.session_state.quiz["total_questions"] += 1
    

# Function to evaluate the user's answer
def evaluate_answer(user_answer):
    correct_answer = st.session_state.quiz["current_correct_answer"]
    feedback = rag.evaluate_answer(user_answer, correct_answer)
    st.session_state.quiz["score"] += 1 if user_answer == correct_answer else 0
    return feedback

# User input for topic and number of questions
st.session_state.quiz["topic"] = st.text_input("Enter the topic for the quiz:")
st.session_state.quiz["num_questions"] = st.slider("Select the number of questions:", 1, 5, 1)

if st.button("Start Quiz"):
    generate_question()

# Display the current question and answer options
if st.session_state.quiz["current_question"]:
    st.markdown(st.session_state.quiz["current_question"])
    user_answer = st.radio("Select your answer:", st.session_state.quiz["current_choices"])
    if st.button("Submit Answer"):
        feedback = evaluate_answer(user_answer)
        st.markdown(feedback)
        st.markdown(st.session_state.quiz["current_explanation"])
        if st.session_state.quiz["total_questions"] < st.session_state.quiz["num_questions"]:
            if st.button("Next Question"):
                generate_question()
        else:
            st.markdown(f"You answered {st.session_state.quiz['score']} out of {st.session_state.quiz['num_questions']} questions correctly. Great job, keep it up!")
            st.session_state.quiz = {
                "topic": "",
                "num_questions": 1,
                "questions": [],
                "current_question": None,
                "current_choices": [],
                "current_correct_answer": None,
                "current_explanation": None,
                "score": 0,
                "total_questions": 0
            }