import json

class LearningAssistant:
    def __init__(self, embedding_db, llm):
        self.embedding_db = embedding_db
        self.llm = llm

    def _create_question_prompt(self, context: str, topic: str) -> str:
        return """
    You are a helpful assistant as part of a learning application. Your goal is to test the knowledge of the users on the given topic."
        Your instructions:
        1. Generate a new relevant question based on the context and the topic provided. The question should be relevant to the given topic and the answer should be found within the given context. In case the selected topic is not found in the context, say that the topic is not found and give a list of possible topics consisting of 2-3 most relevant examples. You are only to generate questions based on topics which are found in the context. If the topic is found within the context, only generate questions for which the answer can also be found within the context.
        2. Provide 4 answer choices for the question, one of which should be correct and the other three should be incorrect but plausible. Answer choices should be formulated clearly and concisely.
        3. From the list of answer choices, select the list index of the correct answer (0, 1, 2, or 3) and return it as an integer.
        4. Provide an explanation for the correct answer. The explanation should give the user additional context and help them better understand the topic.
        5. Return the output in a dictionary format with the structure as in example below:
        
    {
    "question_text": "What is the primary process by which LLMs generate human-like responses?",
    "answer_choices": [
        "LLMs generate responses by searching the internet for relevant information.",
        "LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets.",
        "LLMs generate responses by combining language generation with real-time data retrieval.",
        "LLMs generate responses by using a predefined set of rules and templates."
    ],
    "correct_answer_idx": 1,
    "explanation": "Large Language Models (LLMs) generate human-like responses by learning patterns, structures, and relationships in text from massive datasets. This process involves training the model on a vast amount of text data, allowing it to predict and generate language based on given prompts. Understanding this fundamental mechanism is crucial for comprehending how LLMs operate and the nature of the responses they produce."
    }
    6. You are only to return the dictionary as the output.

    """+f"""
        Context: {context}
        Topic: {topic}
        """

    def call_llm(self, topic: str) -> dict:
        documents = self.embedding_db.retrieve(topic)
        context = "\n".join(documents)
        prompt = self._create_question_prompt(context, topic)
        response = self.llm.call(prompt)
        print(response)
        
        try:
            response_dict = eval(response)
            print(response_dict)
            question_text = response_dict["question_text"]
            answer_choices = response_dict["answer_choices"]
            correct_answer = response_dict["answer_choices"][response_dict["correct_answer_idx"]]
            explanation = response_dict["explanation"]
            result = {
                "question_text": question_text,
                "answer_choices": answer_choices,
                "correct_answer": correct_answer,
                "explanation": explanation
            }
            print(result)
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            return {}
        
        
    def evaluate_answer(self, user_answer: str, correct_answer: str) -> str:
        """Evaluate the user's answer and provide feedback."""
        if user_answer == correct_answer:
            return "Correct! Well done."
        else:
            return f"Incorrect. The correct answer is: {correct_answer}"   


