import json

from chat_solution.embedding_db import EmbeddingDatabase

class LearningAssistant:
    def __init__(self, embedding_db, llm):
        self.embedding_db : EmbeddingDatabase = embedding_db

        self.llm = llm
        self.conversation_history = []

    def _create_question_prompt(self, context: str, query: str) -> str:

        chat_history = "\n".join([f"user: {query}\nassistant: {response}" for query, response in self.conversation_history])

        return f""" You are a helpful AI knowledge quiz chat assistant. Your goal is to test the knowledge of the users on the given topic."
When the user asks for a topic (the topic could also come in the form of a question), you should:
- Generate a new relevant question based on the context and the topic provided. The topic you select on the context does not need to be exactly the same, but should be related to the topic.
The question should be relevant to the given topic and the answer should be found within the given context. In case the selected topic is not found in the context, say that the topic is not found and give a list of possible topics consisting of 2-3 most relevant examples. You are only to generate questions based on topics which are found in the context. If the topic is found within the context, only generate questions for which the answer can also be found within the context.
- Provide 4 answer choices for the question, one of which should be correct and the other three should be incorrect but plausible. Answer choices should be formulated clearly and concisely.
- Mark the index of the correct answer in the answer choices list with the pattern (CORRECT) in the end
- Provide an explanation for the correct answer after the user selected an answer. The explanation should give the user additional context and help them better understand the topic.
if the user answers with a number, it is because they selected an answer to the previous question. In this case, you should evaluate if the answer is correct or not and provide feedback to the user.

<newquestion>
Chat history: 
New Context: the context provided to you
User input: How do LLMs generate responses?
Assistant:<startanswer>
1. LLMs generate responses by searching the internet for relevant information. 
2. LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets. (CORRECT)
3. LLMs generate responses by combining language generation with real-time data retrieval.
4. LMs generate responses by using a predefined set of rules and templates.<endanswer>
<endquestion>
<newquestion>
Chat history:<starthistory>
user: How do LLMs generate responses?
assistant: 1. LLMs generate responses by searching the internet for relevant information. 
2. LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets. (CORRECT)
3. LLMs generate responses by combining language generation with real-time data retrieval.
4. LMs generate responses by using a predefined set of rules and templates.<endhistory>
New Context: the context provided to you
User input: 2.
Assistant:<startanswer>Correct! LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets.
<endanswer>
<endquestion>

Just continue with 1 question at a time. Be direct and concise.
Chat history: {chat_history}
New Context: {context}
User input: {query}
Assistant:<startanswer>"""

    def call_llm(self, query: str) -> dict:
        documents = self.embedding_db.retrieve(query)

        context = "\n".join(documents)
        # only append the last 1 messages to the context
        prompt = self._create_question_prompt(context, query)
        response = self.llm.call(prompt)
        self.conversation_history.append((query, response))

        return response
