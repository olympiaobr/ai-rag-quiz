import json

from chat_solution.embedding_db import EmbeddingDatabase

class LearningAssistant:
    def __init__(self, embedding_db, llm):
        self.embedding_db : EmbeddingDatabase = embedding_db

        self.llm = llm
        self.conversation_history = []

    def _create_question_prompt(self, documents: str, query: str) -> str:

        chat_history = ""
        i = 1
        for query, response in self.conversation_history:
            chat_history += f"Interaction {i}\nUser: {query}\nAssistant: {response}\n"
            i += 1

        return f""" You are a helpful AI knowledge quiz chat assistant. Your goal is to test the knowledge of the users on the given topic."
The user gives you a topic or a question and you should generate a new relevant quiz based on the context.
- Generate a new relevant quiz question based on the context and the topic provided. The topic you select on the context does not need to be exactly the same, but should be related to the topic.
- The question should be relevant to the given topic and the answer should be found within the given context. If not say: "You did not equip me with the knowledge to answer this question."
- Provide 4 answer choices for the question, one of which should be correct and the other three should be incorrect but plausible. Answer choices should be formulated clearly and concisely.
- Mark the index of the correct answer in the answer choices list with the pattern (CORRECT) in the end
- Provide an explanation for the correct answer after the user selected an answer. The explanation should give the user additional context and help them better understand the topic.
if the user answers with a number, it is because they selected an answer to the previous question. In this case, you should evaluate if the answer is correct or not and provide feedback to the user.
- Do not mention the context in your response.

Example below:
Interaction 1
New Context: LLMs are large language models that can generate responses to user queries. They are trained on massive datasets to learn patterns, structures, and relationships in text. They can generate responses by combining language generation with real-time data retrieval.
User input: How do LLMs generate responses?
Assistant:
Question: How do LLMs generate responses?
1. LLMs generate responses by searching the internet for relevant information. 
2. LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets. (CORRECT)
3. LLMs generate responses by combining language generation with real-time data retrieval.
4. LMs generate responses by using a predefined set of rules and templates.
Interaction 2
User input: 3
Assistant:Incorrect! There is no need of realtime data retrieval.
Interaction 3
User input: 2
Assistant:Correct! LLMs generate responses by learning patterns, structures, and relationships in text from massive datasets.

Now we start the conversation history.
{chat_history}

Just with the next answer
New Context: {documents}
Interaction {i+1}
User input: {query}
Assistant:"""

    def call_llm(self, query: str) -> dict:
        documents = self.embedding_db.retrieve(query)
        print(documents)
        # only append the last 1 messages to the context
        prompt = self._create_question_prompt(documents, query)
        response = self.llm.call(prompt)
        self.conversation_history.append((query, response))

        return response
