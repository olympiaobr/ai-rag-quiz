from chat_solution.embedding_db import EmbeddingDatabase
from chat_solution.llm import LargeLanguageModel

class MyRAG:
    _instance = None

    def get_instance():
        if not MyRAG._instance:
            MyRAG._instance = MyRAG()
        return MyRAG._instance
    
    def __init__(self):
        self.embedding_db = EmbeddingDatabase()
        self.llm = LargeLanguageModel()
        self.conversation_history = []
        self.documents_retrieved = []
        self.instructions = """You are an AI assistant focused on delivering concise, practical answers for advanced AI and machine learning queries.
- Use technical language where appropriate but provide clarifications for niche terms.
- Tailor your responses based on retrieved documents or explicitly state, "I don't have the required information" if the context doesn't support an answer.
- For numerical user input, assume it's related to prior feedback or options and respond accordingly.
"""

        self.examples = """<startexample>
    Interaction 1
    New Context: Neural networks use layers of interconnected nodes to simulate the behavior of the human brain for tasks such as image recognition, natural language processing, and more.
    User input: What are neural networks?
    Assistant:
    Question: What are neural networks?
    1. Networks that transmit data between machines. 
    2. Layers of interconnected nodes simulating the brain's behavior for tasks like image recognition. (CORRECT)
    3. Computer programs that automate simple repetitive tasks.
    4. Databases optimized for natural language queries.
    Interaction 2
    User input: 2
    Assistant: Correct! Neural networks use interconnected layers of nodes to replicate the brain's functionality for AI tasks.
    </endexample>
    """

    def query(self, query: str) -> dict:
        documents = None
        if not query.isnumeric():
            documents = self.embedding_db.retrieve(query)
        self.documents_retrieved = documents

        prompt = self._get_prompt(documents, query)
        response = self.llm.call(prompt)
        self.conversation_history.append((query, response))

        return response

    def _get_prompt(self, documents: str, query: str) -> str:
        chat_history = ""
        i = 1
        for old_query, response in self.conversation_history:
            chat_history += f"Interaction {i}\nUser: {old_query}\nAssistant: {response}\n"
            i += 1
        
        new_context_str = f"\nNew Context: {documents}" if documents else ""

        self.complete_prompt = f"""{self.instructions}
{self.examples}
Now we start the conversation history:
{chat_history}

Just predict the next answer:
Interaction {i+1} {new_context_str}
User input: {query}
Assistant:"""
        return self.complete_prompt
