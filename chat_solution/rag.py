from chat_solution.embedding_db import EmbeddingDatabase
from chat_solution.llm import LargeLanguageModel


class QuizRag:
    def __init__(self):
        self.embedding_db = EmbeddingDatabase()
        self.llm = LargeLanguageModel()
        self.documents_retrieved = []

    def query(self, query: str) -> str:
        self.documents_retrieved = self.embedding_db.retrieve(query)
        print(f"Found {len(self.documents_retrieved)} documents, first 500 characters: {self.documents_retrieved[:500]}")
        context = "\n".join(self.documents_retrieved)
        prompt = self._create_prompt(context, query)
        result = self.llm.call(prompt)
        print(f"Result: {result}")
        
        return result

    def _create_prompt(self, context: str, message: str) -> str:
        chat_instructions = """"
        You are a helpful assistant.
        Respond questions with the context provided.
        If you happen to  know the answer but its not in the context, respond with what you know but make it clear that it does not come from the context.
        """

        return f"""{chat_instructions}

        Answer the question only using the provided content.

        Context: {context}
        
        User Question: {message}
        """
