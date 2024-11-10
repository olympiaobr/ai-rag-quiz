from embedding_db import EmbeddingDatabase
from llm import LargeLanguageModel


class QuestionAnsweringRAG:
    def __init__(self, llm: LargeLanguageModel, embedding_db: EmbeddingDatabase):
        self.llm = llm
        self.embedding_db = embedding_db

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

    def query(self, query: str) -> str:
        documents = self.embedding_db.retrieve(query)
        print(f"Found {len(documents)} documents, first 500 characters: {documents[:500]}")
        context = "\n".join(documents)
        prompt = self._create_prompt(context, query)
        
        return self.llm.call(prompt)
