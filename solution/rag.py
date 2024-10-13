from llm import LargeLanguageModel
from embedding_db import EmbeddingDatabase


class QuestionAnsweringRAG:

    def __init__(self, llm, embedding_db: EmbeddingDatabase):
        self.llm = llm
        self.embedding_db = embedding_db

    def _create_prompt(self, context, message):
        return f"""Answer the question only using the provided content.

        Context: {context}
        
        User Question: {message}

        Respond in a natural way. If the information cannot be found in the context, respond with "It is out of my pay grade" and be more rude.
        """

    def query(self, query):
        model = LargeLanguageModel()
        documents = self.embedding_db.retrieve(query)
        context = "\n".join(documents)
        prompt = self._create_prompt(context, query)
        return model.call(prompt)
