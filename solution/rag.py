class QuestionAnsweringRAG:

    def __init__(self, llm, embedding_model, dataset):
        pass

    def _create_prompt(self, context, message):
        return f"""Answer the question using the provided content. Respond with 'It is out of my pay grade' if you don't know and the information cannot be found in the context.

        Context: {context}
        
        User Question: {message}"""
