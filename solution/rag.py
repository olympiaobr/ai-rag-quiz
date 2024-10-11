from llm import MistralModel


class QuestionAnsweringRAG:

    def __init__(self):
        pass

    def _create_prompt(self, context, message):
        return f"""Answer the question using the provided content. Respond with 'It is out of my pay grade' if you don't know and the information cannot be found in the context.

        Context: {context}
        
        User Question: {message}"""

    def call(self, query):
        model = MistralModel()
        return model.call(query)
