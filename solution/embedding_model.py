from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel(object):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


if __name__ == "__main__":
    model = EmbeddingModel()
    print(len(model.create_embedding("Things to do in Berlin")))
    print(model.similarity("Cappuccino", "Coffe with milk"))
    print(model.similarity("Cappuccino", "Coffe without milk"))
    print(model.similarity("Cappuccino", "Sparkling Water"))
    print(model.similarity("Cappuccino", "Italian Hot Drink"))
    print(model.similarity("Cappuccino", "Chinese Hot Drink"))
