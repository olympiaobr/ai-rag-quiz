from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel(object):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def create_embedding(self, text):
        return self.model.encode(text)

    def _compute_cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def similarity(self, text1, text2):
        emb1 = self.create_embedding(text1)
        emb2 = self.create_embedding(text2)
        return self._compute_cosine_similarity(emb1, emb2)


if __name__ == "__main__":
    model = EmbeddingModel()
    print(len(model.create_embedding("Things to do in Berlin")))
    print(model.similarity("Cappuccino", "Coffe with milk"))
    print(model.similarity("Cappuccino", "Coffe without milk"))
    print(model.similarity("Cappuccino", "Sparkling Water"))
    print(model.similarity("Cappuccino", "Italian Hot Drink"))
    print(model.similarity("Cappuccino", "Chinese Hot Drink"))
