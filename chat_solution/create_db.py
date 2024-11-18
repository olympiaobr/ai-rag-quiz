from embedding_db import EmbeddingDatabase
from embedding_model import EmbeddingModel


def create_text_chunks(text: str, chunk_size: int, overlap_size: int) -> list:
    """Create overlapping text chunks from the extracted text."""
    chunks = []
    for i in range(0, len(text) - chunk_size + 1, chunk_size - overlap_size):
        chunks.append(text[i : i + chunk_size])
    print(f"Created {len(chunks)} chunks of size {chunk_size} with overlap {overlap_size}")
    return chunks

if __name__ == "__main__":
    # Extract text from file and create chunks
    text = open("../data/data_example.md", "r").read()
    text_chunks = create_text_chunks(text, chunk_size=700, overlap_size=200)

    # Initialize embedding model and database
    model = EmbeddingModel()
    db = EmbeddingDatabase(model)

    # Add text chunks to the database and save the state
    db.add_documents(text_chunks)
    db.save_state()
    print("Database saved successfully")

