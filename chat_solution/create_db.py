from chat_solution.embedding_db import EmbeddingDatabase
from chat_solution.embedding_model import EmbeddingModel


def create_text_chunks(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    """Create overlapping text chunks from the extracted text."""
    chunks = []
    for i in range(0, len(text) - chunk_size + 1, chunk_size - overlap_size):
        chunks.append(text[i : i + chunk_size])
    print(f"Created {len(chunks)} chunks of size {chunk_size} with overlap {overlap_size}")
    return chunks

def create_text_chunks_from_workshop_data() -> list[str]:
    text = open("../data/data_example.md", "r").read()
    return create_text_chunks(text, chunk_size=700, overlap_size=200)


def create_db() -> EmbeddingDatabase:
    db = EmbeddingDatabase()

    text_chunks = create_text_chunks_from_workshop_data()

    # Add text chunks to the database and save the state
    db.add_documents(text_chunks)
    db.save_state()
    print("Database saved successfully")
    return db


if __name__ == "__main__":
    create_db()
