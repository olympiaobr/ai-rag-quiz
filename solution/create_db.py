import PyPDF2
from embedding_db import EmbeddingDatabase
from embedding_model import EmbeddingModel

DATA_PATH = "../data/food_lab_green_chapter.pdf"


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            page_text = page_text.replace("\t", " ")
            text = text + page_text
    return text


def create_text_chunks(text, chunk_size, overlap_size):
    chunks = []

    for i in range(0, len(text) - chunk_size + 1, chunk_size - overlap_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


text = extract_text_from_pdf(DATA_PATH)
text_chunks = create_text_chunks(text, 1000, 200)

model = EmbeddingModel()
db = EmbeddingDatabase(model)
db.add_documents(text_chunks)
db.save_state()
