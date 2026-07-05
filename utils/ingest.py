from utils.pdf_reader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import generate_embeddings
from utils.vector_store import (
    store_embeddings,
    database_exists
)


def ingest_pdf(pdf_path):
    """
    Process the PDF only if the vector database is empty.
    """

    if database_exists():
        print("✅ Existing vector database found. Skipping ingestion.\n")
        return

    print("📄 Loading PDF...")

    text = load_pdf(pdf_path)

    print("✂️ Splitting text...")

    chunks = split_text(text)

    print("🧠 Generating embeddings...")

    embeddings = generate_embeddings(chunks)

    print("💾 Storing embeddings...")

    store_embeddings(chunks, embeddings)

    print("✅ PDF ingestion completed.\n")