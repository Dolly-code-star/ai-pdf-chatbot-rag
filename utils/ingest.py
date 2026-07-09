import os

from utils.pdf_reader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import generate_embeddings
from utils.vector_store import (
    store_embeddings,
    database_exists
)


def ingest_pdfs(folder_path):
    """
    Process all PDFs in the folder and store embeddings.
    """

    if database_exists():
        print("✅ Existing vector database found. Skipping ingestion.\n")
        return

    pdf_files = [
        file for file in os.listdir(folder_path)
        if file.endswith(".pdf")
    ]

    if not pdf_files:
        print("❌ No PDF files found.")
        return

    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file}")

        pdf_path = os.path.join(folder_path, pdf_file)

        text = load_pdf(pdf_path)
        chunks = split_text(text)
        embeddings = generate_embeddings(chunks)

        store_embeddings(
            chunks,
            embeddings,
            pdf_file
     )

        print(f"✅ Finished: {pdf_file}")