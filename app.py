from utils.pdf_reader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import generate_embeddings

text = load_pdf("data/machine_learning.pdf")

chunks = split_text(text)

embeddings = generate_embeddings(chunks)

print(f"Total Chunks: {len(chunks)}")
print(f"Total Embeddings: {len(embeddings)}")

print("\nFirst Chunk:\n")
print(chunks[0])

print("\nEmbedding Dimension:")
print(len(embeddings[0]))