from utils.pdf_reader import load_pdf
from utils.text_splitter import split_text

text = load_pdf("data/machine_learning.pdf")

chunks = split_text(text)

print(f"Total Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0])