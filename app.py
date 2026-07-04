from utils.pdf_reader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import generate_embeddings
from utils.vector_store import (
    store_embeddings,
    search_documents
)
from chatbot import ask_llm


# Step 1: Load PDF
text = load_pdf("data/machine_learning.pdf")

# Step 2: Split into chunks
chunks = split_text(text)

# Step 3: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 4: Store embeddings in ChromaDB
store_embeddings(chunks, embeddings)

# Step 5: User question
query = "What is Machine Learning?"

# Step 6: Convert question to embedding
query_embedding = generate_embeddings([query])[0]

# Step 7: Search ChromaDB
results = search_documents(query_embedding)

# Step 8: Retrieve most relevant chunk
context = results["documents"][0][0]

print("\n==============================")
print("Retrieved Context")
print("==============================")
print(context)

# Step 9: Ask Gemini
answer = ask_llm(context, query)

print("\n==============================")
print("Gemini Answer")
print("==============================")
print(answer)

print("\n==============================")
print("Project Information")
print("==============================")
print(f"Total Chunks: {len(chunks)}")
print(f"Total Embeddings: {len(embeddings)}")
print(f"Embedding Dimension: {len(embeddings[0])}")