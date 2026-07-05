from utils.pdf_reader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import generate_embeddings
from utils.vector_store import (
    store_embeddings,
    search_documents
)
from chatbot import ask_llm


print("=" * 50)
print("🤖 AI PDF Chatbot Started")
print("=" * 50)

# Step 1: Load PDF
text = load_pdf("data/machine_learning.pdf")

# Step 2: Split text
chunks = split_text(text)

# Step 3: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 4: Store embeddings
store_embeddings(chunks, embeddings)

print("\n✅ PDF processed successfully!")
print("You can now ask questions.")
print("Type 'exit' to quit.\n")


while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("\n👋 Goodbye!")
        break

    # Generate embedding for the question
    query_embedding = generate_embeddings([query])[0]

    # Search database
    results = search_documents(query_embedding)

    # Retrieve relevant context
    context = results["documents"][0][0]

    # Ask Gemini
    answer = ask_llm(context, query)

    print("\n🤖 Bot:")
    print(answer)
    print("\n" + "-" * 50)