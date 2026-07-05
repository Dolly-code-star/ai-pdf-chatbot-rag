from utils.ingest import ingest_pdf
from utils.embeddings import generate_embeddings
from utils.vector_store import search_documents
from chatbot import ask_llm

print("=" * 50)
print("🤖 AI PDF Chatbot Started")
print("=" * 50)

# Process the PDF
ingest_pdf("data/machine_learning.pdf")

print("You can now ask questions.")
print("Type 'exit' to quit.\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("\n👋 Goodbye!")
        break

    # Generate embedding for the user's question
    query_embedding = generate_embeddings([query])[0]

    # Search the vector database
    results = search_documents(query_embedding)

    # Retrieve the most relevant chunk
    context = results["documents"][0][0]

    # Ask Gemini
    answer = ask_llm(context, query)

    print("\n🤖 Bot:")
    print(answer)
    print("\n" + "-" * 50)