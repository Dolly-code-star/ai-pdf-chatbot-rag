from utils.ingest import ingest_pdfs
from utils.embeddings import generate_embeddings
from utils.vector_store import search_documents
from chatbot import ask_llm

print("=" * 50)
print("🤖 AI PDF Chatbot Started")
print("=" * 50)

# Process the PDF
ingest_pdfs("data")

print("You can now ask questions.")
print("Type 'exit' to quit.\n")

chat_history = []

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
    source = results["metadatas"][0][0]["source"]

    # Ask Gemini
    answer = ask_llm(context, query)
    chat_history.append(
        {
            "question": query,
            "answer": answer
       }
    )

    print("\n🤖 Bot:")
    print(answer)
    print(f"\n📄 Source: {source}")
    print("\n📝 Conversation History:")

for chat in chat_history:
    print(f"Q: {chat['question']}")
    print(f"A: {chat['answer']}")
    print("-" * 30)
    print("\n" + "-" * 50)