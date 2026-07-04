import chromadb

client = chromadb.PersistentClient(path="db")

collection = client.get_or_create_collection(
    name="pdf_documents")

def store_embeddings(chunks, embeddings):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(i)],
            documents=[chunks[i]],
            embeddings=[embeddings[i]]
        )
        
print("Embeddings stored successfully!")