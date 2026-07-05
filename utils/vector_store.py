import chromadb

client = chromadb.PersistentClient(path="db")

collection = client.get_or_create_collection(
    name="pdf_documents"
)


def store_embeddings(chunks, embeddings):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding.tolist()]
        )

    print("✅ Embeddings stored successfully!")


def search_documents(query_embedding, n_results=2):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results


def database_exists():
    """
    Returns True if embeddings already exist.
    """
    return collection.count() > 0