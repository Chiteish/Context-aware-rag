from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    if not chunks:
        raise ValueError("No text chunks found to index. The document might be empty or image-only.")

    # Embed in small batches to avoid RAM spike on Render free tier
    # Instead of embedding all chunks at once, we build FAISS incrementally
    BATCH_SIZE = 32

    first_batch = chunks[:BATCH_SIZE]
    vector_store = FAISS.from_documents(first_batch, embedding_model)

    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_store = FAISS.from_documents(batch, embedding_model)
        vector_store.merge_from(batch_store)

    return vector_store