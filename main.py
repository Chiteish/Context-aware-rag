from src.load_documents import load_pdf
from src.chunk_text import split_documents
from src.embeddings import get_embeddings_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.rag_pipeline import build_rag_chain

file_path = "data/UNIT 2 cybersecurity.pdf"

documents = load_pdf(file_path)

chunks = split_documents(documents)

embedding_model = get_embeddings_model()

vector_store = create_vector_store(chunks, embedding_model)

# Specific questions
retriever = get_retriever(vector_store, k=3)

# Broad questions
retriever = get_retriever(vector_store, k=12)

qa_chain = build_rag_chain(retriever)

session_id = "user_session_1"

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    result = qa_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )

    print("\nAnswer:\n", result["answer"].content)