
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings_model():
    # all-MiniLM-L2-v2 is ~80MB vs ~400MB for L6-v2
    # Same quality for RAG, much lower memory — critical for Render free tier
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L2-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": 2,          # smaller batch = lower peak RAM during embedding
            "normalize_embeddings": True
        }
    )

    return embedding_model
