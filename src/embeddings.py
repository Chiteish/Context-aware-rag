
import os
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_community.embeddings import CohereEmbeddings


# Lightweight wrapper: calls Groq's embedding endpoint — no local PyTorch model loaded.
# This keeps RAM usage under Render's 512MB free tier limit.
class GroqEmbeddings:
    """Calls Groq's OpenAI-compatible embeddings endpoint."""

    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1.5"):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings_model():
    return GroqEmbeddings()

