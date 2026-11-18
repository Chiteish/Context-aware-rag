
import os
from functools import lru_cache
from langchain_core.embeddings import Embeddings


# Uses Google's gemini-embedding-001 via API — no local PyTorch/model loaded.
# Free tier: no cost. Keeps RAM well under Render's 512MB limit.
class GoogleEmbeddings(Embeddings):
    """Calls Google Gemini embeddings endpoint using the new google-genai SDK."""

    def __init__(self, model: str = "gemini-embedding-001"):
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types
        results = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            results.append(response.embeddings[0].values)
        return results

    def embed_query(self, text: str) -> list[float]:
        from google.genai import types
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


@lru_cache(maxsize=1)
def get_embeddings_model():
    return GoogleEmbeddings()

