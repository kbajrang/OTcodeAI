from sentence_transformers import SentenceTransformer
from src.config.settings import settings


class BGEEmbedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embedding_model)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
