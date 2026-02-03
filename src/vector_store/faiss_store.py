from __future__ import annotations

import os
import numpy as np
import faiss
from src.config.settings import settings


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: list[list[float]]) -> None:
        arr = np.array(vectors, dtype="float32")
        self.index.add(arr)

    def save(self) -> None:
        os.makedirs(os.path.dirname(settings.vector_db_path), exist_ok=True)
        faiss.write_index(self.index, settings.vector_db_path)

    def load(self) -> None:
        self.index = faiss.read_index(settings.vector_db_path)

    def search(self, vector: list[float], k: int = 5) -> tuple[list[float], list[int]]:
        arr = np.array([vector], dtype="float32")
        scores, indices = self.index.search(arr, k)
        return scores[0].tolist(), indices[0].tolist()
