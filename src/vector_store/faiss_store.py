from __future__ import annotations

import os
import numpy as np
import faiss
from src.config.settings import settings


class FaissStore:
    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or 0
        self.index: faiss.Index | None = (
            faiss.IndexFlatIP(dim) if dim is not None else None
        )

    def add(self, vectors: list[list[float]]) -> None:
        if not vectors:
            return
        arr = np.array(vectors, dtype="float32")
        if self.index is None:
            self.dim = int(arr.shape[1])
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(arr)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Cannot save FAISS index: no vectors have been added")
        os.makedirs(os.path.dirname(settings.vector_db_path), exist_ok=True)
        faiss.write_index(self.index, settings.vector_db_path)

    def load(self) -> None:
        self.index = faiss.read_index(settings.vector_db_path)
        self.dim = int(self.index.d)

    def search(self, vector: list[float], k: int = 5) -> tuple[list[float], list[int]]:
        if self.index is None:
            raise ValueError("FAISS index is not loaded")
        arr = np.array([vector], dtype="float32")
        scores, indices = self.index.search(arr, k)
        return scores[0].tolist(), indices[0].tolist()
