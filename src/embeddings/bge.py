from __future__ import annotations

from typing import Any

import numpy as np

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BGEEmbedder:
    """Embedding adapter.

    - Primary: `sentence-transformers` (BGE).
    - Fallback (Windows-friendly): scikit-learn `HashingVectorizer` (no torch).
    """

    def __init__(self) -> None:
        self._mode: str = "unknown"
        self._model: Any | None = None
        self._vectorizer: Any | None = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            # Lazy import: avoids hard-failing when torch is missing/broken on Windows.
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(settings.embedding_model)
            self._mode = "sentence_transformers"
            logger.info("Embedder backend: sentence-transformers (%s)", settings.embedding_model)
            return
        except Exception as exc:
            logger.warning(
                "sentence-transformers unavailable (%s). Falling back to HashingVectorizer embeddings.",
                exc,
            )

        from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

        self._vectorizer = HashingVectorizer(
            n_features=int(getattr(settings, "fallback_embedding_dim", 2048)),
            alternate_sign=False,
            norm=None,
            lowercase=True,
            ngram_range=(3, 5),
            analyzer="char_wb",
        )
        self._mode = "hashing_vectorizer"
        logger.info("Embedder backend: hashing-vectorizer (dim=%s)", self._vectorizer.n_features)

    def encode(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
        if not texts:
            return []

        if self._mode == "sentence_transformers":
            bs = int(batch_size or settings.embedding_batch_size or 32)
            bs = max(1, bs)
            vecs = self._model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=bs,
                normalize_embeddings=True,
                show_progress_bar=len(texts) >= 256,
            )
            return vecs.tolist()

        # HashingVectorizer path (no batching needed; keep signature consistent).
        mat = self._vectorizer.transform(texts)  # type: ignore[union-attr]
        arr = mat.astype(np.float32).toarray()
        # L2 normalize for inner-product FAISS (IndexFlatIP).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        arr = arr / norms
        return arr.tolist()
