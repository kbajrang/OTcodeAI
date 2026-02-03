from __future__ import annotations

from typing import Any

import requests

from src.config.settings import settings
from src.embeddings.bge import BGEEmbedder
from src.graph.store import load_graph
from src.utils.logging import get_logger
from src.vector_store.faiss_store import FaissStore
from src.vector_store.metadata import load_metadata

logger = get_logger(__name__)


class GraphRAGPipeline:
    def __init__(self) -> None:
        self.embedder = BGEEmbedder()
        self.graph = None
        self.vector_store = None
        self.metadata: list[dict[str, Any]] = []
        self._load_indexes()

    def _load_indexes(self) -> None:
        try:
            self.graph = load_graph()
        except FileNotFoundError:
            logger.warning("Graph file not found. Run the indexer first.")
            self.graph = None
        try:
            self.metadata = load_metadata()
            if self.metadata:
                vector = self.embedder.encode([self.metadata[0]["code"]])[0]
                self.vector_store = FaissStore(dim=len(vector))
                self.vector_store.load()
        except FileNotFoundError:
            logger.warning("Vector index not found. Run the indexer first.")
            self.vector_store = None

    def _build_context(self, top_indices: list[int]) -> str:
        context_chunks: list[str] = []
        for idx in top_indices:
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            context_chunks.append(
                f"ID: {item.get('id')}\nType: {item.get('type')}\n"
                f"File: {item.get('file_path')}\nCode:\n{item.get('code')}\n"
            )
            if self.graph and item.get("id") in self.graph:
                neighbors = list(self.graph.successors(item["id"]))
                if neighbors:
                    context_chunks.append(
                        f"Related: {', '.join(neighbors[:5])}\n"
                    )
        return "\n---\n".join(context_chunks)

    def _query_ollama(self, question: str, context: str) -> str:
        prompt = (
            "You are a code intelligence assistant. Use the provided context to answer "
            "the user's question with concise architectural reasoning. If the context "
            "is insufficient, say what is missing.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
        }
        url = f"{settings.ollama_base_url}/api/generate"
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def answer(self, question: str) -> str:
        logger.info("Received question: %s", question)
        if not self.vector_store or not self.metadata:
            return (
                "Vector index is missing. Run the indexer to build embeddings and graph data."
            )
        query_vector = self.embedder.encode([question])[0]
        _, indices = self.vector_store.search(query_vector, k=5)
        context = self._build_context(indices)
        if not context:
            return "No relevant context found for the question."
        try:
            return self._query_ollama(question, context)
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            return f"Ollama request failed: {exc}"
