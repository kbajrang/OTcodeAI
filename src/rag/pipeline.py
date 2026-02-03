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
                f"Module: {item.get('module')}\nFile: {item.get('file_path')}\n"
                f"Code:\n{item.get('code')}\n"
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
        generate_url = f"{settings.ollama_base_url}/api/generate"
        response = requests.post(generate_url, json=payload, timeout=120)
        if response.status_code == 404:
            chat_url = f"{settings.ollama_base_url}/api/chat"
            chat_payload = {
                "model": settings.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            chat_response = requests.post(chat_url, json=chat_payload, timeout=120)
            if chat_response.status_code != 404:
                chat_response.raise_for_status()
                data = chat_response.json()
                message = data.get("message", {})
                return message.get("content", "").strip()
            return self._ollama_not_found_message()
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def _ollama_not_found_message(self) -> str:
        try:
            tags_url = f"{settings.ollama_base_url}/api/tags"
            tags_response = requests.get(tags_url, timeout=10)
            if tags_response.ok:
                models = [
                    model.get("name")
                    for model in tags_response.json().get("models", [])
                    if model.get("name")
                ]
                if models:
                    model_list = ", ".join(models[:10])
                    return (
                        "Ollama endpoint not found or model missing. "
                        f"Available models: {model_list}. "
                        "Update settings.llm_model to match one of these."
                    )
        except requests.RequestException:
            pass
        return (
            "Ollama endpoint not found or model missing. "
            "Ensure Ollama is running and settings.llm_model matches `ollama list`."
        )

    def answer(self, question: str) -> str:
        logger.info("Received question: %s", question)
        if not self.vector_store or not self.metadata:
            # Try to reload in case indexing completed after app startup.
            self._load_indexes()
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
