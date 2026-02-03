from src.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRAGPipeline:
    def __init__(self) -> None:
        # TODO: wire LlamaIndex, graph traversal, and vector retrieval
        pass

    def answer(self, question: str) -> str:
        logger.info("Received question: %s", question)
        return "GraphRAG pipeline stub response."
