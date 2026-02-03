from src.parsers.treesitter_parser import parse_repository
from src.graph.build_graph import build_code_graph
from src.graph.store import save_graph
from src.embeddings.bge import BGEEmbedder
from src.vector_store.faiss_store import FaissStore
from src.vector_store.metadata import save_metadata
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run(repo_path: str) -> None:
    parsed_units = parse_repository(repo_path)
    graph = build_code_graph(parsed_units)
    save_graph(graph)
    texts = [unit["code"] for unit in parsed_units if unit.get("code")]
    metadata = [
        {
            "id": unit.get("id"),
            "type": unit.get("type"),
            "name": unit.get("name"),
            "file_path": unit.get("file_path"),
            "code": unit.get("code"),
        }
        for unit in parsed_units
        if unit.get("code")
    ]
    if texts:
        embedder = BGEEmbedder()
        vectors = embedder.encode(texts)
        store = FaissStore(dim=len(vectors[0]))
        store.add(vectors)
        store.save()
        save_metadata(metadata)
        logger.info("Vector index built with %s entries", len(metadata))
    else:
        logger.warning("No code units available for vector indexing")
    logger.info("Indexing complete")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.indexer.run_indexer <repo_path>")

    run(sys.argv[1])
