from src.parsers.treesitter_parser import parse_repository
from src.graph.build_graph import build_code_graph
from src.graph.store import save_graph
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run(repo_path: str) -> None:
    parsed_units = parse_repository(repo_path)
    graph = build_code_graph(parsed_units)
    save_graph(graph)
    logger.info("Indexing complete")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.indexer.run_indexer <repo_path>")

    run(sys.argv[1])
