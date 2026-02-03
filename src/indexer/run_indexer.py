import os

from src.parsers.treesitter_parser import parse_repository
from src.graph.build_graph import build_code_graph
from src.graph.store import save_graph
from src.embeddings.bge import BGEEmbedder
from src.vector_store.faiss_store import FaissStore
from src.vector_store.metadata import save_metadata
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _attach_module(
    parsed_units: list[dict], module_name: str
) -> list[dict]:
    id_map: dict[str, str] = {}
    for unit in parsed_units:
        unit_id = unit.get("id")
        if unit_id:
            id_map[unit_id] = f"module::{module_name}::{unit_id}"
    updated_units: list[dict] = []
    for unit in parsed_units:
        unit_id = unit.get("id")
        if unit_id and unit_id in id_map:
            unit["id"] = id_map[unit_id]
        if unit.get("file_path"):
            unit["file_path"] = f"{module_name}/{unit['file_path']}"
        unit["module"] = module_name
        for edge in unit.get("edges", []):
            if edge.get("source") in id_map:
                edge["source"] = id_map[edge["source"]]
            if edge.get("target") in id_map:
                edge["target"] = id_map[edge["target"]]
        updated_units.append(unit)
    return updated_units


def _discover_modules(repo_root: str) -> list[tuple[str, str]]:
    modules: list[tuple[str, str]] = []
    for entry in os.scandir(repo_root):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        modules.append((entry.name, entry.path))
    return modules


def _has_code_files(repo_root: str) -> bool:
    for root, _, files in os.walk(repo_root):
        for name in files:
            if name.endswith((".py", ".js", ".ts", ".java")):
                return True
    return False


def run(repo_path: str) -> None:
    parsed_units: list[dict] = []
    modules = _discover_modules(repo_path)
    if modules and not _has_code_files(repo_path):
        for module_name, module_path in modules:
            module_units = parse_repository(module_path)
            parsed_units.extend(_attach_module(module_units, module_name))
    else:
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
            "module": unit.get("module"),
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
