from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from src.embeddings.bge import BGEEmbedder
from src.graph.build_graph import build_code_graph
from src.graph.store import save_graph
from src.parsers.treesitter_parser import parse_repository
from src.utils.logging import get_logger
from src.vector_store.faiss_store import FaissStore
from src.vector_store.metadata import save_metadata

logger = get_logger(__name__)

StageCallback = Callable[[str, str | None], None]


class IndexStats(BaseModel):
    repo_path: str
    workspace: bool = False
    modules_indexed: int = 0
    parsed_units: int = 0
    vector_entries: int = 0
    message: str | None = None


JAVA_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+(?P<package>[A-Za-z0-9_\.]+)\s*;")


def _extract_java_package(code: str) -> str | None:
    match = JAVA_PACKAGE_RE.search(code or "")
    if not match:
        return None
    return match.group("package").strip() or None


def _guess_java_package_from_path(file_path: str) -> str | None:
    path = (file_path or "").replace("\\", "/")
    for marker in ("/src/main/java/", "/src/test/java/"):
        if marker in path:
            pkg = path.split(marker, 1)[1]
            pkg = pkg.rsplit("/", 1)[0]  # drop filename
            pkg = pkg.replace("/", ".").strip(".")
            return pkg or None
    return None


def _resolve_java_import_edges(parsed_units: list[dict]) -> None:
    """Resolve Java import edges to file nodes when possible.

    The parser emits edges like target="module::com.example.Foo". If a corresponding
    Foo.java file exists in the indexed repo, rewrite the edge target to that file
    node ID so the graph can traverse across files.
    """

    fqn_to_file_id: dict[str, str] = {}
    for unit in parsed_units:
        if unit.get("type") != "file":
            continue
        file_path = unit.get("file_path") or ""
        if not file_path.lower().endswith(".java"):
            continue
        code = unit.get("code") or ""
        package = _extract_java_package(code) or _guess_java_package_from_path(file_path)
        if not package:
            continue
        class_name = Path(file_path.replace("\\", "/")).name.rsplit(".", 1)[0]
        if not class_name:
            continue
        fqn = f"module::{package}.{class_name}"
        unit_id = unit.get("id")
        if unit_id:
            fqn_to_file_id[fqn] = unit_id

    if not fqn_to_file_id:
        return

    for unit in parsed_units:
        for edge in unit.get("edges", []) or []:
            if edge.get("type") != "imports":
                continue
            target = edge.get("target")
            if target in fqn_to_file_id:
                edge["target"] = fqn_to_file_id[target]
                edge["resolved"] = True


def _attach_module(parsed_units: list[dict], module_name: str) -> list[dict]:
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


def _discover_workspace_modules(workspace_root: str) -> list[tuple[str, str]]:
    ignore_dirs = {
        ".git",
        ".idea",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "target",
        "data",
    }
    modules: list[tuple[str, str]] = []
    for entry in os.scandir(workspace_root):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name in ignore_dirs:
            continue
        modules.append((entry.name, entry.path))
    return modules


def run(
    repo_path: str,
    *,
    workspace: bool = False,
    module_name: str | None = None,
    on_stage: StageCallback | None = None,
) -> IndexStats:
    stats = IndexStats(repo_path=repo_path, workspace=workspace)
    if on_stage:
        on_stage("start", f"Indexing: {repo_path}")

    parsed_units: list[dict] = []
    modules_indexed = 0

    if workspace:
        workspace_modules = _discover_workspace_modules(repo_path)
        if on_stage:
            on_stage("discover", f"Discovered {len(workspace_modules)} modules")
        for mod_name, mod_path in workspace_modules:
            if on_stage:
                on_stage("parsing", f"Parsing module: {mod_name}")
            module_units = parse_repository(mod_path)
            parsed_units.extend(_attach_module(module_units, mod_name))
            modules_indexed += 1
    else:
        if on_stage:
            on_stage("parsing", "Parsing repository")
        parsed_units = parse_repository(repo_path)
        if module_name:
            parsed_units = _attach_module(parsed_units, module_name)
            modules_indexed = 1

    stats.modules_indexed = modules_indexed
    stats.parsed_units = len(parsed_units)

    if on_stage:
        on_stage("graph", "Building dependency graph")
    _resolve_java_import_edges(parsed_units)
    graph = build_code_graph(parsed_units)
    save_graph(graph)

    vector_units = [unit for unit in parsed_units if unit.get("code") and unit.get("embed", True)]
    texts = [unit["code"] for unit in vector_units]
    metadata = [
        {
            "id": unit.get("id"),
            "type": unit.get("type"),
            "name": unit.get("name"),
            "file_path": unit.get("file_path"),
            "module": unit.get("module"),
            "start_line": unit.get("start_line"),
            "end_line": unit.get("end_line"),
            "code": unit.get("code"),
        }
        for unit in vector_units
    ]
    if texts:
        if on_stage:
            on_stage("embeddings", f"Embedding {len(texts)} chunks")
        embedder = BGEEmbedder()
        vectors = embedder.encode(texts)
        store = FaissStore(dim=len(vectors[0]))
        store.add(vectors)
        store.save()
        save_metadata(metadata)
        stats.vector_entries = len(metadata)
        stats.message = f"Vector index built with {len(metadata)} entries"
        logger.info("Vector index built with %s entries", len(metadata))
    else:
        stats.message = "No code units available for vector indexing"
        logger.warning("No code units available for vector indexing")

    if on_stage:
        on_stage("done", "Indexing complete")
    logger.info("Indexing complete")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GraphRAG index for a codebase.")
    parser.add_argument("repo_path", help="Path to repository or workspace directory")
    parser.add_argument(
        "--workspace",
        action="store_true",
        help="Treat each top-level subdirectory as a module and index them together",
    )
    parser.add_argument(
        "--module",
        dest="module_name",
        default=None,
        help="Attach a module name when indexing a single repo (useful for external codebases)",
    )
    args = parser.parse_args()

    run(args.repo_path, workspace=args.workspace, module_name=args.module_name)
