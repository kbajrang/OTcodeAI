"""
Lightweight regex-based repository parser.

Despite the historical name `treesitter_parser`, this module does not perform Tree-sitter
AST parsing. It uses regex and line-based heuristics to extract basic units (files,
classes, functions, imports) and to chunk code for embedding.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

PY_FUNCTION_RE = re.compile(r"^\s*def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
PY_CLASS_RE = re.compile(r"^\s*class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[(:]")
PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?P<from>[A-Za-z0-9_\.]+)\s+import|import\s+(?P<import>[A-Za-z0-9_\.]+))"
)
JS_FUNCTION_RE = re.compile(r"\bfunction\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
JS_CLASS_RE = re.compile(r"\bclass\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
JS_IMPORT_RE = re.compile(r"\bimport\s+.*?from\s+[\"'](?P<module>[^\"']+)[\"']")
JAVA_CLASS_RE = re.compile(r"\bclass\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?P<module>[A-Za-z0-9_\.\*]+)\s*;")
JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private|static|final|synchronized|abstract|native|strictfp|default|\s)+"
    r"[A-Za-z0-9_<>\[\],\s]+\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)


def _iter_line_chunks(lines: list[str], chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        return [(0, len(lines))]
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < len(lines):
        end = min(len(lines), start + chunk_size)
        chunks.append((start, end))
        if end >= len(lines):
            break
        start = end - overlap
    return chunks


def _iter_code_files(repo_path: str, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    ignore_dirs = {
        ".git",
        ".idea",
        ".openapi-generator",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "target",
        "__pycache__",
    }
    for root, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for name in filenames:
            if any(name.endswith(ext) for ext in extensions):
                files.append(Path(root) / name)
    return files


def _extract_units_from_file(file_path: Path, repo_root: Path) -> list[dict]:
    units: list[dict] = []
    rel_path = str(file_path.relative_to(repo_root))
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1")

    file_node_id = f"file::{rel_path}"
    file_unit = {
        "id": file_node_id,
        "type": "file",
        "name": file_path.name,
        "file_path": rel_path,
        "embed": False,
        "code": content,
        "edges": [],
    }

    lines = content.splitlines()
    for line in lines:
        if file_path.suffix == ".py":
            match_func = PY_FUNCTION_RE.match(line)
            match_class = PY_CLASS_RE.match(line)
            match_import = PY_IMPORT_RE.match(line)
            if match_func:
                name = match_func.group("name")
                unit_id = f"{file_node_id}::function::{name}"
                units.append(
                    {
                        "id": unit_id,
                        "type": "function",
                        "name": name,
                        "file_path": rel_path,
                        "embed": False,
                        "code": line.strip(),
                        "edges": [],
                    }
                )
                file_unit["edges"].append(
                    {"source": file_node_id, "target": unit_id, "type": "contains"}
                )
            if match_class:
                name = match_class.group("name")
                unit_id = f"{file_node_id}::class::{name}"
                units.append(
                    {
                        "id": unit_id,
                        "type": "class",
                        "name": name,
                        "file_path": rel_path,
                        "embed": False,
                        "code": line.strip(),
                        "edges": [],
                    }
                )
                file_unit["edges"].append(
                    {"source": file_node_id, "target": unit_id, "type": "contains"}
                )
            if match_import:
                module_name = match_import.group("from") or match_import.group("import")
                if module_name:
                    file_unit["edges"].append(
                        {
                            "source": file_node_id,
                            "target": f"module::{module_name}",
                            "type": "imports",
                        }
                    )
        elif file_path.suffix == ".java":
            match_class = JAVA_CLASS_RE.search(line)
            match_import = JAVA_IMPORT_RE.match(line)
            match_method = JAVA_METHOD_RE.match(line)
            if match_class:
                name = match_class.group("name")
                unit_id = f"{file_node_id}::class::{name}"
                units.append(
                    {
                        "id": unit_id,
                        "type": "class",
                        "name": name,
                        "file_path": rel_path,
                        "embed": False,
                        "code": line.strip(),
                        "edges": [],
                    }
                )
                file_unit["edges"].append(
                    {"source": file_node_id, "target": unit_id, "type": "contains"}
                )
            if match_method:
                name = match_method.group("name")
                unit_id = f"{file_node_id}::function::{name}"
                units.append(
                    {
                        "id": unit_id,
                        "type": "function",
                        "name": name,
                        "file_path": rel_path,
                        "embed": False,
                        "code": line.strip(),
                        "edges": [],
                    }
                )
                file_unit["edges"].append(
                    {"source": file_node_id, "target": unit_id, "type": "contains"}
                )
            if match_import:
                module_name = match_import.group("module")
                if module_name:
                    file_unit["edges"].append(
                        {
                            "source": file_node_id,
                            "target": f"module::{module_name}",
                            "type": "imports",
                        }
                    )
        else:
            match_func = JS_FUNCTION_RE.search(line)
            match_class = JS_CLASS_RE.search(line)
            match_import = JS_IMPORT_RE.search(line)
            if match_func:
                name = match_func.group("name")
                unit_id = f"{file_node_id}::function::{name}"
                units.append(
                    {
                        "id": unit_id,
                        "type": "function",
                        "name": name,
                        "file_path": rel_path,
                        "embed": False,
                        "code": line.strip(),
                        "edges": [],
                    }
                )
                file_unit["edges"].append(
                    {"source": file_node_id, "target": unit_id, "type": "contains"}
                )
            if match_class:
                name = match_class.group("name")
                unit_id = f"{file_node_id}::class::{name}"
                units.append(
                    {
                        "id": unit_id,
                        "type": "class",
                        "name": name,
                        "file_path": rel_path,
                        "embed": False,
                        "code": line.strip(),
                        "edges": [],
                    }
                )
                file_unit["edges"].append(
                    {"source": file_node_id, "target": unit_id, "type": "contains"}
                )
            if match_import:
                module_name = match_import.group("module")
                if module_name:
                    file_unit["edges"].append(
                        {
                            "source": file_node_id,
                            "target": f"module::{module_name}",
                            "type": "imports",
                        }
                    )

    chunk_spans = _iter_line_chunks(
        lines,
        chunk_size=settings.chunk_size_lines,
        overlap=settings.chunk_overlap_lines,
    )
    for chunk_index, (start, end) in enumerate(chunk_spans):
        chunk_id = f"{file_node_id}::chunk::{chunk_index}"
        chunk_code = "\n".join(lines[start:end]).strip()
        if not chunk_code:
            continue
        units.append(
            {
                "id": chunk_id,
                "type": "chunk",
                "name": f"chunk_{chunk_index}",
                "file_path": rel_path,
                "start_line": start + 1,
                "end_line": end,
                "embed": True,
                "code": chunk_code,
                "edges": [],
            }
        )
        file_unit["edges"].append(
            {"source": file_node_id, "target": chunk_id, "type": "contains"}
        )

    units.append(file_unit)
    return units


def parse_repository(repo_path: str) -> list[dict]:
    """Parse code and return a list of structured units."""
    logger.info("Parsing repository at %s", repo_path)
    repo_root = Path(repo_path)
    extensions = {".py", ".js", ".ts", ".java"}
    units: list[dict] = []
    for file_path in _iter_code_files(repo_path, extensions):
        units.extend(_extract_units_from_file(file_path, repo_root))
    logger.info("Parsed %s units", len(units))
    return units
