from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import settings
from src.utils.paths import PROJECT_ROOT


_MODULE_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    path: str


def modules_dir() -> Path:
    return PROJECT_ROOT / settings.modules_dir


def list_modules() -> list[ModuleInfo]:
    root = modules_dir()
    if not root.exists():
        return []
    modules: list[ModuleInfo] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        modules.append(ModuleInfo(name=entry.name, path=str(entry)))
    return modules


def sanitize_module_name(name: str) -> str:
    cleaned = _MODULE_NAME_SAFE_RE.sub("_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Module name is empty after sanitization")
    return cleaned


def import_module(
    source_path: str,
    module_name: str,
    *,
    overwrite: bool = False,
) -> ModuleInfo:
    src = Path(source_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")
    if not src.is_dir():
        raise ValueError(f"Source path is not a directory: {src}")

    name = sanitize_module_name(module_name)
    dest_root = modules_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = (dest_root / name).resolve()

    if dest.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dest}. Set overwrite=true to replace."
            )
        shutil.rmtree(dest)

    ignore = shutil.ignore_patterns(
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "target",
        ".idea",
        ".pytest_cache",
    )
    shutil.copytree(src, dest, ignore=ignore, dirs_exist_ok=False)
    return ModuleInfo(name=name, path=str(dest))

