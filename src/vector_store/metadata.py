import json
import os
from typing import Any

from src.config.settings import settings


def save_metadata(items: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(settings.vector_metadata_path), exist_ok=True)
    with open(settings.vector_metadata_path, "w", encoding="utf-8") as file:
        json.dump(items, file, ensure_ascii=False, indent=2)


def load_metadata() -> list[dict[str, Any]]:
    with open(settings.vector_metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)
