"""
Backwards-compatible import path.

The original implementation lived in `src.parsers.treesitter_parser`, but the current
parser is regex/heuristic-based and does not use Tree-sitter AST parsing.

New code should import from `src.parsers.lightweight_parser` instead.
"""

from __future__ import annotations

from src.parsers.lightweight_parser import parse_repository

__all__ = ["parse_repository"]
