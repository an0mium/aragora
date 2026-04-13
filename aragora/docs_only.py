"""Shared docs-safe path helpers for scope narrowing."""

from __future__ import annotations

import re
from typing import Any

_DOCS_SAFE_PREFIXES = ("docs/", "docs-site/")
_DOCS_SAFE_FILENAMES = frozenset(
    {
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "LICENSE.md",
    }
)


def normalize_docs_path(path: Any) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized.removeprefix("./")
    return normalized.rstrip("/")


def canonical_docs_container_scope(path: Any) -> str | None:
    normalized = normalize_docs_path(path)
    if normalized in {"docs", "docs/**"}:
        return "docs"
    if normalized in {"docs-site", "docs-site/**"}:
        return "docs-site"
    return None


def is_docs_safe_top_level_file(path: Any) -> bool:
    normalized = normalize_docs_path(path)
    if not normalized or "/" in normalized:
        return False
    return normalized in _DOCS_SAFE_FILENAMES


def is_docs_safe_path(path: Any) -> bool:
    normalized = normalize_docs_path(path)
    if not normalized:
        return False
    if canonical_docs_container_scope(normalized) is not None:
        return True
    if any(normalized.startswith(prefix) for prefix in _DOCS_SAFE_PREFIXES):
        return True
    return is_docs_safe_top_level_file(normalized)


def infer_docs_safe_hints(text: str) -> list[str]:
    hints: list[str] = []
    for raw in re.split(r"\s+", text or ""):
        token = raw.strip().strip("`'\".,;:()[]{}<>")
        normalized = normalize_docs_path(token)
        if not is_docs_safe_path(normalized):
            continue
        hints.append(normalized)
    return list(dict.fromkeys(hints))
