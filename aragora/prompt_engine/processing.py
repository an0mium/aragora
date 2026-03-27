"""Shared prompt-engine preprocessing helpers."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from aragora.utils.json_helpers import extract_json_from_text


@lru_cache(maxsize=4096)
def prompt_hash(prompt: str) -> str:
    """Generate a stable short hash for a prompt-sized string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def append_context_block(prompt: str, context: dict[str, Any] | None) -> str:
    """Append a serialized context block when context is present."""
    if not context:
        return prompt
    return f"{prompt}\n\nAdditional context:\n{json.dumps(context, indent=2)}"


def format_answered_questions(
    answered_questions: list[Any] | None,
    *,
    header: str,
) -> str:
    """Render answered questions as a prompt block."""
    if not answered_questions:
        return ""

    lines = []
    for question in answered_questions:
        if getattr(question, "is_answered", False):
            lines.append(f"Q: {question.question}\nA: {question.answer}")

    return f"{header}\n" + "\n\n".join(lines) if lines else ""


def format_km_context(
    items: list[dict[str, Any]],
    *,
    limit: int,
    content_chars: int,
    include_source: bool = False,
) -> str:
    """Render Knowledge Mound items into a compact prompt block."""
    if not items:
        return ""

    lines: list[str] = []
    for item in items[:limit]:
        title = item.get("title", item.get("document_id", "Unknown"))
        content = str(item.get("content", ""))[:content_chars]
        if include_source:
            source = item.get("metadata", {}).get("source", "km")
            lines.append(f"- [{source}] {title}: {content}")
        else:
            lines.append(f"- {title}: {content}")
    return "\n".join(lines)


def extract_json_object_text(response: str) -> str | None:
    """Extract the first JSON object-like payload from a response."""
    text = response.strip()
    if not text:
        return None

    extracted = extract_json_from_text(text).strip()
    return extracted if extracted.startswith("{") else None


def parse_json_object(response: str) -> dict[str, Any] | None:
    """Parse the first JSON object-like payload from a response."""
    text = extract_json_object_text(response)
    if text is None:
        return None

    try:
        data = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None
