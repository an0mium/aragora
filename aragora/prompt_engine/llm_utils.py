"""Shared helpers for prompt-engine LLM input/output handling."""

from __future__ import annotations

import json
from typing import Any


def append_json_context(prompt: str, context: dict[str, Any] | None) -> str:
    """Append compact JSON context to a prompt."""
    if not context:
        return prompt
    return f"{prompt}\n\nAdditional context:\n{json.dumps(context, separators=(',', ':'))}"


def format_answered_questions(
    questions: list[Any] | None,
    *,
    header: str,
) -> str:
    """Render answered clarifying questions for prompt injection."""
    if not questions:
        return ""

    lines = [
        f"Q: {question.question}\nA: {question.answer}"
        for question in questions
        if getattr(question, "is_answered", False)
    ]
    if not lines:
        return ""
    return f"{header}\n" + "\n\n".join(lines)


def format_knowledge_items(
    items: list[dict[str, Any]] | None,
    *,
    max_items: int,
    content_limit: int,
    include_source: bool = False,
) -> str:
    """Render knowledge-mound matches in a compact, reusable text format."""
    if not items:
        return ""

    lines: list[str] = []
    for item in items[:max_items]:
        title = item.get("title", item.get("document_id", "Unknown"))
        content = str(item.get("content", ""))[:content_limit]
        if include_source:
            source = item.get("metadata", {}).get("source", "km")
            lines.append(f"- [{source}] {title}: {content}")
        else:
            lines.append(f"- {title}: {content}")
    return "\n".join(lines)


def parse_json_mapping(
    text: str,
    *,
    unwrap_single_nested: bool = False,
    repair_truncated: bool = False,
) -> dict[str, Any] | None:
    """Parse a JSON object from an LLM response with minimal overhead."""
    normalized = _strip_code_fences(text)
    parsed = _load_json_mapping(normalized, unwrap_single_nested=unwrap_single_nested)
    if parsed is not None:
        return parsed

    start = normalized.find("{")
    if start < 0:
        return None

    candidate = _extract_balanced_object(normalized, start)
    if candidate is not None:
        parsed = _load_json_mapping(candidate, unwrap_single_nested=unwrap_single_nested)
        if parsed is not None:
            return parsed

    end = normalized.rfind("}") + 1
    if end > start:
        parsed = _load_json_mapping(
            normalized[start:end],
            unwrap_single_nested=unwrap_single_nested,
        )
        if parsed is not None:
            return parsed

    if repair_truncated:
        repaired = repair_truncated_json(normalized[start:])
        if repaired is not None:
            return _unwrap_single_nested_dict(repaired) if unwrap_single_nested else repaired

    return None


def repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Attempt to repair a truncated JSON object by closing open structures."""
    if not text or not text.lstrip().startswith("{"):
        return None

    last_good = text.rfind("}")
    while last_good > 0:
        candidate = text[: last_good + 1]
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if open_braces >= 0 and open_brackets >= 0:
            repaired = candidate + "]" * open_brackets + "}" * open_braces
            parsed = _load_json_mapping(repaired, unwrap_single_nested=False)
            if parsed is not None:
                return parsed
        last_good = text.rfind("}", 0, last_good)

    return None


def _load_json_mapping(
    text: str,
    *,
    unwrap_single_nested: bool,
) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return _unwrap_single_nested_dict(parsed) if unwrap_single_nested else parsed


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    newline = stripped.find("\n")
    if newline == -1:
        body = stripped[3:]
        if body.lower().startswith("json"):
            body = body[4:]
    else:
        body = stripped[newline + 1 :]

    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def _extract_balanced_object(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape_next = False

    for index in range(start, len(text)):
        char = text[index]
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _unwrap_single_nested_dict(data: dict[str, Any]) -> dict[str, Any]:
    if len(data) != 1:
        return data
    inner = next(iter(data.values()))
    return inner if isinstance(inner, dict) else data
