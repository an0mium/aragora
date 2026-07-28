"""Shared helpers for debate message consumers."""

from __future__ import annotations

from typing import Any, Iterable


def unique_debate_messages(messages: Iterable[Any]) -> list[Any]:
    """Deduplicate mirrored debate messages while preserving first-seen order."""
    unique: list[Any] = []
    seen: set[tuple[str, str, int, str]] = set()
    for message in messages:
        key = (
            str(getattr(message, "agent", "") or "").strip(),
            str(getattr(message, "role", "") or "").strip(),
            int(getattr(message, "round", 0) or 0),
            str(getattr(message, "content", "") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(message)
    return unique
