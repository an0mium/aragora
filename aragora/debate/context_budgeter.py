"""
Context budgeter for debate prompt construction.

Provides lightweight token budgeting and truncation for mixed context inputs
to prevent prompt bloat while preserving higher-priority context first.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: 4 characters per token
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""

    if not text:
        return text

    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    # Preserve leading whitespace to keep formatting stable
    suffix = "\n...[truncated]"
    leading = text[: len(text) - len(text.lstrip())]
    body = text[len(leading) :]
    max_body_chars = max(0, max_chars - len(leading) - len(suffix))
    trimmed = body[:max_body_chars].rstrip()
    return f"{leading}{trimmed}{suffix}"


def _parse_section_map(raw: str) -> dict[str, int]:
    if not raw:
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        if raw.startswith("{"):
            parsed = json.loads(raw)
            return {str(k): int(v) for k, v in parsed.items()}
    except (ValueError, TypeError) as exc:
        logger.debug("Failed to parse context section map JSON: %s", exc)

    mapping: dict[str, int] = {}
    for entry in raw.split(","):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        try:
            mapping[key] = int(value)
        except ValueError:
            continue
    return mapping


DEFAULT_TOTAL_TOKENS = int(os.environ.get("ARAGORA_CONTEXT_TOTAL_TOKENS", "4500"))

DEFAULT_SECTION_LIMITS: dict[str, int] = {
    "env_context": 1400,
    "historical": 800,
    "continuum": 600,
    "supermemory": 600,
    "belief": 300,
    "dissent": 300,
    "patterns": 300,
    "calibration": 200,
    "elo": 200,
    "evidence": 600,
    "trending": 200,
    "audience": 200,
    "codebase": 500,
    "claude_mem": 400,
    "memory_fabric": 600,
}

SECTION_LIMITS = {
    **DEFAULT_SECTION_LIMITS,
    **_parse_section_map(os.environ.get("ARAGORA_CONTEXT_SECTION_LIMITS", "")),
}


# Runtime overrides for the admin ``PUT /api/v1/context/budget`` endpoint.
# Kept as module-level Python state rather than being written back to
# ``os.environ`` — see ``aragora/server/handlers/context_budget.py`` — so
# admin updates don't corrupt the process-global environment that
# subprocess invocations and unrelated readers inherit. The import-time
# constants above remain the defaults; overrides only win when set.
_runtime_total_tokens: int | None = None
_runtime_section_limits: dict[str, int] | None = None


def get_total_tokens() -> int:
    """Return the runtime override for total tokens, or the import-time default."""
    if _runtime_total_tokens is not None:
        return _runtime_total_tokens
    return DEFAULT_TOTAL_TOKENS


def get_section_limits() -> dict[str, int]:
    """Return the runtime-overridden section limits, or the import-time defaults."""
    if _runtime_section_limits is not None:
        return dict(_runtime_section_limits)
    return dict(SECTION_LIMITS)


def set_total_tokens(value: int | None) -> None:
    """Install (or clear) the runtime override for total tokens.

    Pass ``None`` to reset to the import-time default.
    """
    global _runtime_total_tokens
    _runtime_total_tokens = value


def set_section_limits(limits: Mapping[str, int] | None) -> None:
    """Install (or clear) the runtime override for section limits.

    Pass ``None`` to reset. A non-``None`` mapping replaces the override
    wholesale — callers merge with ``SECTION_LIMITS`` themselves if they
    want partial overrides.
    """
    global _runtime_section_limits
    _runtime_section_limits = dict(limits) if limits is not None else None


@dataclass
class ContextSection:
    """Represents a single context section to be budgeted."""

    key: str
    content: str
    max_tokens: int | None = None


@dataclass
class ContextSectionResult:
    key: str
    content: str
    tokens: int
    truncated: bool


class ContextBudgeter:
    """Applies token budgets to ordered context sections."""

    def __init__(
        self,
        total_tokens: int | None = None,
        section_limits: Mapping[str, int] | None = None,
    ) -> None:
        # Read via the module-level getters so admin runtime overrides
        # take effect without callers having to thread the values
        # through every construction site. Explicit constructor args
        # still win over runtime overrides.
        self.total_tokens = total_tokens if total_tokens is not None else get_total_tokens()
        self.section_limits = (
            dict(section_limits) if section_limits is not None else get_section_limits()
        )

    def section_limit(self, key: str) -> int | None:
        return self.section_limits.get(key)

    def truncate_section(self, key: str, content: str, max_tokens: int | None = None) -> str:
        if not content:
            return content
        limit = max_tokens if max_tokens is not None else self.section_limit(key)
        if limit is None:
            return content
        if _estimate_tokens(content) <= limit:
            return content
        return _truncate_text(content, limit)

    def apply(
        self,
        sections: Iterable[ContextSection],
        total_tokens: int | None = None,
    ) -> list[ContextSectionResult]:
        budget = self.total_tokens if total_tokens is None else total_tokens
        remaining = max(budget, 0)
        results: list[ContextSectionResult] = []

        for section in sections:
            if not section.content:
                continue
            if remaining <= 0:
                break

            max_tokens = section.max_tokens
            if max_tokens is None:
                max_tokens = self.section_limits.get(section.key)

            allowed = remaining
            if max_tokens is not None:
                allowed = min(allowed, max_tokens)

            if allowed <= 0:
                continue

            original_tokens = _estimate_tokens(section.content)
            if original_tokens <= allowed:
                content = section.content
                truncated = False
                used_tokens = original_tokens
            else:
                content = _truncate_text(section.content, allowed)
                truncated = True
                used_tokens = _estimate_tokens(content)

            results.append(
                ContextSectionResult(
                    key=section.key,
                    content=content,
                    tokens=used_tokens,
                    truncated=truncated,
                )
            )
            remaining -= used_tokens

        return results


__all__ = [
    "ContextBudgeter",
    "ContextSection",
    "ContextSectionResult",
]
