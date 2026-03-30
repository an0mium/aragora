"""Helpers for canonical receipt verdict vocabulary."""

from __future__ import annotations

from typing import Any

_CANONICAL_VERDICT_ALIASES: dict[str, tuple[str, ...]] = {
    "PASS": (
        "PASS",
        "PASSED",
        "APPROVED",
        "APPROVE",
        "YES",
        "CONSENSUS",
    ),
    "CONDITIONAL": (
        "CONDITIONAL",
        "APPROVED_WITH_CONDITIONS",
        "NEEDS_REVIEW",
        "WARN",
        "WARNING",
    ),
    "FAIL": (
        "FAIL",
        "FAILED",
        "REJECTED",
        "REJECT",
        "NO",
    ),
    "UNKNOWN": (
        "UNKNOWN",
        "INCONCLUSIVE",
        "",
    ),
}

_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in _CANONICAL_VERDICT_ALIASES.items()
    for alias in aliases
}


def canonicalize_receipt_verdict(verdict: Any) -> str:
    """Return the canonical PASS/CONDITIONAL/FAIL/UNKNOWN receipt verdict."""
    normalized = str(verdict or "").strip().replace("-", "_").replace(" ", "_").upper()
    if not normalized:
        return "UNKNOWN"
    return _ALIAS_TO_CANONICAL.get(normalized, normalized)


def receipt_verdict_aliases(verdict: Any) -> tuple[str, ...]:
    """Return raw verdict spellings that should match the given filter."""
    canonical = canonicalize_receipt_verdict(verdict)
    return _CANONICAL_VERDICT_ALIASES.get(canonical, (canonical,))


__all__ = ["canonicalize_receipt_verdict", "receipt_verdict_aliases"]
