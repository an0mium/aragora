"""Canonical receipt verdict helpers shared across CLI and storage paths."""

from __future__ import annotations

from typing import Final

_VERDICT_ALIASES: Final[dict[str, frozenset[str]]] = {
    "PASS": frozenset({"PASS", "PASSED", "CONSENSUS", "APPROVE", "APPROVED", "YES"}),
    "CONDITIONAL": frozenset(
        {"CONDITIONAL", "WARN", "WARNING", "NEEDS_REVIEW", "APPROVED_WITH_CONDITIONS"}
    ),
    "FAIL": frozenset({"FAIL", "FAILED", "REJECT", "REJECTED", "NO"}),
}
_VERDICT_LOOKUP: Final[dict[str, str]] = {
    alias: canonical for canonical, aliases in _VERDICT_ALIASES.items() for alias in aliases
}


def normalize_receipt_verdict(verdict: object) -> str:
    """Return the canonical receipt verdict label for a stored or displayed value."""
    normalized = str(verdict or "").strip()
    if not normalized:
        return ""
    upper = normalized.upper()
    return _VERDICT_LOOKUP.get(upper, upper)


def receipt_verdict_aliases(verdict: object) -> tuple[str, ...]:
    """Return all known aliases that should match a requested verdict filter."""
    canonical = normalize_receipt_verdict(verdict)
    if not canonical:
        return ()
    aliases = _VERDICT_ALIASES.get(canonical)
    if aliases is None:
        return (canonical,)
    return tuple(sorted(aliases))
