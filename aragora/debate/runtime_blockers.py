"""Runtime stderr classification for dogfood/debate quality reporting."""

from __future__ import annotations

import re
from typing import Any


_WARNING_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("resource_warning", re.compile(r"\bresourcewarning\b", re.IGNORECASE)),
    ("unclosed_connector", re.compile(r"\bunclosed connector\b", re.IGNORECASE)),
    ("unclosed_transport", re.compile(r"\bunclosed transport\b", re.IGNORECASE)),
    ("tracemalloc_hint", re.compile(r"\benable tracemalloc\b", re.IGNORECASE)),
    ("leaked_semaphore_warning", re.compile(r"\bleaked semaphore objects\b", re.IGNORECASE)),
]

_BLOCKER_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("debate_timeout", re.compile(r"\bdebate timed out after\b", re.IGNORECASE)),
    ("agent_timeout", re.compile(r"\bagent .* timed out\b", re.IGNORECASE)),
    ("generic_timeout", re.compile(r"\btimed out after\b", re.IGNORECASE)),
    ("insufficient_credits", re.compile(r"\binsufficient credits\b", re.IGNORECASE)),
    ("quota", re.compile(r"\bquota\b", re.IGNORECASE)),
    ("api_error", re.compile(r"\bapierror\b|\bapi error\b", re.IGNORECASE)),
    ("unknown_consensus_mode", re.compile(r"\bunknown consensus mode\b", re.IGNORECASE)),
    ("attribute_error", re.compile(r"\battributeerror\b", re.IGNORECASE)),
    ("runtime_error", re.compile(r"\bruntimeerror\b", re.IGNORECASE)),
    ("connection_error", re.compile(r"\bconnectionerror\b", re.IGNORECASE)),
    ("quality_gate_failure", re.compile(r"\bdebate failed quality gate\b", re.IGNORECASE)),
]

_TRACEBACK_PATTERN = re.compile(r"traceback\s+\(most recent call last\):", re.IGNORECASE)


def classify_stderr_signals(stderr: str) -> dict[str, Any]:
    """Classify stderr text into runtime blockers vs warning-only signals."""
    text = stderr or ""
    blockers: set[str] = set()
    warnings: set[str] = set()

    for label, pattern in _WARNING_RULES:
        if pattern.search(text):
            warnings.add(label)

    if _TRACEBACK_PATTERN.search(text):
        # ResourceWarning noise often contains "traceback" in informational hints.
        if "resourcewarning" in text.lower():
            warnings.add("resource_warning_traceback_hint")
        else:
            blockers.add("traceback")

    for label, pattern in _BLOCKER_RULES:
        if pattern.search(text):
            blockers.add(label)

    nonempty = bool(text.strip())
    warning_only = nonempty and not blockers and bool(warnings)

    return {
        "runtime_blockers": sorted(blockers),
        "warning_signals": sorted(warnings),
        "warning_only": warning_only,
    }
