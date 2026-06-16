"""Comment verdict parsing helpers for review-queue evidence gates."""

from __future__ import annotations

import re


def has_blocking_or_negative_verdict(body: str) -> bool:
    """Return True for explicit evidence comments that report blockers."""
    negative_verdict_prefixes = (
        "fail",
        "failed",
        "failing",
        "fails",
        "failure",
        "block",
        "blocked",
        "blocking",
        "request changes",
        "request_changes",
        "changes requested",
        "reject",
        "rejected",
        "not ready",
        "needs repair",
    )
    non_blocking_prefixes = (
        "none",
        "none found",
        "no",
        "no blockers",
        "no blocking findings",
        "not found",
        "0",
        "zero",
        "false",
        "n/a",
        "not applicable",
        "[]",
    )

    def _starts_with_phrase(value: str, phrases: tuple[str, ...]) -> bool:
        return any(re.match(rf"{re.escape(phrase)}(?!\w)", value) for phrase in phrases)

    def _strip_decoration(text: str) -> str:
        return re.sub(r"^(?:[#>\-*+\s]+|\d+[.)]\s+)+", "", text.strip())

    def _normalize_value(text: str) -> str:
        text = text.replace("**", "").replace("__", "")
        text = re.sub(r"[-_]+", " ", text)
        return re.sub(r"\s+", " ", text.strip().strip("*_").strip().lower())

    lines = [raw_line.strip() for raw_line in str(body or "").splitlines()]
    for idx, stripped in enumerate(lines):
        if not stripped:
            continue
        priority_marker_line = _strip_decoration(stripped)
        if re.match(
            r"^(?:\*\*)?\[(?:p0|p1)\](?:\*\*)?(?:\s|$|[:.;—–-])", priority_marker_line, re.I
        ):
            return True
        line = _strip_decoration(stripped).replace("**", "").replace("__", "")
        match = re.match(r"^(?P<label>[^:—–-]+?)\s*(?::|—|–|-)\s*(?P<value>.*)$", line)
        if not match:
            continue
        normalized_label = re.sub(r"\s+", " ", match.group("label").strip().lower())
        normalized_label = normalized_label.strip("*_ ")
        normalized_value = _normalize_value(match.group("value"))
        if normalized_label in {"verdict", "decision", "recommendation"}:
            if _starts_with_phrase(normalized_value, negative_verdict_prefixes):
                return True
            continue
        if normalized_label not in {"blocking finding", "blocking findings", "blocker", "blockers"}:
            continue
        candidate = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", normalized_value)
        if candidate in {"-", "*", "[]", "[ ]", "—", "–"}:
            continue
        if not candidate:
            follow = next((entry for entry in lines[idx + 1 :] if entry), "")
            is_list_item = bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", follow))
            if not is_list_item and (follow.startswith("#") or re.match(r"^[^:]+?:\s+\S", follow)):
                continue
            candidate = _normalize_value(_strip_decoration(follow))
        if candidate and not _starts_with_phrase(candidate, non_blocking_prefixes):
            return True
    return False
