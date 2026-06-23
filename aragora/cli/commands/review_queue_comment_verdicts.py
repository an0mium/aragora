"""Comment verdict parsing helpers for review-queue evidence gates."""

from __future__ import annotations

import os
import re

_SEVERITY_GATED_DISSENT_ENV = "ARAGORA_ENABLE_SEVERITY_GATED_MODEL_DISSENT"
_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))

# Explicit "no Pn finding" heads. A `[P0]`/`[P1]` line is blocking UNLESS the text
# before its first colon is EXACTLY one of these — models emit `"[P1] None:"`,
# `"[P1] N/A"`, `"[P1] no issues: ..."` to declare the absence of a finding. Matched
# exactly (not as a prefix) so a real finding that merely starts with "none"/"no"
# (e.g. "[P1] None of the inputs are validated") still blocks.
_NO_FINDING_HEADS = frozenset(
    {
        "none",
        "none found",
        "none identified",
        "none noted",
        "none here",
        "n/a",
        "na",
        "nil",
        "0",
        "zero",
        "false",
        "[]",
        "not applicable",
        "no issues",
        "no issue",
        "no findings",
        "no finding",
        "no blockers",
        "no blocker",
        "no blocking",
        "no blocking findings",
        "no concerns",
        "no concern",
        "no critical issues",
        "no critical issue",
        "no critical findings",
    }
)


def severity_gated_model_dissent_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether model-review dissent blocks only on explicit high-severity findings."""
    source = os.environ if env is None else env
    return str(source.get(_SEVERITY_GATED_DISSENT_ENV, "")).strip().lower() in _TRUE_VALUES


def _starts_with_phrase(value: str, phrases: tuple[str, ...]) -> bool:
    return any(re.match(rf"{re.escape(phrase)}(?!\w)", value) for phrase in phrases)


def _strip_decoration(text: str) -> str:
    return re.sub(r"^(?:[#>\-*+\s]+|\d+[.)]\s+)+", "", text.strip())


def _normalize_value(text: str) -> str:
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"[-_]+", " ", text)
    return re.sub(r"\s+", " ", text.strip().strip("*_").strip().lower())


def _priority_marker(line: str) -> tuple[str, str] | None:
    priority_marker_line = _strip_decoration(line)
    match = re.match(
        r"^(?:\*\*)?\[(?P<priority>p[0-3])\](?:\*\*)?(?:\s|$|[:.;—–-])",
        priority_marker_line,
        re.I,
    )
    if not match:
        return None
    rest = re.sub(
        r"^(?:\*\*)?\[(?:p[0-3])\](?:\*\*)?\s*",
        "",
        priority_marker_line,
        flags=re.I,
    )
    head = _normalize_value(rest).split(":", 1)[0].strip(" .;—–-")
    if head in _NO_FINDING_HEADS:
        return None
    return match.group("priority").upper(), rest


def highest_finding_priority(body: str) -> str | None:
    """Return the most severe real ``[P0]``-style finding marker in ``body``."""
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    found: list[str] = []
    for stripped in (raw_line.strip() for raw_line in str(body or "").splitlines()):
        if not stripped:
            continue
        marker = _priority_marker(stripped)
        if marker:
            found.append(marker[0])
    return min(found, key=lambda item: order[item]) if found else None


def has_high_severity_finding(body: str) -> bool:
    """Return True when a real ``[P0]``/``[P1]`` finding is present."""
    return highest_finding_priority(body) in {"P0", "P1"}


def _has_explicit_blocker_label(body: str) -> bool:
    """Return True for explicit blocker fields independent of verdict wording."""
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

    lines = [raw_line.strip() for raw_line in str(body or "").splitlines()]
    for idx, stripped in enumerate(lines):
        if not stripped:
            continue
        line = _strip_decoration(stripped).replace("**", "").replace("__", "")
        match = re.match(r"^(?P<label>[^:—–-]+?)\s*(?::|—|–|-)\s*(?P<value>.*)$", line)
        if not match:
            continue
        normalized_label = re.sub(r"\s+", " ", match.group("label").strip().lower())
        normalized_label = normalized_label.strip("*_ ")
        if normalized_label not in {"blocking finding", "blocking findings", "blocker", "blockers"}:
            continue
        candidate = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", _normalize_value(match.group("value")))
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

    lines = [raw_line.strip() for raw_line in str(body or "").splitlines()]
    for idx, stripped in enumerate(lines):
        if not stripped:
            continue
        marker = _priority_marker(stripped)
        if marker:
            if marker[0] in {"P0", "P1"}:
                return True
            # explicit "[Pn] None:/N/A/no issues" non-finding -> keep scanning
            continue
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
        if _has_explicit_blocker_label("\n".join(lines[idx:])):
            return True
    return False


def has_blocking_model_dissent(
    body: str,
    *,
    severity_gated: bool | None = None,
) -> bool:
    """Return True when a model-review comment should block merge quorum.

    The legacy rule treats any negative verdict as blocking. The opt-in
    severity-gated rule keeps evidence counting strict but only lets exact-head
    model dissent block the merge packet when it carries a real P0/P1 finding or
    an explicit blocker field. P2/P3-only CHANGES-REQUESTED comments remain
    advisory and can be tracked as follow-up debt without moving the PR head.
    """
    if severity_gated is None:
        severity_gated = severity_gated_model_dissent_enabled()
    if not severity_gated:
        return has_blocking_or_negative_verdict(body)
    if not has_blocking_or_negative_verdict(body):
        return False
    if has_high_severity_finding(body) or _has_explicit_blocker_label(body):
        return True
    # Fail closed when a negative verdict does not carry explicit severity metadata.
    # The opt-in gate only downgrades model dissent that states a real low-severity
    # P2/P3 finding; bare CHANGES-REQUESTED remains blocking.
    return highest_finding_priority(body) not in {"P2", "P3"}
