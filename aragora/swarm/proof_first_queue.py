"""Proof-first queue governance for canonical boss-ready work."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from aragora.swarm.roadmap_priority import (
    RoadmapPriority,
    RoadmapPriorityPolicy,
    extract_roadmap_codes,
    load_roadmap_priority_policy,
)

if TYPE_CHECKING:
    from aragora.swarm.boss_feed import GitHubIssue


TW01_ALLOW_FLAG_ENV = "ARAGORA_PROOF_FIRST_ALLOW_TW01"
_TRUTHY_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
_TW01_TITLE_PREFIX = "[tw-01]"
_TW01_FILES_HEADING_RE = re.compile(r"^#+\s*files\b", re.IGNORECASE | re.MULTILINE)
_TW01_VALIDATION_HEADING_RE = re.compile(r"^#+\s*validation\b", re.IGNORECASE | re.MULTILINE)
_TW01_PYTEST_RE = re.compile(r"\bpytest\b", re.IGNORECASE)
_TW01_FILE_PATH_RE = re.compile(r"`([^`\n]+?\.[A-Za-z0-9]+)`")
# Accept a broader set of file path hints (bullet lists, bare lines).
_TW01_BARE_FILE_PATH_RE = re.compile(r"(?:^|\s)([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+)", re.MULTILINE)

_BENCHMARK_TERMS = (
    "benchmark",
    "corpus",
    "scorecard",
    "truth artifact",
    "truth surface",
    "truth metrics",
    "freshness",
    "stale",
    "linkage",
    "tw-01",
    "tw-02",
)
_RESCUE_TERMS = (
    "rescue",
    "productization",
    "repeated rescue class",
    "tw-03",
)
_DOCS_TERMS = (
    "docs",
    "status",
    "roadmap",
    "positioning",
    "commercial",
    "external claims",
)
_PROOF_TERMS = (
    "proof",
    "truth",
    "benchmark",
    "rescue",
    "measured",
    "corpus",
)
_DRIFT_TERMS = (
    "drift",
    "reconcile",
    "align",
    "narrower",
    "outrun",
    "current gate",
)


@dataclass(frozen=True)
class ProofFirstQueueDecision:
    allowed: bool
    lane: str
    reason: str
    matched_terms: tuple[str, ...] = ()
    roadmap_codes: tuple[str, ...] = ()
    blocked_codes: tuple[str, ...] = ()


def _normalize_text(*parts: str) -> str:
    return " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())


def _matched_terms(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in terms if term in text)


def _load_policy(
    *,
    repo_root: Path | str | None,
    roadmap_policy: RoadmapPriorityPolicy | None,
) -> RoadmapPriorityPolicy | None:
    if roadmap_policy is not None:
        return roadmap_policy
    if repo_root is None:
        return None
    return load_roadmap_priority_policy(Path(repo_root))


def _tw01_allow_enabled(env: dict[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    value = str(source.get(TW01_ALLOW_FLAG_ENV, "")).strip().lower()
    return value in _TRUTHY_VALUES


def _extract_section_body(body: str, heading_re: re.Pattern[str]) -> tuple[str, bool]:
    """Return ``(section_text, present)`` for the first matching markdown heading.

    ``section_text`` contains the lines following the heading up to the next
    markdown heading.  ``present`` is ``True`` iff the heading was matched.
    """
    match = heading_re.search(body)
    if match is None:
        return "", False
    remainder = body[match.end() :]
    next_heading = re.search(r"^#+\s+\S", remainder, re.MULTILINE)
    if next_heading is not None:
        section = remainder[: next_heading.start()]
    else:
        section = remainder
    return section.strip(), True


def _tw01_parse_target_files(files_section: str) -> tuple[str, ...]:
    """Extract candidate file paths from the ``## Files`` section."""
    paths: list[str] = []
    for match in _TW01_FILE_PATH_RE.findall(files_section):
        cleaned = match.strip()
        if cleaned:
            paths.append(cleaned)
    if paths:
        # Preserve declaration order but deduplicate.
        seen: set[str] = set()
        deduped: list[str] = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return tuple(deduped)

    # Fall back to bare-token path detection when no backticks were used.
    bare_matches = _TW01_BARE_FILE_PATH_RE.findall(files_section)
    seen = set()
    deduped = []
    for raw in bare_matches:
        cleaned = raw.strip(" \t-*")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return tuple(deduped)


def _tw01_any_target_exists(
    paths: tuple[str, ...], repo_root: Path | str | None
) -> tuple[bool, tuple[str, ...]]:
    """Return ``(exists, present_paths)`` for the provided target paths."""
    if not paths:
        return False, ()
    if repo_root is None:
        # Without a repo root we cannot verify the file exists.  Err on the
        # side of caution and reject.
        return False, ()
    base = Path(repo_root)
    present: list[str] = []
    for path in paths:
        resolved = base / path
        try:
            if resolved.exists():
                present.append(path)
        except OSError:
            continue
    return bool(present), tuple(present)


def _evaluate_tw01_candidate(
    title: str,
    body: str,
    *,
    repo_root: Path | str | None,
    roadmap_codes: tuple[str, ...],
) -> ProofFirstQueueDecision:
    """Apply the TW-01 strict acceptance gate."""
    files_section, files_present = _extract_section_body(body, _TW01_FILES_HEADING_RE)
    if not files_present:
        return ProofFirstQueueDecision(
            allowed=False,
            lane="tw01_rejected",
            reason="TW-01 body missing `## Files` section",
            matched_terms=("tw-01",),
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )

    target_files = _tw01_parse_target_files(files_section)
    if not target_files:
        return ProofFirstQueueDecision(
            allowed=False,
            lane="tw01_rejected",
            reason="TW-01 `## Files` section lists no target files",
            matched_terms=("tw-01",),
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )

    validation_section, validation_present = _extract_section_body(
        body, _TW01_VALIDATION_HEADING_RE
    )
    if not validation_present:
        return ProofFirstQueueDecision(
            allowed=False,
            lane="tw01_rejected",
            reason="TW-01 body missing `## Validation` section",
            matched_terms=("tw-01",),
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )
    if not _TW01_PYTEST_RE.search(validation_section):
        return ProofFirstQueueDecision(
            allowed=False,
            lane="tw01_rejected",
            reason="TW-01 `## Validation` section must include a pytest command",
            matched_terms=("tw-01",),
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )

    exists, present_paths = _tw01_any_target_exists(target_files, repo_root)
    if not exists:
        return ProofFirstQueueDecision(
            allowed=False,
            lane="tw01_rejected",
            reason=("TW-01 target files do not exist in repo: " + ", ".join(target_files[:5])),
            matched_terms=("tw-01",),
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )

    return ProofFirstQueueDecision(
        allowed=True,
        lane="test_authoring",
        reason="TW-01 test-authoring issue with valid Files and Validation sections",
        matched_terms=("tw-01",) + present_paths,
        roadmap_codes=roadmap_codes,
        blocked_codes=(),
    )


def classify_proof_first_queue_issue(
    title: str,
    body: str = "",
    *,
    labels: list[str] | tuple[str, ...] | set[str] = (),
    repo_root: Path | str | None = None,
    roadmap_policy: RoadmapPriorityPolicy | None = None,
    env: dict[str, str] | None = None,
) -> ProofFirstQueueDecision:
    """Return whether an issue belongs in the canonical proof-first queue.

    The ``env`` argument is consulted for opt-in feature flags, notably
    :data:`TW01_ALLOW_FLAG_ENV` (``ARAGORA_PROOF_FIRST_ALLOW_TW01``).  When
    the flag is truthy, titles starting with ``[TW-01]`` are routed through
    the dedicated :func:`_evaluate_tw01_candidate` acceptance gate.  With the
    flag off the classifier behaves exactly as before.
    """

    normalized_text = _normalize_text(title, body, " ".join(str(label) for label in labels))
    policy = _load_policy(repo_root=repo_root, roadmap_policy=roadmap_policy)
    roadmap_codes = extract_roadmap_codes(f"{title}\n{body}")

    if policy is not None:
        priority = policy.priority_for_text(title, body)
        if priority.priority is RoadmapPriority.DO_NOW:
            return ProofFirstQueueDecision(
                allowed=True,
                lane="roadmap_do_now",
                reason="references a current do-now roadmap lane",
                matched_terms=(),
                roadmap_codes=priority.codes,
                blocked_codes=(),
            )
        if priority.priority.blocks_boss_ready:
            return ProofFirstQueueDecision(
                allowed=False,
                lane="blocked_roadmap_lane",
                reason="references delayed or avoided roadmap work",
                matched_terms=(),
                roadmap_codes=priority.codes,
                blocked_codes=priority.blocked_codes,
            )

    normalized_title = str(title or "").strip().lower()
    if normalized_title.startswith(_TW01_TITLE_PREFIX):
        if _tw01_allow_enabled(env):
            return _evaluate_tw01_candidate(
                title,
                body,
                repo_root=repo_root,
                roadmap_codes=roadmap_codes,
            )
        # With the flag off, TW-01 items continue to fall through to the
        # pre-existing lanes (typically non_canonical).

    benchmark_matches = _matched_terms(normalized_text, _BENCHMARK_TERMS)
    if "[tw-02]" in normalized_text or (
        "benchmark" in benchmark_matches
        and any(
            term in benchmark_matches
            for term in ("corpus", "scorecard", "truth artifact", "freshness", "stale")
        )
    ):
        return ProofFirstQueueDecision(
            allowed=True,
            lane="benchmark_regression",
            reason="matches benchmark/truth publication follow-up signals",
            matched_terms=benchmark_matches,
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )

    rescue_matches = _matched_terms(normalized_text, _RESCUE_TERMS)
    if "[tw-03]" in normalized_text or (
        "rescue" in rescue_matches
        and any(
            term in rescue_matches for term in ("productization", "repeated rescue class", "tw-03")
        )
    ):
        return ProofFirstQueueDecision(
            allowed=True,
            lane="rescue_productization",
            reason="matches repeated rescue-class productization signals",
            matched_terms=rescue_matches,
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )

    docs_matches = _matched_terms(normalized_text, _DOCS_TERMS)
    proof_matches = _matched_terms(normalized_text, _PROOF_TERMS)
    drift_matches = _matched_terms(normalized_text, _DRIFT_TERMS)
    if docs_matches and proof_matches and drift_matches:
        return ProofFirstQueueDecision(
            allowed=True,
            lane="docs_proof_drift",
            reason="matches docs/proof drift reconciliation signals",
            matched_terms=docs_matches + proof_matches + drift_matches,
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )

    return ProofFirstQueueDecision(
        allowed=False,
        lane="non_canonical",
        reason="does not match proof-first benchmark, rescue, or docs-proof lanes",
        matched_terms=benchmark_matches
        + rescue_matches
        + docs_matches
        + proof_matches
        + drift_matches,
        roadmap_codes=roadmap_codes,
        blocked_codes=(),
    )


def filter_noncanonical_boss_ready_issues(
    issues: list["GitHubIssue"],
    *,
    repo_root: Path | str,
    repo_slug_for_issue: Callable[["GitHubIssue"], str | None],
    comment_and_update_issue: Callable[..., Any],
) -> list["GitHubIssue"]:
    """Remove boss-ready from non-canonical issues and exclude them from dispatch."""

    kept: list[GitHubIssue] = []
    for issue in issues:
        if "boss-ready" not in issue.labels:
            kept.append(issue)
            continue

        decision = classify_proof_first_queue_issue(
            issue.title,
            issue.body or "",
            labels=issue.labels,
            repo_root=repo_root,
        )
        if decision.allowed:
            kept.append(issue)
            continue

        detail = ", ".join(
            decision.blocked_codes or decision.roadmap_codes or decision.matched_terms
        )
        if not detail:
            detail = decision.lane
        comment = (
            "Boss removed `boss-ready` because this issue is outside the canonical "
            f"proof-first queue for the current tranche ({detail})."
        )
        issue.labels = [value for value in issue.labels if value != "boss-ready"]
        if repo := repo_slug_for_issue(issue):
            comment_and_update_issue(
                issue.number,
                repo,
                comment,
                remove_labels=("boss-ready",),
            )
    return kept


__all__ = [
    "ProofFirstQueueDecision",
    "TW01_ALLOW_FLAG_ENV",
    "classify_proof_first_queue_issue",
    "filter_noncanonical_boss_ready_issues",
]
