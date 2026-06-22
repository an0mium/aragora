"""Proof-first queue governance for canonical boss-ready work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
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
_REV4_STAGING_CORPUS_PATH = Path("tests/benchmarks/corpus_rev4.json")


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


@lru_cache(maxsize=8)
def _staged_rev4_issue_numbers(repo_root: str) -> frozenset[int]:
    path = Path(repo_root) / _REV4_STAGING_CORPUS_PATH
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return frozenset()
    issues = payload.get("issues") if isinstance(payload, dict) else None
    if not isinstance(issues, list):
        return frozenset()
    numbers: set[int] = set()
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        try:
            issue_number = int(issue.get("issue_id") or 0)
        except (TypeError, ValueError):
            continue
        if issue_number > 0:
            numbers.add(issue_number)
    return frozenset(numbers)


def _is_explicit_staged_rev4_issue(
    *,
    issue_number: int | None,
    labels: list[str] | tuple[str, ...] | set[str],
    repo_root: Path | str | None,
) -> bool:
    if "boss-ready" not in {str(label).strip() for label in labels}:
        return False
    if issue_number is None or repo_root is None:
        return False
    try:
        normalized_issue_number = int(issue_number)
    except (TypeError, ValueError):
        return False
    if normalized_issue_number <= 0:
        return False
    return normalized_issue_number in _staged_rev4_issue_numbers(str(Path(repo_root)))


def classify_proof_first_queue_issue(
    title: str,
    body: str = "",
    *,
    labels: list[str] | tuple[str, ...] | set[str] = (),
    issue_number: int | None = None,
    repo_root: Path | str | None = None,
    roadmap_policy: RoadmapPriorityPolicy | None = None,
) -> ProofFirstQueueDecision:
    """Return whether an issue belongs in the canonical proof-first queue."""

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

    if _is_explicit_staged_rev4_issue(
        issue_number=issue_number,
        labels=labels,
        repo_root=repo_root,
    ):
        return ProofFirstQueueDecision(
            allowed=True,
            lane="staged_rev4_corpus",
            reason="explicit boss-ready issue is present in the staged rev-4 corpus",
            matched_terms=("boss-ready", "corpus_rev4"),
            roadmap_codes=roadmap_codes,
            blocked_codes=(),
        )

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
            issue_number=issue.number,
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
    "classify_proof_first_queue_issue",
    "filter_noncanonical_boss_ready_issues",
]
