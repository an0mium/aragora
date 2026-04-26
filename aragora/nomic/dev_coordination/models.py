"""Dataclasses, enums, and error types for development coordination.

These are the pure-data layer of ``aragora.nomic.dev_coordination`` and have
no dependencies on the SQLite store, classification jungle, or CLI.  They
depend only on :mod:`aragora.nomic.dev_coordination.utils` for stateless
helpers, and use lazy imports from :mod:`aragora.nomic.dev_coordination.core`
for a handful of predicates (``_claims_overlap``, ``_globs_overlap_any``,
``_developer_task_work_status``) that will migrate out in TCP-3 PR-B.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any

from aragora.nomic.dev_coordination.utils import (
    _artifact_hash,
    _json_loads,
    _normalize_claim,
    _parse_dt,
    _utcnow,
)
from aragora.nomic.global_work_queue import WorkItem, WorkStatus, WorkType

_ACTIVE_LEASE_STATUSES = {"active"}


class LeaseConflictError(ValueError):
    """Raised when a lease overlaps another active lease."""

    def __init__(self, conflicts: list[dict[str, Any]]):
        super().__init__("Lease overlaps existing active work")
        self.conflicts = conflicts


class FileScopeViolationError(ValueError):
    """Raised when a completion touches files outside the claimed scope."""

    def __init__(self, violations: list[dict[str, Any]]):
        super().__init__("Completion violates file-scope ownership")
        self.violations = violations


class LeaseStatus(str, Enum):
    """Lifecycle states for work leases."""

    ACTIVE = "active"
    COMPLETED = "completed"
    RELEASED = "released"
    EXPIRED = "expired"


class IntegrationDecisionType(str, Enum):
    """Integrator verdict for completed work."""

    PENDING_REVIEW = "pending_review"
    MERGE = "merge"
    CHERRY_PICK = "cherry_pick"
    REQUEST_CHANGES = "request_changes"
    DISCARD = "discard"
    SALVAGE = "salvage"


class SalvageStatus(str, Enum):
    """Lifecycle states for salvage candidates."""

    DETECTED = "detected"
    CLAIMED = "claimed"
    PORTED = "ported"
    DISCARDED = "discarded"


@dataclass(slots=True)
class WorkLease:
    """A bounded claim over a task, worktree, and write scope."""

    lease_id: str
    task_id: str
    title: str
    owner_agent: str
    owner_session_id: str
    branch: str
    worktree_path: str
    allowed_globs: list[str] = field(default_factory=list)
    claimed_paths: list[str] = field(default_factory=list)
    expected_tests: list[str] = field(default_factory=list)
    status: str = LeaseStatus.ACTIVE.value
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: _utcnow().isoformat())
    expires_at: str = field(default_factory=lambda: (_utcnow() + timedelta(hours=8)).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in _ACTIVE_LEASE_STATUSES and not self.is_expired

    @property
    def is_expired(self) -> bool:
        return _parse_dt(self.expires_at) <= _utcnow()

    def overlaps(self, allowed_globs: list[str], claimed_paths: list[str]) -> bool:
        # Lazy imports break the models<->core cycle.  Scope helpers move to
        # ``scope_rules.py`` in TCP-3 PR-B; until then they live in core.
        from aragora.nomic.dev_coordination.core import (
            _claims_overlap,
            _globs_overlap_any,
        )

        other_globs = [_normalize_claim(x) for x in allowed_globs if str(x).strip()]
        other_paths = [_normalize_claim(x) for x in claimed_paths if str(x).strip()]
        if self.claimed_paths and _claims_overlap(self.claimed_paths, other_globs, other_paths):
            return True
        return _globs_overlap_any(self.allowed_globs, other_globs, self.claimed_paths, other_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "title": self.title,
            "owner_agent": self.owner_agent,
            "owner_session_id": self.owner_session_id,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "allowed_globs": list(self.allowed_globs),
            "claimed_paths": list(self.claimed_paths),
            "expected_tests": list(self.expected_tests),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> WorkLease:
        return cls(
            lease_id=row["lease_id"],
            task_id=row["task_id"],
            title=row["title"],
            owner_agent=row["owner_agent"],
            owner_session_id=row["owner_session_id"],
            branch=row["branch"],
            worktree_path=row["worktree_path"],
            allowed_globs=_json_loads(row["allowed_globs_json"], []),
            claimed_paths=_json_loads(row["claimed_paths_json"], []),
            expected_tests=_json_loads(row["expected_tests_json"], []),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            metadata=_json_loads(row["metadata_json"], {}),
        )


@dataclass(slots=True)
class CompletionReceipt:
    """A bounded worker output ready for integration review."""

    receipt_id: str
    lease_id: str
    task_id: str
    owner_agent: str
    owner_session_id: str
    branch: str
    worktree_path: str
    base_sha: str = ""
    head_sha: str = ""
    commit_shas: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    validations_run: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    outcome: str = "completed"
    risks: list[str] = field(default_factory=list)
    pr_url: str = ""
    pr_number: int | None = None
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())
    artifact_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.validations_run and self.tests_run:
            self.validations_run = list(self.tests_run)
        if not self.artifact_hash:
            self.artifact_hash = _artifact_hash(
                {
                    "lease_id": self.lease_id,
                    "task_id": self.task_id,
                    "owner_agent": self.owner_agent,
                    "owner_session_id": self.owner_session_id,
                    "branch": self.branch,
                    "worktree_path": self.worktree_path,
                    "base_sha": self.base_sha,
                    "head_sha": self.head_sha,
                    "commit_shas": self.commit_shas,
                    "changed_paths": self.changed_paths,
                    "tests_run": self.tests_run,
                    "validations_run": self.validations_run,
                    "assumptions": self.assumptions,
                    "blockers": self.blockers,
                    "outcome": self.outcome,
                    "risks": self.risks,
                    "pr_url": self.pr_url,
                    "pr_number": self.pr_number,
                    "confidence": self.confidence,
                    "metadata": self.metadata,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "owner_agent": self.owner_agent,
            "owner_session_id": self.owner_session_id,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "commit_shas": list(self.commit_shas),
            "changed_paths": list(self.changed_paths),
            "tests_run": list(self.tests_run),
            "validations_run": list(self.validations_run),
            "assumptions": list(self.assumptions),
            "blockers": list(self.blockers),
            "outcome": self.outcome,
            "risks": list(self.risks),
            "pr_url": self.pr_url or None,
            "pr_number": self.pr_number,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "artifact_hash": self.artifact_hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> CompletionReceipt:
        return cls(
            receipt_id=row["receipt_id"],
            lease_id=row["lease_id"],
            task_id=row["task_id"],
            owner_agent=row["owner_agent"],
            owner_session_id=row["owner_session_id"],
            branch=row["branch"],
            worktree_path=row["worktree_path"],
            base_sha=row["base_sha"],
            head_sha=row["head_sha"],
            commit_shas=_json_loads(row["commit_shas_json"], []),
            changed_paths=_json_loads(row["changed_paths_json"], []),
            tests_run=_json_loads(row["tests_run_json"], []),
            validations_run=_json_loads(row["validations_run_json"], []),
            assumptions=_json_loads(row["assumptions_json"], []),
            blockers=_json_loads(row["blockers_json"], []),
            outcome=row["outcome"],
            risks=_json_loads(row["risks_json"], []),
            pr_url=row["pr_url"],
            pr_number=row["pr_number"],
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            artifact_hash=row["artifact_hash"],
            metadata=_json_loads(row["metadata_json"], {}),
        )


@dataclass(slots=True)
class DeveloperTask:
    """Canonical task-queue projection for supervised swarm work."""

    task_key: str
    task_id: str
    run_id: str
    goal: str
    title: str
    status: str
    priority: int = 50
    owner_agent: str = ""
    reviewer_agent: str = ""
    blocked_by: list[str] = field(default_factory=list)
    acceptance_checks: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)
    lease_id: str | None = None
    owner_session_id: str | None = None
    branch: str | None = None
    worktree_path: str | None = None
    receipt_id: str | None = None
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: _utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "goal": self.goal,
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "owner_agent": self.owner_agent or None,
            "reviewer_agent": self.reviewer_agent or None,
            "blocked_by": list(self.blocked_by),
            "acceptance_checks": list(self.acceptance_checks),
            "allowed_paths": list(self.allowed_paths),
            "lease_id": self.lease_id,
            "owner_session_id": self.owner_session_id,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "receipt_id": self.receipt_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    def to_work_item(self) -> WorkItem:
        # ``_developer_task_work_status`` moves to ``work_order_rules.py`` in
        # TCP-3 PR-B; until then it lives in core.
        from aragora.nomic.dev_coordination.core import _developer_task_work_status

        status = _developer_task_work_status(self.status)
        blockers = list(self.blocked_by)
        if status == WorkStatus.BLOCKED and not blockers:
            blockers = [f"task_status:{self.status}"]
        return WorkItem(
            id=f"task:{self.task_key}",
            work_type=WorkType.CUSTOM,
            title=self.title,
            description=self.goal or self.title,
            status=status,
            created_at=_parse_dt(self.created_at),
            updated_at=_parse_dt(self.updated_at),
            base_priority=self.priority,
            assigned_to=self.owner_session_id or self.owner_agent or None,
            blockers=blockers,
            tags=["developer-task", "swarm", self.status],
            metadata=self.to_dict(),
        )


@dataclass(slots=True)
class IntegrationDecision:
    """Integrator verdict for a completion receipt."""

    decision_id: str
    lease_id: str
    receipt_id: str
    decision: str
    target_branch: str
    rationale: str
    chosen_commits: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    decided_by: str = ""
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "lease_id": self.lease_id,
            "receipt_id": self.receipt_id,
            "decision": self.decision,
            "target_branch": self.target_branch,
            "rationale": self.rationale,
            "chosen_commits": list(self.chosen_commits),
            "followups": list(self.followups),
            "decided_by": self.decided_by,
            "created_at": self.created_at,
        }

    def to_work_item(self) -> WorkItem:
        priority = 85 if self.decision == IntegrationDecisionType.PENDING_REVIEW.value else 50
        return WorkItem(
            id=f"integration:{self.decision_id}",
            work_type=WorkType.CUSTOM,
            title=f"Integration review for receipt {self.receipt_id[:8]}",
            description=self.rationale or f"{self.decision} for lease {self.lease_id}",
            status=WorkStatus.READY,
            created_at=_parse_dt(self.created_at),
            updated_at=_parse_dt(self.created_at),
            base_priority=priority,
            tags=["integration", self.decision, self.target_branch],
            metadata=self.to_dict(),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> IntegrationDecision:
        return cls(
            decision_id=row["decision_id"],
            lease_id=row["lease_id"],
            receipt_id=row["receipt_id"],
            decision=row["decision"],
            target_branch=row["target_branch"],
            rationale=row["rationale"],
            chosen_commits=_json_loads(row["chosen_commits_json"], []),
            followups=_json_loads(row["followups_json"], []),
            decided_by=row["decided_by"],
            created_at=row["created_at"],
        )


@dataclass(slots=True)
class SalvageCandidate:
    """Potentially useful abandoned work discovered in stashes or worktrees."""

    candidate_id: str
    source_kind: str
    source_ref: str
    branch: str = ""
    worktree_path: str = ""
    stash_ref: str = ""
    head_sha: str = ""
    changed_paths: list[str] = field(default_factory=list)
    summary: str = ""
    likely_value: float = 0.0
    status: str = SalvageStatus.DETECTED.value
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: _utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "stash_ref": self.stash_ref,
            "head_sha": self.head_sha,
            "changed_paths": list(self.changed_paths),
            "summary": self.summary,
            "likely_value": self.likely_value,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    def to_work_item(self) -> WorkItem:
        return WorkItem(
            id=f"salvage:{self.candidate_id}",
            work_type=WorkType.MAINTENANCE,
            title=f"Salvage {self.source_kind} {self.source_ref}",
            description=self.summary or f"Review salvage candidate from {self.source_kind}",
            status=WorkStatus.READY,
            created_at=_parse_dt(self.created_at),
            updated_at=_parse_dt(self.updated_at),
            base_priority=max(10, min(100, int(self.likely_value * 100))),
            tags=["salvage", self.source_kind, self.status],
            metadata=self.to_dict(),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SalvageCandidate:
        return cls(
            candidate_id=row["candidate_id"],
            source_kind=row["source_kind"],
            source_ref=row["source_ref"],
            branch=row["branch"],
            worktree_path=row["worktree_path"],
            stash_ref=row["stash_ref"],
            head_sha=row["head_sha"],
            changed_paths=_json_loads(row["changed_paths_json"], []),
            summary=row["summary"],
            likely_value=float(row["likely_value"]),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_json_loads(row["metadata_json"], {}),
        )


__all__ = [
    "LeaseConflictError",
    "FileScopeViolationError",
    "LeaseStatus",
    "IntegrationDecisionType",
    "SalvageStatus",
    "WorkLease",
    "CompletionReceipt",
    "DeveloperTask",
    "IntegrationDecision",
    "SalvageCandidate",
]
