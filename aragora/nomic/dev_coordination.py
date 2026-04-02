"""Development coordination primitives for concurrent multi-agent work.

This module adds the missing control plane for high-churn concurrent work:
- Work leases with explicit write scopes and expected tests
- Completion receipts for bounded worker outputs
- Integration decisions for an explicit integrator lane
- Salvage candidates for dirty worktrees and stashes

The design intentionally builds on existing Aragora orchestration patterns:
- EventBus for cross-worktree signaling
- GlobalWorkQueue-compatible work item projection
- Receipt-style content hashes for auditability
- Git-common-dir local state so agents coordinate without tracked-file churn
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shlex
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from aragora.docs_only import (
    canonical_docs_container_scope,
    infer_docs_safe_hints,
    is_docs_safe_path,
    is_docs_safe_top_level_file,
)
from aragora.nomic.event_bus import EventBus
from aragora.nomic.global_work_queue import GlobalWorkQueue, WorkItem, WorkStatus, WorkType
from aragora.worktree.fleet import FleetCoordinationStore

UTC = timezone.utc
if TYPE_CHECKING:
    from aragora.swarm.lane_telemetry import LaneTelemetryCollector

_LANE_TELEMETRY: LaneTelemetryCollector | None = None
_ACTIVE_LEASE_STATUSES = {"active"}
_PENDING_INTEGRATION_DECISIONS = {"pending_review"}
_OPEN_SALVAGE_STATUSES = {"detected", "claimed"}
_OPEN_SUPERVISOR_RUN_STATUSES = {"planned", "active", "needs_human"}
_DUPLICATE_BRANCH_DELIVERABLE_ARCHIVE_GRACE_HOURS = 24.0
_SUPERSEDED_WAITING_CONFLICT_ARCHIVE_GRACE_HOURS = 24.0
_CLEAN_EXIT_NO_DELIVERABLE_ARCHIVE_GRACE_HOURS = 24.0
_FAILED_NO_DELIVERABLE_ARCHIVE_GRACE_HOURS = 24.0
_WORK_ORDER_LEASING_FAILED_ARCHIVE_GRACE_HOURS = 24.0
_WORKER_TYPE_BLOCKED_ARCHIVE_GRACE_HOURS = 24.0
_SQLITE_BUSY_TIMEOUT_MS = 60_000
_OPEN_DEVELOPER_TASK_STATUSES = {
    "queued",
    "leased",
    "dispatched",
    "active",
    "waiting_conflict",
    "dispatch_failed",
    "completed",
    "needs_human",
    "changes_requested",
    "timed_out",
    "failed",
    "integrating",
}
_QUEUEABLE_DEVELOPER_TASK_STATUSES = {
    "queued",
    "leased",
    "dispatched",
    "active",
    "waiting_conflict",
    "dispatch_failed",
    "needs_human",
    "changes_requested",
    "timed_out",
    "failed",
}
_REAPED_NO_RECEIPT_ARCHIVE_GRACE_HOURS = 6.0
_REAPED_NO_RECEIPT_BLOCKERS = {"stale_lease_reaped", "expired_lease_reaped"}
_TEST_FILE_PATTERN = re.compile(r"(tests/[\w./-]+\.py)")
_DOCS_ONLY_GENERATE_CAPABILITY_MATRIX_COMMANDS = (
    "python3 scripts/generate_capability_matrix.py",
    "python3 scripts/generate_capability_matrix.py --out docs-site/docs/contributing/capability-matrix.md",
)


def _is_docs_only_path(path: Any) -> bool:
    normalized = str(path or "").strip().removeprefix("./")
    if not normalized:
        return False
    return normalized.startswith("docs/") or normalized.endswith((".md", ".mdx", ".rst", ".txt"))


def _get_lane_telemetry() -> LaneTelemetryCollector:
    global _LANE_TELEMETRY
    if _LANE_TELEMETRY is None:
        from aragora.swarm.lane_telemetry import LaneTelemetryCollector

        _LANE_TELEMETRY = LaneTelemetryCollector()
    return _LANE_TELEMETRY


def _normalize_completion_outcome(
    *,
    outcome: str,
    commit_shas: list[str],
    changed_paths: list[str],
    pr_url: str,
    pr_number: int | None,
) -> str:
    normalized = str(outcome or "").strip()
    lowered = normalized.lower()
    if lowered and lowered != "completed":
        return normalized
    has_deliverable = bool(pr_url.strip() or pr_number is not None or commit_shas or changed_paths)
    return "deliverable_created" if has_deliverable else "clean_exit_no_deliverable"


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


class DevCoordinationStore:
    """SQLite-backed coordination state for concurrent development."""

    def __init__(
        self,
        repo_root: Path | None = None,
        db_path: Path | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.db_path = db_path or (
            self._git_common_dir(self.repo_root) / "aragora-agent-state" / "dev_coordination.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_bus = event_bus or EventBus(repo_root=self.repo_root)
        self.fleet_store = FleetCoordinationStore(self.repo_root)
        self._ensure_schema()

    @staticmethod
    def _git_common_dir(repo_root: Path) -> Path:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                proc.stderr.strip() or f"Failed to resolve git common dir for {repo_root}"
            )
        return Path(proc.stdout.strip()).resolve()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    owner_agent TEXT NOT NULL,
                    owner_session_id TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    worktree_path TEXT NOT NULL,
                    allowed_globs_json TEXT NOT NULL,
                    claimed_paths_json TEXT NOT NULL,
                    expected_tests_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_leases_status ON leases(status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_leases_worktree ON leases(worktree_path, status);

                CREATE TABLE IF NOT EXISTS completion_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    lease_id TEXT NOT NULL,
                    owner_agent TEXT NOT NULL,
                    owner_session_id TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    worktree_path TEXT NOT NULL,
                    commit_shas_json TEXT NOT NULL,
                    changed_paths_json TEXT NOT NULL,
                    tests_run_json TEXT NOT NULL,
                    assumptions_json TEXT NOT NULL,
                    blockers_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    artifact_hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_receipts_lease ON completion_receipts(lease_id, created_at);

                CREATE TABLE IF NOT EXISTS integration_decisions (
                    decision_id TEXT PRIMARY KEY,
                    lease_id TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    target_branch TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    chosen_commits_json TEXT NOT NULL,
                    followups_json TEXT NOT NULL,
                    decided_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_integration_receipt ON integration_decisions(receipt_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_integration_decision ON integration_decisions(decision, created_at);

                CREATE TABLE IF NOT EXISTS salvage_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    source_kind TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    worktree_path TEXT NOT NULL,
                    stash_ref TEXT NOT NULL,
                    head_sha TEXT NOT NULL,
                    changed_paths_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    likely_value REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_salvage_source ON salvage_candidates(source_kind, source_ref);
                CREATE INDEX IF NOT EXISTS idx_salvage_status ON salvage_candidates(status, updated_at);

                CREATE TABLE IF NOT EXISTS supervisor_runs (
                    run_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    target_branch TEXT NOT NULL,
                    status TEXT NOT NULL,
                    supervisor_agents_json TEXT NOT NULL,
                    approval_policy_json TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    work_orders_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_supervisor_runs_status ON supervisor_runs(status, updated_at);
                """
            )
            self._ensure_completion_receipt_columns(conn)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_completion_receipt_columns(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(completion_receipts)").fetchall()
        }
        required_columns = {
            "task_id": "TEXT NOT NULL DEFAULT ''",
            "base_sha": "TEXT NOT NULL DEFAULT ''",
            "head_sha": "TEXT NOT NULL DEFAULT ''",
            "validations_run_json": "TEXT NOT NULL DEFAULT '[]'",
            "outcome": "TEXT NOT NULL DEFAULT 'completed'",
            "risks_json": "TEXT NOT NULL DEFAULT '[]'",
            "pr_url": "TEXT NOT NULL DEFAULT ''",
            "pr_number": "INTEGER",
            "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for name, ddl in required_columns.items():
            if name in columns:
                continue
            conn.execute(f"ALTER TABLE completion_receipts ADD COLUMN {name} {ddl}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_receipts_task ON completion_receipts(task_id, created_at)"
        )

    def status_summary(
        self,
        *,
        include_integrator_artifacts: bool = False,
        integrator_limit: int = 200,
    ) -> dict[str, Any]:
        active_leases = self.list_active_leases()
        pending_integrations = self.list_integration_decisions(only_pending=True)
        salvage = self.list_salvage_candidates(statuses=sorted(_OPEN_SALVAGE_STATUSES))
        supervisor_runs = self.list_supervisor_runs(statuses=sorted(_OPEN_SUPERVISOR_RUN_STATUSES))
        developer_tasks = self.list_developer_tasks(open_only=True)
        scope_violations = []
        for lease in active_leases:
            violation = lease.metadata.get("last_scope_violation")
            if not isinstance(violation, dict):
                continue
            scope_violations.append(
                {
                    "lease_id": lease.lease_id,
                    "task_id": lease.task_id,
                    "title": lease.title,
                    "owner_agent": lease.owner_agent,
                    "owner_session_id": lease.owner_session_id,
                    "branch": lease.branch,
                    "worktree_path": lease.worktree_path,
                    **violation,
                }
            )
        payload = {
            "db_path": str(self.db_path),
            "fleet_path": str(self.fleet_store.path),
            "active_leases": [item.to_dict() for item in active_leases],
            "pending_integrations": [item.to_dict() for item in pending_integrations],
            "open_salvage_candidates": [item.to_dict() for item in salvage],
            "supervisor_runs": supervisor_runs,
            "developer_tasks": [item.to_dict() for item in developer_tasks],
            "scope_violations": scope_violations,
            "counts": {
                "active_leases": len(active_leases),
                "pending_integrations": len(pending_integrations),
                "open_salvage_candidates": len(salvage),
                "supervisor_runs": len(supervisor_runs),
                "open_developer_tasks": len(developer_tasks),
                "fleet_claims": len(self.fleet_store.list_claims()),
                "fleet_merge_queue": len(self.fleet_store.list_merge_queue()),
                "scope_violations": len(scope_violations),
            },
        }
        if include_integrator_artifacts:
            payload["integrator"] = self.integrator_snapshot(limit=integrator_limit)
        return payload

    def create_supervisor_run(
        self,
        *,
        goal: str,
        target_branch: str,
        supervisor_agents: dict[str, Any],
        approval_policy: dict[str, Any],
        spec: dict[str, Any],
        work_orders: list[dict[str, Any]],
        status: str = "planned",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utcnow().isoformat()
        record = {
            "run_id": str(uuid.uuid4())[:12],
            "goal": goal,
            "target_branch": target_branch,
            "status": status,
            "supervisor_agents": dict(supervisor_agents),
            "approval_policy": dict(approval_policy),
            "spec": dict(spec),
            "work_orders": [dict(item) for item in work_orders],
            "metadata": dict(metadata or {}),
            "created_at": now,
            "updated_at": now,
        }
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO supervisor_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["run_id"],
                    record["goal"],
                    record["target_branch"],
                    record["status"],
                    _json_dump(record["supervisor_agents"]),
                    _json_dump(record["approval_policy"]),
                    _json_dump(record["spec"]),
                    _json_dump(record["work_orders"]),
                    _json_dump(record["metadata"]),
                    record["created_at"],
                    record["updated_at"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return record

    def get_supervisor_run(self, run_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM supervisor_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._supervisor_run_from_row(row)

    def list_supervisor_runs(
        self,
        *,
        statuses: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM supervisor_runs ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        finally:
            conn.close()
        runs = [self._supervisor_run_from_row(row) for row in rows]
        if statuses is None:
            return runs
        allowed = set(statuses)
        return [item for item in runs if str(item.get("status", "")) in allowed]

    def list_developer_tasks(
        self,
        *,
        open_only: bool = False,
        run_id: str | None = None,
        limit: int = 200,
    ) -> list[DeveloperTask]:
        tasks: list[DeveloperTask] = []
        for run in self.list_supervisor_runs(limit=max(1, int(limit))):
            current_run_id = str(run.get("run_id", "")).strip()
            if run_id and current_run_id != str(run_id).strip():
                continue
            for raw_item in run.get("work_orders", []):
                if not isinstance(raw_item, dict):
                    continue
                task = self._developer_task_from_run(run, raw_item)
                if open_only and task.status not in _OPEN_DEVELOPER_TASK_STATUSES:
                    continue
                tasks.append(task)
        tasks.sort(key=lambda item: item.updated_at, reverse=True)
        return tasks

    def backfill_missing_blocker_metadata(self) -> int:
        """Fill structured blocker fields for historical lanes that only stored free-text errors."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            updated = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    normalized = _backfill_work_order_blocker_metadata(item)
                    if not normalized:
                        continue
                    changed = True
                    updated += 1
                if not changed:
                    continue
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return updated

    def backfill_missing_verification_plans(self) -> int:
        """Infer verification commands for historical merge-gate rows missing them."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            updated = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not _work_order_should_backfill_verification_plan(item):
                        continue
                    inferred_tests = _inferred_expected_tests_for_work_order(item)
                    if not inferred_tests:
                        continue
                    item["expected_tests"] = list(inferred_tests)
                    success_criteria = dict(item.get("success_criteria") or {})
                    if "tests" not in success_criteria:
                        success_criteria["tests"] = (
                            inferred_tests[0] if len(inferred_tests) == 1 else list(inferred_tests)
                        )
                    item["success_criteria"] = success_criteria
                    merge_gate = _merge_gate_state_for_work_order(item)
                    item["merge_gate"] = merge_gate
                    item["verification_missing_reason"] = merge_gate.get(
                        "verification_missing_reason"
                    )
                    blocked_reasons = [
                        str(reason).strip()
                        for reason in merge_gate.get("blocked_reasons", [])
                        if str(reason).strip()
                    ]
                    if blocked_reasons:
                        item["dispatch_error"] = blocked_reasons[0]
                        item["blockers"] = blocked_reasons
                    if not merge_gate.get("verification_missing_reason"):
                        item["failure_reason"] = "merge_gate_failed"
                        item["worker_outcome"] = "merge_gate_failed"
                        item["blocking_question"] = _default_blocking_question_for_reason(
                            "merge_gate_failed"
                        )
                        item["blocker"] = {
                            "reason": "merge_gate_failed",
                            "question": item["blocking_question"],
                        }
                    changed = True
                    updated += 1
                if not changed:
                    continue
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return updated

    def rehabilitate_docs_only_missing_verification_plan_work_orders(self) -> int:
        """Restore docs-only deliverables that were blocked only by absent test commands."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            updated = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not _work_order_should_rehabilitate_docs_only_missing_verification_plan(
                        item
                    ):
                        continue
                    item["merge_gate"] = _merge_gate_state_for_work_order(item)
                    item["verification_missing_reason"] = None
                    item["status"] = "completed"
                    item["review_status"] = "pending_heterogeneous_review"
                    item["worker_outcome"] = "completed"
                    for key in (
                        "failure_reason",
                        "blocking_question",
                        "blocker",
                        "dispatch_error",
                    ):
                        item.pop(key, None)
                    item["blockers"] = []
                    changed = True
                    updated += 1
                if not changed:
                    continue
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return updated

    def rehabilitate_deliverable_backed_clean_exit_no_deliverable_work_orders(self) -> int:
        """Restore contradictory clean-exit lanes that already have receipt-backed deliverables."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            updated = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not _work_order_should_rehabilitate_deliverable_backed_clean_exit_no_deliverable(
                        item
                    ):
                        continue
                    item["merge_gate"] = _merge_gate_state_for_work_order(item)
                    item["status"] = "completed"
                    item["review_status"] = "pending_heterogeneous_review"
                    item["worker_outcome"] = "completed"
                    for key in (
                        "failure_reason",
                        "blocking_question",
                        "blocker",
                        "dispatch_error",
                    ):
                        item.pop(key, None)
                    item["blockers"] = []
                    changed = True
                    updated += 1
                if not changed:
                    continue
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return updated

    def reclassify_branch_snapshot_stale_review_work_orders(self) -> int:
        """Move deliverable-backed branch-stale lanes into the review bucket."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            updated = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not _work_order_should_reclassify_branch_snapshot_stale_review(item):
                        continue
                    item["status"] = "changes_requested"
                    item["review_status"] = "changes_requested"
                    changed = True
                    updated += 1
                if not changed:
                    continue
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return updated

    def reclassify_deliverable_changes_requested_work_orders(self) -> int:
        """Align status with review bucket for deliverable-backed lanes already marked changes_requested."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            updated = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not _work_order_should_reclassify_deliverable_changes_requested(item):
                        continue
                    item["status"] = "changes_requested"
                    changed = True
                    updated += 1
                if not changed:
                    continue
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return updated

    def archive_reaped_no_receipt_work_orders(
        self,
        *,
        grace_period_hours: float = _REAPED_NO_RECEIPT_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Archive stale reaped work orders that never produced a receipt."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_should_archive_reaped_no_receipt(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status,
                    ):
                        continue
                    metadata = dict(item.get("metadata") or {})
                    archive_reason = (
                        _work_order_reap_failure_reason(item, lease_status=lease_status)
                        or "stale_lease_reaped"
                    )
                    metadata.update(
                        {
                            "archived_due_to": "reaped_no_receipt",
                            "archived_at": now.isoformat(),
                            "archive_reason": archive_reason,
                            "previous_status": _optional_text(item.get("status")) or "needs_human",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = archive_reason
                    changed = True
                    archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)

            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_scope_violation_no_deliverable_work_orders(
        self,
        *,
        grace_period_hours: float = _REAPED_NO_RECEIPT_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Archive old scope-violation work orders that never produced a deliverable."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_should_archive_scope_violation_no_deliverable(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status,
                    ):
                        continue
                    metadata = dict(item.get("metadata") or {})
                    metadata.update(
                        {
                            "archived_due_to": "scope_violation_no_deliverable",
                            "archived_at": now.isoformat(),
                            "archive_reason": "scope_violation",
                            "previous_status": _optional_text(item.get("status")) or "blocked",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "scope_violation"
                    changed = True
                    archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)

            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_failed_no_deliverable_work_orders(
        self,
        *,
        grace_period_hours: float = _FAILED_NO_DELIVERABLE_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Archive old failed lanes that never produced a receipt or deliverable."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_should_archive_failed_no_deliverable(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status,
                    ):
                        continue
                    metadata = dict(item.get("metadata") or {})
                    archive_reason = _work_order_failed_no_deliverable_reason(item)
                    metadata.update(
                        {
                            "archived_due_to": "failed_no_deliverable",
                            "archived_at": now.isoformat(),
                            "archive_reason": archive_reason,
                            "previous_status": _optional_text(item.get("status")) or "failed",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = archive_reason
                    changed = True
                    archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_clean_exit_no_deliverable_work_orders(
        self,
        *,
        grace_period_hours: float = _CLEAN_EXIT_NO_DELIVERABLE_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Archive old clean-exit lanes that never produced a receipt or deliverable."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_should_archive_clean_exit_no_deliverable(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status,
                    ):
                        continue
                    metadata = dict(item.get("metadata") or {})
                    archive_reason = _work_order_clean_exit_no_deliverable_reason(item)
                    metadata.update(
                        {
                            "archived_due_to": "clean_exit_no_deliverable",
                            "archived_at": now.isoformat(),
                            "archive_reason": archive_reason,
                            "previous_status": _optional_text(item.get("status")) or "completed",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = archive_reason
                    changed = True
                    archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_work_order_leasing_failed_work_orders(
        self,
        *,
        grace_period_hours: float = _WORK_ORDER_LEASING_FAILED_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Archive stale leasing failures that never produced a receipt or deliverable."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_should_archive_work_order_leasing_failed(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status,
                    ):
                        continue
                    metadata = dict(item.get("metadata") or {})
                    metadata.update(
                        {
                            "archived_due_to": "work_order_leasing_failed",
                            "archived_at": now.isoformat(),
                            "archive_reason": "work_order_leasing_failed",
                            "previous_status": _optional_text(item.get("status")) or "needs_human",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "work_order_leasing_failed"
                    changed = True
                    archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_worker_type_blocked_work_orders(
        self,
        *,
        grace_period_hours: float = _WORKER_TYPE_BLOCKED_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Archive stale capacity/worker-type blocked lanes with no artifact."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_should_archive_worker_type_blocked(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status,
                    ):
                        continue
                    metadata = dict(item.get("metadata") or {})
                    metadata.update(
                        {
                            "archived_due_to": "worker_type_blocked",
                            "archived_at": now.isoformat(),
                            "archive_reason": "worker_type_blocked",
                            "previous_status": _optional_text(item.get("status")) or "needs_human",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "worker_type_blocked"
                    changed = True
                    archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_superseded_clean_exit_no_deliverable_work_orders(self) -> int:
        """Archive no-op helper lanes when same-run deliverable siblings already cover the scope."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                deliverable_items = [
                    item
                    for item in record["work_orders"]
                    if isinstance(item, dict) and _work_order_has_concrete_deliverable(item)
                ]
                changed = False
                for item in record["work_orders"]:
                    if not _work_order_should_archive_superseded_clean_exit_no_deliverable(item):
                        continue
                    overlapping_deliverables = [
                        sibling
                        for sibling in deliverable_items
                        if sibling is not item and _work_orders_overlap_by_scope(item, sibling)
                    ]
                    if not overlapping_deliverables:
                        if not _looks_like_helper_clean_exit_no_deliverable(item):
                            continue
                        overlapping_siblings = [
                            sibling
                            for sibling in record["work_orders"]
                            if isinstance(sibling, dict)
                            and sibling is not item
                            and _work_orders_overlap_by_scope(item, sibling)
                            and _work_order_is_live_overlap_sibling(sibling)
                        ]
                        if not overlapping_siblings:
                            continue
                        keeper = max(
                            overlapping_siblings,
                            key=lambda sibling: _live_overlap_sibling_priority(sibling, run=record),
                        )
                        archive_reason = "helper_clean_exit_no_deliverable"
                    else:
                        keeper = max(
                            overlapping_deliverables,
                            key=lambda sibling: _duplicate_branch_deliverable_priority(
                                sibling, run=record
                            ),
                        )
                        archive_reason = "superseded_clean_exit_no_deliverable"
                    keeper_id = _optional_text(
                        keeper.get("work_order_id"),
                        keeper.get("task_id"),
                    )
                    metadata = dict(item.get("metadata") or {})
                    metadata.update(
                        {
                            "archived_due_to": "superseded_clean_exit_no_deliverable",
                            "archived_at": now,
                            "archive_reason": archive_reason,
                            "canonical_work_order_id": keeper_id or None,
                            "previous_status": _optional_text(item.get("status")) or "needs_human",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "clean_exit_no_deliverable"
                    changed = True
                    archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_superseded_stale_lease_reaped_work_orders(self) -> int:
        """Archive helper stale-lease rows when an overlapping same-run sibling still owns the work."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not _work_order_should_archive_superseded_stale_lease_reaped(item):
                        continue
                    overlapping_siblings = [
                        sibling
                        for sibling in record["work_orders"]
                        if isinstance(sibling, dict)
                        and sibling is not item
                        and _work_orders_overlap_by_scope(item, sibling)
                        and _work_order_is_live_overlap_sibling(sibling)
                    ]
                    if not overlapping_siblings:
                        continue
                    keeper = max(
                        overlapping_siblings,
                        key=lambda sibling: _live_overlap_sibling_priority(sibling, run=record),
                    )
                    keeper_id = _optional_text(
                        keeper.get("work_order_id"),
                        keeper.get("task_id"),
                    )
                    metadata = dict(item.get("metadata") or {})
                    metadata.update(
                        {
                            "archived_due_to": "superseded_stale_lease_reaped",
                            "archived_at": now,
                            "archive_reason": "helper_stale_lease_reaped",
                            "canonical_work_order_id": keeper_id or None,
                            "previous_status": _optional_text(item.get("status")) or "needs_human",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "stale_lease_reaped"
                    changed = True
                    archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_duplicate_work_order_leasing_failed_work_orders(self) -> int:
        """Collapse duplicate no-artifact leasing failures across runs."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            records = [self._supervisor_run_from_row(row) for row in rows]
            grouped: dict[
                tuple[str, tuple[str, ...]], list[tuple[dict[str, Any], dict[str, Any]]]
            ] = {}
            for record in records:
                goal_key = _canonical_goal_key(record.get("goal"))
                if not goal_key:
                    continue
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_is_duplicate_work_order_leasing_failed_candidate(
                        item,
                        run=record,
                        lease_status=lease_status,
                    ):
                        continue
                    scope_key = _canonical_work_order_scope_key(item)
                    if not scope_key:
                        continue
                    grouped.setdefault((goal_key, scope_key), []).append((record, item))

            archived = 0
            changed_run_ids: set[str] = set()
            for _, siblings in grouped.items():
                if len(siblings) < 2:
                    continue
                keeper_record, keeper_item = max(
                    siblings,
                    key=lambda pair: _duplicate_work_order_leasing_failed_priority(
                        pair[1], run=pair[0]
                    ),
                )
                keeper_run_id = _optional_text(keeper_record.get("run_id"))
                keeper_id = _optional_text(
                    keeper_item.get("work_order_id"),
                    keeper_item.get("task_id"),
                )
                for record, item in siblings:
                    if record is keeper_record and item is keeper_item:
                        continue
                    metadata = dict(item.get("metadata") or {})
                    if _optional_text(metadata.get("archived_due_to")):
                        continue
                    metadata.update(
                        {
                            "archived_due_to": "duplicate_work_order_leasing_failed",
                            "archived_at": now,
                            "archive_reason": "duplicate_work_order_leasing_failed",
                            "canonical_run_id": keeper_run_id or None,
                            "canonical_work_order_id": keeper_id or None,
                            "previous_status": _optional_text(item.get("status")) or "needs_human",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "work_order_leasing_failed"
                    archived += 1
                    changed_run_ids.add(_optional_text(record.get("run_id")))

            for record in records:
                run_id = _optional_text(record.get("run_id"))
                if run_id not in changed_run_ids:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_duplicate_branch_deliverable_work_orders(
        self,
        *,
        grace_period_hours: float = _DUPLICATE_BRANCH_DELIVERABLE_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Collapse same-run duplicate deliverable siblings that point at the same branch."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            archived = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                grouped: dict[str, list[dict[str, Any]]] = {}
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    branch = _optional_text(item.get("branch"))
                    if branch:
                        grouped.setdefault(branch, []).append(item)
                changed = False
                for branch, items in grouped.items():
                    eligible = [
                        item
                        for item in items
                        if _work_order_should_archive_duplicate_branch_deliverable(
                            item,
                            run=record,
                            cutoff=cutoff,
                            lease_status=lease_status_by_id.get(
                                _optional_text(item.get("lease_id"))
                            ),
                        )
                    ]
                    if len(eligible) < 2:
                        continue
                    keeper = max(
                        eligible,
                        key=lambda item: _duplicate_branch_deliverable_priority(item, run=record),
                    )
                    keeper_id = _optional_text(
                        keeper.get("work_order_id"),
                        keeper.get("task_id"),
                    )
                    for item in eligible:
                        if item is keeper:
                            continue
                        metadata = dict(item.get("metadata") or {})
                        metadata.update(
                            {
                                "archived_due_to": "duplicate_branch_deliverable",
                                "archived_at": now.isoformat(),
                                "archive_reason": f"duplicate_branch:{branch}",
                                "duplicate_branch": branch,
                                "canonical_work_order_id": keeper_id or None,
                                "previous_status": _optional_text(item.get("status"))
                                or "completed",
                            }
                        )
                        item["metadata"] = metadata
                        item["status"] = "discarded"
                        if not _optional_text(item.get("failure_reason")):
                            item["failure_reason"] = "duplicate_branch_deliverable"
                        changed = True
                        archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_superseded_waiting_conflict_work_orders(
        self,
        *,
        grace_period_hours: float = _SUPERSEDED_WAITING_CONFLICT_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Archive stale waiting_conflict siblings covered by deliverables or duplicate same-scope siblings."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            archived = 0
            records = [self._supervisor_run_from_row(row) for row in rows]
            for record in records:
                deliverable_items = [
                    item
                    for item in record["work_orders"]
                    if isinstance(item, dict) and _work_order_has_concrete_deliverable(item)
                ]
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_should_archive_superseded_waiting_conflict(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status,
                    ):
                        continue
                    overlapping_deliverables = [
                        sibling
                        for sibling in deliverable_items
                        if sibling is not item and _work_orders_overlap_by_scope(item, sibling)
                    ]
                    if not overlapping_deliverables:
                        continue
                    keeper = max(
                        overlapping_deliverables,
                        key=lambda sibling: _duplicate_branch_deliverable_priority(
                            sibling, run=record
                        ),
                    )
                    keeper_id = _optional_text(
                        keeper.get("work_order_id"),
                        keeper.get("task_id"),
                    )
                    metadata = dict(item.get("metadata") or {})
                    metadata.update(
                        {
                            "archived_due_to": "superseded_waiting_conflict",
                            "archived_at": now.isoformat(),
                            "archive_reason": "overlapping_deliverable_sibling",
                            "canonical_work_order_id": keeper_id or None,
                            "previous_status": "waiting_conflict",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "superseded_waiting_conflict"
                    changed = True
                    archived += 1
                if not changed:
                    eligible_waiting = [
                        item
                        for item in record["work_orders"]
                        if isinstance(item, dict)
                        and _work_order_should_archive_superseded_waiting_conflict(
                            item,
                            run=record,
                            cutoff=cutoff,
                            lease_status=lease_status_by_id.get(
                                _optional_text(item.get("lease_id"))
                            ),
                        )
                    ]
                    grouped_waiting: dict[tuple[str, ...], list[dict[str, Any]]] = {}
                    for item in eligible_waiting:
                        scope_key = _canonical_work_order_scope_key(item)
                        if not scope_key:
                            continue
                        grouped_waiting.setdefault(scope_key, []).append(item)
                    for siblings in grouped_waiting.values():
                        if len(siblings) < 2:
                            continue
                        keeper = min(
                            siblings,
                            key=lambda sibling: _duplicate_waiting_conflict_priority(
                                sibling, run=record
                            ),
                        )
                        keeper_id = _optional_text(
                            keeper.get("work_order_id"),
                            keeper.get("task_id"),
                        )
                        for item in siblings:
                            if item is keeper:
                                continue
                            metadata = dict(item.get("metadata") or {})
                            metadata.update(
                                {
                                    "archived_due_to": "superseded_waiting_conflict",
                                    "archived_at": now.isoformat(),
                                    "archive_reason": "duplicate_waiting_conflict_sibling",
                                    "canonical_work_order_id": keeper_id or None,
                                    "previous_status": "waiting_conflict",
                                }
                            )
                            item["metadata"] = metadata
                            item["status"] = "discarded"
                            if not _optional_text(item.get("failure_reason")):
                                item["failure_reason"] = "superseded_waiting_conflict"
                            changed = True
                            archived += 1
                    remaining_waiting = [
                        item
                        for item in record["work_orders"]
                        if isinstance(item, dict)
                        and _optional_text(item.get("status")).lower() == "waiting_conflict"
                        and not _optional_text(item.get("receipt_id"))
                        and not _work_order_has_concrete_deliverable(item)
                    ]
                    for item in remaining_waiting:
                        containing_siblings = [
                            sibling
                            for sibling in remaining_waiting
                            if sibling is not item
                            and _optional_text(sibling.get("status")).lower() == "waiting_conflict"
                            and _work_order_scope_contains(sibling, item)
                            and not _work_order_scope_contains(item, sibling)
                        ]
                        if not containing_siblings:
                            continue
                        keeper = max(
                            containing_siblings,
                            key=lambda sibling: _containing_waiting_conflict_priority(
                                sibling, run=record
                            ),
                        )
                        keeper_id = _optional_text(
                            keeper.get("work_order_id"),
                            keeper.get("task_id"),
                        )
                        metadata = dict(item.get("metadata") or {})
                        if _optional_text(metadata.get("archived_due_to")):
                            continue
                        metadata.update(
                            {
                                "archived_due_to": "superseded_waiting_conflict",
                                "archived_at": now.isoformat(),
                                "archive_reason": "contained_waiting_conflict_sibling",
                                "canonical_work_order_id": keeper_id or None,
                                "previous_status": "waiting_conflict",
                            }
                        )
                        item["metadata"] = metadata
                        item["status"] = "discarded"
                        if not _optional_text(item.get("failure_reason")):
                            item["failure_reason"] = "superseded_waiting_conflict"
                        changed = True
                        archived += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)

            grouped_waiting_by_goal: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
            for record in records:
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    if not _work_order_should_archive_superseded_waiting_conflict(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status_by_id.get(_optional_text(item.get("lease_id"))),
                    ):
                        continue
                    group_key = _superseded_waiting_conflict_group_key(item, run=record)
                    if not group_key:
                        continue
                    grouped_waiting_by_goal.setdefault(group_key, []).append((record, item))

            changed_run_ids: set[str] = set()
            for siblings in grouped_waiting_by_goal.values():
                for record, item in siblings:
                    if _optional_text(item.get("status")).lower() != "waiting_conflict":
                        continue
                    metadata = dict(item.get("metadata") or {})
                    if _optional_text(metadata.get("archived_due_to")):
                        continue
                    record_run_id = _optional_text(record.get("run_id"))
                    containing_siblings = [
                        (sibling_record, sibling)
                        for sibling_record, sibling in siblings
                        if sibling is not item
                        and _optional_text(sibling_record.get("run_id")) != record_run_id
                        and _optional_text(sibling.get("status")).lower() == "waiting_conflict"
                        and _work_order_scope_contains(sibling, item)
                        and not _work_order_scope_contains(item, sibling)
                    ]
                    if not containing_siblings:
                        continue
                    keeper_record, keeper = max(
                        containing_siblings,
                        key=lambda pair: _containing_waiting_conflict_priority(
                            pair[1], run=pair[0]
                        ),
                    )
                    metadata.update(
                        {
                            "archived_due_to": "superseded_waiting_conflict",
                            "archived_at": now.isoformat(),
                            "archive_reason": "cross_run_contained_waiting_conflict_sibling",
                            "canonical_run_id": _optional_text(keeper_record.get("run_id")) or None,
                            "canonical_work_order_id": _optional_text(
                                keeper.get("work_order_id"),
                                keeper.get("task_id"),
                            )
                            or None,
                            "previous_status": "waiting_conflict",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "superseded_waiting_conflict"
                    changed_run_ids.add(record_run_id)
                    archived += 1

            for record in records:
                run_id = _optional_text(record.get("run_id"))
                if run_id not in changed_run_ids:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return archived

    def archive_duplicate_waiting_conflict_work_orders(self) -> int:
        """Collapse duplicate no-artifact waiting_conflict rows across runs."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            records = [self._supervisor_run_from_row(row) for row in rows]
            grouped: dict[
                tuple[str, tuple[str, ...]], list[tuple[dict[str, Any], dict[str, Any]]]
            ] = {}
            for record in records:
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_is_duplicate_waiting_conflict_candidate(
                        item,
                        run=record,
                        lease_status=lease_status,
                    ):
                        continue
                    group_key = _duplicate_waiting_conflict_group_key(item, run=record)
                    if not group_key:
                        continue
                    grouped.setdefault(group_key, []).append((record, item))

            archived = 0
            changed_run_ids: set[str] = set()
            for _, siblings in grouped.items():
                if len(siblings) < 2:
                    continue
                keeper_record, keeper_item = max(
                    siblings,
                    key=lambda pair: _duplicate_waiting_conflict_priority(pair[1], run=pair[0]),
                )
                keeper_run_id = _optional_text(keeper_record.get("run_id"))
                keeper_id = _optional_text(
                    keeper_item.get("work_order_id"),
                    keeper_item.get("task_id"),
                )
                for record, item in siblings:
                    if record is keeper_record and item is keeper_item:
                        continue
                    metadata = dict(item.get("metadata") or {})
                    if _optional_text(metadata.get("archived_due_to")):
                        continue
                    metadata.update(
                        {
                            "archived_due_to": "duplicate_waiting_conflict",
                            "archived_at": now,
                            "archive_reason": "duplicate_waiting_conflict",
                            "canonical_run_id": keeper_run_id or None,
                            "canonical_work_order_id": keeper_id or None,
                            "previous_status": _optional_text(item.get("status"))
                            or "waiting_conflict",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "duplicate_waiting_conflict"
                    archived += 1
                    changed_run_ids.add(_optional_text(record.get("run_id")))

            waiting_by_scope: dict[
                tuple[str, ...], list[tuple[dict[str, Any], dict[str, Any]]]
            ] = {}
            for record in records:
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_is_duplicate_waiting_conflict_candidate(
                        item,
                        run=record,
                        lease_status=lease_status,
                    ):
                        continue
                    scope_key = _canonical_work_order_scope_key(item)
                    if not scope_key:
                        continue
                    waiting_by_scope.setdefault(scope_key, []).append((record, item))

            for siblings in waiting_by_scope.values():
                umbrella_candidates = [
                    (record, item)
                    for record, item in siblings
                    if _work_order_is_broad_explicit_pytest_umbrella(item, run=record)
                ]
                if not umbrella_candidates:
                    continue
                keeper_record, keeper_item = max(
                    umbrella_candidates,
                    key=lambda pair: _duplicate_waiting_conflict_priority(pair[1], run=pair[0]),
                )
                keeper_run_id = _optional_text(keeper_record.get("run_id"))
                keeper_id = _optional_text(
                    keeper_item.get("work_order_id"),
                    keeper_item.get("task_id"),
                )
                for record, item in siblings:
                    if record is keeper_record and item is keeper_item:
                        continue
                    if not _work_order_is_specific_pytest_child(item, run=record):
                        continue
                    metadata = dict(item.get("metadata") or {})
                    if _optional_text(metadata.get("archived_due_to")):
                        continue
                    metadata.update(
                        {
                            "archived_due_to": "duplicate_waiting_conflict",
                            "archived_at": now,
                            "archive_reason": "broader_explicit_pytest_waiting_conflict",
                            "canonical_run_id": keeper_run_id or None,
                            "canonical_work_order_id": keeper_id or None,
                            "previous_status": _optional_text(item.get("status"))
                            or "waiting_conflict",
                        }
                    )
                    item["metadata"] = metadata
                    item["status"] = "discarded"
                    if not _optional_text(item.get("failure_reason")):
                        item["failure_reason"] = "duplicate_waiting_conflict"
                    archived += 1
                    changed_run_ids.add(_optional_text(record.get("run_id")))

            for record in records:
                run_id = _optional_text(record.get("run_id"))
                if run_id not in changed_run_ids:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)

            if archived:
                conn.commit()
            else:
                conn.rollback()
            return archived
        finally:
            conn.close()

    def rehabilitate_narrowed_waiting_conflict_work_orders(
        self,
        *,
        grace_period_hours: float = _SUPERSEDED_WAITING_CONFLICT_ARCHIVE_GRACE_HOURS,
    ) -> int:
        """Narrow stale waiting-conflict scopes and requeue lanes that are no longer truly blocked."""
        now = _utcnow()
        grace_period = timedelta(hours=max(0.0, float(grace_period_hours)))
        cutoff = now - grace_period
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            lease_status_by_id = {
                str(row["lease_id"]).strip(): str(row["status"]).strip()
                for row in conn.execute("SELECT lease_id, status FROM leases").fetchall()
                if str(row["lease_id"]).strip()
            }
            records = [self._supervisor_run_from_row(row) for row in rows]
            updated = 0
            for record in records:
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    lease_status = lease_status_by_id.get(_optional_text(item.get("lease_id")))
                    if not _work_order_should_rehabilitate_narrowed_waiting_conflict(
                        item,
                        run=record,
                        cutoff=cutoff,
                        lease_status=lease_status,
                    ):
                        continue
                    narrowed_scope = _narrow_waiting_conflict_scope_from_explicit_paths(
                        item,
                        run=record,
                        repo_root=self.repo_root,
                    )
                    if not narrowed_scope:
                        continue

                    original_scope = [
                        str(path).strip()
                        for path in item.get("file_scope", [])
                        if str(path).strip()
                    ]
                    item_changed = False
                    if _canonical_work_order_scope_key({"file_scope": narrowed_scope}) != (
                        _canonical_work_order_scope_key(item)
                    ):
                        item["file_scope"] = list(narrowed_scope)
                        metadata = dict(item.get("metadata") or {})
                        metadata["waiting_conflict_scope_narrowed_at"] = now.isoformat()
                        metadata["waiting_conflict_original_scope"] = original_scope
                        item["metadata"] = metadata
                        item_changed = True

                    lease_conflicts = self._find_conflicting_leases_locked(
                        conn,
                        allowed_globs=_work_order_scope_patterns(item),
                        claimed_paths=[],
                        owner_session_id=_optional_text(item.get("owner_session_id")),
                    )
                    sibling_conflicts = _blocking_waiting_conflict_siblings(
                        item,
                        run=record,
                        records=records,
                    )
                    item["conflicts"] = [*lease_conflicts, *sibling_conflicts]

                    if not lease_conflicts and not sibling_conflicts:
                        metadata = dict(item.get("metadata") or {})
                        metadata["waiting_conflict_requeued_at"] = now.isoformat()
                        metadata["waiting_conflict_previous_scope"] = original_scope
                        metadata["waiting_conflict_requeue_reason"] = (
                            "narrowed_scope_cleared_container_only_blockers"
                        )
                        item["metadata"] = metadata
                        item["status"] = "queued"
                        item["blockers"] = []
                        item["conflicts"] = []
                        item.pop("review_status", None)
                        for key in (
                            "failure_reason",
                            "blocking_question",
                            "blocker",
                            "dispatch_error",
                        ):
                            item.pop(key, None)
                        item_changed = True
                    else:
                        item["status"] = "waiting_conflict"
                        item["failure_reason"] = "waiting_conflict"
                        _backfill_work_order_blocker_metadata(item)
                        item["blockers"] = _developer_task_blockers(item)

                    if not item_changed:
                        continue
                    changed = True
                    updated += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = now.isoformat()
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return updated

    def backfill_file_scope_from_changed_paths(self) -> int:
        """Repair historical empty-scope rows from concrete changed-path evidence."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            backfilled = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    if not _work_order_should_backfill_file_scope_from_changed_paths(item):
                        continue
                    changed_paths = [
                        str(path).strip()
                        for path in item.get("changed_paths", []) or []
                        if str(path).strip()
                    ]
                    if not changed_paths:
                        continue
                    item["file_scope"] = list(dict.fromkeys(changed_paths))
                    metadata = dict(item.get("metadata") or {})
                    metadata["backfilled_file_scope_from_changed_paths"] = True
                    metadata["file_scope_backfilled_at"] = now
                    item["metadata"] = metadata
                    changed = True
                    backfilled += 1
                if not changed:
                    continue
                record["updated_at"] = now
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return backfilled

    def backfill_missing_completion_receipts(self) -> int:
        """Attach or synthesize missing receipts for stored deliverable work orders."""
        self.backfill_file_scope_from_changed_paths()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
        finally:
            conn.close()

        backfilled = 0
        for row in rows:
            record = self._supervisor_run_from_row(row)
            changed = False
            for item in record["work_orders"]:
                if not isinstance(item, dict):
                    continue
                if not _work_order_should_backfill_receipt(item):
                    continue
                lease_id = _optional_text(item.get("lease_id"))
                if not lease_id:
                    continue
                existing = self.list_completion_receipts(lease_id=lease_id, limit=1)
                if existing:
                    receipt = existing[0]
                else:
                    try:
                        receipt = self.record_completion(
                            lease_id=lease_id,
                            owner_agent=_optional_text(item.get("target_agent")),
                            owner_session_id=_optional_text(item.get("owner_session_id")),
                            branch=_optional_text(item.get("branch")),
                            worktree_path=_optional_text(item.get("worktree_path")),
                            base_sha=_optional_text(item.get("initial_head")),
                            head_sha=_optional_text(item.get("head_sha")),
                            commit_shas=list(item.get("commit_shas", []) or []),
                            changed_paths=list(item.get("changed_paths", []) or []),
                            tests_run=list(item.get("tests_run", []) or []),
                            validations_run=list(item.get("validations_run", []) or []),
                            assumptions=[],
                            blockers=[
                                str(blocker).strip()
                                for blocker in item.get("blockers", [])
                                if str(blocker).strip()
                            ],
                            outcome=_work_order_receipt_outcome(item),
                            risks=[
                                str(blocker).strip()
                                for blocker in item.get("blockers", [])
                                if str(blocker).strip()
                            ],
                            pr_url=_optional_text(item.get("pr_url"), item.get("adopted_pr")),
                            pr_number=_extract_pr_number(
                                _optional_text(item.get("pr_url"), item.get("adopted_pr"))
                            ),
                            confidence=float(item.get("confidence", 0.0) or 0.0),
                            metadata={
                                "task_key": _optional_text(item.get("task_key")) or None,
                                "verification_results": list(
                                    item.get("verification_results", []) or []
                                ),
                                "worker_outcome": _optional_text(item.get("worker_outcome"))
                                or None,
                                "approval_required": bool(item.get("approval_required", False)),
                                "risk_level": _optional_text(item.get("risk_level")) or None,
                                "success_criteria": dict(item.get("success_criteria") or {}),
                                "backfilled_receipt": True,
                            },
                            require_session_ownership=False,
                        )
                    except FileScopeViolationError:
                        if not self._rehydrate_lease_scope_from_work_order(item):
                            continue
                        try:
                            receipt = self.record_completion(
                                lease_id=lease_id,
                                owner_agent=_optional_text(item.get("target_agent")),
                                owner_session_id=_optional_text(item.get("owner_session_id")),
                                branch=_optional_text(item.get("branch")),
                                worktree_path=_optional_text(item.get("worktree_path")),
                                base_sha=_optional_text(item.get("initial_head")),
                                head_sha=_optional_text(item.get("head_sha")),
                                commit_shas=list(item.get("commit_shas", []) or []),
                                changed_paths=list(item.get("changed_paths", []) or []),
                                tests_run=list(item.get("tests_run", []) or []),
                                validations_run=list(item.get("validations_run", []) or []),
                                assumptions=[],
                                blockers=[
                                    str(blocker).strip()
                                    for blocker in item.get("blockers", [])
                                    if str(blocker).strip()
                                ],
                                outcome=_work_order_receipt_outcome(item),
                                risks=[
                                    str(blocker).strip()
                                    for blocker in item.get("blockers", [])
                                    if str(blocker).strip()
                                ],
                                pr_url=_optional_text(item.get("pr_url"), item.get("adopted_pr")),
                                pr_number=_extract_pr_number(
                                    _optional_text(item.get("pr_url"), item.get("adopted_pr"))
                                ),
                                confidence=float(item.get("confidence", 0.0) or 0.0),
                                metadata={
                                    "task_key": _optional_text(item.get("task_key")) or None,
                                    "verification_results": list(
                                        item.get("verification_results", []) or []
                                    ),
                                    "worker_outcome": _optional_text(item.get("worker_outcome"))
                                    or None,
                                    "approval_required": bool(item.get("approval_required", False)),
                                    "risk_level": _optional_text(item.get("risk_level")) or None,
                                    "success_criteria": dict(item.get("success_criteria") or {}),
                                    "backfilled_receipt": True,
                                    "lease_scope_rehydrated": True,
                                },
                                require_session_ownership=False,
                            )
                        except (FileScopeViolationError, KeyError, ValueError):
                            continue
                    except (KeyError, ValueError):
                        continue
                item["receipt_id"] = receipt.receipt_id
                item["confidence"] = receipt.confidence
                changed = True
                backfilled += 1
            if not changed:
                continue
            self.update_supervisor_run(record["run_id"], work_orders=record["work_orders"])
        return backfilled

    def _rehydrate_lease_scope_from_work_order(self, work_order: dict[str, Any]) -> bool:
        lease_id = _optional_text(work_order.get("lease_id"))
        if not lease_id:
            return False
        patterns = _work_order_scope_patterns(work_order)
        if not patterns:
            return False
        changed_paths = [
            _normalize_claim(str(item))
            for item in work_order.get("changed_paths", []) or []
            if _normalize_claim(str(item))
        ]
        if changed_paths and not all(
            any(_path_matches_glob(path, pattern) for pattern in patterns) for path in changed_paths
        ):
            return False
        allowed_globs = [pattern for pattern in patterns if _has_wildcard(pattern)]
        claimed_paths = [pattern for pattern in patterns if not _has_wildcard(pattern)]
        if not allowed_globs and not claimed_paths:
            claimed_paths = list(patterns)

        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM leases WHERE lease_id = ?", (lease_id,)).fetchone()
            if row is None:
                return False
            lease = WorkLease.from_row(row)
            if lease.claimed_paths or lease.allowed_globs:
                return False
            metadata = _json_loads(row["metadata_json"], {})
            metadata.pop("last_scope_violation", None)
            conn.execute(
                """
                UPDATE leases
                SET allowed_globs_json = ?, claimed_paths_json = ?, metadata_json = ?, updated_at = ?
                WHERE lease_id = ?
                """,
                (
                    _json_dump(allowed_globs),
                    _json_dump(claimed_paths),
                    _json_dump(metadata),
                    now,
                    lease_id,
                ),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    @staticmethod
    def _run_verification_commands_sync(
        worktree_path: str,
        commands: list[str],
        *,
        timeout: float = 900.0,
    ) -> list[dict[str, Any]]:
        from aragora.swarm.worker_launcher import WorkerLauncher

        results: list[dict[str, Any]] = []
        verification_env = WorkerLauncher._verification_environment(worktree_path)
        for raw_command in commands:
            command = str(raw_command).strip()
            if not command:
                continue
            execution_command = WorkerLauncher._prepare_verification_command(command)
            command_timeout = _verification_timeout_for_command(command, timeout)
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    ["/bin/bash", "-lc", execution_command],
                    cwd=worktree_path,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                    check=False,
                    env=verification_env,
                )
                exit_code = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
            except subprocess.TimeoutExpired as exc:
                exit_code = -1
                stdout = exc.stdout or ""
                stderr = exc.stderr or f"Timed out after {int(command_timeout)}s"
            except OSError as exc:
                exit_code = -2
                stdout = ""
                stderr = str(exc)
            results.append(
                {
                    "command": command,
                    "exit_code": exit_code,
                    "passed": exit_code == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "duration_seconds": round(time.monotonic() - started, 3),
                }
            )
        return results

    def _resolve_verification_worktree(
        self,
        work_order: dict[str, Any],
    ) -> tuple[str, Path | None]:
        worktree_path = Path(str(work_order.get("worktree_path") or "").strip())
        if worktree_path.is_dir():
            head_sha = str(work_order.get("head_sha") or "").strip()
            if head_sha:
                proc = subprocess.run(
                    ["git", "-C", str(worktree_path), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                actual_sha = proc.stdout.strip() if proc.returncode == 0 else ""
                if not actual_sha or not (
                    actual_sha.startswith(head_sha) or head_sha.startswith(actual_sha)
                ):
                    pass  # fall through to create fresh worktree
                else:
                    return str(worktree_path), None
            else:
                return str(worktree_path), None

        subprocess.run(
            ["git", "-C", str(self.repo_root), "worktree", "prune"],
            capture_output=True,
            text=True,
            check=False,
        )
        ref_candidates = [
            _optional_text(work_order.get("branch")),
            _optional_text(work_order.get("head_sha")),
        ]
        ref = ""
        for candidate in ref_candidates:
            if not candidate:
                continue
            proc = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-parse", "--verify", "--quiet", candidate],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                ref = candidate
                break
        if not ref:
            return "", None

        parent = self.repo_root / ".worktrees" / "verification-replay"
        parent.mkdir(parents=True, exist_ok=True)
        temp_path = parent / f"verify-{str(uuid.uuid4())[:8]}"
        proc = subprocess.run(
            ["git", "-C", str(self.repo_root), "worktree", "add", "--detach", str(temp_path), ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return "", None
        return str(temp_path), temp_path

    def _cleanup_verification_worktree(self, worktree_path: Path | None) -> None:
        if worktree_path is None:
            return
        subprocess.run(
            ["git", "-C", str(self.repo_root), "worktree", "remove", "--force", str(worktree_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(self.repo_root), "worktree", "prune"],
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _update_completion_receipt_verification_locked(
        conn: sqlite3.Connection,
        *,
        receipt_id: str,
        verification_results: list[dict[str, Any]],
        replayed_at: str,
    ) -> bool:
        row = conn.execute(
            "SELECT * FROM completion_receipts WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchone()
        if row is None:
            return False
        receipt = CompletionReceipt.from_row(row)
        tests_run = [
            str(entry.get("command", "")).strip()
            for entry in verification_results
            if str(entry.get("command", "")).strip()
        ]
        metadata = dict(receipt.metadata)
        metadata["verification_results"] = [dict(entry) for entry in verification_results]
        metadata["verification_replayed"] = True
        metadata["verification_replayed_at"] = replayed_at
        updated_receipt = CompletionReceipt(
            receipt_id=receipt.receipt_id,
            lease_id=receipt.lease_id,
            task_id=receipt.task_id,
            owner_agent=receipt.owner_agent,
            owner_session_id=receipt.owner_session_id,
            branch=receipt.branch,
            worktree_path=receipt.worktree_path,
            base_sha=receipt.base_sha,
            head_sha=receipt.head_sha,
            commit_shas=list(receipt.commit_shas),
            changed_paths=list(receipt.changed_paths),
            tests_run=tests_run,
            validations_run=tests_run,
            assumptions=list(receipt.assumptions),
            blockers=list(receipt.blockers),
            outcome=receipt.outcome,
            risks=list(receipt.risks),
            pr_url=receipt.pr_url,
            pr_number=receipt.pr_number,
            confidence=receipt.confidence,
            created_at=receipt.created_at,
            metadata=metadata,
        )
        conn.execute(
            """
            UPDATE completion_receipts
            SET tests_run_json = ?, validations_run_json = ?, artifact_hash = ?, metadata_json = ?
            WHERE receipt_id = ?
            """,
            (
                _json_dump(updated_receipt.tests_run),
                _json_dump(updated_receipt.validations_run),
                updated_receipt.artifact_hash,
                _json_dump(updated_receipt.metadata),
                receipt_id,
            ),
        )
        return True

    def _replay_merge_gate_failures(
        self,
        *,
        should_replay: Any,
        metadata_flag: str,
        prepare_commands: Any | None = None,
        merge_existing_results: bool = False,
        task_keys: list[str] | None = None,
        limit: int | None = None,
        timeout: float = 900.0,
    ) -> int:
        normalized_task_keys = {
            str(task_key).strip() for task_key in (task_keys or []) if str(task_key).strip()
        }
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            candidate_ids: list[tuple[str, str]] = []
            attempted = 0
            for row in rows:
                if isinstance(limit, int) and limit > 0 and attempted >= limit:
                    break
                record = self._supervisor_run_from_row(row)
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    if isinstance(limit, int) and limit > 0 and attempted >= limit:
                        break
                    if not _merge_gate_replay_matches_task_keys(
                        record["run_id"], item, normalized_task_keys
                    ):
                        continue
                    if not should_replay(item):
                        continue
                    work_order_id = _work_order_identifier(item)
                    if not work_order_id:
                        continue
                    candidate_ids.append((record["run_id"], work_order_id))
                    attempted += 1
        finally:
            conn.close()

        replayed = 0
        for run_id, work_order_id in candidate_ids:
            record = self.get_supervisor_run(run_id)
            if not record:
                continue
            item = _find_work_order(record, work_order_id)
            if item is None or not _merge_gate_replay_matches_task_keys(
                run_id, item, normalized_task_keys
            ):
                continue
            if not should_replay(item):
                continue

            commands = [
                str(command).strip()
                for command in item.get("expected_tests", [])
                if str(command).strip()
            ]
            if prepare_commands is not None:
                prepared_commands = prepare_commands(item)
                if prepared_commands is None:
                    continue
                commands = [
                    str(command).strip() for command in prepared_commands if str(command).strip()
                ]
            if not commands:
                continue

            worktree_path, cleanup_path = self._resolve_verification_worktree(item)
            if not worktree_path:
                continue
            try:
                verification_results = self._run_verification_commands_sync(
                    worktree_path,
                    commands,
                    timeout=timeout,
                )
            finally:
                self._cleanup_verification_worktree(cleanup_path)
            if not verification_results:
                continue

            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM supervisor_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    continue
                record = self._supervisor_run_from_row(row)
                item = _find_work_order(record, work_order_id)
                if item is None or not _merge_gate_replay_matches_task_keys(
                    run_id, item, normalized_task_keys
                ):
                    continue
                if not should_replay(item):
                    continue
                if prepare_commands is not None and prepare_commands(item) is None:
                    continue

                existing_results = [
                    dict(entry)
                    for entry in item.get("verification_results", [])
                    if isinstance(entry, dict) and str(entry.get("command", "")).strip()
                ]
                existing_tests_run = [
                    str(command).strip()
                    for command in item.get("tests_run", [])
                    if str(command).strip()
                ]
                effective_results = [dict(entry) for entry in verification_results]
                effective_tests_run = [
                    str(entry.get("command", "")).strip()
                    for entry in effective_results
                    if str(entry.get("command", "")).strip()
                ]
                if merge_existing_results:
                    seen_commands = {
                        _canonical_verification_command(entry.get("command", ""))
                        for entry in effective_results
                        if _canonical_verification_command(entry.get("command", ""))
                    }
                    for entry in existing_results:
                        canonical = _canonical_verification_command(entry.get("command", ""))
                        if canonical and canonical not in seen_commands:
                            effective_results.append(dict(entry))
                            seen_commands.add(canonical)
                    seen_tests = {
                        _canonical_verification_command(command)
                        for command in effective_tests_run
                        if _canonical_verification_command(command)
                    }
                    for command in existing_tests_run:
                        canonical = _canonical_verification_command(command)
                        if canonical and canonical not in seen_tests:
                            effective_tests_run.append(command)
                            seen_tests.add(canonical)

                item["tests_run"] = effective_tests_run
                item["verification_results"] = [dict(entry) for entry in effective_results]
                item["merge_gate"] = _merge_gate_state_for_work_order(item)
                metadata = dict(item.get("metadata") or {})
                metadata[metadata_flag] = True
                metadata[f"{metadata_flag}_at"] = _utcnow().isoformat()
                item["metadata"] = metadata
                receipt_id = _optional_text(item.get("receipt_id"))
                if receipt_id:
                    self._update_completion_receipt_verification_locked(
                        conn,
                        receipt_id=receipt_id,
                        verification_results=effective_results,
                        replayed_at=metadata[f"{metadata_flag}_at"],
                    )
                if item["merge_gate"]["checks_passed"]:
                    item["status"] = "completed"
                    item["review_status"] = "pending_heterogeneous_review"
                    item["worker_outcome"] = "completed"
                    item["blockers"] = []
                    for key in (
                        "failure_reason",
                        "blocking_question",
                        "blocker",
                        "dispatch_error",
                    ):
                        item.pop(key, None)
                else:
                    item["status"] = "needs_human"
                    item["review_status"] = "changes_requested"
                    item["worker_outcome"] = "merge_gate_failed"
                    item["failure_reason"] = "merge_gate_failed"
                    item["dispatch_error"] = (
                        item["merge_gate"]["blocked_reasons"][0]
                        if item["merge_gate"]["blocked_reasons"]
                        else "merge gate blocked"
                    )
                    item["blockers"] = list(item["merge_gate"]["blocked_reasons"])
                    _backfill_work_order_blocker_metadata(item)
                    if _work_order_should_reclassify_deliverable_changes_requested(item):
                        item["status"] = "changes_requested"

                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = _utcnow().isoformat()
                self._persist_supervisor_run(conn, record)
                conn.commit()
                replayed += 1
            finally:
                conn.close()
        return replayed

    def replay_missing_verification_for_merge_gate_failures(
        self,
        *,
        task_keys: list[str] | None = None,
        limit: int | None = None,
        timeout: float = 900.0,
    ) -> int:
        return self._replay_merge_gate_failures(
            should_replay=_work_order_should_replay_missing_verification,
            metadata_flag="verification_replayed",
            task_keys=task_keys,
            limit=limit,
            timeout=timeout,
        )

    def replay_environment_blocked_merge_gate_failures(
        self,
        *,
        task_keys: list[str] | None = None,
        limit: int | None = None,
        timeout: float = 900.0,
    ) -> int:
        return self._replay_merge_gate_failures(
            should_replay=_work_order_should_replay_environment_blocked_verification,
            metadata_flag="verification_environment_replayed",
            task_keys=task_keys,
            limit=limit,
            timeout=timeout,
        )

    def replay_docs_only_merge_gate_failures(
        self,
        *,
        task_keys: list[str] | None = None,
        limit: int | None = None,
        timeout: float = 900.0,
    ) -> int:
        return self._replay_merge_gate_failures(
            should_replay=_work_order_should_replay_docs_only_merge_gate_failure,
            metadata_flag="verification_docs_replayed",
            prepare_commands=_docs_only_replay_commands_for_work_order,
            task_keys=task_keys,
            limit=limit,
            timeout=timeout,
        )

    def replay_missing_required_merge_gate_failures(
        self,
        *,
        task_keys: list[str] | None = None,
        limit: int | None = None,
        timeout: float = 900.0,
    ) -> int:
        return self._replay_merge_gate_failures(
            should_replay=_work_order_should_replay_missing_required_merge_gate_failure,
            metadata_flag="verification_missing_required_replayed",
            prepare_commands=_missing_required_replay_commands_for_work_order,
            merge_existing_results=True,
            task_keys=task_keys,
            limit=limit,
            timeout=timeout,
        )

    def replay_narrow_pytest_merge_gate_failures(
        self,
        *,
        task_keys: list[str] | None = None,
        limit: int | None = None,
        timeout: float = 900.0,
    ) -> int:
        return self._replay_merge_gate_failures(
            should_replay=_work_order_should_replay_narrow_pytest_merge_gate_failure,
            metadata_flag="verification_narrow_pytest_replayed",
            prepare_commands=_narrow_pytest_replay_commands_for_work_order,
            task_keys=task_keys,
            limit=limit,
            timeout=timeout,
        )

    def replay_targeted_merge_gate_failures(
        self,
        *,
        task_keys: list[str] | None = None,
        limit: int | None = None,
        timeout: float = 900.0,
    ) -> int:
        def _prepare(item: dict[str, Any]) -> list[str] | None:
            targeted_commands = _targeted_replay_expected_tests_for_work_order(item)
            if not targeted_commands:
                return None
            previous_expected = [
                str(command).strip()
                for command in item.get("expected_tests", [])
                if str(command).strip()
            ]
            metadata = dict(item.get("metadata") or {})
            if previous_expected and not metadata.get(
                "verification_targeted_previous_expected_tests"
            ):
                metadata["verification_targeted_previous_expected_tests"] = list(previous_expected)
            item["metadata"] = metadata
            item["expected_tests"] = list(targeted_commands)
            success_criteria = dict(item.get("success_criteria") or {})
            success_criteria["tests"] = (
                targeted_commands[0] if len(targeted_commands) == 1 else list(targeted_commands)
            )
            item["success_criteria"] = success_criteria
            item["tests_run"] = []
            item["verification_results"] = []
            return targeted_commands

        return self._replay_merge_gate_failures(
            should_replay=_work_order_should_replay_targeted_merge_gate_failure,
            metadata_flag="verification_targeted_replayed",
            prepare_commands=_prepare,
            task_keys=task_keys,
            limit=limit,
            timeout=timeout,
        )

    def reclassify_branch_stale_merge_gate_failures(
        self,
        *,
        limit: int | None = None,
        timeout: float = 900.0,
        task_keys: list[str] | None = None,
    ) -> int:
        requested_task_keys = {str(item).strip() for item in (task_keys or []) if str(item).strip()}

        def _task_key_for(record: dict[str, Any], item: dict[str, Any], work_order_id: str) -> str:
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                task_key = _optional_text(metadata.get("task_key"))
                if task_key:
                    return task_key
            task_key = _optional_text(item.get("task_key"))
            if task_key:
                return task_key
            if record.get("run_id") and work_order_id:
                return f"{record['run_id']}:{work_order_id}"
            return ""

        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            candidate_ids: list[tuple[str, str]] = []
            attempted = 0
            for row in rows:
                if isinstance(limit, int) and limit > 0 and attempted >= limit:
                    break
                record = self._supervisor_run_from_row(row)
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    if isinstance(limit, int) and limit > 0 and attempted >= limit:
                        break
                    if not (
                        _work_order_should_reclassify_branch_stale_merge_gate_failure(item)
                        or _work_order_should_reclassify_branch_stale_verification_target_missing(
                            item, repo_root=self.repo_root
                        )
                    ):
                        continue
                    work_order_id = _work_order_identifier(item)
                    if not work_order_id:
                        continue
                    task_key = _task_key_for(record, item, work_order_id)
                    if requested_task_keys and task_key not in requested_task_keys:
                        continue
                    candidate_ids.append((record["run_id"], work_order_id))
                    attempted += 1
        finally:
            conn.close()

        reclassified = 0
        for run_id, work_order_id in candidate_ids:
            record = self.get_supervisor_run(run_id)
            if not record:
                continue
            item = _find_work_order(record, work_order_id)
            is_missing_target = False
            if item is None:
                continue
            if _work_order_should_reclassify_branch_stale_merge_gate_failure(item):
                pass
            elif _work_order_should_reclassify_branch_stale_verification_target_missing(
                item, repo_root=self.repo_root
            ):
                is_missing_target = True
            else:
                continue
            task_key = _task_key_for(record, item, work_order_id)
            if requested_task_keys and task_key not in requested_task_keys:
                continue
            target_ref = _optional_text(record.get("target_branch")) or "main"
            commands = _mainline_verification_commands_for_work_order(item)
            verification_results: list[dict[str, Any]] = []
            missing_paths: list[str] = []
            if is_missing_target:
                missing_paths = _mainline_missing_repo_paths_for_work_order(
                    item, repo_root=self.repo_root
                )
                if not missing_paths:
                    continue
            else:
                if not commands:
                    continue
                worktree_path, cleanup_path = self._resolve_verification_worktree(
                    {"branch": target_ref}
                )
                if not worktree_path and target_ref != "main":
                    target_ref = "main"
                    worktree_path, cleanup_path = self._resolve_verification_worktree(
                        {"branch": target_ref}
                    )
                if not worktree_path:
                    continue
                try:
                    verification_results = self._run_verification_commands_sync(
                        worktree_path,
                        commands,
                        timeout=timeout,
                    )
                finally:
                    self._cleanup_verification_worktree(cleanup_path)
                if not verification_results or not all(
                    bool(entry.get("passed", False)) for entry in verification_results
                ):
                    continue

            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM supervisor_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    continue
                refreshed_record = self._supervisor_run_from_row(row)
                refreshed_item = _find_work_order(refreshed_record, work_order_id)
                refreshed_is_missing_target = False
                if refreshed_item is None:
                    continue
                if _work_order_should_reclassify_branch_stale_merge_gate_failure(refreshed_item):
                    pass
                elif _work_order_should_reclassify_branch_stale_verification_target_missing(
                    refreshed_item, repo_root=self.repo_root
                ):
                    refreshed_is_missing_target = True
                else:
                    continue
                refreshed_task_key = _task_key_for(refreshed_record, refreshed_item, work_order_id)
                if requested_task_keys and refreshed_task_key not in requested_task_keys:
                    continue

                checked_at = _utcnow().isoformat()
                metadata = dict(refreshed_item.get("metadata") or {})
                metadata["mainline_verification_checked_at"] = checked_at
                if refreshed_is_missing_target:
                    metadata["mainline_verification_target_missing"] = True
                    metadata["mainline_missing_paths"] = list(missing_paths)
                    metadata["mainline_verification_commands"] = list(commands)
                else:
                    metadata["mainline_verification_passed"] = True
                    metadata["mainline_verification_commands"] = list(commands)
                    metadata["mainline_verification_results"] = [
                        dict(entry) for entry in verification_results
                    ]
                refreshed_item["metadata"] = metadata
                refreshed_item["status"] = "changes_requested"
                refreshed_item["review_status"] = "changes_requested"
                refreshed_item["worker_outcome"] = "branch_snapshot_stale"
                refreshed_item["failure_reason"] = "branch_snapshot_stale"
                refreshed_item["blocking_question"] = _default_blocking_question_for_reason(
                    "branch_snapshot_stale"
                )
                if refreshed_is_missing_target:
                    refreshed_item["dispatch_error"] = (
                        "branch snapshot stale: referenced verification targets no longer exist "
                        f"on {target_ref}"
                    )
                else:
                    refreshed_item["dispatch_error"] = (
                        f"branch snapshot stale: merge-gate verification now passes on {target_ref}"
                    )
                refreshed_item["blockers"] = ["branch_snapshot_stale"]
                refreshed_item["blocker"] = {
                    "reason": "branch_snapshot_stale",
                    "question": refreshed_item["blocking_question"],
                }

                refreshed_record["status"] = self._derive_supervisor_run_status(
                    refreshed_record["work_orders"]
                )
                refreshed_record["updated_at"] = checked_at
                self._persist_supervisor_run(conn, refreshed_record)
                conn.commit()
                reclassified += 1
            finally:
                conn.close()

        return reclassified

    def reconcile_merge_gate_failed_work_orders(self) -> int:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM supervisor_runs ORDER BY updated_at DESC").fetchall()
            reconciled = 0
            for row in rows:
                record = self._supervisor_run_from_row(row)
                changed = False
                for item in record["work_orders"]:
                    if not isinstance(item, dict):
                        continue
                    if not _work_order_should_reconcile_merge_gate_failure(item):
                        continue
                    item["merge_gate"] = _merge_gate_state_for_work_order(item)
                    metadata = dict(item.get("metadata") or {})
                    metadata["merge_gate_reconciled"] = True
                    metadata["merge_gate_reconciled_at"] = _utcnow().isoformat()
                    item["metadata"] = metadata
                    if item["merge_gate"]["checks_passed"]:
                        item["status"] = "completed"
                        item["review_status"] = "pending_heterogeneous_review"
                        item["worker_outcome"] = "completed"
                        item["blockers"] = []
                        for key in (
                            "failure_reason",
                            "blocking_question",
                            "blocker",
                            "dispatch_error",
                        ):
                            item.pop(key, None)
                    else:
                        item["status"] = "needs_human"
                        item["review_status"] = "changes_requested"
                        item["worker_outcome"] = "merge_gate_failed"
                        item["failure_reason"] = "merge_gate_failed"
                        item["dispatch_error"] = (
                            item["merge_gate"]["blocked_reasons"][0]
                            if item["merge_gate"]["blocked_reasons"]
                            else "merge gate blocked"
                        )
                        item["blockers"] = list(item["merge_gate"]["blocked_reasons"])
                        _backfill_work_order_blocker_metadata(item)
                    changed = True
                    reconciled += 1
                if not changed:
                    continue
                record["status"] = self._derive_supervisor_run_status(record["work_orders"])
                record["updated_at"] = _utcnow().isoformat()
                self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return reconciled

    def get_developer_task(self, task_key: str) -> DeveloperTask | None:
        for task in self.list_developer_tasks(limit=500):
            if task.task_key == str(task_key).strip():
                return task
        return None

    def update_supervisor_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        work_orders: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM supervisor_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown supervisor run: {run_id}")
            record = self._supervisor_run_from_row(row)
            if status is not None:
                record["status"] = status
            if work_orders is not None:
                record["work_orders"] = [dict(item) for item in work_orders]
            if metadata:
                record["metadata"] = {
                    **dict(record.get("metadata") or {}),
                    **dict(metadata),
                }
            record["updated_at"] = _utcnow().isoformat()
            self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()
        return record

    def list_active_leases(self) -> list[WorkLease]:
        now = _utcnow()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM leases ORDER BY created_at ASC").fetchall()
        finally:
            conn.close()
        leases = [WorkLease.from_row(row) for row in rows]
        active: list[WorkLease] = []
        for lease in leases:
            if lease.status != LeaseStatus.ACTIVE.value:
                continue
            if _parse_dt(lease.expires_at) <= now:
                continue
            active.append(lease)
        return active

    def list_leases(
        self,
        *,
        statuses: list[str] | None = None,
        limit: int | None = 500,
    ) -> list[WorkLease]:
        query = "SELECT * FROM leases ORDER BY updated_at DESC"
        params: tuple[Any, ...] = ()
        if isinstance(limit, int) and limit > 0:
            query += " LIMIT ?"
            params = (limit,)
        conn = self._connect()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        leases = [WorkLease.from_row(row) for row in rows]
        if statuses is None:
            return leases
        allowed = set(statuses)
        return [item for item in leases if item.status in allowed]

    def reap_expired_leases(self) -> list[WorkLease]:
        now = _utcnow()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM leases WHERE status = ?",
                (LeaseStatus.ACTIVE.value,),
            ).fetchall()
            expired = [
                WorkLease.from_row(row) for row in rows if _parse_dt(row["expires_at"]) <= now
            ]
            for lease in expired:
                conn.execute(
                    "UPDATE leases SET status = ?, updated_at = ? WHERE lease_id = ?",
                    (LeaseStatus.EXPIRED.value, now.isoformat(), lease.lease_id),
                )
            conn.commit()
        finally:
            conn.close()

        for lease in expired:
            self._release_fleet_claims_for_lease(lease)
            self._publish(
                "task_expired",
                track=lease.branch,
                data={
                    "lease_id": lease.lease_id,
                    "task_id": lease.task_id,
                    "worktree_path": lease.worktree_path,
                },
            )
            self._sync_supervisor_run_from_lease(
                lease.metadata,
                update={"status": "needs_human", "failure_reason": "expired_lease_reaped"},
            )
        self.backfill_missing_completion_receipts()
        self.backfill_missing_blocker_metadata()
        self.archive_reaped_no_receipt_work_orders()
        self.archive_scope_violation_no_deliverable_work_orders()
        self.archive_failed_no_deliverable_work_orders()
        self.archive_clean_exit_no_deliverable_work_orders()
        self.archive_work_order_leasing_failed_work_orders()
        self.archive_worker_type_blocked_work_orders()
        self.archive_duplicate_work_order_leasing_failed_work_orders()
        self.archive_duplicate_branch_deliverable_work_orders()
        self.archive_superseded_waiting_conflict_work_orders()
        self.archive_duplicate_waiting_conflict_work_orders()
        self.rehabilitate_narrowed_waiting_conflict_work_orders()
        return expired

    def cleanup_stale_supervisor_runs(
        self,
        *,
        max_age_hours: float = 24.0,
        limit: int = 200,
    ) -> int:
        """Mark stale supervisor runs as completed so duplicate detection doesn't block new dispatches.

        A run is stale when:
        - Its status is 'needs_human' or 'completed'
        - ALL of its work orders are in non-actionable states (discarded, needs_human,
          dispatch_failed, failed, timed_out, completed, scope_violation)
        - No work order has an active lease or running process

        This prevents the duplicate_open_work_order detector in
        _suppress_duplicate_open_work_orders from permanently blocking new work
        orders that share file scope with old, abandoned runs.

        Returns:
            Number of runs cleaned up.
        """
        terminal_wo_statuses = {
            "completed",
            "discarded",
            "needs_human",
            "dispatch_failed",
            "failed",
            "timed_out",
            "scope_violation",
            "cancelled",
        }
        runs = self.list_supervisor_runs(limit=limit)
        cleaned = 0
        for run in runs:
            status = str(run.get("status", "")).strip().lower()
            if status not in ("needs_human", "completed"):
                continue
            work_orders = run.get("work_orders", [])
            if not work_orders:
                continue
            all_terminal = all(
                str(wo.get("status", "")).strip().lower() in terminal_wo_statuses
                for wo in work_orders
                if isinstance(wo, dict)
            )
            if not all_terminal:
                continue
            # Mark non-terminal work orders as completed
            changed = False
            for wo in work_orders:
                if not isinstance(wo, dict):
                    continue
                wo_status = str(wo.get("status", "")).strip().lower()
                if wo_status not in ("completed",):
                    wo["status"] = "completed"
                    changed = True
            if changed or status == "needs_human":
                self.update_supervisor_run(
                    run["run_id"],
                    status="completed",
                    work_orders=work_orders,
                )
                cleaned += 1
        return cleaned

    def reap_stale_leases(
        self,
        *,
        stale_threshold_seconds: float = 1800.0,
    ) -> list[WorkLease]:
        """Release active leases whose worker process is dead.

        A lease is stale when its metadata ``worker_pid`` is no longer running,
        **or** no ``worker_pid`` is recorded and the lease has not been
        heartbeated within *stale_threshold_seconds* (default 30 min).

        Complements ``reap_expired_leases`` (TTL only) and the conflict-path
        reaping in ``SwarmSupervisor._release_orphaned_conflict_leases``.
        """
        now = _utcnow()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM leases WHERE status = ?",
                (LeaseStatus.ACTIVE.value,),
            ).fetchall()
        finally:
            conn.close()

        stale: list[WorkLease] = []
        for row in rows:
            lease = WorkLease.from_row(row)
            if _parse_dt(lease.expires_at) <= now:
                continue  # reap_expired_leases handles TTL expiry.

            metadata = lease.metadata or {}
            raw_pid = metadata.get("worker_pid")

            if raw_pid is not None:
                probe = _safe_kill_probe(raw_pid)
                if probe is None or isinstance(probe, PermissionError):
                    continue  # Process alive (or owned by another user).
            else:
                updated = _parse_dt(lease.updated_at)
                if (now - updated).total_seconds() < stale_threshold_seconds:
                    continue

            stale.append(lease)

        if not stale:
            return stale

        conn = self._connect()
        try:
            for lease in stale:
                conn.execute(
                    "UPDATE leases SET status = ?, updated_at = ? WHERE lease_id = ?",
                    (LeaseStatus.EXPIRED.value, now.isoformat(), lease.lease_id),
                )
            conn.commit()
        finally:
            conn.close()

        for lease in stale:
            self._release_fleet_claims_for_lease(lease)
            reason = (
                "worker_pid_dead"
                if (lease.metadata or {}).get("worker_pid") is not None
                else "heartbeat_timeout"
            )
            self._publish(
                "lease_stale",
                track=lease.branch,
                data={
                    "lease_id": lease.lease_id,
                    "task_id": lease.task_id,
                    "worktree_path": lease.worktree_path,
                    "reason": reason,
                },
            )
            self._sync_supervisor_run_from_lease(
                lease.metadata,
                update={"status": "needs_human", "failure_reason": "stale_lease_reaped"},
            )

        self.backfill_missing_completion_receipts()
        self.backfill_missing_blocker_metadata()
        self.archive_reaped_no_receipt_work_orders()
        self.archive_scope_violation_no_deliverable_work_orders()
        self.archive_failed_no_deliverable_work_orders()
        self.archive_clean_exit_no_deliverable_work_orders()
        self.archive_work_order_leasing_failed_work_orders()
        self.archive_worker_type_blocked_work_orders()
        self.archive_duplicate_work_order_leasing_failed_work_orders()
        self.archive_duplicate_branch_deliverable_work_orders()
        self.archive_superseded_waiting_conflict_work_orders()
        self.archive_duplicate_waiting_conflict_work_orders()
        self.rehabilitate_narrowed_waiting_conflict_work_orders()
        return stale

    def list_completion_receipts(
        self,
        lease_id: str | None = None,
        *,
        task_id: str | None = None,
        limit: int | None = None,
    ) -> list[CompletionReceipt]:
        suffix = ""
        params: list[Any] = []
        if isinstance(limit, int) and limit > 0:
            suffix = " LIMIT ?"
        conn = self._connect()
        try:
            if lease_id:
                params = [lease_id]
                if suffix:
                    params.append(limit)
                rows = conn.execute(
                    "SELECT * FROM completion_receipts WHERE lease_id = ? ORDER BY created_at DESC"
                    + suffix,
                    tuple(params),
                ).fetchall()
            elif task_id:
                params = [task_id]
                if suffix:
                    params.append(limit)
                rows = conn.execute(
                    "SELECT * FROM completion_receipts WHERE task_id = ? ORDER BY created_at DESC"
                    + suffix,
                    tuple(params),
                ).fetchall()
            else:
                params = [limit] if suffix else []
                rows = conn.execute(
                    "SELECT * FROM completion_receipts ORDER BY created_at DESC" + suffix,
                    tuple(params),
                ).fetchall()
        finally:
            conn.close()
        return [CompletionReceipt.from_row(row) for row in rows]

    def get_completion_receipt(self, receipt_id: str) -> CompletionReceipt | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM completion_receipts WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else CompletionReceipt.from_row(row)

    def list_integration_decisions(
        self,
        *,
        only_pending: bool = False,
        receipt_id: str | None = None,
        limit: int | None = None,
    ) -> list[IntegrationDecision]:
        suffix = ""
        params: list[Any] = []
        if isinstance(limit, int) and limit > 0:
            suffix = " LIMIT ?"
        conn = self._connect()
        try:
            if receipt_id:
                params = [receipt_id]
                if suffix:
                    params.append(limit)
                rows = conn.execute(
                    "SELECT * FROM integration_decisions WHERE receipt_id = ? ORDER BY created_at DESC"
                    + suffix,
                    tuple(params),
                ).fetchall()
            else:
                params = [limit] if suffix else []
                rows = conn.execute(
                    "SELECT * FROM integration_decisions ORDER BY created_at DESC" + suffix,
                    tuple(params),
                ).fetchall()
        finally:
            conn.close()
        decisions = [IntegrationDecision.from_row(row) for row in rows]
        if only_pending:
            return [item for item in decisions if item.decision in _PENDING_INTEGRATION_DECISIONS]
        return decisions

    def integrator_snapshot(self, *, limit: int = 200) -> dict[str, Any]:
        bounded_limit = max(1, int(limit))
        return {
            "generated_at": _utcnow().isoformat(),
            "leases": [item.to_dict() for item in self.list_leases(limit=bounded_limit)],
            "developer_tasks": [
                item.to_dict()
                for item in self.list_developer_tasks(open_only=False, limit=bounded_limit)
            ],
            "completion_receipts": [
                item.to_dict() for item in self.list_completion_receipts(limit=bounded_limit)
            ],
            "integration_decisions": [
                item.to_dict() for item in self.list_integration_decisions(limit=bounded_limit)
            ],
            "salvage_candidates": [
                item.to_dict()
                for item in self.list_salvage_candidates(statuses=sorted(_OPEN_SALVAGE_STATUSES))
            ],
        }

    def list_salvage_candidates(self, statuses: list[str] | None = None) -> list[SalvageCandidate]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM salvage_candidates ORDER BY updated_at DESC"
            ).fetchall()
        finally:
            conn.close()
        items = [SalvageCandidate.from_row(row) for row in rows]
        if statuses is None:
            return items
        allowed = set(statuses)
        return [item for item in items if item.status in allowed]

    def find_conflicting_leases(
        self,
        *,
        allowed_globs: list[str],
        claimed_paths: list[str],
        owner_session_id: str | None = None,
        exclude_lease_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.fleet_store.reap_stale_claims()
        normalized_globs = [_normalize_claim(item) for item in allowed_globs if str(item).strip()]
        normalized_paths = [_normalize_claim(item) for item in claimed_paths if str(item).strip()]
        conflicts: list[dict[str, Any]] = []
        active_leases = self.list_active_leases()
        tracked_sessions = {lease.owner_session_id for lease in active_leases}
        for lease in active_leases:
            if exclude_lease_id and lease.lease_id == exclude_lease_id:
                continue
            if lease.overlaps(normalized_globs, normalized_paths):
                conflicts.append(
                    {
                        "lease_id": lease.lease_id,
                        "task_id": lease.task_id,
                        "title": lease.title,
                        "owner_agent": lease.owner_agent,
                        "owner_session_id": lease.owner_session_id,
                        "branch": lease.branch,
                        "worktree_path": lease.worktree_path,
                        "allowed_globs": lease.allowed_globs,
                        "claimed_paths": lease.claimed_paths,
                        "expires_at": lease.expires_at,
                    }
                )
        for claim in self.fleet_store.list_claims():
            session_id = str(claim.get("session_id", "")).strip()
            if owner_session_id and session_id == owner_session_id:
                continue
            if session_id in tracked_sessions:
                continue
            path = _normalize_claim(str(claim.get("path", "")))
            if not path:
                continue
            if not _claims_overlap([path], normalized_globs, normalized_paths):
                continue
            conflicts.append(
                {
                    "source": "fleet_claim",
                    "session_id": session_id,
                    "branch": str(claim.get("branch", "")),
                    "path": path,
                    "mode": str(claim.get("mode", "exclusive")),
                }
            )
        return conflicts

    def claim_lease(
        self,
        *,
        task_id: str,
        title: str,
        owner_agent: str,
        owner_session_id: str,
        branch: str,
        worktree_path: str,
        allowed_globs: list[str] | None = None,
        claimed_paths: list[str] | None = None,
        expected_tests: list[str] | None = None,
        ttl_hours: float = 8.0,
        metadata: dict[str, Any] | None = None,
        allow_overlap: bool = False,
    ) -> WorkLease:
        normalized_globs = [
            _normalize_claim(item) for item in allowed_globs or [] if str(item).strip()
        ]
        normalized_paths = [
            _normalize_claim(item) for item in claimed_paths or [] if str(item).strip()
        ]
        self.reap_expired_leases()
        self.reap_stale_leases()
        self.fleet_store.reap_stale_claims()
        now = _utcnow()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conflicts = self._find_conflicting_leases_locked(
                conn,
                allowed_globs=normalized_globs,
                claimed_paths=normalized_paths,
                owner_session_id=owner_session_id,
            )
            if conflicts and not allow_overlap:
                raise LeaseConflictError(conflicts)

            lease = WorkLease(
                lease_id=str(uuid.uuid4())[:12],
                task_id=task_id,
                title=title or task_id,
                owner_agent=owner_agent,
                owner_session_id=owner_session_id,
                branch=branch,
                worktree_path=str(Path(worktree_path).resolve()),
                allowed_globs=normalized_globs,
                claimed_paths=normalized_paths,
                expected_tests=list(expected_tests or []),
                status=LeaseStatus.ACTIVE.value,
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
                expires_at=(now + timedelta(hours=ttl_hours)).isoformat(),
                metadata=dict(metadata or {}),
            )
            conn.execute(
                "INSERT INTO leases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    lease.lease_id,
                    lease.task_id,
                    lease.title,
                    lease.owner_agent,
                    lease.owner_session_id,
                    lease.branch,
                    lease.worktree_path,
                    _json_dump(lease.allowed_globs),
                    _json_dump(lease.claimed_paths),
                    _json_dump(lease.expected_tests),
                    lease.status,
                    lease.created_at,
                    lease.updated_at,
                    lease.expires_at,
                    _json_dump(lease.metadata),
                ),
            )
            conn.commit()
        except LeaseConflictError:
            conn.rollback()
            self._publish(
                "conflict_detected",
                track=branch,
                data={
                    "task_id": task_id,
                    "worktree_path": worktree_path,
                    "conflicts": conflicts,
                },
            )
            raise
        finally:
            conn.close()

        self._publish(
            "task_claimed",
            track=branch,
            data={
                "lease_id": lease.lease_id,
                "task_id": task_id,
                "title": lease.title,
                "files": lease.claimed_paths or lease.allowed_globs,
                "expected_tests": lease.expected_tests,
                "worktree_path": lease.worktree_path,
            },
        )
        claim_paths = self._fleet_claim_paths(lease)
        if claim_paths:
            self.fleet_store.claim_paths(
                session_id=lease.owner_session_id,
                paths=claim_paths,
                branch=lease.branch,
                mode="exclusive",
            )
        self._sync_supervisor_run_from_lease(
            lease.metadata,
            update={
                "status": "leased",
                "lease_id": lease.lease_id,
                "owner_session_id": lease.owner_session_id,
                "branch": lease.branch,
                "worktree_path": lease.worktree_path,
                "target_agent": lease.owner_agent,
                "expected_tests": list(lease.expected_tests),
            },
        )
        return lease

    def _find_conflicting_leases_locked(
        self,
        conn: sqlite3.Connection,
        *,
        allowed_globs: list[str],
        claimed_paths: list[str],
        owner_session_id: str | None = None,
        exclude_lease_id: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_globs = [_normalize_claim(item) for item in allowed_globs if str(item).strip()]
        normalized_paths = [_normalize_claim(item) for item in claimed_paths if str(item).strip()]
        conflicts: list[dict[str, Any]] = []
        now = _utcnow()
        rows = conn.execute("SELECT * FROM leases ORDER BY created_at ASC").fetchall()
        active_leases = [
            lease
            for lease in (WorkLease.from_row(row) for row in rows)
            if lease.status == LeaseStatus.ACTIVE.value and _parse_dt(lease.expires_at) > now
        ]
        tracked_sessions = {lease.owner_session_id for lease in active_leases}
        for lease in active_leases:
            if exclude_lease_id and lease.lease_id == exclude_lease_id:
                continue
            if lease.overlaps(normalized_globs, normalized_paths):
                conflicts.append(
                    {
                        "lease_id": lease.lease_id,
                        "task_id": lease.task_id,
                        "title": lease.title,
                        "owner_agent": lease.owner_agent,
                        "owner_session_id": lease.owner_session_id,
                        "branch": lease.branch,
                        "worktree_path": lease.worktree_path,
                        "allowed_globs": lease.allowed_globs,
                        "claimed_paths": lease.claimed_paths,
                        "expires_at": lease.expires_at,
                    }
                )
        for claim in self.fleet_store.list_claims():
            session_id = str(claim.get("session_id", "")).strip()
            if owner_session_id and session_id == owner_session_id:
                continue
            if session_id in tracked_sessions:
                continue
            path = _normalize_claim(str(claim.get("path", "")))
            if not path:
                continue
            if not _claims_overlap([path], normalized_globs, normalized_paths):
                continue
            conflicts.append(
                {
                    "source": "fleet_claim",
                    "session_id": session_id,
                    "branch": str(claim.get("branch", "")),
                    "path": path,
                    "mode": str(claim.get("mode", "exclusive")),
                }
            )
        return conflicts

    def heartbeat_lease(self, lease_id: str, ttl_hours: float | None = None) -> WorkLease:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM leases WHERE lease_id = ?", (lease_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown lease_id: {lease_id}")
            lease = WorkLease.from_row(row)
            ttl = (
                ttl_hours
                if ttl_hours is not None
                else max(
                    1.0,
                    (_parse_dt(lease.expires_at) - _parse_dt(lease.updated_at)).total_seconds()
                    / 3600,
                )
            )
            now = _utcnow()
            conn.execute(
                "UPDATE leases SET updated_at = ?, expires_at = ? WHERE lease_id = ?",
                (now.isoformat(), (now + timedelta(hours=ttl)).isoformat(), lease_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM leases WHERE lease_id = ?", (lease_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        lease = WorkLease.from_row(row)
        claim_paths = self._fleet_claim_paths(lease)
        if claim_paths:
            self.fleet_store.claim_paths(
                session_id=lease.owner_session_id,
                paths=claim_paths,
                branch=lease.branch,
                mode="exclusive",
            )
        return lease

    def persist_scope_violation(
        self,
        lease_id: str,
        *,
        changed_paths: list[str],
        violations: list[dict[str, Any]],
    ) -> None:
        """Write scope-violation metadata into a lease without releasing it.

        The lease remains active so that ``status_summary()`` — which scans
        ``list_active_leases()`` for ``last_scope_violation`` — can surface the
        violation to fleet/integrator views.  This mirrors the metadata write in
        ``record_completion()`` but is callable from the supervisor's early-kill
        path where a full completion receipt is not available.
        """
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT metadata_json FROM leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if row is None:
                return  # Unknown lease — nothing to update
            metadata = {
                **_json_loads(row["metadata_json"], {}),
                "last_scope_violation": {
                    "detected_at": now,
                    "changed_paths": changed_paths,
                    "violations": violations,
                },
            }
            conn.execute(
                "UPDATE leases SET updated_at = ?, metadata_json = ? WHERE lease_id = ?",
                (now, _json_dump(metadata), lease_id),
            )
            conn.commit()
        finally:
            conn.close()
        self._publish(
            "scope_violation_detected",
            track="",
            data={
                "lease_id": lease_id,
                "changed_paths": changed_paths,
                "violations": violations,
            },
        )

    def update_lease_metadata(self, lease_id: str, updates: dict[str, Any]) -> None:
        """Merge *updates* into a lease's metadata JSON."""
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT metadata_json FROM leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if row is None:
                return
            metadata = {**_json_loads(row["metadata_json"], {}), **updates}
            conn.execute(
                "UPDATE leases SET updated_at = ?, metadata_json = ? WHERE lease_id = ?",
                (now, _json_dump(metadata), lease_id),
            )
            conn.commit()
        finally:
            conn.close()

    def release_lease(self, lease_id: str, status: LeaseStatus = LeaseStatus.RELEASED) -> WorkLease:
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE leases SET status = ?, updated_at = ? WHERE lease_id = ?",
                (status.value, now, lease_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM leases WHERE lease_id = ?", (lease_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        lease = WorkLease.from_row(row)
        self._release_fleet_claims_for_lease(lease)
        self._publish(
            "task_completed",
            track=lease.branch,
            data={
                "lease_id": lease.lease_id,
                "status": lease.status,
                "worktree_path": lease.worktree_path,
            },
        )
        return lease

    def record_completion(
        self,
        *,
        lease_id: str,
        owner_agent: str,
        owner_session_id: str,
        branch: str,
        worktree_path: str,
        base_sha: str | None = None,
        head_sha: str | None = None,
        commit_shas: list[str] | None = None,
        changed_paths: list[str] | None = None,
        tests_run: list[str] | None = None,
        validations_run: list[str] | None = None,
        assumptions: list[str] | None = None,
        blockers: list[str] | None = None,
        outcome: str = "completed",
        risks: list[str] | None = None,
        pr_url: str | None = None,
        pr_number: int | None = None,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
        require_session_ownership: bool = True,
    ) -> CompletionReceipt:
        normalized_changed_paths = [
            _normalize_claim(item) for item in changed_paths or [] if str(item).strip()
        ]
        now = _utcnow().isoformat()
        conn = self._connect()
        try:
            lease_row = conn.execute(
                "SELECT * FROM leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if lease_row is None:
                raise KeyError(f"Unknown lease_id: {lease_id}")
            lease = WorkLease.from_row(lease_row)
            lease_metadata = _json_loads(lease_row["metadata_json"], {})
            receipt_metadata = {
                **dict(metadata or {}),
                "supervisor_run_id": lease_metadata.get("supervisor_run_id"),
                "work_order_id": lease_metadata.get("work_order_id"),
                "task_key": lease_metadata.get("task_key"),
                "reviewer_agent": lease_metadata.get("reviewer_agent"),
                "risk_level": lease_metadata.get("risk_level"),
            }
            if (str(pr_url or "").strip() or pr_number is not None) and not str(
                receipt_metadata.get("pr_created_at", "")
            ).strip():
                receipt_metadata["pr_created_at"] = now
            normalized_outcome = _normalize_completion_outcome(
                outcome=str(outcome or "completed").strip() or "completed",
                commit_shas=list(commit_shas or []),
                changed_paths=normalized_changed_paths,
                pr_url=str(pr_url or "").strip(),
                pr_number=pr_number,
            )
            receipt = CompletionReceipt(
                receipt_id=str(uuid.uuid4())[:12],
                lease_id=lease_id,
                task_id=lease.task_id,
                owner_agent=owner_agent,
                owner_session_id=owner_session_id,
                branch=branch,
                worktree_path=str(Path(worktree_path).resolve()),
                base_sha=str(base_sha or lease_metadata.get("base_sha") or "").strip(),
                head_sha=str(head_sha or "").strip(),
                commit_shas=list(commit_shas or []),
                changed_paths=normalized_changed_paths,
                tests_run=list(tests_run or []),
                validations_run=list(validations_run or tests_run or []),
                assumptions=list(assumptions or []),
                blockers=list(blockers or []),
                outcome=normalized_outcome,
                risks=list(risks or blockers or []),
                pr_url=str(pr_url or "").strip(),
                pr_number=pr_number,
                confidence=float(confidence),
                created_at=now,
                metadata=receipt_metadata,
            )
            violations = self._validate_completion_scope(
                lease,
                changed_paths=receipt.changed_paths,
                owner_session_id=owner_session_id,
                branch=branch,
                require_session_ownership=require_session_ownership,
            )
            if violations:
                metadata = {
                    **_json_loads(lease_row["metadata_json"], {}),
                    "last_scope_violation": {
                        "detected_at": now,
                        "changed_paths": list(receipt.changed_paths),
                        "violations": violations,
                    },
                }
                conn.execute(
                    "UPDATE leases SET updated_at = ?, metadata_json = ? WHERE lease_id = ?",
                    (now, _json_dump(metadata), lease_id),
                )
                conn.commit()
                self._publish(
                    "scope_violation_detected",
                    track=branch,
                    data={
                        "lease_id": lease_id,
                        "owner_session_id": owner_session_id,
                        "changed_paths": receipt.changed_paths,
                        "violations": violations,
                    },
                )
                raise FileScopeViolationError(violations)
            conn.execute(
                """
                INSERT INTO completion_receipts (
                    receipt_id, lease_id, owner_agent, owner_session_id, branch, worktree_path,
                    commit_shas_json, changed_paths_json, tests_run_json, assumptions_json,
                    blockers_json, confidence, created_at, artifact_hash, task_id, base_sha,
                    head_sha, validations_run_json, outcome, risks_json, pr_url, pr_number,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id,
                    receipt.lease_id,
                    receipt.owner_agent,
                    receipt.owner_session_id,
                    receipt.branch,
                    receipt.worktree_path,
                    _json_dump(receipt.commit_shas),
                    _json_dump(receipt.changed_paths),
                    _json_dump(receipt.tests_run),
                    _json_dump(receipt.assumptions),
                    _json_dump(receipt.blockers),
                    receipt.confidence,
                    receipt.created_at,
                    receipt.artifact_hash,
                    receipt.task_id,
                    receipt.base_sha,
                    receipt.head_sha,
                    _json_dump(receipt.validations_run),
                    receipt.outcome,
                    _json_dump(receipt.risks),
                    receipt.pr_url,
                    receipt.pr_number,
                    _json_dump(receipt.metadata),
                ),
            )
            conn.execute(
                "UPDATE leases SET status = ?, updated_at = ?, metadata_json = ? WHERE lease_id = ?",
                (
                    LeaseStatus.COMPLETED.value,
                    now,
                    _json_dump(
                        {
                            **lease_metadata,
                            "last_receipt_id": receipt.receipt_id,
                        }
                    ),
                    lease_id,
                ),
            )
            pending = IntegrationDecision(
                decision_id=str(uuid.uuid4())[:12],
                lease_id=lease_id,
                receipt_id=receipt.receipt_id,
                decision=IntegrationDecisionType.PENDING_REVIEW.value,
                target_branch="main",
                rationale="Awaiting integrator review",
                chosen_commits=list(receipt.commit_shas),
                followups=[],
                decided_by="system",
                created_at=now,
            )
            conn.execute(
                "INSERT INTO integration_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pending.decision_id,
                    pending.lease_id,
                    pending.receipt_id,
                    pending.decision,
                    pending.target_branch,
                    pending.rationale,
                    _json_dump(pending.chosen_commits),
                    _json_dump(pending.followups),
                    pending.decided_by,
                    pending.created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        self._release_fleet_claims_for_lease(lease)

        self._publish(
            "task_completed",
            track=branch,
            data={
                "lease_id": lease_id,
                "receipt_id": receipt.receipt_id,
                "files": receipt.changed_paths,
                "tests_run": receipt.tests_run,
                "validations_run": receipt.validations_run,
                "confidence": receipt.confidence,
                "task_id": receipt.task_id,
                "base_sha": receipt.base_sha,
                "head_sha": receipt.head_sha,
                "outcome": receipt.outcome,
                "risks": receipt.risks,
                "pr_url": receipt.pr_url or None,
                "pr_number": receipt.pr_number,
                "pr_created_at": receipt.metadata.get("pr_created_at"),
                "metadata": dict(receipt.metadata),
            },
        )
        self._publish(
            "merge_ready",
            track=branch,
            data={
                "lease_id": lease_id,
                "receipt_id": receipt.receipt_id,
                "commit_shas": receipt.commit_shas,
                "artifact_hash": receipt.artifact_hash,
                "task_id": receipt.task_id,
            },
        )
        self.fleet_store.enqueue_merge(
            session_id=owner_session_id,
            branch=branch,
            title=f"{owner_agent}: {lease_id}",
            metadata={
                "lease_id": lease_id,
                "receipt_id": receipt.receipt_id,
                "task_id": receipt.task_id,
                "tests_run": receipt.tests_run,
                "validations_run": receipt.validations_run,
                "changed_paths": receipt.changed_paths,
                "confidence": receipt.confidence,
                "artifact_hash": receipt.artifact_hash,
                "base_sha": receipt.base_sha,
                "head_sha": receipt.head_sha,
                "outcome": receipt.outcome,
                "risks": receipt.risks,
                "pr_url": receipt.pr_url or None,
                "pr_number": receipt.pr_number,
                "pr_created_at": receipt.metadata.get("pr_created_at"),
            },
        )
        self._sync_supervisor_run_from_lease(
            lease.metadata,
            update={
                "status": "completed",
                "receipt_id": receipt.receipt_id,
                "changed_paths": list(receipt.changed_paths),
                "tests_run": list(receipt.tests_run),
                "confidence": receipt.confidence,
                "review_status": "pending_heterogeneous_review",
            },
        )
        return receipt

    def record_integration_decision(
        self,
        *,
        receipt_id: str,
        decision: IntegrationDecisionType,
        decided_by: str,
        rationale: str,
        target_branch: str = "main",
        chosen_commits: list[str] | None = None,
        followups: list[str] | None = None,
        lease_id: str | None = None,
    ) -> IntegrationDecision:
        conn = self._connect()
        try:
            latest = conn.execute(
                "SELECT * FROM integration_decisions WHERE receipt_id = ? ORDER BY created_at DESC LIMIT 1",
                (receipt_id,),
            ).fetchone()
            if latest is None and lease_id is None:
                receipt_row = conn.execute(
                    "SELECT * FROM completion_receipts WHERE receipt_id = ?",
                    (receipt_id,),
                ).fetchone()
                if receipt_row is None:
                    raise KeyError(f"Unknown receipt_id: {receipt_id}")
                lease_id = receipt_row["lease_id"]
            decision_row = IntegrationDecision(
                decision_id=str(uuid.uuid4())[:12],
                lease_id=lease_id or latest["lease_id"],
                receipt_id=receipt_id,
                decision=decision.value,
                target_branch=target_branch,
                rationale=rationale,
                chosen_commits=list(
                    chosen_commits
                    or (_json_loads(latest["chosen_commits_json"], []) if latest else [])
                ),
                followups=list(followups or []),
                decided_by=decided_by,
                created_at=_utcnow().isoformat(),
            )
            conn.execute(
                "INSERT INTO integration_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_row.decision_id,
                    decision_row.lease_id,
                    decision_row.receipt_id,
                    decision_row.decision,
                    decision_row.target_branch,
                    decision_row.rationale,
                    _json_dump(decision_row.chosen_commits),
                    _json_dump(decision_row.followups),
                    decision_row.decided_by,
                    decision_row.created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        event_type = (
            "merge_completed"
            if decision in {IntegrationDecisionType.MERGE, IntegrationDecisionType.CHERRY_PICK}
            else "conflict_detected"
        )
        self._publish(
            event_type,
            track=decision_row.target_branch,
            data=decision_row.to_dict(),
        )
        queue_item = self._find_fleet_queue_item(receipt_id=receipt_id)
        if queue_item is not None:
            queue_status = {
                IntegrationDecisionType.MERGE: "integrating",
                IntegrationDecisionType.CHERRY_PICK: "integrating",
                IntegrationDecisionType.REQUEST_CHANGES: "needs_human",
                IntegrationDecisionType.DISCARD: "blocked",
                IntegrationDecisionType.SALVAGE: "blocked",
            }.get(decision)
            if queue_status:
                self.fleet_store.update_merge_queue_item(
                    item_id=str(queue_item.get("id", "")),
                    status=queue_status,
                    metadata={
                        "integration_decision_id": decision_row.decision_id,
                        "integration_decision": decision_row.decision,
                        "chosen_commits": decision_row.chosen_commits,
                        "followups": decision_row.followups,
                    },
                )
        conn = self._connect()
        try:
            lease_row = conn.execute(
                "SELECT metadata_json FROM leases WHERE lease_id = ?",
                (decision_row.lease_id,),
            ).fetchone()
        finally:
            conn.close()
        lease_metadata = _json_loads(lease_row["metadata_json"], {}) if lease_row else {}
        self._sync_supervisor_run_from_lease(
            lease_metadata,
            update={
                "integration_decision": decision_row.decision,
                "integration_decision_id": decision_row.decision_id,
                "integration_followups": list(decision_row.followups),
                "status": {
                    IntegrationDecisionType.PENDING_REVIEW.value: "needs_human",
                    IntegrationDecisionType.MERGE.value: "integrating",
                    IntegrationDecisionType.CHERRY_PICK.value: "integrating",
                    IntegrationDecisionType.REQUEST_CHANGES.value: "changes_requested",
                    IntegrationDecisionType.DISCARD.value: "discarded",
                    IntegrationDecisionType.SALVAGE.value: "salvage",
                }.get(decision_row.decision, "needs_human"),
            },
        )
        return decision_row

    def upsert_salvage_candidate(
        self,
        *,
        source_kind: str,
        source_ref: str,
        branch: str = "",
        worktree_path: str = "",
        stash_ref: str = "",
        head_sha: str = "",
        changed_paths: list[str] | None = None,
        summary: str = "",
        likely_value: float = 0.0,
        status: SalvageStatus = SalvageStatus.DETECTED,
        metadata: dict[str, Any] | None = None,
    ) -> SalvageCandidate:
        now = _utcnow().isoformat()
        candidate_id = hashlib.sha1(
            f"{source_kind}:{source_ref}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]
        candidate = SalvageCandidate(
            candidate_id=candidate_id,
            source_kind=source_kind,
            source_ref=source_ref,
            branch=branch,
            worktree_path=str(Path(worktree_path).resolve()) if worktree_path else "",
            stash_ref=stash_ref,
            head_sha=head_sha,
            changed_paths=[
                _normalize_claim(item) for item in changed_paths or [] if str(item).strip()
            ],
            summary=summary,
            likely_value=float(likely_value),
            status=status.value,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO salvage_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_kind, source_ref) DO UPDATE SET
                    branch = excluded.branch,
                    worktree_path = excluded.worktree_path,
                    stash_ref = excluded.stash_ref,
                    head_sha = excluded.head_sha,
                    changed_paths_json = excluded.changed_paths_json,
                    summary = excluded.summary,
                    likely_value = excluded.likely_value,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    candidate.candidate_id,
                    candidate.source_kind,
                    candidate.source_ref,
                    candidate.branch,
                    candidate.worktree_path,
                    candidate.stash_ref,
                    candidate.head_sha,
                    _json_dump(candidate.changed_paths),
                    candidate.summary,
                    candidate.likely_value,
                    candidate.status,
                    candidate.created_at,
                    candidate.updated_at,
                    _json_dump(candidate.metadata),
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM salvage_candidates WHERE source_kind = ? AND source_ref = ?",
                (source_kind, source_ref),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise RuntimeError("Failed to persist salvage candidate")
        return SalvageCandidate.from_row(row)

    def pending_work_items(self) -> list[WorkItem]:
        items: list[WorkItem] = []
        items.extend(
            item.to_work_item() for item in self.list_integration_decisions(only_pending=True)
        )
        items.extend(
            item.to_work_item()
            for item in self.list_salvage_candidates(statuses=sorted(_OPEN_SALVAGE_STATUSES))
        )
        return items

    def developer_task_work_items(self) -> list[WorkItem]:
        items: list[WorkItem] = []
        for task in self.list_developer_tasks(open_only=True):
            if task.status not in _QUEUEABLE_DEVELOPER_TASK_STATUSES:
                continue
            items.append(task.to_work_item())
        return items

    async def sync_developer_task_queue(
        self,
        queue: GlobalWorkQueue | None = None,
        *,
        complete_missing: bool = True,
    ) -> dict[str, int]:
        """Project open developer tasks into the global work queue."""
        self.backfill_missing_completion_receipts()
        self.backfill_missing_blocker_metadata()
        self.backfill_missing_verification_plans()
        self.rehabilitate_docs_only_missing_verification_plan_work_orders()
        self.rehabilitate_deliverable_backed_clean_exit_no_deliverable_work_orders()
        self.reclassify_branch_snapshot_stale_review_work_orders()
        self.reclassify_deliverable_changes_requested_work_orders()
        self.archive_superseded_clean_exit_no_deliverable_work_orders()
        self.archive_superseded_stale_lease_reaped_work_orders()
        self.archive_reaped_no_receipt_work_orders()
        self.archive_scope_violation_no_deliverable_work_orders()
        self.archive_failed_no_deliverable_work_orders()
        self.archive_clean_exit_no_deliverable_work_orders()
        self.archive_work_order_leasing_failed_work_orders()
        self.archive_worker_type_blocked_work_orders()
        self.archive_duplicate_work_order_leasing_failed_work_orders()
        self.archive_duplicate_branch_deliverable_work_orders()
        self.archive_superseded_waiting_conflict_work_orders()
        self.archive_duplicate_waiting_conflict_work_orders()
        self.rehabilitate_narrowed_waiting_conflict_work_orders()
        work_queue = queue or GlobalWorkQueue(storage_dir=self.repo_root / ".work_queue")
        await work_queue.initialize()

        desired_items = {item.id: item for item in self.developer_task_work_items()}
        existing_items = {item.id: item for item in await work_queue.list_items(limit=10_000)}

        counts = {
            "created": 0,
            "updated": 0,
            "reopened": 0,
            "completed": 0,
            "skipped_active": 0,
            "open_items": len(desired_items),
        }

        for item_id, item in desired_items.items():
            existing = existing_items.get(item_id)
            if existing and existing.status in (WorkStatus.CLAIMED, WorkStatus.IN_PROGRESS):
                counts["skipped_active"] += 1
                continue
            await work_queue.upsert(item, allow_reopen=True, preserve_claimed=True)
            if existing is None:
                counts["created"] += 1
            elif existing.status in (WorkStatus.COMPLETED, WorkStatus.FAILED):
                counts["reopened"] += 1
            else:
                counts["updated"] += 1

        if complete_missing:
            for item_id, existing in existing_items.items():
                if not item_id.startswith("task:") or item_id in desired_items:
                    continue
                if existing.status in (
                    WorkStatus.CLAIMED,
                    WorkStatus.IN_PROGRESS,
                    WorkStatus.COMPLETED,
                    WorkStatus.FAILED,
                ):
                    continue
                completed = await work_queue.complete(
                    item_id,
                    result={"source": "dev_coordination", "reason": "task_no_longer_open"},
                )
                if completed is not None:
                    counts["completed"] += 1

        await work_queue.reprioritize()
        return counts

    async def sync_pending_work_queue(
        self,
        queue: GlobalWorkQueue | None = None,
        *,
        complete_missing: bool = True,
    ) -> dict[str, int]:
        """Project pending integration/salvage items into the global work queue."""
        work_queue = queue or GlobalWorkQueue(storage_dir=self.repo_root / ".work_queue")
        await work_queue.initialize()

        desired_items = {item.id: item for item in self.pending_work_items()}
        existing_items = {item.id: item for item in await work_queue.list_items(limit=10_000)}
        managed_prefixes = ("integration:", "salvage:")

        counts = {
            "created": 0,
            "updated": 0,
            "reopened": 0,
            "completed": 0,
            "skipped_active": 0,
            "open_items": len(desired_items),
        }

        for item_id, item in desired_items.items():
            existing = existing_items.get(item_id)
            if existing and existing.status in (WorkStatus.CLAIMED, WorkStatus.IN_PROGRESS):
                counts["skipped_active"] += 1
                continue

            await work_queue.upsert(item, allow_reopen=True, preserve_claimed=True)
            if existing is None:
                counts["created"] += 1
            elif existing.status in (WorkStatus.COMPLETED, WorkStatus.FAILED):
                counts["reopened"] += 1
            else:
                counts["updated"] += 1

        if complete_missing:
            for item_id, existing in existing_items.items():
                if not item_id.startswith(managed_prefixes) or item_id in desired_items:
                    continue
                if existing.status in (
                    WorkStatus.CLAIMED,
                    WorkStatus.IN_PROGRESS,
                    WorkStatus.COMPLETED,
                    WorkStatus.FAILED,
                ):
                    continue
                completed = await work_queue.complete(
                    item_id,
                    result={"source": "dev_coordination", "reason": "no_longer_pending"},
                )
                if completed is not None:
                    counts["completed"] += 1

        await work_queue.reprioritize()
        return counts

    def scan_salvage_sources(
        self,
        *,
        include_worktrees: bool = True,
        include_stashes: bool = True,
        max_stashes: int = 25,
    ) -> list[SalvageCandidate]:
        candidates: list[SalvageCandidate] = []
        if include_worktrees:
            candidates.extend(self._scan_worktrees())
        if include_stashes:
            candidates.extend(self._scan_stashes(max_stashes=max_stashes))
        return candidates

    def _scan_worktrees(self) -> list[SalvageCandidate]:
        proc = subprocess.run(
            ["git", "-C", str(self.repo_root), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return []

        candidates: list[SalvageCandidate] = []
        for path, branch in _parse_worktree_entries(proc.stdout):
            if not branch or branch in {"main", "master"}:
                continue
            dirty_proc = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            ahead_proc = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-list", "--count", f"origin/main..{branch}"],
                capture_output=True,
                text=True,
                check=False,
            )
            head_proc = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            ahead = int(ahead_proc.stdout.strip() or "0") if ahead_proc.returncode == 0 else 0
            status_lines = [line for line in dirty_proc.stdout.splitlines() if line.strip()]
            if not status_lines and ahead == 0:
                continue
            changed_paths = _status_paths(status_lines)
            if ahead > 0:
                diff_proc = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(self.repo_root),
                        "diff",
                        "--name-only",
                        f"origin/main...{branch}",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                changed_paths.extend(
                    _normalize_claim(item) for item in diff_proc.stdout.splitlines() if item.strip()
                )
            summary = f"worktree {branch} dirty={bool(status_lines)} ahead={ahead}"
            candidate = self.upsert_salvage_candidate(
                source_kind="worktree",
                source_ref=branch,
                branch=branch,
                worktree_path=str(path),
                head_sha=head_proc.stdout.strip() if head_proc.returncode == 0 else "",
                changed_paths=sorted(set(changed_paths)),
                summary=summary,
                likely_value=_estimate_salvage_value(
                    ahead=ahead, changed_paths=changed_paths, dirty=bool(status_lines)
                ),
                metadata={"ahead": ahead, "dirty": bool(status_lines)},
            )
            candidates.append(candidate)
        return candidates

    def _scan_stashes(self, *, max_stashes: int = 25) -> list[SalvageCandidate]:
        proc = subprocess.run(
            ["git", "-C", str(self.repo_root), "stash", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        candidates: list[SalvageCandidate] = []
        for line in proc.stdout.splitlines()[:max_stashes]:
            if not line.strip() or ":" not in line:
                continue
            source_ref, summary = line.split(":", 1)
            source_ref = source_ref.strip()
            summary = summary.strip()
            names_proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_root),
                    "stash",
                    "show",
                    "--name-only",
                    "--include-untracked",
                    source_ref,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            changed_paths = [
                _normalize_claim(item) for item in names_proc.stdout.splitlines() if item.strip()
            ]
            if not changed_paths:
                continue
            candidate = self.upsert_salvage_candidate(
                source_kind="stash",
                source_ref=source_ref,
                stash_ref=source_ref,
                changed_paths=changed_paths,
                summary=summary,
                likely_value=_estimate_salvage_value(
                    ahead=0, changed_paths=changed_paths, dirty=True
                ),
                metadata={"summary": summary},
            )
            candidates.append(candidate)
        return candidates

    def _publish(self, event_type: str, *, track: str, data: dict[str, Any]) -> None:
        try:
            self.event_bus.publish(event_type, track=track, data=data)
        except (OSError, ValueError, RuntimeError):
            # Coordination state must not fail closed if the event bus is unavailable.
            pass

    @staticmethod
    def _fleet_claim_paths(lease: WorkLease) -> list[str]:
        return lease.claimed_paths or lease.allowed_globs

    def _release_fleet_claims_for_lease(self, lease: WorkLease) -> None:
        release_paths = self._fleet_claim_paths(lease)
        if release_paths:
            self.fleet_store.release_paths(
                session_id=lease.owner_session_id,
                paths=release_paths,
            )

    def _find_fleet_queue_item(self, *, receipt_id: str) -> dict[str, Any] | None:
        for item in self.fleet_store.list_merge_queue():
            metadata = item.get("metadata")
            if isinstance(metadata, dict) and str(metadata.get("receipt_id", "")) == receipt_id:
                return item
        return None

    def _validate_completion_scope(
        self,
        lease: WorkLease,
        *,
        changed_paths: list[str],
        owner_session_id: str,
        branch: str,
        require_session_ownership: bool = True,
    ) -> list[dict[str, Any]]:
        scope_patterns = list(dict.fromkeys([*lease.claimed_paths, *lease.allowed_globs]))
        protected_patterns = list(
            dict.fromkeys(
                _normalize_claim(item)
                for key in ("forbidden_paths", "forbidden_globs", "hot_paths", "hot_globs")
                for item in lease.metadata.get(key, [])
                if str(item).strip()
            )
        )
        violations: list[dict[str, Any]] = []

        if changed_paths and not scope_patterns:
            return [
                {
                    "type": "undeclared_scope",
                    "message": "Lease has no declared file scope for the recorded changes.",
                    "paths": list(changed_paths),
                }
            ]

        for path in changed_paths:
            if scope_patterns and not any(
                _path_matches_glob(path, pattern) for pattern in scope_patterns
            ):
                violations.append(
                    {
                        "type": "out_of_scope",
                        "path": path,
                        "allowed_scope": list(scope_patterns),
                    }
                )
            if protected_patterns and any(
                _path_matches_glob(path, pattern) for pattern in protected_patterns
            ):
                violations.append(
                    {
                        "type": "protected_path",
                        "path": path,
                        "protected_scope": list(protected_patterns),
                    }
                )

        if require_session_ownership:
            audit = self.fleet_store.audit_session_paths(
                session_id=owner_session_id,
                paths=changed_paths,
                branch=branch,
            )
            for path in audit["unowned_paths"]:
                violations.append({"type": "unowned_path", "path": path})
            for conflict in audit["conflicts"]:
                violations.append(
                    {
                        "type": "conflicting_claim",
                        "path": str(conflict.get("path", "")),
                        "conflicting_path": str(conflict.get("conflicting_path", "")),
                        "session_id": str(conflict.get("session_id", "")),
                        "branch": str(conflict.get("branch", "")),
                        "mode": str(conflict.get("mode", "")),
                    }
                )
        return violations

    def mark_supervisor_run_merged(
        self,
        *,
        receipt_id: str,
        merge_commit_sha: str | None = None,
        merged_at: str | None = None,
    ) -> None:
        receipt = self.get_completion_receipt(receipt_id)
        if receipt is None:
            return
        conn = self._connect()
        try:
            lease_row = conn.execute(
                "SELECT metadata_json FROM leases WHERE lease_id = ?",
                (receipt.lease_id,),
            ).fetchone()
        finally:
            conn.close()
        lease_metadata = _json_loads(lease_row["metadata_json"], {}) if lease_row else {}
        merged_at_text = str(merged_at or _utcnow().isoformat()).strip() or _utcnow().isoformat()
        update = {"status": "merged", "merged_at": merged_at_text}
        if merge_commit_sha:
            update["merge_commit_sha"] = merge_commit_sha
        self._sync_supervisor_run_from_lease(lease_metadata, update=update)
        self._record_supervisor_merge_telemetry(
            lease_metadata,
            receipt=receipt,
            merge_commit_sha=merge_commit_sha,
            merged_at=merged_at_text,
        )

    def _record_supervisor_merge_telemetry(
        self,
        lease_metadata: dict[str, Any] | None,
        *,
        receipt: CompletionReceipt,
        merge_commit_sha: str | None,
        merged_at: str,
    ) -> None:
        from aragora.swarm.lane_telemetry import LaneTelemetryRecord

        if not isinstance(lease_metadata, dict):
            lease_metadata = {}
        run_id = str(lease_metadata.get("supervisor_run_id", "")).strip()
        work_order_id = str(lease_metadata.get("work_order_id", "")).strip()
        task_key = str(lease_metadata.get("task_key", "")).strip()
        lane_id = task_key or (
            f"{run_id}:{work_order_id}" if run_id and work_order_id else work_order_id or run_id
        )
        if not lane_id:
            return

        collector = _get_lane_telemetry()
        existing = collector.get_lane("supervisor_work_order", lane_id)
        deliverable_type = str(existing.deliverable_type if existing else "").strip()
        if not deliverable_type:
            if receipt.pr_url or receipt.pr_number is not None:
                deliverable_type = "pr"
            elif receipt.branch and receipt.commit_shas:
                deliverable_type = "branch"
        terminal_outcome = str(existing.terminal_outcome if existing else "").strip()
        if not terminal_outcome:
            terminal_outcome = str(receipt.outcome or "").strip()
        if terminal_outcome == "completed":
            terminal_outcome = (
                "deliverable_created" if deliverable_type else "clean_exit_no_deliverable"
            )
        if not terminal_outcome:
            if deliverable_type == "adopted_pr":
                terminal_outcome = "pr_adopted"
            elif deliverable_type:
                terminal_outcome = "deliverable_created"
            else:
                terminal_outcome = "unknown"

        time_to_merge_seconds = existing.time_to_merge_seconds if existing else None
        try:
            time_to_merge_seconds = max(
                0.0,
                (_parse_dt(merged_at) - _parse_dt(receipt.created_at)).total_seconds(),
            )
        except (TypeError, ValueError):
            pass
        time_to_pr_seconds = existing.time_to_pr_seconds if existing else None
        pr_created_at = str((receipt.metadata or {}).get("pr_created_at", "")).strip()
        try:
            if pr_created_at:
                time_to_pr_seconds = max(
                    0.0,
                    (_parse_dt(pr_created_at) - _parse_dt(receipt.created_at)).total_seconds(),
                )
        except (TypeError, ValueError):
            pass

        metadata = dict(existing.metadata if existing else {})
        metadata.update(
            {
                "status": "merged",
                "merge_commit_sha": merge_commit_sha or metadata.get("merge_commit_sha"),
                "merged_at": merged_at,
                "receipt_outcome": receipt.outcome or None,
            }
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="supervisor_work_order",
                lane_id=lane_id,
                run_id=run_id or (existing.run_id if existing else ""),
                task_id=(existing.task_id if existing else "") or task_key or work_order_id,
                work_order_id=work_order_id or (existing.work_order_id if existing else ""),
                terminal_outcome=terminal_outcome,
                worker_outcome=(existing.worker_outcome if existing else "") or "",
                deliverable_type=deliverable_type,
                receipt_id=receipt.receipt_id or (existing.receipt_id if existing else ""),
                human_intervention_required=False,
                duration_seconds=existing.duration_seconds if existing else 0.0,
                pr_url=receipt.pr_url or (existing.pr_url if existing else ""),
                pr_number=receipt.pr_number
                if receipt.pr_number is not None
                else (existing.pr_number if existing else None),
                merge_ref=merge_commit_sha or (existing.merge_ref if existing else ""),
                merged_at=merged_at,
                time_to_pr_seconds=time_to_pr_seconds,
                time_to_merge_seconds=time_to_merge_seconds,
                false_success_candidate=False
                if deliverable_type
                else bool(existing.false_success_candidate if existing else False),
                timestamp=existing.timestamp if existing else _utcnow().timestamp(),
                metadata=metadata,
            )
        )

    @staticmethod
    def _supervisor_run_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "goal": row["goal"],
            "target_branch": row["target_branch"],
            "status": row["status"],
            "supervisor_agents": _json_loads(row["supervisor_agents_json"], {}),
            "approval_policy": _json_loads(row["approval_policy_json"], {}),
            "spec": _json_loads(row["spec_json"], {}),
            "work_orders": _json_loads(row["work_orders_json"], []),
            "metadata": _json_loads(row["metadata_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _developer_task_from_run(
        run: dict[str, Any],
        work_order: dict[str, Any],
    ) -> DeveloperTask:
        run_id = str(run.get("run_id", "")).strip()
        task_id = (
            str(work_order.get("work_order_id", "")).strip()
            or str(work_order.get("task_id", "")).strip()
            or "task"
        )
        status = str(work_order.get("status", "")).strip().lower() or "queued"
        metadata = dict(work_order.get("metadata") or {})
        lease_id = _optional_text(work_order.get("lease_id"))
        return DeveloperTask(
            task_key=f"{run_id}:{task_id}",
            task_id=task_id,
            run_id=run_id,
            goal=str(run.get("goal", "")).strip(),
            title=str(work_order.get("title", "")).strip() or task_id,
            status=status,
            priority=_developer_task_priority(work_order),
            owner_agent=_optional_text(
                work_order.get("target_agent"),
                metadata.get("owner_agent"),
            )
            or "",
            reviewer_agent=_optional_text(
                work_order.get("reviewer_agent"),
                metadata.get("reviewer_agent"),
            )
            or "",
            blocked_by=_developer_task_blockers(work_order),
            acceptance_checks=_developer_task_acceptance_checks(work_order),
            allowed_paths=[
                str(item).strip() for item in work_order.get("file_scope", []) if str(item).strip()
            ],
            lease_id=lease_id or None,
            owner_session_id=_optional_text(work_order.get("owner_session_id")) or None,
            branch=_optional_text(work_order.get("branch")) or None,
            worktree_path=_optional_text(work_order.get("worktree_path")) or None,
            receipt_id=_optional_text(work_order.get("receipt_id")) or None,
            created_at=str(run.get("created_at", _utcnow().isoformat())),
            updated_at=_developer_task_updated_at(work_order, run),
            metadata={
                **metadata,
                "base_sha": _optional_text(work_order.get("base_sha")) or None,
                "head_sha": _optional_text(work_order.get("head_sha")) or None,
                "commit_shas": [
                    str(item).strip()
                    for item in work_order.get("commit_shas", [])
                    if str(item).strip()
                ],
                "changed_paths": [
                    str(item).strip()
                    for item in work_order.get("changed_paths", [])
                    if str(item).strip()
                ],
                "changed_files": [
                    str(item).strip()
                    for item in work_order.get("changed_files", [])
                    if str(item).strip()
                ],
                "pr_url": _optional_text(work_order.get("pr_url")) or None,
                "pr_number": work_order.get("pr_number"),
                "adopted_pr": _optional_text(work_order.get("adopted_pr")) or None,
                "worker_outcome": _optional_text(work_order.get("worker_outcome")) or None,
                "failure_reason": _optional_text(work_order.get("failure_reason")) or None,
                "dispatch_error": _optional_text(work_order.get("dispatch_error")) or None,
                "blocking_question": _optional_text(work_order.get("blocking_question")) or None,
                "blocker": dict(work_order.get("blocker") or {})
                if isinstance(work_order.get("blocker"), dict)
                else None,
                "target_branch": str(run.get("target_branch", "")).strip() or None,
                "run_status": str(run.get("status", "")).strip() or None,
                "success_criteria": dict(work_order.get("success_criteria") or {}),
                "dependency_ids": [
                    str(item).strip()
                    for item in work_order.get("dependency_ids", [])
                    if str(item).strip()
                ],
                "risk_level": str(work_order.get("risk_level", "")).strip() or None,
                "estimated_complexity": str(work_order.get("estimated_complexity", "")).strip()
                or None,
                "approval_required": bool(work_order.get("approval_required", False)),
            },
        )

    @staticmethod
    def _derive_supervisor_run_status(work_orders: list[dict[str, Any]]) -> str:
        statuses = {str(item.get("status", "")).strip() for item in work_orders if item}
        if not statuses:
            return "planned"
        terminal = {
            "merged",
            "discarded",
            "salvage",
            "completed",
            "failed",
            "timed_out",
            "scope_violation",
        }
        if statuses <= terminal:
            return "completed"
        if (
            "changes_requested" in statuses
            or "needs_human" in statuses
            or "dispatch_failed" in statuses
        ):
            return "needs_human"
        forward_progress = {"queued", "leased", "dispatched"}
        non_terminal = statuses - terminal
        if non_terminal and not (non_terminal & forward_progress):
            return "needs_human"
        return "active"

    def _persist_supervisor_run(
        self,
        conn: sqlite3.Connection,
        record: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            UPDATE supervisor_runs
            SET goal = ?, target_branch = ?, status = ?, supervisor_agents_json = ?,
                approval_policy_json = ?, spec_json = ?, work_orders_json = ?,
                metadata_json = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (
                record["goal"],
                record["target_branch"],
                record["status"],
                _json_dump(record.get("supervisor_agents", {})),
                _json_dump(record.get("approval_policy", {})),
                _json_dump(record.get("spec", {})),
                _json_dump(record.get("work_orders", [])),
                _json_dump(record.get("metadata", {})),
                record["updated_at"],
                record["run_id"],
            ),
        )

    def _sync_supervisor_run_from_lease(
        self,
        lease_metadata: dict[str, Any] | None,
        *,
        update: dict[str, Any],
    ) -> None:
        if not lease_metadata:
            return
        run_id = str(lease_metadata.get("supervisor_run_id", "")).strip()
        work_order_id = str(lease_metadata.get("work_order_id", "")).strip()
        if not run_id or not work_order_id:
            return
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM supervisor_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return
            record = self._supervisor_run_from_row(row)
            changed = False
            for item in record["work_orders"]:
                if str(item.get("work_order_id", "")).strip() != work_order_id:
                    continue
                reaped_failure = _optional_text(update.get("failure_reason"))
                if reaped_failure in {
                    "stale_lease_reaped",
                    "expired_lease_reaped",
                } and _work_order_has_concrete_deliverable(item):
                    preserved_update = {
                        key: value
                        for key, value in update.items()
                        if key not in {"status", "failure_reason", "blocking_question", "blocker"}
                    }
                    if not preserved_update:
                        break
                    item.update(preserved_update)
                    changed = True
                    break
                item.update(update)
                changed = True
                break
            if not changed:
                return
            record["status"] = self._derive_supervisor_run_status(record["work_orders"])
            record["updated_at"] = _utcnow().isoformat()
            self._persist_supervisor_run(conn, record)
            conn.commit()
        finally:
            conn.close()


def _optional_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _work_order_has_concrete_deliverable(work_order: dict[str, Any]) -> bool:
    receipt_id = _optional_text(work_order.get("receipt_id"))
    pr_url = _optional_text(work_order.get("pr_url"))
    adopted_pr = _optional_text(work_order.get("adopted_pr"))
    branch = _optional_text(work_order.get("branch"))
    commit_shas = [
        str(item).strip() for item in work_order.get("commit_shas", []) if str(item).strip()
    ]
    return bool(receipt_id or pr_url or adopted_pr or (branch and commit_shas))


def _flatten_acceptance_value(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        items: list[str] = []
        for key, nested in value.items():
            for text in _flatten_acceptance_value(nested):
                items.append(f"{key}: {text}")
        return items
    if isinstance(value, list):
        items: list[str] = []
        for nested in value:
            items.extend(_flatten_acceptance_value(nested))
        return items
    text = str(value or "").strip()
    return [text] if text else []


def _work_order_identifier(work_order: dict[str, Any]) -> str:
    return (
        str(work_order.get("work_order_id", "")).strip()
        or str(work_order.get("task_id", "")).strip()
        or str(work_order.get("id", "")).strip()
    )


def _find_work_order(record: dict[str, Any], work_order_id: str) -> dict[str, Any] | None:
    target = str(work_order_id).strip()
    if not target:
        return None
    for item in record.get("work_orders", []):
        if not isinstance(item, dict):
            continue
        if _work_order_identifier(item) == target:
            return item
    return None


def _developer_task_acceptance_checks(work_order: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    for entry in work_order.get("expected_tests", []):
        text = str(entry).strip()
        if text and text not in checks:
            checks.append(text)
    for entry in _flatten_acceptance_value(work_order.get("success_criteria", {})):
        if entry not in checks:
            checks.append(entry)
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict):
        for entry in metadata.get("acceptance_criteria", []):
            text = str(entry).strip()
            if text and text not in checks:
                checks.append(text)
    return checks


def _extract_tests_value(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _inferred_expected_tests_for_work_order(work_order: dict[str, Any]) -> list[str]:
    inferred: list[str] = []
    seen: set[str] = set()
    deferred_overbroad: list[str] = []
    deferred_seen: set[str] = set()

    def _append(command: str) -> None:
        normalized = str(command).strip()
        canonical = _canonical_verification_command(normalized)
        dedupe_key = canonical or normalized
        if not normalized or dedupe_key in seen:
            return
        if _is_overbroad_pytest_command(normalized):
            if dedupe_key in deferred_seen:
                return
            deferred_seen.add(dedupe_key)
            deferred_overbroad.append(normalized)
            return
        seen.add(dedupe_key)
        inferred.append(normalized)

    for entry in work_order.get("expected_tests", []):
        _append(str(entry))

    success_criteria = work_order.get("success_criteria")
    if isinstance(success_criteria, dict):
        for entry in _extract_tests_value(success_criteria.get("tests")):
            _append(entry)

    metadata = work_order.get("metadata")
    if isinstance(metadata, dict):
        for entry in metadata.get("acceptance_criteria", []):
            text = str(entry).strip()
            if text.startswith("python -m pytest") or text.startswith("pytest"):
                _append(text)
            for match in _TEST_FILE_PATTERN.findall(text):
                _append(f"python -m pytest {match} -q")

    for path in work_order.get("file_scope", []):
        normalized = str(path).strip()
        if normalized.startswith("tests/") and normalized.endswith(".py"):
            _append(f"python -m pytest {normalized} -q")

    for path in work_order.get("changed_paths", []):
        normalized = str(path).strip()
        if normalized.startswith("tests/") and normalized.endswith(".py"):
            _append(f"python -m pytest {normalized} -q")

    if not inferred:
        inferred.extend(deferred_overbroad)

    return inferred


def _merge_gate_state_for_work_order(work_order: dict[str, Any]) -> dict[str, Any]:
    expected_checks = _inferred_expected_tests_for_work_order(work_order)
    verification_results = [
        dict(entry)
        for entry in work_order.get("verification_results", [])
        if isinstance(entry, dict) and str(entry.get("command", "")).strip()
    ]
    missing_checks = [
        command
        for command in expected_checks
        if not any(
            _verification_command_covers_expected(entry.get("command", ""), command)
            for entry in verification_results
        )
    ]
    failed_checks = [
        dict(entry)
        for entry in verification_results
        if any(
            _verification_command_covers_expected(entry.get("command", ""), command)
            for command in expected_checks
        )
        and not bool(entry.get("passed", False))
    ]

    blocked_reasons: list[str] = []
    verification_missing_reason: str | None = None
    if not expected_checks:
        candidates = [
            str(path).strip()
            for path in work_order.get("changed_paths", []) or work_order.get("file_scope", [])
            if str(path).strip()
        ]
        if candidates and all(_is_docs_only_path(path) for path in candidates):
            return {
                "enabled": True,
                "expected_checks": expected_checks,
                "verification_results": verification_results,
                "verification_missing_reason": None,
                "checks_passed": True,
                "human_approval_required": True,
                "merge_eligible": True,
                "blocked_reasons": [],
            }
        verification_missing_reason = "missing_verification_plan"
        blocked_reasons.append(
            "merge gate blocked: missing verification plan or verification command"
        )
    if missing_checks:
        blocked_reasons.append(
            "merge gate blocked: required verification did not run: "
            + ", ".join(missing_checks[:3])
        )
    if failed_checks:
        first = failed_checks[0]
        reason = (
            "merge gate blocked: verification failed: "
            f"{first.get('command', '')} (exit {first.get('exit_code', -1)})"
        )
        stderr = str(first.get("stderr", "")).strip()
        if stderr:
            reason = f"{reason} - {stderr.splitlines()[0][:200]}"
        blocked_reasons.append(reason)

    checks_passed = not blocked_reasons
    return {
        "enabled": True,
        "expected_checks": expected_checks,
        "verification_results": verification_results,
        "verification_missing_reason": verification_missing_reason,
        "checks_passed": checks_passed,
        "human_approval_required": True,
        "merge_eligible": checks_passed,
        "blocked_reasons": blocked_reasons,
    }


def _canonical_verification_command(command: Any) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    for prefix in ("bash -lc ", "/bin/bash -lc "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
                text = text[1:-1]
            break
    from aragora.swarm.worker_launcher import WorkerLauncher

    text = re.sub(r"^(?P<prefix>\s*)python3(?=\s|$)", r"\g<prefix>python", text)
    return WorkerLauncher._normalize_verification_command(text).strip()


def _pytest_command_targets(command: Any) -> list[str]:
    text = _canonical_verification_command(command)
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return []
    if not tokens:
        return []
    start = 0
    if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
        start = 3
    elif tokens[0].endswith("pytest"):
        start = 1
    else:
        return []

    targets: list[str] = []
    skip_next = False
    options_with_values = {"-k", "-m", "--maxfail", "--timeout", "--tb", "-c", "--rootdir"}
    for token in tokens[start:]:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        normalized = _normalize_claim(token.rstrip("/"))
        if token.endswith("/") or "/" in normalized or normalized.endswith(".py"):
            targets.append(normalized)
    return targets


def _is_overbroad_pytest_command(command: Any) -> bool:
    targets = _pytest_command_targets(command)
    if not targets:
        return False
    return any(not target.endswith(".py") for target in targets)


def _verification_command_covers_expected(recorded_command: Any, expected_command: Any) -> bool:
    recorded = _canonical_verification_command(recorded_command)
    expected = _canonical_verification_command(expected_command)
    if not recorded or not expected:
        return False
    if recorded == expected:
        return True
    recorded_targets = _pytest_command_targets(recorded)
    expected_targets = _pytest_command_targets(expected)
    if not recorded_targets or not expected_targets:
        return False
    for expected_target in expected_targets:
        if not any(
            expected_target == recorded_target
            or expected_target.startswith(recorded_target.rstrip("/") + "/")
            for recorded_target in recorded_targets
        ):
            return False
    return True


def _verification_timeout_for_command(command: Any, default_timeout: float) -> float:
    canonical = _canonical_verification_command(command)
    if not canonical:
        return default_timeout
    targets = _pytest_command_targets(canonical)
    if len(targets) == 1 and targets[0].endswith(".py"):
        return max(default_timeout, 300.0)
    return default_timeout


def _developer_task_blockers(work_order: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for value in work_order.get("blockers", []):
        text = str(value).strip()
        if text and text not in blockers:
            blockers.append(text)
    for key in ("dispatch_error", "failure_reason"):
        text = str(work_order.get(key, "")).strip()
        if text and text not in blockers:
            blockers.append(text)
    if isinstance(work_order.get("scope_violation"), dict):
        blockers.append("scope_violation")
    return blockers


def _default_blocking_question_for_reason(reason_code: str) -> str:
    mapping = {
        "waiting_conflict": (
            "Which overlapping lane should finish, be discarded, or be split before this task can proceed?"
        ),
        "clean_exit_no_deliverable": (
            "What concrete branch, commit, or PR should this lane produce before rerunning?"
        ),
        "merge_gate_failed": (
            "Which required verification or acceptance check must pass before approval?"
        ),
        "branch_snapshot_stale": (
            "Should this deliverable be rebased, regenerated, or otherwise refreshed on current main before review?"
        ),
        "missing_verification_plan": (
            "Which verification command or acceptance check should be added before rerunning?"
        ),
        "verification_target_missing": (
            "Which current verification target should replace the missing path before rerunning?"
        ),
        "scope_violation": (
            "Which files should stay in scope, or should this lane be split before rerunning?"
        ),
        "worker_exited_without_receipt": (
            "Should this lane be rerun, or recovered manually from the existing worktree?"
        ),
        "worker_no_progress_timeout": (
            "Should this stalled lane be rerun, split, or investigated in its current worktree?"
        ),
        "worker_timeout_with_salvage": (
            "Should the recovered timed-out deliverable be adopted, amended, or rerun before integration?"
        ),
        "worker_timeout_no_deliverable": (
            "Should this timed-out lane be rerun, split, or investigated before retrying?"
        ),
        "worker_crash_with_salvage": (
            "Should the recovered crashed deliverable be adopted, amended, or rerun before integration?"
        ),
        "worker_crash": (
            "Should this crashed lane be rerun, reassigned, or investigated before retrying?"
        ),
        "worker_type_blocked": (
            "Which worker type or capacity issue must be resolved before rerunning this lane?"
        ),
        "work_order_leasing_failed": (
            "What missing environment, resource, or policy input must be resolved first?"
        ),
        "stale_lease_reaped": ("Should this stale lane be requeued, recovered, or discarded?"),
        "expired_lease_reaped": ("Should this expired lane be requeued, recovered, or discarded?"),
    }
    return mapping.get(reason_code, "What human input is required before rerunning this lane?")


def _merge_gate_verification_target_missing(work_order: dict[str, Any]) -> bool:
    haystacks = [_optional_text(work_order.get("dispatch_error")).lower()]
    for entry in work_order.get("verification_results", []):
        if not isinstance(entry, dict):
            continue
        haystacks.extend(
            _optional_text(entry.get(field)).lower()
            for field in ("stdout", "stderr")
            if _optional_text(entry.get(field))
        )
    combined = "\n".join(text for text in haystacks if text)
    return "file or directory not found:" in combined


def _infer_missing_failure_reason_for_work_order(work_order: dict[str, Any]) -> str:
    status = _optional_text(work_order.get("status")).lower()
    if status == "waiting_conflict":
        return "waiting_conflict"

    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and bool(metadata.get("mainline_verification_passed")):
        return "branch_snapshot_stale"

    merge_gate = work_order.get("merge_gate")
    if isinstance(merge_gate, dict):
        missing = _optional_text(merge_gate.get("verification_missing_reason")).lower()
        if missing:
            return missing

    worker_outcome = _optional_text(work_order.get("worker_outcome")).lower()
    if worker_outcome == "branch_snapshot_stale":
        return "branch_snapshot_stale"
    if worker_outcome == "clean_exit_no_effect":
        return "clean_exit_no_deliverable"
    if worker_outcome == "merge_gate_failed":
        dispatch_error = _optional_text(work_order.get("dispatch_error")).lower()
        if "missing verification plan" in dispatch_error:
            return "missing_verification_plan"
        if _merge_gate_verification_target_missing(work_order):
            return "verification_target_missing"
        return "merge_gate_failed"
    if worker_outcome == "scope_violation":
        return "scope_violation"
    if worker_outcome == "timeout_with_salvage":
        return "worker_timeout_with_salvage"
    if worker_outcome == "timeout_no_progress":
        return "worker_no_progress_timeout"
    if worker_outcome == "crash_with_salvage":
        return "worker_crash_with_salvage"

    lowered = _optional_text(work_order.get("dispatch_error")).lower()
    if "branch snapshot stale" in lowered or "passes on main" in lowered:
        return "branch_snapshot_stale"
    if "missing verification plan" in lowered:
        return "missing_verification_plan"
    if _merge_gate_verification_target_missing(work_order):
        return "verification_target_missing"
    if "merge gate" in lowered:
        return "merge_gate_failed"
    if "scope" in lowered and "ownership" in lowered:
        return "scope_violation"
    if "without receipt or exit marker" in lowered:
        return "worker_exited_without_receipt"
    if "no-progress timeout" in lowered:
        return "worker_no_progress_timeout"
    if "recoverable deliverable" in lowered and "timed out" in lowered:
        return "worker_timeout_with_salvage"
    if "recoverable deliverable" in lowered and "non-zero" in lowered:
        return "worker_crash_with_salvage"
    if "timed out before producing a deliverable" in lowered:
        return "worker_timeout_no_deliverable"
    if "crashed before producing a deliverable" in lowered:
        return "worker_crash"
    if "no commits and no changed paths" in lowered or "no real deliverables" in lowered:
        return "clean_exit_no_deliverable"
    if "dispatch blocked" in lowered:
        return "worker_type_blocked"
    if "autopilot ensure failed" in lowered or "lease" in lowered or "worktree" in lowered:
        return "work_order_leasing_failed"

    return _optional_text(work_order.get("failure_reason")) or "needs_human"


def _backfill_work_order_blocker_metadata(work_order: dict[str, Any]) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status not in {
        "waiting_conflict",
        "needs_human",
        "changes_requested",
        "dispatch_failed",
        "failed",
        "timed_out",
        "scope_violation",
    }:
        return False

    changed = False
    failure_reason = _optional_text(work_order.get("failure_reason"))
    inferred = _infer_missing_failure_reason_for_work_order(work_order)
    if inferred and (not failure_reason or failure_reason == "merge_gate_failed"):
        if failure_reason != inferred:
            work_order["failure_reason"] = inferred
            failure_reason = inferred
            changed = True
    elif (
        inferred
        and inferred != failure_reason
        and failure_reason
        in {
            "merge_gate_failed",
            "needs_human",
        }
    ):
        work_order["failure_reason"] = inferred
        failure_reason = inferred
        changed = True

    blocking_question = _optional_text(work_order.get("blocking_question"))
    if failure_reason and (
        not blocking_question
        or (
            inferred
            and inferred == failure_reason
            and blocking_question == _default_blocking_question_for_reason("merge_gate_failed")
        )
    ):
        work_order["blocking_question"] = _default_blocking_question_for_reason(failure_reason)
        blocking_question = _optional_text(work_order.get("blocking_question"))
        changed = True

    blocker = work_order.get("blocker")
    if (
        not isinstance(blocker, dict)
        or not _optional_text(blocker.get("reason"))
        or not _optional_text(blocker.get("question"))
    ) and (failure_reason or blocking_question):
        work_order["blocker"] = {
            "reason": failure_reason or "needs_human",
            "question": blocking_question
            or _default_blocking_question_for_reason(failure_reason or "needs_human"),
        }
        changed = True

    return changed


def _work_order_should_backfill_verification_plan(work_order: dict[str, Any]) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status not in {"needs_human", "changes_requested", "completed"}:
        return False
    if _infer_missing_failure_reason_for_work_order(work_order) != "missing_verification_plan":
        return False
    return not [
        str(item).strip() for item in work_order.get("expected_tests", []) if str(item).strip()
    ]


def _work_order_should_rehabilitate_docs_only_missing_verification_plan(
    work_order: dict[str, Any],
) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status not in {"needs_human", "changes_requested"}:
        return False
    if _infer_missing_failure_reason_for_work_order(work_order) != "missing_verification_plan":
        return False
    candidates = [
        str(path).strip()
        for path in work_order.get("changed_paths", []) or work_order.get("file_scope", [])
        if str(path).strip()
    ]
    if not candidates or not all(_is_docs_only_path(path) for path in candidates):
        return False
    return _work_order_has_concrete_deliverable(work_order)


def _work_order_should_rehabilitate_deliverable_backed_clean_exit_no_deliverable(
    work_order: dict[str, Any],
) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status not in {"needs_human", "changes_requested"}:
        return False
    if _infer_missing_failure_reason_for_work_order(work_order) != "clean_exit_no_deliverable":
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    return _work_order_has_concrete_deliverable(work_order)


def _work_order_should_reclassify_branch_snapshot_stale_review(
    work_order: dict[str, Any],
) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status != "needs_human":
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "branch_snapshot_stale":
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    if not _work_order_has_concrete_deliverable(work_order):
        return False
    metadata = work_order.get("metadata")
    if not isinstance(metadata, dict) or not bool(metadata.get("mainline_verification_passed")):
        return False
    return True


def _work_order_should_reclassify_deliverable_changes_requested(
    work_order: dict[str, Any],
) -> bool:
    if not isinstance(work_order, dict):
        return False
    if _optional_text(work_order.get("status")).lower() != "needs_human":
        return False
    if _optional_text(work_order.get("review_status")).lower() != "changes_requested":
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    if not _work_order_has_concrete_deliverable(work_order):
        return False
    return _optional_text(work_order.get("failure_reason")).lower() in {
        "verification_target_missing",
        "merge_gate_failed",
        "branch_snapshot_stale",
    }


def _work_order_should_replay_missing_verification(work_order: dict[str, Any]) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status not in {"needs_human", "changes_requested"}:
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "merge_gate_failed":
        return False
    if not _work_order_has_concrete_deliverable(work_order):
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    expected_tests = [
        str(item).strip() for item in work_order.get("expected_tests", []) if str(item).strip()
    ]
    if not expected_tests:
        return False
    verification_results = [
        dict(entry)
        for entry in work_order.get("verification_results", [])
        if isinstance(entry, dict) and str(entry.get("command", "")).strip()
    ]
    if verification_results:
        return False
    tests_run = [str(item).strip() for item in work_order.get("tests_run", []) if str(item).strip()]
    return not tests_run


def _verification_result_looks_environment_blocked(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if bool(result.get("passed", False)):
        return False
    haystack = "\n".join(
        str(result.get(key, "")).lower()
        for key in ("stdout", "stderr")
        if str(result.get(key, "")).strip()
    )
    return any(
        marker in haystack
        for marker in (
            "no module named 'aragora_debate'",
            "no module named 'pydantic_settings'",
            "cannot find module 'next/jest'",
            "jest: command not found",
            "cannot find module 'react'",
            "react/jsx-runtime",
            "cannot execute binary file",
        )
    )


def _docs_only_verification_script_path(command: Any) -> str:
    normalized = _canonical_verification_command(command)
    if not normalized:
        return ""
    try:
        tokens = shlex.split(normalized)
    except ValueError:
        return ""
    if not tokens:
        return ""
    interpreter = Path(tokens[0]).name
    if len(tokens) < 2 or not interpreter.startswith("python"):
        return ""
    return tokens[1]


def _is_docs_only_verification_command(command: Any) -> bool:
    script = _docs_only_verification_script_path(command)
    return script in {
        "scripts/reconcile_status_docs.py",
        "scripts/check_capability_matrix_sync.py",
        "scripts/check_version_alignment.py",
        "scripts/generate_capability_matrix.py",
    }


def _docs_only_merge_gate_needs_capability_matrix_generation(
    work_order: dict[str, Any],
    commands: list[str],
) -> bool:
    if any(
        _docs_only_verification_script_path(command)
        in {
            "scripts/reconcile_status_docs.py",
            "scripts/check_capability_matrix_sync.py",
        }
        for command in commands
        if str(command).strip()
    ):
        return True
    haystacks = [str(work_order.get("dispatch_error", "")).lower()]
    for entry in work_order.get("verification_results", []):
        if not isinstance(entry, dict):
            continue
        haystacks.extend(
            str(entry.get(field, "")).lower() for field in ("stdout", "stderr") if entry.get(field)
        )
    combined = "\n".join(haystacks)
    return any(
        marker in combined
        for marker in (
            "capability matrix files are out of date",
            "capability_matrix.md",
            "generate_capability_matrix.py",
        )
    )


def _docs_only_replay_commands_for_work_order(work_order: dict[str, Any]) -> list[str] | None:
    commands: list[str] = []
    seen: set[str] = set()

    def _append(command: Any) -> None:
        text = str(command).strip()
        canonical = _canonical_verification_command(text)
        if not text or not canonical or canonical in seen:
            return
        seen.add(canonical)
        commands.append(text)

    for source in (
        work_order.get("expected_tests", []),
        work_order.get("tests_run", []),
    ):
        for entry in source:
            if _is_docs_only_verification_command(entry):
                _append(entry)
    for entry in work_order.get("verification_results", []):
        if not isinstance(entry, dict):
            continue
        command = entry.get("command", "")
        if _is_docs_only_verification_command(command):
            _append(command)

    if not commands:
        return None

    if _docs_only_merge_gate_needs_capability_matrix_generation(work_order, commands):
        generated: list[str] = []
        for command in _DOCS_ONLY_GENERATE_CAPABILITY_MATRIX_COMMANDS:
            canonical = _canonical_verification_command(command)
            if canonical in seen:
                continue
            seen.add(canonical)
            generated.append(command)
        commands = generated + commands

    return commands


def _work_order_should_replay_docs_only_merge_gate_failure(work_order: dict[str, Any]) -> bool:
    if not _work_order_should_reconcile_merge_gate_failure(work_order):
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    candidates = [
        str(path).strip()
        for path in work_order.get("changed_paths", []) or work_order.get("file_scope", [])
        if str(path).strip()
    ]
    if not candidates or not all(_is_docs_only_path(path) for path in candidates):
        return False
    return _docs_only_replay_commands_for_work_order(work_order) is not None


def _missing_required_replay_commands_for_work_order(
    work_order: dict[str, Any],
) -> list[str] | None:
    if not _work_order_should_reconcile_merge_gate_failure(work_order):
        return None
    if not _optional_text(work_order.get("receipt_id")):
        return None

    merge_gate = _merge_gate_state_for_work_order(work_order)
    expected_checks = [
        str(command).strip()
        for command in merge_gate.get("expected_checks", [])
        if str(command).strip()
    ]
    verification_results = [
        dict(entry)
        for entry in work_order.get("verification_results", [])
        if isinstance(entry, dict) and str(entry.get("command", "")).strip()
    ]
    if not expected_checks:
        return None

    missing_checks = [
        command
        for command in expected_checks
        if not any(
            _verification_command_covers_expected(entry.get("command", ""), command)
            for entry in verification_results
        )
    ]
    failed_checks = [
        dict(entry)
        for entry in verification_results
        if any(
            _verification_command_covers_expected(entry.get("command", ""), command)
            for command in expected_checks
        )
        and not bool(entry.get("passed", False))
    ]
    if failed_checks or not missing_checks:
        return None
    return missing_checks


def _work_order_should_replay_missing_required_merge_gate_failure(
    work_order: dict[str, Any],
) -> bool:
    return _missing_required_replay_commands_for_work_order(work_order) is not None


def _narrow_pytest_replay_commands_for_work_order(work_order: dict[str, Any]) -> list[str] | None:
    commands: list[str] = []
    seen: set[str] = set()

    def _append(command: Any) -> None:
        text = str(command).strip()
        canonical = _canonical_verification_command(text)
        if not text or not canonical or canonical in seen:
            return
        if not _pytest_command_targets(text) or _is_overbroad_pytest_command(text):
            return
        seen.add(canonical)
        commands.append(text)

    for source in (
        work_order.get("expected_tests", []),
        work_order.get("tests_run", []),
    ):
        for entry in source:
            _append(entry)
    for entry in work_order.get("verification_results", []):
        if not isinstance(entry, dict):
            continue
        _append(entry.get("command", ""))

    return commands or None


def _work_order_should_replay_narrow_pytest_merge_gate_failure(work_order: dict[str, Any]) -> bool:
    if not _work_order_should_reconcile_merge_gate_failure(work_order):
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    return _narrow_pytest_replay_commands_for_work_order(work_order) is not None


def _work_order_should_replay_environment_blocked_verification(work_order: dict[str, Any]) -> bool:
    if not _work_order_should_reconcile_merge_gate_failure(work_order):
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    verification_results = [
        dict(entry)
        for entry in work_order.get("verification_results", [])
        if isinstance(entry, dict) and str(entry.get("command", "")).strip()
    ]
    return any(
        _verification_result_looks_environment_blocked(entry) for entry in verification_results
    )


def _targeted_replay_expected_tests_for_work_order(work_order: dict[str, Any]) -> list[str]:
    targeted: list[str] = []
    seen: set[str] = set()

    def _append(command: str) -> None:
        display = str(command).strip()
        normalized = _canonical_verification_command(display)
        if (
            not display
            or not normalized
            or normalized in seen
            or _is_overbroad_pytest_command(normalized)
        ):
            return
        seen.add(normalized)
        targeted.append(display)

    success_criteria = work_order.get("success_criteria")
    if isinstance(success_criteria, dict):
        for entry in _extract_tests_value(success_criteria.get("tests")):
            _append(entry)

    metadata = work_order.get("metadata")
    if isinstance(metadata, dict):
        for entry in metadata.get("acceptance_criteria", []):
            text = str(entry).strip()
            if text.startswith("python -m pytest") or text.startswith("pytest"):
                _append(text)
            for match in _TEST_FILE_PATTERN.findall(text):
                _append(f"python -m pytest {match} -q")

    for entry in work_order.get("tests_run", []):
        _append(str(entry))

    for entry in work_order.get("verification_results", []):
        if isinstance(entry, dict):
            _append(str(entry.get("command", "")))

    for path in work_order.get("file_scope", []):
        normalized = str(path).strip()
        if normalized.startswith("tests/") and normalized.endswith(".py"):
            _append(f"python -m pytest {normalized} -q")

    for path in work_order.get("changed_paths", []):
        normalized = str(path).strip()
        if normalized.startswith("tests/") and normalized.endswith(".py"):
            _append(f"python -m pytest {normalized} -q")

    return targeted


def _work_order_should_replay_targeted_merge_gate_failure(work_order: dict[str, Any]) -> bool:
    if not _work_order_should_reconcile_merge_gate_failure(work_order):
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    current_expected = [
        _canonical_verification_command(command)
        for command in work_order.get("expected_tests", [])
        if str(command).strip()
    ]
    if not any(_is_overbroad_pytest_command(command) for command in current_expected):
        return False
    targeted = _targeted_replay_expected_tests_for_work_order(work_order)
    if not targeted:
        return False
    return set(targeted) != {command for command in current_expected if command}


def _mainline_verification_commands_for_work_order(work_order: dict[str, Any]) -> list[str]:
    targeted = _targeted_replay_expected_tests_for_work_order(work_order)
    if targeted:
        return list(targeted)

    commands: list[str] = []
    seen: set[str] = set()
    for entry in work_order.get("expected_tests", []):
        display = str(entry).strip()
        normalized = _canonical_verification_command(display)
        if not display or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        commands.append(display)
    return commands


def _mainline_candidate_repo_paths_for_work_order(work_order: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    def _append(raw: Any) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        target = text.split("::", 1)[0].strip()
        normalized = _normalize_claim(target)
        if not normalized or any(token in normalized for token in ("*", "?", "[", "]", "{", "}")):
            return
        if normalized in seen:
            return
        seen.add(normalized)
        paths.append(normalized)

    for entry in work_order.get("file_scope", []) or []:
        _append(entry)
    for entry in work_order.get("changed_paths", []) or []:
        _append(entry)
    for command in _mainline_verification_commands_for_work_order(work_order):
        for target in _pytest_command_targets(command):
            _append(target)
    return paths


def _mainline_missing_repo_paths_for_work_order(
    work_order: dict[str, Any],
    *,
    repo_root: Path,
) -> list[str]:
    root = repo_root.resolve()
    missing: list[str] = []
    for candidate in _mainline_candidate_repo_paths_for_work_order(work_order):
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if not resolved.exists():
            missing.append(candidate)
    return missing


def _work_order_should_reclassify_branch_stale_verification_target_missing(
    work_order: dict[str, Any],
    *,
    repo_root: Path,
) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status not in {"needs_human", "changes_requested"}:
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "verification_target_missing":
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    if not _work_order_has_concrete_deliverable(work_order):
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and (
        bool(metadata.get("mainline_verification_passed"))
        or bool(metadata.get("mainline_verification_target_missing"))
    ):
        return False
    candidate_paths = _mainline_candidate_repo_paths_for_work_order(work_order)
    if not candidate_paths:
        return False
    missing_paths = _mainline_missing_repo_paths_for_work_order(work_order, repo_root=repo_root)
    return bool(missing_paths) and len(missing_paths) == len(candidate_paths)


def _work_order_should_reclassify_branch_stale_merge_gate_failure(
    work_order: dict[str, Any],
) -> bool:
    if not _work_order_should_reconcile_merge_gate_failure(work_order):
        return False
    if not _optional_text(work_order.get("receipt_id")):
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and bool(metadata.get("mainline_verification_passed")):
        return False
    return bool(_mainline_verification_commands_for_work_order(work_order))


def _work_order_should_reconcile_merge_gate_failure(work_order: dict[str, Any]) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status not in {"needs_human", "changes_requested"}:
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "merge_gate_failed":
        return False
    if not _work_order_has_concrete_deliverable(work_order):
        return False
    verification_results = [
        dict(entry)
        for entry in work_order.get("verification_results", [])
        if isinstance(entry, dict) and str(entry.get("command", "")).strip()
    ]
    return bool(verification_results)


def _merge_gate_replay_matches_task_keys(
    run_id: str,
    work_order: dict[str, Any],
    task_keys: set[str],
) -> bool:
    if not task_keys:
        return True
    metadata = work_order.get("metadata")
    metadata_task_key = ""
    if isinstance(metadata, dict):
        metadata_task_key = _optional_text(metadata.get("task_key"))
    explicit = _optional_text(work_order.get("task_key"))
    derived = ""
    work_order_id = _work_order_identifier(work_order)
    if run_id and work_order_id:
        derived = f"{run_id}:{work_order_id}"
    return metadata_task_key in task_keys or explicit in task_keys or derived in task_keys


def _work_order_reap_failure_reason(
    work_order: dict[str, Any],
    *,
    lease_status: str | None = None,
) -> str:
    for blocker in _developer_task_blockers(work_order):
        normalized = blocker.strip().lower()
        if normalized in _REAPED_NO_RECEIPT_BLOCKERS:
            return normalized
    status = _optional_text(work_order.get("status")).lower()
    normalized_lease_status = _optional_text(lease_status).lower()
    if status in {"leased", "dispatched", "active", "integrating"} and normalized_lease_status in {
        "released",
        "expired",
    }:
        return (
            "expired_lease_reaped"
            if normalized_lease_status == LeaseStatus.EXPIRED.value
            else "stale_lease_reaped"
        )
    return ""


def _work_order_should_archive_reaped_no_receipt(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status not in _OPEN_DEVELOPER_TASK_STATUSES:
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if (
        isinstance(metadata, dict)
        and _optional_text(metadata.get("archived_due_to")) == "reaped_no_receipt"
    ):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    if not _work_order_reap_failure_reason(work_order, lease_status=lease_status):
        return False
    updated_at = _parse_dt(_developer_task_updated_at(work_order, run))
    return updated_at <= cutoff


def _work_order_has_scope_violation(work_order: dict[str, Any]) -> bool:
    if _optional_text(work_order.get("status")).lower() == "scope_violation":
        return True
    if isinstance(work_order.get("scope_violation"), dict):
        return True
    lowered_blockers = {
        blocker.strip().lower()
        for blocker in _developer_task_blockers(work_order)
        if blocker.strip()
    }
    return "scope_violation" in lowered_blockers


def _work_order_should_archive_scope_violation_no_deliverable(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status not in (_OPEN_DEVELOPER_TASK_STATUSES | {"scope_violation"}):
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    if not _work_order_has_scope_violation(work_order):
        return False
    updated_at = _parse_dt(_developer_task_updated_at(work_order, run))
    return updated_at <= cutoff


def _work_order_failed_no_deliverable_reason(work_order: dict[str, Any]) -> str:
    for blocker in _developer_task_blockers(work_order):
        text = blocker.strip()
        if text:
            return text
    return _optional_text(work_order.get("failure_reason")) or "failed_no_deliverable"


def _work_order_clean_exit_no_deliverable_reason(work_order: dict[str, Any]) -> str:
    status = _optional_text(work_order.get("status")).lower()
    metadata = work_order.get("metadata")
    candidate_fields = [
        work_order.get("worker_outcome"),
        work_order.get("last_run_outcome"),
        work_order.get("failure_reason"),
    ]
    if isinstance(metadata, dict):
        candidate_fields.extend(
            [
                metadata.get("outcome"),
                metadata.get("terminal_outcome"),
                metadata.get("receipt_outcome"),
            ]
        )
    for candidate in candidate_fields:
        if _optional_text(candidate).lower() == "clean_exit_no_deliverable":
            return "clean_exit_no_deliverable"
    for blocker in _developer_task_blockers(work_order):
        normalized = blocker.strip().lower()
        if normalized in {
            "clean_exit_no_deliverable",
            "run ended without a concrete deliverable.",
        }:
            return blocker.strip()
    if (
        status == "completed"
        and _work_order_receipt_outcome(work_order) == "clean_exit_no_deliverable"
        and not _developer_task_blockers(work_order)
        and not _optional_text(work_order.get("failure_reason"))
    ):
        return "clean_exit_no_deliverable"
    return ""


def _work_order_should_archive_failed_no_deliverable(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    timeout_like_needs_human = False
    if status == "needs_human":
        worker_outcome = _optional_text(work_order.get("worker_outcome")).lower()
        failure_reason = _optional_text(work_order.get("failure_reason")).lower()
        blockers = {blocker.strip().lower() for blocker in _developer_task_blockers(work_order)}
        timeout_like_needs_human = (
            worker_outcome
            in {
                "timeout_no_progress",
                "timeout_with_salvage",
            }
            or any("timeout" in blocker for blocker in blockers)
            or "timeout" in failure_reason
        )
    if status not in {"failed", "dispatch_failed", "timed_out"} and not timeout_like_needs_human:
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    updated_at = _parse_dt(_developer_task_updated_at(work_order, run))
    return updated_at <= cutoff


def _work_order_should_archive_clean_exit_no_deliverable(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status not in {"completed", "needs_human"}:
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    if not _work_order_clean_exit_no_deliverable_reason(work_order):
        return False
    updated_at = _parse_dt(_work_order_clean_exit_no_deliverable_staleness_anchor(work_order, run))
    return updated_at <= cutoff


def _work_order_should_archive_work_order_leasing_failed(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status != "needs_human":
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "work_order_leasing_failed":
        return False
    updated_at = _parse_dt(_work_order_leasing_failed_staleness_anchor(work_order, run))
    return updated_at <= cutoff


def _work_order_should_archive_worker_type_blocked(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status != "needs_human":
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "worker_type_blocked":
        return False
    updated_at = _parse_dt(_work_order_leasing_failed_staleness_anchor(work_order, run))
    return updated_at <= cutoff


def _work_order_leasing_failed_staleness_anchor(
    work_order: dict[str, Any],
    run: dict[str, Any],
) -> str:
    for value in (
        *(
            work_order.get(key)
            for key in (
                "last_observed_at",
                "last_progress_at",
                "completed_at",
                "dispatched_at",
                "leased_at",
                "started_at",
            )
        ),
        run.get("created_at"),
        run.get("updated_at"),
        _utcnow().isoformat(),
    ):
        text = str(value or "").strip()
        if text and text.lower() != "none":
            return text
    return _utcnow().isoformat()


def _work_order_clean_exit_no_deliverable_staleness_anchor(
    work_order: dict[str, Any],
    run: dict[str, Any],
) -> str:
    for value in (
        *(
            work_order.get(key)
            for key in (
                "last_observed_at",
                "last_progress_at",
                "completed_at",
                "dispatched_at",
                "leased_at",
                "started_at",
            )
        ),
        run.get("created_at"),
        run.get("updated_at"),
        _utcnow().isoformat(),
    ):
        text = str(value or "").strip()
        if text and text.lower() != "none":
            return text
    return _utcnow().isoformat()


def _work_order_should_archive_superseded_clean_exit_no_deliverable(
    work_order: dict[str, Any],
) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status != "needs_human":
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "clean_exit_no_deliverable":
        return False
    return _optional_text(work_order.get("worker_outcome")).lower() == "clean_exit_no_effect"


_HELPER_CLEAN_EXIT_TITLE_PREFIXES = (
    "read existing ",
    "inspect ",
    "understand ",
    "run tests and validate",
    "validate implementation",
    "review existing ",
    "analyze existing ",
)


def _looks_like_helper_clean_exit_no_deliverable(work_order: dict[str, Any]) -> bool:
    title = " ".join(_optional_text(work_order.get("title")).lower().split())
    return any(title.startswith(prefix) for prefix in _HELPER_CLEAN_EXIT_TITLE_PREFIXES)


def _work_order_should_archive_superseded_stale_lease_reaped(
    work_order: dict[str, Any],
) -> bool:
    if not isinstance(work_order, dict):
        return False
    status = _optional_text(work_order.get("status")).lower()
    if status != "needs_human":
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "stale_lease_reaped":
        return False
    return _looks_like_helper_clean_exit_no_deliverable(work_order)


def _work_order_is_live_overlap_sibling(work_order: dict[str, Any]) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status in {"discarded", "superseded", "merged"}:
        return False
    return status in {
        "queued",
        "leased",
        "dispatched",
        "active",
        "completed",
        "changes_requested",
        "needs_human",
    }


def _live_overlap_sibling_priority(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> tuple[int, int, int, str]:
    status = _optional_text(work_order.get("status")).lower()
    status_rank = {
        "completed": 6,
        "changes_requested": 5,
        "active": 4,
        "dispatched": 3,
        "leased": 2,
        "queued": 1,
        "needs_human": 0,
    }.get(status, -1)
    return (
        status_rank,
        1 if _work_order_has_concrete_deliverable(work_order) else 0,
        len(_work_order_scope_patterns(work_order)),
        _developer_task_updated_at(work_order, run),
    )


def _work_order_is_duplicate_work_order_leasing_failed_candidate(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    lease_status: str | None,
) -> bool:
    if _optional_text(work_order.get("status")).lower() != "needs_human":
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("failure_reason")).lower() != "work_order_leasing_failed":
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    return bool(_canonical_work_order_scope_key(work_order)) and bool(
        _canonical_goal_key(run.get("goal"))
    )


def _work_order_is_duplicate_waiting_conflict_candidate(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    lease_status: str | None,
) -> bool:
    if _optional_text(work_order.get("status")).lower() != "waiting_conflict":
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    return bool(_duplicate_waiting_conflict_group_key(work_order, run=run))


def _waiting_conflict_candidate_text(work_order: dict[str, Any], *, run: dict[str, Any]) -> str:
    metadata = work_order.get("metadata") or {}
    acceptance = metadata.get("acceptance_criteria") if isinstance(metadata, dict) else []
    parts: list[str] = [
        _optional_text(run.get("goal")),
        _optional_text(work_order.get("title")),
        _optional_text(work_order.get("description")),
    ]
    if isinstance(acceptance, list):
        parts.extend(str(item).strip() for item in acceptance if str(item).strip())
    return " ".join(part for part in parts if part).lower()


def _work_order_source_name(work_order: dict[str, Any]) -> str:
    metadata = work_order.get("metadata") or {}
    return _optional_text(work_order.get("source"), metadata.get("source")).lower()


def _work_order_is_broad_explicit_pytest_umbrella(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> bool:
    if _work_order_source_name(work_order) != "explicit_spec_work_order":
        return False
    text = _waiting_conflict_candidate_text(work_order, run=run)
    if "pytest" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "comprehensive pytest",
            "thorough pytest",
            "cover every",
            "internal helper",
            "helper function",
        )
    )


def _work_order_is_specific_pytest_child(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> bool:
    text = _waiting_conflict_candidate_text(work_order, run=run)
    if "pytest" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "write one pytest test",
            "one pytest test",
            "single pytest test",
        )
    )


def _duplicate_waiting_conflict_group_key(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> tuple[str, str, tuple[str, ...]] | None:
    scope_key = _canonical_work_order_scope_key(work_order)
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict):
        tranche_lane_id = _optional_text(metadata.get("tranche_lane_id"))
        if tranche_lane_id:
            return ("tranche_lane_id", tranche_lane_id, scope_key)
    goal_key = _canonical_goal_key(run.get("goal"))
    if not goal_key:
        return None
    return ("goal", goal_key, scope_key)


def _superseded_waiting_conflict_group_key(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> str | None:
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict):
        tranche_lane_id = _optional_text(metadata.get("tranche_lane_id"))
        if tranche_lane_id:
            return f"lane:{tranche_lane_id}"
    goal_key = _canonical_goal_key(run.get("goal"))
    if not goal_key:
        return None
    return f"goal:{goal_key}"


def _duplicate_work_order_leasing_failed_priority(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        _developer_task_updated_at(work_order, run),
        _optional_text(run.get("created_at")),
        _optional_text(work_order.get("work_order_id"), work_order.get("task_id")),
    )


def _work_order_should_archive_duplicate_branch_deliverable(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status in {"discarded", "superseded", "merged"}:
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if not _optional_text(work_order.get("branch")):
        return False
    if not _work_order_has_concrete_deliverable(work_order):
        return False
    updated_at = _parse_dt(_developer_task_updated_at(work_order, run))
    return updated_at <= cutoff


def _work_order_scope_patterns(work_order: dict[str, Any]) -> list[str]:
    patterns = [
        _canonical_scope_pattern(str(item))
        for item in work_order.get("file_scope", []) or []
        if _canonical_scope_pattern(str(item))
    ]
    if patterns:
        return _collapse_scope_patterns(patterns)
    changed_paths = [
        _canonical_scope_pattern(str(item))
        for item in work_order.get("changed_paths", []) or []
        if _canonical_scope_pattern(str(item))
    ]
    return _collapse_scope_patterns(changed_paths)


def _canonical_work_order_scope_key(work_order: dict[str, Any]) -> tuple[str, ...]:
    patterns = _work_order_scope_patterns(work_order)
    if not patterns:
        return ()
    return tuple(sorted(dict.fromkeys(patterns)))


def _canonical_scope_pattern(value: str) -> str:
    clean = _normalize_claim(value)
    if not clean:
        return ""
    if clean.endswith("/**"):
        clean = clean[:-3].rstrip("/")
    return clean


def _collapse_scope_patterns(patterns: list[str]) -> list[str]:
    collapsed: list[str] = []
    unique_patterns = list(
        dict.fromkeys(_canonical_scope_pattern(item) for item in patterns if item)
    )
    for pattern in unique_patterns:
        if not pattern:
            continue
        if any(
            other != pattern and _path_matches_glob(pattern, other)
            for other in unique_patterns
            if other
        ):
            continue
        collapsed.append(pattern)
    return collapsed


def _canonical_goal_key(value: Any) -> str:
    text = str(value or "").strip()
    for paragraph in re.split(r"\n\s*\n", text):
        candidate = " ".join(paragraph.split()).strip()
        if not candidate:
            continue
        lower = candidate.lower()
        if lower.startswith(
            (
                "## ",
                "validation",
                "allowed write scope",
                "verification commands",
                "source issue context",
                "acceptance criteria",
                "context",
                "goal",
            )
        ):
            continue
        first_sentence = re.split(r"(?<=[.!?])\s+", candidate, maxsplit=1)[0]
        normalized = " ".join(first_sentence.split()).strip().lower()
        if normalized:
            return normalized
    return " ".join(text.split()).strip().lower()


def _claim_contains(container: str, containee: str) -> bool:
    clean_container = _normalize_claim(container)
    clean_containee = _normalize_claim(containee)
    if not clean_container or not clean_containee:
        return False
    return _path_matches_glob(clean_containee, clean_container)


def _work_order_scope_contains(container: dict[str, Any], containee: dict[str, Any]) -> bool:
    container_patterns = _work_order_scope_patterns(container)
    containee_patterns = _work_order_scope_patterns(containee)
    if not container_patterns or not containee_patterns:
        return False
    return all(
        any(
            _claim_contains(container_pattern, containee_pattern)
            for container_pattern in container_patterns
        )
        for containee_pattern in containee_patterns
    )


def _work_orders_overlap_by_scope(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    first_globs = _work_order_scope_patterns(first)
    second_globs = _work_order_scope_patterns(second)
    first_paths = [
        _normalize_claim(str(item))
        for item in first.get("changed_paths", []) or []
        if _normalize_claim(str(item))
    ]
    second_paths = [
        _normalize_claim(str(item))
        for item in second.get("changed_paths", []) or []
        if _normalize_claim(str(item))
    ]
    if not first_globs and not first_paths:
        return False
    if not second_globs and not second_paths:
        return False
    return _globs_overlap_any(first_globs, second_globs, first_paths, second_paths)


def _work_order_should_archive_superseded_waiting_conflict(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    if _optional_text(work_order.get("status")).lower() != "waiting_conflict":
        return False
    if _optional_text(lease_status).lower() == LeaseStatus.ACTIVE.value:
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(work_order.get("receipt_id")) or _work_order_has_concrete_deliverable(
        work_order
    ):
        return False
    if not _work_order_scope_patterns(work_order):
        return False
    updated_at = _parse_dt(_waiting_conflict_staleness_anchor(work_order, run))
    return updated_at <= cutoff


def _work_order_should_rehabilitate_narrowed_waiting_conflict(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    cutoff: datetime,
    lease_status: str | None,
) -> bool:
    if not _work_order_should_archive_superseded_waiting_conflict(
        work_order,
        run=run,
        cutoff=cutoff,
        lease_status=lease_status,
    ):
        return False
    return bool(
        _narrow_waiting_conflict_scope_from_explicit_paths(
            work_order,
            run=run,
            repo_root=None,
        )
    )


def _waiting_conflict_staleness_anchor(work_order: dict[str, Any], run: dict[str, Any]) -> str:
    for value in (
        *(
            work_order.get(key)
            for key in (
                "last_observed_at",
                "last_progress_at",
                "completed_at",
                "dispatched_at",
                "leased_at",
                "started_at",
            )
        ),
        run.get("created_at"),
        run.get("updated_at"),
        _utcnow().isoformat(),
    ):
        text = str(value or "").strip()
        if text and text.lower() != "none":
            return text
    return _utcnow().isoformat()


def _duplicate_branch_deliverable_priority(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> tuple[int, int, int, str]:
    commit_shas = [
        str(item).strip() for item in work_order.get("commit_shas", []) if str(item).strip()
    ]
    return (
        1 if _optional_text(work_order.get("receipt_id")) else 0,
        len(commit_shas),
        1 if _optional_text(work_order.get("head_sha")) else 0,
        _developer_task_updated_at(work_order, run),
    )


def _duplicate_waiting_conflict_priority(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> tuple[int, str, str]:
    return (
        -len(_work_order_scope_patterns(work_order)),
        _waiting_conflict_staleness_anchor(work_order, run),
        _optional_text(work_order.get("work_order_id"), work_order.get("task_id")),
    )


def _containing_waiting_conflict_priority(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> tuple[int, int, str]:
    patterns = _work_order_scope_patterns(work_order)
    return (
        len(patterns),
        -sum(len(pattern) for pattern in patterns),
        _optional_text(work_order.get("work_order_id"), work_order.get("task_id")),
    )


def _is_concrete_repo_path_hint(path: str, *, repo_root: Path | None) -> bool:
    clean = _normalize_claim(path)
    if not clean or any(token in clean for token in ("*", "?", "[", "]", "{", "}")):
        return False
    name = clean.rsplit("/", 1)[-1]
    if repo_root is not None:
        candidate = (repo_root / clean).resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError:
            return False
        if candidate.is_file():
            return True
    return "." in name


def _waiting_conflict_inference_text(work_order: dict[str, Any], run: dict[str, Any]) -> str:
    spec = run.get("spec")
    metadata = work_order.get("metadata")
    spec_acceptance = spec.get("acceptance_criteria") if isinstance(spec, dict) else []
    spec_constraints = spec.get("constraints") if isinstance(spec, dict) else []
    metadata_acceptance = metadata.get("acceptance_criteria") if isinstance(metadata, dict) else []
    metadata_constraints = metadata.get("constraints") if isinstance(metadata, dict) else []
    parts = [
        run.get("goal"),
        spec.get("raw_goal") if isinstance(spec, dict) else None,
        spec.get("refined_goal") if isinstance(spec, dict) else None,
        work_order.get("title"),
        work_order.get("description"),
        metadata.get("description") if isinstance(metadata, dict) else None,
    ]
    if isinstance(spec_acceptance, list):
        parts.extend(spec_acceptance)
    if isinstance(spec_constraints, list):
        parts.extend(spec_constraints)
    if isinstance(metadata_acceptance, list):
        parts.extend(metadata_acceptance)
    if isinstance(metadata_constraints, list):
        parts.extend(metadata_constraints)
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def _explicit_scope_paths_for_waiting_conflict(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    repo_root: Path | None,
) -> list[str]:
    from aragora.swarm.spec import SwarmSpec

    original_scope = [
        _canonical_scope_pattern(str(path))
        for path in work_order.get("file_scope", []) or []
        if _canonical_scope_pattern(str(path))
    ]
    if not original_scope:
        return []
    explicit_paths: list[str] = []
    for path in SwarmSpec.infer_file_scope_hints(_waiting_conflict_inference_text(work_order, run)):
        clean = _normalize_claim(path)
        if not _is_concrete_repo_path_hint(clean, repo_root=repo_root):
            continue
        if not any(_path_matches_glob(clean, scope) for scope in original_scope):
            continue
        if clean not in explicit_paths:
            explicit_paths.append(clean)
    return explicit_paths


def _docs_only_scope_hints_for_waiting_conflict(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
) -> list[str]:
    metadata = work_order.get("metadata") or {}
    constraints = metadata.get("constraints") if isinstance(metadata, dict) else []
    if not isinstance(constraints, list) or not any(
        "documentation only" in str(item).strip().lower() for item in constraints
    ):
        spec = run.get("spec")
        spec_constraints = spec.get("constraints") if isinstance(spec, dict) else []
        if not isinstance(spec_constraints, list) or not any(
            "documentation only" in str(item).strip().lower() for item in spec_constraints
        ):
            return []

    original_scope = [
        _canonical_scope_pattern(str(path))
        for path in work_order.get("file_scope", []) or []
        if _canonical_scope_pattern(str(path))
    ]
    if not original_scope:
        return []

    def _scope_supports_docs_only_hint(path: str) -> bool:
        if any(
            scope == path or path.startswith(f"{scope}/") or scope.startswith(f"{path}/")
            for scope in original_scope
        ):
            return True
        if not is_docs_safe_top_level_file(path):
            return False
        return any(canonical_docs_container_scope(scope) == "docs" for scope in original_scope)

    doc_hints: list[str] = []
    for path in infer_docs_safe_hints(_waiting_conflict_inference_text(work_order, run)):
        clean = _canonical_scope_pattern(path)
        if _scope_supports_docs_only_hint(clean):
            doc_hints.append(clean)
    if any(
        is_docs_safe_path(hint) and canonical_docs_container_scope(hint) is None
        for hint in doc_hints
    ):
        doc_hints = [hint for hint in doc_hints if canonical_docs_container_scope(hint) is None]
    collapsed = _collapse_scope_patterns(doc_hints)
    if collapsed:
        return collapsed
    return list(
        dict.fromkeys(
            scope
            for scope in (canonical_docs_container_scope(path) for path in original_scope)
            if scope is not None
        )
    )


def _narrow_waiting_conflict_scope_from_explicit_paths(
    work_order: dict[str, Any],
    *,
    run: dict[str, Any],
    repo_root: Path | None,
) -> list[str]:
    original_scope = [
        _canonical_scope_pattern(str(path))
        for path in work_order.get("file_scope", []) or []
        if _canonical_scope_pattern(str(path))
    ]
    if not original_scope:
        return []
    explicit_paths = _explicit_scope_paths_for_waiting_conflict(
        work_order,
        run=run,
        repo_root=repo_root,
    )
    if explicit_paths:
        narrowed_scope: list[str] = []
        replaced = False
        for scope in original_scope:
            contains_explicit = any(_path_matches_glob(path, scope) for path in explicit_paths)
            if (
                contains_explicit
                and scope not in explicit_paths
                and not _is_concrete_repo_path_hint(
                    scope,
                    repo_root=repo_root,
                )
            ):
                replaced = True
                continue
            narrowed_scope.append(scope)
        if replaced:
            return _collapse_scope_patterns(narrowed_scope + explicit_paths)

    docs_only_scope = _docs_only_scope_hints_for_waiting_conflict(work_order, run=run)
    if docs_only_scope and tuple(docs_only_scope) != tuple(original_scope):
        return docs_only_scope
    return []


def _waiting_conflict_sibling_can_be_ignored(
    candidate: dict[str, Any],
    sibling: dict[str, Any],
) -> bool:
    if _optional_text(sibling.get("status")).lower() != "waiting_conflict":
        return False
    metadata = sibling.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    if _optional_text(sibling.get("receipt_id")) or _work_order_has_concrete_deliverable(sibling):
        return False
    return _work_order_scope_contains(sibling, candidate) and not _work_order_scope_contains(
        candidate,
        sibling,
    )


def _blocking_waiting_conflict_siblings(
    candidate: dict[str, Any],
    *,
    run: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_id = _optional_text(run.get("run_id"))
    work_order_id = _work_order_identifier(candidate)
    conflicts: list[dict[str, Any]] = []
    for sibling_record in records:
        sibling_run_id = _optional_text(sibling_record.get("run_id"))
        for sibling in sibling_record.get("work_orders", []):
            if not isinstance(sibling, dict):
                continue
            if sibling is candidate and sibling_run_id == run_id:
                continue
            if (
                work_order_id
                and sibling_run_id == run_id
                and _work_order_identifier(sibling) == work_order_id
            ):
                continue
            status = _optional_text(sibling.get("status")).lower()
            if status in {"discarded", "superseded", "merged"}:
                continue
            metadata = sibling.get("metadata")
            if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
                continue
            if not _work_orders_overlap_by_scope(candidate, sibling):
                continue
            if _waiting_conflict_sibling_can_be_ignored(candidate, sibling):
                continue
            conflicts.append(
                {
                    "source": "work_order",
                    "run_id": sibling_run_id or None,
                    "work_order_id": _work_order_identifier(sibling) or None,
                    "status": status or None,
                    "title": _optional_text(sibling.get("title")) or None,
                    "allowed_globs": _work_order_scope_patterns(sibling),
                }
            )
    return conflicts


def _work_order_receipt_outcome(work_order: dict[str, Any]) -> str:
    if _optional_text(work_order.get("adopted_pr")):
        return "pr_adopted"
    if _work_order_has_concrete_deliverable(work_order):
        return "deliverable_created"
    return "clean_exit_no_deliverable"


def _work_order_should_backfill_file_scope_from_changed_paths(work_order: dict[str, Any]) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status in {"discarded", "superseded", "merged", "scope_violation"}:
        return False
    if any(str(path).strip() for path in work_order.get("file_scope", []) or []):
        return False
    if not any(str(path).strip() for path in work_order.get("changed_paths", []) or []):
        return False
    if isinstance(work_order.get("scope_violation"), dict):
        return False
    metadata = work_order.get("metadata")
    if isinstance(metadata, dict) and _optional_text(metadata.get("archived_due_to")):
        return False
    return True


def _work_order_should_backfill_receipt(work_order: dict[str, Any]) -> bool:
    status = _optional_text(work_order.get("status")).lower()
    if status in {"queued", "leased", "dispatched", "active", "integrating", "discarded"}:
        return False
    if _optional_text(work_order.get("receipt_id")):
        return False
    if not _optional_text(work_order.get("lease_id")):
        return False
    return _work_order_receipt_outcome(work_order) in {"deliverable_created", "pr_adopted"}


def _extract_pr_number(pr_reference: str) -> int | None:
    text = str(pr_reference or "").strip().rstrip("/")
    if not text:
        return None
    tail = text.rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _developer_task_priority(work_order: dict[str, Any]) -> int:
    explicit = work_order.get("priority")
    if isinstance(explicit, int):
        return max(0, min(100, explicit))
    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        if normalized.isdigit():
            return max(0, min(100, int(normalized)))
        named = {
            "critical": 95,
            "high": 80,
            "medium": 60,
            "normal": 50,
            "low": 35,
        }
        if normalized in named:
            return named[normalized]
    risk_level = str(work_order.get("risk_level", "")).strip().lower()
    risk_priority = {
        "critical": 90,
        "high": 75,
        "review": 60,
        "medium": 55,
        "low": 45,
    }
    return risk_priority.get(risk_level, 50)


def _developer_task_updated_at(work_order: dict[str, Any], run: dict[str, Any]) -> str:
    for value in (
        *(
            work_order.get(key)
            for key in (
                "last_observed_at",
                "last_progress_at",
                "completed_at",
                "dispatched_at",
                "leased_at",
                "started_at",
            )
        ),
        run.get("updated_at"),
        _utcnow().isoformat(),
    ):
        text = str(value or "").strip()
        if text and text.lower() != "none":
            return text
    return _utcnow().isoformat()


def _developer_task_work_status(status: str) -> WorkStatus:
    normalized = str(status or "").strip().lower()
    if normalized == "queued":
        return WorkStatus.READY
    if normalized == "leased":
        return WorkStatus.CLAIMED
    if normalized in {"dispatched", "active"}:
        return WorkStatus.IN_PROGRESS
    if normalized in {"waiting_conflict", "dispatch_failed", "needs_human", "changes_requested"}:
        return WorkStatus.BLOCKED
    if normalized in {"timed_out", "failed"}:
        return WorkStatus.FAILED
    if normalized in {"merged", "discarded", "salvage", "completed", "integrating"}:
        return WorkStatus.COMPLETED
    return WorkStatus.PENDING


def _safe_kill_probe(raw_pid: Any) -> Exception | None:
    """Probe whether *raw_pid* is alive.  Returns ``None`` if alive, the exception otherwise."""
    import os as _os

    try:
        _os.kill(int(raw_pid), 0)
        return None
    except (ProcessLookupError, PermissionError, TypeError, ValueError, OSError) as exc:
        return exc


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _json_dump(value: Any) -> str:
    return json.dumps(_json_compatible(value), sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _artifact_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_json_compatible(payload), sort_keys=True).encode()
    ).hexdigest()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _json_compatible(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    return value


def _normalize_claim(value: str) -> str:
    return value.strip().strip("/")


def _has_wildcard(pattern: str) -> bool:
    return any(token in pattern for token in ("*", "?", "["))


def _path_matches_glob(path: str, pattern: str) -> bool:
    clean_path = _normalize_claim(path)
    clean_pattern = _normalize_claim(pattern)
    if not clean_pattern:
        return False
    if _has_wildcard(clean_pattern):
        if clean_pattern.endswith("/**"):
            prefix = clean_pattern[:-3].rstrip("/")
            return clean_path == prefix or clean_path.startswith(f"{prefix}/")
        return PurePosixPath(clean_path).match(clean_pattern)
    return clean_path == clean_pattern or clean_path.startswith(f"{clean_pattern}/")


def _glob_overlap(first: str, second: str) -> bool:
    a = _normalize_claim(first)
    b = _normalize_claim(second)
    if not a or not b:
        return False
    if a == b:
        return True
    a_wild = _has_wildcard(a)
    b_wild = _has_wildcard(b)
    if not a_wild and not b_wild:
        return a.startswith(f"{b}/") or b.startswith(f"{a}/")
    if not a_wild:
        return _path_matches_glob(a, b)
    if not b_wild:
        return _path_matches_glob(b, a)
    a_prefix = a.split("*")[0]
    b_prefix = b.split("*")[0]
    if a_prefix and b_prefix and (a_prefix.startswith(b_prefix) or b_prefix.startswith(a_prefix)):
        return True
    return False


def _globs_overlap_any(
    first_globs: list[str],
    second_globs: list[str],
    first_paths: list[str],
    second_paths: list[str],
) -> bool:
    for path in first_paths:
        if _claims_overlap([path], second_globs, second_paths):
            return True
    for path in second_paths:
        if _claims_overlap([path], first_globs, first_paths):
            return True
    for left in first_globs:
        for right in second_globs:
            if _glob_overlap(left, right):
                return True
    return False


def _claims_overlap(
    claimed_paths: list[str], allowed_globs: list[str], other_paths: list[str]
) -> bool:
    for claimed in claimed_paths:
        for glob in allowed_globs:
            if _path_matches_glob(claimed, glob) or _path_matches_glob(glob, claimed):
                return True
        for other in other_paths:
            if _glob_overlap(claimed, other):
                return True
    return False


def _parse_worktree_entries(raw: str) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    current_path: Path | None = None
    current_branch: str | None = None
    for line in raw.splitlines():
        text = line.strip()
        if text.startswith("worktree "):
            current_path = Path(text[len("worktree ") :]).resolve()
            current_branch = None
        elif text.startswith("branch refs/heads/"):
            current_branch = text[len("branch refs/heads/") :]
        elif text == "" and current_path is not None and current_branch is not None:
            entries.append((current_path, current_branch))
            current_path = None
            current_branch = None
    if current_path is not None and current_branch is not None:
        entries.append((current_path, current_branch))
    return entries


def _status_paths(lines: list[str]) -> list[str]:
    paths: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        path = text[3:] if len(text) > 3 else text
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(_normalize_claim(path))
    return paths


def _estimate_salvage_value(*, ahead: int, changed_paths: list[str], dirty: bool) -> float:
    value = 0.2
    if dirty:
        value += 0.2
    value += min(0.4, ahead * 0.1)
    value += min(0.2, len(set(changed_paths)) * 0.02)
    return max(0.0, min(1.0, value))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dev coordination control plane")
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--db", default=None, help="Optional explicit SQLite path")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show coordination status")
    status.add_argument("--json", action="store_true")

    claim = sub.add_parser("claim", help="Claim a bounded work lease")
    claim.add_argument("--task-id", required=True)
    claim.add_argument("--title", required=True)
    claim.add_argument("--agent", required=True)
    claim.add_argument("--session-id", required=True)
    claim.add_argument("--branch", required=True)
    claim.add_argument("--worktree", required=True)
    claim.add_argument("--write-scope", action="append", default=[])
    claim.add_argument("--claimed-path", action="append", default=[])
    claim.add_argument("--test", action="append", default=[])
    claim.add_argument("--ttl-hours", type=float, default=8.0)
    claim.add_argument("--allow-overlap", action="store_true")
    claim.add_argument("--json", action="store_true")

    complete = sub.add_parser("complete", help="Record a completion receipt")
    complete.add_argument("--lease-id", required=True)
    complete.add_argument("--agent", required=True)
    complete.add_argument("--session-id", required=True)
    complete.add_argument("--branch", required=True)
    complete.add_argument("--worktree", required=True)
    complete.add_argument("--base-sha", default=None)
    complete.add_argument("--head-sha", default=None)
    complete.add_argument("--commit", action="append", default=[])
    complete.add_argument("--changed-path", action="append", default=[])
    complete.add_argument("--test", action="append", default=[])
    complete.add_argument("--validation", action="append", default=[])
    complete.add_argument("--assumption", action="append", default=[])
    complete.add_argument("--blocker", action="append", default=[])
    complete.add_argument("--outcome", default="completed")
    complete.add_argument("--risk", action="append", default=[])
    complete.add_argument("--pr-url", default=None)
    complete.add_argument("--pr-number", type=int, default=None)
    complete.add_argument("--confidence", type=float, default=0.0)
    complete.add_argument("--json", action="store_true")

    heartbeat = sub.add_parser("heartbeat", help="Refresh a lease TTL")
    heartbeat.add_argument("--lease-id", required=True)
    heartbeat.add_argument("--ttl-hours", type=float, default=None)
    heartbeat.add_argument("--json", action="store_true")

    decide = sub.add_parser("decide", help="Record an integration decision")
    decide.add_argument("--receipt-id", required=True)
    decide.add_argument(
        "--decision",
        required=True,
        choices=[item.value for item in IntegrationDecisionType],
    )
    decide.add_argument("--decided-by", required=True)
    decide.add_argument("--rationale", required=True)
    decide.add_argument("--target-branch", default="main")
    decide.add_argument("--commit", action="append", default=[])
    decide.add_argument("--follow-up", action="append", default=[])
    decide.add_argument("--lease-id", default=None)
    decide.add_argument("--json", action="store_true")

    salvage = sub.add_parser("scan-salvage", help="Scan stashes/worktrees for salvage candidates")
    salvage.add_argument("--no-worktrees", action="store_true")
    salvage.add_argument("--no-stashes", action="store_true")
    salvage.add_argument("--max-stashes", type=int, default=25)
    salvage.add_argument("--json", action="store_true")

    sync_queue = sub.add_parser(
        "sync-queue",
        help="Project pending integration/salvage items into the global work queue",
    )
    sync_queue.add_argument("--json", action="store_true")

    tasks = sub.add_parser("tasks", help="List canonical developer tasks")
    tasks.add_argument("--open-only", action="store_true")
    tasks.add_argument("--json", action="store_true")

    sync_tasks = sub.add_parser(
        "sync-tasks",
        help="Project open developer tasks into the global work queue",
    )
    sync_tasks.add_argument("--json", action="store_true")

    reap = sub.add_parser("reap", help="Expire stale leases and release mirrored fleet claims")
    reap.add_argument("--json", action="store_true")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    db_path = Path(args.db).resolve() if args.db else None
    store = DevCoordinationStore(repo_root=Path(args.repo), db_path=db_path)

    if args.command == "status":
        payload = store.status_summary()
        if args.json:
            print(json.dumps(payload, indent=2))  # noqa: T201
        else:
            counts = payload["counts"]
            print(  # noqa: T201
                f"active_leases={counts['active_leases']} "
                f"pending_integrations={counts['pending_integrations']} "
                f"open_salvage_candidates={counts['open_salvage_candidates']} "
                f"open_developer_tasks={counts['open_developer_tasks']}"
            )
        return 0

    if args.command == "claim":
        try:
            lease = store.claim_lease(
                task_id=args.task_id,
                title=args.title,
                owner_agent=args.agent,
                owner_session_id=args.session_id,
                branch=args.branch,
                worktree_path=args.worktree,
                allowed_globs=args.write_scope,
                claimed_paths=args.claimed_path,
                expected_tests=args.test,
                ttl_hours=args.ttl_hours,
                allow_overlap=args.allow_overlap,
            )
        except LeaseConflictError as exc:
            if args.json:
                print(json.dumps({"ok": False, "conflicts": exc.conflicts}, indent=2))  # noqa: T201
            else:
                print("lease_conflict", json.dumps(exc.conflicts, indent=2))  # noqa: T201
            return 2
        if args.json:
            print(json.dumps({"ok": True, "lease": lease.to_dict()}, indent=2))  # noqa: T201
        else:
            print(lease.lease_id)  # noqa: T201
        return 0

    if args.command == "complete":
        receipt = store.record_completion(
            lease_id=args.lease_id,
            owner_agent=args.agent,
            owner_session_id=args.session_id,
            branch=args.branch,
            worktree_path=args.worktree,
            base_sha=args.base_sha,
            head_sha=args.head_sha,
            commit_shas=args.commit,
            changed_paths=args.changed_path,
            tests_run=args.test,
            validations_run=args.validation,
            assumptions=args.assumption,
            blockers=args.blocker,
            outcome=args.outcome,
            risks=args.risk,
            pr_url=args.pr_url,
            pr_number=args.pr_number,
            confidence=args.confidence,
        )
        if args.json:
            print(json.dumps({"ok": True, "receipt": receipt.to_dict()}, indent=2))  # noqa: T201
        else:
            print(receipt.receipt_id)  # noqa: T201
        return 0

    if args.command == "heartbeat":
        lease = store.heartbeat_lease(
            lease_id=args.lease_id,
            ttl_hours=args.ttl_hours,
        )
        if args.json:
            print(json.dumps({"ok": True, "lease": lease.to_dict()}, indent=2))  # noqa: T201
        else:
            print(lease.lease_id)  # noqa: T201
        return 0

    if args.command == "decide":
        decision = store.record_integration_decision(
            receipt_id=args.receipt_id,
            decision=IntegrationDecisionType(args.decision),
            decided_by=args.decided_by,
            rationale=args.rationale,
            target_branch=args.target_branch,
            chosen_commits=args.commit,
            followups=args.follow_up,
            lease_id=args.lease_id,
        )
        if args.json:
            print(json.dumps({"ok": True, "decision": decision.to_dict()}, indent=2))  # noqa: T201
        else:
            print(decision.decision_id)  # noqa: T201
        return 0

    if args.command == "scan-salvage":
        items = store.scan_salvage_sources(
            include_worktrees=not args.no_worktrees,
            include_stashes=not args.no_stashes,
            max_stashes=args.max_stashes,
        )
        payload = {
            "ok": True,
            "count": len(items),
            "candidates": [item.to_dict() for item in items],
        }
        if args.json:
            print(json.dumps(payload, indent=2))  # noqa: T201
        else:
            print(payload["count"])  # noqa: T201
        return 0

    if args.command == "sync-queue":
        payload = {
            "ok": True,
            "counts": asyncio.run(store.sync_pending_work_queue()),
        }
        if args.json:
            print(json.dumps(payload, indent=2))  # noqa: T201
        else:
            print(json.dumps(payload["counts"], indent=2))  # noqa: T201
        return 0

    if args.command == "tasks":
        payload = {
            "ok": True,
            "count": 0,
            "tasks": [
                item.to_dict() for item in store.list_developer_tasks(open_only=args.open_only)
            ],
        }
        payload["count"] = len(payload["tasks"])
        if args.json:
            print(json.dumps(payload, indent=2))  # noqa: T201
        else:
            print(payload["count"])  # noqa: T201
        return 0

    if args.command == "sync-tasks":
        payload = {
            "ok": True,
            "counts": asyncio.run(store.sync_developer_task_queue()),
        }
        if args.json:
            print(json.dumps(payload, indent=2))  # noqa: T201
        else:
            print(json.dumps(payload["counts"], indent=2))  # noqa: T201
        return 0

    if args.command == "reap":
        expired = store.reap_expired_leases()
        stale = store.reap_stale_leases()
        all_reaped = expired + stale
        payload = {
            "ok": True,
            "count": len(all_reaped),
            "expired": len(expired),
            "stale": len(stale),
            "leases": [lease.to_dict() for lease in all_reaped],
        }
        if args.json:
            print(json.dumps(payload, indent=2))  # noqa: T201
        else:
            print(payload["count"])  # noqa: T201
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
