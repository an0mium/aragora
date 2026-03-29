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
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

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


def _get_lane_telemetry() -> LaneTelemetryCollector:
    global _LANE_TELEMETRY
    if _LANE_TELEMETRY is None:
        from aragora.swarm.lane_telemetry import LaneTelemetryCollector

        _LANE_TELEMETRY = LaneTelemetryCollector()
    return _LANE_TELEMETRY


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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
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
        return expired

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
                outcome=str(outcome or "completed").strip() or "completed",
                risks=list(risks or blockers or []),
                pr_url=str(pr_url or "").strip(),
                pr_number=pr_number,
                confidence=float(confidence),
                metadata={
                    **dict(metadata or {}),
                    "supervisor_run_id": lease_metadata.get("supervisor_run_id"),
                    "work_order_id": lease_metadata.get("work_order_id"),
                    "task_key": lease_metadata.get("task_key"),
                    "reviewer_agent": lease_metadata.get("reviewer_agent"),
                    "risk_level": lease_metadata.get("risk_level"),
                },
            )
            violations = self._validate_completion_scope(
                lease,
                changed_paths=receipt.changed_paths,
                owner_session_id=owner_session_id,
                branch=branch,
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
                time_to_pr_seconds=existing.time_to_pr_seconds if existing else None,
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
    for key in (
        "last_observed_at",
        "last_progress_at",
        "completed_at",
        "dispatched_at",
        "leased_at",
        "started_at",
    ):
        text = str(work_order.get(key, "")).strip()
        if text:
            return text
    return str(run.get("updated_at", _utcnow().isoformat()))


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
    return json.dumps(value, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _artifact_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
