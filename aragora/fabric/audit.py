"""
Audit trail for Agent Fabric decisions.

Provides durable logging and querying of policy decisions, budget checks,
and task execution events for compliance and debugging.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from aragora.fabric.models import PolicyEffect

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecisionLog:
    """Audit log for policy decisions."""

    timestamp: datetime
    trace_id: str
    correlation_id: str
    agent_id: str
    action: str
    context: dict[str, Any]
    effect: PolicyEffect
    rule_id: str | None
    reason: str


@dataclass
class BudgetDecisionLog:
    """Audit log for budget decisions."""

    timestamp: datetime
    trace_id: str
    correlation_id: str
    agent_id: str
    resource_type: str  # tokens, compute, cost
    requested: float
    available: float
    allowed: bool
    reason: str


@dataclass
class TaskExecutionLog:
    """Audit log for task execution events."""

    timestamp: datetime
    trace_id: str
    correlation_id: str
    agent_id: str
    task_id: str
    event_type: Literal["started", "completed", "failed", "cancelled", "timeout"]
    duration_seconds: float | None = None
    result_summary: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _serialize_log(log: PolicyDecisionLog | BudgetDecisionLog | TaskExecutionLog) -> dict[str, Any]:
    """Serialize a log entry to a JSON-compatible dict."""
    data = asdict(log)

    # Convert datetime to ISO format string
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()

    # Convert PolicyEffect enum to string
    if isinstance(data.get("effect"), PolicyEffect):
        data["effect"] = data["effect"].value
    elif "effect" in data and hasattr(data["effect"], "value"):
        data["effect"] = data["effect"].value

    # Add log type for deserialization
    if isinstance(log, PolicyDecisionLog):
        data["_log_type"] = "policy"
    elif isinstance(log, BudgetDecisionLog):
        data["_log_type"] = "budget"
    elif isinstance(log, TaskExecutionLog):
        data["_log_type"] = "task"

    return data


def _deserialize_log(
    data: dict[str, Any],
) -> PolicyDecisionLog | BudgetDecisionLog | TaskExecutionLog | None:
    """Deserialize a dict to a log entry."""
    log_type = data.pop("_log_type", None)

    # Convert ISO string back to datetime
    if "timestamp" in data and isinstance(data["timestamp"], str):
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

    if log_type == "policy":
        # Convert effect string back to PolicyEffect enum
        if "effect" in data and isinstance(data["effect"], str):
            data["effect"] = PolicyEffect(data["effect"])
        return PolicyDecisionLog(**data)
    elif log_type == "budget":
        return BudgetDecisionLog(**data)
    elif log_type == "task":
        return TaskExecutionLog(**data)

    return None


class AuditStore:
    """
    Persistent storage for audit logs.

    Features:
    - JSON file-based durable storage
    - Thread-safe write operations
    - Query by trace_id, agent_id, or time range
    - Automatic file rotation support
    """

    def __init__(self, audit_dir: Path | None = None) -> None:
        """
        Initialize the audit store.

        Args:
            audit_dir: Directory for audit log files. Defaults to ./audit_logs
        """
        self._audit_dir = audit_dir or Path("./audit_logs")
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._policy_file = self._audit_dir / "policy_decisions.jsonl"
        self._budget_file = self._audit_dir / "budget_decisions.jsonl"
        self._task_file = self._audit_dir / "task_events.jsonl"

    def log_policy_decision(self, log: PolicyDecisionLog) -> None:
        """
        Log a policy decision.

        Args:
            log: Policy decision log entry
        """
        self._append_log(self._policy_file, log)
        logger.debug(
            f"Policy decision logged: trace_id={log.trace_id}, "
            f"agent={log.agent_id}, effect={log.effect.value}"
        )

    def log_budget_decision(self, log: BudgetDecisionLog) -> None:
        """
        Log a budget decision.

        Args:
            log: Budget decision log entry
        """
        self._append_log(self._budget_file, log)
        logger.debug(
            f"Budget decision logged: trace_id={log.trace_id}, "
            f"agent={log.agent_id}, allowed={log.allowed}"
        )

    def log_task_event(self, log: TaskExecutionLog) -> None:
        """
        Log a task execution event.

        Args:
            log: Task execution log entry
        """
        self._append_log(self._task_file, log)
        logger.debug(
            f"Task event logged: trace_id={log.trace_id}, "
            f"task={log.task_id}, event={log.event_type}"
        )

    def _append_log(
        self,
        file_path: Path,
        log: PolicyDecisionLog | BudgetDecisionLog | TaskExecutionLog,
    ) -> None:
        """Append a log entry to a file."""
        data = _serialize_log(log)
        with self._lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")

    def _read_logs(self, file_path: Path) -> list[dict[str, Any]]:
        """Read all logs from a file."""
        logs = []
        if not file_path.exists():
            return logs

        with self._lock:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in audit log: {line[:100]}")
        return logs

    def query_by_trace_id(self, trace_id: str) -> list[dict[str, Any]]:
        """
        Query all audit logs by trace ID.

        Args:
            trace_id: The trace ID to search for

        Returns:
            List of matching log entries
        """
        results = []

        for file_path in [self._policy_file, self._budget_file, self._task_file]:
            for log in self._read_logs(file_path):
                if log.get("trace_id") == trace_id:
                    results.append(log)

        # Sort by timestamp
        results.sort(key=lambda x: x.get("timestamp", ""))
        return results

    def query_by_agent_id(self, agent_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        Query all audit logs by agent ID.

        Args:
            agent_id: The agent ID to search for
            limit: Maximum number of results to return

        Returns:
            List of matching log entries (most recent first)
        """
        results = []

        for file_path in [self._policy_file, self._budget_file, self._task_file]:
            for log in self._read_logs(file_path):
                if log.get("agent_id") == agent_id:
                    results.append(log)

        # Sort by timestamp descending (most recent first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]

    def query_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        log_type: Literal["policy", "budget", "task", "all"] = "all",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Query audit logs within a time range.

        Args:
            start_time: Start of the time range (inclusive)
            end_time: End of the time range (inclusive)
            log_type: Type of logs to query, or "all" for all types
            limit: Maximum number of results to return

        Returns:
            List of matching log entries (oldest first)
        """
        results = []
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()

        files_to_query = []
        if log_type in ("policy", "all"):
            files_to_query.append(self._policy_file)
        if log_type in ("budget", "all"):
            files_to_query.append(self._budget_file)
        if log_type in ("task", "all"):
            files_to_query.append(self._task_file)

        for file_path in files_to_query:
            for log in self._read_logs(file_path):
                timestamp = log.get("timestamp", "")
                if start_iso <= timestamp <= end_iso:
                    results.append(log)

        # Sort by timestamp ascending (oldest first)
        results.sort(key=lambda x: x.get("timestamp", ""))
        return results[:limit]

    def query_policy_decisions(
        self,
        agent_id: str | None = None,
        effect: PolicyEffect | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query policy decision logs with filters.

        Args:
            agent_id: Filter by agent ID
            effect: Filter by policy effect
            limit: Maximum number of results to return

        Returns:
            List of matching log entries (most recent first)
        """
        results = []

        for log in self._read_logs(self._policy_file):
            if agent_id and log.get("agent_id") != agent_id:
                continue
            if effect and log.get("effect") != effect.value:
                continue
            results.append(log)

        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]

    def query_budget_decisions(
        self,
        agent_id: str | None = None,
        allowed: bool | None = None,
        resource_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query budget decision logs with filters.

        Args:
            agent_id: Filter by agent ID
            allowed: Filter by allowed status
            resource_type: Filter by resource type (tokens, compute, cost)
            limit: Maximum number of results to return

        Returns:
            List of matching log entries (most recent first)
        """
        results = []

        for log in self._read_logs(self._budget_file):
            if agent_id and log.get("agent_id") != agent_id:
                continue
            if allowed is not None and log.get("allowed") != allowed:
                continue
            if resource_type and log.get("resource_type") != resource_type:
                continue
            results.append(log)

        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]

    def query_task_events(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query task execution logs with filters.

        Args:
            agent_id: Filter by agent ID
            task_id: Filter by task ID
            event_type: Filter by event type
            limit: Maximum number of results to return

        Returns:
            List of matching log entries (most recent first)
        """
        results = []

        for log in self._read_logs(self._task_file):
            if agent_id and log.get("agent_id") != agent_id:
                continue
            if task_id and log.get("task_id") != task_id:
                continue
            if event_type and log.get("event_type") != event_type:
                continue
            results.append(log)

        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]

    def get_stats(self) -> dict[str, Any]:
        """
        Get audit store statistics.

        Returns:
            Dictionary with counts and file sizes
        """
        policy_logs = self._read_logs(self._policy_file)
        budget_logs = self._read_logs(self._budget_file)
        task_logs = self._read_logs(self._task_file)

        return {
            "policy_decisions_count": len(policy_logs),
            "budget_decisions_count": len(budget_logs),
            "task_events_count": len(task_logs),
            "total_logs": len(policy_logs) + len(budget_logs) + len(task_logs),
            "policy_file_size_bytes": self._policy_file.stat().st_size
            if self._policy_file.exists()
            else 0,
            "budget_file_size_bytes": self._budget_file.stat().st_size
            if self._budget_file.exists()
            else 0,
            "task_file_size_bytes": self._task_file.stat().st_size
            if self._task_file.exists()
            else 0,
        }

    def clear(self) -> None:
        """
        Clear all audit logs.

        Warning: This permanently deletes all audit data.
        """
        with self._lock:
            for file_path in [self._policy_file, self._budget_file, self._task_file]:
                if file_path.exists():
                    file_path.unlink()
        logger.warning("Audit logs cleared")
