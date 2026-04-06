"""Worker process data types and constants for supervised swarm runs.

Extracted from ``worker_launcher.py`` to keep data definitions separate from
launch orchestration logic.  Everything exported here is re-exported from
``worker_launcher`` for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from aragora.pipeline.execution_mode import ExecutionMode

UTC = timezone.utc
MAX_WORKER_LOG_TAIL_CHARS = 4000

# Session artifacts that autonomous workers should never treat as deliverable
# output.  These are infrastructure metadata files created by the harness, not
# user work product.  They must be stripped from changed_paths before any
# result is qualified.
SESSION_ARTIFACTS: frozenset[str] = frozenset(
    {
        ".codex_session_meta.json",
        ".codex_session.log",
        ".codex_session_active",
        ".swarm_worker_stdout.log",
        ".swarm_worker_stderr.log",
    }
)

# Runtime dependency directories can appear in worker status output when the
# harness reuses local frontend tooling state. They are environment noise, not
# lane deliverables.
IGNORED_RUNTIME_PATH_PARTS: frozenset[str] = frozenset({"node_modules"})


def is_ignored_changed_path(path: str) -> bool:
    """Return True when a changed path is harness/runtime noise."""
    clean = str(path).strip().removeprefix("./").rstrip("/")
    if not clean:
        return False
    pure_path = PurePosixPath(clean)
    if pure_path.name in SESSION_ARTIFACTS:
        return True
    return any(part in IGNORED_RUNTIME_PATH_PARTS for part in pure_path.parts)


# Exit codes where the worker likely completed its work but the process was
# terminated by a transport-level signal (e.g. broken pipe). Only these codes
# are eligible for salvage. Other non-zero exits must preserve raw exit truth
# even if they left behind a recoverable artifact.
_SALVAGEABLE_EXIT_CODES: frozenset[int] = frozenset(
    {
        1,  # Generic error — worker may have produced partial work
        2,  # Misuse of shell builtins
        130,  # SIGINT — Ctrl-C, worker may have committed before interrupt
        137,  # SIGKILL — force-killed, check for commits
        141,  # SIGPIPE — stdout pipe closed before process finished writing
        143,  # SIGTERM — graceful termination, worker may have committed
    }
)

DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 900.0


@dataclass(slots=True)
class WorkerProcess:
    """Tracks a running worker subprocess."""

    work_order_id: str
    agent: str
    worktree_path: str
    branch: str
    pid: int | None = None
    session_id: str = ""
    lease_id: str = ""
    receipt_id: str = ""
    approval_id: str = ""
    git_write_approval_id: str = ""
    push_approval_id: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    diff: str = ""
    initial_head: str = ""
    head_sha: str = ""
    commit_shas: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    expected_tests: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    dispatch_action_id: str = ""
    push_action_id: str = ""
    admin_approved: bool = False

    @property
    def is_running(self) -> bool:
        return self.exit_code is None and self.pid is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_order_id": self.work_order_id,
            "agent": self.agent,
            "worktree_path": self.worktree_path,
            "branch": self.branch,
            "pid": self.pid,
            "session_id": self.session_id,
            "lease_id": self.lease_id,
            "receipt_id": self.receipt_id,
            "approval_id": self.approval_id,
            "git_write_approval_id": self.git_write_approval_id,
            "push_approval_id": self.push_approval_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exit_code": self.exit_code,
            "head_sha": self.head_sha,
            "commit_shas": list(self.commit_shas),
            "changed_paths": list(self.changed_paths),
            "expected_tests": list(self.expected_tests),
            "tests_run": list(self.tests_run),
            "verification_results": [dict(item) for item in self.verification_results],
            "dispatch_action_id": self.dispatch_action_id,
            "push_action_id": self.push_action_id,
            "admin_approved": self.admin_approved,
        }


@dataclass(slots=True)
class LaunchConfig:
    """Configuration for worker launches."""

    claude_path: str = "claude"
    codex_path: str = "codex"
    timeout_seconds: float = 2400.0
    no_progress_timeout_seconds: float = 3600.0  # 60 min — large repos need context loading time
    claude_model: str | None = None
    codex_model: str | None = None
    claude_profile: str | None = None
    claude_profile_script: str | None = None
    auto_commit: bool = True
    use_managed_session_script: bool = True
    base_branch: str = "main"
    detach: bool = False
    require_explicit_approval: bool = True
    # Security: dangerous CLI flags are OFF by default (Crux 1 fix).
    allow_claude_dangerously_skip_permissions: bool = False
    allow_codex_full_auto: bool = False
    execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS

    def __post_init__(self) -> None:
        """Validate that dangerous flags are only used in AUTONOMOUS mode."""
        if self.execution_mode != ExecutionMode.AUTONOMOUS:
            if self.allow_claude_dangerously_skip_permissions:
                raise ValueError(
                    "allow_claude_dangerously_skip_permissions requires execution_mode=AUTONOMOUS"
                )
            if self.allow_codex_full_auto:
                raise ValueError("allow_codex_full_auto requires execution_mode=AUTONOMOUS")
