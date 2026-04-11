"""Long-running Boss loop MVP: GitHub-issue-backed task feed with runner freshness.

Pulls candidate work from GitHub issues, selects one eligible task at a time,
requires fresh eligible runners, runs supervised worker execution with bounded
retries, and emits periodic status reports with truthful stop conditions.

Usage:
    loop = BossLoop(config=BossLoopConfig(max_iterations=20))
    final_status = await loop.run()
    print(json.dumps(final_status.to_dict(), indent=2))
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.pipeline.execution_mode import ExecutionMode
from aragora.swarm.debate_gate import DebateGate, DebateGateConfig, DebateGateRequest
from aragora.swarm.env_utils import git_safe_env
from aragora.swarm.terminal_truth import (
    extract_run_deliverable,
    extract_run_worker_outcome,
    qualify_work_order_terminal_state,
    qualify_run_terminal_state,
)
from aragora.swarm.lane_telemetry import LaneTelemetryCollector, LaneTelemetryRecord

# Backwards-compatible re-exports from extracted modules
from aragora.swarm.boss_feed import (  # noqa: F401
    GitHubIssue,
    GitHubIssueFeed,
    IssueEligibilityReport,
    build_issue_eligibility_report,
    fetch_open_pr_changed_paths,
    infer_issue_lane_hints,
    infer_issue_scope_entries,
    select_eligible_issue,
    scope_entries_overlap,
)
from aragora.swarm.boss_freshness import RunnerFreshnessResult, check_runner_freshness  # noqa: F401
from aragora.swarm.boss_validation import (  # noqa: F401
    _compose_issue_dispatch_goal,
    _should_replace_with_focused_tests,
    check_pre_dispatch_gate,
    sanitize_issue_body_for_dispatch,
    extract_issue_validation_contract,
    extract_pre_dispatch_validation_commands,
    extract_declared_new_file_paths,
    find_missing_pre_dispatch_validation_targets,
    run_pre_dispatch_validation_commands,
    discover_focused_tests,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc
_LANE_TELEMETRY = LaneTelemetryCollector()
_GITHUB_ISSUE_URL_RE = re.compile(r"github\.com/(?P<repo>[^/]+/[^/]+)/issues/(?P<number>\d+)")
_GITHUB_PR_URL_RE = re.compile(r"github\.com/[^/]+/[^/]+/pull/(?P<number>\d+)")

# The 5 CI checks required for merge.  Draft PRs run only these;
# ready PRs run the full 32-workflow suite.  The boss loop creates
# PRs as drafts and auto-promotes them once these 5 pass.
_REQUIRED_CHECK_NAMES: frozenset[str] = frozenset(
    {
        "lint",
        "typecheck",
        "sdk-parity",
        "Generate & Validate",
        "TypeScript SDK Type Check",
    }
)
_ALREADY_DONE_MARKERS = (
    "already implemented",
    "already exists",
    "no changes needed",
    "no code changes needed",
    "nothing to commit",
    "there's nothing to commit",
)
_BOSS_PUBLISH_COMMENT_MARKER = "<!-- aragora-boss-loop-publish -->"


def _strict_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------------------
# Boss Loop Status & Stop Conditions
# ---------------------------------------------------------------------------


class BossStopReason(str, Enum):
    """Why the Boss loop stopped."""

    MAX_ITERATIONS = "max_iterations"
    NO_FRESH_RUNNER = "no_fresh_runner"
    NO_SUITABLE_ISSUE = "no_suitable_issue"
    WORKER_FAILED = "worker_failed"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    NEEDS_HUMAN = "needs_human"
    MANUAL_STOP = "manual_stop"
    ISSUE_FEED_ERROR = "issue_feed_error"
    AUTO_UPDATE = "auto_update"
    STILL_RUNNING = "still_running"


@dataclass
class BossIterationStatus:
    """Status payload for a single Boss loop iteration."""

    iteration: int
    run_id: str
    timestamp: str
    runner_freshness: dict[str, Any]
    selected_issue: dict[str, Any] | None
    worker_status: str
    stop_reason: str | None
    needs_human_reasons: list[str]
    next_actions: list[str]
    elapsed_seconds: float = 0.0
    error: str | None = None
    worker_outcome: str | None = None
    configured_max_parallel_dispatches: int | None = None
    effective_parallel_dispatches: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "iteration": self.iteration,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "runner_freshness": dict(self.runner_freshness),
            "selected_issue": dict(self.selected_issue) if self.selected_issue else None,
            "worker_status": self.worker_status,
            "stop_reason": self.stop_reason,
            "needs_human_reasons": list(self.needs_human_reasons),
            "next_actions": list(self.next_actions),
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "configured_max_parallel_dispatches": self.configured_max_parallel_dispatches,
            "effective_parallel_dispatches": self.effective_parallel_dispatches,
        }
        if self.worker_outcome is not None:
            result["worker_outcome"] = self.worker_outcome
        return result


@dataclass
class BossLoopResult:
    """Final result of a Boss loop run."""

    run_id: str
    iterations_completed: int
    total_elapsed_seconds: float
    stop_reason: str
    issues_attempted: list[dict[str, Any]]
    issues_completed: list[dict[str, Any]]
    issues_failed: list[dict[str, Any]]
    iteration_statuses: list[dict[str, Any]]
    needs_human_reasons: list[str]
    next_actions: list[str]
    sanitation_summary: list[str] = field(default_factory=list)
    configured_max_parallel_dispatches: int = 1
    effective_parallel_dispatches_observed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "boss-loop",
            "run_id": self.run_id,
            "iterations_completed": self.iterations_completed,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "stop_reason": self.stop_reason,
            "issues_attempted": list(self.issues_attempted),
            "issues_completed": list(self.issues_completed),
            "issues_failed": list(self.issues_failed),
            "iteration_statuses": list(self.iteration_statuses),
            "needs_human_reasons": list(self.needs_human_reasons),
            "next_actions": list(self.next_actions),
            "sanitation_summary": list(self.sanitation_summary),
            "configured_max_parallel_dispatches": self.configured_max_parallel_dispatches,
            "effective_parallel_dispatches_observed": self.effective_parallel_dispatches_observed,
        }


@dataclass(slots=True)
class _BossDeliverableArtifact:
    """Minimal artifact wrapper for boss-loop PR publication."""

    metadata: dict[str, Any]
    branch: str | None = None
    urls: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Boss Loop Config
# ---------------------------------------------------------------------------


@dataclass
class BossLoopConfig:
    """Configuration for the long-running Boss loop."""

    # Iteration bounds
    max_iterations: int = 50
    iteration_interval_seconds: float = 30.0
    auto_update_enabled: bool = False
    auto_update_interval_iterations: int = 10

    # Runner freshness
    freshness_ttl_seconds: float = 3600.0  # 1 hour
    registry_path: str | None = None

    # Issue feed
    repo: str | None = None
    label_filter: str | None = None
    issue_number: int | None = None
    issue_numbers: list[int] | None = None
    issue_limit: int = 25
    skip_labels: set[str] = field(
        default_factory=lambda: {"wontfix", "duplicate", "invalid", "boss-stuck"}
    )
    require_labels: set[str] | None = None
    require_validation_contract: bool = True

    # Retry / self-correction
    max_consecutive_failures: int = 3
    max_retries_per_issue: int = 3  # initial attempt + 1 ping-pong + 1 repair

    # Dispatch
    target_branch: str = "main"
    budget_limit_usd: float = 5.0
    dispatch_enabled: bool = True
    default_target_agent: str | None = None
    model_rotation: list[str] = field(default_factory=lambda: ["claude", "codex"])
    default_reviewer_agent: str | None = None
    allowed_runner_profiles: set[str] | None = None
    runner_rotation_interval_seconds: float = 1800.0
    verified_runner_target: int | None = None
    runner_probe_limit: int | None = None
    dispatch_max_ticks: int = 720
    max_parallel_dispatches: int = 1

    # Autonomy: when True, treat needs_human with a deliverable as completed
    # instead of stopping the loop. Only stop when there's genuinely no output.
    auto_continue_on_needs_human: bool = False

    # Ping-pong: when a worker hits needs_human with no deliverable but a
    # non-trivial transcript, retry with the OTHER agent type using a
    # structured handoff prompt from the failed agent's output.
    enable_ping_pong_retry: bool = False

    # Fix-forward: max repair attempts when verification fails.
    # Each repair dispatches a targeted fix task using only the failing test output.
    max_repair_attempts: int = 2

    # Verification: use focused tests (only files touched by the worker) instead
    # of the full test suite.  Dramatically reduces false negatives from
    # pre-existing failures in unrelated modules.
    use_focused_verification: bool = True

    # Value-per-cost ranking: when True, rank eligible issues by estimated
    # value/cost before selecting.  This pushes the loop toward high-leverage
    # work instead of processing issues in arbitrary GitHub order.
    use_value_ranking: bool = True

    # Scope conflict guardrail: skip issues whose inferred file scope already
    # appears in an open PR or another issue selected in the same batch.
    avoid_open_pr_scope_conflicts: bool = True

    # Micro-decomposition: break broad issues into single-file work orders.
    # Workers succeed on focused tasks but timeout on broad ones in large repos.
    use_micro_decomposition: bool = True

    # Pre-dispatch gate: optionally use LLM parsing for issue bodies that do
    # not fit the deterministic regex contract. Disabled by default to avoid
    # hidden network calls, provider costs, and nondeterministic dispatch gates.
    use_llm_pre_dispatch_gate: bool = False

    # Security: opt-in flags for dangerous worker CLI behavior (Crux 1).
    allow_claude_dangerously_skip_permissions: bool = False
    allow_codex_full_auto: bool = False
    execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS

    # Autonomous post-processing: publish verified branch deliverables and
    # optionally close already-resolved no-op issues.
    auto_publish_deliverables: bool = False
    use_debate_publish_gate: bool = False
    debate_publish_gate_fail_closed: bool = False
    debate_publish_gate_agent: str | None = None
    debate_publish_gate_timeout_seconds: float = 90.0
    max_open_auto_publish_prs: int = 4
    auto_close_already_done_issues: bool = False

    # Decomposition guardrails
    max_decomposition_depth: int = 3
    max_total_sub_issues_per_run: int = 15
    max_decomposed_issue_ticks: int = 30

    # Fast-fail circuit breaker: if N consecutive iterations complete in under
    # threshold_seconds, skip decomposed issues and log a warning.
    fast_fail_circuit_breaker_window: int = 5
    fast_fail_threshold_seconds: float = 30.0

    # Reporting
    status_report_interval: int = 5  # every N iterations
    metrics_jsonl_path: str | None = ".aragora/overnight/boss_metrics.jsonl"


# ---------------------------------------------------------------------------
# Boss Loop
# ---------------------------------------------------------------------------


def _classify_terminal_run_outcome(run_dict: dict[str, Any]) -> str:
    """Map a supervisor run dict to a stable, shared terminal outcome."""
    return qualify_run_terminal_state(run_dict).terminal_outcome


def _qualify_worker_result_terminal_state(worker_result: dict[str, Any]) -> tuple[str, str]:
    """Normalize legacy flat worker_result payloads into canonical terminal truth."""
    issue_resolution = worker_result.get("issue_resolution")
    if (
        isinstance(issue_resolution, dict)
        and str(issue_resolution.get("action", "")).strip() == "closed"
    ):
        return "issue_already_resolved", ""
    deliverable = worker_result.get("deliverable")
    adapted: dict[str, Any] = {
        "status": worker_result.get("status"),
        "worker_outcome": worker_result.get("worker_outcome"),
        "failure_reason": worker_result.get("error"),
        "blockers": list(worker_result.get("reasons", []) or []),
    }
    if isinstance(deliverable, dict):
        deliverable_type = str(deliverable.get("type", "")).strip().lower()
        if deliverable_type == "branch":
            adapted["branch"] = deliverable.get("branch")
            adapted["commit_shas"] = deliverable.get("commit_shas") or []
        elif deliverable_type == "pr":
            adapted["pr_url"] = deliverable.get("pr_url") or worker_result.get("pr_url")
        elif deliverable_type == "adopted_pr":
            adapted["adopted_pr"] = (
                deliverable.get("adopted_pr")
                or deliverable.get("pr_url")
                or worker_result.get("pr_url")
            )
    qualification = qualify_work_order_terminal_state(adapted)
    return qualification.terminal_outcome, qualification.deliverable_type or ""


def _freshness_to_dict(freshness: Any) -> dict[str, Any]:
    """Best-effort conversion for custom freshness checker payloads."""
    if isinstance(freshness, RunnerFreshnessResult):
        return freshness.to_dict()
    to_dict = getattr(freshness, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    if isinstance(freshness, dict):
        return dict(freshness)
    return {}


def _freshness_is_fresh(freshness: Any, freshness_dict: dict[str, Any]) -> bool:
    """Require a real boolean freshness signal before dispatching work."""
    if hasattr(freshness, "fresh"):
        return getattr(freshness, "fresh") is True
    return freshness_dict.get("fresh") is True


async def dispatch_bounded_spec(
    spec: Any,
    *,
    target_branch: str = "main",
    budget_limit_usd: float = 5.0,
    max_ticks: int = 360,
    wait_for_completion: bool = True,
    repo_path: Any | None = None,
    default_target_agent: str | None = None,
    default_reviewer_agent: str | None = None,
    use_managed_session_script: bool = True,
    selected_runner: dict[str, Any] | None = None,
    worker_env: dict[str, str] | None = None,
    allow_claude_dangerously_skip_permissions: bool = False,
    allow_codex_full_auto: bool = False,
    execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS,
) -> dict[str, Any]:
    # Auto-detect Claude profile from environment if no runner specified
    if selected_runner is None:
        profile = os.environ.get("ARAGORA_CLAUDE_PROFILE", "").strip()
        if profile:
            repo_root = repo_path or Path.cwd()
            selected_runner = {
                "runner_type": "claude",
                "profile": profile,
                "command_path": str(Path(repo_root) / "scripts" / "claude_profile.sh"),
                "cost_class": "subscription",
            }
            logger.info("Using Claude profile %r from ARAGORA_CLAUDE_PROFILE", profile)
    """Dispatch one bounded spec via the supervisor-backed Boss path.

    This reuses the Boss loop's concrete-deliverable gate so higher-level
    orchestrators do not implement their own divergent run classification.
    """
    from aragora.swarm.commander import SwarmCommander
    from aragora.swarm.config import SwarmCommanderConfig
    from aragora.swarm.supervisor import SwarmApprovalPolicy

    if not spec.is_dispatch_bounded():
        return {
            "status": "failed",
            "outcome": "blocked",
            "error": spec.dispatch_gate_reason(),
        }

    try:
        config = SwarmCommanderConfig(
            budget_limit_usd=budget_limit_usd,
            require_approval=True,
        )
        commander = SwarmCommander(config=config)
        run = await commander.run_supervised_from_spec(
            spec,
            repo_path=repo_path,
            target_branch=target_branch,
            max_concurrency=1,
            approval_policy=SwarmApprovalPolicy(
                require_merge_approval=True,
                require_external_action_approval=True,
            ),
            dispatch=True,
            wait=wait_for_completion,
            interval_seconds=5.0,
            max_ticks=max_ticks,
            force_collect_on_max_ticks=True,
            default_target_agent=default_target_agent,
            default_reviewer_agent=default_reviewer_agent,
            use_managed_session_script=use_managed_session_script,
            default_target_runner=selected_runner,
            worker_env=worker_env,
            allow_claude_dangerously_skip_permissions=allow_claude_dangerously_skip_permissions,
            allow_codex_full_auto=allow_codex_full_auto,
            execution_mode=execution_mode,
        )
        run_dict = run.to_dict()
        run_status = str(run_dict.get("status", "")).strip().lower()

        # --- Diagnostic: trace work order results ---
        work_orders = run_dict.get("work_orders", [])
        for wo in work_orders:
            wo_id = str(wo.get("work_order_id", ""))[:30]
            wo_status = wo.get("status")
            wo_exit = wo.get("exit_code")
            wo_commits = len(wo.get("commit_shas", []))
            wo_changed = len(wo.get("changed_paths", []))
            wo_pid = wo.get("pid")
            wo_wt = str(wo.get("worktree_path", ""))[-50:]
            logger.info(
                "dispatch_bounded_spec work_order %s: status=%s exit=%s commits=%d "
                "changed=%d pid=%s worktree=...%s",
                wo_id,
                wo_status,
                wo_exit,
                wo_commits,
                wo_changed,
                wo_pid,
                wo_wt,
            )

        if not wait_for_completion and run_status not in {"completed", "needs_human"}:
            return {
                "status": "running",
                "outcome": "dispatched",
                "run": run_dict,
                "run_id": run_dict.get("run_id"),
            }
        qualification = qualify_run_terminal_state(run_dict)
        outcome = qualification.terminal_outcome
        deliverable = qualification.deliverable
        reasons = qualification.reasons or (
            [qualification.blocked_reason] if qualification.blocked_reason else []
        )
        logger.info(
            "dispatch_bounded_spec terminal: outcome=%s deliverable=%s "
            "blocked_reason=%s run_status=%s",
            outcome,
            bool(deliverable),
            qualification.blocked_reason,
            run_status,
        )
        worker_receipt_id = _first_receipt_id_from_run(run_dict)
        if outcome in {"deliverable_created", "pr_adopted"}:
            return {
                "status": "completed",
                "outcome": outcome,
                "run": run_dict,
                "run_id": run_dict.get("run_id"),
                "deliverable": deliverable,
                "receipt_id": worker_receipt_id,
            }
        if outcome == "clean_exit_no_deliverable":
            return {
                "status": "needs_human",
                "outcome": outcome,
                "run": run_dict,
                "run_id": run_dict.get("run_id"),
                "deliverable": deliverable,
                "receipt_id": worker_receipt_id,
                "reasons": reasons
                or [
                    "Run reported completed but produced no concrete deliverable "
                    "(no pushed branch, no PR, no committed artifact)."
                ],
            }
        if outcome in {"needs_human", "blocked", "crash", "timeout"}:
            return {
                "status": "needs_human",
                "outcome": outcome,
                "run": run_dict,
                "run_id": run_dict.get("run_id"),
                "deliverable": deliverable,
                "receipt_id": worker_receipt_id,
                "reasons": reasons
                or [
                    qualification.blocked_reason
                    or "Worker requires human review before integration."
                ],
            }
        return {
            "status": "failed",
            "outcome": outcome,
            "run": run_dict,
            "run_id": run_dict.get("run_id"),
            "error": f"Run ended with status: {run_dict.get('status', '')}",
        }
    except ValueError as exc:
        return {"status": "failed", "outcome": "blocked", "error": str(exc)}
    except Exception as exc:
        logger.warning("Bounded spec dispatch failed: %s", exc)
        return {"status": "failed", "outcome": "crash", "error": str(exc)}


def _extract_deliverable(run_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first concrete deliverable on the run, if any."""
    return extract_run_deliverable(run_dict)


def _extract_worker_outcome(run_dict: dict[str, Any]) -> str | None:
    """Extract the first non-empty ``worker_outcome`` from a run."""
    return extract_run_worker_outcome(run_dict)


def _first_receipt_id_from_run(run_dict: dict[str, Any]) -> str | None:
    for work_order in run_dict.get("work_orders", []):
        if not isinstance(work_order, dict):
            continue
        receipt_id = str(work_order.get("receipt_id", "")).strip()
        if receipt_id:
            return receipt_id
    return None


def _backbone_dispatch_status(result: dict[str, Any]) -> str:
    """Preserve the dispatch status when mirroring it into the backbone ledger."""
    status = str(result.get("status", "")).strip().lower()
    return status or "failed"


def _dispatch_result_started(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    run_id = result.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return True
    run = result.get("run")
    if not isinstance(run, dict):
        return False
    embedded_run_id = run.get("run_id")
    if isinstance(embedded_run_id, str) and embedded_run_id.strip():
        return True
    work_orders = run.get("work_orders")
    if isinstance(work_orders, list):
        return True
    run_status = run.get("status")
    return isinstance(run_status, str) and bool(run_status.strip())


class BossLoop:
    """Long-running Boss loop: pull issues, check freshness, dispatch, report.

    The loop is bounded by ``max_iterations`` and stops truthfully when:
    - No fresh runner is available
    - No suitable issue exists in the feed
    - Consecutive worker failures exceed the threshold
    - A worker hits a needs-human condition
    - Max iterations reached

    Each iteration emits a ``BossIterationStatus`` suitable for JSON logging
    or machine-readable output.
    """

    def __init__(
        self,
        config: BossLoopConfig | None = None,
        *,
        issue_feed: GitHubIssueFeed | None = None,
        freshness_checker: Any | None = None,
        env: dict[str, str] | None = None,
        exec_argv: list[str] | None = None,
    ) -> None:
        self.config = config or BossLoopConfig()
        self.run_id = f"boss-{uuid.uuid4().hex[:12]}"
        self._configured_parallel_dispatches = max(1, int(self.config.max_parallel_dispatches or 1))
        self._current_effective_parallel_dispatches: int | None = None
        self._max_effective_parallel_dispatches_observed: int | None = None
        self._feed = issue_feed or GitHubIssueFeed(
            repo=self.config.repo,
            label_filter=self.config.label_filter,
            issue_numbers=self.config.issue_numbers,
            limit=self.config.issue_limit,
        )
        self._freshness_checker = freshness_checker or check_runner_freshness
        self._env = env
        self._exec_argv = exec_argv
        self._attempted_issues: list[dict[str, Any]] = []
        self._completed_issues: list[dict[str, Any]] = []
        self._failed_issues: list[dict[str, Any]] = []
        self._iteration_statuses: list[BossIterationStatus] = []
        self._consecutive_failures = 0
        self._issue_attempt_counts: dict[int | str, int] = {}
        self._pending_handoff_prompts: dict[int, tuple[str, str | None]] = {}
        self._stop_reason: str | None = None
        self._last_sanitation_summary: list[str] = []
        # Decomposition guardrails
        self._total_sub_issues_created: int = 0
        self._ticks_spent_on_decomposed_issues: int = 0
        # Fast-fail circuit breaker
        self._recent_elapsed: list[float] = []
        # Deferred publish retry queue: (issue, worker_result) pairs
        self._deferred_publish_queue: list[tuple[Any, dict[str, Any]]] = []

    def _git_cmd(self, args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            env=git_safe_env(self._env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    def _git_rev_parse(self, ref: str) -> str | None:
        try:
            result = self._git_cmd(["rev-parse", ref], timeout=10.0)
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def _git_is_clean(self) -> bool:
        try:
            result = self._git_cmd(["status", "--porcelain"], timeout=10.0)
        except subprocess.TimeoutExpired:
            return False
        if result.returncode != 0:
            return False
        return not bool(result.stdout.strip())

    def _git_is_ancestor(self, ancestor: str, descendant: str) -> bool:
        try:
            result = self._git_cmd(["merge-base", "--is-ancestor", ancestor, descendant])
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0

    def _restart_after_update(self) -> None:
        if not self._exec_argv:
            self._stop_reason = BossStopReason.AUTO_UPDATE.value
            return
        logger.info("Boss loop auto-update: restarting with %s", " ".join(self._exec_argv))
        os.execv(sys.executable, [sys.executable, *self._exec_argv])

    def _maybe_auto_update(self, iteration: int) -> bool:
        if not self.config.auto_update_enabled:
            return False
        interval = max(1, int(self.config.auto_update_interval_iterations or 1))
        if iteration % interval != 0:
            return False
        if not (Path.cwd() / ".git").exists():
            logger.debug("Boss loop auto-update skipped: no git repo detected.")
            return False
        if not self._git_is_clean():
            logger.info("Boss loop auto-update skipped: working tree is dirty.")
            return False
        target_branch = self.config.target_branch or "main"
        fetch = self._git_cmd(["fetch", "--no-tags", "origin", target_branch], timeout=60.0)
        if fetch.returncode != 0:
            logger.warning("Boss loop auto-update fetch failed: %s", fetch.stderr.strip())
            return False
        local_head = self._git_rev_parse("HEAD")
        remote_head = self._git_rev_parse(f"origin/{target_branch}")
        if not local_head or not remote_head or local_head == remote_head:
            return False
        if not self._git_is_ancestor(local_head, remote_head):
            logger.warning(
                "Boss loop auto-update skipped: local HEAD diverged from origin/%s.",
                target_branch,
            )
            return False
        merge = self._git_cmd(["merge", "--ff-only", remote_head], timeout=60.0)
        if merge.returncode != 0:
            logger.warning("Boss loop auto-update merge failed: %s", merge.stderr.strip())
            return False
        logger.info("Boss loop auto-updated to %s; restarting.", remote_head)
        self._restart_after_update()
        return True

    def _decorate_iteration_status(
        self,
        status: BossIterationStatus,
        *,
        effective_parallel_dispatches: int | None = None,
    ) -> BossIterationStatus:
        if status.configured_max_parallel_dispatches is None:
            status.configured_max_parallel_dispatches = self._configured_parallel_dispatches
        effective = effective_parallel_dispatches
        if effective is None:
            effective = self._current_effective_parallel_dispatches
        if status.effective_parallel_dispatches is None:
            status.effective_parallel_dispatches = effective
        if status.effective_parallel_dispatches is not None:
            self._max_effective_parallel_dispatches_observed = max(
                int(status.effective_parallel_dispatches),
                int(self._max_effective_parallel_dispatches_observed or 0),
            )
        return status

    @staticmethod
    def _issue_payload(issue: GitHubIssue) -> dict[str, Any]:
        payload = issue.to_dict()
        lane_hints = infer_issue_lane_hints(issue)
        if lane_hints:
            payload["lane_hints"] = list(lane_hints)
            if len(lane_hints) == 1:
                payload["lane_id"] = lane_hints[0]
        return payload

    def _blocked_issue_scopes(self) -> set[str]:
        if not self.config.avoid_open_pr_scope_conflicts:
            return set()
        blocked = self._coordination_blocked_scopes()
        repo = str(self.config.repo).strip() if isinstance(self.config.repo, str) else ""
        if not repo:
            feed_repo = getattr(self._feed, "repo", None)
            repo = str(feed_repo).strip() if isinstance(feed_repo, str) else ""
        if not repo:
            return blocked
        blocked.update(fetch_open_pr_changed_paths(repo=repo))
        return blocked

    def _coordination_blocked_scopes(self) -> set[str]:
        blocked: set[str] = set()
        try:
            store = DevCoordinationStore(repo_root=Path.cwd().resolve())
        except Exception:
            logger.debug(
                "Failed to open coordination store for boss-loop scope blocking", exc_info=True
            )
            return blocked

        try:
            for lease in store.list_active_leases():
                blocked.update(
                    str(path).strip()
                    for path in [*lease.claimed_paths, *lease.allowed_globs]
                    if str(path).strip()
                )
        except Exception:
            logger.debug("Failed to collect active lease scope claims", exc_info=True)

        try:
            store.fleet_store.reap_stale_claims()
            blocked.update(
                str(claim.get("path", "")).strip()
                for claim in store.fleet_store.list_claims()
                if isinstance(claim, dict) and str(claim.get("path", "")).strip()
            )
        except Exception:
            logger.debug("Failed to collect active fleet claim scopes", exc_info=True)

        return blocked

    def _extract_iteration_metrics(self, worker_result: dict[str, Any]) -> tuple[int, int, int]:
        """Summarize changed files and test verification from a worker run."""
        run_dict = worker_result.get("run")
        if not isinstance(run_dict, dict):
            return 0, 0, 0

        changed_files: list[str] = []
        tests_run: list[str] = []
        tests_passed = 0
        saw_verification_results = False

        for work_order in run_dict.get("work_orders", []):
            if not isinstance(work_order, dict):
                continue

            changed_files.extend(
                str(path).strip()
                for path in work_order.get("changed_paths", [])
                if str(path).strip()
            )
            tests_run.extend(
                str(command).strip()
                for command in work_order.get("tests_run", [])
                if str(command).strip()
            )

            verification_results = work_order.get("verification_results", [])
            if not isinstance(verification_results, list):
                continue

            for verification in verification_results:
                if not isinstance(verification, dict):
                    continue
                saw_verification_results = True
                if verification.get("passed") is True:
                    tests_passed += 1

        unique_changed_files = list(dict.fromkeys(changed_files))
        unique_tests_run = list(dict.fromkeys(tests_run))
        if (
            not saw_verification_results
            and unique_tests_run
            and str(worker_result.get("status", "")).strip().lower() == "completed"
        ):
            tests_passed = len(unique_tests_run)

        return len(unique_changed_files), len(unique_tests_run), tests_passed

    def _append_iteration_metrics(
        self,
        *,
        iteration: int,
        issue_number: int | None,
        worker_result: dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        """Append one JSONL row for a finalized boss-loop iteration."""
        metrics_path_text = str(self.config.metrics_jsonl_path or "").strip()
        if not metrics_path_text:
            return

        try:
            files_changed, tests_run, tests_passed = self._extract_iteration_metrics(worker_result)
            metrics_path = Path(metrics_path_text)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract prompt/context instrumentation from work order data
            run_dict = worker_result.get("run")
            prompt_chars = 0
            enriched_context_chars = 0
            prompt_version = "v2"  # Context-first prompt (post a11848eac)
            is_decomposed = bool(
                re.search(
                    r"\[from #\d+\]",
                    str((worker_result.get("receipt_metadata") or {}).get("issue_title", "")),
                )
            )
            worker_outcome = str(worker_result.get("outcome", "")).strip()
            has_deliverable = bool(worker_result.get("deliverable"))
            publish_action = (
                str((worker_result.get("publish_result") or {}).get("action", "")).strip() or None
            )

            # Read prompt size from persisted WorkerProcess fields
            if isinstance(run_dict, dict):
                for wo in run_dict.get("work_orders", []):
                    if isinstance(wo, dict):
                        prompt_chars += int(wo.get("prompt_chars", 0) or 0)
                        enriched_context_chars += int(wo.get("enriched_context_chars", 0) or 0)

            payload = {
                "iteration": int(iteration),
                "issue_number": issue_number,
                "worker_status": str(worker_result.get("status", "")).strip() or "unknown",
                "worker_outcome": worker_outcome or None,
                "elapsed_seconds": float(elapsed_seconds or 0.0),
                "files_changed": files_changed,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "prompt_version": prompt_version,
                "prompt_chars": prompt_chars,
                "enriched_context_chars": enriched_context_chars,
                "is_decomposed_issue": is_decomposed,
                "has_deliverable": has_deliverable,
                "publish_action": publish_action,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        except Exception as exc:
            logger.debug("Boss metrics emission skipped: %s", exc)

    def _normalized_model_rotation(self) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for item in self.config.model_rotation:
            runner_type = str(item).strip().lower()
            if not runner_type or runner_type in seen:
                continue
            seen.add(runner_type)
            normalized.append(runner_type)
        return normalized

    def _selected_issues_need_retry_routing(self, issues: list[GitHubIssue]) -> bool:
        for issue in issues:
            issue_number = int(getattr(issue, "number", 0) or 0)
            if issue_number <= 0:
                continue
            if issue_number in self._pending_handoff_prompts:
                return True
            if int(self._issue_attempt_counts.get(issue_number, 0) or 0) > 0:
                return True
        return False

    def _filter_mixed_retry_routing_batch(
        self,
        issues: list[GitHubIssue],
    ) -> list[GitHubIssue]:
        """Keep retry-routed work isolated from fresh issues in one batch.

        A mixed batch forces `_requested_runner_type_for_freshness()` to widen
        the runner pool for every selected issue. That is correct for retry
        work, but it lets fresh issues piggy-back onto retry-specific routing.
        When both kinds are present, dispatch only the retry-routed issues in
        this iteration and leave fresh work for the next pass.
        """
        if len(issues) <= 1:
            return issues

        retry_routed: list[GitHubIssue] = []
        fresh: list[GitHubIssue] = []
        for issue in issues:
            if self._selected_issues_need_retry_routing([issue]):
                retry_routed.append(issue)
            else:
                fresh.append(issue)
        if retry_routed and fresh:
            return retry_routed
        return issues

    def _already_maxed_issue_numbers(self, issues: list[GitHubIssue]) -> set[int]:
        already_maxed: set[int] = set()
        for num, count in self._issue_attempt_counts.items():
            if count >= self.config.max_retries_per_issue:
                try:
                    issue_number = int(num)
                except (TypeError, ValueError):
                    continue
                already_maxed.add(issue_number)
                if count == self.config.max_retries_per_issue and self.config.repo:
                    self._auto_decompose_stuck_issue(issue_number, issues)
        return already_maxed

    def _existing_open_pr_skip_status(
        self,
        *,
        iteration: int,
        timestamp: str,
        runner_freshness: dict[str, Any],
        issue: GitHubIssue,
        elapsed_seconds: float,
    ) -> BossIterationStatus | None:
        existing_pr = self._has_open_pr_for_issue(issue.number)
        if not existing_pr:
            return None
        logger.info(
            "boss_loop_skip_existing_pr issue=#%s pr=%s",
            issue.number,
            existing_pr,
        )
        issue_dict = self._issue_payload(issue)
        self._completed_issues.append(issue_dict)
        self._issue_attempt_counts[issue.number] = max(
            self._issue_attempt_counts.get(issue.number, 0),
            self.config.max_retries_per_issue,
        )
        return BossIterationStatus(
            iteration=iteration,
            run_id=self.run_id,
            timestamp=timestamp,
            runner_freshness=runner_freshness,
            selected_issue=issue_dict,
            worker_status="completed",
            stop_reason=None,
            needs_human_reasons=[],
            next_actions=[f"Skipped: issue #{issue.number} already has open PR {existing_pr}."],
            elapsed_seconds=elapsed_seconds,
        )

    def _requested_runner_type_for_freshness(
        self,
        selected_issues: list[GitHubIssue],
    ) -> str | None:
        # Broaden the freshness pool only for the issue(s) we are about to
        # dispatch when they are actually on a retry/handoff path. Historical
        # retries on unrelated issues must not let fresh issues bypass the
        # default target runner requirement.
        if (
            self._selected_issues_need_retry_routing(selected_issues)
            and len(self._normalized_model_rotation()) > 1
        ):
            return None
        return self.config.default_target_agent

    def _refresh_runner_heartbeats(self) -> None:
        """Update heartbeat timestamps for all registered runners.

        Called at the top of each iteration so that ``check_runner_freshness``
        does not reject runners whose ``updated_at`` drifted past the TTL
        while the boss loop was still running.
        """
        from aragora.swarm.runner_registry import (
            LocalRunnerRegistry,
            RunnerInspection,
            authorization_context_from_env,
        )

        owner_context = authorization_context_from_env(self._env)
        if owner_context is None:
            return

        registry = (
            LocalRunnerRegistry(path=self.config.registry_path)
            if self.config.registry_path
            else LocalRunnerRegistry()
        )

        for reg in registry.list_registrations():
            runner_id = str(reg.get("runner_id", "")).strip()
            if not runner_id:
                continue
            inspection = RunnerInspection(
                runner_id=runner_id,
                runner_type=str(reg.get("runner_type", "codex")).strip(),
                availability=str(reg.get("availability", "unknown")).strip(),
                available=_strict_bool(reg.get("available")) is True,
                auth_mode=str(reg.get("auth_mode", "unknown")).strip(),
                command_path=reg.get("command_path"),
                profile=reg.get("profile"),
            )
            try:
                registry.heartbeat(inspection, owner_context=owner_context)
            except Exception:
                logger.debug("Failed to refresh heartbeat for runner %s", runner_id, exc_info=True)

    def _requested_target_agent_for_issue(self, issue_number: int) -> str | None:
        attempt_count = max(0, int(self._issue_attempt_counts.get(issue_number, 0) or 0))
        default_target = str(self.config.default_target_agent or "").strip().lower() or None
        if attempt_count <= 1:
            return default_target

        rotation = self._normalized_model_rotation()
        if not rotation:
            return default_target
        if default_target and default_target in rotation:
            base_index = rotation.index(default_target)
            return rotation[(base_index + attempt_count - 1) % len(rotation)]
        if default_target:
            return rotation[(attempt_count - 2) % len(rotation)]
        return rotation[(attempt_count - 2) % len(rotation)]

    def _extract_worker_agent(self, worker_result: dict[str, Any]) -> str | None:
        for key in ("target_agent", "runner_type"):
            value = str(worker_result.get(key, "")).strip().lower()
            if value:
                return value

        receipt_metadata = worker_result.get("receipt_metadata")
        if isinstance(receipt_metadata, dict):
            for key in ("actual_target_agent", "requested_target_agent", "runner_type"):
                value = str(receipt_metadata.get(key, "")).strip().lower()
                if value:
                    return value

        run = worker_result.get("run")
        if not isinstance(run, dict):
            return None
        work_orders = run.get("work_orders", [])
        if not isinstance(work_orders, list):
            return None
        for work_order in work_orders:
            if not isinstance(work_order, dict):
                continue
            value = str(work_order.get("target_agent", "")).strip().lower()
            if value:
                return value
        return None

    def _pending_handoff_candidates(
        self,
        issues: list[GitHubIssue],
        *,
        blocked_scopes: set[str] | None = None,
    ) -> list[GitHubIssue]:
        if not self._pending_handoff_prompts:
            return []

        issue_by_number = {int(issue.number): issue for issue in issues}
        candidates: list[GitHubIssue] = []
        stale_issue_numbers: list[int] = []

        for issue_number in list(self._pending_handoff_prompts):
            issue = issue_by_number.get(issue_number)
            if issue is None:
                stale_issue_numbers.append(issue_number)
                continue
            if self.config.issue_number is not None and issue_number != self.config.issue_number:
                continue
            if (
                select_eligible_issue(
                    [issue],
                    skip_labels=self.config.skip_labels,
                    require_labels=self.config.require_labels,
                    blocked_scopes=blocked_scopes,
                )
                is None
            ):
                stale_issue_numbers.append(issue_number)
                continue
            candidates.append(issue)

        for issue_number in stale_issue_numbers:
            self._pending_handoff_prompts.pop(issue_number, None)

        return candidates

    def _target_issue_miss_guidance(self, issue_number: int) -> tuple[list[str], list[str]]:
        reasons = [
            f"Target issue #{issue_number} was not found in the issue feed or is not eligible under current filters/retry state."
        ]
        next_actions = [
            f"Verify issue #{issue_number} is still open, eligible, and has not exceeded retry limits.",
            "Remove --boss-issue-number to return to feed-driven selection.",
        ]
        fetch_issue = getattr(self._feed, "_fetch_issue", None)
        if not callable(fetch_issue):
            return reasons, next_actions
        try:
            issue = fetch_issue(issue_number, allow_closed=True)
        except TypeError:
            try:
                issue = fetch_issue(issue_number)
            except Exception:
                return reasons, next_actions
        except Exception:
            return reasons, next_actions
        if not isinstance(issue, GitHubIssue):
            return reasons, next_actions

        state = str(issue.state or "").strip().lower()
        if state and state != "open":
            return (
                [
                    f"Target issue #{issue_number} is {state} and cannot be selected by the open-issue boss feed."
                ],
                [
                    f"Reopen issue #{issue_number} if it should be eligible for Boss dispatch.",
                    "Remove --boss-issue-number to return to feed-driven selection.",
                ],
            )

        labels = {str(label).strip() for label in issue.labels if str(label).strip()}
        skipped = sorted(labels & set(self.config.skip_labels or set()))
        if skipped:
            return (
                [f"Target issue #{issue_number} is excluded by skip labels: {', '.join(skipped)}."],
                [
                    f"Remove skip labels from issue #{issue_number} or adjust --label-filter/skip-label settings.",
                    "Remove --boss-issue-number to return to feed-driven selection.",
                ],
            )

        required = set(self.config.require_labels or set())
        missing_labels = sorted(required - labels)
        if missing_labels:
            return (
                [
                    f"Target issue #{issue_number} is missing required labels: {', '.join(missing_labels)}."
                ],
                [
                    f"Add the required labels to issue #{issue_number} or adjust --require-label settings.",
                    "Remove --boss-issue-number to return to feed-driven selection.",
                ],
            )

        if not issue.title:
            return (
                [f"Target issue #{issue_number} is missing a title and cannot be selected."],
                [
                    f"Add a non-empty title to issue #{issue_number}.",
                    "Remove --boss-issue-number to return to feed-driven selection.",
                ],
            )
        return reasons, next_actions

    @staticmethod
    def _skip_label_summary(report: IssueEligibilityReport) -> str | None:
        if not report.skipped_by_label:
            return None
        parts: list[str] = []
        for label, numbers in sorted(report.skipped_by_label.items()):
            issue_refs = ", ".join(f"#{number}" for number in numbers[:3])
            if len(numbers) > 3:
                issue_refs = f"{issue_refs}, +{len(numbers) - 3} more"
            parts.append(f"{label} ({len(numbers)}: {issue_refs})")
        return "Skipped by label: " + "; ".join(parts)

    @staticmethod
    def _skip_sanitation_summary(report: IssueEligibilityReport) -> list[str]:
        if not report.skipped_by_sanitation:
            return []
        parts: list[str] = []
        for reason, numbers in sorted(report.skipped_by_sanitation.items()):
            issue_refs = ", ".join(f"#{number}" for number in numbers[:3])
            if len(numbers) > 3:
                issue_refs = f"{issue_refs}, +{len(numbers) - 3} more"
            parts.append(f"{reason} ({len(numbers)}: {issue_refs})")
        return parts

    def _log_issue_skip_summary(self, report: IssueEligibilityReport) -> None:
        summary = self._skip_label_summary(report)
        if summary:
            logger.info("Boss loop %s", summary)
        sanitation = self._skip_sanitation_summary(report)
        if sanitation:
            logger.info("Boss loop skipped by sanitation: %s", "; ".join(sanitation))
            self._last_sanitation_summary = sanitation
        else:
            self._last_sanitation_summary = []

    def _no_suitable_issue_guidance(
        self,
        *,
        already_maxed: set[int],
        report: IssueEligibilityReport,
    ) -> tuple[list[str], list[str]]:
        needs_human_reasons = ["No suitable open issue found in the GitHub feed."]
        next_actions = [
            "Create a new issue with actionable scope, or adjust label filters.",
            (
                "Eligible dispatch candidates after filters: "
                f"{report.eligible_count}, already maxed retries: {len(already_maxed)}"
            ),
        ]
        summary = self._skip_label_summary(report)
        if summary:
            needs_human_reasons.append(summary)
            next_actions.append(summary)
        sanitation = self._skip_sanitation_summary(report)
        if sanitation:
            sanitation_summary = "Skipped by sanitation: " + "; ".join(sanitation)
            needs_human_reasons.append(sanitation_summary)
            next_actions.append(sanitation_summary)
        return needs_human_reasons, next_actions

    def _emit_terminal_receipt(self, result: BossLoopResult) -> None:
        try:
            from aragora.receipts.provenance import emit_operational_receipt

            attempted = len(result.issues_attempted)
            completed = len(result.issues_completed)
            failed = len(result.issues_failed)
            if completed > 0:
                verdict = "pass"
            elif result.stop_reason in {
                BossStopReason.NO_FRESH_RUNNER.value,
                BossStopReason.NO_SUITABLE_ISSUE.value,
                BossStopReason.NEEDS_HUMAN.value,
            }:
                verdict = "blocked"
            else:
                verdict = "fail"

            emit_operational_receipt(
                source="boss_loop",
                action="run_completed",
                actor="boss-loop",
                inputs={
                    "run_id": self.run_id,
                    "repo": self.config.repo,
                    "label_filter": self.config.label_filter,
                    "max_iterations": self.config.max_iterations,
                    "max_retries_per_issue": self.config.max_retries_per_issue,
                    "max_consecutive_failures": self.config.max_consecutive_failures,
                    "budget_limit_usd": self.config.budget_limit_usd,
                    "configured_max_parallel_dispatches": result.configured_max_parallel_dispatches,
                },
                outputs={
                    "iterations_completed": result.iterations_completed,
                    "stop_reason": result.stop_reason,
                    "issues_attempted": attempted,
                    "issues_completed": completed,
                    "issues_failed": failed,
                    "effective_parallel_dispatches_observed": (
                        result.effective_parallel_dispatches_observed
                    ),
                    "needs_human_reasons": list(result.needs_human_reasons),
                    "next_actions": list(result.next_actions),
                    "sanitation_summary": list(result.sanitation_summary),
                },
                verdict=verdict,
                confidence=(completed / attempted) if attempted else 0.0,
                duration_seconds=result.total_elapsed_seconds,
            )
        except Exception as exc:
            logger.debug("Boss loop operational receipt skipped: %s", exc)

    @staticmethod
    def _extract_worker_transcript(worker_result: dict[str, Any]) -> str:
        """Extract the worker's stdout transcript from the run dict."""
        run = worker_result.get("run")
        if not isinstance(run, dict):
            return ""
        work_orders = run.get("work_orders", [])
        if not isinstance(work_orders, list):
            return ""
        parts = []
        for wo in work_orders:
            if isinstance(wo, dict):
                for key in ("stdout_tail", "transcript", "log_tail"):
                    tail = str(wo.get(key, "")).strip()
                    if tail:
                        parts.append(tail)
                        break
        return "\n---\n".join(parts)

    @staticmethod
    def _extract_worker_files_changed(worker_result: dict[str, Any]) -> list[str]:
        """Extract changed file paths from the run dict."""
        run = worker_result.get("run")
        if not isinstance(run, dict):
            return []
        work_orders = run.get("work_orders", [])
        files: list[str] = []
        for wo in work_orders:
            if isinstance(wo, dict):
                paths = wo.get("changed_paths", [])
                if isinstance(paths, list):
                    files.extend(str(p) for p in paths if str(p).strip())
        return files

    def _repo_slug_for_issue(self, issue: GitHubIssue) -> str | None:
        configured_repo = str(self.config.repo or "").strip()
        if configured_repo:
            return configured_repo
        match = _GITHUB_ISSUE_URL_RE.search(str(issue.url or "").strip())
        if match is None:
            return None
        repo = str(match.group("repo") or "").strip()
        return repo or None

    @staticmethod
    def _pr_number_from_url(url: str | None) -> int | None:
        match = _GITHUB_PR_URL_RE.search(str(url or "").strip())
        if match is None:
            return None
        try:
            return int(match.group("number"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _already_done_comment(worker_result: dict[str, Any]) -> str | None:
        run = worker_result.get("run")
        if not isinstance(run, dict):
            return None
        work_orders = [item for item in run.get("work_orders", []) if isinstance(item, dict)]
        if not work_orders:
            return None
        if any(
            str(item.get("worker_outcome", "")).strip() != "clean_exit_no_effect"
            or item.get("commit_shas")
            or item.get("changed_paths")
            for item in work_orders
        ):
            return None

        evidence_phrase: str | None = None
        passed_checks = 0
        tests_run: set[str] = set()
        for item in work_orders:
            for verification in item.get("verification_results", []) or []:
                if isinstance(verification, dict) and verification.get("passed") is True:
                    passed_checks += 1
            for test_cmd in item.get("tests_run", []) or []:
                text = str(test_cmd).strip()
                if text:
                    tests_run.add(text)
            text_blob = "\n".join(
                str(item.get(key, "")).strip()
                for key in ("stdout_tail", "stderr_tail", "blocker", "failure_reason")
                if str(item.get(key, "")).strip()
            ).lower()
            for marker in _ALREADY_DONE_MARKERS:
                if marker in text_blob:
                    evidence_phrase = marker
                    break
            if evidence_phrase:
                break

        if evidence_phrase is None:
            return None
        verification_detail = ""
        if passed_checks:
            verification_detail = f" Verification passed on {passed_checks} check(s)."
        elif tests_run:
            verification_detail = (
                " Verification commands were run: " + ", ".join(sorted(tests_run)[:3]) + "."
            )
        return (
            "Already implemented — autonomous verification found no code changes were needed, "
            f"and worker logs indicated '{evidence_phrase}'.{verification_detail}"
        )

    def _maybe_auto_close_already_done_issue(
        self,
        issue: GitHubIssue,
        worker_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.config.auto_close_already_done_issues:
            return None
        comment = self._already_done_comment(worker_result)
        if comment is None:
            return None
        repo_slug = self._repo_slug_for_issue(issue)
        if repo_slug is None:
            return {
                "action": "skipped",
                "reason": "missing_repo_slug",
                "issue_number": issue.number,
            }
        try:
            proc = subprocess.run(
                [
                    "gh",
                    "issue",
                    "close",
                    str(issue.number),
                    "--repo",
                    repo_slug,
                    "--comment",
                    comment,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Boss auto-close failed for issue #%s: %s", issue.number, exc)
            return {
                "action": "failed",
                "reason": type(exc).__name__,
                "issue_number": issue.number,
                "repo": repo_slug,
            }
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            logger.warning("Boss auto-close failed for issue #%s: %s", issue.number, detail)
            return {
                "action": "failed",
                "reason": "gh_issue_close_failed",
                "detail": detail or "gh issue close failed",
                "issue_number": issue.number,
                "repo": repo_slug,
            }
        worker_result["outcome"] = "issue_already_resolved"
        return {
            "action": "closed",
            "reason": "already_implemented",
            "issue_number": issue.number,
            "repo": repo_slug,
            "comment": comment,
        }

    def _auto_decompose_stuck_issue(
        self,
        issue_number: int | str,
        issues: list[GitHubIssue],
    ) -> None:
        """When an issue exhausts retries, try to decompose it into sub-issues.

        Uses the TaskDecomposer to break the issue into smaller pieces, then
        creates sub-issues on GitHub with the boss-ready label. If decomposition
        fails or produces nothing useful, falls back to labeling boss-stuck.
        """
        import subprocess

        repo = self.config.repo or ""
        issue = next((i for i in issues if i.number == int(issue_number)), None)
        if not issue:
            return

        # Guard: cap decomposition depth to prevent runaway recursion.
        lineage_root, decomposition_depth = self._decomposition_lineage(issue)
        if decomposition_depth >= self.config.max_decomposition_depth:
            self._label_boss_stuck(
                issue_number,
                repo,
                f"Decomposition depth {decomposition_depth} reached limit of "
                f"{self.config.max_decomposition_depth}. Needs manual attention.",
            )
            return

        # Guard: cap total sub-issues created per run to prevent runaway fan-out.
        if self._total_sub_issues_created >= self.config.max_total_sub_issues_per_run:
            logger.warning(
                "Skipping decomposition for issue #%s: per-run budget of %d sub-issues exhausted",
                issue.number,
                self.config.max_total_sub_issues_per_run,
            )
            self._label_boss_stuck(
                issue_number,
                repo,
                f"Per-run sub-issue budget ({self.config.max_total_sub_issues_per_run}) exhausted.",
            )
            return

        # Check if a PR was already merged for this issue — if so, close it
        try:
            pr_check = subprocess.run(
                [
                    "gh",
                    "pr",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "merged",
                    "--search",
                    f"#{issue.number}",
                    "--limit",
                    "1",
                    "--json",
                    "number",
                    "--jq",
                    ".[0].number",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if pr_check.returncode == 0 and pr_check.stdout.strip():
                self._label_boss_stuck(
                    issue_number,
                    repo,
                    f"PR #{pr_check.stdout.strip()} already merged for this issue.",
                )
                return
        except Exception:
            pass

        open_pr_changed_paths: set[str] = set()
        if repo:
            try:
                open_pr_changed_paths = fetch_open_pr_changed_paths(repo=repo)
            except Exception as exc:
                logger.debug("Open PR scope lookup failed during auto-decomposition: %s", exc)

        # Collect existing boss-ready issues to avoid duplicate fan-out.
        existing_titles: set[str] = set()
        existing_decomposition_signatures: list[dict[str, Any]] = []
        try:
            proc = subprocess.run(
                [
                    "gh",
                    "issue",
                    "list",
                    "--repo",
                    repo,
                    "--label",
                    "boss-ready",
                    "--state",
                    "open",
                    "--limit",
                    "100",
                    "--json",
                    "number,title,body,url",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0:
                raw_issues = json.loads(proc.stdout or "[]")
                if isinstance(raw_issues, list):
                    for raw_issue in raw_issues:
                        if not isinstance(raw_issue, dict):
                            continue
                        try:
                            existing_number = int(raw_issue.get("number", 0) or 0)
                        except (TypeError, ValueError):
                            existing_number = 0
                        if existing_number == int(issue.number):
                            continue
                        existing_title = str(raw_issue.get("title", "") or "").strip()
                        existing_body = str(raw_issue.get("body", "") or "").strip()
                        if existing_title:
                            existing_titles.add(existing_title.lower())
                        signature = self._decomposition_issue_signature(
                            number=existing_number,
                            title=existing_title,
                            body=existing_body,
                            url=str(raw_issue.get("url", "") or "").strip(),
                        )
                        if signature["scopes"]:
                            existing_decomposition_signatures.append(signature)
                else:
                    existing_titles = {
                        line.strip().lower() for line in proc.stdout.splitlines() if line.strip()
                    }
        except Exception:
            pass

        parent_signature = self._decomposition_issue_signature(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            url=issue.url,
        )

        # Try LLM-based decomposition
        sub_issues_created = 0
        decomposition_candidates = 0
        covered_candidates = 0
        try:
            from aragora.nomic.task_decomposer import TaskDecomposer

            decomposer = TaskDecomposer()
            result = decomposer.analyze(
                issue.body or issue.title,
                file_scope_hints=list(self._extract_file_scope_hints(issue.body or "")),
            )

            if result.should_decompose and result.subtasks:
                for subtask in result.subtasks[:3]:  # Cap at 3 sub-issues
                    # Validate subtask quality before creating an issue
                    sub_desc = (subtask.description or "").strip()
                    if len(sub_desc) < 40:
                        logger.debug(
                            "Skipping malformed subtask (description too short): %r",
                            sub_desc[:80],
                        )
                        continue
                    sub_title = (subtask.title or sub_desc[:60]).strip()[:80]
                    title = f"[from #{issue.number}] {sub_title}"
                    if title.lower() in existing_titles:
                        decomposition_candidates += 1
                        covered_candidates += 1
                        continue
                    valid_scope = self._decomposition_scope_entries(subtask.file_scope or [])
                    if not valid_scope:
                        logger.debug(
                            "Skipping subtask with no valid file scope: %r",
                            sub_title,
                        )
                        continue
                    # Annotate files that don't exist on disk as (new file)
                    scope_lines_parts = []
                    for f in valid_scope:
                        if (Path.cwd() / f).exists():
                            scope_lines_parts.append(f"- `{f}`")
                        else:
                            scope_lines_parts.append(f"- `{f}` (new file)")
                    scope_lines = "\n".join(scope_lines_parts)
                    # Build specific validation command
                    test_files = [f for f in valid_scope if f.startswith("tests/")]
                    src_files = [
                        f for f in valid_scope if f.endswith(".py") and not f.startswith("tests/")
                    ]
                    # Only use pytest if the test file exists; otherwise use ruff
                    existing_test_files = [f for f in test_files if (Path.cwd() / f).exists()]
                    if existing_test_files:
                        validation_cmd = f"`python3 -m pytest {existing_test_files[0]} -q`"
                    elif src_files:
                        validation_cmd = f"`ruff check {' '.join(src_files)}`"
                    elif test_files:
                        # Test file doesn't exist yet — validate source instead
                        related_src = [
                            f.replace("tests/", "aragora/").replace("test_", "") for f in test_files
                        ]
                        existing_src = [f for f in related_src if (Path.cwd() / f).exists()]
                        if existing_src:
                            validation_cmd = f"`ruff check {' '.join(existing_src)}`"
                        else:
                            validation_cmd = "`ruff check` on the changed files passes"
                    else:
                        validation_cmd = "`pytest` on the changed files passes"
                    decomposition_candidates += 1
                    signature = self._decomposition_candidate_signature(
                        title=sub_title,
                        description=sub_desc,
                        scope_entries=valid_scope,
                        validation_command=validation_cmd,
                    )
                    if decomposition_depth > 0 and self._decomposition_candidate_restates_parent(
                        candidate=signature,
                        parent=parent_signature,
                        candidate_text=f"{sub_title}\n{sub_desc}",
                    ):
                        covered_candidates += 1
                        continue
                    if self._decomposition_candidate_is_covered(
                        signature=signature,
                        existing_signatures=existing_decomposition_signatures,
                        open_pr_changed_paths=open_pr_changed_paths,
                    ):
                        covered_candidates += 1
                        continue
                    child_depth = decomposition_depth + 1
                    body = (
                        f"Auto-decomposed from #{issue.number} after {self.config.max_retries_per_issue} "
                        f"failed autonomous attempts.\n\n"
                        f"## Decomposition Lineage\n"
                        f"- Root issue: #{lineage_root}\n"
                        f"- Parent issue: #{issue.number}\n"
                        f"- Depth: {child_depth}\n\n"
                        f"## Task\n{sub_desc}\n\n"
                        f"## Files\n{scope_lines}\n\n"
                        f"## Acceptance\n"
                        f"{validation_cmd}\n\n"
                        f"## Constraints\n"
                        f"- Single-file change preferred\n"
                        f"- Under 100 lines of new/changed code\n"
                        f"- Estimated complexity: {subtask.estimated_complexity}\n"
                    )
                    try:
                        proc = subprocess.run(
                            [
                                "gh",
                                "issue",
                                "create",
                                "--repo",
                                repo,
                                "--title",
                                title,
                                "--body",
                                body,
                                "--label",
                                "boss-ready",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        if proc.returncode == 0:
                            sub_issues_created += 1
                            self._total_sub_issues_created += 1
                    except Exception:
                        pass

        except Exception as exc:
            logger.debug("Auto-decomposition failed for #%s: %s", issue_number, exc)

        # Comment on the parent issue
        if sub_issues_created > 0:
            comment = (
                f"Boss loop exhausted {self.config.max_retries_per_issue} attempts. "
                f"Auto-decomposed into {sub_issues_created} smaller sub-issues with `boss-ready` label."
            )
        elif decomposition_candidates > 0 and covered_candidates == decomposition_candidates:
            comment = (
                f"Boss loop exhausted {self.config.max_retries_per_issue} attempts. "
                "All decomposition candidates are already covered by the parent task, "
                "existing open `boss-ready` issues, or open PRs."
            )
        else:
            comment = (
                f"Boss loop exhausted {self.config.max_retries_per_issue} attempts without "
                f"producing a deliverable. The issue may be too complex for autonomous workers."
            )

        self._label_boss_stuck(issue_number, repo, comment)

    @staticmethod
    def _decomposition_lineage(issue: GitHubIssue) -> tuple[int, int]:
        body = issue.body or ""
        title = issue.title or ""
        title_parents = [
            int(match) for match in re.findall(r"\[from #(\d+)\]", title) if match.isdigit()
        ]

        root_match = re.search(r"\bRoot issue:\s*#?(\d+)", body, flags=re.IGNORECASE)
        depth_match = re.search(r"\bDepth:\s*(\d+)", body, flags=re.IGNORECASE)
        legacy_parent_match = re.search(
            r"\bAuto-decomposed from #(\d+)",
            body,
            flags=re.IGNORECASE,
        )

        if root_match:
            root_issue = int(root_match.group(1))
        elif title_parents:
            root_issue = title_parents[0]
        elif legacy_parent_match:
            root_issue = int(legacy_parent_match.group(1))
        else:
            root_issue = int(issue.number)

        if depth_match:
            depth = int(depth_match.group(1))
        elif title_parents:
            depth = len(title_parents)
        elif legacy_parent_match:
            depth = 1
        else:
            depth = 0

        return root_issue, depth

    @staticmethod
    def _decomposition_scope_entries(
        paths: list[str] | tuple[str, ...] | set[str],
    ) -> tuple[str, ...]:
        scope_issue = GitHubIssue(
            number=0,
            title="",
            body="\n".join(f"`{path}`" for path in paths if str(path or "").strip()),
            labels=[],
            url="",
            state="OPEN",
            created_at="",
        )
        return tuple(infer_issue_scope_entries(scope_issue))

    @staticmethod
    def _normalize_decomposition_validation(command: str) -> str:
        text = str(command or "").strip()
        if not text:
            return ""
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = text.replace("python -m pytest", "python3 -m pytest")
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    @staticmethod
    def _normalize_decomposition_intent(*parts: str) -> str:
        text = " ".join(str(part or "") for part in parts)
        text = re.sub(r"\[from #\d+\]", " ", text.lower())
        tokens = re.findall(r"[a-z0-9_/.]+", text)
        stopwords = {
            "a",
            "add",
            "all",
            "and",
            "are",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "keep",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }
        return " ".join(token for token in tokens if token not in stopwords)

    @classmethod
    def _decomposition_issue_signature(
        cls,
        *,
        number: int,
        title: str,
        body: str,
        url: str,
    ) -> dict[str, Any]:
        issue = GitHubIssue(
            number=number,
            title=title,
            body=body,
            labels=["boss-ready"],
            url=url,
            state="OPEN",
            created_at="",
        )
        return {
            "scopes": tuple(infer_issue_scope_entries(issue)),
            "validations": tuple(
                cls._normalize_decomposition_validation(command)
                for command in extract_pre_dispatch_validation_commands(body)
                if cls._normalize_decomposition_validation(command)
            ),
            "intent": cls._normalize_decomposition_intent(title, body),
        }

    @classmethod
    def _decomposition_candidate_signature(
        cls,
        *,
        title: str,
        description: str,
        scope_entries: tuple[str, ...],
        validation_command: str,
    ) -> dict[str, Any]:
        validation = cls._normalize_decomposition_validation(validation_command)
        return {
            "scopes": tuple(scope_entries),
            "validations": (validation,) if validation else (),
            "intent": cls._normalize_decomposition_intent(title, description),
        }

    @staticmethod
    def _decomposition_scope_sets_overlap(
        left: tuple[str, ...] | set[str],
        right: tuple[str, ...] | set[str],
    ) -> bool:
        return any(
            scope_entries_overlap(left_scope, right_scope)
            for left_scope in left
            for right_scope in right
        )

    @classmethod
    def _decomposition_intents_overlap(cls, left: str, right: str) -> bool:
        left_tokens = set((left or "").split())
        right_tokens = set((right or "").split())
        if not left_tokens or not right_tokens:
            return False
        shared = left_tokens & right_tokens
        return len(shared) >= max(3, min(len(left_tokens), len(right_tokens)) // 2)

    @staticmethod
    def _decomposition_intent_is_generic(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(text or "").lower())
        generic_phrases = (
            "fix failing tests",
            "repair failing tests",
            "ensure comprehensive coverage",
            "comprehensive test coverage",
            "comprehensive unit tests",
            "execute and verify",
            "run and verify",
        )
        return any(phrase in normalized for phrase in generic_phrases)

    @staticmethod
    def _decomposition_generic_same_scope_overlap(left: str, right: str) -> bool:
        left_tokens = set((left or "").split())
        right_tokens = set((right or "").split())
        if not left_tokens or not right_tokens:
            return False
        shared = left_tokens & right_tokens
        if len(shared) < 3:
            return False
        generic_tokens = {
            "coverage",
            "comprehensive",
            "execute",
            "failing",
            "fix",
            "repair",
            "run",
            "suite",
            "test",
            "tests",
            "unit",
            "validate",
            "verify",
        }
        # Same-file generic test/coverage decompositions can differ in exact
        # validation command when a test file is being created. Require at
        # least one shared domain token so unrelated same-file tasks do not
        # suppress each other.
        return bool(shared - generic_tokens)

    @classmethod
    def _decomposition_candidate_restates_parent(
        cls,
        *,
        candidate: dict[str, Any],
        parent: dict[str, Any],
        candidate_text: str,
    ) -> bool:
        if not cls._decomposition_intent_is_generic(candidate_text):
            return False
        candidate_scopes = tuple(candidate.get("scopes") or ())
        parent_scopes = tuple(parent.get("scopes") or ())
        if not candidate_scopes or not parent_scopes:
            return False
        if not cls._decomposition_scope_sets_overlap(candidate_scopes, parent_scopes):
            return False
        return cls._decomposition_intents_overlap(
            str(candidate.get("intent") or ""),
            str(parent.get("intent") or ""),
        )

    @classmethod
    def _decomposition_candidate_is_covered(
        cls,
        *,
        signature: dict[str, Any],
        existing_signatures: list[dict[str, Any]],
        open_pr_changed_paths: set[str],
    ) -> bool:
        scopes = tuple(signature.get("scopes") or ())
        validations = set(signature.get("validations") or ())
        intent = str(signature.get("intent") or "")
        if scopes and cls._decomposition_scope_sets_overlap(scopes, open_pr_changed_paths):
            return True
        for existing in existing_signatures:
            existing_scopes = tuple(existing.get("scopes") or ())
            if not cls._decomposition_scope_sets_overlap(scopes, existing_scopes):
                continue
            existing_validations = set(existing.get("validations") or ())
            if validations and existing_validations and validations & existing_validations:
                return True
            if cls._decomposition_intents_overlap(intent, str(existing.get("intent") or "")):
                return True
            if cls._decomposition_generic_same_scope_overlap(
                intent, str(existing.get("intent") or "")
            ):
                return True
        return False

    @staticmethod
    def _extract_file_scope_hints(body: str) -> list[str]:
        """Extract file paths from an issue body.

        Handles backtick-wrapped paths and strips escaped backticks from
        GitHub markdown rendering.
        """
        import re

        # Strip escaped backticks that GitHub API sometimes returns
        cleaned = body.replace("\\`", "`")
        # Match paths starting with known top-level directories
        return re.findall(
            r"`((?:aragora|tests|scripts|docs|docs-site|sdk|contracts)/[a-zA-Z0-9_/.*-]+(?:\.\w+)?)`",
            cleaned,
        )

    @staticmethod
    def _label_boss_stuck(issue_number: int | str, repo: str, comment: str) -> None:
        """Label an issue as boss-stuck, remove boss-ready, and comment."""
        import subprocess

        try:
            subprocess.run(
                ["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", comment],
                capture_output=True,
                timeout=15,
            )
            # Add boss-stuck AND remove boss-ready — an issue should never be both
            subprocess.run(
                [
                    "gh",
                    "issue",
                    "edit",
                    str(issue_number),
                    "--repo",
                    repo,
                    "--add-label",
                    "boss-stuck",
                    "--remove-label",
                    "boss-ready",
                ],
                capture_output=True,
                timeout=15,
            )
        except Exception:
            pass

    @staticmethod
    def _reuse_existing_published_branch_deliverable(
        worker_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        deliverable = worker_result.get("deliverable")
        if not isinstance(deliverable, dict):
            return None
        if str(deliverable.get("type", "")).strip().lower() != "branch":
            return None

        publish_result = worker_result.get("publish_result")
        if not BossLoop._publish_result_succeeded(publish_result):
            return None

        pr_url = BossLoop._published_pr_url(worker_result)
        if not pr_url or not isinstance(publish_result, dict):
            return None

        branch = (
            str(publish_result.get("branch") or deliverable.get("branch") or "").strip() or None
        )
        commit_shas = [
            str(item).strip()
            for item in deliverable.get("commit_shas", []) or []
            if str(item).strip()
        ]
        worker_result["deliverable"] = {
            **dict(deliverable),
            "type": "pr",
            "branch": branch,
            "commit_shas": commit_shas,
            "pr_url": pr_url,
        }
        worker_result["pr_url"] = pr_url
        worker_result["pr_number"] = BossLoop._pr_number_from_url(pr_url)
        return dict(publish_result)

    @staticmethod
    def _debate_gate_changed_files(worker_result: dict[str, Any]) -> list[str]:
        changed_files: list[str] = []
        seen: set[str] = set()
        run = worker_result.get("run")
        if isinstance(run, dict):
            for work_order in run.get("work_orders", []) or []:
                if not isinstance(work_order, dict):
                    continue
                for path in work_order.get("changed_paths", []) or []:
                    text = str(path).strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    changed_files.append(text)
        deliverable = worker_result.get("deliverable")
        if isinstance(deliverable, dict):
            for path in deliverable.get("changed_paths", []) or []:
                text = str(path).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                changed_files.append(text)
        return changed_files

    def _run_debate_publish_gate(
        self,
        issue: GitHubIssue,
        worker_result: dict[str, Any],
        *,
        branch: str,
        commit_shas: list[str],
    ) -> dict[str, Any] | None:
        if not self.config.use_debate_publish_gate:
            return None
        gate = DebateGate(
            repo_root=Path.cwd().resolve(),
            config=DebateGateConfig(
                enabled=True,
                fail_closed=bool(self.config.debate_publish_gate_fail_closed),
                agent_type=(
                    str(self.config.debate_publish_gate_agent or "").strip()
                    or str(self.config.default_reviewer_agent or "").strip()
                    or "codex"
                ),
                timeout_seconds=float(self.config.debate_publish_gate_timeout_seconds or 90.0),
            ),
        )
        result = gate.evaluate(
            DebateGateRequest(
                issue_number=issue.number,
                issue_title=issue.title,
                issue_body=issue.body,
                source_branch=branch,
                target_branch=self.config.target_branch,
                commit_shas=list(commit_shas),
                changed_files=self._debate_gate_changed_files(worker_result),
                tests_run=[
                    str(item).strip()
                    for item in worker_result.get("tests_run", []) or []
                    if str(item).strip()
                ],
                verification_results=[
                    dict(item)
                    for item in worker_result.get("verification_results", []) or []
                    if isinstance(item, dict)
                ],
                receipt_id=str(worker_result.get("receipt_id", "")).strip() or None,
            )
        ).to_dict()
        worker_result["debate_gate_result"] = dict(result)
        if result.get("verdict") == "blocked":
            blocked_reason = (
                f"Debate publish gate blocked publication: "
                f"{str(result.get('reason', '')).strip() or 'human review required before PR publication.'}"
            )
            reasons = worker_result.get("reasons")
            normalized_reasons = (
                [str(item).strip() for item in reasons if str(item).strip()]
                if isinstance(reasons, list)
                else []
            )
            if blocked_reason not in normalized_reasons:
                normalized_reasons.append(blocked_reason)
            worker_result["reasons"] = normalized_reasons
        return result

    def _maybe_publish_deliverable(
        self,
        issue: GitHubIssue,
        worker_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.config.auto_publish_deliverables:
            return None
        if str(worker_result.get("status", "")).strip() not in {"completed", "needs_human"}:
            return None
        deliverable = worker_result.get("deliverable")
        if not isinstance(deliverable, dict):
            return None
        existing_publish = self._reuse_existing_published_branch_deliverable(worker_result)
        if existing_publish is not None:
            return existing_publish
        deliverable = worker_result.get("deliverable")
        if not isinstance(deliverable, dict):
            return None
        deliverable_type = str(deliverable.get("type", "")).strip().lower()
        prior_publish_result = worker_result.get("publish_result")
        if not isinstance(prior_publish_result, dict):
            receipt_metadata = worker_result.get("receipt_metadata")
            if isinstance(receipt_metadata, dict):
                candidate_publish_result = receipt_metadata.get("publish_result")
                if isinstance(candidate_publish_result, dict):
                    prior_publish_result = dict(candidate_publish_result)
        prior_publish_action = (
            str(prior_publish_result.get("action", "")).strip()
            if isinstance(prior_publish_result, dict)
            else ""
        )
        existing_pr_url = str(
            deliverable.get("pr_url")
            or deliverable.get("adopted_pr")
            or worker_result.get("pr_url")
            or (
                prior_publish_result.get("pr_url") if isinstance(prior_publish_result, dict) else ""
            )
            or ""
        ).strip()
        if (
            deliverable_type not in {"pr", "adopted_pr"}
            and existing_pr_url
            and (
                deliverable.get("pr_url")
                or deliverable.get("adopted_pr")
                or worker_result.get("pr_url")
                or (
                    isinstance(prior_publish_result, dict)
                    and (
                        prior_publish_result.get("published") is True
                        or prior_publish_action
                        in {"pr_created", "existing_pr", "discovered_after_push"}
                    )
                )
            )
        ):
            branch = str(
                deliverable.get("branch")
                or (
                    prior_publish_result.get("branch")
                    if isinstance(prior_publish_result, dict)
                    else ""
                )
                or ""
            ).strip()
            normalized_deliverable = {
                **dict(deliverable),
                "type": "pr",
                "pr_url": existing_pr_url,
            }
            if branch:
                normalized_deliverable["branch"] = branch
            worker_result["deliverable"] = normalized_deliverable
            worker_result["pr_url"] = existing_pr_url
            worker_result["pr_number"] = self._pr_number_from_url(existing_pr_url)
            normalized_publish_result = (
                dict(prior_publish_result) if isinstance(prior_publish_result, dict) else {}
            )
            normalized_publish_result.update(
                {
                    "action": "existing_pr",
                    "published": True,
                    "branch": branch or None,
                    "pr_url": existing_pr_url,
                }
            )
            return normalized_publish_result
        if deliverable_type in {"pr", "adopted_pr"}:
            pr_url = str(
                deliverable.get("pr_url")
                or deliverable.get("adopted_pr")
                or worker_result.get("pr_url")
                or ""
            ).strip()
            if pr_url:
                worker_result["pr_url"] = pr_url
                worker_result["pr_number"] = self._pr_number_from_url(pr_url)
            return {
                "action": "existing_pr",
                "published": True,
                "branch": str(deliverable.get("branch", "")).strip() or None,
                "pr_url": pr_url or None,
            }
        if deliverable_type != "branch":
            return None
        branch = str(deliverable.get("branch", "")).strip()
        commit_shas = [
            str(item).strip()
            for item in deliverable.get("commit_shas", []) or []
            if str(item).strip()
        ]
        if not branch or not commit_shas:
            return None
        max_open_prs = max(int(self.config.max_open_auto_publish_prs or 0), 0)
        if max_open_prs <= 0:
            logger.info(
                "Boss publish deferred for issue #%s branch %s: max_open_auto_publish_prs=%d",
                issue.number,
                branch,
                max_open_prs,
            )
            return {
                "action": "deferred_due_to_open_boss_prs",
                "reason": "max_open_auto_publish_prs_zero",
                "branch": branch,
                "max_open_prs": max_open_prs,
                "open_prs": [],
            }
        open_boss_prs = self._list_open_boss_harvest_prs()
        if len(open_boss_prs) >= max_open_prs:
            logger.info(
                "Boss publish deferred for issue #%s branch %s: %d open boss-harvest PR(s) "
                "already exist (limit=%d)",
                issue.number,
                branch,
                len(open_boss_prs),
                max_open_prs,
            )
            return {
                "action": "deferred_due_to_open_boss_prs",
                "reason": "open_boss_harvest_pr_limit",
                "branch": branch,
                "max_open_prs": max_open_prs,
                "open_prs": open_boss_prs,
            }
        try:
            from aragora.ralph.github_control import GitHubControl
            from aragora.swarm.pr_registry import PullRequestRegistry
            from aragora.swarm.tranche_integrate import publish_lane_deliverable

            repo_root = Path.cwd().resolve()
            try:
                harvest_result = self._harvest_worker_commits_for_publish(
                    issue=issue,
                    repo_root=repo_root,
                    source_branch=branch,
                    commit_shas=commit_shas,
                )
            except Exception as exc:
                logger.warning(
                    "Boss auto-harvest failed for issue #%s branch %s; "
                    "publishing original branch: %s",
                    issue.number,
                    branch,
                    exc,
                )
                harvest_result = {
                    "action": "harvest_failed",
                    "reason": type(exc).__name__,
                    "branch": branch,
                    "source_branch": branch,
                    "commit_shas": commit_shas,
                    "error": str(exc),
                }
            if harvest_result is not None:
                worker_result["harvest_result"] = harvest_result
                harvest_action = str(harvest_result.get("action", "")).strip()
                if harvest_action != "harvest_failed":
                    branch = str(harvest_result.get("branch") or branch).strip() or branch
                else:
                    branch_has_diff = self._publish_branch_has_target_diff(
                        repo_root=repo_root,
                        branch=branch,
                    )
                    if branch_has_diff is False:
                        logger.warning(
                            "Boss publish skipped for issue #%s branch %s: %s",
                            issue.number,
                            branch,
                            "harvest_failed_empty_diff",
                        )
                        return {
                            "action": "skipped_empty_publish_branch",
                            "published": False,
                            "reason": "harvest_failed_empty_diff",
                            "branch": branch,
                            "source_branch": branch,
                            "commit_shas": commit_shas,
                            "harvest_result": dict(harvest_result),
                        }
                    if branch_has_diff is None:
                        logger.warning(
                            "Boss publish fallback could not verify diff for issue #%s branch %s; "
                            "continuing branch publish",
                            issue.number,
                            branch,
                        )
            debate_gate_result = None
            if self.config.use_debate_publish_gate:
                debate_gate_result = self._run_debate_publish_gate(
                    issue,
                    worker_result,
                    branch=branch,
                    commit_shas=commit_shas,
                )
            if isinstance(debate_gate_result, dict):
                worker_result.setdefault("debate_gate_result", dict(debate_gate_result))
            if isinstance(debate_gate_result, dict) and not debate_gate_result.get(
                "publication_allowed", True
            ):
                blocked_reason = "Debate publish gate blocked publication: " + (
                    str(debate_gate_result.get("reason", "")).strip()
                    or "human review required before PR publication."
                )
                reasons = worker_result.get("reasons")
                normalized_reasons = (
                    [str(item).strip() for item in reasons if str(item).strip()]
                    if isinstance(reasons, list)
                    else []
                )
                if blocked_reason not in normalized_reasons:
                    normalized_reasons.append(blocked_reason)
                worker_result["reasons"] = normalized_reasons
                logger.info(
                    "Boss publish blocked by debate gate for issue #%s branch %s: %s",
                    issue.number,
                    branch,
                    str(debate_gate_result.get("reason", "")).strip() or "no reason provided",
                )
                return {
                    "action": "blocked_by_debate_gate",
                    "published": False,
                    "branch": branch,
                    "pr_url": None,
                    "reason": str(debate_gate_result.get("reason", "")).strip()
                    or "Debate gate blocked publication.",
                    "concerns": list(debate_gate_result.get("concerns", []) or []),
                }
            artifact = _BossDeliverableArtifact(
                branch=branch,
                metadata={
                    "branch": branch,
                    "source_branch": str(deliverable.get("branch", "")).strip() or None,
                    "deliverable": {
                        **dict(deliverable),
                        "branch": branch,
                        "commit_shas": commit_shas,
                    },
                    "receipt_id": worker_result.get("receipt_id"),
                    "harvest_result": dict(worker_result.get("harvest_result") or {}),
                },
            )
            publish_result = publish_lane_deliverable(
                artifact,
                manifest_id=f"boss-{self.run_id}-issue-{issue.number}",
                github=GitHubControl(repo_root=repo_root),
                registry=PullRequestRegistry(state_dir=repo_root / ".aragora"),
                repo_root=repo_root,
                target_branch=self.config.target_branch,
                artifact_store=None,
            )
        except Exception as exc:
            logger.warning(
                "Boss publish failed for issue #%s branch %s: %s",
                issue.number,
                branch,
                exc,
            )
            return {
                "action": "failed",
                "reason": type(exc).__name__,
                "branch": branch,
            }

        pr_url = str(publish_result.get("pr_url", "")).strip()
        if pr_url:
            worker_result["deliverable"] = {
                **dict(deliverable),
                "type": "pr",
                "branch": branch,
                "commit_shas": commit_shas,
                "pr_url": pr_url,
            }
            worker_result["pr_url"] = pr_url
            worker_result["pr_number"] = self._pr_number_from_url(pr_url)
        return dict(publish_result)

    def _list_open_boss_harvest_prs(self) -> list[dict[str, Any]]:
        """List open boss-harvest PRs so auto-publish can avoid queue fan-out."""
        repo = str(self.config.repo or "").strip()
        if not repo:
            return []
        cmd: list[str] = [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,headRefName,isDraft,url",
            "--limit",
            "100",
            "-R",
            repo,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                logger.debug(
                    "Failed to list open boss-harvest PRs: %s",
                    (proc.stderr or proc.stdout or "").strip(),
                )
                return []
            entries = json.loads(proc.stdout or "[]")
        except Exception as exc:
            logger.debug("Exception listing open boss-harvest PRs: %s", exc)
            return []

        open_boss_prs: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            head_ref = str(entry.get("headRefName", "")).strip()
            if not head_ref.startswith("aragora/boss-harvest/issue-"):
                continue
            open_boss_prs.append(
                {
                    "number": entry.get("number"),
                    "headRefName": head_ref,
                    "isDraft": bool(entry.get("isDraft")),
                    "url": str(entry.get("url", "")).strip() or None,
                }
            )
        return open_boss_prs

    def _has_open_pr_for_issue(self, issue_number: int) -> str | None:
        """Check if there is already an open boss-loop PR for the given issue.

        Returns the PR URL if found, otherwise ``None``.  Uses the cached
        ``_list_open_boss_harvest_prs`` results when available — the branch
        naming convention ``aragora/boss-harvest/issue-{N}`` encodes the issue
        number so a substring match is sufficient.
        """
        open_prs = self._list_open_boss_harvest_prs()
        suffix = f"issue-{issue_number}"
        for pr in open_prs:
            head_ref = str(pr.get("headRefName", ""))
            if head_ref.endswith(suffix) or f"issue-{issue_number}-" in head_ref:
                return str(pr.get("url") or "")
        return None

    def _git_repo_cmd(
        self,
        repo_root: Path,
        args: list[str],
        *,
        timeout: float = 30.0,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            env=git_safe_env(self._env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    def _verify_git_commit_ref(self, repo_root: Path, ref: str) -> str | None:
        ref = str(ref or "").strip()
        if not ref:
            return None
        try:
            proc = self._git_repo_cmd(
                repo_root,
                ["rev-parse", "--verify", f"{ref}^{{commit}}"],
                timeout=10.0,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return ref

    def _publish_branch_has_target_diff(
        self,
        *,
        repo_root: Path,
        branch: str,
    ) -> bool | None:
        """Return whether a branch has publishable diff against target branch."""
        branch = str(branch or "").strip()
        if not branch:
            return None

        target_branch = str(self.config.target_branch or "main").strip() or "main"
        base_ref = self._verify_git_commit_ref(repo_root, f"origin/{target_branch}")
        if base_ref is None:
            try:
                self._git_repo_cmd(
                    repo_root,
                    ["fetch", "--no-tags", "origin", target_branch],
                    timeout=60.0,
                )
            except Exception:
                pass
            base_ref = self._verify_git_commit_ref(repo_root, f"origin/{target_branch}")
        if base_ref is None:
            base_ref = self._verify_git_commit_ref(repo_root, target_branch)
        if base_ref is None:
            return None

        branch_ref = self._verify_git_commit_ref(repo_root, branch)
        if branch_ref is None:
            remote_branch = branch.removeprefix("origin/")
            remote_ref = f"origin/{remote_branch}"
            branch_ref = self._verify_git_commit_ref(repo_root, remote_ref)
            if branch_ref is None:
                try:
                    self._git_repo_cmd(
                        repo_root,
                        [
                            "fetch",
                            "--no-tags",
                            "origin",
                            f"refs/heads/{remote_branch}:refs/remotes/origin/{remote_branch}",
                        ],
                        timeout=60.0,
                    )
                except Exception:
                    pass
                branch_ref = self._verify_git_commit_ref(repo_root, remote_ref)
        if branch_ref is None:
            return None

        try:
            diff_proc = self._git_repo_cmd(
                repo_root,
                ["diff", "--quiet", f"{base_ref}...{branch_ref}", "--"],
                timeout=30.0,
            )
        except Exception:
            return None
        if diff_proc.returncode == 0:
            return False
        if diff_proc.returncode == 1:
            return True
        logger.debug(
            "Failed to diff publish branch %s against %s: %s",
            branch_ref,
            base_ref,
            (diff_proc.stderr or diff_proc.stdout or "").strip(),
        )
        return None

    def _harvest_worker_commits_for_publish(
        self,
        *,
        issue: GitHubIssue,
        repo_root: Path,
        source_branch: str,
        commit_shas: list[str],
    ) -> dict[str, Any] | None:
        unique_commit_shas = list(
            dict.fromkeys(str(sha).strip() for sha in commit_shas if str(sha).strip())
        )
        if not source_branch or not unique_commit_shas:
            return None

        base_ref = self.config.target_branch
        remote_base_ref = f"origin/{self.config.target_branch}"
        verify_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", remote_base_ref],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if verify_proc.returncode == 0:
            base_ref = remote_base_ref

        safe_run_id = re.sub(r"[^A-Za-z0-9._/-]+", "-", self.run_id).strip("-") or "boss"
        harvest_branch = f"aragora/boss-harvest/issue-{issue.number}-{safe_run_id}"
        with tempfile.TemporaryDirectory(prefix="aragora-boss-harvest-") as temp_dir:
            add_proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "worktree",
                    "add",
                    "--force",
                    "-B",
                    harvest_branch,
                    temp_dir,
                    base_ref,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if add_proc.returncode != 0:
                detail = (add_proc.stderr or add_proc.stdout or "").strip()
                raise RuntimeError(detail or f"git worktree add failed for {harvest_branch}")

            try:
                for sha in unique_commit_shas:
                    cherry_pick_proc = subprocess.run(
                        ["git", "-C", temp_dir, "cherry-pick", "-x", sha],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    if cherry_pick_proc.returncode != 0:
                        subprocess.run(
                            ["git", "-C", temp_dir, "cherry-pick", "--abort"],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                        detail = (cherry_pick_proc.stderr or cherry_pick_proc.stdout or "").strip()
                        raise RuntimeError(detail or f"git cherry-pick failed for {sha}")

                push_proc = subprocess.run(
                    ["git", "-C", temp_dir, "push", "-u", "origin", f"HEAD:{harvest_branch}"],
                    env=git_safe_env(),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                push_detail = (push_proc.stderr or push_proc.stdout or "").strip()
                if push_proc.returncode != 0:
                    logger.warning(
                        "Boss auto-harvest push failed for issue #%s branch %s: %s",
                        issue.number,
                        harvest_branch,
                        push_detail or "git push failed",
                    )
                return {
                    "action": "harvested",
                    "source_branch": source_branch,
                    "branch": harvest_branch,
                    "base_ref": base_ref,
                    "commit_shas": unique_commit_shas,
                    "pushed": push_proc.returncode == 0,
                    "push_error": None
                    if push_proc.returncode == 0
                    else push_detail or "git push failed",
                }
            finally:
                remove_proc = subprocess.run(
                    ["git", "-C", str(repo_root), "worktree", "remove", "--force", temp_dir],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if remove_proc.returncode != 0:
                    detail = (remove_proc.stderr or remove_proc.stdout or "").strip()
                    logger.warning(
                        "Boss auto-harvest cleanup failed for branch %s: %s",
                        harvest_branch,
                        detail or "git worktree remove failed",
                    )

    @staticmethod
    def _published_pr_url(worker_result: dict[str, Any]) -> str | None:
        publish_result = worker_result.get("publish_result")
        pr_url = str(
            publish_result.get("pr_url")
            if isinstance(publish_result, dict) and publish_result.get("pr_url")
            else worker_result.get("pr_url")
            or (
                worker_result.get("deliverable", {}).get("pr_url")
                if isinstance(worker_result.get("deliverable"), dict)
                else ""
            )
            or ""
        ).strip()
        return pr_url or None

    @staticmethod
    def _publish_result_succeeded(publish_result: Any) -> bool:
        return isinstance(publish_result, dict) and publish_result.get("published") is True

    @staticmethod
    def _published_deliverable_comment(worker_result: dict[str, Any]) -> str | None:
        publish_result = worker_result.get("publish_result")
        if not BossLoop._publish_result_succeeded(publish_result):
            return None
        pr_url = BossLoop._published_pr_url(worker_result)
        if pr_url is None:
            return None
        branch = str(
            publish_result.get("branch")
            or (
                worker_result.get("deliverable", {}).get("branch")
                if isinstance(worker_result.get("deliverable"), dict)
                else ""
            )
            or ""
        ).strip()
        action = str(publish_result.get("action", "")).strip()
        detail = str(publish_result.get("detail", "")).strip()
        lines = [
            "Aragora boss loop published a deliverable for human review.",
            "",
            f"- PR: {pr_url}",
        ]
        if branch:
            lines.append(f"- Branch: `{branch}`")
        if action:
            lines.append(f"- Publish action: `{action}`")
        if detail:
            lines.append(f"- Detail: {detail}")
        lines.extend(
            [
                "",
                "This status comment is updated in place on boss-loop retries.",
                _BOSS_PUBLISH_COMMENT_MARKER,
            ]
        )
        return "\n".join(lines)

    def _maybe_comment_published_deliverable(
        self,
        issue: GitHubIssue,
        worker_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        comment = self._published_deliverable_comment(worker_result)
        if comment is None:
            return None
        repo_slug = self._repo_slug_for_issue(issue)
        if repo_slug is None:
            return {
                "commented": False,
                "action": "skipped",
                "reason": "missing_repo_slug",
                "issue_number": issue.number,
            }
        try:
            from aragora.ralph.github_control import GitHubControl

            result = GitHubControl(repo_root=Path.cwd().resolve()).upsert_issue_comment(
                repo=repo_slug,
                issue_number=issue.number,
                body=comment,
                marker=_BOSS_PUBLISH_COMMENT_MARKER,
            )
        except Exception as exc:
            logger.warning("Boss publish comment failed for issue #%s: %s", issue.number, exc)
            return {
                "commented": False,
                "action": "comment_failed",
                "reason": type(exc).__name__,
                "issue_number": issue.number,
                "repo": repo_slug,
            }
        if not isinstance(result, dict):
            return {
                "commented": False,
                "action": "comment_failed",
                "reason": "invalid_comment_result",
                "issue_number": issue.number,
                "repo": repo_slug,
            }
        normalized = dict(result)
        normalized["issue_number"] = issue.number
        normalized["repo"] = repo_slug
        return normalized

    @staticmethod
    def _apply_postprocess_metadata(worker_result: dict[str, Any]) -> dict[str, Any]:
        receipt_metadata = worker_result.get("receipt_metadata")
        if not isinstance(receipt_metadata, dict):
            receipt_metadata = {}
            worker_result["receipt_metadata"] = receipt_metadata

        postprocess: dict[str, Any] = {}
        publish_result = worker_result.get("publish_result")
        if isinstance(publish_result, dict):
            normalized_publish = dict(publish_result)
            receipt_metadata["publish_result"] = normalized_publish
            postprocess["publish_result"] = normalized_publish
        harvest_result = worker_result.get("harvest_result")
        if isinstance(harvest_result, dict):
            normalized_harvest = dict(harvest_result)
            receipt_metadata["harvest_result"] = normalized_harvest
            postprocess["harvest_result"] = normalized_harvest
        debate_gate_result = worker_result.get("debate_gate_result")
        if isinstance(debate_gate_result, dict):
            normalized_gate = dict(debate_gate_result)
            receipt_metadata["debate_gate_result"] = normalized_gate
            postprocess["debate_gate_result"] = normalized_gate
        issue_comment_result = worker_result.get("issue_comment_result")
        if isinstance(issue_comment_result, dict):
            normalized_comment = dict(issue_comment_result)
            receipt_metadata["issue_comment_result"] = normalized_comment
            postprocess["issue_comment_result"] = normalized_comment
        issue_resolution = worker_result.get("issue_resolution")
        if isinstance(issue_resolution, dict):
            normalized_resolution = dict(issue_resolution)
            receipt_metadata["issue_resolution"] = normalized_resolution
            postprocess["issue_resolution"] = normalized_resolution
        for key in (
            "postprocess_promoted_from_status",
            "postprocess_promoted_from_outcome",
        ):
            value = receipt_metadata.get(key)
            if value is not None:
                postprocess[key] = value
        return postprocess

    @staticmethod
    def _promote_published_deliverable(worker_result: dict[str, Any]) -> bool:
        publish_result = worker_result.get("publish_result")
        if not BossLoop._publish_result_succeeded(publish_result):
            return False
        deliverable = worker_result.get("deliverable")
        if not isinstance(deliverable, dict):
            return False
        deliverable_type = str(deliverable.get("type", "")).strip().lower()
        if deliverable_type not in {"pr", "adopted_pr"}:
            return False
        if str(worker_result.get("status", "")).strip() != "needs_human":
            return False

        prior_status = str(worker_result.get("status", "")).strip()
        prior_outcome = str(worker_result.get("outcome", "")).strip()
        worker_result["status"] = "completed"
        worker_result["outcome"] = "pr_adopted"
        receipt_metadata = worker_result.get("receipt_metadata")
        if not isinstance(receipt_metadata, dict):
            receipt_metadata = {}
            worker_result["receipt_metadata"] = receipt_metadata
        receipt_metadata["postprocess_promoted_from_status"] = prior_status or None
        receipt_metadata["postprocess_promoted_from_outcome"] = prior_outcome or None
        return True

    @staticmethod
    def _published_pr_followup(worker_result: dict[str, Any]) -> str | None:
        publish_result = worker_result.get("publish_result")
        if not isinstance(publish_result, dict):
            return None
        pr_url = BossLoop._published_pr_url(worker_result)
        if not pr_url:
            return None
        action = str(publish_result.get("action", "")).strip()
        if action in {"existing_pr", "discovered_after_push"}:
            return (
                f"Auto-continuing: existing PR {pr_url} captures the deliverable for human review."
            )
        if action == "pr_created":
            return f"Auto-continuing: published PR {pr_url} for human review."
        return f"Auto-continuing: deliverable is available at {pr_url} for human review."

    @staticmethod
    def _debate_gate_followup(worker_result: dict[str, Any]) -> str | None:
        gate_result = worker_result.get("debate_gate_result")
        if not isinstance(gate_result, dict):
            return None
        verdict = str(gate_result.get("verdict", "")).strip()
        if verdict != "blocked":
            return None
        reason = str(gate_result.get("reason", "")).strip()
        if not reason:
            reason = "human review required before PR publication."
        return f"Publish skipped by debate gate: {reason}"

    def _convert_pr_to_draft(self, worker_result: dict[str, Any]) -> None:
        """Convert a newly-created PR to draft so only the 5 required checks run.

        The boss loop later promotes draft PRs to ready once those checks pass
        (see ``_promote_ready_drafts``).  This avoids triggering all 32 CI
        workflows on every PR while 35+ PRs compete for runner time.
        """
        pr_url = self._published_pr_url(worker_result)
        if not pr_url:
            return
        pr_number = self._pr_number_from_url(pr_url)
        if pr_number is None:
            return
        repo = str(self.config.repo or "").strip()
        cmd: list[str] = ["gh", "pr", "ready", "--undo", str(pr_number)]
        if repo:
            cmd.extend(["-R", repo])
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0:
                logger.info(
                    "Converted PR #%d to draft to limit CI to required checks only",
                    pr_number,
                )
                worker_result["draft_converted"] = True
            else:
                detail = (proc.stderr or proc.stdout or "").strip()
                # If already a draft, treat as success
                if "already a draft" in detail.lower():
                    logger.debug("PR #%d already a draft", pr_number)
                    worker_result["draft_converted"] = True
                else:
                    logger.warning(
                        "Failed to convert PR #%d to draft: %s",
                        pr_number,
                        detail or "gh pr ready --undo failed",
                    )
        except Exception as exc:
            logger.warning("Exception converting PR #%d to draft: %s", pr_number, exc)

    def _promote_ready_drafts(self) -> list[int]:
        """Promote draft PRs whose 5 required checks have all passed.

        Returns the list of PR numbers that were promoted.
        """
        repo = str(self.config.repo or "").strip()
        if not repo:
            return []

        # List open draft PRs authored by the current user (or any, if no filter)
        list_cmd: list[str] = [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--draft",
            "--json",
            "number,headRefName",
            "--limit",
            "100",
            "-R",
            repo,
        ]
        try:
            list_proc = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if list_proc.returncode != 0:
                logger.debug(
                    "Failed to list draft PRs for promotion: %s",
                    (list_proc.stderr or "").strip(),
                )
                return []
            draft_prs = json.loads(list_proc.stdout or "[]")
        except Exception as exc:
            logger.debug("Exception listing draft PRs for promotion: %s", exc)
            return []

        promoted: list[int] = []
        for pr_entry in draft_prs:
            pr_num = pr_entry.get("number")
            if not isinstance(pr_num, int):
                continue
            head_ref_name = pr_entry.get("headRefName")
            ownership = self._draft_promotion_ownership(head_ref_name)
            if ownership is None:
                logger.debug(
                    "Skipping draft PR #%d promotion: head ref %r is not queue-owned or boss-owned",
                    pr_num,
                    head_ref_name,
                )
                continue
            if self._all_required_checks_passed(pr_num, repo):
                ready_cmd: list[str] = [
                    "gh",
                    "pr",
                    "ready",
                    str(pr_num),
                    "-R",
                    repo,
                ]
                try:
                    ready_proc = subprocess.run(
                        ready_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    if ready_proc.returncode == 0:
                        logger.info(
                            "Promoted draft PR #%d to ready — all %d required checks passed",
                            pr_num,
                            len(_REQUIRED_CHECK_NAMES),
                        )
                        promoted.append(pr_num)
                    else:
                        logger.debug(
                            "Failed to promote PR #%d: %s",
                            pr_num,
                            (ready_proc.stderr or "").strip(),
                        )
                except Exception as exc:
                    logger.debug("Exception promoting PR #%d: %s", pr_num, exc)
        return promoted

    def _drain_deferred_publish_queue(self) -> int:
        """Retry deferred branch publishes now that PR slots may have opened.

        Returns count of successfully published branches.
        """
        if not self._deferred_publish_queue:
            return 0
        published = 0
        remaining: list[tuple[Any, dict[str, Any]]] = []
        for issue, worker_result in self._deferred_publish_queue:
            result = self._maybe_publish_deliverable(issue, worker_result)
            if result is not None and result.get("published"):
                published += 1
                logger.info(
                    "Deferred publish succeeded for issue #%s: %s",
                    issue.number,
                    result.get("pr_url", "branch pushed"),
                )
            else:
                remaining.append((issue, worker_result))
        self._deferred_publish_queue = remaining
        if published:
            logger.info(
                "Drained %d deferred publishes, %d remaining",
                published,
                len(remaining),
            )
        return published

    @staticmethod
    def _draft_promotion_ownership(head_ref_name: object) -> str | None:
        """Classify whether a draft PR is explicitly owned by boss-loop drafting."""
        if not isinstance(head_ref_name, str):
            return None
        normalized = head_ref_name.strip()
        if normalized.startswith("aragora/boss-harvest/issue-"):
            return "boss-owned"
        if normalized.startswith("codex/swarm-"):
            return "queue-owned"
        return None

    @staticmethod
    def _all_required_checks_passed(pr_number: int, repo: str) -> bool:
        """Return True if all 5 required CI checks have conclusion==success."""
        checks_cmd: list[str] = [
            "gh",
            "pr",
            "checks",
            str(pr_number),
            "--json",
            "name,state,conclusion",
            "-R",
            repo,
        ]
        try:
            proc = subprocess.run(
                checks_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                return False
            checks = json.loads(proc.stdout or "[]")
        except Exception:
            return False

        # Build a map of check name -> conclusion
        check_conclusions: dict[str, str] = {}
        for check in checks:
            name = str(check.get("name", "")).strip()
            conclusion = str(check.get("conclusion", "")).strip().upper()
            if name in _REQUIRED_CHECK_NAMES:
                check_conclusions[name] = conclusion

        # All 5 must be present and successful
        for required in _REQUIRED_CHECK_NAMES:
            if check_conclusions.get(required) != "SUCCESS":
                return False
        return True

    def _postprocess_issue_result(
        self,
        issue: GitHubIssue,
        worker_result: dict[str, Any],
    ) -> dict[str, Any]:
        publish_result = self._maybe_publish_deliverable(issue, worker_result)
        if publish_result is not None:
            worker_result["publish_result"] = publish_result
            # Queue deferred publishes for retry later when PR slots open
            if str(publish_result.get("action", "")).startswith("deferred"):
                self._deferred_publish_queue.append((issue, worker_result))
        issue_comment_result = self._maybe_comment_published_deliverable(issue, worker_result)
        if issue_comment_result is not None:
            worker_result["issue_comment_result"] = issue_comment_result
        issue_resolution = self._maybe_auto_close_already_done_issue(issue, worker_result)
        if issue_resolution is not None:
            worker_result["issue_resolution"] = issue_resolution
        self._apply_postprocess_metadata(worker_result)
        self._promote_published_deliverable(worker_result)
        # Convert newly-published PRs to draft so only the 5 required CI
        # checks run.  They will be promoted to ready by _promote_ready_drafts
        # once those checks pass.
        self._convert_pr_to_draft(worker_result)
        return worker_result

    def _log_value_outcome(
        self,
        issue_dict: dict[str, Any],
        worker_status: str,
        elapsed_seconds: float,
    ) -> None:
        """Log outcome for value-per-cost calibration and cross-loop signals."""
        issue_num = issue_dict.get("number", 0)
        try:
            from aragora.swarm.value_estimator import OutcomeRecord, log_outcome

            log_outcome(
                OutcomeRecord(
                    issue_number=issue_num,
                    predicted_score=0.0,
                    predicted_p_success=0.0,
                    did_merge=worker_status == "completed",
                    needed_human_rescue=worker_status == "needs_human",
                    actual_minutes=elapsed_seconds / 60.0,
                    worker_status=worker_status,
                )
            )
        except Exception as exc:
            logger.debug("Value outcome logging skipped: %s", exc)

        # Emit cross-loop outcome signal
        try:
            from aragora.swarm.outcome_signals import OutcomeSignal, get_signal_bus

            get_signal_bus().emit(
                OutcomeSignal(
                    source_loop="boss",
                    signal_type="completed" if worker_status == "completed" else "failed",
                    entity_id=str(issue_num),
                    entity_title=issue_dict.get("title", ""),
                    elapsed_seconds=elapsed_seconds,
                    did_merge=worker_status == "completed",
                    needed_human_rescue=worker_status == "needs_human",
                    failure_reason=worker_status if worker_status != "completed" else "",
                )
            )
        except Exception as exc:
            logger.debug("Outcome signal emission skipped: %s", exc)

    def _emit_lane_receipt(
        self,
        worker_result: dict[str, Any],
        issue_dict: dict[str, Any],
        elapsed: float,
    ) -> str | None:
        try:
            from aragora.receipts.lane import LaneCompletionReceipt, emit_lane_receipt

            terminal_outcome = str(worker_result.get("outcome", "")).strip().lower()
            deliverable = worker_result.get("deliverable")
            deliverable_present = isinstance(deliverable, dict) and bool(deliverable)
            if terminal_outcome in {
                "deliverable_created",
                "pr_adopted",
                "issue_already_resolved",
            }:
                receipt_outcome = "pass"
            elif deliverable_present and terminal_outcome in {"crash", "timeout"}:
                receipt_outcome = "blocked"
            elif terminal_outcome in {
                "needs_human",
                "blocked",
                "clean_exit_no_deliverable",
                "preview_only",
            }:
                receipt_outcome = "blocked"
            elif terminal_outcome in {"crash", "timeout"}:
                receipt_outcome = "fail"
            else:
                receipt_outcome = "unknown"

            receipt = LaneCompletionReceipt(
                task_id=str(issue_dict.get("number", "")),
                lease_id=str(worker_result.get("lease_id", self.run_id)),
                agent_id=str(worker_result.get("agent_id", "boss-loop")),
                base_sha=worker_result.get("base_sha"),
                head_sha=worker_result.get("head_sha"),
                changed_files=list(worker_result.get("changed_files", [])),
                validations_run=list(worker_result.get("validations_run", [])),
                outcome=receipt_outcome,
                risks=list(worker_result.get("risks", [])),
                pr_url=worker_result.get("pr_url"),
                pr_number=worker_result.get("pr_number"),
                branch=worker_result.get("branch"),
                duration_seconds=elapsed,
                metadata={
                    **dict(worker_result.get("receipt_metadata") or {}),
                    "terminal_outcome": terminal_outcome or None,
                    "worker_receipt_id": worker_result.get("receipt_id"),
                    "blocked_reasons": list(worker_result.get("reasons", [])),
                },
            )
            receipt_id = emit_lane_receipt(receipt)
            self._record_lane_telemetry(worker_result, issue_dict, elapsed, receipt_id)
            return receipt_id
        except Exception as exc:
            logger.debug("Lane receipt emission skipped: %s", exc)
            self._record_lane_telemetry(worker_result, issue_dict, elapsed, None)
            return None

    def _record_lane_telemetry(
        self,
        worker_result: dict[str, Any],
        issue_dict: dict[str, Any],
        elapsed: float,
        lane_receipt_id: str | None,
    ) -> None:
        terminal_outcome = str(worker_result.get("outcome", "")).strip().lower()
        deliverable = worker_result.get("deliverable")
        deliverable_type = ""
        pr_url = ""
        pr_number: int | None = None
        if isinstance(deliverable, dict):
            deliverable_type = str(deliverable.get("type", "")).strip()
            pr_url = str(
                deliverable.get("pr_url")
                or worker_result.get("pr_url")
                or deliverable.get("adopted_pr")
                or ""
            ).strip()
        if isinstance(worker_result.get("pr_number"), int):
            pr_number = int(worker_result["pr_number"])
        if not terminal_outcome:
            terminal_outcome, normalized_deliverable_type = _qualify_worker_result_terminal_state(
                worker_result
            )
            if normalized_deliverable_type:
                deliverable_type = normalized_deliverable_type
            if not terminal_outcome:
                terminal_outcome = "unknown"
        receipt_id = str(lane_receipt_id or worker_result.get("receipt_id") or "").strip()
        false_success_candidate = (
            terminal_outcome
            in {
                "deliverable_created",
                "pr_adopted",
            }
            and not deliverable_type
        )
        try:
            _LANE_TELEMETRY.record_lane(
                LaneTelemetryRecord(
                    lane_kind="boss_dispatch",
                    lane_id=str(
                        worker_result.get("run_id")
                        or worker_result.get("lease_id")
                        or issue_dict.get("number")
                        or ""
                    ).strip(),
                    run_id=str(worker_result.get("run_id", "")).strip(),
                    task_id=str(issue_dict.get("number", "")).strip(),
                    terminal_outcome=terminal_outcome,
                    worker_outcome=str(worker_result.get("worker_outcome", "")).strip(),
                    deliverable_type=deliverable_type,
                    receipt_id=receipt_id,
                    human_intervention_required=terminal_outcome
                    not in {
                        "deliverable_created",
                        "pr_adopted",
                        "preview_only",
                        "issue_already_resolved",
                    },
                    duration_seconds=float(elapsed or 0.0),
                    pr_url=pr_url,
                    pr_number=pr_number,
                    false_success_candidate=false_success_candidate,
                    metadata={
                        "issue_title": str(issue_dict.get("title", "")).strip() or None,
                        "worker_status": str(worker_result.get("status", "")).strip() or None,
                        "reasons": list(worker_result.get("reasons", []) or []),
                    },
                )
            )
        except Exception:
            logger.debug("Boss lane telemetry emission skipped", exc_info=True)

    def _emit_live_status(self, on_status: Any | None, status: BossIterationStatus) -> None:
        if on_status is None:
            return
        status = self._decorate_iteration_status(status)
        try:
            on_status(status)
        except Exception:
            pass

    async def run(
        self,
        *,
        on_status: Any | None = None,
    ) -> BossLoopResult:
        """Run the Boss loop until a stop condition is met.

        Args:
            on_status: Optional callback ``(BossIterationStatus) -> None``
                called after each iteration for live reporting.

        Returns:
            BossLoopResult with the final summary.
        """
        start_time = time.monotonic()
        iteration = 0
        logger.info(
            "Boss loop starting: configured_max_parallel_dispatches=%d",
            self._configured_parallel_dispatches,
        )

        # Clean stale supervisor runs that would block dispatch via
        # duplicate_open_work_order detection.  Previous runs with
        # needs_human/discarded work orders accumulate across sessions
        # and permanently block new dispatches for the same file scopes.
        try:
            from aragora.nomic.dev_coordination import DevCoordinationStore

            store = DevCoordinationStore()
            cleaned = store.cleanup_stale_supervisor_runs(max_age_hours=0.25)
            if cleaned:
                logger.info("Cleaned %d stale supervisor runs before starting boss loop", cleaned)
            archived_leasing_failures = store.archive_work_order_leasing_failed_work_orders(
                grace_period_hours=0.0
            )
            if archived_leasing_failures:
                logger.info(
                    "Archived %d stale work_order_leasing_failed lanes before starting boss loop",
                    archived_leasing_failures,
                )
        except Exception:
            logger.debug("Stale supervisor run cleanup skipped", exc_info=True)

        while iteration < self.config.max_iterations:
            iteration += 1

            # Refresh runner heartbeats so registrations do not go stale
            # while the boss loop is running continuously.
            self._refresh_runner_heartbeats()

            statuses = await self._run_iteration_statuses(iteration, on_status=on_status)
            statuses = [self._decorate_iteration_status(status) for status in statuses]
            self._iteration_statuses.extend(statuses)

            for status in statuses:
                self._emit_live_status(on_status, status)

            terminal_status = next(
                (
                    status
                    for status in statuses
                    if status.stop_reason
                    and status.stop_reason != BossStopReason.STILL_RUNNING.value
                ),
                None,
            )
            if terminal_status is not None:
                self._stop_reason = terminal_status.stop_reason
                break

            # Periodic status logging
            if iteration % self.config.status_report_interval == 0:
                logger.info(
                    "Boss loop iteration %d/%d: attempted=%d completed=%d failed=%d",
                    iteration,
                    self.config.max_iterations,
                    len(self._attempted_issues),
                    len(self._completed_issues),
                    len(self._failed_issues),
                )

            # Promote draft PRs whose required checks have passed.
            # This converts them to ready-for-review, triggering the
            # full CI suite only when the 5 fast required checks pass.
            try:
                promoted = self._promote_ready_drafts()
                if promoted:
                    logger.info(
                        "Promoted %d draft PR(s) to ready: %s",
                        len(promoted),
                        promoted,
                    )
            except Exception:
                logger.debug("Draft PR promotion check skipped", exc_info=True)

            # Retry deferred branch publishes now that slots may have opened
            try:
                self._drain_deferred_publish_queue()
            except Exception:
                logger.debug("Deferred publish drain skipped", exc_info=True)

            if self._maybe_auto_update(iteration):
                break

            # Inter-iteration sleep (skipped after last iteration)
            if iteration < self.config.max_iterations:
                import asyncio

                await asyncio.sleep(self.config.iteration_interval_seconds)

        if not self._stop_reason:
            self._stop_reason = BossStopReason.MAX_ITERATIONS.value

        total_elapsed = time.monotonic() - start_time
        result = BossLoopResult(
            run_id=self.run_id,
            iterations_completed=iteration,
            total_elapsed_seconds=total_elapsed,
            stop_reason=self._stop_reason,
            issues_attempted=list(self._attempted_issues),
            issues_completed=list(self._completed_issues),
            issues_failed=list(self._failed_issues),
            iteration_statuses=[s.to_dict() for s in self._iteration_statuses],
            needs_human_reasons=self._collect_needs_human_reasons(),
            next_actions=self._derive_next_actions(),
            sanitation_summary=list(self._last_sanitation_summary),
            configured_max_parallel_dispatches=self._configured_parallel_dispatches,
            effective_parallel_dispatches_observed=self._max_effective_parallel_dispatches_observed,
        )
        self._emit_terminal_receipt(result)
        return result

    async def _run_iteration_statuses(
        self,
        iteration: int,
        *,
        on_status: Any | None = None,
    ) -> list[BossIterationStatus]:
        if int(self.config.max_parallel_dispatches or 1) <= 1:
            return [await self._run_iteration(iteration, on_status=on_status)]
        return await self._run_iteration_batch(iteration, on_status=on_status)

    async def _run_iteration(
        self,
        iteration: int,
        *,
        on_status: Any | None = None,
    ) -> BossIterationStatus:
        """Execute a single Boss loop iteration."""
        now = datetime.now(UTC).isoformat()
        iter_start = time.monotonic()
        freshness_dict: dict[str, Any] = {}
        self._current_effective_parallel_dispatches = 1
        # Step 1: Fetch issues from GitHub
        try:
            issues = self._feed.fetch()
        except Exception as exc:
            logger.warning("Issue feed error: %s", exc)
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=now,
                runner_freshness=freshness_dict,
                selected_issue=None,
                worker_status="blocked",
                stop_reason=BossStopReason.ISSUE_FEED_ERROR.value,
                needs_human_reasons=["GitHub issue feed is unreachable."],
                next_actions=["Check GitHub CLI authentication and network."],
                elapsed_seconds=time.monotonic() - iter_start,
                error="issue_feed_error",
            )

        # Step 2: Select eligible issue
        # Skip issues that have exceeded retry limits and auto-label them
        already_maxed = self._already_maxed_issue_numbers(issues)
        blocked_scopes = self._blocked_issue_scopes()
        pending_handoffs = self._pending_handoff_candidates(
            issues,
            blocked_scopes=blocked_scopes,
        )
        pending_issue_numbers = {issue.number for issue in pending_handoffs}
        candidate_issues = [
            i for i in issues if i.number in pending_issue_numbers or i.number not in already_maxed
        ]
        eligibility_report = build_issue_eligibility_report(
            candidate_issues,
            skip_labels=self.config.skip_labels,
            require_labels=self.config.require_labels,
            blocked_scopes=blocked_scopes,
        )
        self._log_issue_skip_summary(eligibility_report)
        if pending_handoffs:
            selected: GitHubIssue | None = pending_handoffs[0]
        elif self.config.issue_number is not None:
            target_issue = next(
                (issue for issue in candidate_issues if issue.number == self.config.issue_number),
                None,
            )
            selected = (
                select_eligible_issue(
                    [target_issue],
                    skip_labels=self.config.skip_labels,
                    require_labels=self.config.require_labels,
                    blocked_scopes=blocked_scopes,
                )
                if target_issue is not None
                else None
            )
        else:
            selected = select_eligible_issue(
                candidate_issues,
                skip_labels=self.config.skip_labels,
                require_labels=self.config.require_labels,
                blocked_scopes=blocked_scopes,
                use_value_ranking=self.config.use_value_ranking,
            )

        if selected is None:
            if self.config.issue_number is not None:
                needs_human_reasons, next_actions = self._target_issue_miss_guidance(
                    self.config.issue_number
                )
            else:
                needs_human_reasons, next_actions = self._no_suitable_issue_guidance(
                    already_maxed=already_maxed,
                    report=eligibility_report,
                )
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=now,
                runner_freshness=freshness_dict,
                selected_issue=None,
                worker_status="idle",
                stop_reason=BossStopReason.NO_SUITABLE_ISSUE.value,
                needs_human_reasons=needs_human_reasons,
                next_actions=next_actions,
                elapsed_seconds=time.monotonic() - iter_start,
            )

        # Step 3: Check runner freshness only when there is eligible work to dispatch
        # Circuit breaker: skip decomposed issues if budget exhausted or fast-fail detected
        is_selected_decomposed = bool(re.search(r"\[from #\d+\]", selected.title or ""))
        if is_selected_decomposed:
            if self._ticks_spent_on_decomposed_issues >= self.config.max_decomposed_issue_ticks:
                logger.warning(
                    "Skipping decomposed issue #%s: tick budget (%d/%d) exhausted",
                    selected.number,
                    self._ticks_spent_on_decomposed_issues,
                    self.config.max_decomposed_issue_ticks,
                )
                return BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=now,
                    runner_freshness={},
                    selected_issue={"number": selected.number, "title": selected.title},
                    worker_status="skipped",
                    stop_reason=None,
                    needs_human_reasons=[],
                    next_actions=["Decomposed-issue tick budget exhausted; skipping."],
                    elapsed_seconds=time.monotonic() - iter_start,
                )
            window = self.config.fast_fail_circuit_breaker_window
            threshold = self.config.fast_fail_threshold_seconds
            if len(self._recent_elapsed) >= window and all(
                e < threshold for e in self._recent_elapsed[-window:]
            ):
                logger.warning(
                    "Fast-fail circuit breaker: last %d iterations all under %.0fs, skipping decomposed #%s",
                    window,
                    threshold,
                    selected.number,
                )
                return BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=now,
                    runner_freshness={},
                    selected_issue={"number": selected.number, "title": selected.title},
                    worker_status="skipped",
                    stop_reason=None,
                    needs_human_reasons=[],
                    next_actions=[
                        "Fast-fail circuit breaker triggered; skipping decomposed issue."
                    ],
                    elapsed_seconds=time.monotonic() - iter_start,
                )

        freshness = self._freshness_checker(
            freshness_ttl_seconds=self.config.freshness_ttl_seconds,
            registry_path=self.config.registry_path,
            env=self._env,
            requested_runner_type=self._requested_runner_type_for_freshness([selected]),
            allowed_profiles=self.config.allowed_runner_profiles,
            rotation_interval_seconds=self.config.runner_rotation_interval_seconds,
            verified_runner_target=self.config.verified_runner_target,
            runner_probe_limit=self.config.runner_probe_limit,
        )
        freshness_dict = _freshness_to_dict(freshness)

        if not _freshness_is_fresh(freshness, freshness_dict):
            blocked_reason = (
                freshness.blocked_reason
                if hasattr(freshness, "blocked_reason")
                else freshness_dict.get("blocked_reason", "runner_not_fresh")
            )
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=now,
                runner_freshness=freshness_dict,
                selected_issue=None,
                worker_status="blocked",
                stop_reason=BossStopReason.NO_FRESH_RUNNER.value,
                needs_human_reasons=[f"No fresh runner: {blocked_reason}"],
                next_actions=[
                    "Re-register or refresh the Codex runner before resuming the Boss loop.",
                    f"Blocked reason: {blocked_reason}",
                ],
                elapsed_seconds=time.monotonic() - iter_start,
            )

        # Step 3b: Pre-dispatch guard — skip issues that already have an open PR
        existing_pr_status = self._existing_open_pr_skip_status(
            iteration=iteration,
            timestamp=now,
            runner_freshness=freshness_dict,
            issue=selected,
            elapsed_seconds=time.monotonic() - iter_start,
        )
        if existing_pr_status is not None:
            return existing_pr_status

        # Step 4: Dispatch supervised work for this issue
        issue_dict = self._issue_payload(selected)
        self._attempted_issues.append(issue_dict)
        self._issue_attempt_counts[selected.number] = (
            self._issue_attempt_counts.get(selected.number, 0) + 1
        )
        requested_target_agent = (
            self._pending_handoff_prompts.get(selected.number, (None, None))[1]
            or self._requested_target_agent_for_issue(selected.number)
            or self.config.default_target_agent
            or ""
        )
        self._emit_live_status(
            on_status,
            BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=now,
                runner_freshness=freshness_dict,
                selected_issue=issue_dict,
                worker_status="dispatching",
                stop_reason=None,
                needs_human_reasons=[],
                next_actions=[
                    f"Dispatching issue #{selected.number} with "
                    f"{str(requested_target_agent).strip() or 'default routing'}."
                ],
                elapsed_seconds=time.monotonic() - iter_start,
            ),
        )

        worker_result = await self._dispatch_issue(selected, freshness)
        return self._finalize_worker_result(
            iteration=iteration,
            timestamp=now,
            runner_freshness=freshness_dict,
            issue=selected,
            issue_dict=issue_dict,
            worker_result=worker_result,
            elapsed_seconds=time.monotonic() - iter_start,
        )

    async def _run_iteration_batch(
        self,
        iteration: int,
        *,
        on_status: Any | None = None,
    ) -> list[BossIterationStatus]:
        import asyncio

        now = datetime.now(UTC).isoformat()
        iter_start = time.monotonic()
        freshness_dict: dict[str, Any] = {}
        self._current_effective_parallel_dispatches = None

        try:
            issues = self._feed.fetch()
        except Exception as exc:
            logger.warning("Issue feed error: %s", exc)
            return [
                BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=now,
                    runner_freshness=freshness_dict,
                    selected_issue=None,
                    worker_status="blocked",
                    stop_reason=BossStopReason.ISSUE_FEED_ERROR.value,
                    needs_human_reasons=["GitHub issue feed is unreachable."],
                    next_actions=["Check GitHub CLI authentication and network."],
                    elapsed_seconds=time.monotonic() - iter_start,
                    error="issue_feed_error",
                )
            ]

        already_maxed = self._already_maxed_issue_numbers(issues)
        blocked_scopes = self._blocked_issue_scopes()
        pending_handoffs = self._pending_handoff_candidates(
            issues,
            blocked_scopes=blocked_scopes,
        )
        pending_issue_numbers = {issue.number for issue in pending_handoffs}
        candidate_issues = [
            i for i in issues if i.number in pending_issue_numbers or i.number not in already_maxed
        ]
        eligibility_report = build_issue_eligibility_report(
            candidate_issues,
            skip_labels=self.config.skip_labels,
            require_labels=self.config.require_labels,
            blocked_scopes=blocked_scopes,
        )
        self._log_issue_skip_summary(eligibility_report)
        ordered_candidates = pending_handoffs + [
            issue for issue in candidate_issues if issue.number not in pending_issue_numbers
        ]
        selected_issues = self._select_issues_for_iteration(
            ordered_candidates,
            limit=None,
            blocked_scopes=blocked_scopes,
        )
        selected_issues = self._filter_mixed_retry_routing_batch(selected_issues)
        serialize_retry_routed_batch = self._selected_issues_need_retry_routing(selected_issues)
        if serialize_retry_routed_batch and len(selected_issues) > 1:
            logger.info(
                "Boss loop serializing retry-routed parallel batch: selected=%s",
                [issue.number for issue in selected_issues],
            )
            selected_issues = selected_issues[:1]

        if not selected_issues:
            if self.config.issue_number is not None:
                needs_human_reasons, next_actions = self._target_issue_miss_guidance(
                    self.config.issue_number
                )
            else:
                needs_human_reasons, next_actions = self._no_suitable_issue_guidance(
                    already_maxed=already_maxed,
                    report=eligibility_report,
                )
            return [
                BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=now,
                    runner_freshness=freshness_dict,
                    selected_issue=None,
                    worker_status="idle",
                    stop_reason=BossStopReason.NO_SUITABLE_ISSUE.value,
                    needs_human_reasons=needs_human_reasons,
                    next_actions=next_actions,
                    elapsed_seconds=time.monotonic() - iter_start,
                )
            ]

        freshness = self._freshness_checker(
            freshness_ttl_seconds=self.config.freshness_ttl_seconds,
            registry_path=self.config.registry_path,
            env=self._env,
            requested_runner_type=self._requested_runner_type_for_freshness(selected_issues),
            allowed_profiles=self.config.allowed_runner_profiles,
            rotation_interval_seconds=self.config.runner_rotation_interval_seconds,
        )
        freshness_dict = _freshness_to_dict(freshness)

        if not _freshness_is_fresh(freshness, freshness_dict):
            blocked_reason = (
                freshness.blocked_reason
                if hasattr(freshness, "blocked_reason")
                else freshness_dict.get("blocked_reason", "runner_not_fresh")
            )
            return [
                BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=now,
                    runner_freshness=freshness_dict,
                    selected_issue=None,
                    worker_status="blocked",
                    stop_reason=BossStopReason.NO_FRESH_RUNNER.value,
                    needs_human_reasons=[f"No fresh runner: {blocked_reason}"],
                    next_actions=[
                        "Re-register or refresh the Codex runner before resuming the Boss loop.",
                        f"Blocked reason: {blocked_reason}",
                    ],
                    elapsed_seconds=time.monotonic() - iter_start,
                )
            ]

        parallel_limit = self._parallel_dispatch_limit(freshness)
        if serialize_retry_routed_batch:
            parallel_limit = 1
        statuses: list[BossIterationStatus] = []
        dispatchable_issues: list[GitHubIssue] = []
        for issue in selected_issues:
            existing_pr_status = self._existing_open_pr_skip_status(
                iteration=iteration,
                timestamp=now,
                runner_freshness=freshness_dict,
                issue=issue,
                elapsed_seconds=time.monotonic() - iter_start,
            )
            if existing_pr_status is not None:
                statuses.append(existing_pr_status)
                continue
            dispatchable_issues.append(issue)
        if not dispatchable_issues:
            parallel_limit = 0
        self._current_effective_parallel_dispatches = parallel_limit
        logger.info(
            "Boss loop iteration %d parallel dispatches: configured=%d effective=%d",
            iteration,
            self._configured_parallel_dispatches,
            parallel_limit,
        )
        if not dispatchable_issues:
            return statuses

        pending_issues = list(dispatchable_issues)
        active_tasks: dict[
            asyncio.Task[dict[str, Any]], tuple[GitHubIssue, dict[str, Any], float]
        ] = {}
        stop_launching = False

        while pending_issues and len(active_tasks) < parallel_limit:
            issue = pending_issues.pop(0)
            issue_dict = self._issue_payload(issue)
            self._attempted_issues.append(issue_dict)
            self._issue_attempt_counts[issue.number] = (
                self._issue_attempt_counts.get(issue.number, 0) + 1
            )
            requested_target_agent = (
                self._pending_handoff_prompts.get(issue.number, (None, None))[1]
                or self._requested_target_agent_for_issue(issue.number)
                or self.config.default_target_agent
                or ""
            )
            self._emit_live_status(
                on_status,
                BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=now,
                    runner_freshness=freshness_dict,
                    selected_issue=issue_dict,
                    worker_status="dispatching",
                    stop_reason=None,
                    needs_human_reasons=[],
                    next_actions=[
                        f"Dispatching issue #{issue.number} with "
                        f"{str(requested_target_agent).strip() or 'default routing'}."
                    ],
                    elapsed_seconds=time.monotonic() - iter_start,
                ),
            )
            task = asyncio.create_task(self._dispatch_issue(issue, freshness))
            active_tasks[task] = (issue, issue_dict, time.monotonic())

        while active_tasks:
            done, _pending = await asyncio.wait(
                active_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                issue, issue_dict, started_at = active_tasks.pop(task)
                worker_result = task.result()
                status = self._finalize_worker_result(
                    iteration=iteration,
                    timestamp=now,
                    runner_freshness=freshness_dict,
                    issue=issue,
                    issue_dict=issue_dict,
                    worker_result=worker_result,
                    elapsed_seconds=time.monotonic() - started_at,
                )
                statuses.append(status)
                if status.stop_reason and status.stop_reason != BossStopReason.STILL_RUNNING.value:
                    stop_launching = True

                while not stop_launching and pending_issues and len(active_tasks) < parallel_limit:
                    next_issue = pending_issues.pop(0)
                    next_issue_dict = self._issue_payload(next_issue)
                    self._attempted_issues.append(next_issue_dict)
                    self._issue_attempt_counts[next_issue.number] = (
                        self._issue_attempt_counts.get(next_issue.number, 0) + 1
                    )
                    requested_target_agent = (
                        self._pending_handoff_prompts.get(next_issue.number, (None, None))[1]
                        or self._requested_target_agent_for_issue(next_issue.number)
                        or self.config.default_target_agent
                        or ""
                    )
                    self._emit_live_status(
                        on_status,
                        BossIterationStatus(
                            iteration=iteration,
                            run_id=self.run_id,
                            timestamp=now,
                            runner_freshness=freshness_dict,
                            selected_issue=next_issue_dict,
                            worker_status="dispatching",
                            stop_reason=None,
                            needs_human_reasons=[],
                            next_actions=[
                                f"Dispatching issue #{next_issue.number} with "
                                f"{str(requested_target_agent).strip() or 'default routing'}."
                            ],
                            elapsed_seconds=time.monotonic() - iter_start,
                        ),
                    )
                    next_task = asyncio.create_task(self._dispatch_issue(next_issue, freshness))
                    active_tasks[next_task] = (
                        next_issue,
                        next_issue_dict,
                        time.monotonic(),
                    )

        return statuses

    def _parallel_dispatch_limit(self, freshness: RunnerFreshnessResult) -> int:
        configured_limit = max(1, int(self.config.max_parallel_dispatches or 1))
        details = freshness.details if isinstance(freshness.details, dict) else {}
        routing = details.get("routing") if isinstance(details, dict) else {}
        selected_runners = routing.get("selected_runners") if isinstance(routing, dict) else None
        if not isinstance(selected_runners, list):
            return configured_limit
        available_capacity = 0
        any_capacity_reported = False
        for item in selected_runners:
            if not isinstance(item, dict):
                continue
            cap = max(0, int(item.get("available_capacity", 0) or 0))
            if cap > 0:
                any_capacity_reported = True
            available_capacity += cap
        if not any_capacity_reported:
            # Runners are selected (passed eligibility) but none report explicit
            # capacity — trust the configured parallel limit instead of degrading
            # to serial dispatch.
            return configured_limit
        return max(1, min(configured_limit, available_capacity))

    @staticmethod
    def _semantic_dedup_issues(issues: list[GitHubIssue]) -> list[GitHubIssue]:
        """Use LLM to cluster semantically duplicate issues, keep one per cluster."""
        if len(issues) < 6:
            return issues
        import json as _json
        import os
        import re

        try:
            from aragora.agents.base import create_agent

            agent = None
            if os.environ.get("OPENROUTER_API_KEY"):
                agent = create_agent(
                    "openrouter", name="dedup", role="proposer", model="deepseek/deepseek-chat"
                )
            elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                agent = create_agent(
                    "gemini", name="dedup", role="proposer", model="gemini-2.0-flash"
                )
            if agent is None:
                return issues

            issue_map = {
                str(i.number): re.sub(r"\[from #\d+\]\s*", "", i.title).strip() for i in issues
            }
            prompt = (
                "You are deduplicating GitHub issues. Group semantically equivalent tasks. "
                "Return ONLY a JSON array of arrays: [[num1,num2],[num3],...]\n\n"
                + "\n".join(f"#{num}: {title}" for num, title in issue_map.items())
            )

            import asyncio

            try:
                asyncio.get_running_loop()
                return issues  # Can't call asyncio.run inside running loop
            except RuntimeError:
                pass

            raw = asyncio.run(agent.generate(prompt))
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return issues

            clusters = _json.loads(match.group())
            if not isinstance(clusters, list):
                return issues

            kept = {int(c[0]) for c in clusters if isinstance(c, list) and c}
            all_clustered = {int(n) for c in clusters if isinstance(c, list) for n in c}
            deduped = [i for i in issues if i.number in kept or i.number not in all_clustered]

            logger.info("Semantic dedup: %d → %d issues", len(issues), len(deduped))
            return deduped
        except Exception as exc:
            logger.debug("Semantic dedup skipped: %s", exc)
            return issues

    @staticmethod
    def _scope_hint_is_specific(scope_entry: str) -> bool:
        basename = str(scope_entry or "").rstrip("/").rsplit("/", 1)[-1]
        return "*" in scope_entry or "." in basename

    @staticmethod
    def _scope_hint_is_validation_command_scope(issue: GitHubIssue, scope_entry: str) -> bool:
        if BossLoop._scope_hint_is_specific(scope_entry):
            return False
        scope = str(scope_entry or "").strip().rstrip("/")
        if not scope:
            return False
        validation_command_re = re.compile(
            r"\b(pytest|ruff|mypy|npm|pnpm|yarn|python\s+-m\s+pytest)\b",
            re.IGNORECASE,
        )
        for line in str(issue.body or "").splitlines():
            normalized = line.strip().strip("-* ").replace("`", "").rstrip("/")
            if scope in normalized and validation_command_re.search(normalized):
                return True
        return False

    @staticmethod
    def _parallel_claim_scope_entries(issue: GitHubIssue) -> list[str]:
        return [
            scope_entry
            for scope_entry in infer_issue_scope_entries(issue)
            if not BossLoop._scope_hint_is_validation_command_scope(issue, scope_entry)
        ]

    @staticmethod
    def _has_explicit_parallel_lane_hint(issue: GitHubIssue) -> bool:
        if any(
            re.match(r"^(?:lane|area)[:/]", str(label or "").strip(), re.IGNORECASE)
            for label in issue.labels
        ):
            return True
        combined = "\n".join(
            part for part in (str(issue.title or "").strip(), str(issue.body or "").strip()) if part
        )
        return bool(re.search(r"(?im)^(?:lane|owner lane|lane id)\s*:", combined))

    def _select_issues_for_iteration(
        self,
        issues: list[GitHubIssue],
        *,
        limit: int | None,
        blocked_scopes: set[str] | None = None,
    ) -> list[GitHubIssue]:
        blocked_scope_entries = set(blocked_scopes or set())
        if limit is not None and limit <= 1:
            if self.config.issue_number is not None:
                target_issue = next(
                    (issue for issue in issues if issue.number == self.config.issue_number),
                    None,
                )
                selected = (
                    select_eligible_issue(
                        [target_issue],
                        skip_labels=self.config.skip_labels,
                        require_labels=self.config.require_labels,
                        blocked_scopes=blocked_scope_entries,
                    )
                    if target_issue is not None
                    else None
                )
                return [selected] if selected is not None else []
            selected = select_eligible_issue(
                issues,
                skip_labels=self.config.skip_labels,
                require_labels=self.config.require_labels,
                blocked_scopes=blocked_scope_entries,
            )
            return [selected] if selected is not None else []

        if self.config.issue_number is not None:
            target_issue = next(
                (issue for issue in issues if issue.number == self.config.issue_number),
                None,
            )
            selected = (
                select_eligible_issue(
                    [target_issue],
                    skip_labels=self.config.skip_labels,
                    require_labels=self.config.require_labels,
                    blocked_scopes=blocked_scope_entries,
                )
                if target_issue is not None
                else None
            )
            return [selected] if selected is not None else []

        # Semantic dedup: LLM clusters near-duplicate issues before dispatch.
        # Skip-labeled issues are excluded first so a quarantined issue like
        # `boss-stuck` cannot suppress an unlabeled twin in the same cluster.
        pre_dedup_issues = [
            issue
            for issue in issues
            if not (set(issue.labels) & set(self.config.skip_labels or set()))
        ]
        issues = self._semantic_dedup_issues(pre_dedup_issues)

        selected_issues: list[GitHubIssue] = []
        # Track file scopes to avoid dispatching overlapping work in parallel.
        claimed_scopes: set[str] = set(blocked_scope_entries)
        # Track coarse ownership lanes so one batch does not dispatch multiple
        # issues into the same domain when explicit metadata or scope hints exist.
        claimed_lanes: set[str] = set()
        # Track title stems to avoid dispatching near-duplicate sub-issues
        claimed_stems: set[str] = set()

        for issue in issues:
            candidate = select_eligible_issue(
                [issue],
                skip_labels=self.config.skip_labels,
                require_labels=self.config.require_labels,
                blocked_scopes=claimed_scopes,
            )
            if candidate is None:
                continue

            # Deduplicate by explicit edit scope. Broad validation commands such
            # as `pytest -q tests/swarm/` are not exclusive edit claims.
            claim_scope_hints = set(self._parallel_claim_scope_entries(candidate))
            if claim_scope_hints and claim_scope_hints & claimed_scopes:
                continue
            lane_hints = (
                set(infer_issue_lane_hints(candidate))
                if claim_scope_hints or self._has_explicit_parallel_lane_hint(candidate)
                else set()
            )
            if lane_hints and lane_hints & claimed_lanes:
                continue

            # Deduplicate by title stem — strip [from #N] prefix and compare
            import re

            stem = re.sub(r"\[from #\d+\]\s*", "", candidate.title).strip().lower()[:60]
            if stem in claimed_stems:
                continue

            selected_issues.append(candidate)
            claimed_scopes.update(claim_scope_hints)
            claimed_lanes.update(lane_hints)
            claimed_stems.add(stem)
            if limit is not None and len(selected_issues) >= limit:
                break
        return selected_issues

    def _finalize_worker_result(
        self,
        *,
        iteration: int,
        timestamp: str,
        runner_freshness: dict[str, Any],
        issue: GitHubIssue,
        issue_dict: dict[str, Any],
        worker_result: dict[str, Any],
        elapsed_seconds: float,
    ) -> BossIterationStatus:
        issue_number = int(issue.number)

        # Track decomposed-issue ticks and fast-fail patterns
        is_decomposed = bool(re.search(r"\[from #\d+\]", issue.title or ""))
        if is_decomposed:
            self._ticks_spent_on_decomposed_issues += 1

        # Update fast-fail window
        self._recent_elapsed.append(elapsed_seconds)
        if len(self._recent_elapsed) > self.config.fast_fail_circuit_breaker_window:
            self._recent_elapsed = self._recent_elapsed[
                -self.config.fast_fail_circuit_breaker_window :
            ]

        if worker_result.get("status") == "running":
            self._consecutive_failures = 0
            worker_run_id = str(worker_result.get("run_id", "")).strip()
            next_actions = [
                (
                    f"Supervisor run {worker_run_id} is active for issue #{issue.number}; "
                    "the boss loop returned after this bounded dispatch tick."
                )
                if worker_run_id
                else (
                    f"Issue #{issue.number} dispatched successfully; "
                    "the boss loop returned after this bounded dispatch tick."
                ),
                "Inspect the active supervisor run before starting another live boss-loop tick.",
            ]
            self._append_iteration_metrics(
                iteration=iteration,
                issue_number=issue_number,
                worker_result=worker_result,
                elapsed_seconds=elapsed_seconds,
            )
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=timestamp,
                runner_freshness=runner_freshness,
                selected_issue=issue_dict,
                worker_status="running",
                stop_reason=None,
                needs_human_reasons=[],
                next_actions=next_actions,
                elapsed_seconds=elapsed_seconds,
                worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
            )

        issue_resolution = worker_result.get("issue_resolution")
        if (
            isinstance(issue_resolution, dict)
            and str(issue_resolution.get("action", "")).strip() == "closed"
        ):
            self._completed_issues.append(issue_dict)
            self._consecutive_failures = 0
            self._emit_lane_receipt(worker_result, issue_dict, elapsed_seconds)
            self._log_value_outcome(issue_dict, "completed", elapsed_seconds)
            self._append_iteration_metrics(
                iteration=iteration,
                issue_number=issue_number,
                worker_result=worker_result,
                elapsed_seconds=elapsed_seconds,
            )
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=timestamp,
                runner_freshness=runner_freshness,
                selected_issue=issue_dict,
                worker_status="completed",
                stop_reason=None,
                needs_human_reasons=[],
                next_actions=["Issue auto-closed as already implemented; proceeding."],
                elapsed_seconds=elapsed_seconds,
                worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
            )

        if worker_result.get("status") == "completed":
            self._completed_issues.append(issue_dict)
            self._issue_attempt_counts[issue_number] = max(
                self._issue_attempt_counts.get(issue_number, 0),
                self.config.max_retries_per_issue,
            )
            self._consecutive_failures = 0
            self._emit_lane_receipt(worker_result, issue_dict, elapsed_seconds)
            self._log_value_outcome(issue_dict, "completed", elapsed_seconds)
            next_action = (
                self._published_pr_followup(worker_result)
                or self._debate_gate_followup(worker_result)
                or "Proceeding to next issue."
            )
            self._append_iteration_metrics(
                iteration=iteration,
                issue_number=issue_number,
                worker_result=worker_result,
                elapsed_seconds=elapsed_seconds,
            )
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=timestamp,
                runner_freshness=runner_freshness,
                selected_issue=issue_dict,
                worker_status="completed",
                stop_reason=None,
                needs_human_reasons=[],
                next_actions=[next_action],
                elapsed_seconds=elapsed_seconds,
            )

        if worker_result.get("status") == "needs_human":
            _terminal_outcome, normalized_deliverable_type = _qualify_worker_result_terminal_state(
                worker_result
            )
            has_deliverable = bool(normalized_deliverable_type)
            raw_deliverable = worker_result.get("deliverable")
            has_untyped_deliverable = isinstance(raw_deliverable, dict) and bool(raw_deliverable)
            pr_url = self._published_pr_url(worker_result)
            self._emit_lane_receipt(worker_result, issue_dict, elapsed_seconds)

            # Deliverable = terminal: the worker produced a concrete artifact
            # (commit, branch, or PR).  Do NOT retry the issue — it already
            # has work product that either needs review or was published.
            if has_deliverable:
                self._completed_issues.append(issue_dict)
                self._consecutive_failures = 0
                # Mark the issue as exhausted so it is never re-dispatched in
                # this loop run.
                self._issue_attempt_counts[issue_number] = max(
                    self._issue_attempt_counts.get(issue_number, 0),
                    self.config.max_retries_per_issue,
                )
                self._log_value_outcome(issue_dict, "completed", elapsed_seconds)
                logger.info(
                    "boss_loop_terminal_deliverable issue=#%s pr=%s deliverable_type=%s",
                    issue_dict.get("number", "?"),
                    pr_url or "(pending publish)",
                    normalized_deliverable_type,
                )
                self._append_iteration_metrics(
                    iteration=iteration,
                    issue_number=issue_number,
                    worker_result=worker_result,
                    elapsed_seconds=elapsed_seconds,
                )
                next_action = (
                    self._published_pr_followup(worker_result)
                    or self._debate_gate_followup(worker_result)
                    or (
                        f"Terminal: deliverable ({normalized_deliverable_type}) for issue "
                        f"#{issue_dict.get('number', '?')}"
                        f"{f' PR {pr_url}' if pr_url else ''}. Proceeding to next issue."
                    )
                )
                return BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=timestamp,
                    runner_freshness=runner_freshness,
                    selected_issue=issue_dict,
                    worker_status="completed",
                    stop_reason=None,
                    needs_human_reasons=[],
                    next_actions=[next_action],
                    elapsed_seconds=elapsed_seconds,
                    worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
                )

            if self.config.auto_continue_on_needs_human and has_deliverable:
                self._failed_issues.append(issue_dict)
                self._consecutive_failures = 0
                logger.info(
                    "boss_loop_auto_continue issue=#%s (recoverable deliverable still blocked)",
                    issue_dict.get("number", "?"),
                )
                self._append_iteration_metrics(
                    iteration=iteration,
                    issue_number=issue_number,
                    worker_result=worker_result,
                    elapsed_seconds=elapsed_seconds,
                )
                return BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=timestamp,
                    runner_freshness=runner_freshness,
                    selected_issue=issue_dict,
                    worker_status="needs_human",
                    stop_reason=None,
                    needs_human_reasons=worker_result.get(
                        "reasons",
                        ["Recovered deliverable requires human review before integration."],
                    ),
                    next_actions=[
                        "Auto-continuing: recovered deliverable is receipt-backed but still blocked on human review."
                    ],
                    elapsed_seconds=elapsed_seconds,
                    worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
                )
            self._failed_issues.append(issue_dict)
            self._log_value_outcome(issue_dict, "needs_human", elapsed_seconds)

            # Fix-forward: if verification failed and we haven't exhausted
            # repair attempts, re-dispatch with a targeted repair prompt
            issue_num = issue_dict.get("number", 0)
            repair_key = f"repair_{issue_num}"
            repair_count = self._issue_attempt_counts.get(repair_key, 0)
            reasons = worker_result.get("reasons", [])
            has_verification_failure = any(
                "verification failed" in str(r).lower()
                or "exit 1" in str(r).lower()
                or "test" in str(r).lower()
                for r in reasons
            )

            if (
                self.config.auto_continue_on_needs_human
                and has_verification_failure
                and repair_count < self.config.max_repair_attempts
            ):
                self._issue_attempt_counts[repair_key] = repair_count + 1
                logger.info(
                    "boss_loop_repair issue=#%s attempt=%d/%d (verification failed, dispatching fix)",
                    issue_num,
                    repair_count + 1,
                    self.config.max_repair_attempts,
                )
                # Don't count as consecutive failure — we're actively repairing
                self._append_iteration_metrics(
                    iteration=iteration,
                    issue_number=issue_number,
                    worker_result=worker_result,
                    elapsed_seconds=elapsed_seconds,
                )
                return BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=timestamp,
                    runner_freshness=runner_freshness,
                    selected_issue=issue_dict,
                    worker_status="repairing",
                    stop_reason=None,
                    needs_human_reasons=[],
                    next_actions=[
                        f"Repair attempt {repair_count + 1}/{self.config.max_repair_attempts} "
                        f"for issue #{issue_num} — fixing verification failures."
                    ],
                    elapsed_seconds=elapsed_seconds,
                    worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
                )

            # Ping-pong retry: dispatch to the OTHER agent with transcript context
            if self.config.enable_ping_pong_retry and not has_verification_failure:
                pp_key = f"pingpong_{issue_num}"
                pp_count = self._issue_attempt_counts.get(pp_key, 0)
                transcript = self._extract_worker_transcript(worker_result)
                if pp_count < 1 and len(transcript.strip()) > 50:
                    self._issue_attempt_counts[pp_key] = pp_count + 1
                    previous_agent = self._extract_worker_agent(worker_result) or "unknown"
                    rotation = list(self.config.model_rotation or ["claude", "codex"])
                    next_agent = rotation[0] if previous_agent == rotation[-1] else rotation[-1]

                    from aragora.swarm.ping_pong import build_handoff_prompt

                    handoff = build_handoff_prompt(
                        goal=f"[Issue #{issue_num}] {issue_dict.get('title', '')}",
                        previous_transcript=transcript,
                        previous_agent=previous_agent,
                        next_agent=next_agent,
                        round_number=1,
                        files_changed=self._extract_worker_files_changed(worker_result),
                        remaining_issues=[str(r) for r in reasons[:5]],
                    )
                    self._pending_handoff_prompts[issue_num] = (handoff, next_agent)
                    logger.info(
                        "boss_loop_ping_pong issue=#%s from=%s to=%s transcript_len=%d",
                        issue_num,
                        previous_agent,
                        next_agent,
                        len(transcript),
                    )
                    self._append_iteration_metrics(
                        iteration=iteration,
                        issue_number=issue_number,
                        worker_result=worker_result,
                        elapsed_seconds=elapsed_seconds,
                    )
                    return BossIterationStatus(
                        iteration=iteration,
                        run_id=self.run_id,
                        timestamp=timestamp,
                        runner_freshness=runner_freshness,
                        selected_issue=issue_dict,
                        worker_status="ping_pong_retry",
                        stop_reason=None,
                        needs_human_reasons=[],
                        next_actions=[
                            f"Ping-pong handoff: {previous_agent} → {next_agent} "
                            f"for issue #{issue_num}"
                        ],
                        elapsed_seconds=elapsed_seconds,
                        worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
                    )

            if self.config.auto_continue_on_needs_human:
                self._consecutive_failures += 1
                if has_untyped_deliverable:
                    logger.warning(
                        "boss_loop_skip issue=#%s "
                        "(needs_human, untyped deliverable, auto-continue on)",
                        issue_dict.get("number", "?"),
                    )
                    next_actions = [
                        "Auto-continuing: worker returned a deliverable that still needs human review."
                    ]
                else:
                    logger.warning(
                        "boss_loop_skip issue=#%s (needs_human, no deliverable, auto-continue on)",
                        issue_dict.get("number", "?"),
                    )
                    next_actions = ["Skipping to next issue (auto-continue mode)."]
                self._append_iteration_metrics(
                    iteration=iteration,
                    issue_number=issue_number,
                    worker_result=worker_result,
                    elapsed_seconds=elapsed_seconds,
                )
                return BossIterationStatus(
                    iteration=iteration,
                    run_id=self.run_id,
                    timestamp=timestamp,
                    runner_freshness=runner_freshness,
                    selected_issue=issue_dict,
                    worker_status="needs_human",
                    stop_reason=None,
                    needs_human_reasons=worker_result.get(
                        "reasons", ["Worker requires human input."]
                    ),
                    next_actions=next_actions,
                    elapsed_seconds=elapsed_seconds,
                    worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
                )
            next_actions = [
                str(item).strip()
                for item in worker_result.get("next_actions", [])
                if str(item).strip()
            ] or ["Review the worker output and decide next steps."]
            self._append_iteration_metrics(
                iteration=iteration,
                issue_number=issue_number,
                worker_result=worker_result,
                elapsed_seconds=elapsed_seconds,
            )
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=timestamp,
                runner_freshness=runner_freshness,
                selected_issue=issue_dict,
                worker_status="needs_human",
                stop_reason=BossStopReason.NEEDS_HUMAN.value,
                needs_human_reasons=worker_result.get("reasons", ["Worker requires human input."]),
                next_actions=next_actions,
                elapsed_seconds=elapsed_seconds,
                worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
            )

        self._failed_issues.append(issue_dict)
        self._consecutive_failures += 1
        self._emit_lane_receipt(worker_result, issue_dict, elapsed_seconds)
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            self._append_iteration_metrics(
                iteration=iteration,
                issue_number=issue_number,
                worker_result=worker_result,
                elapsed_seconds=elapsed_seconds,
            )
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=timestamp,
                runner_freshness=runner_freshness,
                selected_issue=issue_dict,
                worker_status="failed",
                stop_reason=BossStopReason.CONSECUTIVE_FAILURES.value,
                needs_human_reasons=[
                    f"Consecutive failures reached threshold ({self.config.max_consecutive_failures})."
                ],
                next_actions=[
                    "Investigate the last failures before resuming.",
                    f"Error: {worker_result.get('error', 'unknown')}",
                ],
                elapsed_seconds=elapsed_seconds,
                error=worker_result.get("error"),
                worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
            )

        self._append_iteration_metrics(
            iteration=iteration,
            issue_number=issue_number,
            worker_result=worker_result,
            elapsed_seconds=elapsed_seconds,
        )
        return BossIterationStatus(
            iteration=iteration,
            run_id=self.run_id,
            timestamp=timestamp,
            runner_freshness=runner_freshness,
            selected_issue=issue_dict,
            worker_status="failed",
            stop_reason=None,
            needs_human_reasons=[],
            next_actions=[
                f"Issue #{issue.number} failed (attempt "
                f"{self._issue_attempt_counts[issue.number]}/{self.config.max_retries_per_issue}). "
                "Will retry with next iteration.",
            ],
            elapsed_seconds=elapsed_seconds,
            error=worker_result.get("error"),
            worker_outcome=str(worker_result.get("outcome", "")).strip() or None,
        )

    async def _dispatch_issue(
        self,
        issue: GitHubIssue,
        freshness: RunnerFreshnessResult,
    ) -> dict[str, Any]:
        """Dispatch a supervised worker for the given issue.

        Returns a dict with at minimum ``{"status": "completed"|"failed"|"needs_human"}``.
        """
        from aragora.swarm.spec import SwarmSpec

        refinement: dict[str, Any] = {}
        refined_prompt = ""
        pending_handoff = self._pending_handoff_prompts.get(issue.number)
        if pending_handoff is not None:
            refined_prompt = str(pending_handoff[0]).strip()

        try:
            from aragora.swarm.prompt_refiner import (
                build_refinement_worker_env,
                refine_worker_prompt,
            )

            refinement = await refine_worker_prompt(
                issue.title,
                issue.body or "",
                repo_path=Path.cwd(),
            )
            refinement_worker_env = build_refinement_worker_env(refinement)
            if refinement.get("context_gathered"):
                if pending_handoff is None:
                    refined_prompt = str(refinement.get("refined_prompt", "")).strip()
                logger.info(
                    "Refined prompt for #%s: %d relevant files, %d test patterns",
                    issue.number,
                    len(refinement.get("files_to_change", [])),
                    len(refinement.get("test_patterns", [])),
                )
        except Exception as exc:
            logger.debug("Prompt refinement skipped: %s", exc)
            refinement_worker_env = {}

        body_lines = [str(line).strip() for line in str(issue.body or "").splitlines()]
        scope_hints = list(
            dict.fromkeys(
                [
                    *[
                        str(path).strip()
                        for path in refinement.get("files_to_change", [])
                        if str(path).strip()
                    ],
                    *SwarmSpec.infer_file_scope_hints(issue.body or ""),
                ]
            )
        )
        constraints = list(
            dict.fromkeys(
                [
                    *SwarmSpec.infer_constraints(body_lines),
                    *[
                        str(item).strip()
                        for item in refinement.get("constraints", [])
                        if str(item).strip()
                    ],
                ]
            )
        )
        goal = _compose_issue_dispatch_goal(
            issue.number,
            issue.title,
            issue_body=issue.body or "",
            refined_prompt=refined_prompt,
        )
        if "git add -A && git commit" not in goal:
            goal += (
                "\n\n## CRITICAL: You MUST commit your changes\n"
                "After making changes, run:\n"
                "```\ngit add -A && git commit -m 'fix: description of changes'\n```\n"
                "If you do not commit, your work will be lost."
            )

        spec = SwarmSpec(
            raw_goal=goal,
            refined_goal=goal,
            constraints=constraints,
            budget_limit_usd=self.config.budget_limit_usd,
            file_scope_hints=scope_hints,
            requires_approval=True,
            interrogation_turns=0,
            user_expertise="developer",
        )

        # Micro-decompose: convert broad issues into single-file work orders.
        # Workers succeed on focused tasks but fail on broad ones in large repos.
        if self.config.use_micro_decomposition and scope_hints:
            try:
                from aragora.swarm.micro_decomposer import build_micro_work_orders

                validation_contract_raw = extract_issue_validation_contract(issue.body)
                micro_orders = build_micro_work_orders(
                    goal=goal,
                    file_scope_hints=scope_hints,
                    acceptance_criteria=list(validation_contract_raw)
                    if validation_contract_raw
                    else None,
                    constraints=constraints,
                    repo_root=Path.cwd(),
                )
                if micro_orders:
                    spec.work_orders = micro_orders
                    # Clear spec-level hints so supervisor doesn't merge all
                    # files into every work order's scope
                    spec.file_scope_hints = []
                    logger.info(
                        "Micro-decomposed issue #%s into %d work orders",
                        issue.number,
                        len(micro_orders),
                    )
            except Exception as exc:
                logger.debug("Micro-decomposition skipped: %s", exc)

        validation_contract = extract_issue_validation_contract(issue.body)
        if validation_contract and self.config.use_focused_verification:
            # Replace broad test suite commands with focused verification
            # that only tests files related to the worker's changes
            focused_tests = discover_focused_tests(Path.cwd())
            focused = []
            for criterion in validation_contract:
                if _should_replace_with_focused_tests(criterion):
                    if focused_tests:
                        test_list = " ".join(focused_tests[:20])
                        focused.append(f"python -m pytest --timeout=30 -x -q {test_list}")
                    else:
                        # No focused tests found — keep the original criterion
                        # rather than running an empty pytest invocation
                        focused.append(criterion)
                else:
                    focused.append(criterion)
            spec.acceptance_criteria = focused
        elif validation_contract:
            spec.acceptance_criteria = list(validation_contract)

        if self.config.require_validation_contract and not bool(
            getattr(spec, "acceptance_criteria", None)
        ):
            return {
                "status": "needs_human",
                "reasons": [
                    f"Issue #{issue.number} lacks an explicit validation contract or acceptance criteria."
                ],
                "next_actions": [
                    "Add an Acceptance Criteria, Validation, Definition of Done, or Test Plan section to the issue body.",
                    "Include at least one concrete verification step such as a pytest command or observable success criterion.",
                ],
            }

        self._attach_issue_handoff_metadata(spec, issue)

        gate = await check_pre_dispatch_gate(
            issue.body or "",
            repo_root=Path.cwd(),
            use_llm=self.config.use_llm_pre_dispatch_gate,
        )
        logger.info(
            "pre_dispatch_gate issue=#%s method=%s pass=%s sanitation=%s missing=%s",
            issue.number,
            gate["method"],
            gate["pass"],
            gate["sanitation_ok"],
            gate.get("unresolved_missing", []),
        )
        if not gate["sanitation_ok"]:
            return {
                "status": "needs_human",
                "outcome": "sanitation_failed",
                "reasons": [
                    f"Issue #{issue.number} failed sanitation: {gate.get('sanitation_reason', 'unknown')}"
                ],
                "next_actions": [
                    "Rewrite the issue body with a clear task description.",
                ],
            }
        if gate.get("unresolved_missing"):
            if gate.get("missing_targets") and not gate["unresolved_missing"]:
                logger.info(
                    "boss_loop_missing_validation_targets_allowed issue=#%s targets=%s",
                    issue.number,
                    ", ".join(gate["missing_targets"]),
                )
            else:
                targets_text = ", ".join(gate["unresolved_missing"])
                return {
                    "status": "needs_human",
                    "outcome": "verification_target_missing",
                    "reasons": [
                        f"Issue #{issue.number} references missing validation targets: {targets_text}"
                    ],
                    "next_actions": [
                        "Refresh the issue's Acceptance Criteria or Test Plan so pytest points at current repo paths.",
                        "Update the Files/Reference section or add explicit work orders before rerunning Boss dispatch.",
                    ],
                }

        if not spec.is_dispatch_bounded():
            return {
                "status": "needs_human",
                "reasons": [
                    f"Issue #{issue.number} is not safely dispatchable: {spec.dispatch_gate_reason()}"
                ],
                "next_actions": [
                    "Add file-scope hints, constraints, acceptance criteria, or explicit work orders before dispatch.",
                ],
            }

        if not self.config.dispatch_enabled:
            return {
                "status": "needs_human",
                "outcome": "preview_only",
                "reasons": [
                    f"No-dispatch preview only for issue #{issue.number}; supervised execution was intentionally skipped."
                ],
                "next_actions": [
                    "Review the selected issue and derived validation contract.",
                    "Rerun without --no-dispatch to execute the bounded Boss loop lane.",
                ],
            }

        requested_target_agent = (
            str(pending_handoff[1]).strip().lower()
            if pending_handoff is not None and str(pending_handoff[1]).strip()
            else self._requested_target_agent_for_issue(issue.number)
        )

        # --- Backbone ledger: register dispatch intent ---
        backbone_run_id = None
        runtime = None
        try:
            from aragora.pipeline.backbone_runtime import BackboneRuntime
            from aragora.pipeline.backbone_contracts import RunLedger

            runtime = BackboneRuntime()
            ledger = RunLedger(
                run_id=f"boss-{self.run_id}-issue{issue.number}",
                entrypoint="boss_loop",
                status="dispatching",
                metadata={"issue_number": issue.number, "issue_title": issue.title},
            )
            runtime.create_run(ledger)
            backbone_run_id = ledger.run_id
        except Exception:
            pass  # Never block autonomous dispatch

        claimed_runner_id: str | None = None
        selected_runner, claimed_runner_id = self._claim_runner_for_dispatch(
            freshness,
            requested_target_agent=requested_target_agent,
        )
        if selected_runner is None:
            selected_runner = self._selected_runner_for_dispatch(
                freshness,
                requested_target_agent=requested_target_agent,
            )
        try:
            result = await dispatch_bounded_spec(
                spec,
                target_branch=self.config.target_branch,
                budget_limit_usd=self.config.budget_limit_usd,
                max_ticks=self.config.dispatch_max_ticks,
                wait_for_completion=True,
                default_target_agent=requested_target_agent,
                default_reviewer_agent=self.config.default_reviewer_agent,
                # The supervisor already provisions the worktree, so the session
                # script wrapper is redundant and crashes on bash 3.2 (macOS
                # default) due to ${VAR,,} syntax.  Matches tranche_queue.py.
                use_managed_session_script=False,
                selected_runner=selected_runner,
                worker_env=refinement_worker_env or None,
                allow_claude_dangerously_skip_permissions=self.config.allow_claude_dangerously_skip_permissions,
                allow_codex_full_auto=self.config.allow_codex_full_auto,
                execution_mode=self.config.execution_mode,
            )
        finally:
            if claimed_runner_id:
                self._release_runner_claim(claimed_runner_id)

        # --- Backbone ledger: record dispatch outcome ---
        if backbone_run_id and runtime is not None:
            try:
                runtime.update_run(
                    backbone_run_id,
                    status=_backbone_dispatch_status(result),
                    execution_id=result.get("run_id"),
                    receipt_id=result.get("receipt_id"),
                )
            except Exception:
                pass  # Never block autonomous dispatch
        if pending_handoff is not None:
            dispatch_started = _dispatch_result_started(result)
            if result.get("status") != "failed" or dispatch_started:
                self._pending_handoff_prompts.pop(issue.number, None)
        result["receipt_metadata"] = self._receipt_metadata_for_result(
            result,
            issue=issue,
            freshness=freshness,
            selected_runner=selected_runner,
            requested_target_agent=requested_target_agent,
        )
        dispatch_status = _backbone_dispatch_status(result)
        result = self._postprocess_issue_result(issue, result)
        postprocess_metadata = self._apply_postprocess_metadata(result)
        if (
            backbone_run_id
            and runtime is not None
            and (_backbone_dispatch_status(result) != dispatch_status or bool(postprocess_metadata))
        ):
            try:
                runtime.update_run(
                    backbone_run_id,
                    status=_backbone_dispatch_status(result),
                    execution_id=result.get("run_id"),
                    receipt_id=result.get("receipt_id"),
                    metadata={"boss_postprocess": postprocess_metadata}
                    if postprocess_metadata
                    else {},
                )
            except Exception:
                pass  # Never block autonomous dispatch
        if result.get("status") == "failed":
            error = str(result.get("error", "")).strip()
            if error:
                logger.warning("Boss dispatch failed for issue #%d: %s", issue.number, error)
        return result

    def _attach_issue_handoff_metadata(self, spec: Any, issue: GitHubIssue) -> None:
        repo_slug = self._repo_slug_for_issue(issue) or ""
        work_orders = getattr(spec, "work_orders", None)
        if not isinstance(work_orders, list):
            return
        for index, work_order in enumerate(work_orders, start=1):
            if not isinstance(work_order, dict):
                continue
            work_order_id = str(work_order.get("work_order_id", "")).strip() or f"work-{index}"
            metadata = dict(work_order.get("metadata") or {})
            metadata.setdefault("issue_number", issue.number)
            metadata.setdefault("issue_title", issue.title)
            if repo_slug:
                metadata.setdefault("boss_repo", repo_slug)
            repo_part = repo_slug or "unknown-repo"
            metadata.setdefault(
                "handoff_key",
                f"github-issue:{repo_part}:{issue.number}:{work_order_id}",
            )
            work_order["metadata"] = metadata

    def _receipt_metadata_for_result(
        self,
        result: dict[str, Any],
        *,
        issue: GitHubIssue,
        freshness: RunnerFreshnessResult,
        selected_runner: dict[str, Any] | None = None,
        requested_target_agent: str | None = None,
    ) -> dict[str, Any]:
        details = freshness.details if isinstance(freshness.details, dict) else {}
        routing = details.get("routing") if isinstance(details, dict) else {}
        selected_runner_payload: dict[str, Any] = dict(selected_runner or {})
        if not selected_runner_payload and isinstance(routing, dict):
            selected_runners = routing.get("selected_runners")
            if isinstance(selected_runners, list) and selected_runners:
                first = selected_runners[0]
                if isinstance(first, dict):
                    selected_runner_payload = dict(first)

        deliverable = (
            result.get("deliverable") if isinstance(result.get("deliverable"), dict) else {}
        )
        actual_target_agent = None
        actual_reviewer_agent = None
        run = result.get("run")
        if isinstance(run, dict):
            work_orders = run.get("work_orders", [])
            if isinstance(work_orders, list):
                for work_order in work_orders:
                    if not isinstance(work_order, dict):
                        continue
                    if (
                        deliverable
                        and deliverable.get("work_order_id")
                        and work_order.get("work_order_id") != deliverable.get("work_order_id")
                    ):
                        continue
                    actual_target_agent = str(work_order.get("target_agent", "")).strip() or None
                    actual_reviewer_agent = (
                        str(work_order.get("reviewer_agent", "")).strip() or None
                    )
                    break

        return {
            "issue_number": issue.number,
            "requested_target_agent": requested_target_agent,
            "requested_reviewer_agent": self.config.default_reviewer_agent,
            "actual_target_agent": actual_target_agent,
            "actual_reviewer_agent": actual_reviewer_agent,
            "runner_id": selected_runner_payload.get("runner_id"),
            "runner_type": selected_runner_payload.get("runner_type"),
            "runner_profile": selected_runner_payload.get("profile"),
            "cost_class": selected_runner_payload.get("cost_class"),
            "fallback_reason": routing.get("fallback_reason")
            if isinstance(routing, dict)
            else None,
        }

    def _runner_candidates_for_dispatch(
        self,
        freshness: RunnerFreshnessResult,
        *,
        requested_target_agent: str | None = None,
    ) -> list[dict[str, Any]]:
        details = freshness.details if isinstance(freshness.details, dict) else {}
        routing = details.get("routing") if isinstance(details, dict) else {}
        if not isinstance(routing, dict):
            return []
        selected_runners = routing.get("selected_runners")
        if not isinstance(selected_runners, list):
            return []
        requested = (
            str(requested_target_agent or self.config.default_target_agent or "").strip().lower()
        )
        candidates: list[dict[str, Any]] = []
        for item in selected_runners:
            if not isinstance(item, dict):
                continue
            runner_type = str(item.get("runner_type", "")).strip().lower()
            if requested and runner_type == requested:
                candidates.append(dict(item))
        for item in selected_runners:
            if isinstance(item, dict):
                runner_id = str(item.get("runner_id", "")).strip()
                if runner_id and all(
                    str(candidate.get("runner_id", "")).strip() != runner_id
                    for candidate in candidates
                ):
                    candidates.append(dict(item))
        return candidates

    def _selected_runner_for_dispatch(
        self,
        freshness: RunnerFreshnessResult,
        *,
        requested_target_agent: str | None = None,
    ) -> dict[str, Any] | None:
        candidates = self._runner_candidates_for_dispatch(
            freshness,
            requested_target_agent=requested_target_agent,
        )
        return dict(candidates[0]) if candidates else None

    def _claim_runner_for_dispatch(
        self,
        freshness: RunnerFreshnessResult,
        *,
        requested_target_agent: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        from aragora.swarm.runner_registry import (
            LocalRunnerRegistry,
            authorization_context_with_defaults,
        )

        owner_context = authorization_context_with_defaults(repo_root=Path.cwd(), env=self._env)
        registry = (
            LocalRunnerRegistry(path=self.config.registry_path)
            if self.config.registry_path
            else LocalRunnerRegistry()
        )
        for selected_runner in self._runner_candidates_for_dispatch(
            freshness,
            requested_target_agent=requested_target_agent,
        ):
            runner_id = str(selected_runner.get("runner_id", "")).strip()
            if not runner_id:
                continue
            claimed = registry.claim_runner(runner_id, owner_context=owner_context)
            if claimed is not None:
                return claimed, runner_id
        return None, None

    def _release_runner_claim(self, runner_id: str) -> None:
        from aragora.swarm.runner_registry import (
            LocalRunnerRegistry,
            authorization_context_with_defaults,
        )

        normalized_runner_id = str(runner_id).strip()
        if not normalized_runner_id:
            return
        owner_context = authorization_context_with_defaults(repo_root=Path.cwd(), env=self._env)
        registry = (
            LocalRunnerRegistry(path=self.config.registry_path)
            if self.config.registry_path
            else LocalRunnerRegistry()
        )
        registry.release_runner_claim(normalized_runner_id, owner_context=owner_context)

    def _collect_needs_human_reasons(self) -> list[str]:
        """Collect all needs-human reasons across iterations."""
        reasons: list[str] = []
        for status in self._iteration_statuses:
            reasons.extend(status.needs_human_reasons)
        return list(dict.fromkeys(reasons))

    def _derive_next_actions(self) -> list[str]:
        """Derive final next actions based on stop reason."""
        if self._stop_reason in {
            BossStopReason.NO_FRESH_RUNNER.value,
            BossStopReason.NO_SUITABLE_ISSUE.value,
            BossStopReason.ISSUE_FEED_ERROR.value,
        }:
            for status in reversed(self._iteration_statuses):
                if status.stop_reason == self._stop_reason and status.next_actions:
                    return list(status.next_actions)
        if self._stop_reason == BossStopReason.MAX_ITERATIONS.value:
            for status in reversed(self._iteration_statuses):
                if status.worker_status == "running" and status.next_actions:
                    return list(status.next_actions)
            return [
                f"Boss loop completed {len(self._iteration_statuses)} iterations.",
                "Review completed and failed issues, then restart if needed.",
            ]
        if self._stop_reason == BossStopReason.NO_FRESH_RUNNER.value:
            return [
                "Re-register or refresh an eligible runner.",
                "Run `aragora swarm runner register` to update registration.",
            ]
        if self._stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value:
            return [
                "No actionable issues found.",
                "Create issues with concrete scope, or adjust --label-filter.",
            ]
        if self._stop_reason == BossStopReason.CONSECUTIVE_FAILURES.value:
            return [
                f"{self._consecutive_failures} consecutive failures.",
                "Investigate the last failures before resuming.",
            ]
        if self._stop_reason == BossStopReason.NEEDS_HUMAN.value:
            for status in reversed(self._iteration_statuses):
                if status.stop_reason == BossStopReason.NEEDS_HUMAN.value and status.next_actions:
                    return list(status.next_actions)
            return [
                "Worker reached a decision boundary requiring human input.",
                "Review the worker output and decide next steps.",
            ]
        if self._stop_reason == BossStopReason.AUTO_UPDATE.value:
            return [
                "Boss loop stopped to apply a code update.",
                "Restart the boss loop to pick up the latest changes.",
            ]
        return ["Boss loop stopped. Check iteration statuses for details."]
