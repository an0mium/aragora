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
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aragora.pipeline.execution_mode import ExecutionMode
from aragora.swarm.terminal_truth import (
    extract_run_deliverable,
    extract_run_worker_outcome,
    qualify_work_order_terminal_state,
    qualify_run_terminal_state,
)
from aragora.swarm.lane_telemetry import LaneTelemetryCollector, LaneTelemetryRecord

# Backwards-compatible re-exports from extracted modules
from aragora.swarm.boss_feed import GitHubIssue, GitHubIssueFeed, select_eligible_issue  # noqa: F401
from aragora.swarm.boss_freshness import RunnerFreshnessResult, check_runner_freshness  # noqa: F401
from aragora.swarm.boss_validation import (  # noqa: F401
    _compose_issue_dispatch_goal,
    _should_replace_with_focused_tests,
    sanitize_issue_body_for_dispatch,
    extract_issue_validation_contract,
    extract_pre_dispatch_validation_commands,
    run_pre_dispatch_validation_commands,
    discover_focused_tests,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc
_LANE_TELEMETRY = LaneTelemetryCollector()
_GITHUB_ISSUE_URL_RE = re.compile(r"github\.com/(?P<repo>[^/]+/[^/]+)/issues/(?P<number>\d+)")
_GITHUB_PR_URL_RE = re.compile(r"github\.com/[^/]+/[^/]+/pull/(?P<number>\d+)")
_ALREADY_DONE_MARKERS = (
    "already implemented",
    "already exists",
    "no changes needed",
    "no code changes needed",
    "nothing to commit",
    "there's nothing to commit",
)
_BOSS_PUBLISH_COMMENT_MARKER = "<!-- aragora-boss-loop-publish -->"


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

    # Runner freshness
    freshness_ttl_seconds: float = 3600.0  # 1 hour
    registry_path: str | None = None

    # Issue feed
    repo: str | None = None
    label_filter: str | None = None
    issue_number: int | None = None
    issue_numbers: list[int] | None = None
    issue_limit: int = 25
    skip_labels: set[str] = field(default_factory=lambda: {"wontfix", "duplicate", "invalid"})
    require_labels: set[str] | None = None
    require_validation_contract: bool = True

    # Retry / self-correction
    max_consecutive_failures: int = 3
    max_retries_per_issue: int = 5  # Generous: allows initial attempt + 2 repairs + 2 retries

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

    # Micro-decomposition: break broad issues into single-file work orders.
    # Workers succeed on focused tasks but timeout on broad ones in large repos.
    use_micro_decomposition: bool = True

    # Security: opt-in flags for dangerous worker CLI behavior (Crux 1).
    allow_claude_dangerously_skip_permissions: bool = False
    allow_codex_full_auto: bool = False
    execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS

    # Autonomous post-processing: publish verified branch deliverables and
    # optionally close already-resolved no-op issues.
    auto_publish_deliverables: bool = False
    auto_close_already_done_issues: bool = False

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
    ) -> None:
        self.config = config or BossLoopConfig()
        self.run_id = f"boss-{uuid.uuid4().hex[:12]}"
        self._feed = issue_feed or GitHubIssueFeed(
            repo=self.config.repo,
            label_filter=self.config.label_filter,
            issue_numbers=self.config.issue_numbers,
            limit=self.config.issue_limit,
        )
        self._freshness_checker = freshness_checker or check_runner_freshness
        self._env = env
        self._attempted_issues: list[dict[str, Any]] = []
        self._completed_issues: list[dict[str, Any]] = []
        self._failed_issues: list[dict[str, Any]] = []
        self._iteration_statuses: list[BossIterationStatus] = []
        self._consecutive_failures = 0
        self._issue_attempt_counts: dict[int | str, int] = {}
        self._pending_handoff_prompts: dict[int, tuple[str, str | None]] = {}
        self._stop_reason: str | None = None

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
            payload = {
                "iteration": int(iteration),
                "issue_number": issue_number,
                "worker_status": str(worker_result.get("status", "")).strip() or "unknown",
                "elapsed_seconds": float(elapsed_seconds or 0.0),
                "files_changed": files_changed,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
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

    def _has_retryable_attempts(self) -> bool:
        return any(
            isinstance(issue_number, int) and attempt_count > 0
            for issue_number, attempt_count in self._issue_attempt_counts.items()
        )

    def _requested_runner_type_for_freshness(self) -> str | None:
        # Once retries are in play, keep the freshness pool broad enough that
        # dispatch can rotate to the next runner type instead of reusing the
        # original default forever.
        if self._has_retryable_attempts() and len(self._normalized_model_rotation()) > 1:
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
                available=bool(reg.get("available", False)),
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

    def _pending_handoff_candidates(self, issues: list[GitHubIssue]) -> list[GitHubIssue]:
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
                },
                outputs={
                    "iterations_completed": result.iterations_completed,
                    "stop_reason": result.stop_reason,
                    "issues_attempted": attempted,
                    "issues_completed": completed,
                    "issues_failed": failed,
                    "needs_human_reasons": list(result.needs_human_reasons),
                    "next_actions": list(result.next_actions),
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
        deliverable_type = str(deliverable.get("type", "")).strip().lower()
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
        try:
            from aragora.ralph.github_control import GitHubControl
            from aragora.swarm.pr_registry import PullRequestRegistry
            from aragora.swarm.tranche_integrate import publish_lane_deliverable

            repo_root = Path.cwd().resolve()
            artifact = _BossDeliverableArtifact(
                branch=branch,
                metadata={
                    "branch": branch,
                    "deliverable": {
                        **dict(deliverable),
                        "branch": branch,
                        "commit_shas": commit_shas,
                    },
                    "receipt_id": worker_result.get("receipt_id"),
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
    def _published_deliverable_comment(worker_result: dict[str, Any]) -> str | None:
        publish_result = worker_result.get("publish_result")
        if not isinstance(publish_result, dict):
            return None
        if not bool(publish_result.get("published")):
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
        if not isinstance(publish_result, dict):
            return False
        if not bool(publish_result.get("published")):
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

    def _postprocess_issue_result(
        self,
        issue: GitHubIssue,
        worker_result: dict[str, Any],
    ) -> dict[str, Any]:
        publish_result = self._maybe_publish_deliverable(issue, worker_result)
        if publish_result is not None:
            worker_result["publish_result"] = publish_result
        issue_comment_result = self._maybe_comment_published_deliverable(issue, worker_result)
        if issue_comment_result is not None:
            worker_result["issue_comment_result"] = issue_comment_result
        issue_resolution = self._maybe_auto_close_already_done_issue(issue, worker_result)
        if issue_resolution is not None:
            worker_result["issue_resolution"] = issue_resolution
        self._apply_postprocess_metadata(worker_result)
        self._promote_published_deliverable(worker_result)
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

    @staticmethod
    def _emit_live_status(on_status: Any | None, status: BossIterationStatus) -> None:
        if on_status is None:
            return
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
        # Skip issues that have exceeded retry limits
        already_maxed = {
            num
            for num, count in self._issue_attempt_counts.items()
            if count >= self.config.max_retries_per_issue
        }
        pending_handoffs = self._pending_handoff_candidates(issues)
        pending_issue_numbers = {issue.number for issue in pending_handoffs}
        candidate_issues = [
            i for i in issues if i.number in pending_issue_numbers or i.number not in already_maxed
        ]
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
                )
                if target_issue is not None
                else None
            )
        else:
            selected = select_eligible_issue(
                candidate_issues,
                skip_labels=self.config.skip_labels,
                require_labels=self.config.require_labels,
                use_value_ranking=self.config.use_value_ranking,
            )

        if selected is None:
            if self.config.issue_number is not None:
                needs_human_reasons, next_actions = self._target_issue_miss_guidance(
                    self.config.issue_number
                )
            else:
                needs_human_reasons = ["No suitable open issue found in the GitHub feed."]
                next_actions = [
                    "Create a new issue with actionable scope, or adjust label filters.",
                    f"Issues checked: {len(issues)}, already maxed retries: {len(already_maxed)}",
                ]
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
        freshness = self._freshness_checker(
            freshness_ttl_seconds=self.config.freshness_ttl_seconds,
            registry_path=self.config.registry_path,
            env=self._env,
            requested_runner_type=self._requested_runner_type_for_freshness(),
            allowed_profiles=self.config.allowed_runner_profiles,
            rotation_interval_seconds=self.config.runner_rotation_interval_seconds,
            verified_runner_target=self.config.verified_runner_target,
            runner_probe_limit=self.config.runner_probe_limit,
        )
        freshness_dict = _freshness_to_dict(freshness)

        if not (freshness.fresh if hasattr(freshness, "fresh") else freshness_dict.get("fresh")):
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

        # Step 4: Dispatch supervised work for this issue
        issue_dict = selected.to_dict()
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

        already_maxed = {
            num
            for num, count in self._issue_attempt_counts.items()
            if count >= self.config.max_retries_per_issue
        }
        pending_handoffs = self._pending_handoff_candidates(issues)
        pending_issue_numbers = {issue.number for issue in pending_handoffs}
        candidate_issues = [
            i for i in issues if i.number in pending_issue_numbers or i.number not in already_maxed
        ]
        ordered_candidates = pending_handoffs + [
            issue for issue in candidate_issues if issue.number not in pending_issue_numbers
        ]
        selected_issues = self._select_issues_for_iteration(
            ordered_candidates,
            limit=None,
        )

        if not selected_issues:
            if self.config.issue_number is not None:
                needs_human_reasons, next_actions = self._target_issue_miss_guidance(
                    self.config.issue_number
                )
            else:
                needs_human_reasons = ["No suitable open issue found in the GitHub feed."]
                next_actions = [
                    "Create a new issue with actionable scope, or adjust label filters.",
                    f"Issues checked: {len(issues)}, already maxed retries: {len(already_maxed)}",
                ]
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
            requested_runner_type=self._requested_runner_type_for_freshness(),
            allowed_profiles=self.config.allowed_runner_profiles,
            rotation_interval_seconds=self.config.runner_rotation_interval_seconds,
        )
        freshness_dict = _freshness_to_dict(freshness)

        if not (freshness.fresh if hasattr(freshness, "fresh") else freshness_dict.get("fresh")):
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
        pending_issues = list(selected_issues)
        active_tasks: dict[
            asyncio.Task[dict[str, Any]], tuple[GitHubIssue, dict[str, Any], float]
        ] = {}
        statuses: list[BossIterationStatus] = []
        stop_launching = False

        while pending_issues and len(active_tasks) < parallel_limit:
            issue = pending_issues.pop(0)
            issue_dict = issue.to_dict()
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
                    next_issue_dict = next_issue.to_dict()
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
        for item in selected_runners:
            if not isinstance(item, dict):
                continue
            available_capacity += max(0, int(item.get("available_capacity", 0) or 0))
        if available_capacity <= 0:
            return 1
        return max(1, min(configured_limit, available_capacity))

    def _select_issues_for_iteration(
        self,
        issues: list[GitHubIssue],
        *,
        limit: int | None,
    ) -> list[GitHubIssue]:
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
                    )
                    if target_issue is not None
                    else None
                )
                return [selected] if selected is not None else []
            selected = select_eligible_issue(
                issues,
                skip_labels=self.config.skip_labels,
                require_labels=self.config.require_labels,
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
                )
                if target_issue is not None
                else None
            )
            return [selected] if selected is not None else []

        selected_issues: list[GitHubIssue] = []
        for issue in issues:
            candidate = select_eligible_issue(
                [issue],
                skip_labels=self.config.skip_labels,
                require_labels=self.config.require_labels,
            )
            if candidate is None:
                continue
            selected_issues.append(candidate)
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
            self._consecutive_failures = 0
            self._emit_lane_receipt(worker_result, issue_dict, elapsed_seconds)
            self._log_value_outcome(issue_dict, "completed", elapsed_seconds)
            next_action = self._published_pr_followup(worker_result) or "Proceeding to next issue."
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
            has_deliverable = bool(worker_result.get("deliverable"))
            self._emit_lane_receipt(worker_result, issue_dict, elapsed_seconds)
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
                logger.warning(
                    "boss_loop_skip issue=#%s (needs_human, no deliverable, auto-continue on)",
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
                        "reasons", ["Worker requires human input."]
                    ),
                    next_actions=["Skipping to next issue (auto-continue mode)."],
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
            dispatch_started = bool(result.get("run") or result.get("run_id"))
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
        return ["Boss loop stopped. Check iteration statuses for details."]
