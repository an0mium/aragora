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
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# GitHub Issue Feed
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GitHubIssue:
    """Minimal representation of a GitHub issue suitable for task selection."""

    number: int
    title: str
    body: str
    labels: list[str]
    url: str
    state: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "labels": list(self.labels),
            "url": self.url,
            "state": self.state,
            "created_at": self.created_at,
        }


class GitHubIssueFeed:
    """Pull open issues from a GitHub repo via the ``gh`` CLI.

    Only issues in the ``open`` state are considered.  The feed is intentionally
    simple: it does not cache, does not paginate beyond the configured limit,
    and does not filter on anything except state and optional label match.
    """

    def __init__(
        self,
        *,
        repo: str | None = None,
        label_filter: str | None = None,
        limit: int = 25,
    ) -> None:
        self.repo = repo  # "owner/repo" or None for current repo
        self.label_filter = label_filter
        self.limit = max(1, min(limit, 100))

    def fetch(self) -> list[GitHubIssue]:
        """Fetch open issues from GitHub. Returns empty list on failure."""
        cmd = [
            "gh",
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            str(self.limit),
            "--json",
            "number,title,body,labels,url,state,createdAt",
        ]
        if self.repo:
            cmd.extend(["--repo", self.repo])
        if self.label_filter:
            cmd.extend(["--label", self.label_filter])

        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("gh issue list failed: %s", exc)
            return []

        if proc.returncode != 0:
            logger.warning("gh issue list returned %d: %s", proc.returncode, proc.stderr.strip())
            return []

        try:
            raw_issues = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            logger.warning("gh issue list produced invalid JSON")
            return []

        if not isinstance(raw_issues, list):
            return []

        issues: list[GitHubIssue] = []
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            labels_raw = item.get("labels") or []
            labels = [
                str(lbl.get("name", "") if isinstance(lbl, dict) else lbl).strip()
                for lbl in labels_raw
                if str(lbl.get("name", "") if isinstance(lbl, dict) else lbl).strip()
            ]
            issues.append(
                GitHubIssue(
                    number=int(item.get("number", 0)),
                    title=str(item.get("title", "")).strip(),
                    body=str(item.get("body", "")).strip(),
                    labels=labels,
                    url=str(item.get("url", "")).strip(),
                    state=str(item.get("state", "OPEN")).strip(),
                    created_at=str(item.get("createdAt", "")).strip(),
                )
            )
        return issues


def select_eligible_issue(
    issues: list[GitHubIssue],
    *,
    skip_labels: set[str] | None = None,
    require_labels: set[str] | None = None,
) -> GitHubIssue | None:
    """Select the first open issue that passes eligibility filters.

    Selection is intentionally simple and truthful:
    - Must be in ``open`` state
    - Must have a non-empty title
    - Must not carry any label in ``skip_labels``
    - If ``require_labels`` is set, must carry at least one

    Returns ``None`` with no improvisation if nothing qualifies.
    """
    _skip = skip_labels or set()
    for issue in issues:
        if issue.state.upper() != "OPEN":
            continue
        if not issue.title:
            continue
        if _skip & set(issue.labels):
            continue
        if require_labels and not (require_labels & set(issue.labels)):
            continue
        return issue
    return None


# ---------------------------------------------------------------------------
# Runner Freshness
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunnerFreshnessResult:
    """Result of a runner freshness check."""

    fresh: bool
    runner_ids: list[str]
    checked_at: str
    blocked_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fresh": self.fresh,
            "runner_ids": list(self.runner_ids),
            "checked_at": self.checked_at,
            "blocked_reason": self.blocked_reason,
            "details": dict(self.details),
        }


def check_runner_freshness(
    *,
    freshness_ttl_seconds: float = 300.0,
    registry_path: str | None = None,
    env: dict[str, str] | None = None,
) -> RunnerFreshnessResult:
    """Verify that at least one registered runner is fresh and eligible.

    Freshness means:
    1. The runner registry resolves to at least one eligible runner
    2. A live re-inspection of the Codex CLI confirms it is still available
    3. The runner's registration is not older than ``freshness_ttl_seconds``

    This is a synchronous check suitable for calling at each Boss loop iteration.
    """
    from aragora.swarm.runner_registry import (
        CodexRunnerInspector,
        LocalRunnerRegistry,
        authorization_context_from_env,
    )

    now = datetime.now(UTC)
    checked_at = now.isoformat()
    owner_context = authorization_context_from_env(env)

    if owner_context is None:
        return RunnerFreshnessResult(
            fresh=False,
            runner_ids=[],
            checked_at=checked_at,
            blocked_reason="missing_owner_context",
        )

    registry = LocalRunnerRegistry(path=registry_path) if registry_path else LocalRunnerRegistry()
    routing = registry.resolve_boss_routing(owner_context=owner_context)

    if routing.is_blocked:
        return RunnerFreshnessResult(
            fresh=False,
            runner_ids=[],
            checked_at=checked_at,
            blocked_reason=routing.blocked_reason,
            details={"routing": routing.to_dict()},
        )

    # Live re-inspection: is the Codex CLI still responding?
    inspector = CodexRunnerInspector(env=env)
    live = inspector.inspect()

    if not live.available:
        return RunnerFreshnessResult(
            fresh=False,
            runner_ids=routing.selected_runner_ids,
            checked_at=checked_at,
            blocked_reason="runner_not_responding",
            details={"live_inspection": live.to_dict()},
        )

    # Check registration age against TTL
    registrations = registry.list_registrations()
    stale_ids: list[str] = []
    for reg in registrations:
        runner_id = str(reg.get("runner_id", "")).strip()
        if runner_id not in routing.selected_runner_ids:
            continue
        updated_at = str(reg.get("updated_at") or reg.get("registered_at") or "").strip()
        if not updated_at:
            stale_ids.append(runner_id)
            continue
        try:
            reg_time = datetime.fromisoformat(updated_at)
            if reg_time.tzinfo is None:
                reg_time = reg_time.replace(tzinfo=UTC)
            age = (now - reg_time).total_seconds()
            if age > freshness_ttl_seconds:
                stale_ids.append(runner_id)
        except ValueError:
            stale_ids.append(runner_id)

    fresh_ids = [rid for rid in routing.selected_runner_ids if rid not in stale_ids]

    if not fresh_ids:
        return RunnerFreshnessResult(
            fresh=False,
            runner_ids=routing.selected_runner_ids,
            checked_at=checked_at,
            blocked_reason="all_runners_stale",
            details={
                "stale_ids": stale_ids,
                "freshness_ttl_seconds": freshness_ttl_seconds,
            },
        )

    return RunnerFreshnessResult(
        fresh=True,
        runner_ids=fresh_ids,
        checked_at=checked_at,
        details={
            "live_available": live.available,
            "live_auth_mode": live.auth_mode,
        },
    )


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

    def to_dict(self) -> dict[str, Any]:
        return {
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
    issue_limit: int = 25
    skip_labels: set[str] = field(default_factory=lambda: {"wontfix", "duplicate", "invalid"})
    require_labels: set[str] | None = None

    # Retry / self-correction
    max_consecutive_failures: int = 3
    max_retries_per_issue: int = 2

    # Dispatch
    target_branch: str = "main"
    budget_limit_usd: float = 5.0

    # Reporting
    status_report_interval: int = 5  # every N iterations


# ---------------------------------------------------------------------------
# Boss Loop
# ---------------------------------------------------------------------------


def _classify_terminal_run_outcome(run_dict: dict[str, Any]) -> str:
    """Map a supervisor run dict to a stable outcome classification."""
    status = str(run_dict.get("status", "")).strip().lower()
    if status == "completed":
        deliverable = _extract_deliverable(run_dict)
        if deliverable is None:
            return "clean_exit_no_deliverable"
        if deliverable.get("type") == "adopted_pr":
            return "pr_adopted"
        return "deliverable_created"
    if status == "needs_human":
        return "needs_human"

    details = json.dumps(run_dict, sort_keys=True).lower()
    if "timeout" in details:
        return "timeout"
    if "exit_code" in details or "traceback" in details or "crash" in details:
        return "crash"
    return "blocked"


async def dispatch_bounded_spec(
    spec: Any,
    *,
    target_branch: str = "main",
    budget_limit_usd: float = 5.0,
    max_ticks: int = 360,
    repo_path: Any | None = None,
    default_target_agent: str | None = None,
    default_reviewer_agent: str | None = None,
) -> dict[str, Any]:
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
            wait=True,
            interval_seconds=5.0,
            max_ticks=max_ticks,
            default_target_agent=default_target_agent,
            default_reviewer_agent=default_reviewer_agent,
        )
        run_dict = run.to_dict()
        outcome = _classify_terminal_run_outcome(run_dict)
        deliverable = _extract_deliverable(run_dict)
        if outcome in {"deliverable_created", "pr_adopted"}:
            return {
                "status": "completed",
                "outcome": outcome,
                "run": run_dict,
                "run_id": run_dict.get("run_id"),
                "deliverable": deliverable,
            }
        if outcome == "clean_exit_no_deliverable":
            return {
                "status": "needs_human",
                "outcome": outcome,
                "run": run_dict,
                "run_id": run_dict.get("run_id"),
                "reasons": [
                    "Run reported completed but produced no concrete deliverable "
                    "(no pushed branch, no PR, no committed artifact)."
                ],
            }
        if outcome == "needs_human":
            reasons: list[str] = []
            for wo in run_dict.get("work_orders", []):
                if isinstance(wo, dict):
                    for blocker in wo.get("blockers", []):
                        reasons.append(str(blocker))
                    err = wo.get("dispatch_error")
                    if err:
                        reasons.append(str(err))
            return {
                "status": "needs_human",
                "outcome": outcome,
                "run": run_dict,
                "run_id": run_dict.get("run_id"),
                "reasons": reasons or ["Worker reached needs_human state."],
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
    """Check a completed run for a concrete deliverable.

    A run is only considered to have produced a real deliverable if at least
    one work order has:
    - A non-empty ``pr_url``, OR
    - A non-empty ``branch`` with at least one ``commit_sha``, OR
    - An explicit ``adopted_pr`` reference

    Returns a summary dict describing the deliverable, or ``None`` if the run
    produced only a dirty local worktree with no pushed/committed artifact.
    """
    work_orders = run_dict.get("work_orders", [])
    for wo in work_orders:
        if not isinstance(wo, dict):
            continue
        wo_status = str(wo.get("status", "")).strip()
        if wo_status not in {"completed", "merged"}:
            continue

        pr_url = str(wo.get("pr_url", "")).strip()
        if pr_url:
            return {"type": "pr", "pr_url": pr_url, "work_order_id": wo.get("work_order_id")}

        adopted_pr = str(wo.get("adopted_pr", "")).strip()
        if adopted_pr:
            return {
                "type": "adopted_pr",
                "adopted_pr": adopted_pr,
                "work_order_id": wo.get("work_order_id"),
            }

        branch = str(wo.get("branch", "")).strip()
        commit_shas = [s for s in wo.get("commit_shas", []) if str(s).strip()]
        if branch and commit_shas:
            return {
                "type": "branch",
                "branch": branch,
                "commit_shas": commit_shas,
                "work_order_id": wo.get("work_order_id"),
            }

    return None


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
            limit=self.config.issue_limit,
        )
        self._freshness_checker = freshness_checker or check_runner_freshness
        self._env = env
        self._attempted_issues: list[dict[str, Any]] = []
        self._completed_issues: list[dict[str, Any]] = []
        self._failed_issues: list[dict[str, Any]] = []
        self._iteration_statuses: list[BossIterationStatus] = []
        self._consecutive_failures = 0
        self._issue_attempt_counts: dict[int, int] = {}
        self._stop_reason: str | None = None

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

        while iteration < self.config.max_iterations:
            iteration += 1

            status = await self._run_iteration(iteration)
            self._iteration_statuses.append(status)

            if on_status is not None:
                try:
                    on_status(status)
                except Exception:
                    pass

            if status.stop_reason and status.stop_reason != BossStopReason.STILL_RUNNING.value:
                self._stop_reason = status.stop_reason
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
        return BossLoopResult(
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

    async def _run_iteration(self, iteration: int) -> BossIterationStatus:
        """Execute a single Boss loop iteration."""
        now = datetime.now(UTC).isoformat()
        iter_start = time.monotonic()

        # Step 1: Check runner freshness
        freshness = self._freshness_checker(
            freshness_ttl_seconds=self.config.freshness_ttl_seconds,
            registry_path=self.config.registry_path,
            env=self._env,
        )
        freshness_dict = freshness.to_dict() if hasattr(freshness, "to_dict") else dict(freshness)

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

        # Step 2: Fetch issues from GitHub
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

        # Step 3: Select eligible issue
        # Skip issues that have exceeded retry limits
        already_maxed = {
            num
            for num, count in self._issue_attempt_counts.items()
            if count >= self.config.max_retries_per_issue
        }
        candidate_issues = [i for i in issues if i.number not in already_maxed]

        selected = select_eligible_issue(
            candidate_issues,
            skip_labels=self.config.skip_labels,
            require_labels=self.config.require_labels,
        )

        if selected is None:
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=now,
                runner_freshness=freshness_dict,
                selected_issue=None,
                worker_status="idle",
                stop_reason=BossStopReason.NO_SUITABLE_ISSUE.value,
                needs_human_reasons=["No suitable open issue found in the GitHub feed."],
                next_actions=[
                    "Create a new issue with actionable scope, or adjust label filters.",
                    f"Issues checked: {len(issues)}, already maxed retries: {len(already_maxed)}",
                ],
                elapsed_seconds=time.monotonic() - iter_start,
            )

        # Step 4: Dispatch supervised work for this issue
        issue_dict = selected.to_dict()
        self._attempted_issues.append(issue_dict)
        self._issue_attempt_counts[selected.number] = (
            self._issue_attempt_counts.get(selected.number, 0) + 1
        )

        worker_result = await self._dispatch_issue(selected, freshness)

        elapsed = time.monotonic() - iter_start

        if worker_result.get("status") == "completed":
            self._completed_issues.append(issue_dict)
            self._consecutive_failures = 0
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=now,
                runner_freshness=freshness_dict,
                selected_issue=issue_dict,
                worker_status="completed",
                stop_reason=None,
                needs_human_reasons=[],
                next_actions=["Proceeding to next issue."],
                elapsed_seconds=elapsed,
            )

        if worker_result.get("status") == "needs_human":
            self._failed_issues.append(issue_dict)
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=now,
                runner_freshness=freshness_dict,
                selected_issue=issue_dict,
                worker_status="needs_human",
                stop_reason=BossStopReason.NEEDS_HUMAN.value,
                needs_human_reasons=worker_result.get("reasons", ["Worker requires human input."]),
                next_actions=["Review the worker output and decide next steps."],
                elapsed_seconds=elapsed,
            )

        # Worker failed
        self._failed_issues.append(issue_dict)
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.config.max_consecutive_failures:
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=now,
                runner_freshness=freshness_dict,
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
                elapsed_seconds=elapsed,
                error=worker_result.get("error"),
            )

        return BossIterationStatus(
            iteration=iteration,
            run_id=self.run_id,
            timestamp=now,
            runner_freshness=freshness_dict,
            selected_issue=issue_dict,
            worker_status="failed",
            stop_reason=None,
            needs_human_reasons=[],
            next_actions=[
                f"Issue #{selected.number} failed (attempt "
                f"{self._issue_attempt_counts[selected.number]}/{self.config.max_retries_per_issue}). "
                "Will retry with next iteration.",
            ],
            elapsed_seconds=elapsed,
            error=worker_result.get("error"),
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

        goal = f"[Issue #{issue.number}] {issue.title}"
        body_context = issue.body[:500] if issue.body else ""
        if body_context:
            goal = f"{goal}\n\n{body_context}"

        spec = SwarmSpec.from_direct_goal(
            goal,
            budget_limit_usd=self.config.budget_limit_usd,
            requires_approval=True,
            user_expertise="developer",
        )

        if not spec.is_dispatch_bounded():
            return {
                "status": "failed",
                "error": f"Issue #{issue.number} produced an under-specified spec: "
                f"{spec.dispatch_gate_reason()}",
            }

        result = await dispatch_bounded_spec(
            spec,
            target_branch=self.config.target_branch,
            budget_limit_usd=self.config.budget_limit_usd,
            max_ticks=360,
        )
        if result.get("status") == "failed":
            error = str(result.get("error", "")).strip()
            if error:
                logger.warning("Boss dispatch failed for issue #%d: %s", issue.number, error)
        return result

    def _collect_needs_human_reasons(self) -> list[str]:
        """Collect all needs-human reasons across iterations."""
        reasons: list[str] = []
        for status in self._iteration_statuses:
            reasons.extend(status.needs_human_reasons)
        return list(dict.fromkeys(reasons))

    def _derive_next_actions(self) -> list[str]:
        """Derive final next actions based on stop reason."""
        if self._stop_reason == BossStopReason.MAX_ITERATIONS.value:
            return [
                f"Boss loop completed {len(self._iteration_statuses)} iterations.",
                "Review completed and failed issues, then restart if needed.",
            ]
        if self._stop_reason == BossStopReason.NO_FRESH_RUNNER.value:
            return [
                "Re-register or refresh the Codex runner.",
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
            return [
                "Worker reached a decision boundary requiring human input.",
                "Review the worker output and decide next steps.",
            ]
        return ["Boss loop stopped. Check iteration statuses for details."]
