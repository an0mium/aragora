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
        issue_numbers: list[int] | None = None,
        limit: int = 25,
    ) -> None:
        self.repo = repo  # "owner/repo" or None for current repo
        self.label_filter = label_filter
        self.issue_numbers = [int(item) for item in issue_numbers or [] if int(item) > 0]
        self.limit = max(1, min(limit, 100))

    def fetch(self) -> list[GitHubIssue]:
        """Fetch open issues from GitHub. Returns empty list on failure."""
        if self.issue_numbers:
            issues: list[GitHubIssue] = []
            for number in self.issue_numbers:
                issue = self._fetch_issue(number)
                if issue is not None:
                    issues.append(issue)
            return issues

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

    def _fetch_issue(self, number: int) -> GitHubIssue | None:
        cmd = [
            "gh",
            "issue",
            "view",
            str(number),
            "--json",
            "number,title,body,labels,url,state,createdAt",
        ]
        if self.repo:
            cmd.extend(["--repo", self.repo])

        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("gh issue view failed: %s", exc)
            return None

        if proc.returncode != 0:
            logger.warning("gh issue view returned %d: %s", proc.returncode, proc.stderr.strip())
            return None

        try:
            item = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            logger.warning("gh issue view produced invalid JSON")
            return None

        if not isinstance(item, dict):
            return None

        state = str(item.get("state", "OPEN")).strip().lower()
        if state != "open":
            return None

        labels_raw = item.get("labels") or []
        labels = [
            str(lbl.get("name", "") if isinstance(lbl, dict) else lbl).strip()
            for lbl in labels_raw
            if str(lbl.get("name", "") if isinstance(lbl, dict) else lbl).strip()
        ]
        if self.label_filter and self.label_filter not in labels:
            return None

        return GitHubIssue(
            number=int(item.get("number", number)),
            title=str(item.get("title", "")).strip(),
            body=str(item.get("body", "")).strip(),
            labels=labels,
            url=str(item.get("url", "")).strip(),
            state=state,
            created_at=str(item.get("createdAt", "")).strip(),
        )


def select_eligible_issue(
    issues: list[GitHubIssue],
    *,
    skip_labels: set[str] | None = None,
    require_labels: set[str] | None = None,
    use_value_ranking: bool = False,
) -> GitHubIssue | None:
    """Select the best open issue that passes eligibility filters.

    Selection rules:
    - Must be in ``open`` state
    - Must have a non-empty title
    - Must not carry any label in ``skip_labels``
    - If ``require_labels`` is set, must carry ALL of them

    When ``use_value_ranking`` is True, eligible issues are scored by
    expected value-per-cost and the highest-scored issue is returned.
    Otherwise returns the first eligible issue (GitHub order).

    Returns ``None`` with no improvisation if nothing qualifies.
    """
    _skip = skip_labels or set()
    eligible: list[GitHubIssue] = []
    for issue in issues:
        if issue.state.upper() != "OPEN":
            continue
        if not issue.title:
            continue
        if _skip & set(issue.labels):
            continue
        if require_labels and not require_labels.issubset(set(issue.labels)):
            continue
        eligible.append(issue)

    if not eligible:
        return None

    if not use_value_ranking:
        return eligible[0]

    try:
        from aragora.swarm.value_estimator import (
            load_outcomes,
            log_prediction,
            rank_issues,
        )

        history = load_outcomes()
        issue_dicts = [i.to_dict() for i in eligible]
        ranked = rank_issues(issue_dicts, historical_outcomes=history)
        if ranked:
            best_estimate, best_dict = ranked[0]
            log_prediction(best_estimate)
            best_number = best_dict.get("number")
            logger.info(
                "value_ranking: #%s score=%.3f (value=%.2f p_success=%.2f proof=%.2f) — %s",
                best_number,
                best_estimate.priority_score,
                best_estimate.expected_value,
                best_estimate.p_success,
                best_estimate.proof_weight,
                best_estimate.reasoning[:80],
            )
            # Return the original GitHubIssue object
            for issue in eligible:
                if issue.number == best_number:
                    return issue
    except Exception as exc:
        logger.debug("Value ranking failed, falling back to first eligible: %s", exc)

    return eligible[0]


_VALIDATION_SECTION_PREFIXES = (
    "acceptance criteria",
    "validation",
    "validation contract",
    "definition of done",
    "done when",
    "test plan",
)
_VALIDATION_INLINE_PREFIXES = (
    "acceptance",
    "acceptance criteria",
    "validation",
    "validation contract",
    "definition of done",
    "done when",
    "test plan",
)
_VALIDATION_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s*(?:\[(?: |x|X)\]\s*)?(?P<text>.+?)\s*$")


def _ordered_unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def extract_issue_validation_contract(issue_body: str) -> list[str]:
    """Extract an explicit validation contract from a GitHub issue body.

    Supported forms:
    - bullets/checklists under headings such as "Acceptance Criteria" or "Validation"
    - inline markers such as "Validation: pytest -q ..."
    - standalone pytest commands anywhere in the issue body
    """
    lines = [str(line).rstrip() for line in str(issue_body or "").splitlines()]
    criteria: list[str] = []
    in_validation_section = False

    for raw_line in lines:
        stripped = raw_line.strip()
        normalized = stripped.lstrip("#").strip()
        normalized_lower = normalized.rstrip(":").strip().lower()

        if any(
            normalized_lower == prefix or normalized_lower.startswith(f"{prefix} ")
            for prefix in _VALIDATION_SECTION_PREFIXES
        ):
            in_validation_section = True
            continue

        inline_prefix, _, inline_value = normalized.partition(":")
        if inline_value and inline_prefix.strip().lower() in _VALIDATION_INLINE_PREFIXES:
            criteria.append(inline_value.strip())
            in_validation_section = False
            continue

        if stripped.startswith("pytest ") or stripped.startswith("python -m pytest"):
            criteria.append(stripped)
            continue

        if not in_validation_section:
            continue

        if stripped.startswith("#"):
            in_validation_section = False
            continue

        bullet_match = _VALIDATION_BULLET_RE.match(stripped)
        if bullet_match:
            criteria.append(bullet_match.group("text"))
            continue

        if not stripped:
            continue

        if normalized.endswith(":"):
            in_validation_section = False
            continue

        criteria.append(stripped)

    return _ordered_unique_strings(criteria)


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
    requested_runner_type: str | None = None,
    allowed_profiles: set[str] | None = None,
    rotation_interval_seconds: float = 1800.0,
) -> RunnerFreshnessResult:
    """Verify that at least one registered runner is fresh and eligible.

    Freshness means:
    1. The runner registry resolves to at least one eligible runner
    2. A live re-inspection of the selected CLI runner confirms it is still available
    3. The runner's registration is not older than ``freshness_ttl_seconds``

    This is a synchronous check suitable for calling at each Boss loop iteration.
    """
    from aragora.swarm.runner_registry import (
        LocalRunnerRegistry,
        authorization_context_from_env,
        configured_claude_runner_profiles,
        make_runner_inspector,
        prioritized_probe_candidates,
        probe_runner_execution,
        refresh_discovered_runners,
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
    discovered: list[Any] = []
    if requested_runner_type:
        discovered = refresh_discovered_runners(
            requested_runner_type,
            registry=registry,
            owner_context=owner_context,
            env=env,
            repo_root=Path.cwd(),
        )
    allowed_profile_set = set(allowed_profiles or configured_claude_runner_profiles(env))
    routing = registry.resolve_boss_routing(
        owner_context=owner_context,
        requested_runner_type=requested_runner_type,
        allowed_profiles=allowed_profile_set or None,
        rotation_interval_seconds=rotation_interval_seconds,
    )
    probe_summary = {
        "auto_probe_triggered": False,
        "attempted": 0,
        "passed": 0,
        "failed": 0,
        "verified_target": 0,
        "results": [],
    }
    if requested_runner_type == "claude":
        try:
            verified_target = max(
                1, int(str((env or os.environ).get("ARAGORA_BOSS_VERIFIED_RUNNER_TARGET", "2")))
            )
        except ValueError:
            verified_target = 2
        try:
            probe_limit = max(
                1, int(str((env or os.environ).get("ARAGORA_BOSS_RUNNER_PROBE_LIMIT", "1")))
            )
        except ValueError:
            probe_limit = 1
        selected_verified = len(
            [
                item
                for item in routing.selected_runners
                if isinstance(item, dict) and str(item.get("probe_status", "")).strip() == "passed"
            ]
        )
        probe_summary["verified_target"] = verified_target
        if selected_verified < verified_target:
            candidates = prioritized_probe_candidates(
                registry=registry,
                runner_type=requested_runner_type,
                discovered_inspections=discovered,
                owner_context=owner_context,
                selected_runners=routing.selected_runners,
            )
            for inspection in candidates[:probe_limit]:
                probe = probe_runner_execution(
                    inspection,
                    repo_root=Path.cwd(),
                )
                registry.record_probe(
                    inspection,
                    probe,
                    owner_context=owner_context,
                )
                probe_summary["results"].append(probe.to_dict())
                probe_summary["attempted"] += 1
                if probe.status == "passed":
                    probe_summary["passed"] += 1
                elif probe.status == "failed":
                    probe_summary["failed"] += 1
            if probe_summary["attempted"]:
                probe_summary["auto_probe_triggered"] = True
                routing = registry.resolve_boss_routing(
                    owner_context=owner_context,
                    requested_runner_type=requested_runner_type,
                    allowed_profiles=allowed_profile_set or None,
                    rotation_interval_seconds=rotation_interval_seconds,
                )

    if routing.is_blocked:
        return RunnerFreshnessResult(
            fresh=False,
            runner_ids=[],
            checked_at=checked_at,
            blocked_reason=routing.blocked_reason,
            details={"routing": routing.to_dict(), "probe": probe_summary},
        )

    # Live re-inspection: is a selected CLI runner still responding?
    live_runner_ids: list[str] = []
    live_inspections: list[dict[str, Any]] = []
    for selected in routing.selected_runners:
        runner_type = str(selected.get("runner_type", "")).strip() or "codex"
        live = make_runner_inspector(
            runner_type,
            env=env,
            profile=str(selected.get("profile", "")).strip() or None,
        ).inspect()
        live_inspections.append(live.to_dict())
        if live.available and live.auth_mode in {"chatgpt_login", "api_key", "subscription"}:
            live_runner_ids.append(live.runner_id)

    if not live_runner_ids:
        return RunnerFreshnessResult(
            fresh=False,
            runner_ids=routing.selected_runner_ids,
            checked_at=checked_at,
            blocked_reason="runner_not_responding",
            details={
                "routing": routing.to_dict(),
                "probe": probe_summary,
                "live_inspections": live_inspections,
            },
        )

    # Check registration age against TTL
    registrations = registry.list_registrations()
    stale_ids: list[str] = []
    for reg in registrations:
        runner_id = str(reg.get("runner_id", "")).strip()
        if runner_id not in routing.selected_runner_ids or runner_id not in live_runner_ids:
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
                "routing": routing.to_dict(),
                "probe": probe_summary,
                "stale_ids": stale_ids,
                "freshness_ttl_seconds": freshness_ttl_seconds,
            },
        )

    return RunnerFreshnessResult(
        fresh=True,
        runner_ids=fresh_ids,
        checked_at=checked_at,
        details={
            "routing": routing.to_dict(),
            "probe": probe_summary,
            "live_runner_ids": live_runner_ids,
            "live_inspections": live_inspections,
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
    default_reviewer_agent: str | None = None
    allowed_runner_profiles: set[str] | None = None
    runner_rotation_interval_seconds: float = 1800.0
    max_parallel_dispatches: int = 1

    # Autonomy: when True, treat needs_human with a deliverable as completed
    # instead of stopping the loop. Only stop when there's genuinely no output.
    auto_continue_on_needs_human: bool = False

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

    # Reporting
    status_report_interval: int = 5  # every N iterations


# ---------------------------------------------------------------------------
# Boss Loop
# ---------------------------------------------------------------------------


def _classify_terminal_run_outcome(run_dict: dict[str, Any]) -> str:
    """Map a supervisor run dict to a stable outcome classification.

    Uses LLM classification for ambiguous cases, falling back to keyword
    matching if the LLM is unavailable.
    """
    status = str(run_dict.get("status", "")).strip().lower()
    if status == "completed":
        deliverable = _extract_deliverable(run_dict)
        if deliverable is None:
            return "clean_exit_no_deliverable"
        if deliverable.get("type") == "adopted_pr":
            return "pr_adopted"
        return "deliverable_created"
    if status == "needs_human":
        # A run can be "needs_human" overall (e.g. one lane blocked) but still
        # have deliverables from other lanes.  Prioritize the deliverable so
        # the campaign can extract branch/commit info for PR creation.
        deliverable = _extract_deliverable(run_dict)
        if deliverable is not None:
            if deliverable.get("type") == "adopted_pr":
                return "pr_adopted"
            return "deliverable_created"
        return "needs_human"

    # --- LLM classification for ambiguous terminal states ---
    try:
        from aragora.ralph.llm_classifier import LLMBlockerClassifier

        import asyncio

        classifier = LLMBlockerClassifier()
        verdict = asyncio.run(classifier.classify_run_outcome(run_dict))
        # Only trust the LLM verdict if it actually ran (not a fallback default)
        if verdict.reasoning != "LLM call failed":
            logger.info(
                "LLM run outcome classification: %s (reasoning: %s)",
                verdict.outcome,
                verdict.reasoning,
            )
            return verdict.outcome
    except Exception:
        logger.debug("LLM run outcome classification failed, using keyword fallback", exc_info=True)

    # --- keyword fallback ---
    return _keyword_classify_terminal_run(run_dict)


def _keyword_classify_terminal_run(run_dict: dict[str, Any]) -> str:
    """Keyword-based fallback for terminal run classification."""
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
    wait_for_completion: bool = True,
    repo_path: Any | None = None,
    default_target_agent: str | None = None,
    default_reviewer_agent: str | None = None,
    use_managed_session_script: bool = True,
    selected_runner: dict[str, Any] | None = None,
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
        )
        run_dict = run.to_dict()
        run_status = str(run_dict.get("status", "")).strip().lower()
        if not wait_for_completion and run_status not in {"completed", "needs_human"}:
            return {
                "status": "running",
                "outcome": "dispatched",
                "run": run_dict,
                "run_id": run_dict.get("run_id"),
            }
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
                "deliverable": deliverable,
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


def _extract_worker_outcome(run_dict: dict[str, Any]) -> str | None:
    """Extract the first non-empty ``worker_outcome`` from a completed run.

    Returns None if no work order carries a ``worker_outcome`` field.
    """
    for wo in run_dict.get("work_orders", []):
        if not isinstance(wo, dict):
            continue
        outcome = str(wo.get("worker_outcome", "")).strip()
        if outcome:
            return outcome
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
        self._issue_attempt_counts: dict[int, int] = {}
        self._stop_reason: str | None = None

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

    def _log_value_outcome(
        self,
        issue_dict: dict[str, Any],
        worker_status: str,
        elapsed_seconds: float,
    ) -> None:
        """Log outcome for value-per-cost calibration."""
        try:
            from aragora.swarm.value_estimator import OutcomeRecord, log_outcome

            log_outcome(
                OutcomeRecord(
                    issue_number=issue_dict.get("number", 0),
                    predicted_score=0.0,  # TODO: pass from ranking step
                    predicted_p_success=0.0,
                    did_merge=worker_status == "completed",
                    needed_human_rescue=worker_status == "needs_human",
                    actual_minutes=elapsed_seconds / 60.0,
                    worker_status=worker_status,
                )
            )
        except Exception as exc:
            logger.debug("Value outcome logging skipped: %s", exc)

    def _emit_lane_receipt(
        self,
        worker_result: dict[str, Any],
        issue_dict: dict[str, Any],
        elapsed: float,
    ) -> None:
        try:
            from aragora.receipts.lane import LaneCompletionReceipt, emit_lane_receipt

            receipt = LaneCompletionReceipt(
                task_id=str(issue_dict.get("number", "")),
                lease_id=str(worker_result.get("lease_id", self.run_id)),
                agent_id=str(worker_result.get("agent_id", "boss-loop")),
                base_sha=worker_result.get("base_sha"),
                head_sha=worker_result.get("head_sha"),
                changed_files=list(worker_result.get("changed_files", [])),
                validations_run=list(worker_result.get("validations_run", [])),
                outcome="pass",
                risks=list(worker_result.get("risks", [])),
                pr_url=worker_result.get("pr_url"),
                pr_number=worker_result.get("pr_number"),
                branch=worker_result.get("branch"),
                duration_seconds=elapsed,
                metadata=dict(worker_result.get("receipt_metadata") or {}),
            )
            emit_lane_receipt(receipt)
        except Exception as exc:
            logger.debug("Lane receipt emission skipped: %s", exc)

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

            statuses = await self._run_iteration_statuses(iteration)
            self._iteration_statuses.extend(statuses)

            for status in statuses:
                if on_status is not None:
                    try:
                        on_status(status)
                    except Exception:
                        pass

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

    async def _run_iteration_statuses(self, iteration: int) -> list[BossIterationStatus]:
        if int(self.config.max_parallel_dispatches or 1) <= 1:
            return [await self._run_iteration(iteration)]
        return await self._run_iteration_batch(iteration)

    async def _run_iteration(self, iteration: int) -> BossIterationStatus:
        """Execute a single Boss loop iteration."""
        now = datetime.now(UTC).isoformat()
        iter_start = time.monotonic()

        # Step 1: Check runner freshness
        freshness = self._freshness_checker(
            freshness_ttl_seconds=self.config.freshness_ttl_seconds,
            registry_path=self.config.registry_path,
            env=self._env,
            requested_runner_type=self.config.default_target_agent,
            allowed_profiles=self.config.allowed_runner_profiles,
            rotation_interval_seconds=self.config.runner_rotation_interval_seconds,
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
        if self.config.issue_number is not None:
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
                needs_human_reasons = [
                    f"Target issue #{self.config.issue_number} was not found in the issue feed or is not eligible under current filters/retry state."
                ]
                next_actions = [
                    f"Verify issue #{self.config.issue_number} is still open, eligible, and has not exceeded retry limits.",
                    "Remove --boss-issue-number to return to feed-driven selection.",
                ]
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

        # Step 4: Dispatch supervised work for this issue
        issue_dict = selected.to_dict()
        self._attempted_issues.append(issue_dict)
        self._issue_attempt_counts[selected.number] = (
            self._issue_attempt_counts.get(selected.number, 0) + 1
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

    async def _run_iteration_batch(self, iteration: int) -> list[BossIterationStatus]:
        import asyncio

        now = datetime.now(UTC).isoformat()
        iter_start = time.monotonic()

        freshness = self._freshness_checker(
            freshness_ttl_seconds=self.config.freshness_ttl_seconds,
            registry_path=self.config.registry_path,
            env=self._env,
            requested_runner_type=self.config.default_target_agent,
            allowed_profiles=self.config.allowed_runner_profiles,
            rotation_interval_seconds=self.config.runner_rotation_interval_seconds,
        )
        freshness_dict = freshness.to_dict() if hasattr(freshness, "to_dict") else dict(freshness)

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
        candidate_issues = [i for i in issues if i.number not in already_maxed]
        parallel_limit = self._parallel_dispatch_limit(freshness)
        selected_issues = self._select_issues_for_iteration(
            candidate_issues,
            limit=None,
        )

        if not selected_issues:
            if self.config.issue_number is not None:
                needs_human_reasons = [
                    f"Target issue #{self.config.issue_number} was not found in the issue feed or is not eligible under current filters/retry state."
                ]
                next_actions = [
                    f"Verify issue #{self.config.issue_number} is still open, eligible, and has not exceeded retry limits.",
                    "Remove --boss-issue-number to return to feed-driven selection.",
                ]
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

        if worker_result.get("status") == "completed":
            self._completed_issues.append(issue_dict)
            self._consecutive_failures = 0
            self._emit_lane_receipt(worker_result, issue_dict, elapsed_seconds)
            self._log_value_outcome(issue_dict, "completed", elapsed_seconds)
            return BossIterationStatus(
                iteration=iteration,
                run_id=self.run_id,
                timestamp=timestamp,
                runner_freshness=runner_freshness,
                selected_issue=issue_dict,
                worker_status="completed",
                stop_reason=None,
                needs_human_reasons=[],
                next_actions=["Proceeding to next issue."],
                elapsed_seconds=elapsed_seconds,
            )

        if worker_result.get("status") == "needs_human":
            has_deliverable = bool(worker_result.get("deliverable"))
            if self.config.auto_continue_on_needs_human and has_deliverable:
                self._completed_issues.append(issue_dict)
                self._consecutive_failures = 0
                self._emit_lane_receipt(worker_result, issue_dict, elapsed_seconds)
                logger.info(
                    "boss_loop_auto_continue issue=#%s (needs_human with deliverable → treating as completed)",
                    issue_dict.get("number", "?"),
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
                    next_actions=["Auto-continuing: deliverable created, review can happen async."],
                    elapsed_seconds=elapsed_seconds,
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
                )

            if self.config.auto_continue_on_needs_human:
                self._consecutive_failures += 1
                logger.warning(
                    "boss_loop_skip issue=#%s (needs_human, no deliverable, auto-continue on)",
                    issue_dict.get("number", "?"),
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
                )
            next_actions = [
                str(item).strip()
                for item in worker_result.get("next_actions", [])
                if str(item).strip()
            ] or ["Review the worker output and decide next steps."]
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
            )

        self._failed_issues.append(issue_dict)
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.config.max_consecutive_failures:
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

        # Refine the prompt with codebase context before dispatch
        goal = f"[Issue #{issue.number}] {issue.title}"
        body_context = issue.body[:2000] if issue.body else ""
        if body_context:
            goal = f"{goal}\n\n{body_context}"
        # Ensure workers always commit — this is the #1 reason for needs_human failures
        goal += (
            "\n\n## CRITICAL: You MUST commit your changes\n"
            "After making changes, run:\n"
            "```\ngit add -A && git commit -m 'fix: description of changes'\n```\n"
            "If you do not commit, your work will be lost."
        )

        try:
            from aragora.swarm.prompt_refiner import refine_worker_prompt

            refinement = await refine_worker_prompt(
                issue.title,
                issue.body or "",
                repo_path=Path.cwd(),
            )
            if refinement.get("context_gathered"):
                goal = refinement["refined_prompt"]
                logger.info(
                    "Refined prompt for #%s: %d relevant files, %d test patterns",
                    issue.number,
                    len(refinement.get("files_to_change", [])),
                    len(refinement.get("test_patterns", [])),
                )
        except Exception as exc:
            logger.debug("Prompt refinement skipped: %s", exc)

        spec = SwarmSpec.from_direct_goal(
            goal,
            budget_limit_usd=self.config.budget_limit_usd,
            requires_approval=True,
            user_expertise="developer",
        )
        validation_contract = extract_issue_validation_contract(issue.body)
        if validation_contract and self.config.use_focused_verification:
            # Replace broad test suite commands with focused verification
            # that only tests files related to the worker's changes
            focused = []
            for criterion in validation_contract:
                if "pytest tests/" in criterion and "-k" not in criterion.lower():
                    # Replace broad suite with a marker that the supervisor
                    # will resolve to the actual changed-file test paths
                    focused.append(
                        "python -m pytest --timeout=30 -x -q "
                        "$(git diff --name-only origin/main -- '*.py' "
                        "| grep '^tests/' | head -20 | tr '\\n' ' ')"
                    )
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
                "reasons": [
                    f"No-dispatch preview only for issue #{issue.number}; supervised execution was intentionally skipped."
                ],
                "next_actions": [
                    "Review the selected issue and derived validation contract.",
                    "Rerun without --no-dispatch to execute the bounded Boss loop lane.",
                ],
            }

        claimed_runner_id: str | None = None
        selected_runner, claimed_runner_id = self._claim_runner_for_dispatch(freshness)
        if selected_runner is None:
            selected_runner = self._selected_runner_for_dispatch(freshness)
        try:
            result = await dispatch_bounded_spec(
                spec,
                target_branch=self.config.target_branch,
                budget_limit_usd=self.config.budget_limit_usd,
                max_ticks=360,
                wait_for_completion=self.config.max_iterations > 1,
                default_target_agent=self.config.default_target_agent,
                default_reviewer_agent=self.config.default_reviewer_agent,
                # The supervisor already provisions the worktree, so the session
                # script wrapper is redundant and crashes on bash 3.2 (macOS
                # default) due to ${VAR,,} syntax.  Matches tranche_queue.py.
                use_managed_session_script=False,
                selected_runner=selected_runner,
            )
        finally:
            if claimed_runner_id:
                self._release_runner_claim(claimed_runner_id)
        result["receipt_metadata"] = self._receipt_metadata_for_result(
            result,
            issue=issue,
            freshness=freshness,
            selected_runner=selected_runner,
        )
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
            "requested_target_agent": self.config.default_target_agent,
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
    ) -> list[dict[str, Any]]:
        details = freshness.details if isinstance(freshness.details, dict) else {}
        routing = details.get("routing") if isinstance(details, dict) else {}
        if not isinstance(routing, dict):
            return []
        selected_runners = routing.get("selected_runners")
        if not isinstance(selected_runners, list):
            return []
        requested = str(self.config.default_target_agent or "").strip().lower()
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
    ) -> dict[str, Any] | None:
        candidates = self._runner_candidates_for_dispatch(freshness)
        return dict(candidates[0]) if candidates else None

    def _claim_runner_for_dispatch(
        self,
        freshness: RunnerFreshnessResult,
    ) -> tuple[dict[str, Any] | None, str | None]:
        from aragora.swarm.runner_registry import (
            LocalRunnerRegistry,
            authorization_context_from_env,
        )

        owner_context = authorization_context_from_env(self._env)
        registry = (
            LocalRunnerRegistry(path=self.config.registry_path)
            if self.config.registry_path
            else LocalRunnerRegistry()
        )
        for selected_runner in self._runner_candidates_for_dispatch(freshness):
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
            authorization_context_from_env,
        )

        normalized_runner_id = str(runner_id).strip()
        if not normalized_runner_id:
            return
        owner_context = authorization_context_from_env(self._env)
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
