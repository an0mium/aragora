"""Ralph campaign supervisor — step-based autonomous incident commander.

Owns the outer loop above CampaignExecutor: run iterations, classify blockers,
generate repair tasks, track PR state, and resume after merge.  Each ``step()``
call advances the state machine by exactly one action.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from aragora.ralph.classifier import BlockerKind, classify_blocker
from aragora.ralph.repair import RepairTask, generate_repair_task

logger = logging.getLogger(__name__)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SupervisorStatus(str, Enum):
    RUNNING = "running"
    WAITING_FOR_PR = "waiting_for_pr"
    WAITING_FOR_MERGE = "waiting_for_merge"
    RESUMING = "resuming"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    STOPPED = "stopped"


class SupervisorAction(str, Enum):
    """The action taken by the last ``step()`` call."""

    CAMPAIGN_ITERATION = "campaign_iteration"
    BLOCKER_CLASSIFIED = "blocker_classified"
    REPAIR_GENERATED = "repair_generated"
    PR_CHECKED = "pr_checked"
    CAMPAIGN_RESUMED = "campaign_resumed"
    CAMPAIGN_COMPLETED = "campaign_completed"
    ESCALATED = "escalated"
    NOOP = "noop"


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

_DEFAULT_MAX_REPAIR_ATTEMPTS = 2


@dataclass(slots=True)
class SupervisorState:
    """Persistent state for the ralph campaign supervisor."""

    supervisor_id: str = ""
    campaign_manifest_path: str = ""
    campaign_id: str = ""
    status: str = SupervisorStatus.RUNNING.value
    current_step: int = 0
    last_campaign_result: dict[str, Any] = field(default_factory=dict)
    last_stop_reason: str = ""
    active_blocker: str | None = None
    blocker_history: list[dict[str, Any]] = field(default_factory=list)
    repair_attempts: int = 0
    max_repair_attempts: int = _DEFAULT_MAX_REPAIR_ATTEMPTS
    active_repair_pr: str | None = None
    active_repair_branch: str | None = None
    active_repair_task: dict[str, Any] | None = None
    merge_commit_sha: str | None = None
    resume_cursor: str | None = None
    budget_spent_usd: float = 0.0
    escalation_reason: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "supervisor_id": self.supervisor_id,
            "campaign_manifest_path": self.campaign_manifest_path,
            "campaign_id": self.campaign_id,
            "status": self.status,
            "current_step": self.current_step,
            "last_campaign_result": self.last_campaign_result,
            "last_stop_reason": self.last_stop_reason,
            "active_blocker": self.active_blocker,
            "blocker_history": list(self.blocker_history),
            "repair_attempts": self.repair_attempts,
            "max_repair_attempts": self.max_repair_attempts,
            "active_repair_pr": self.active_repair_pr,
            "active_repair_branch": self.active_repair_branch,
            "active_repair_task": self.active_repair_task,
            "merge_commit_sha": self.merge_commit_sha,
            "resume_cursor": self.resume_cursor,
            "budget_spent_usd": self.budget_spent_usd,
            "escalation_reason": self.escalation_reason,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupervisorState:
        return cls(
            supervisor_id=str(data.get("supervisor_id", "")),
            campaign_manifest_path=str(data.get("campaign_manifest_path", "")),
            campaign_id=str(data.get("campaign_id", "")),
            status=str(data.get("status", SupervisorStatus.RUNNING.value)),
            current_step=int(data.get("current_step", 0)),
            last_campaign_result=dict(data.get("last_campaign_result") or {}),
            last_stop_reason=str(data.get("last_stop_reason", "")),
            active_blocker=data.get("active_blocker"),
            blocker_history=list(data.get("blocker_history") or []),
            repair_attempts=int(data.get("repair_attempts", 0)),
            max_repair_attempts=int(data.get("max_repair_attempts", _DEFAULT_MAX_REPAIR_ATTEMPTS)),
            active_repair_pr=data.get("active_repair_pr"),
            active_repair_branch=data.get("active_repair_branch"),
            active_repair_task=data.get("active_repair_task"),
            merge_commit_sha=data.get("merge_commit_sha"),
            resume_cursor=data.get("resume_cursor"),
            budget_spent_usd=float(data.get("budget_spent_usd", 0.0)),
            escalation_reason=data.get("escalation_reason"),
            updated_at=str(data.get("updated_at", "")),
        )


def load_supervisor_state(path: Path) -> SupervisorState:
    if not path.exists():
        raise FileNotFoundError(f"Supervisor state not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid supervisor state: expected dict, got {type(data).__name__}")
    return SupervisorState.from_dict(data)


def save_supervisor_state(path: Path, state: SupervisorState) -> None:
    state.updated_at = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(state.to_dict(), default_flow_style=False, sort_keys=False)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StepResult:
    action: str
    status: str
    detail: str = ""
    repair_task: RepairTask | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "action": self.action,
            "status": self.status,
            "detail": self.detail,
        }
        if self.repair_task:
            d["repair_task"] = self.repair_task.to_dict()
        return d


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class RalphSupervisor:
    """Step-based campaign supervisor.

    Each call to :meth:`step` reads the current state, performs exactly one
    action, saves the updated state, and returns a :class:`StepResult`.

    Designed for cron/ralph-loop invocation::

        while True:
            result = supervisor.step()
            if result.status in ("completed", "escalated", "stopped"):
                break
    """

    def __init__(
        self,
        *,
        state_path: Path,
        repo_root: Path | None = None,
        merge_policy: str = "manual_review_required",
    ) -> None:
        self.state_path = state_path.resolve()
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.merge_policy = merge_policy

    # -- public API --

    @classmethod
    def start(
        cls,
        *,
        manifest_path: Path,
        state_path: Path,
        repo_root: Path | None = None,
        merge_policy: str = "manual_review_required",
        max_repair_attempts: int = _DEFAULT_MAX_REPAIR_ATTEMPTS,
    ) -> "RalphSupervisor":
        """Initialize a new supervisor run and persist state."""
        manifest_path = manifest_path.resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Campaign manifest not found: {manifest_path}")

        # Read campaign_id from manifest.
        manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        campaign_id = str(manifest_data.get("campaign_id", "unknown"))

        state = SupervisorState(
            supervisor_id=f"ralph-{uuid.uuid4().hex[:12]}",
            campaign_manifest_path=str(manifest_path),
            campaign_id=campaign_id,
            status=SupervisorStatus.RUNNING.value,
            max_repair_attempts=max_repair_attempts,
        )
        state_path = state_path.resolve()
        save_supervisor_state(state_path, state)

        supervisor = cls(
            state_path=state_path,
            repo_root=repo_root,
            merge_policy=merge_policy,
        )
        logger.info(
            "Ralph supervisor started: %s (campaign=%s)",
            state.supervisor_id,
            campaign_id,
        )
        return supervisor

    def step(self) -> StepResult:
        """Advance the supervisor state machine by one action."""
        state = load_supervisor_state(self.state_path)
        state.current_step += 1

        if state.status in (
            SupervisorStatus.COMPLETED.value,
            SupervisorStatus.ESCALATED.value,
            SupervisorStatus.STOPPED.value,
        ):
            return StepResult(
                action=SupervisorAction.NOOP.value,
                status=state.status,
                detail="Supervisor is in terminal state.",
            )

        if state.status == SupervisorStatus.WAITING_FOR_MERGE.value:
            result = self._step_check_merge(state)
            save_supervisor_state(self.state_path, state)
            return result

        if state.status == SupervisorStatus.WAITING_FOR_PR.value:
            result = self._step_check_pr(state)
            save_supervisor_state(self.state_path, state)
            return result

        if state.status == SupervisorStatus.RESUMING.value:
            result = self._step_resume(state)
            save_supervisor_state(self.state_path, state)
            return result

        # status == RUNNING: run one campaign iteration.
        if state.active_blocker:
            result = self._step_handle_blocker(state)
            save_supervisor_state(self.state_path, state)
            return result

        result = self._step_campaign_iteration(state)
        save_supervisor_state(self.state_path, state)
        return result

    def status(self) -> dict[str, Any]:
        """Return current supervisor state as dict."""
        state = load_supervisor_state(self.state_path)
        return state.to_dict()

    def stop(self) -> StepResult:
        """Gracefully stop the supervisor."""
        state = load_supervisor_state(self.state_path)
        state.status = SupervisorStatus.STOPPED.value
        save_supervisor_state(self.state_path, state)
        return StepResult(
            action=SupervisorAction.NOOP.value,
            status=SupervisorStatus.STOPPED.value,
            detail="Supervisor stopped by operator.",
        )

    # -- step implementations --

    def _step_campaign_iteration(self, state: SupervisorState) -> StepResult:
        """Run one CampaignExecutor.execute_once() iteration."""
        from aragora.swarm.campaign import CampaignExecutor

        manifest_path = Path(state.campaign_manifest_path)
        executor = CampaignExecutor(
            manifest_path=manifest_path,
            repo_root=self.repo_root,
        )

        try:
            result = asyncio.run(executor.execute_once())
        except Exception as exc:
            logger.warning("Campaign iteration failed: %s", exc)
            state.last_campaign_result = {"error": str(exc)}
            state.active_blocker = BlockerKind.INFRA_FAILURE.value
            state.blocker_history.append(
                {
                    "step": state.current_step,
                    "kind": BlockerKind.INFRA_FAILURE.value,
                    "detail": str(exc),
                }
            )
            return StepResult(
                action=SupervisorAction.BLOCKER_CLASSIFIED.value,
                status=state.status,
                detail=f"Campaign iteration raised: {type(exc).__name__}: {exc}",
            )

        state.last_campaign_result = result
        stop_reason = str(result.get("stop_reason", ""))
        state.last_stop_reason = stop_reason

        # Update budget from manifest.
        try:
            manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            exec_state = manifest_data.get("execution_state", {})
            state.budget_spent_usd = float(exec_state.get("total_cost_usd", 0.0))
        except Exception:
            pass

        if stop_reason == "campaign_complete":
            state.status = SupervisorStatus.COMPLETED.value
            return StepResult(
                action=SupervisorAction.CAMPAIGN_COMPLETED.value,
                status=SupervisorStatus.COMPLETED.value,
                detail="All projects completed or skipped.",
            )

        if stop_reason == "still_running":
            return StepResult(
                action=SupervisorAction.CAMPAIGN_ITERATION.value,
                status=SupervisorStatus.RUNNING.value,
                detail=f"Dispatched: {result.get('dispatched_projects', [])}",
            )

        # Campaign is blocked. Classify.
        manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        blocker = classify_blocker(
            stop_reason=stop_reason,
            manifest_dict=manifest_data,
        )

        if blocker is None:
            return StepResult(
                action=SupervisorAction.CAMPAIGN_ITERATION.value,
                status=SupervisorStatus.RUNNING.value,
                detail="No blocker classified; will retry.",
            )

        state.active_blocker = blocker.value
        state.blocker_history.append(
            {"step": state.current_step, "kind": blocker.value, "stop_reason": stop_reason}
        )

        return StepResult(
            action=SupervisorAction.BLOCKER_CLASSIFIED.value,
            status=state.status,
            detail=f"Blocker classified: {blocker.value}",
        )

    def _step_handle_blocker(self, state: SupervisorState) -> StepResult:
        """Generate a repair task for a deterministic blocker, or escalate."""
        blocker_kind_str = state.active_blocker or ""
        try:
            blocker_kind = BlockerKind(blocker_kind_str)
        except ValueError:
            return self._escalate(state, f"Unknown blocker kind: {blocker_kind_str}")

        if not blocker_kind.is_deterministic:
            return self._escalate(state, f"Non-deterministic blocker: {blocker_kind.value}")

        if state.repair_attempts >= state.max_repair_attempts:
            return self._escalate(
                state,
                f"Exceeded max repair attempts ({state.max_repair_attempts}) "
                f"for blocker: {blocker_kind.value}",
            )

        # Find affected project IDs.
        affected: list[str] = []
        try:
            manifest_path = Path(state.campaign_manifest_path)
            manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            for proj in manifest_data.get("projects", []):
                if proj.get("status") in ("blocked", "failed", "skipped"):
                    affected.append(str(proj.get("project_id", "")))
        except Exception:
            pass

        repair = generate_repair_task(blocker_kind, affected_project_ids=affected)
        if repair is None:
            return self._escalate(state, f"No repair template for: {blocker_kind.value}")

        state.repair_attempts += 1
        state.active_repair_task = repair.to_dict()
        state.status = SupervisorStatus.WAITING_FOR_PR.value

        return StepResult(
            action=SupervisorAction.REPAIR_GENERATED.value,
            status=SupervisorStatus.WAITING_FOR_PR.value,
            detail=f"Repair task generated: {repair.title}",
            repair_task=repair,
        )

    def _step_check_pr(self, state: SupervisorState) -> StepResult:
        """Check if a repair PR has been opened."""
        if state.active_repair_pr:
            # PR exists, wait for merge.
            state.status = SupervisorStatus.WAITING_FOR_MERGE.value
            return StepResult(
                action=SupervisorAction.PR_CHECKED.value,
                status=SupervisorStatus.WAITING_FOR_MERGE.value,
                detail=f"PR found: {state.active_repair_pr}. Waiting for merge.",
            )

        # Check if a PR was created on the repair branch.
        if state.active_repair_branch:
            pr_url = self._find_pr_for_branch(state.active_repair_branch)
            if pr_url:
                state.active_repair_pr = pr_url
                state.status = SupervisorStatus.WAITING_FOR_MERGE.value
                return StepResult(
                    action=SupervisorAction.PR_CHECKED.value,
                    status=SupervisorStatus.WAITING_FOR_MERGE.value,
                    detail=f"PR discovered: {pr_url}. Waiting for merge.",
                )

        return StepResult(
            action=SupervisorAction.PR_CHECKED.value,
            status=SupervisorStatus.WAITING_FOR_PR.value,
            detail="No PR found yet. Still waiting.",
        )

    def _step_check_merge(self, state: SupervisorState) -> StepResult:
        """Check if the repair PR has been merged."""
        pr_url = state.active_repair_pr
        if not pr_url:
            state.status = SupervisorStatus.WAITING_FOR_PR.value
            return StepResult(
                action=SupervisorAction.PR_CHECKED.value,
                status=SupervisorStatus.WAITING_FOR_PR.value,
                detail="No PR URL. Reverting to waiting_for_pr.",
            )

        merged, merge_sha = self._check_pr_merged(pr_url)
        if not merged:
            return StepResult(
                action=SupervisorAction.PR_CHECKED.value,
                status=SupervisorStatus.WAITING_FOR_MERGE.value,
                detail=f"PR {pr_url} not yet merged.",
            )

        state.merge_commit_sha = merge_sha
        state.status = SupervisorStatus.RESUMING.value
        return StepResult(
            action=SupervisorAction.PR_CHECKED.value,
            status=SupervisorStatus.RESUMING.value,
            detail=f"PR merged (SHA: {merge_sha}). Ready to resume campaign.",
        )

    def _step_resume(self, state: SupervisorState) -> StepResult:
        """Resume campaign after a repair PR was merged."""
        # Fetch latest main.
        try:
            subprocess.run(
                ["git", "fetch", "origin", "main"],
                capture_output=True,
                cwd=str(self.repo_root),
                timeout=30,
            )
        except Exception as exc:
            logger.debug("git fetch failed: %s", exc)

        # Clear blocker state.
        state.active_blocker = None
        state.active_repair_pr = None
        state.active_repair_branch = None
        state.active_repair_task = None
        state.merge_commit_sha = None
        state.status = SupervisorStatus.RUNNING.value

        return StepResult(
            action=SupervisorAction.CAMPAIGN_RESUMED.value,
            status=SupervisorStatus.RUNNING.value,
            detail="Campaign resumed after repair merge. Next step will run iteration.",
        )

    # -- helpers --

    def _escalate(self, state: SupervisorState, reason: str) -> StepResult:
        state.status = SupervisorStatus.ESCALATED.value
        state.escalation_reason = reason
        logger.warning("Ralph supervisor escalating: %s", reason)
        return StepResult(
            action=SupervisorAction.ESCALATED.value,
            status=SupervisorStatus.ESCALATED.value,
            detail=reason,
        )

    def _find_pr_for_branch(self, branch: str) -> str | None:
        """Use gh CLI to find an open PR for the given branch."""
        try:
            result = subprocess.run(
                ["gh", "pr", "list", "--head", branch, "--json", "url", "--limit", "1"],
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                timeout=15,
            )
            if result.returncode == 0:
                prs = json.loads(result.stdout)
                if prs and isinstance(prs, list):
                    return str(prs[0].get("url", ""))
        except Exception as exc:
            logger.debug("gh pr list failed: %s", exc)
        return None

    def _check_pr_merged(self, pr_url: str) -> tuple[bool, str | None]:
        """Check if a PR has been merged using gh CLI."""
        try:
            result = subprocess.run(
                ["gh", "pr", "view", pr_url, "--json", "state,mergeCommit"],
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                timeout=15,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data.get("state") == "MERGED":
                    merge_commit = data.get("mergeCommit", {})
                    sha = str(merge_commit.get("oid", "")) if merge_commit else None
                    return True, sha
        except Exception as exc:
            logger.debug("gh pr view failed: %s", exc)
        return False, None
