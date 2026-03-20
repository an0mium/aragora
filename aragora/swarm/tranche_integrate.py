from __future__ import annotations

from pathlib import Path
from typing import Any

from aragora.nomic.dev_coordination import DevCoordinationStore, IntegrationDecisionType
from aragora.ralph.github_control import (
    GitHubCheck,
    GitHubControl,
    _check_is_green,
    _partition_checks,
)
from aragora.swarm.pr_registry import PullRequestRegistry
from aragora.swarm.tranche import TrancheArtifactStore
from aragora.swarm.tranche_state import (
    LANE_STATUS_COMPLETED,
    LANE_STATUS_NEEDS_HUMAN,
    TRANCHE_STATUS_NEEDS_HUMAN,
    _utcnow,
)


def discover_lane_pr(
    artifact: Any,
    *,
    github: GitHubControl | Any | None = None,
    repo_root: Path | None = None,
) -> str | None:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    deliverable = metadata.get("deliverable", {})
    if not isinstance(deliverable, dict):
        deliverable = {}

    candidates = [
        deliverable.get("pr_url"),
        metadata.get("pr_url"),
        *list(getattr(artifact, "urls", []) or []),
    ]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if _looks_like_pr_url(value):
            return value

    branch = str(
        deliverable.get("branch") or metadata.get("branch") or getattr(artifact, "branch", "") or ""
    ).strip()
    if not branch:
        return None

    control = github or GitHubControl(repo_root=(repo_root or Path.cwd()).resolve())
    return _optional_text(control.find_pr_for_branch(branch))


def register_pr(
    pr_url: str | None,
    branch: str | None,
    registry: PullRequestRegistry,
    *,
    creator: str = "tranche-integrate",
) -> dict[str, Any] | None:
    normalized_pr_url = str(pr_url or "").strip()
    normalized_branch = str(branch or "").strip()
    if not normalized_pr_url or not normalized_branch:
        return None

    existing = registry.get(normalized_branch)
    if isinstance(existing, dict) and str(existing.get("pr_url", "")).strip() == normalized_pr_url:
        return existing

    entry = registry.register(normalized_branch, normalized_pr_url, creator=creator)
    if hasattr(entry, "__dict__"):
        return dict(entry.__dict__)
    if isinstance(entry, dict):
        return dict(entry)
    return {
        "branch": normalized_branch,
        "pr_url": normalized_pr_url,
        "creator": creator,
    }


def classify_check_results(checks: list[GitHubCheck | dict[str, Any]]) -> str:
    normalized = [_normalize_check(item) for item in checks]
    required_contexts = {item.name for item in normalized if item.required}
    required, _, _required_green = _partition_checks(
        normalized,
        required_contexts=required_contexts,
        required_known=bool(required_contexts),
    )

    if any(_check_is_failed(item) for item in required):
        return "checks_failed"
    if any(not _check_is_green(item) for item in required):
        return "checks_pending"
    return "checks_passed"


def assess_lane_integration(
    *,
    artifact: Any,
    checks: str,
    review_status: str,
    merge_policy: str = "confirm",
    autonomy_mode: str = "adaptive",
    approve: bool = False,
) -> dict[str, Any]:
    normalized_checks = str(checks or "").strip().lower()
    normalized_review = str(review_status or "").strip().lower()
    normalized_policy = str(merge_policy or "confirm").strip().lower()
    normalized_autonomy = str(autonomy_mode or "adaptive").strip().lower()

    recommendation = "request_changes"
    executed = False
    rationale = "Review requested changes."

    if normalized_review not in {"passed", "approved"}:
        recommendation = "request_changes"
        rationale = "Review must pass before integration."
    elif normalized_checks == "checks_failed":
        recommendation = "request_changes"
        rationale = "Required checks are failing."
    elif normalized_checks != "checks_passed":
        recommendation = "awaiting_confirmation"
        rationale = "Required checks are still pending."
    else:
        recommendation = "merge"
        rationale = "Review passed and required checks are green."
        if normalized_policy == "manual":
            rationale = "Manual merge policy requires a human merge."
        elif normalized_autonomy == "fire_and_forget" and normalized_policy == "auto":
            executed = True
            rationale = "Fire-and-forget mode auto-executes merge for auto merge policy."
        elif normalized_autonomy == "checkpoint" and not approve:
            rationale = "Checkpoint mode requires explicit approval before merge."
        elif approve:
            executed = normalized_policy in {"auto", "confirm"}
            rationale = "Explicit approval granted for merge execution."

    return {
        "recommendation": recommendation,
        "executed": executed,
        "checks": normalized_checks,
        "review_status": normalized_review,
        "merge_policy": normalized_policy,
        "autonomy_mode": normalized_autonomy,
        "rationale": rationale,
        "lane_id": str(getattr(artifact, "lane_id", "") or "").strip() or None,
    }


def execute_lane_merge(
    pr_url: str,
    *,
    github: GitHubControl | Any,
    branch: str | None,
    registry: PullRequestRegistry | Any,
    required_checks_green: bool,
    allow_admin: bool = False,
) -> dict[str, Any]:
    merge_call = github.merge_pr(
        pr_url,
        required_checks_green=required_checks_green,
        allow_admin=allow_admin,
    )
    result = merge_call.to_dict() if hasattr(merge_call, "to_dict") else dict(merge_call)
    result["pr_url"] = str(pr_url or "").strip() or None
    result["branch"] = _optional_text(branch)
    if result.get("merged") and branch:
        registry.close(str(branch).strip(), outcome="merged")
    return result


def cascade_after_merge(
    manifest: Any,
    merged_lane_id: str,
    *,
    artifact_store: TrancheArtifactStore | Any,
    github: GitHubControl | Any,
    registry: PullRequestRegistry | Any,
    base_branch: str,
    run_state: Any | None = None,
) -> dict[str, Any]:
    """Inspect dependent lane PRs after a merge.

    Tranche lanes default to flat PRs against the manifest base branch. This
    cascade only handles explicit lane-to-lane dependencies where downstream PRs
    may need simple retargeting or manual restack after an upstream merge.
    """

    normalized_merged_lane_id = str(merged_lane_id or "").strip()
    normalized_base_branch = str(base_branch or "main").strip() or "main"
    lane_ids = {
        str(getattr(lane, "lane_id", "") or "").strip()
        for lane in getattr(manifest, "lanes", [])
        if str(getattr(lane, "lane_id", "") or "").strip()
    }
    downstream: list[dict[str, Any]] = []

    for lane in getattr(manifest, "lanes", []):
        lane_id = str(getattr(lane, "lane_id", "") or "").strip()
        dependencies = [
            str(item).strip()
            for item in getattr(lane, "dependencies", []) or []
            if str(item).strip() in lane_ids
        ]
        if not lane_id or lane_id == normalized_merged_lane_id:
            continue
        if normalized_merged_lane_id not in dependencies:
            continue

        artifact = artifact_store.load(manifest.manifest_id, lane_id)
        lane_branch = _lane_branch_hint(lane)
        branch = _artifact_branch(artifact) or lane_branch
        pr_url = discover_lane_pr(artifact, github=github) if artifact is not None else None
        if not pr_url and branch:
            pr_url = _optional_text(github.find_pr_for_branch(branch))
        if pr_url and branch:
            register_pr(pr_url, branch, registry)

        action = "ok"
        reason = "Downstream PR already targets the tranche base branch."

        if artifact is None:
            action = "missing_pr"
            reason = "Downstream lane has no tranche artifact yet."
        elif not pr_url:
            action = "missing_pr"
            reason = "Downstream lane has no discoverable PR."
        else:
            snapshot = github.fetch_gate_snapshot(pr_url)
            state = str(getattr(snapshot, "state", "") or "").strip().upper()
            current_base = _optional_text(getattr(snapshot, "base_branch", None))
            merge_state = str(getattr(snapshot, "merge_state_status", "") or "").strip().upper()

            if state != "OPEN":
                action = "needs_restack"
                reason = f"Downstream PR is {state.lower() or 'closed'} after the upstream merge."
            elif current_base and current_base != normalized_base_branch:
                retarget = github.retarget_pr_base(pr_url, normalized_base_branch)
                if retarget.get("retargeted"):
                    action = "retargeted"
                    reason = (
                        f"Retargeted downstream PR from {current_base} to {normalized_base_branch}."
                    )
                else:
                    action = "needs_restack"
                    reason = str(retarget.get("detail", "") or "Failed to retarget downstream PR.")
            elif not current_base:
                action = "needs_restack"
                reason = "Downstream PR base branch could not be determined."
            elif merge_state == "DIRTY":
                action = "needs_restack"
                reason = "Downstream PR has merge conflicts after the upstream merge."

        downstream.append(
            {
                "lane_id": lane_id,
                "pr_url": pr_url,
                "action": action,
                "reason": reason,
            }
        )

    report = {
        "merged_lane_id": normalized_merged_lane_id,
        "downstream": downstream,
        "clean": all(item["action"] in {"ok", "retargeted"} for item in downstream),
        "needs_human": any(
            item["action"] in {"needs_restack", "missing_pr"} for item in downstream
        ),
    }
    if run_state is not None:
        _apply_cascade_to_run_state(run_state, report)
    return report


async def integrate_lane(
    *,
    manifest: Any,
    artifact: Any,
    approve: bool,
    repo_root: Path | None = None,
    github: GitHubControl | Any | None = None,
    registry: PullRequestRegistry | Any | None = None,
    store: DevCoordinationStore | Any | None = None,
    artifact_store: TrancheArtifactStore | Any | None = None,
    target_branch: str = "main",
    decided_by: str = "tranche-integrate",
    rationale: str | None = None,
    allow_admin: bool = False,
    run_state: Any | None = None,
    autonomy_mode: str = "adaptive",
) -> dict[str, Any]:
    repo = (repo_root or Path.cwd()).resolve()
    github_obj = github or GitHubControl(repo_root=repo)
    registry_obj = registry or PullRequestRegistry()
    artifact_store_obj = artifact_store or TrancheArtifactStore(repo_root=repo)

    review_status = _artifact_review_status(artifact)
    pr_url = discover_lane_pr(artifact, github=github_obj, repo_root=repo)
    branch = _artifact_branch(artifact)
    if pr_url and branch:
        register_pr(pr_url, branch, registry_obj)

    gate_payload: dict[str, Any] | None = None
    checks = "checks_pending"
    if pr_url:
        snapshot = github_obj.fetch_gate_snapshot(pr_url)
        gate_payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else None
        checks = classify_check_results(
            list(getattr(snapshot, "required_checks", []))
            + list(getattr(snapshot, "advisory_checks", []))
        )

    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    assessment = assess_lane_integration(
        artifact=artifact,
        checks=checks,
        review_status=review_status,
        merge_policy=str(metadata.get("merge_policy", "confirm") or "confirm"),
        autonomy_mode=str(autonomy_mode or "adaptive").strip() or "adaptive",
        approve=approve,
    )

    merge_result: dict[str, Any] | None = None
    cascade_report: dict[str, Any] | None = None
    if approve and assessment.get("recommendation") == "merge":
        coord_store = store or DevCoordinationStore(repo_root=repo)
        await record_lane_integration(
            artifact=artifact,
            decision="merge",
            rationale=str(
                rationale or "Tranche integrate approved merge after green checks and review."
            ).strip(),
            decided_by=str(decided_by or "tranche-integrate").strip() or "tranche-integrate",
            store=coord_store,
            target_branch=str(target_branch or "main").strip() or "main",
        )
        if pr_url and assessment.get("executed"):
            merge_result = execute_lane_merge(
                pr_url,
                github=github_obj,
                branch=branch,
                registry=registry_obj,
                required_checks_green=checks == "checks_passed",
                allow_admin=allow_admin,
            )
            assessment["executed"] = bool(merge_result.get("merged", False))
            if merge_result.get("merged"):
                cascade_report = cascade_after_merge(
                    manifest,
                    str(getattr(artifact, "lane_id", "") or "").strip(),
                    artifact_store=artifact_store_obj,
                    github=github_obj,
                    registry=registry_obj,
                    base_branch=str(target_branch or "main").strip() or "main",
                    run_state=run_state,
                )

    return {
        "lane_id": str(getattr(artifact, "lane_id", "") or "").strip() or None,
        "status": getattr(artifact, "status", None),
        "pr_url": pr_url,
        "checks": checks,
        "review_status": review_status,
        "gate": gate_payload,
        "merge_result": merge_result,
        "cascade_report": cascade_report,
        **assessment,
    }


async def record_lane_integration(
    *,
    artifact: Any,
    decision: str,
    rationale: str,
    decided_by: str,
    store: DevCoordinationStore,
    target_branch: str = "main",
) -> dict[str, Any]:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    receipt_id = str(metadata.get("receipt_id", "") or "").strip()
    lease_id = str(metadata.get("lease_id", "") or "").strip()
    if not receipt_id:
        raise ValueError("Lane artifact is missing receipt_id metadata.")

    decision_enum = IntegrationDecisionType(str(decision or "").strip().lower())
    result = store.record_integration_decision(
        receipt_id=receipt_id,
        lease_id=lease_id or None,
        decision=decision_enum,
        decided_by=str(decided_by or "").strip() or "tranche-integrate",
        rationale=str(rationale or "").strip() or "Integration decision recorded.",
        target_branch=str(target_branch or "main").strip() or "main",
    )
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return dict(result)
    return {
        "receipt_id": receipt_id,
        "lease_id": lease_id or None,
        "decision": decision_enum.value,
        "target_branch": str(target_branch or "main").strip() or "main",
    }


def _normalize_check(check: GitHubCheck | dict[str, Any]) -> GitHubCheck:
    if isinstance(check, GitHubCheck):
        return check
    if not isinstance(check, dict):
        raise TypeError("check must be a dict or GitHubCheck")
    return GitHubCheck(
        name=str(check.get("name", "")).strip(),
        status=str(check.get("status", "") or "").strip().upper()
        or _status_from_conclusion(check.get("conclusion")),
        conclusion=_optional_text(check.get("conclusion"), upper=True),
        required=bool(check.get("required", False)),
        details_url=_optional_text(check.get("details_url")),
    )


def _status_from_conclusion(conclusion: Any) -> str:
    normalized = _optional_text(conclusion, upper=True)
    if not normalized:
        return "UNKNOWN"
    if normalized in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
        return "COMPLETED"
    if normalized in {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"}:
        return "COMPLETED"
    return normalized


def _check_is_failed(check: GitHubCheck) -> bool:
    conclusion = str(check.conclusion or "").strip().upper()
    if conclusion in {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"}:
        return True
    return check.status in {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"}


def _looks_like_pr_url(value: str) -> bool:
    return value.startswith("https://github.com/") and "/pull/" in value


def _artifact_review_status(artifact: Any) -> str:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    review_payload = metadata.get("review", {})
    if isinstance(review_payload, dict):
        review_status = str(review_payload.get("status", "") or "").strip() or "pending"
    else:
        review_status = "pending"
    if review_status != "pending":
        return review_status

    artifact_status = str(getattr(artifact, "status", "") or "").strip()
    if artifact_status == "review_passed":
        return "passed"
    if artifact_status == "changes_requested":
        return "changes_requested"
    if artifact_status == "review_blocked":
        return "blocked_nonreviewable"
    return "pending"


def _artifact_branch(artifact: Any) -> str | None:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    deliverable = metadata.get("deliverable", {})
    if not isinstance(deliverable, dict):
        deliverable = {}
    return _optional_text(
        deliverable.get("branch") or metadata.get("branch") or getattr(artifact, "branch", None)
    )


def _lane_branch_hint(lane: Any) -> str | None:
    branch = getattr(lane, "branch", {})
    if isinstance(branch, dict):
        return _optional_text(branch.get("current"))
    return None


def _apply_cascade_to_run_state(run_state: Any, report: dict[str, Any]) -> None:
    if not hasattr(run_state, "lane_states") or not isinstance(run_state.lane_states, dict):
        return

    merged_lane_id = _optional_text(report.get("merged_lane_id"))
    now = _utcnow()
    if merged_lane_id:
        merged_lane = run_state.lane_states.get(merged_lane_id)
        if merged_lane is not None:
            merged_lane.status = LANE_STATUS_COMPLETED
            merged_lane.last_updated = now

    needs_human = False
    for item in report.get("downstream", []):
        if not isinstance(item, dict):
            continue
        lane_id = _optional_text(item.get("lane_id"))
        action = str(item.get("action", "") or "").strip()
        if not lane_id:
            continue
        if action not in {"needs_restack", "missing_pr"}:
            continue
        needs_human = True
        lane_state = run_state.lane_states.get(lane_id)
        if lane_state is not None:
            lane_state.status = LANE_STATUS_NEEDS_HUMAN
            lane_state.last_updated = now

    if needs_human:
        run_state.status = TRANCHE_STATUS_NEEDS_HUMAN
    run_state.updated_at = now


def _optional_text(value: Any, *, upper: bool = False) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text.upper() if upper else text
