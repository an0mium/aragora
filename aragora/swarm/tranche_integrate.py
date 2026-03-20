from __future__ import annotations

import logging
import subprocess
from pathlib import Path, PurePosixPath
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

logger = logging.getLogger(__name__)

_LOW_RISK_PROTECTED_PREFIXES = (
    ".github/workflows/",
    "deploy/",
    "infra/",
    "infrastructure/",
    "terraform/",
    "k8s/",
    "kubernetes/",
    "helm/",
    "docker/",
    "migrations/",
    "alembic/",
)
_LOW_RISK_PROTECTED_PARTS = frozenset(
    {
        "auth",
        "oauth",
        "sso",
        "security",
        "billing",
        "payment",
        "payments",
        "migrations",
        "migration",
        "schema",
        "schemas",
        "alembic",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "token",
        "tokens",
    }
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
    manifest: Any | None = None,
    merge_policy: str = "confirm",
    autonomy_mode: str = "adaptive",
    approve: bool = False,
) -> dict[str, Any]:
    normalized_checks = str(checks or "").strip().lower()
    normalized_review = str(review_status or "").strip().lower()
    normalized_policy = _lane_policy_value(
        manifest,
        artifact,
        key="merge_policy",
        fallback=str(merge_policy or "confirm").strip().lower() or "confirm",
    )
    normalized_autonomy = str(autonomy_mode or "adaptive").strip().lower()
    merge_class = _lane_policy_value(manifest, artifact, key="merge_class", fallback="manual")
    low_risk_policy = (
        _evaluate_low_risk_merge_policy(manifest=manifest, artifact=artifact)
        if merge_class == "low_risk"
        else None
    )

    recommendation = "request_changes"
    executed = False
    rationale = "Review requested changes."

    if merge_class == "low_risk":
        if normalized_review not in {"passed", "approved"}:
            recommendation = "needs_human"
            rationale = "Low-risk auto-merge requires a passing review."
        elif normalized_checks == "checks_failed":
            recommendation = "needs_human"
            rationale = "Low-risk auto-merge requires green required checks."
        elif normalized_checks != "checks_passed":
            recommendation = "awaiting_checks"
            rationale = "Required checks are still pending before low-risk auto-merge can proceed."
        elif isinstance(low_risk_policy, dict) and not bool(low_risk_policy.get("eligible", False)):
            recommendation = "needs_human"
            rationale = (
                "Low-risk auto-merge is not eligible; leave the PR for human review: "
                + _format_low_risk_reasons(low_risk_policy)
            )
        elif normalized_autonomy == "fire_and_forget" and normalized_policy == "auto":
            recommendation = "merge"
            executed = True
            rationale = "Low-risk auto-merge policy passed and fire-and-forget executed the merge."
        elif approve and normalized_policy in {"auto", "confirm"}:
            recommendation = "merge"
            executed = True
            rationale = (
                "Low-risk auto-merge policy passed and explicit approval executed the merge."
            )
        else:
            recommendation = "needs_human"
            rationale = "Low-risk lane is merge-eligible, but explicit approval is required."
    elif normalized_review not in {"passed", "approved"}:
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
        "merge_class": merge_class,
        "merge_policy": normalized_policy,
        "autonomy_mode": normalized_autonomy,
        "rationale": rationale,
        "low_risk_policy": low_risk_policy,
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


def publish_lane_deliverable(
    artifact: Any,
    *,
    manifest_id: str,
    github: GitHubControl | Any,
    registry: PullRequestRegistry | Any,
    repo_root: Path,
    target_branch: str,
    artifact_store: TrancheArtifactStore | Any | None = None,
) -> dict[str, Any]:
    branch = _artifact_branch(artifact)
    if not branch:
        return {
            "published": False,
            "action": "skipped",
            "branch": None,
            "pr_url": None,
            "detail": "Lane artifact has no branch deliverable to publish.",
        }

    existing_pr = discover_lane_pr(artifact, github=github, repo_root=repo_root)
    if existing_pr:
        register_pr(existing_pr, branch, registry)
        _persist_published_pr(
            artifact,
            manifest_id=manifest_id,
            pr_url=existing_pr,
            branch=branch,
            artifact_store=artifact_store,
        )
        return {
            "published": True,
            "action": "existing_pr",
            "branch": branch,
            "pr_url": existing_pr,
            "detail": "Existing PR discovered for lane branch.",
        }

    commit_shas = _artifact_commit_shas(artifact)
    if not commit_shas:
        return {
            "published": False,
            "action": "skipped",
            "branch": branch,
            "pr_url": None,
            "detail": "Lane artifact has no committed branch deliverable to publish.",
        }

    push_result = _push_branch_to_origin(repo_root, branch)
    if not push_result["pushed"]:
        _record_publish_attempt(
            artifact,
            manifest_id=manifest_id,
            publish_result=push_result,
            artifact_store=artifact_store,
        )
        return {
            "published": False,
            "action": "push_failed",
            "branch": branch,
            "pr_url": None,
            "detail": str(push_result.get("detail", "") or "Failed to push branch."),
        }

    discovered_pr = _optional_text(github.find_pr_for_branch(branch))
    if discovered_pr:
        register_pr(discovered_pr, branch, registry)
        _persist_published_pr(
            artifact,
            manifest_id=manifest_id,
            pr_url=discovered_pr,
            branch=branch,
            artifact_store=artifact_store,
        )
        return {
            "published": True,
            "action": "discovered_after_push",
            "branch": branch,
            "pr_url": discovered_pr,
            "detail": "Branch pushed and existing PR discovered.",
        }

    try:
        created_pr = github.create_pr_for_branch(branch, target_branch)
    except Exception as exc:
        logger.warning("pr create failed for branch %s: %s", branch, exc)
        failure = {
            "published": False,
            "action": "pr_create_failed",
            "branch": branch,
            "pr_url": None,
            "detail": "gh pr create failed. Check logs for detail.",
        }
        _record_publish_attempt(
            artifact,
            manifest_id=manifest_id,
            publish_result=failure,
            artifact_store=artifact_store,
        )
        return failure

    register_pr(created_pr, branch, registry)
    _persist_published_pr(
        artifact,
        manifest_id=manifest_id,
        pr_url=created_pr,
        branch=branch,
        artifact_store=artifact_store,
    )
    return {
        "published": True,
        "action": "pr_created",
        "branch": branch,
        "pr_url": created_pr,
        "detail": f"Branch pushed and PR created against {target_branch}.",
    }


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
    publish_result: dict[str, Any] | None = None
    if not pr_url:
        publish_result = publish_lane_deliverable(
            artifact,
            manifest_id=str(getattr(manifest, "manifest_id", "") or "").strip(),
            github=github_obj,
            registry=registry_obj,
            repo_root=repo,
            target_branch=str(target_branch or "main").strip() or "main",
            artifact_store=artifact_store_obj,
        )
        pr_url = _optional_text(publish_result.get("pr_url")) if publish_result else None
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
        manifest=manifest,
        checks=checks,
        review_status=review_status,
        merge_policy=str(metadata.get("merge_policy", "confirm") or "confirm"),
        autonomy_mode=str(autonomy_mode or "adaptive").strip() or "adaptive",
        approve=approve,
    )
    if not pr_url:
        detail = (
            str(publish_result.get("detail", "")).strip()
            if isinstance(publish_result, dict)
            else "No PR could be discovered or published for the lane deliverable."
        )
        assessment = {
            **assessment,
            "recommendation": "needs_human",
            "executed": False,
            "rationale": detail
            or "No PR could be discovered or published for the lane deliverable.",
        }

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
        "publish_result": publish_result,
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
    review_payload = _artifact_review_payload(artifact)
    review_status = str(review_payload.get("status", "") or "").strip() or "pending"
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


def _artifact_review_payload(artifact: Any) -> dict[str, Any]:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        return {}
    review_payload = metadata.get("review")
    return dict(review_payload) if isinstance(review_payload, dict) else {}


def _artifact_review_tier(artifact: Any) -> int | None:
    review_payload = _artifact_review_payload(artifact)
    value = review_payload.get("tier")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _artifact_changed_files(artifact: Any) -> list[str]:
    review_payload = _artifact_review_payload(artifact)
    value = review_payload.get("changed_files", [])
    if not isinstance(value, list):
        return []
    return [text for text in (_normalize_repo_path(item) for item in value) if text]


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


def _artifact_commit_shas(artifact: Any) -> list[str]:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        return []
    deliverable = metadata.get("deliverable", {})
    if not isinstance(deliverable, dict):
        return []
    return [str(item).strip() for item in deliverable.get("commit_shas", []) if str(item).strip()]


def _manifest_lane(manifest: Any | None, lane_id: str) -> Any | None:
    normalized_lane_id = str(lane_id or "").strip()
    if not normalized_lane_id or manifest is None:
        return None
    lane_getter = getattr(manifest, "lane", None)
    if callable(lane_getter):
        try:
            lane = lane_getter(normalized_lane_id)
        except Exception:
            lane = None
        if lane is not None:
            return lane
    for lane in getattr(manifest, "lanes", []) or []:
        if str(getattr(lane, "lane_id", "") or "").strip() == normalized_lane_id:
            return lane
    return None


def _lane_metadata(manifest: Any | None, artifact: Any) -> dict[str, Any]:
    lane = _manifest_lane(manifest, str(getattr(artifact, "lane_id", "") or "").strip())
    metadata = getattr(lane, "metadata", {}) if lane is not None else {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def _lane_policy_value(
    manifest: Any | None,
    artifact: Any,
    *,
    key: str,
    fallback: str,
) -> str:
    metadata = getattr(artifact, "metadata", {})
    if isinstance(metadata, dict):
        value = _optional_text(metadata.get(key))
        if value:
            return value.lower()
    lane_metadata = _lane_metadata(manifest, artifact)
    value = _optional_text(lane_metadata.get(key))
    if value:
        return value.lower()
    return str(fallback or "").strip().lower() or fallback


def _evaluate_low_risk_merge_policy(*, manifest: Any | None, artifact: Any) -> dict[str, Any]:
    lane_metadata = _lane_metadata(manifest, artifact)
    lane_count = len(getattr(manifest, "lanes", []) or []) if manifest is not None else 0
    tier = _artifact_review_tier(artifact)
    changed_files = _artifact_changed_files(artifact)
    protected_paths = _protected_paths(changed_files)
    reasons: list[str] = []
    if lane_count != 1:
        reasons.append("tranche contains multiple lanes")
    if tier != 1:
        reasons.append(f"review tier is {tier if tier is not None else 'unknown'} instead of 1")
    if not bool(lane_metadata.get("enforce_cross_model_review", True)):
        reasons.append("cross-model review is disabled for this lane")
    if not changed_files:
        reasons.append("changed files were not recorded for the reviewed run")
    if protected_paths:
        reasons.append(
            "touched protected paths: " + ", ".join(sorted(dict.fromkeys(protected_paths)))
        )
    return {
        "eligible": not reasons,
        "lane_count": lane_count,
        "review_tier": tier,
        "changed_files": changed_files,
        "protected_paths": protected_paths,
        "reasons": reasons,
    }


def _format_low_risk_reasons(policy: dict[str, Any] | None) -> str:
    if not isinstance(policy, dict):
        return "low-risk policy details are unavailable."
    reasons = policy.get("reasons", [])
    if not isinstance(reasons, list) or not reasons:
        return "low-risk policy details are unavailable."
    return "; ".join(str(item).strip() for item in reasons if str(item).strip())


def _protected_paths(paths: list[str]) -> list[str]:
    protected: list[str] = []
    seen: set[str] = set()
    for item in paths:
        normalized = _normalize_repo_path(item)
        if not normalized or normalized in seen:
            continue
        if _is_protected_path(normalized):
            seen.add(normalized)
            protected.append(normalized)
    return protected


def _is_protected_path(path: str) -> bool:
    lowered = _normalize_repo_path(path).lower()
    if not lowered:
        return False
    if any(lowered.startswith(prefix) for prefix in _LOW_RISK_PROTECTED_PREFIXES):
        return True
    parts = [part for part in PurePosixPath(lowered).parts if part not in {"", "."}]
    return any(part in _LOW_RISK_PROTECTED_PARTS for part in parts)


def _normalize_repo_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _persist_published_pr(
    artifact: Any,
    *,
    manifest_id: str,
    pr_url: str,
    branch: str,
    artifact_store: TrancheArtifactStore | Any | None,
) -> None:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        artifact.metadata = metadata
    deliverable = metadata.get("deliverable", {})
    if not isinstance(deliverable, dict):
        deliverable = {}
        metadata["deliverable"] = deliverable
    deliverable["branch"] = branch
    deliverable["pr_url"] = pr_url
    metadata["branch"] = branch
    metadata["pr_url"] = pr_url
    metadata["publish"] = {
        "published": True,
        "action": "pr_available",
        "branch": branch,
        "pr_url": pr_url,
        "detail": "Controller confirmed PR availability for the lane branch.",
    }
    urls = list(getattr(artifact, "urls", []) or [])
    if pr_url not in urls:
        urls.append(pr_url)
        artifact.urls = urls
    if artifact_store is not None and manifest_id:
        artifact_store.save(manifest_id, artifact)


def _record_publish_attempt(
    artifact: Any,
    *,
    manifest_id: str,
    publish_result: dict[str, Any],
    artifact_store: TrancheArtifactStore | Any | None,
) -> None:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        artifact.metadata = metadata
    metadata["publish"] = dict(publish_result)
    if artifact_store is not None and manifest_id:
        artifact_store.save(manifest_id, artifact)


def _push_branch_to_origin(repo_root: Path, branch: str) -> dict[str, Any]:
    normalized_branch = str(branch or "").strip()
    if not normalized_branch:
        return {
            "pushed": False,
            "branch": None,
            "detail": "Branch name is required.",
        }
    try:
        result = subprocess.run(
            ["git", "push", "origin", normalized_branch],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("git push failed for branch %s: %s", normalized_branch, exc)
        return {
            "pushed": False,
            "branch": normalized_branch,
            "detail": "git push failed before completion. Check logs for detail.",
        }
    detail = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        logger.warning("git push failed for branch %s: %s", normalized_branch, detail)
        detail = "git push failed. Check logs for detail."
    return {
        "pushed": result.returncode == 0,
        "branch": normalized_branch,
        "detail": detail or ("git push succeeded" if result.returncode == 0 else "git push failed"),
    }


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
