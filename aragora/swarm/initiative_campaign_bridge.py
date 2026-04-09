from __future__ import annotations

from pathlib import Path

from aragora.swarm.campaign import (
    CampaignDependency,
    CampaignExecutionState,
    CampaignManifest,
    CampaignProject,
    CampaignProjectStatus,
    CampaignReviewGate,
    _complexity_cost,
    _refresh_execution_state,
    load_campaign_manifest,
    save_campaign_manifest,
)
from aragora.swarm.initiative_models import InitiativeMilestone, InitiativeRecord, InitiativeSlice
from aragora.swarm.initiative_store import InitiativeStore
from aragora.swarm.spec import SwarmSpec


_SLICE_STATUS_TO_PROJECT_STATUS = {
    "active": CampaignProjectStatus.ACTIVE.value,
    "blocked": CampaignProjectStatus.BLOCKED.value,
    "merged": CampaignProjectStatus.COMPLETED.value,
    "needs_human": CampaignProjectStatus.NEEDS_REVISION.value,
    "queued": CampaignProjectStatus.PENDING.value,
    "superseded": CampaignProjectStatus.SKIPPED.value,
}


def sync_campaign_manifest_for_initiative(
    store: InitiativeStore,
    initiative: InitiativeRecord,
    *,
    planner_model: str = "claude",
    planner_strategy: str = "heuristic",
    worker_model: str = "claude",
    review_model: str = "codex",
) -> Path:
    destination = store.campaign_manifest_path_for(initiative.initiative_id)
    existing_manifest = load_campaign_manifest(destination) if destination.exists() else None
    manifest = manifest_from_initiative(
        initiative,
        source_ref=str(store.path_for(initiative.initiative_id)),
        existing_manifest=existing_manifest,
        planner_model=planner_model,
        planner_strategy=planner_strategy,
        worker_model=worker_model,
        review_model=review_model,
    )
    save_campaign_manifest(destination, manifest)
    return destination


def manifest_from_initiative(
    initiative: InitiativeRecord,
    *,
    source_ref: str,
    existing_manifest: CampaignManifest | None = None,
    planner_model: str = "claude",
    planner_strategy: str = "heuristic",
    worker_model: str = "claude",
    review_model: str = "codex",
) -> CampaignManifest:
    milestones_by_slice = _milestones_by_slice_id(initiative.milestones)
    existing_projects = existing_manifest.project_map() if existing_manifest else {}
    review_required = review_model.strip().lower() != worker_model.strip().lower()
    projects = [
        _project_from_slice(
            initiative=initiative,
            slice_record=slice_record,
            milestone_title=milestones_by_slice.get(slice_record.slice_id),
            existing_project=existing_projects.get(slice_record.slice_id),
            review_model=review_model,
            review_required=review_required,
        )
        for slice_record in initiative.slices
    ]
    planning_findings = list(existing_manifest.planning_findings) if existing_manifest else []
    bridge_finding = f"materialized from initiative record {initiative.initiative_id}"
    if bridge_finding not in planning_findings:
        planning_findings.append(bridge_finding)
    manifest = CampaignManifest(
        campaign_id=initiative.initiative_id,
        created_at=initiative.created_at,
        source_kind="initiative",
        source_ref=source_ref,
        planner_model=planner_model,
        planner_strategy=planner_strategy,
        worker_model=worker_model,
        review_model=review_model,
        enforce_cross_model_review=review_required,
        experiment_id=existing_manifest.experiment_id if existing_manifest else None,
        experiment_label=existing_manifest.experiment_label if existing_manifest else None,
        max_parallel_ready_projects=(
            existing_manifest.max_parallel_ready_projects if existing_manifest else 2
        ),
        max_retries_per_project=existing_manifest.max_retries_per_project
        if existing_manifest
        else 2,
        budget_limit_usd=existing_manifest.budget_limit_usd if existing_manifest else 50.0,
        time_limit_hours=existing_manifest.time_limit_hours if existing_manifest else 8.0,
        projects=projects,
        execution_state=existing_manifest.execution_state
        if existing_manifest
        else CampaignExecutionState(),
        planning_findings=planning_findings,
        manifest_version=existing_manifest.manifest_version if existing_manifest else 1,
    )
    _refresh_execution_state(manifest)
    return manifest


def _project_from_slice(
    *,
    initiative: InitiativeRecord,
    slice_record: InitiativeSlice,
    milestone_title: str | None,
    existing_project: CampaignProject | None,
    review_model: str,
    review_required: bool,
) -> CampaignProject:
    spec = _spec_from_slice(
        initiative=initiative,
        slice_record=slice_record,
        existing_spec=existing_project.spec if existing_project else None,
    )
    dependencies = [
        CampaignDependency(project_id=dependency_id, reason="initiative_dependency")
        for dependency_id in slice_record.dependencies
    ]
    return CampaignProject(
        project_id=slice_record.slice_id,
        title=slice_record.title or slice_record.slice_id,
        source_refs=_dedupe(
            (list(existing_project.source_refs) if existing_project else [])
            + [initiative.initiative_id, slice_record.slice_id]
        ),
        milestone=milestone_title,
        spec=spec,
        file_scope_hints=_dedupe(slice_record.file_scope or list(spec.file_scope_hints)),
        acceptance_criteria=_dedupe(list(spec.acceptance_criteria)),
        constraints=_dedupe(spec.constraints),
        dependencies=dependencies,
        feature_flag=initiative.feature_flag_name
        or (existing_project.feature_flag if existing_project else None),
        feature_flag_required=bool(initiative.feature_flag_name)
        or bool(existing_project.feature_flag_required if existing_project else False),
        estimated_cost_usd=_complexity_cost(slice_record.estimated_complexity),
        status=_project_status(slice_record.status, existing_project),
        retry_count=existing_project.retry_count if existing_project else 0,
        last_run_outcome=existing_project.last_run_outcome if existing_project else None,
        run_id=existing_project.run_id if existing_project else None,
        worker_receipt_id=existing_project.worker_receipt_id if existing_project else None,
        receipt_id=existing_project.receipt_id if existing_project else None,
        pr_url=existing_project.pr_url if existing_project else None,
        adopted_pr=existing_project.adopted_pr if existing_project else None,
        branch=existing_project.branch if existing_project else None,
        commit_shas=list(existing_project.commit_shas) if existing_project else [],
        attempt_history=list(existing_project.attempt_history) if existing_project else [],
        review=_review_gate(existing_project, review_model=review_model, required=review_required),
    )


def _spec_from_slice(
    *,
    initiative: InitiativeRecord,
    slice_record: InitiativeSlice,
    existing_spec: SwarmSpec | None,
) -> SwarmSpec:
    acceptance_criteria = _dedupe(
        list(slice_record.acceptance_criteria)
        + list(slice_record.validations)
        + list(initiative.validations)
    )
    file_scope_hints = _dedupe(list(slice_record.file_scope))
    constraints = list(existing_spec.constraints) if existing_spec else []
    if not constraints:
        constraints = ["Stay within the initiative slice scope."]
    spec = SwarmSpec(
        raw_goal=slice_record.description or slice_record.title,
        refined_goal=slice_record.description or slice_record.title,
        acceptance_criteria=acceptance_criteria,
        constraints=_dedupe(constraints),
        file_scope_hints=file_scope_hints,
        budget_limit_usd=existing_spec.budget_limit_usd if existing_spec else 5.0,
        estimated_complexity=slice_record.estimated_complexity,
        requires_approval=existing_spec.requires_approval if existing_spec else True,
        user_expertise=existing_spec.user_expertise if existing_spec else "developer",
    )
    if existing_spec is not None:
        spec.id = existing_spec.id
        spec.created_at = existing_spec.created_at
        spec.track_hints = list(existing_spec.track_hints)
        spec.work_orders = list(existing_spec.work_orders)
        spec.proactive_suggestions = list(existing_spec.proactive_suggestions)
        spec.research_context = dict(existing_spec.research_context)
        spec.pipeline_stage = existing_spec.pipeline_stage
        spec.obsidian_source = existing_spec.obsidian_source
        spec.epistemic_scores = dict(existing_spec.epistemic_scores)
        spec.interrogation_turns = existing_spec.interrogation_turns
    return spec


def _milestones_by_slice_id(milestones: list[InitiativeMilestone]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for milestone in milestones:
        for slice_id in milestone.slice_ids:
            mapping.setdefault(slice_id, milestone.title or milestone.milestone_id)
    return mapping


def _project_status(
    slice_status: str,
    existing_project: CampaignProject | None,
) -> str:
    if existing_project is not None:
        return existing_project.status
    return _SLICE_STATUS_TO_PROJECT_STATUS.get(
        str(slice_status or "").strip().lower(),
        CampaignProjectStatus.PENDING.value,
    )


def _review_gate(
    existing_project: CampaignProject | None,
    *,
    review_model: str,
    required: bool,
) -> CampaignReviewGate:
    if existing_project is None or existing_project.review is None:
        return CampaignReviewGate(required=required, review_model=review_model)
    gate = CampaignReviewGate.from_dict(existing_project.review.to_dict())
    gate.required = required
    gate.review_model = review_model
    return gate


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen[text] = None
    return list(seen)


__all__ = ["manifest_from_initiative", "sync_campaign_manifest_for_initiative"]
