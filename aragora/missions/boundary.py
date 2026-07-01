"""Zenith-style mission boundary decisions for native missions.

This module is pure at the controller layer: ``evaluate`` only classifies the
next boundary action. ``apply_boundary_decision`` performs the small state
mutation so tests and callers can reason about decisions before applying them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .reconcile import apply_validation_result, inject_validation_features
from .state import Feature, FeatureKind, MissionState, Status


class MissionBoundaryAction(str, Enum):
    """Actions available at a feature, validator, or milestone boundary."""

    CONTINUE = "continue"
    PATCH_PLAN = "patch_plan"
    RETRY = "retry"
    ADD_WORKER = "add_worker"
    ADD_VALIDATOR = "add_validator"
    SYNTHESIZE_SKILL = "synthesize_skill"
    RESET_STRATEGY = "reset_strategy"
    SEAL_MILESTONE = "seal_milestone"
    PARK = "park"
    STOP = "stop"


@dataclass(frozen=True)
class ContractCoverageError:
    """One contract-coverage invariant violation."""

    code: str
    assertion_id: str
    detail: str
    feature_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MissionBoundaryEvent:
    """The observable event that asks the controller for a next action."""

    kind: str
    feature_id: str = ""
    milestone: str = ""
    reason: str = ""
    terminal: bool = False
    risk_tier: int | None = None
    failed_assertions: list[str] = field(default_factory=list)
    discovered_skills: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MissionBoundaryDecision:
    """Pure boundary verdict returned by ``MissionBoundaryController``."""

    action: MissionBoundaryAction
    reason: str
    feature_id: str = ""
    milestone: str = ""
    assertion_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "feature_id": self.feature_id,
            "milestone": self.milestone,
            "assertion_ids": list(self.assertion_ids),
            "metadata": dict(self.metadata),
        }


class MissionBoundaryController:
    """Choose the next mission-control action at explicit boundaries."""

    def __init__(self, *, max_retries: int = 3, operator_tier: int = 3) -> None:
        self.max_retries = max(1, int(max_retries))
        self.operator_tier = max(1, int(operator_tier))

    def evaluate(self, state: MissionState, event: MissionBoundaryEvent) -> MissionBoundaryDecision:
        feature = _feature_or_none(state, event.feature_id)
        milestone = event.milestone or (feature.milestone if feature else "")
        reason = event.reason or "mission boundary reached"

        if event.risk_tier is not None and event.risk_tier >= self.operator_tier:
            return MissionBoundaryDecision(
                MissionBoundaryAction.PARK,
                f"tier-{event.risk_tier} boundary requires operator settlement: {reason}",
                feature_id=event.feature_id,
                milestone=milestone,
                assertion_ids=list(event.failed_assertions),
            )

        if event.terminal:
            return MissionBoundaryDecision(
                MissionBoundaryAction.PARK,
                f"terminal boundary requires operator receipt: {reason}",
                feature_id=event.feature_id,
                milestone=milestone,
                assertion_ids=list(event.failed_assertions),
            )

        if event.kind == "validation_failed":
            assertion_ids = list(event.failed_assertions)
            if not assertion_ids and feature is not None:
                assertion_ids = list(feature.fulfills)
            return MissionBoundaryDecision(
                MissionBoundaryAction.PATCH_PLAN,
                reason,
                feature_id=event.feature_id,
                milestone=milestone,
                assertion_ids=assertion_ids,
            )

        if event.kind == "worker_failed":
            next_retry = (feature.retry_count + 1) if feature is not None else 1
            if next_retry >= self.max_retries:
                return MissionBoundaryDecision(
                    MissionBoundaryAction.PARK,
                    f"retry budget exhausted after {next_retry} attempt(s): {reason}",
                    feature_id=event.feature_id,
                    milestone=milestone,
                    assertion_ids=list(feature.fulfills if feature else event.failed_assertions),
                    metadata={"retry_count": next_retry, "max_retries": self.max_retries},
                )
            return MissionBoundaryDecision(
                MissionBoundaryAction.RETRY,
                reason,
                feature_id=event.feature_id,
                milestone=milestone,
                assertion_ids=list(feature.fulfills if feature else event.failed_assertions),
                metadata={"retry_count": next_retry, "max_retries": self.max_retries},
            )

        if (
            event.kind == "feature_completed"
            and feature is not None
            and feature.kind == FeatureKind.WORK
            and _all_work_completed(state, feature.milestone)
            and not _has_validation_or_gate(state, feature.milestone)
        ):
            return MissionBoundaryDecision(
                MissionBoundaryAction.ADD_VALIDATOR,
                f"milestone {feature.milestone} work is complete; add layered validation",
                feature_id=feature.id,
                milestone=feature.milestone,
                assertion_ids=_milestone_assertions(state, feature.milestone),
            )

        return MissionBoundaryDecision(
            MissionBoundaryAction.CONTINUE,
            reason,
            feature_id=event.feature_id,
            milestone=milestone,
            assertion_ids=list(event.failed_assertions),
        )


def validate_contract_coverage(state: MissionState) -> list[ContractCoverageError]:
    """Validate native contract coverage.

    Every ``VAL-*`` assertion needs exactly one active work owner. Validators and
    gates may cover many assertions, but they do not count as implementation
    ownership.
    """
    if not state.contract:
        return []

    assertion_ids = [item.assertion_id for item in state.contract]
    known = set(assertion_ids)
    owners: dict[str, list[str]] = {assertion_id: [] for assertion_id in assertion_ids}
    errors: list[ContractCoverageError] = []

    for feature in state.features:
        for assertion_id in feature.fulfills:
            if assertion_id not in known:
                errors.append(
                    ContractCoverageError(
                        code="task_targets_unknown_assertion",
                        assertion_id=assertion_id,
                        detail=f"feature {feature.id} targets unknown assertion {assertion_id}",
                        feature_ids=[feature.id],
                    )
                )
                continue
            if feature.kind == FeatureKind.WORK and feature.status != Status.BLOCKED:
                owners[assertion_id].append(feature.id)

    for assertion_id, feature_ids in owners.items():
        if not feature_ids:
            errors.append(
                ContractCoverageError(
                    code="uncovered_assertion",
                    assertion_id=assertion_id,
                    detail=f"assertion {assertion_id} has no active work owner",
                )
            )
        elif len(feature_ids) > 1:
            errors.append(
                ContractCoverageError(
                    code="over_covered_assertion",
                    assertion_id=assertion_id,
                    detail=f"assertion {assertion_id} has multiple active work owners",
                    feature_ids=list(feature_ids),
                )
            )

    return errors


def layered_validation_kinds(state: MissionState, milestone: str) -> tuple[str, ...]:
    """Return layered validation kinds for a milestone."""
    kinds = ["automated", "review"]
    milestone_features = [
        feature
        for feature in state.features
        if feature.milestone == milestone and feature.kind == FeatureKind.WORK
    ]
    if any(_needs_fidelity_validation(feature) for feature in milestone_features):
        kinds.append("fidelity")
    return tuple(kinds)


def apply_boundary_decision(state: MissionState, decision: MissionBoundaryDecision) -> None:
    """Apply a previously evaluated boundary decision to mission state."""
    state.decision_trace.append(decision.to_dict())

    if decision.action == MissionBoundaryAction.RETRY:
        feature = state.get(decision.feature_id)
        feature.retry_count += 1
        feature.status = Status.PENDING
        return

    if decision.action == MissionBoundaryAction.PARK:
        if decision.feature_id:
            state.mark_blocked(decision.feature_id, decision.reason)
        return

    if decision.action == MissionBoundaryAction.ADD_VALIDATOR:
        milestone = decision.milestone
        if milestone:
            inject_validation_features(
                state,
                milestone=milestone,
                validation_kinds=layered_validation_kinds(state, milestone),
                include_gate=True,
            )
        return

    if decision.action == MissionBoundaryAction.PATCH_PLAN:
        _apply_patch_plan(state, decision)


def _apply_patch_plan(state: MissionState, decision: MissionBoundaryDecision) -> None:
    validator = state.get(decision.feature_id)
    apply_validation_result(
        state,
        validator_feature_id=validator.id,
        passed=False,
        reason=decision.reason,
    )
    assertions = list(decision.assertion_ids or validator.fulfills)
    parent_paths = _paths_from_validated_parents(state, validator)
    for assertion_id in assertions:
        feature_id = f"repair-{validator.id}-{_slug(assertion_id)}"
        if _feature_or_none(state, feature_id) is not None:
            continue
        state.insert_feature(
            Feature(
                id=feature_id,
                description=f"Repair {assertion_id}: {decision.reason}",
                milestone=validator.milestone,
                skill="worker",
                kind=FeatureKind.WORK,
                fulfills=[assertion_id],
                metadata={
                    "repair_for": validator.id,
                    "paths": list(parent_paths),
                },
            )
        )


def _needs_fidelity_validation(feature: Feature) -> bool:
    metadata = dict(feature.metadata or {})
    surface = str(metadata.get("surface", "") or "").strip().lower()
    if surface in {"ui", "visual", "benchmark", "external"}:
        return True

    raw_layers = metadata.get("validation_layers", [])
    if isinstance(raw_layers, str):
        layers: Iterable[Any] = [raw_layers]
    elif isinstance(raw_layers, Iterable):
        layers = raw_layers
    else:
        layers = []
    normalized_layers = {str(layer).strip().lower() for layer in layers if str(layer).strip()}
    if normalized_layers & {"fidelity", "visual", "user_testing", "benchmark", "external"}:
        return True

    expected = " ".join(str(item) for item in feature.expected_behavior).lower()
    return any(term in expected for term in ("ui", "visual", "browser", "benchmark", "external"))


def _paths_from_validated_parents(state: MissionState, validator: Feature) -> list[str]:
    paths: set[str] = set()
    validates = validator.metadata.get("validates", [])
    if not isinstance(validates, list):
        return []
    for parent_id in validates:
        parent = _feature_or_none(state, str(parent_id))
        if parent is None:
            continue
        paths.update(_coerce_strings(parent.metadata.get("paths")))
    return sorted(paths)


def _coerce_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = value
    else:
        values = []
    return [str(item).strip() for item in values if str(item).strip()]


def _feature_or_none(state: MissionState, feature_id: str) -> Feature | None:
    if not feature_id:
        return None
    try:
        return state.get(feature_id)
    except KeyError:
        return None


def _all_work_completed(state: MissionState, milestone: str) -> bool:
    work = [
        feature
        for feature in state.features
        if feature.milestone == milestone and feature.kind == FeatureKind.WORK
    ]
    return bool(work) and all(feature.status == Status.COMPLETED for feature in work)


def _has_validation_or_gate(state: MissionState, milestone: str) -> bool:
    return any(
        feature.milestone == milestone
        and (
            feature.kind in {FeatureKind.VALIDATE, FeatureKind.GATE}
            or feature.metadata.get("validation_for") == milestone
        )
        for feature in state.features
    )


def _milestone_assertions(state: MissionState, milestone: str) -> list[str]:
    return sorted(
        {
            assertion_id
            for feature in state.features
            if feature.milestone == milestone
            for assertion_id in feature.fulfills
            if assertion_id
        }
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "item"
