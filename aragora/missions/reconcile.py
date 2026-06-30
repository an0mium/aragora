"""Preserve-first reconciliation primitives for native missions.

This module is deliberately side-effect-light: it classifies repo artifacts and
returns authorizations. Callers still need to perform fresh helper checks before
any destructive cleanup or exact-head merge.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .state import Feature, MissionState, Status


class ArtifactCategory(str, Enum):
    MERGED = "merged"
    OPEN_PR = "open-pr"
    VALUABLE_UNMERGED = "valuable-unmerged"
    DUPLICATE = "duplicate"
    SUPERSEDED = "superseded"
    UNSAFE_DIRTY = "unsafe-dirty"
    NEEDS_HUMAN = "needs-human"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorkArtifact:
    artifact_id: str
    kind: str
    clean: bool | None = None
    already_merged: bool = False
    open_pr: bool = False
    owner_active: bool = False
    unique_commits: bool = False
    represented_elsewhere: bool = False
    superseded: bool = False
    gone: bool = False
    tier: int | None = None
    head_sha: str | None = None
    checks_green: bool = False
    quorum_satisfied: bool = False
    evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkArtifact:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ClassifiedArtifact:
    artifact_id: str
    kind: str
    category: ArtifactCategory
    reason: str
    evidence: list[str] = field(default_factory=list)
    head_sha: str | None = None
    tier: int | None = None


@dataclass(frozen=True)
class ReconcileReport:
    mode: str
    items: list[ClassifiedArtifact]
    authorized_cleanup: list[ClassifiedArtifact] = field(default_factory=list)
    authorized_auto_drain: list[ClassifiedArtifact] = field(default_factory=list)
    parked: list[ClassifiedArtifact] = field(default_factory=list)

    @property
    def unresolved_count(self) -> int:
        return len(self.parked)

    def to_dict(self) -> dict[str, Any]:
        def dump(items: list[ClassifiedArtifact]) -> list[dict[str, Any]]:
            return [
                {
                    **asdict(item),
                    "category": item.category.value,
                }
                for item in items
            ]

        return {
            "mode": self.mode,
            "items": dump(self.items),
            "authorized_cleanup": dump(self.authorized_cleanup),
            "authorized_auto_drain": dump(self.authorized_auto_drain),
            "parked": dump(self.parked),
            "unresolved_count": self.unresolved_count,
            "mutations_executed": False,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class ReconcileMode(str, Enum):
    REPORT = "report"
    SAFE_CLEAN = "safe-clean"
    AUTO_DRAIN = "auto-drain"

    def run(self, artifacts: list[WorkArtifact]) -> ReconcileReport:
        items = [classify_artifact(artifact) for artifact in artifacts]
        authorized_cleanup: list[ClassifiedArtifact] = []
        authorized_auto_drain: list[ClassifiedArtifact] = []

        if self in {ReconcileMode.SAFE_CLEAN, ReconcileMode.AUTO_DRAIN}:
            authorized_cleanup = [
                item for item in items if item.category == ArtifactCategory.MERGED
            ]

        if self == ReconcileMode.AUTO_DRAIN:
            by_id = {artifact.artifact_id: artifact for artifact in artifacts}
            authorized_auto_drain = [
                item for item in items if _is_auto_drain_candidate(by_id[item.artifact_id], item)
            ]

        authorized_ids = {
            item.artifact_id for item in [*authorized_cleanup, *authorized_auto_drain]
        }
        terminal_without_action = {ArtifactCategory.DUPLICATE, ArtifactCategory.SUPERSEDED}
        parked = [
            item
            for item in items
            if item.artifact_id not in authorized_ids
            and item.category not in {ArtifactCategory.MERGED, *terminal_without_action}
        ]
        return ReconcileReport(
            mode=self.value,
            items=items,
            authorized_cleanup=authorized_cleanup,
            authorized_auto_drain=authorized_auto_drain,
            parked=parked,
        )


def classify_artifact(artifact: WorkArtifact) -> ClassifiedArtifact:
    """Classify one artifact with preserve-first ordering."""
    evidence = list(artifact.evidence)
    if artifact.clean is False:
        return _classified(
            artifact, ArtifactCategory.UNSAFE_DIRTY, "artifact has local dirt", evidence
        )
    if artifact.owner_active:
        return _classified(
            artifact, ArtifactCategory.NEEDS_HUMAN, "artifact has an active owner", evidence
        )
    if artifact.open_pr:
        return _classified(artifact, ArtifactCategory.OPEN_PR, "artifact has an open PR", evidence)
    if artifact.already_merged or artifact.gone:
        if artifact.clean is True:
            return _classified(
                artifact, ArtifactCategory.MERGED, "clean artifact is already merged/gone", evidence
            )
        return _classified(
            artifact,
            ArtifactCategory.UNKNOWN,
            "merged/gone artifact lacks fresh clean proof",
            evidence,
        )
    if artifact.superseded:
        return _classified(
            artifact, ArtifactCategory.SUPERSEDED, "artifact is superseded", evidence
        )
    if artifact.represented_elsewhere:
        return _classified(
            artifact, ArtifactCategory.DUPLICATE, "value is represented elsewhere", evidence
        )
    if artifact.unique_commits:
        return _classified(
            artifact,
            ArtifactCategory.VALUABLE_UNMERGED,
            "artifact has unique unmerged commits",
            evidence,
        )
    if artifact.tier is not None and artifact.tier >= 3:
        return _classified(
            artifact,
            ArtifactCategory.NEEDS_HUMAN,
            f"tier-{artifact.tier} artifact needs operator settlement",
            evidence,
        )
    return _classified(artifact, ArtifactCategory.UNKNOWN, "no terminal proof", evidence)


def _classified(
    artifact: WorkArtifact,
    category: ArtifactCategory,
    reason: str,
    evidence: list[str],
) -> ClassifiedArtifact:
    return ClassifiedArtifact(
        artifact_id=artifact.artifact_id,
        kind=artifact.kind,
        category=category,
        reason=reason,
        evidence=evidence,
        head_sha=artifact.head_sha,
        tier=artifact.tier,
    )


def _is_auto_drain_candidate(artifact: WorkArtifact, item: ClassifiedArtifact) -> bool:
    return (
        item.category == ArtifactCategory.OPEN_PR
        and artifact.tier is not None
        and artifact.tier <= 2
        and bool(artifact.head_sha)
        and artifact.checks_green
        and artifact.quorum_satisfied
        and not artifact.owner_active
    )


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class AdmissionPolicy:
    max_unresolved: int = 0
    allowed_goal_keywords: tuple[str, ...] = (
        "cleanup",
        "clean up",
        "reconcile",
        "drain",
        "settle",
        "evidence",
        "repair",
        "merge",
    )

    def evaluate(self, goal: str, report: ReconcileReport) -> AdmissionDecision:
        if report.unresolved_count <= self.max_unresolved:
            return AdmissionDecision(True, "unresolved backlog is within admission limit")
        goal_lc = goal.lower()
        if _goal_targets_backlog(goal_lc, self.allowed_goal_keywords):
            return AdmissionDecision(True, "mission directly targets backlog drain/repair")
        return AdmissionDecision(
            False,
            (
                f"unresolved backlog has {report.unresolved_count} parked artifacts; "
                "new producer work is blocked until cleanup, drain, evidence, settlement, or repair"
            ),
        )


def _goal_targets_backlog(goal_lc: str, allowed_goal_keywords: tuple[str, ...]) -> bool:
    action_patterns = {
        "cleanup": r"\b(cleanup|clean\s+up)\b",
        "clean up": r"\b(cleanup|clean\s+up)\b",
        "reconcile": r"\breconcile\b",
        "drain": r"\bdrain\b",
        "settle": r"\bsettle(?:ment)?\b",
        "evidence": r"\bevidence\b",
        "repair": r"\b(repair|fix)\b",
        "merge": r"\bmerge\b",
    }
    patterns: list[str] = []
    for keyword in allowed_goal_keywords:
        pattern = action_patterns.get(keyword)
        if pattern:
            patterns.append(pattern)
    if not patterns:
        return False

    action_terms = "|".join(f"(?:{pattern})" for pattern in patterns)
    backlog_nouns = (
        r"(?:backlog|queue|queued\s+work|worktree|worktrees|artifact|artifacts|"
        r"pr\s*#?\d+|pr|prs|pull\s+request|pull\s+requests|branch|branches|ci|check|checks)"
    )
    return bool(
        re.search(rf"(?:{action_terms}).{{0,120}}\b{backlog_nouns}\b", goal_lc)
        or re.search(rf"\b{backlog_nouns}\b.{{0,120}}(?:{action_terms})", goal_lc)
    )


def inject_validation_features(
    state: MissionState,
    *,
    milestone: str,
    validation_kinds: tuple[str, ...] = ("tests", "scrutiny"),
) -> list[Feature]:
    """Insert validator features for a completed milestone if not already present."""
    milestone_features = [f for f in state.features if f.milestone == milestone]
    if not milestone_features:
        return []
    non_validators = [
        f for f in milestone_features if f.metadata.get("validation_for") != milestone
    ]
    if not non_validators or not all(f.status == Status.COMPLETED for f in non_validators):
        return []

    preconditions = [f"feature:{f.id}" for f in non_validators]
    fulfills = sorted({assertion for f in non_validators for assertion in f.fulfills})
    path_values: set[str] = set()
    for f in non_validators:
        raw_paths = f.metadata.get("paths") or []
        raw_iterable: Iterable[Any]
        if isinstance(raw_paths, str):
            raw_iterable = [raw_paths]
        elif isinstance(raw_paths, (list, tuple, set)):
            raw_iterable = raw_paths
        else:
            raw_iterable = []
        path_values.update(str(path).strip() for path in raw_iterable if str(path).strip())
    paths = sorted(path_values)
    existing_ids = {f.id for f in state.features}
    injected: list[Feature] = []
    for kind in validation_kinds:
        feature_id = f"validate-{_slug(milestone)}-{_slug(kind)}"
        if feature_id in existing_ids:
            continue
        feature = Feature(
            id=feature_id,
            description=f"Validate milestone {milestone} with {kind}",
            milestone=milestone,
            skill="validator",
            preconditions=list(preconditions),
            fulfills=list(fulfills),
            metadata={
                "validation_for": milestone,
                "validation_kind": kind,
                "validates": [f.id for f in non_validators],
                "paths": paths,
            },
        )
        state.insert_feature(feature)
        injected.append(feature)
    return injected


def apply_validation_result(
    state: MissionState,
    validator_feature_id: str,
    *,
    passed: bool,
    reason: str,
    ledger_path: str | Path | None = None,
) -> None:
    """Apply a validator outcome and reopen parent work on failure."""
    validator = state.get(validator_feature_id)
    if passed:
        validator.status = Status.COMPLETED
        return

    ledger = None
    if ledger_path is not None:
        from .ledger import Ledger

        ledger = Ledger(ledger_path)

    state.mark_blocked(validator_feature_id, reason or "validation failed")
    validates = validator.metadata.get("validates", [])
    if not isinstance(validates, list):
        validates = []
    for parent_id in validates:
        try:
            parent = state.get(str(parent_id))
        except KeyError:
            continue
        parent.status = Status.PENDING
        parent.metadata["validation_reopened_by"] = validator_feature_id
        parent.metadata["validation_reopened_reason"] = reason or "validation failed"
        if ledger is not None:
            parent.metadata["validation_reopened_ledger_done_invalidated"] = ledger.invalidate_done(
                parent.id
            )
        note = f"validation {validator_feature_id} failed"
        if reason:
            note = f"{note}: {reason}"
        if note not in parent.notes:
            parent.notes = (parent.notes + "\n" if parent.notes else "") + note


def write_operator_receipt(
    receipt_dir: str | Path,
    *,
    feature_id: str,
    blocker: str,
    evidence: list[str],
    next_action: str,
    human_required: bool,
) -> Path:
    """Append one structured operator escalation receipt."""
    root = Path(receipt_dir)
    root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    payload = {
        "receipt_id": f"operator-{uuid.uuid4().hex}",
        "created_at": created_at,
        "feature_id": feature_id,
        "blocker": blocker,
        "evidence_checked": list(evidence),
        "next_action": next_action,
        "human_required": human_required,
    }
    path = (
        root
        / f"{created_at.replace(':', '').replace('+', 'Z')}-{_slug(feature_id)}-{uuid.uuid4().hex[:8]}.json"
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"
