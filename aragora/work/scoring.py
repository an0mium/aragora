"""Scoring and routing for the read-only Aragora work board."""

from __future__ import annotations

from datetime import UTC, datetime

from aragora.work.models import WorkItem, WorkRecommendation, WorkScore

RECOMMENDATION_CLASSES = {
    "ready",
    "needs-polish",
    "blocked",
    "duplicate",
    "stale",
    "review-only",
    "human-gated",
}

_TERMINAL_STATUSES = {
    "closed",
    "completed",
    "done",
    "failed",
    "cancelled",
    "canceled",
    "merged",
    "already_satisfied",
    "published",
}

_SWARM_READY_FIELDS = (
    ("objective", "objective missing"),
    ("context", "context missing"),
    ("acceptance_criteria", "acceptance criteria missing"),
    ("mutation_boundary", "mutation boundary missing"),
    ("validation", "tests/validation missing"),
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def stale_factor(updated_at: str | None, *, now: datetime | None = None) -> float:
    """Return freshness score where 1.0 is fresh and 0.0 is very stale."""
    parsed = _parse_time(updated_at)
    if parsed is None:
        return 0.45
    current = now or datetime.now(UTC)
    age_days = max(0.0, (current - parsed).total_seconds() / 86400.0)
    if age_days <= 1:
        return 1.0
    if age_days <= 7:
        return 0.85
    if age_days <= 14:
        return 0.65
    if age_days <= 30:
        return 0.35
    return 0.1


def is_current_status(status: str) -> bool:
    return status.lower() not in _TERMINAL_STATUSES


def _metadata_truthy(item: WorkItem, key: str) -> bool:
    value = item.metadata.get(key)
    if value:
        return True
    nested = item.metadata.get("metadata")
    if isinstance(nested, dict):
        nested_value = nested.get(key)
        if nested_value:
            return True
    return False


def _swarm_ready_blockers(item: WorkItem) -> list[str]:
    blockers: list[str] = []
    for key, message in _SWARM_READY_FIELDS:
        if not _metadata_truthy(item, key):
            blockers.append(message)
    if not (item.owner or item.branch):
        blockers.append("owner missing")
    dependencies_declared = bool(item.dependencies) or _metadata_truthy(
        item, "dependencies_declared"
    )
    if not dependencies_declared:
        blockers.append("dependency clarity missing")
    return blockers


def score_work_item(item: WorkItem, *, now: datetime | None = None) -> WorkScore:
    status = item.status.lower()
    title = item.title.lower()
    source = item.source
    rationale: list[str] = []

    readiness = 0.45
    if source == "github_pr":
        if item.metadata.get("is_draft"):
            readiness = 0.25
            rationale.append("draft PR is not ready for settlement")
        elif item.metadata.get("review_decision") == "REVIEW_REQUIRED":
            readiness = 0.62
            rationale.append("PR is ready for focused review/settlement")
        else:
            readiness = 0.75
            rationale.append("PR is non-draft and has no draft gate")
    elif source == "automation_outbox":
        readiness = 0.72
        rationale.append("automation handoff is pending operator/publisher action")
    elif source == "broker_run":
        readiness = 0.55 if is_current_status(status) else 0.2
        rationale.append(
            "broker run is active" if is_current_status(status) else "broker run is historical"
        )
    elif source in {"bead", "convoy"}:
        readiness = 0.58 if is_current_status(status) else 0.18
        rationale.append(
            "bead/convoy lifecycle is non-terminal"
            if is_current_status(status)
            else "bead/convoy is terminal"
        )
    elif source == "mission_file":
        readiness = 0.4
        rationale.append("mission file is context, not an active claim")

    impact = {
        "github_pr": 0.72,
        "automation_outbox": 0.68,
        "broker_run": 0.6,
        "bead": 0.5,
        "convoy": 0.56,
        "mission_file": 0.42,
        "automation_receipt": 0.25,
    }.get(source, 0.4)
    if any(token in title for token in ("block", "fix", "repair", "health", "proof", "queue")):
        impact = min(1.0, impact + 0.12)
        rationale.append("title signals repair/health/proof impact")

    risk = 0.72
    files = item.metadata.get("files") or []
    if item.metadata.get("tier") in {3, 4, "3", "4"}:
        risk -= 0.2
        rationale.append("higher-tier semantic-risk surface")
    if any(str(path).startswith((".github/", "scripts/")) for path in files):
        risk -= 0.12
    if any(str(path).startswith("docs/") for path in files):
        risk += 0.08
    if source == "mission_file":
        risk += 0.08

    parallel_safety = 0.65
    if len(files) > 8:
        parallel_safety -= 0.2
    if source in {"automation_receipt", "mission_file"}:
        parallel_safety += 0.15
    if item.dependencies:
        parallel_safety -= 0.12

    staleness = stale_factor(item.updated_at or item.created_at, now=now)
    owner_clarity = 0.85 if item.owner or item.branch else 0.35
    if item.owner or item.branch:
        rationale.append("owner/branch is explicit")

    tests = [path for path in files if str(path).startswith("tests/")]
    code = [path for path in files if str(path).startswith("aragora/")]
    if code and tests:
        test_obligation = 0.9
    elif code:
        test_obligation = 0.35
        rationale.append("code surface lacks visible tests in metadata")
    else:
        test_obligation = 0.75

    dependency_clarity = (
        0.85 if item.dependencies or _metadata_truthy(item, "dependencies_declared") else 0.45
    )
    if item.dependencies:
        rationale.append("dependencies are explicit")

    bead_quality = 0.5
    if source in {"bead", "convoy"}:
        blockers = _swarm_ready_blockers(item)
        present = len(_SWARM_READY_FIELDS) + 2 - len(blockers)
        total_fields = len(_SWARM_READY_FIELDS) + 2
        bead_quality = 0.08 + (0.82 * (present / total_fields))
        if item.title and len(item.title) > 8:
            bead_quality += 0.08
        if item.dependencies:
            bead_quality += 0.04
        if blockers:
            bead_quality -= 0.08 * len(blockers)
            rationale.append("bead needs polish: " + ", ".join(blockers[:3]))
    elif source == "github_pr":
        bead_quality = 0.58
    elif source == "automation_outbox":
        bead_quality = 0.62

    dimensions: dict[str, float] = {
        "readiness": _clamp(readiness),
        "impact": _clamp(impact),
        "risk": _clamp(risk),
        "parallel_safety": _clamp(parallel_safety),
        "staleness": _clamp(staleness),
        "owner_clarity": _clamp(owner_clarity),
        "test_obligation": _clamp(test_obligation),
        "dependency_clarity": _clamp(dependency_clarity),
        "bead_quality": _clamp(bead_quality),
    }
    weights = {
        "readiness": 0.22,
        "impact": 0.16,
        "risk": 0.12,
        "parallel_safety": 0.1,
        "staleness": 0.1,
        "owner_clarity": 0.08,
        "test_obligation": 0.08,
        "dependency_clarity": 0.07,
        "bead_quality": 0.07,
    }
    total = sum(dimensions[key] * weights[key] for key in weights)
    return WorkScore(total=_clamp(total), rationale=rationale[:6], **dimensions)


def classify_work_item(item: WorkItem, *, score: WorkScore | None = None) -> tuple[str, list[str]]:
    """Classify work into the stable flywheel recommendation vocabulary."""
    status = item.status.lower()
    blockers: list[str] = []
    score = score or item.score or score_work_item(item)

    if status in {"already_satisfied", "published"}:
        return "duplicate", ["receipt says work was already satisfied or published"]
    if not is_current_status(status) or item.scope == "historical":
        return "stale", ["terminal or historical work item"]
    if item.metadata.get("human_gate") or item.metadata.get("tier") in {3, 4, "3", "4"}:
        return "human-gated", ["human-gated risk surface"]
    if item.metadata.get("blocked") or item.metadata.get("blockers"):
        raw = item.metadata.get("blockers")
        if isinstance(raw, list):
            blockers.extend(str(entry) for entry in raw)
        elif raw:
            blockers.append(str(raw))
        blockers.append("explicit blocker metadata")
        return "blocked", blockers
    if item.source == "github_pr" and item.metadata.get("is_draft"):
        return "review-only", ["draft PR"]
    if item.source in {"bead", "convoy", "automation_outbox"}:
        blockers = _swarm_ready_blockers(item)
        if blockers:
            return "needs-polish", blockers
    if score.bead_quality < 0.45 or score.dependency_clarity < 0.4:
        return "needs-polish", ["weak bead quality or dependency clarity"]
    return "ready", []


def recommended_action(item: WorkItem) -> tuple[str, str, list[str]]:
    blockers: list[str] = []
    if item.source == "github_pr":
        if item.metadata.get("is_draft"):
            blockers.append("draft PR")
            return "keep_draft_until_evidence_ready", "hold", blockers
        if item.metadata.get("review_decision") == "REVIEW_REQUIRED":
            blockers.append("human/model review required")
        return "review_and_settle_when_policy_clean", "high", blockers
    if item.source == "automation_outbox":
        return "publish_or_reconcile_handoff", "high", blockers
    if item.source == "broker_run":
        return "inspect_broker_run", "medium", blockers
    if item.source in {"bead", "convoy"}:
        return "clarify_or_claim_atomic_work", "medium", blockers
    if item.source == "mission_file":
        return "use_as_context_for_decomposition", "low", blockers
    return "inspect_work_item", "medium", blockers


def build_recommendations(items: list[WorkItem]) -> list[WorkRecommendation]:
    scored: list[WorkItem] = []
    for item in items:
        item.score = score_work_item(item)
        scored.append(item)
    scored.sort(
        key=lambda it: (it.score.total if it.score else 0.0, it.updated_at or ""), reverse=True
    )

    recommendations: list[WorkRecommendation] = []
    for rank, item in enumerate(scored, start=1):
        action, priority, blockers = recommended_action(item)
        score = item.score or WorkScore()
        classification, class_blockers = classify_work_item(item, score=score)
        blockers = [*blockers, *class_blockers]
        if classification in {"human-gated", "blocked"}:
            priority = "hold"
        elif classification == "needs-polish" and priority == "high":
            priority = "medium"
        recommendations.append(
            WorkRecommendation(
                rank=rank,
                item_id=item.id,
                classification=classification,
                action=action,
                priority=priority,
                blockers=blockers,
                rationale=score.rationale,
                score=score,
                item=item,
            )
        )
    return recommendations
