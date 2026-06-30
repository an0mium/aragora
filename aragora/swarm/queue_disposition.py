"""Value-preserving queue disposition rules.

The manifest produced from these helpers is a pre-cleanup artifact: it records
why each queue item should be harvested, human-routed, parked, or only closed /
deleted after a recovery manifest exists. It is intentionally conservative.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aragora.swarm.pr_value import (
    CLASS_INFRA,
    CLASS_MAINTENANCE,
    CLASS_PRODUCT,
    CLASS_UNKNOWN,
    classify_value_record,
    label_names,
    parse_iso_datetime,
)

DISPOSITION_HARVEST_NOW = "harvest_now"
DISPOSITION_HUMAN_PACKET = "human_packet"
DISPOSITION_PARK_PRESERVE = "park_preserve"
DISPOSITION_CLOSE_OR_DELETE = "close_or_delete_after_manifest"
DISPOSITIONS = (
    DISPOSITION_HARVEST_NOW,
    DISPOSITION_HUMAN_PACKET,
    DISPOSITION_PARK_PRESERVE,
    DISPOSITION_CLOSE_OR_DELETE,
)

EVALUATION_LOW = "low"
EVALUATION_MEDIUM = "medium"
EVALUATION_HIGH = "high"

PARKED_PREFIXES = ("[AGT-", "[DIC-")
HIGH_VALUE_TERMS = (
    "odr",
    "crux",
    "receipt",
    "api",
    "sdk",
    "routing",
    "goals",
    "gauntlet",
    "server",
    "inbox",
    "decision",
)
HIGH_RISK_TERMS = (
    "crypto",
    "signing",
    "rbac",
    "security",
    "tier4",
    "workflow",
    "deploy",
    "public api",
)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any) -> bool:
    return bool(value)


def _title(record: dict[str, Any]) -> str:
    return str(record.get("title") or "")


def _branch(record: dict[str, Any]) -> str:
    return str(record.get("headRefName") or record.get("branch") or "")


def _head(record: dict[str, Any]) -> str:
    return str(record.get("headRefOid") or record.get("head") or record.get("head_sha") or "")


def _is_stale(record: dict[str, Any], *, now: datetime, stale_days: int) -> bool:
    created = parse_iso_datetime(record.get("createdAt") or record.get("created_at"))
    if created is None:
        return False
    return (now - created).days >= stale_days


def estimate_evaluation_cost(record: dict[str, Any]) -> str:
    """Estimate the cheapness of deciding whether an item has value."""
    changed_files = _coerce_int(record.get("changedFiles") or record.get("changed_files"))
    additions = _coerce_int(record.get("additions"))
    deletions = _coerce_int(record.get("deletions"))
    total_delta = additions + deletions
    if changed_files <= 3 and total_delta <= 500:
        return EVALUATION_LOW
    if changed_files <= 20 and total_delta <= 2000:
        return EVALUATION_MEDIUM
    return EVALUATION_HIGH


def _is_parked_epic(record: dict[str, Any]) -> bool:
    title = _title(record).strip().upper()
    labels = label_names(record)
    return title.startswith(PARKED_PREFIXES) or "vision-layer" in labels


def _has_high_value_signal(record: dict[str, Any], value_class: str) -> bool:
    if value_class == CLASS_PRODUCT or _is_parked_epic(record):
        return True
    title = _title(record).lower()
    return any(term in title for term in HIGH_VALUE_TERMS)


def _has_high_risk_signal(
    record: dict[str, Any], merge_packet_entry: dict[str, Any] | None
) -> bool:
    title = _title(record).lower()
    if any(term in title for term in HIGH_RISK_TERMS):
        return True
    if not merge_packet_entry:
        return False
    tier = _coerce_int(merge_packet_entry.get("tier"), default=-1)
    return (
        tier >= 3
        or _coerce_bool(merge_packet_entry.get("requires_human_risk_settlement"))
        or _coerce_bool(merge_packet_entry.get("requires_human_preapproval"))
    )


def _merge_packet_evidence(entry: dict[str, Any] | None) -> list[str]:
    if not entry:
        return []
    evidence = []
    for key in (
        "tier",
        "tier_name",
        "status",
        "verdict",
        "checks_summary",
        "admin_squash_allowed",
        "requires_human_risk_settlement",
        "requires_human_preapproval",
        "unresolved_dissent",
    ):
        if key in entry:
            evidence.append(f"merge_packet.{key}={entry[key]}")
    return evidence


def _base_item(
    *,
    item_type: str,
    item_id: str,
    branch: str,
    head_sha: str,
    open_pr: int | None,
    value_class: str,
    evaluation_cost: str,
    evidence: list[str],
    disposition: str,
    next_action: str,
    operator_required: bool,
) -> dict[str, Any]:
    return {
        "item_type": item_type,
        "id": item_id,
        "branch": branch,
        "head_sha": head_sha,
        "open_pr": open_pr,
        "value_class": value_class,
        "evaluation_cost": evaluation_cost,
        "evidence": evidence,
        "disposition": disposition,
        "next_action": next_action,
        "operator_required": operator_required,
    }


def classify_pr_disposition(
    record: dict[str, Any],
    *,
    merge_packet_entry: dict[str, Any] | None = None,
    now: datetime | None = None,
    stale_days: int = 14,
) -> dict[str, Any]:
    """Classify one open PR into a value-preserving disposition."""
    if now is None:
        now = datetime.now(timezone.utc)
    number = _coerce_int(record.get("number"))
    value_class = classify_value_record(record)
    cost = estimate_evaluation_cost(record)
    title = _title(record)
    branch = _branch(record)
    head_sha = _head(record)
    is_draft = _coerce_bool(record.get("isDraft"))
    mergeable = str(record.get("mergeable") or "").upper()
    stale = _is_stale(record, now=now, stale_days=stale_days)
    high_value = _has_high_value_signal(record, value_class)
    high_risk = _has_high_risk_signal(record, merge_packet_entry)
    unresolved_dissent = _coerce_bool((merge_packet_entry or {}).get("unresolved_dissent"))
    low_expected_value = value_class in {CLASS_MAINTENANCE, CLASS_INFRA} and not high_value

    evidence = [
        f"title={title}",
        f"value_class={value_class}",
        f"evaluation_cost={cost}",
        f"isDraft={is_draft}",
        f"mergeable={mergeable or 'unknown'}",
    ]
    if stale:
        evidence.append(f"stale_days>={stale_days}")
    if high_value:
        evidence.append("high_value_signal=true")
    if high_risk:
        evidence.append("high_risk_or_human_settlement_signal=true")
    if unresolved_dissent:
        evidence.append("model_dissent_present_but_not_value_proof")
    evidence.extend(_merge_packet_evidence(merge_packet_entry))

    if high_risk and not is_draft:
        return _base_item(
            item_type="pr",
            item_id=str(number),
            branch=branch,
            head_sha=head_sha,
            open_pr=number,
            value_class=value_class,
            evaluation_cost=cost,
            evidence=evidence,
            disposition=DISPOSITION_HUMAN_PACKET,
            next_action="prepare exact-head human-risk packet; do not close as churn",
            operator_required=True,
        )

    if high_value and unresolved_dissent:
        return _base_item(
            item_type="pr",
            item_id=str(number),
            branch=branch,
            head_sha=head_sha,
            open_pr=number,
            value_class=value_class,
            evaluation_cost=cost,
            evidence=evidence,
            disposition=DISPOSITION_PARK_PRESERVE,
            next_action="preserve; repair or adjudicate dissent, never close solely for dissent",
            operator_required=False,
        )

    if high_value and not is_draft and mergeable != "CONFLICTING":
        return _base_item(
            item_type="pr",
            item_id=str(number),
            branch=branch,
            head_sha=head_sha,
            open_pr=number,
            value_class=value_class,
            evaluation_cost=cost,
            evidence=evidence,
            disposition=DISPOSITION_HARVEST_NOW,
            next_action="run one exact-head evidence/merge-packet attempt or settle if already green",
            operator_required=False,
        )

    if _is_parked_epic(record) or high_value or (is_draft and high_risk):
        return _base_item(
            item_type="pr",
            item_id=str(number),
            branch=branch,
            head_sha=head_sha,
            open_pr=number,
            value_class=value_class,
            evaluation_cost=cost,
            evidence=evidence,
            disposition=DISPOSITION_PARK_PRESERVE,
            next_action="preserve parked/high-risk value until a cheap content pass proves duplicate",
            operator_required=high_risk,
        )

    if low_expected_value and (is_draft or stale or mergeable == "CONFLICTING"):
        return _base_item(
            item_type="pr",
            item_id=str(number),
            branch=branch,
            head_sha=head_sha,
            open_pr=number,
            value_class=value_class,
            evaluation_cost=cost,
            evidence=evidence,
            disposition=DISPOSITION_CLOSE_OR_DELETE,
            next_action=(
                "only close after live owner/steering, diffstat, and main-equivalence checks; "
                "write reversible rationale"
            ),
            operator_required=True,
        )

    return _base_item(
        item_type="pr",
        item_id=str(number),
        branch=branch,
        head_sha=head_sha,
        open_pr=number,
        value_class=value_class or CLASS_UNKNOWN,
        evaluation_cost=cost,
        evidence=evidence,
        disposition=DISPOSITION_PARK_PRESERVE,
        next_action="preserve pending cheap content pass; value is not disproven",
        operator_required=False,
    )


def classify_inventory_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Classify one ``codex_worktree_value_inventory`` candidate."""
    classification = str(candidate.get("classification") or "unknown")
    raw_git = candidate.get("git")
    git: dict[str, Any] = raw_git if isinstance(raw_git, dict) else {}
    raw_links = candidate.get("links")
    links: dict[str, Any] = raw_links if isinstance(raw_links, dict) else {}
    branch = str(git.get("branch") or candidate.get("branch") or "")
    head_sha = str(git.get("head") or candidate.get("head") or "")
    open_prs = links.get("open_prs") if isinstance(links.get("open_prs"), list) else []
    open_pr = _coerce_int(open_prs[0], default=0) if open_prs else None
    if open_pr == 0:
        open_pr = None
    path = str(candidate.get("path") or "")
    item_id = str(candidate.get("candidate_id") or path or branch or head_sha)
    evidence = [
        f"inventory.classification={classification}",
        f"inventory.decision={candidate.get('decision')}",
    ]
    evidence.extend(str(item) for item in candidate.get("proof") or [])

    if classification == "unique_unharvested":
        disposition = DISPOSITION_HARVEST_NOW
        next_action = "preserve and represent unique work on an open PR or durable branch"
        operator_required = False
    elif classification in {"active_or_dirty", "open_pr_or_outbox", "receipt_protected"}:
        disposition = DISPOSITION_PARK_PRESERVE
        next_action = "preserve; route to owner, open PR, outbox, or receipt before cleanup"
        operator_required = False
    elif classification in {"patch_equivalent_or_merged", "no_git_cache_residue"}:
        disposition = DISPOSITION_CLOSE_OR_DELETE
        next_action = "delete only after fresh safe_worktree_cleanup inspect and SHA/path manifest"
        operator_required = True
    else:
        disposition = DISPOSITION_PARK_PRESERVE
        next_action = "preserve pending cheap inventory/content pass"
        operator_required = False

    return _base_item(
        item_type="worktree",
        item_id=item_id,
        branch=branch,
        head_sha=head_sha,
        open_pr=open_pr,
        value_class=CLASS_UNKNOWN,
        evaluation_cost=EVALUATION_MEDIUM,
        evidence=evidence,
        disposition=disposition,
        next_action=next_action,
        operator_required=operator_required,
    )


def build_manifest(
    *,
    prs: list[dict[str, Any]],
    merge_packet_entries: dict[int, dict[str, Any]] | None = None,
    inventory_candidates: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
    stale_days: int = 14,
    annotations: list[str] | None = None,
) -> dict[str, Any]:
    """Build the complete disposition manifest payload."""
    if now is None:
        now = datetime.now(timezone.utc)
    merge_packet_entries = merge_packet_entries or {}
    items: list[dict[str, Any]] = []
    for pr in prs:
        if not isinstance(pr, dict):
            continue
        number = _coerce_int(pr.get("number"))
        items.append(
            classify_pr_disposition(
                pr,
                merge_packet_entry=merge_packet_entries.get(number),
                now=now,
                stale_days=stale_days,
            )
        )
    for candidate in inventory_candidates or []:
        if isinstance(candidate, dict):
            items.append(classify_inventory_candidate(candidate))

    by_disposition = {name: 0 for name in DISPOSITIONS}
    by_value_class: dict[str, int] = {}
    for item in items:
        by_disposition[item["disposition"]] = by_disposition.get(item["disposition"], 0) + 1
        by_value_class[item["value_class"]] = by_value_class.get(item["value_class"], 0) + 1

    return {
        "schema_version": "aragora.queue_disposition_manifest.v1",
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": "value_of_information_preserve_first",
        "annotations": list(annotations or []),
        "summary": {
            "total_items": len(items),
            "by_disposition": by_disposition,
            "by_value_class": by_value_class,
            "operator_required": sum(1 for item in items if item["operator_required"]),
        },
        "items": items,
    }
