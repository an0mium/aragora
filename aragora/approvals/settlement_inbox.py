"""Settlement approval inbox packets for merge-risk decisions.

This module is deliberately read-only. It turns the existing review-queue
merge-packet into approval-inbox items when a PR is already at the human
settlement boundary. It does not post comments, statuses, evidence, or merges.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

UTC = timezone.utc

SETTLEMENT_READY_STATUSES = {
    "human_risk_settlement_required",
    "human_preapproval_required",
}


def _parse_ts(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _requested_at(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC).isoformat()
    except ValueError:
        return raw


def _reason_summary(entry: dict[str, Any]) -> list[str]:
    reasons = entry.get("reasons")
    if not isinstance(reasons, list):
        return []
    return [str(reason) for reason in reasons[:8]]


def _settlement_kind(entry: dict[str, Any]) -> str:
    if bool(entry.get("requires_human_preapproval")):
        return "tier4_human_preapproval"
    return "human_risk_settlement"


def _approval_item(entry: dict[str, Any], *, generated_at: str) -> dict[str, Any]:
    pr_number = int(entry.get("pr_number") or 0)
    head_sha = str(entry.get("head_sha") or "").strip()
    target_id = f"settlement-pr-{pr_number}-{head_sha[:12] or 'unknown'}"
    tier = entry.get("tier")
    title = str(entry.get("title") or f"PR #{pr_number}")
    settlement_kind = _settlement_kind(entry)
    reason_summary = _reason_summary(entry)
    description = (
        f"PR #{pr_number} is waiting for "
        f"{'Tier 4 human preapproval' if settlement_kind == 'tier4_human_preapproval' else 'human risk settlement'} "
        f"at exact head {head_sha[:12] or 'unknown'}."
    )
    if reason_summary:
        description = f"{description} Primary reason: {reason_summary[0]}"

    cli_reason = f"Settlement Inbox acceptance for PR #{pr_number} at exact head {head_sha[:12]}"
    approve_cli = [
        "python3",
        "-m",
        "aragora.cli.main",
        "review-queue",
        "record-settlement",
        str(pr_number),
        "--head-sha",
        head_sha,
        "--action",
        "approve",
        "--reason",
        cli_reason,
        "--post-github-status",
    ]
    reject_cli = [
        "python3",
        "-m",
        "aragora.cli.main",
        "review-queue",
        "record-settlement",
        str(pr_number),
        "--head-sha",
        head_sha,
        "--action",
        "request_changes",
        "--reason",
        f"Settlement Inbox rejection for PR #{pr_number} at exact head {head_sha[:12]}",
    ]

    return {
        "id": target_id,
        "kind": "settlement",
        "status": "pending",
        "title": f"Settlement approval: PR #{pr_number} T{tier} {title}",
        "description": description,
        "requested_at": _requested_at(generated_at),
        "requested_by": "review-queue merge-packet",
        "metadata": {
            "pr_number": pr_number,
            "pr_url": entry.get("url"),
            "title": title,
            "head_sha": head_sha,
            "tier": tier,
            "tier_name": entry.get("tier_name"),
            "settlement_kind": settlement_kind,
            "status": entry.get("status"),
            "verdict": entry.get("verdict"),
            "checks_summary": entry.get("checks_summary"),
            "requires_human_risk_settlement": bool(entry.get("requires_human_risk_settlement")),
            "requires_human_preapproval": bool(entry.get("requires_human_preapproval")),
            "counted_model_families": entry.get("counted_model_families") or [],
            "reviewer_signals": entry.get("reviewer_signals") or [],
            "dogfood_evidence": entry.get("dogfood_evidence") or [],
            "reasons": reason_summary,
            "settlement_creator_pin": entry.get("settlement_creator_pin") or {},
            "check_surfaces": entry.get("check_surfaces") or {},
        },
        "actions": {
            "approve": {
                "method": "POST",
                "path": f"/api/v1/settlement-inbox/{target_id}/approve",
                "body": {
                    "pr": pr_number,
                    "head_sha": head_sha,
                    "decision": "approve",
                    "settlement_kind": settlement_kind,
                    "reason": cli_reason,
                },
                "cli_preview": approve_cli,
                "implemented": False,
            },
            "reject": {
                "method": "POST",
                "path": f"/api/v1/settlement-inbox/{target_id}/reject",
                "body": {
                    "pr": pr_number,
                    "head_sha": head_sha,
                    "decision": "reject",
                    "settlement_kind": settlement_kind,
                    "reason": reject_cli[-1],
                },
                "cli_preview": reject_cli,
                "implemented": False,
            },
        },
        "_sort_ts": _parse_ts(generated_at),
    }


def collect_pending_settlement_approvals(
    *,
    limit: int = 20,
    repo: str | None = None,
    review_queue_root: str | None = None,
    merge_packet_builder: Any | None = None,
) -> list[dict[str, Any]]:
    """Return settlement-ready PRs as approval-inbox item dictionaries.

    The merge-packet is the authority for readiness. Entries with pending
    checks, incomplete quorum, unresolved dissent, or any other
    ``repair_or_wait`` state are intentionally excluded.
    """
    if merge_packet_builder is None:
        from aragora.cli.commands.review_queue import _build_merge_authorization_packet

        merge_packet_builder = _build_merge_authorization_packet

    packet = merge_packet_builder(
        pr_refs=[],
        limit=max(1, int(limit or 20)),
        repo_override=repo or os.environ.get("ARAGORA_SETTLEMENT_INBOX_REPO") or None,
        review_queue_root=review_queue_root or os.environ.get("ARAGORA_REVIEW_QUEUE_ROOT") or None,
        execute_reviewers=False,
        ignore_own_quorum_check=False,
    )
    generated_at = str(packet.get("generated_at") or "")
    items = []
    for entry in packet.get("entries", []):
        if not isinstance(entry, dict):
            continue
        try:
            pr_number = int(entry.get("pr_number") or 0)
        except (TypeError, ValueError):
            continue
        head_sha = str(entry.get("head_sha") or "").strip()
        if pr_number <= 0 or not head_sha:
            continue
        status = str(entry.get("status") or "").strip()
        if status not in SETTLEMENT_READY_STATUSES:
            continue
        if bool(entry.get("unresolved_dissent")):
            continue
        items.append(_approval_item(entry, generated_at=generated_at))
    items.sort(key=lambda item: float(item.get("_sort_ts") or 0.0), reverse=True)
    for item in items:
        item.pop("_sort_ts", None)
    return items[: max(1, int(limit or 20))]


__all__ = [
    "SETTLEMENT_READY_STATUSES",
    "collect_pending_settlement_approvals",
]
