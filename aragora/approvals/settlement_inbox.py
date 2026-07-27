"""Settlement approval inbox packets for merge-risk decisions.

This module is deliberately read-only. It turns the existing review-queue
merge-packet into approval-inbox items when a PR is already at the human
settlement boundary. It does not post comments, statuses, evidence, or merges.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections import OrderedDict
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from aragora.cli.commands.review_queue_transport import _GhError

UTC = timezone.utc
logger = logging.getLogger(__name__)

DEFAULT_PACKET_SCAN_LIMIT = 20
MAX_PACKET_SCAN_LIMIT = 100
DEFAULT_PACKET_CACHE_TTL_SECONDS = 30.0
DEFAULT_PACKET_CACHE_MAX_ENTRIES = 32
MAX_PACKET_CACHE_MAX_ENTRIES = 256
EMPTY_PACKET_VERSION = "merge_authorization_packet.v1"

PacketCacheKey = tuple[str | None, str | None, tuple[str, ...], int]
_PACKET_CACHE: OrderedDict[PacketCacheKey, tuple[float, dict[str, Any]]] = OrderedDict()
_PACKET_CACHE_LOCK = RLock()

SETTLEMENT_READY_STATUSES = {
    "human_risk_settlement_required",
    "human_preapproval_required",
}


class SettlementInboxError(RuntimeError):
    """Raised when settlement packet hydration fails for the inbox source."""


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


def _positive_int(value: Any, *, default: int, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(1, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _requested_limit(value: Any) -> int:
    return _positive_int(value, default=DEFAULT_PACKET_SCAN_LIMIT)


def _packet_scan_limit(request_limit: int) -> int:
    configured_limit = _positive_int(
        os.environ.get("ARAGORA_SETTLEMENT_INBOX_PACKET_LIMIT"),
        default=DEFAULT_PACKET_SCAN_LIMIT,
        maximum=MAX_PACKET_SCAN_LIMIT,
    )
    return min(request_limit, configured_limit)


def _cache_ttl_seconds() -> float:
    try:
        ttl = float(os.environ.get("ARAGORA_SETTLEMENT_INBOX_CACHE_TTL_SECONDS", ""))
    except ValueError:
        ttl = DEFAULT_PACKET_CACHE_TTL_SECONDS
    if ttl < 0:
        return 0.0
    return min(ttl, 300.0)


def _cache_max_entries() -> int:
    return _positive_int(
        os.environ.get("ARAGORA_SETTLEMENT_INBOX_CACHE_MAX_ENTRIES"),
        default=DEFAULT_PACKET_CACHE_MAX_ENTRIES,
        maximum=MAX_PACKET_CACHE_MAX_ENTRIES,
    )


def _sync_refresh_enabled() -> bool:
    raw = os.environ.get("ARAGORA_SETTLEMENT_INBOX_ALLOW_SYNC_REFRESH", "")
    return raw.lower().strip() in {"1", "true", "yes", "on"}


def _bounded_queue_scan_enabled() -> bool:
    raw = os.environ.get("ARAGORA_SETTLEMENT_INBOX_ALLOW_BOUNDED_QUEUE_SCAN", "")
    return raw.lower().strip() in {"1", "true", "yes", "on"}


def _configured_pr_refs() -> list[str]:
    raw = os.environ.get("ARAGORA_SETTLEMENT_INBOX_PR_REFS", "")
    return [ref for ref in raw.replace(",", " ").split() if ref.strip()]


def _empty_packet() -> dict[str, Any]:
    return {
        "version": EMPTY_PACKET_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "entries": [],
    }


def _packet_cache_key(
    repo_override: str | None,
    review_queue_root: str | None,
    pr_refs: list[str],
    packet_limit: int,
) -> PacketCacheKey:
    limit_key = packet_limit if not pr_refs and _bounded_queue_scan_enabled() else 0
    return (repo_override, review_queue_root, tuple(pr_refs), limit_key)


def _cache_get_fresh(
    cache_key: PacketCacheKey,
    *,
    now: float,
    ttl: float,
) -> dict[str, Any] | None:
    def fresh_packet_for_key(key: PacketCacheKey) -> dict[str, Any] | None:
        cached = _PACKET_CACHE.get(key)
        if cached is None:
            return None
        if ttl > 0 and now - cached[0] <= ttl:
            _PACKET_CACHE.move_to_end(key)
            return dict(cached[1])
        _PACKET_CACHE.pop(key, None)
        return None

    with _PACKET_CACHE_LOCK:
        cached = fresh_packet_for_key(cache_key)
        if cached is not None:
            return cached
        repo_override, review_queue_root, pr_refs, limit_key = cache_key
        if pr_refs or limit_key <= 0:
            return None
        compatible_keys = sorted(
            (
                key
                for key in _PACKET_CACHE
                if key[0] == repo_override
                and key[1] == review_queue_root
                and key[2] == pr_refs
                and key[3] >= limit_key
            ),
            key=lambda key: key[3],
        )
        for key in compatible_keys:
            cached = fresh_packet_for_key(key)
            if cached is not None:
                return cached
        return None


def _cache_discard(cache_key: PacketCacheKey) -> None:
    with _PACKET_CACHE_LOCK:
        _PACKET_CACHE.pop(cache_key, None)


def _cache_store(cache_key: PacketCacheKey, *, timestamp: float, packet: dict[str, Any]) -> None:
    max_entries = _cache_max_entries()
    with _PACKET_CACHE_LOCK:
        _PACKET_CACHE[cache_key] = (timestamp, dict(packet))
        _PACKET_CACHE.move_to_end(cache_key)
        while len(_PACKET_CACHE) > max_entries:
            _PACKET_CACHE.popitem(last=False)


def _build_merge_packet(
    *,
    merge_packet_builder: Any,
    repo_override: str | None,
    review_queue_root: str | None,
    pr_refs: list[str],
    packet_limit: int,
    use_cache: bool,
    allow_sync_refresh: bool,
) -> dict[str, Any]:
    if not use_cache:
        return _call_merge_packet_builder(
            merge_packet_builder,
            pr_refs=pr_refs,
            limit=packet_limit,
            repo_override=repo_override,
            review_queue_root=review_queue_root,
            execute_reviewers=False,
            ignore_own_quorum_check=False,
        )

    ttl = _cache_ttl_seconds()
    cache_key = _packet_cache_key(repo_override, review_queue_root, pr_refs, packet_limit)
    now = time.monotonic()
    if ttl > 0:
        cached = _cache_get_fresh(cache_key, now=now, ttl=ttl)
        if cached is not None:
            return cached
    if not allow_sync_refresh:
        _cache_discard(cache_key)
        logger.debug("Settlement approval packet cache miss for %s", cache_key)
        return _empty_packet()
    if not pr_refs and not _bounded_queue_scan_enabled():
        return _empty_packet()

    packet = _call_merge_packet_builder(
        merge_packet_builder,
        pr_refs=pr_refs,
        limit=packet_limit,
        repo_override=repo_override,
        review_queue_root=review_queue_root,
        execute_reviewers=False,
        ignore_own_quorum_check=False,
    )
    if ttl > 0:
        _cache_store(cache_key, timestamp=now, packet=packet)
    return packet


def _call_merge_packet_builder(merge_packet_builder: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        packet = merge_packet_builder(**kwargs)
    except (
        _GhError,
        LookupError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        raise SettlementInboxError(str(exc)) from exc
    if not isinstance(packet, dict):
        raise SettlementInboxError("merge packet builder returned non-dict packet")
    return packet


def refresh_settlement_approval_cache(
    *,
    limit: int = DEFAULT_PACKET_SCAN_LIMIT,
    repo: str | None = None,
    pr_refs: list[str] | None = None,
    review_queue_root: str | None = None,
    merge_packet_builder: Any | None = None,
) -> dict[str, Any]:
    """Refresh and return the cached settlement merge packet.

    This intentionally separates the expensive GitHub hydration path from
    approval-inbox reads. A scheduler or explicit operator action can warm the
    cache; request handlers use cached data and never perform a cold scan by
    default.
    """
    requested_limit = _requested_limit(limit)
    packet_limit = _packet_scan_limit(requested_limit)
    if merge_packet_builder is None:
        from aragora.cli.commands.review_queue import _build_merge_authorization_packet

        merge_packet_builder = _build_merge_authorization_packet

    repo_override = repo or os.environ.get("ARAGORA_SETTLEMENT_INBOX_REPO") or None
    queue_root = review_queue_root or os.environ.get("ARAGORA_REVIEW_QUEUE_ROOT") or None
    refs = list(pr_refs) if pr_refs is not None else _configured_pr_refs()
    cache_key = _packet_cache_key(repo_override, queue_root, refs, packet_limit)
    if not refs and not _bounded_queue_scan_enabled():
        packet = _empty_packet()
        _cache_store(cache_key, timestamp=time.monotonic(), packet=packet)
        return packet
    packet = _call_merge_packet_builder(
        merge_packet_builder,
        pr_refs=refs,
        limit=packet_limit,
        repo_override=repo_override,
        review_queue_root=queue_root,
        execute_reviewers=False,
        ignore_own_quorum_check=False,
    )
    _cache_store(cache_key, timestamp=time.monotonic(), packet=packet)
    return packet


def _approval_item(
    entry: dict[str, Any],
    *,
    generated_at: str,
    repo_override: str | None,
    review_queue_root: str | None,
) -> dict[str, Any]:
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
    if repo_override:
        approve_cli.extend(["--repo", repo_override])
    if review_queue_root:
        approve_cli.extend(["--review-queue-root", review_queue_root])
    reject_reason = f"Settlement Inbox rejection for PR #{pr_number} at exact head {head_sha[:12]}"
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
        reject_reason,
    ]
    if repo_override:
        reject_cli.extend(["--repo", repo_override])
    if review_queue_root:
        reject_cli.extend(["--review-queue-root", review_queue_root])

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
                    "repo": repo_override,
                    "review_queue_root": review_queue_root,
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
                    "decision": "request_changes",
                    "settlement_kind": settlement_kind,
                    "reason": reject_reason,
                    "repo": repo_override,
                    "review_queue_root": review_queue_root,
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
    allow_sync_refresh: bool | None = None,
) -> list[dict[str, Any]]:
    """Return settlement-ready PRs as approval-inbox item dictionaries.

    The merge-packet is the authority for readiness. Entries with pending
    checks, incomplete quorum, unresolved dissent, or any other
    ``repair_or_wait`` state are intentionally excluded.
    """
    requested_limit = _requested_limit(limit)
    packet_limit = _packet_scan_limit(requested_limit)
    use_cache = merge_packet_builder is None
    if merge_packet_builder is None:
        from aragora.cli.commands.review_queue import _build_merge_authorization_packet

        merge_packet_builder = _build_merge_authorization_packet

    repo_override = repo or os.environ.get("ARAGORA_SETTLEMENT_INBOX_REPO") or None
    queue_root = review_queue_root or os.environ.get("ARAGORA_REVIEW_QUEUE_ROOT") or None
    refs = _configured_pr_refs()
    packet = _build_merge_packet(
        merge_packet_builder=merge_packet_builder,
        repo_override=repo_override,
        review_queue_root=queue_root,
        pr_refs=refs,
        packet_limit=packet_limit,
        use_cache=use_cache,
        allow_sync_refresh=_sync_refresh_enabled()
        if allow_sync_refresh is None
        else allow_sync_refresh,
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
        items.append(
            _approval_item(
                entry,
                generated_at=generated_at,
                repo_override=repo_override,
                review_queue_root=queue_root,
            )
        )
    # The merge packet has a packet-level generated_at, not per-PR timestamps.
    # Python's stable sort preserves merge-packet queue order for equal times.
    items.sort(key=lambda item: float(item.get("_sort_ts") or 0.0), reverse=True)
    for item in items:
        item.pop("_sort_ts", None)
    return items[:requested_limit]


__all__ = [
    "DEFAULT_PACKET_CACHE_TTL_SECONDS",
    "DEFAULT_PACKET_CACHE_MAX_ENTRIES",
    "DEFAULT_PACKET_SCAN_LIMIT",
    "EMPTY_PACKET_VERSION",
    "MAX_PACKET_CACHE_MAX_ENTRIES",
    "MAX_PACKET_SCAN_LIMIT",
    "SETTLEMENT_READY_STATUSES",
    "SettlementInboxError",
    "collect_pending_settlement_approvals",
    "refresh_settlement_approval_cache",
]
