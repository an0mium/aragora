"""Local-first review queue HTTP surface for the PDB UI v0.

This handler intentionally reuses the existing ``aragora review-queue`` CLI
contracts for queue classification, advisory packets, and settlement receipts.
The web UI becomes a thinner browser layer over the same local truth rather
than a second review system with different semantics.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aragora.cli.commands.review_queue import (
    REVIEW_QUEUE_ARTIFACT_DIR,
    _GhError,
    _build_packet,
    _gh_json,
    _require_clean_worktree,
    _session_id,
    _settle_packet,
)
from aragora.server.handlers.base import BaseHandler, HandlerResult, error_response, json_response
from aragora.server.validation.query_params import safe_query_int
from aragora.server.versioning.compat import strip_version_prefix
from aragora.worktree.fleet import resolve_repo_root

logger = logging.getLogger(__name__)

UTC = timezone.utc
_PR_PATH_RE = re.compile(r"^/api/review-queue/prs/(?P<number>\d+)(?:/(?P<action>[a-z-]+))?$")
_MANUAL_BRIEF_HEADING_RE = re.compile(r"^##\s+#(?P<number>\d+)\s+·\s+`(?P<title>[^`]+)`\s*$", re.M)
_VERDICT_RE = re.compile(
    r"\*\*Verdict:\*\*\s*(?P<verdict>.+?)\s*·\s*\*\*Confidence:\*\*\s*(?P<confidence>\d+)/5(?:.*?)(?:·\s*\*\*Scope:\*\*\s*(?P<scope>[^\n]+))?",
    re.S,
)


@dataclass(slots=True)
class ReviewQueueRow:
    """Queue item plus metadata the web UI needs for ordering and display."""

    number: int
    title: str
    url: str
    head_sha: str
    author: str
    is_draft: bool
    mergeable: str
    review_decision: str
    labels: list[str]
    additions: int
    deletions: int
    changed_files: int
    checks_summary: str
    lane: str
    lane_reason: str
    created_at: str | None
    updated_at: str | None
    status_check_rollup: list[dict[str, Any]]


class ReviewQueueHandler(BaseHandler):
    """Serve the local-first review queue surface used by Aragora Live."""

    ROUTES = [
        "/api/review-queue/prs",
        "/api/review-queue/prs/*",
        "/api/review-queue/prs/*/brief",
        "/api/review-queue/prs/*/approve",
        "/api/review-queue/prs/*/request-changes",
        "/api/review-queue/prs/*/defer",
        "/api/review-queue/stats",
        "/api/v1/review-queue/prs",
        "/api/v1/review-queue/prs/*",
        "/api/v1/review-queue/prs/*/brief",
        "/api/v1/review-queue/prs/*/approve",
        "/api/v1/review-queue/prs/*/request-changes",
        "/api/v1/review-queue/prs/*/defer",
        "/api/v1/review-queue/stats",
    ]

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str, method: str = "GET") -> bool:
        normalized = strip_version_prefix(path)
        return normalized.startswith("/api/review-queue/")

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        method = getattr(handler, "command", "GET").upper() if handler else "GET"
        normalized = strip_version_prefix(path)

        if normalized == "/api/review-queue/stats" and method == "GET":
            return self._handle_stats(handler)

        if normalized == "/api/review-queue/prs" and method == "GET":
            return self._handle_list_prs(query_params, handler)

        match = _PR_PATH_RE.match(normalized)
        if not match:
            return error_response("Review queue endpoint not found", 404)

        pr_number = int(match.group("number"))
        action = str(match.group("action") or "").strip()

        if not action and method == "GET":
            return self._handle_pr_detail(pr_number, handler)
        if action == "brief" and method == "GET":
            return self._handle_brief(pr_number, handler)
        if action == "approve" and method == "POST":
            return self._handle_settlement(pr_number, "approve", handler)
        if action == "request-changes" and method == "POST":
            return self._handle_settlement(pr_number, "request_changes", handler)
        if action == "defer" and method == "POST":
            return self._handle_defer(pr_number, handler)

        return error_response("Method not allowed", 405)

    def _handle_list_prs(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        _, perm_err = self.require_permission_or_error(handler, "approval:read")
        if perm_err:
            return perm_err

        limit = safe_query_int(query_params, "limit", default=30, min_val=1, max_val=100)
        ready_only = _query_bool(query_params, "ready_only", default=False)
        include_parked = _query_bool(query_params, "include_parked", default=False)
        include_deferred = _query_bool(query_params, "include_deferred", default=False)

        repo_root = _resolve_storage_repo_root(_current_repo_root())
        active_deferred = _load_active_deferred(repo_root)

        try:
            queue_items = _build_queue_rows(limit=limit)
        except _GhError as exc:
            logger.warning("review queue list failed: %s", exc)
            return error_response(str(exc), 502)

        payload: list[dict[str, Any]] = []
        for item in queue_items:
            if ready_only and item.lane != "ready_now":
                continue
            if not include_parked and item.lane == "parked":
                continue

            deferred_entry = active_deferred.get(item.number)
            if deferred_entry and not include_deferred:
                continue

            try:
                packet = _build_packet(str(item.number), repo_override=None)
            except _GhError as exc:
                logger.warning("review queue packet hydration failed for #%s: %s", item.number, exc)
                packet = None

            brief = _load_brief(repo_root, pr_number=item.number, head_sha=item.head_sha)
            payload.append(
                _serialize_queue_item(
                    item=item,
                    packet=packet,
                    brief=brief,
                    deferred_entry=deferred_entry,
                )
            )

        payload.sort(key=_queue_sort_key)
        return json_response(
            {
                "prs": payload,
                "count": len(payload),
                "generated_at": datetime.now(UTC).isoformat(),
                "source": "local-review-queue",
            }
        )

    def _handle_pr_detail(self, pr_number: int, handler: Any) -> HandlerResult:
        _, perm_err = self.require_permission_or_error(handler, "approval:read")
        if perm_err:
            return perm_err

        repo_root = _resolve_storage_repo_root(_current_repo_root())
        try:
            packet = _build_packet(str(pr_number), repo_override=None)
            payload = _gh_json(
                [
                    "pr",
                    "view",
                    str(pr_number),
                    "--json",
                    ",".join(
                        [
                            "number",
                            "title",
                            "url",
                            "headRefOid",
                            "baseRefOid",
                            "createdAt",
                            "updatedAt",
                            "body",
                            "files",
                            "statusCheckRollup",
                            "author",
                            "mergeable",
                            "reviewDecision",
                            "isDraft",
                            "labels",
                            "additions",
                            "deletions",
                            "changedFiles",
                        ]
                    ),
                ]
            )
        except _GhError as exc:
            logger.warning("review queue detail failed for #%s: %s", pr_number, exc)
            return error_response(str(exc), 502)

        if not isinstance(payload, dict):
            return error_response(f"PR #{pr_number} not found", 404)

        item = _row_from_payload(payload)
        brief = _load_brief(repo_root, pr_number=pr_number, head_sha=packet.head_sha)
        return json_response(
            {
                "pr": _serialize_queue_item(
                    item=item, packet=packet, brief=brief, deferred_entry=None
                ),
                "packet": packet.to_dict(),
                "brief": brief,
                "checks": _normalize_checks(payload.get("statusCheckRollup") or []),
                "files": _normalize_files(payload.get("files") or []),
                "diff_url": f"{packet.url}/files",
            }
        )

    def _handle_brief(self, pr_number: int, handler: Any) -> HandlerResult:
        _, perm_err = self.require_permission_or_error(handler, "approval:read")
        if perm_err:
            return perm_err

        repo_root = _resolve_storage_repo_root(_current_repo_root())
        try:
            packet = _build_packet(str(pr_number), repo_override=None)
        except _GhError as exc:
            return error_response(str(exc), 502)

        brief = _load_brief(repo_root, pr_number=pr_number, head_sha=packet.head_sha)
        if brief is None:
            return error_response("Brief not found", 404)
        return json_response({"brief": brief})

    def _handle_settlement(self, pr_number: int, action: str, handler: Any) -> HandlerResult:
        _, perm_err = self.require_permission_or_error(handler, "approvals:manage")
        if perm_err:
            return perm_err

        body = self.read_json_body(handler) or {}
        reason = str(body.get("reason", "") or "").strip()
        if action == "request_changes" and not reason:
            return error_response("request_changes requires a non-empty reason", 400)

        worktree_root = _current_repo_root()
        repo_root = _resolve_storage_repo_root(worktree_root)

        try:
            _require_clean_worktree(worktree_root)
            packet = _build_packet(str(pr_number), repo_override=None)
            receipt = _settle_packet(
                packet=packet,
                action=action,
                reason=reason,
                repo_root=repo_root,
                repo_override=None,
                session_id=_session_id(),
            )
            _clear_deferred(repo_root, pr_number)
        except _GhError as exc:
            status = 409 if "head changed" in str(exc) else 400
            logger.warning("review queue settlement failed for #%s: %s", pr_number, exc)
            return error_response(str(exc), status)

        return json_response({"receipt": receipt.to_dict()})

    def _handle_defer(self, pr_number: int, handler: Any) -> HandlerResult:
        _, perm_err = self.require_permission_or_error(handler, "approvals:manage")
        if perm_err:
            return perm_err

        body = self.read_json_body(handler) or {}
        reason = str(body.get("reason", "") or "").strip()
        repo_root = _resolve_storage_repo_root(_current_repo_root())
        deferred_until = datetime.now(UTC) + timedelta(hours=4)
        entry = {
            "pr_number": pr_number,
            "reason": reason,
            "deferred_at": datetime.now(UTC).isoformat(),
            "deferred_until": deferred_until.isoformat(),
        }
        state = _load_deferred_state(repo_root)
        state[str(pr_number)] = entry
        _write_json(_deferred_path(repo_root), state)
        return json_response({"deferred": entry})

    def _handle_stats(self, handler: Any) -> HandlerResult:
        _, perm_err = self.require_permission_or_error(handler, "approval:read")
        if perm_err:
            return perm_err

        repo_root = _resolve_storage_repo_root(_current_repo_root())
        receipts = _load_receipts(repo_root)
        today_local = datetime.now().astimezone().date()
        todays_receipts = [
            receipt
            for receipt in receipts
            if _receipt_local_date(receipt.get("reviewed_at")) == today_local
        ]

        elapsed = sorted(
            float(item.get("elapsed_seconds"))
            for item in todays_receipts
            if isinstance(item.get("elapsed_seconds"), (int, float))
        )
        median = 0.0
        if elapsed:
            mid = len(elapsed) // 2
            median = elapsed[mid] if len(elapsed) % 2 else (elapsed[mid - 1] + elapsed[mid]) / 2

        approvals_today = sum(1 for item in todays_receipts if item.get("action") == "approve")
        return json_response(
            {
                "decisions_today": len(todays_receipts),
                "approvals_today": approvals_today,
                "median_decision_seconds": round(median, 2),
                "streak": len(todays_receipts),
                "source": "local-review-queue",
            }
        )


def _current_repo_root() -> Path:
    return resolve_repo_root(Path.cwd())


def _resolve_storage_repo_root(repo_root: Path) -> Path:
    override = os.environ.get("ARAGORA_REVIEW_QUEUE_STORAGE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if (_review_queue_root(repo_root)).exists():
        return repo_root
    parts = repo_root.parts
    if ".worktrees" in parts:
        index = parts.index(".worktrees")
        candidate = Path(*parts[:index]) if index > 0 else repo_root
        if _review_queue_root(candidate).exists():
            return candidate
    return repo_root


def _review_queue_root(repo_root: Path) -> Path:
    return repo_root / REVIEW_QUEUE_ARTIFACT_DIR


def _briefs_dir(repo_root: Path) -> Path:
    return _review_queue_root(repo_root) / "briefs"


def _deferred_path(repo_root: Path) -> Path:
    return _review_queue_root(repo_root) / "deferred.json"


def _receipts_dir(repo_root: Path) -> Path:
    return _review_queue_root(repo_root) / "receipts"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _query_bool(query_params: dict[str, Any], key: str, *, default: bool) -> bool:
    raw = query_params.get(key)
    if raw is None:
        return default
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _load_deferred_state(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = _deferred_path(repo_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            entries[str(key)] = value
    return entries


def _load_active_deferred(repo_root: Path) -> dict[int, dict[str, Any]]:
    now = datetime.now(UTC)
    active: dict[int, dict[str, Any]] = {}
    dirty = False
    state = _load_deferred_state(repo_root)
    for key, value in state.items():
        until_text = str(value.get("deferred_until", "") or "").strip()
        try:
            until = _parse_iso_datetime(until_text)
        except ValueError:
            dirty = True
            continue
        if until <= now:
            dirty = True
            continue
        try:
            active[int(key)] = value
        except ValueError:
            dirty = True
    if dirty:
        cleaned = {str(number): value for number, value in active.items()}
        _write_json(_deferred_path(repo_root), cleaned)
    return active


def _clear_deferred(repo_root: Path, pr_number: int) -> None:
    state = _load_deferred_state(repo_root)
    key = str(pr_number)
    if key not in state:
        return
    del state[key]
    _write_json(_deferred_path(repo_root), state)


def _load_receipts(repo_root: Path) -> list[dict[str, Any]]:
    directory = _receipts_dir(repo_root)
    if not directory.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            receipts.append(payload)
    receipts.sort(key=lambda item: str(item.get("reviewed_at", "")))
    return receipts


def _receipt_local_date(value: Any) -> datetime.date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _parse_iso_datetime(value).astimezone().date()
    except ValueError:
        return None


def _parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_brief(repo_root: Path, *, pr_number: int, head_sha: str | None) -> dict[str, Any] | None:
    brief = _load_json_brief(repo_root, pr_number=pr_number, head_sha=head_sha)
    if brief is not None:
        return brief
    return _load_manual_markdown_brief(repo_root, pr_number=pr_number)


def _load_json_brief(
    repo_root: Path, *, pr_number: int, head_sha: str | None
) -> dict[str, Any] | None:
    directory = _briefs_dir(repo_root)
    if not directory.exists():
        return None

    candidates = sorted(directory.glob(f"pr-{pr_number}-*.json"))
    if head_sha:
        prefix = head_sha[:12]
        exact = [path for path in candidates if path.name == f"pr-{pr_number}-{prefix}.json"]
        if exact:
            candidates = exact

    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        verdict = _normalize_verdict(payload.get("verdict"))
        return {
            "pr_number": pr_number,
            "title": str(payload.get("title", "") or "").strip() or None,
            "source": "brief_json",
            "source_path": str(path),
            "head_sha": str(payload.get("head_sha", "") or head_sha or "").strip() or None,
            "verdict": verdict,
            "raw_verdict": str(payload.get("verdict", "") or "").strip() or None,
            "confidence": _safe_int(payload.get("confidence")),
            "scope": str(payload.get("scope", "") or "").strip() or None,
            "logic": _clean_text(payload.get("logic")),
            "security": _clean_text(payload.get("security")),
            "maintainability": _clean_text(payload.get("maintainability")),
            "skeptic": _clean_text(payload.get("skeptic")),
            "recommended_action": _clean_text(
                payload.get("recommended_action") or payload.get("recommendedAction")
            ),
        }
    return None


def _load_manual_markdown_brief(repo_root: Path, *, pr_number: int) -> dict[str, Any] | None:
    files = sorted(_review_queue_root(repo_root).glob("manual-briefs-*.md"))
    for path in reversed(files):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        brief = _parse_manual_brief_markdown(content, pr_number=pr_number)
        if brief is not None:
            brief["source"] = "manual_markdown"
            brief["source_path"] = str(path)
            return brief
    return None


def _parse_manual_brief_markdown(content: str, *, pr_number: int) -> dict[str, Any] | None:
    matches = list(_MANUAL_BRIEF_HEADING_RE.finditer(content))
    for index, match in enumerate(matches):
        number = int(match.group("number"))
        if number != pr_number:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        section = content[start:end].strip()
        verdict_match = _VERDICT_RE.search(section)
        verdict_raw = verdict_match.group("verdict").strip() if verdict_match else None
        confidence = (
            int(verdict_match.group("confidence"))
            if verdict_match and verdict_match.group("confidence")
            else None
        )
        scope = (
            verdict_match.group("scope").strip()
            if verdict_match and verdict_match.group("scope")
            else None
        )
        return {
            "pr_number": pr_number,
            "title": match.group("title").strip(),
            "head_sha": None,
            "verdict": _normalize_verdict(verdict_raw),
            "raw_verdict": verdict_raw,
            "confidence": confidence,
            "scope": scope,
            "logic": _extract_markdown_field(section, "Logic"),
            "security": _extract_markdown_field(section, "Security"),
            "maintainability": _extract_markdown_field(section, "Maintainability"),
            "skeptic": _extract_markdown_field(section, "Skeptic"),
            "recommended_action": _extract_markdown_field(section, "Recommended action"),
        }
    return None


def _extract_markdown_field(section: str, label: str) -> str | None:
    labels = [
        "Verdict",
        "Confidence",
        "Scope",
        "Logic",
        "Security",
        "Maintainability",
        "Skeptic",
        "Recommended action",
    ]
    next_labels = [candidate for candidate in labels if candidate != label]
    terminator = "|".join(re.escape(item) for item in next_labels)
    pattern = re.compile(
        rf"\*\*{re.escape(label)}:\*\*\s*(?P<value>.+?)(?=(?:\n\n\*\*(?:{terminator}):\*\*)|\Z)",
        re.S,
    )
    match = pattern.search(section)
    if not match:
        return None
    return _clean_text(match.group("value"))


def _normalize_verdict(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "repair_first" in text or "repair first" in text or "fix before review" in text:
        return "repair_first"
    if "needs_human_attention" in text or "needs human attention" in text or "⚠" in text:
        return "needs_human_attention"
    if "approve_candidate" in text or "approve" in text or "✓" in text:
        return "approve_candidate"
    return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_checks(raw_checks: Iterable[Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for item in raw_checks:
        if not isinstance(item, dict):
            continue
        checks.append(
            {
                "name": str(item.get("name", "") or "").strip(),
                "status": str(item.get("status", "") or "").strip(),
                "conclusion": str(item.get("conclusion", "") or "").strip(),
                "details_url": str(item.get("detailsUrl", "") or "").strip() or None,
            }
        )
    return checks


def _normalize_files(raw_files: Iterable[Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "") or "").strip()
        if not path:
            continue
        files.append(
            {
                "path": path,
                "additions": _safe_int(item.get("additions")) or 0,
                "deletions": _safe_int(item.get("deletions")) or 0,
            }
        )
    return files


def _classify_pr_from_payload(payload: dict[str, Any]) -> Any:
    from aragora.cli.commands.review_queue import _classify_pr

    return _classify_pr(payload)


def _build_queue_rows(*, limit: int) -> list[ReviewQueueRow]:
    fields = ",".join(
        [
            "number",
            "title",
            "url",
            "headRefOid",
            "isDraft",
            "mergeable",
            "reviewDecision",
            "labels",
            "author",
            "additions",
            "deletions",
            "changedFiles",
            "statusCheckRollup",
            "createdAt",
            "updatedAt",
        ]
    )
    payload = _gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            str(limit),
            "--json",
            fields,
        ]
    )
    rows: list[ReviewQueueRow] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        rows.append(_row_from_payload(item))
    rows.sort(
        key=lambda row: (
            _queue_sort_key({"lane": row.lane, "created_at": row.created_at, "number": row.number})
        )
    )
    return rows


def _row_from_payload(payload: dict[str, Any]) -> ReviewQueueRow:
    queue_item = _classify_pr_from_payload(payload)
    return ReviewQueueRow(
        number=queue_item.number,
        title=queue_item.title,
        url=queue_item.url,
        head_sha=queue_item.head_sha,
        author=queue_item.author,
        is_draft=queue_item.is_draft,
        mergeable=queue_item.mergeable,
        review_decision=queue_item.review_decision,
        labels=queue_item.labels,
        additions=queue_item.additions,
        deletions=queue_item.deletions,
        changed_files=queue_item.changed_files,
        checks_summary=queue_item.checks_summary,
        lane=queue_item.lane,
        lane_reason=queue_item.lane_reason,
        created_at=_clean_text(payload.get("createdAt")),
        updated_at=_clean_text(payload.get("updatedAt")),
        status_check_rollup=[
            entry for entry in (payload.get("statusCheckRollup") or []) if isinstance(entry, dict)
        ],
    )


def _check_counts(status_check_rollup: Iterable[Any]) -> dict[str, int]:
    counts = {"success": 0, "failure": 0, "pending": 0, "cancelled": 0, "total": 0}
    for item in status_check_rollup:
        if not isinstance(item, dict):
            continue
        counts["total"] += 1
        status = str(item.get("status", "") or "").upper()
        conclusion = str(item.get("conclusion", "") or "").upper()
        if conclusion == "SUCCESS":
            counts["success"] += 1
        elif conclusion in {"FAILURE", "TIMED_OUT", "ACTION_REQUIRED"}:
            counts["failure"] += 1
        elif conclusion in {"CANCELLED", "SKIPPED", "NEUTRAL", "STALE"}:
            counts["cancelled"] += 1
        elif status in {"IN_PROGRESS", "QUEUED", "PENDING"} or not conclusion:
            counts["pending"] += 1
    return counts


def _serialize_queue_item(
    *,
    item: Any,
    packet: Any | None,
    brief: dict[str, Any] | None,
    deferred_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    created_at = getattr(item, "created_at", None)
    updated_at = getattr(item, "updated_at", None)
    checks_rollup = getattr(item, "status_check_rollup", None) or []
    return {
        "number": item.number,
        "title": item.title,
        "url": item.url,
        "diff_url": f"{item.url}/files",
        "head_sha": item.head_sha,
        "author": item.author,
        "is_draft": item.is_draft,
        "mergeable": item.mergeable,
        "review_decision": item.review_decision,
        "labels": item.labels,
        "additions": item.additions,
        "deletions": item.deletions,
        "changed_files": item.changed_files,
        "checks_summary": item.checks_summary,
        "lane": item.lane,
        "lane_reason": item.lane_reason,
        "created_at": created_at,
        "updated_at": updated_at,
        "status_counts": _check_counts(checks_rollup),
        "touched_subsystems": list(packet.touched_subsystems) if packet else [],
        "high_risk_paths_touched": list(packet.high_risk_paths_touched) if packet else [],
        "machine_recommendation": getattr(packet, "machine_recommendation", None),
        "machine_recommendation_reason": getattr(packet, "machine_recommendation_reason", None),
        "brief": brief,
        "brief_available": brief is not None,
        "deferred_until": deferred_entry.get("deferred_until") if deferred_entry else None,
        "deferred_reason": deferred_entry.get("reason") if deferred_entry else None,
    }


def _queue_sort_key(item: dict[str, Any]) -> tuple[int, str, int]:
    lane_order = {
        "repairable": 0,
        "needs_attention": 1,
        "ready_now": 2,
        "parked": 3,
    }
    created_at = str(item.get("created_at", "") or "")
    return (
        lane_order.get(str(item.get("lane", "") or ""), 99),
        created_at,
        int(item.get("number", 0) or 0),
    )


__all__ = ["ReviewQueueHandler"]
