from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.ralph.github_control import GitHubControl
from aragora.swarm.campaign import CampaignPlanner, locked_manifest_path
from aragora.swarm.pr_registry import PullRequestRegistry
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.tranche import (
    TrancheArtifactStore,
    TrancheExecutor,
    TrancheInspector,
    load_tranche_manifest,
)
from aragora.swarm.tranche_design_review import (
    DesignReviewRecord,
    load_design_review,
    run_design_review,
    save_design_review,
)
from aragora.swarm.tranche_integrate import integrate_lane
from aragora.swarm.tranche_review import review_lane, select_review_tier
from aragora.swarm.tranche_state import (
    TRANCHE_STATUS_ABORTED,
    TRANCHE_STATUS_COMPLETED,
    TRANCHE_STATUS_NEEDS_HUMAN,
)
from aragora.swarm.tranche_submit import (
    campaign_projects_to_candidate_lanes,
    submit_intake_bundle,
)
from aragora.swarm.tranche_watch import (
    DriverAlreadyClaimedError,
    claim_driver,
    load_tranche_run_state,
    refresh_tranche_state,
    release_driver,
    run_state_path_for_manifest,
    watch_loop,
)

logger = logging.getLogger(__name__)

QUEUE_STATUS_PENDING = "pending"
QUEUE_STATUS_RUNNING = "running"
QUEUE_STATUS_COMPLETED = "completed"
QUEUE_STATUS_STOPPED = "stopped"

QUEUE_ITEM_STATUS_PENDING = "pending"
QUEUE_ITEM_STATUS_RUNNING = "running"
QUEUE_ITEM_STATUS_COMPLETED = "completed"
QUEUE_ITEM_STATUS_NEEDS_HUMAN = "needs_human"
QUEUE_ITEM_STATUS_STOPPED = "stopped"

_QUEUE_TERMINAL_STATUSES = frozenset(
    {
        QUEUE_STATUS_COMPLETED,
        QUEUE_STATUS_STOPPED,
    }
)
_QUEUE_ITEM_TERMINAL_STATUSES = frozenset(
    {
        QUEUE_ITEM_STATUS_COMPLETED,
        QUEUE_ITEM_STATUS_NEEDS_HUMAN,
        QUEUE_ITEM_STATUS_STOPPED,
    }
)
_SAFE_QUEUE_AUTONOMY_MODES = frozenset({"adaptive", "checkpoint"})
_SYSTEMIC_REVIEW_MARKERS = (
    "no configured reviewer candidate succeeded",
    "review blocked (billing):",
)
_SYSTEMIC_PUBLISH_ACTIONS = frozenset({"push_failed", "pr_create_failed"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_scope(scope: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in scope:
        value = str(item).strip().removeprefix("./").rstrip("/")
        if not value:
            continue
        if "*" in value:
            normalized.append(value)
            continue
        if "/" in value and "." not in value.rsplit("/", 1)[-1]:
            normalized.append(f"{value}/**")
            continue
        normalized.append(value)
    return list(dict.fromkeys(normalized))


def _resolve_queue_autonomy_mode(
    requested_mode: str | None, *, merge_class: str
) -> tuple[str, bool]:
    requested = _optional_text(requested_mode) or "adaptive"
    if requested == "fire_and_forget" and str(merge_class or "").strip().lower() == "low_risk":
        return requested, False
    if requested not in _SAFE_QUEUE_AUTONOMY_MODES:
        return "adaptive", True
    return requested, False


def _queue_merge_policy(*, merge_class: str) -> str:
    return "auto" if str(merge_class or "").strip().lower() == "low_risk" else "manual"


def _resolve_queue_max_parallel_lanes(value: Any) -> int:
    lanes = int(value or 1)
    if lanes not in {1, 2}:
        raise ValueError("Queue max_parallel_lanes must be 1 or 2.")
    return lanes


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return (
            value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        )
    text = _optional_text(value)
    if not text:
        return _utcnow()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return _utcnow()


def _coerce_optional_datetime(value: Any) -> datetime | None:
    return _coerce_datetime(value) if _optional_text(value) else None


def _load_structured_object(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(raw) or {}
    except ImportError:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Structured queue input must deserialize to an object.")
    return dict(payload)


def _dump_structured_object(data: dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=True, allow_unicode=False)
    except ImportError:
        return json.dumps(data, indent=2, sort_keys=True)


def queue_state_path_for_queue(queue_path: str | Path) -> Path:
    path = Path(queue_path).resolve()
    if path.suffix:
        return path.with_suffix(".queue_state.yaml")
    return path.with_name(f"{path.name}.queue_state.yaml")


@dataclass(slots=True)
class TrancheQueueItem:
    item_id: str
    kind: str
    source: str
    objective_override: str | None = None
    merge_class: str = "manual"
    max_lanes: int = 1
    allowed_write_scope: list[str] = field(default_factory=list)
    autonomy_mode: str | None = None
    verification_commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.item_id,
            "kind": self.kind,
            "source": self.source,
            "merge_class": self.merge_class,
            "max_lanes": self.max_lanes,
        }
        if self.objective_override:
            payload["objective_override"] = self.objective_override
        if self.allowed_write_scope:
            payload["allowed_write_scope"] = list(self.allowed_write_scope)
        if self.autonomy_mode:
            payload["autonomy_mode"] = self.autonomy_mode
        if self.verification_commands:
            payload["verification_commands"] = list(self.verification_commands)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheQueueItem:
        item_id = str(data.get("id", "")).strip()
        kind = str(data.get("kind", "")).strip().lower()
        source = str(data.get("source", "")).strip()
        if not item_id:
            raise ValueError("Queue item id is required.")
        if kind not in {"issue", "intake"}:
            raise ValueError(f"Queue item {item_id} has unsupported kind: {kind!r}")
        if not source:
            raise ValueError(f"Queue item {item_id} is missing source.")
        merge_class = str(data.get("merge_class", "manual")).strip().lower() or "manual"
        if merge_class not in {"low_risk", "manual"}:
            raise ValueError(f"Queue item {item_id} has unsupported merge_class: {merge_class!r}")
        max_lanes = max(1, int(data.get("max_lanes", 1) or 1))
        return cls(
            item_id=item_id,
            kind=kind,
            source=source,
            objective_override=_optional_text(data.get("objective_override")),
            merge_class=merge_class,
            max_lanes=max_lanes,
            allowed_write_scope=_normalize_scope(_string_list(data.get("allowed_write_scope"))),
            autonomy_mode=_optional_text(data.get("autonomy_mode")),
            verification_commands=_string_list(data.get("verification_commands")),
        )


@dataclass(slots=True)
class TrancheQueueManifest:
    queue_id: str
    items: list[TrancheQueueItem]
    manifest_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "queue_id": self.queue_id,
            "items": [item.to_dict() for item in self.items],
        }

    def to_yaml(self) -> str:
        return _dump_structured_object(self.to_dict())

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, default_queue_id: str | None = None
    ) -> TrancheQueueManifest:
        items_raw = data.get("items") or []
        if not isinstance(items_raw, list):
            raise ValueError("Queue manifest items must be a list.")
        items = [TrancheQueueItem.from_dict(item) for item in items_raw if isinstance(item, dict)]
        if not items:
            raise ValueError("Queue manifest must contain at least one item.")
        seen: set[str] = set()
        for item in items:
            if item.item_id in seen:
                raise ValueError(f"Queue manifest has duplicate item id: {item.item_id}")
            seen.add(item.item_id)
        queue_id = str(data.get("queue_id", "")).strip() or (default_queue_id or "").strip()
        if not queue_id:
            raise ValueError("Queue manifest is missing queue_id.")
        return cls(
            queue_id=queue_id,
            items=items,
            manifest_version=max(1, int(data.get("manifest_version", 1) or 1)),
        )

    @classmethod
    def load(cls, path: str | Path) -> TrancheQueueManifest:
        target = Path(path).resolve()
        payload = _load_structured_object(target)
        return cls.from_dict(payload, default_queue_id=target.stem)


@dataclass(slots=True)
class TrancheQueueSource:
    source_id: str
    kind: str
    mode: str
    source: str
    merge_class: str = "manual"
    priority: int = 100
    repo: str | None = None
    limit: int = 10
    objective: str | None = None
    max_lanes: int = 1
    allowed_write_scope: list[str] = field(default_factory=list)
    autonomy_mode: str | None = None
    verification_commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.source_id,
            "kind": self.kind,
            "mode": self.mode,
            "source": self.source,
            "merge_class": self.merge_class,
            "priority": self.priority,
            "max_lanes": self.max_lanes,
        }
        if self.repo:
            payload["repo"] = self.repo
        if self.limit != 10:
            payload["limit"] = self.limit
        if self.objective:
            payload["objective"] = self.objective
        if self.allowed_write_scope:
            payload["allowed_write_scope"] = list(self.allowed_write_scope)
        if self.autonomy_mode:
            payload["autonomy_mode"] = self.autonomy_mode
        if self.verification_commands:
            payload["verification_commands"] = list(self.verification_commands)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheQueueSource:
        kind = str(data.get("kind", "")).strip().lower()
        mode = str(data.get("mode", "execute")).strip().lower() or "execute"
        if kind not in {"doc", "issue", "issue_query"}:
            raise ValueError(f"Unsupported queue source kind: {kind!r}")
        if mode not in {"execute", "synthesize"}:
            raise ValueError(f"Unsupported queue source mode: {mode!r}")
        source = _optional_text(data.get("source"))
        if kind == "doc":
            source = _optional_text(data.get("path")) or _optional_text(data.get("url")) or source
        elif kind == "issue":
            source = _optional_text(data.get("url")) or source
        else:
            source = _optional_text(data.get("query")) or source
        if not source:
            raise ValueError(f"Queue source {kind!r} is missing its locator.")
        source_id = _optional_text(data.get("id")) or _default_queue_source_id(kind, source)
        merge_class = str(data.get("merge_class", "manual")).strip().lower() or "manual"
        if merge_class not in {"low_risk", "manual"}:
            raise ValueError(
                f"Queue source {source_id} has unsupported merge_class: {merge_class!r}"
            )
        return cls(
            source_id=source_id,
            kind=kind,
            mode=mode,
            source=source,
            merge_class=merge_class,
            priority=int(data.get("priority", 100) or 100),
            repo=_optional_text(data.get("repo")),
            limit=max(1, int(data.get("limit", 10) or 10)),
            objective=_optional_text(data.get("objective")),
            max_lanes=max(1, int(data.get("max_lanes", 1) or 1)),
            allowed_write_scope=_normalize_scope(_string_list(data.get("allowed_write_scope"))),
            autonomy_mode=_optional_text(data.get("autonomy_mode")),
            verification_commands=_string_list(data.get("verification_commands")),
        )


@dataclass(slots=True)
class TrancheQueueSourceManifest:
    queue_id: str
    sources: list[TrancheQueueSource]
    manifest_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "queue_id": self.queue_id,
            "sources": [source.to_dict() for source in self.sources],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, default_queue_id: str | None = None
    ) -> TrancheQueueSourceManifest:
        sources_raw = data.get("sources") or []
        if not isinstance(sources_raw, list):
            raise ValueError("Queue source manifest sources must be a list.")
        sources = [
            TrancheQueueSource.from_dict(item) for item in sources_raw if isinstance(item, dict)
        ]
        if not sources:
            raise ValueError("Queue source manifest must contain at least one source.")
        seen: set[str] = set()
        for source in sources:
            if source.source_id in seen:
                raise ValueError(
                    f"Queue source manifest has duplicate source id: {source.source_id}"
                )
            seen.add(source.source_id)
        queue_id = str(data.get("queue_id", "")).strip() or (default_queue_id or "").strip()
        if not queue_id:
            raise ValueError("Queue source manifest is missing queue_id.")
        return cls(
            queue_id=queue_id,
            sources=sources,
            manifest_version=max(1, int(data.get("manifest_version", 1) or 1)),
        )

    @classmethod
    def load(cls, path: str | Path) -> TrancheQueueSourceManifest:
        target = Path(path).resolve()
        payload = _load_structured_object(target)
        return cls.from_dict(payload, default_queue_id=target.stem)


@dataclass(slots=True)
class TrancheQueueItemRunState:
    item_id: str
    status: str = QUEUE_ITEM_STATUS_PENDING
    attempts: int = 0
    manifest_id: str | None = None
    manifest_path: str | None = None
    tranche_dir: str | None = None
    requested_autonomy_mode: str | None = None
    effective_autonomy_mode: str | None = None
    submission_status: str | None = None
    inspection_status: str | None = None
    recommended_action: str | None = None
    design_review_recommendation: str | None = None
    design_review_path: str | None = None
    tranche_status: str | None = None
    pr_urls: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "status": self.status,
            "attempts": self.attempts,
            "manifest_id": self.manifest_id,
            "manifest_path": self.manifest_path,
            "tranche_dir": self.tranche_dir,
            "requested_autonomy_mode": self.requested_autonomy_mode,
            "effective_autonomy_mode": self.effective_autonomy_mode,
            "submission_status": self.submission_status,
            "inspection_status": self.inspection_status,
            "recommended_action": self.recommended_action,
            "design_review_recommendation": self.design_review_recommendation,
            "design_review_path": self.design_review_path,
            "tranche_status": self.tranche_status,
            "pr_urls": list(self.pr_urls),
            "findings": list(self.findings),
            "events": [dict(item) for item in self.events],
            "stop_reason": self.stop_reason,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "result": dict(self.result),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrancheQueueItemRunState:
        data = data or {}
        return cls(
            item_id=str(data.get("item_id", "")).strip(),
            status=str(data.get("status", QUEUE_ITEM_STATUS_PENDING)).strip()
            or QUEUE_ITEM_STATUS_PENDING,
            attempts=max(0, int(data.get("attempts", 0) or 0)),
            manifest_id=_optional_text(data.get("manifest_id")),
            manifest_path=_optional_text(data.get("manifest_path")),
            tranche_dir=_optional_text(data.get("tranche_dir")),
            requested_autonomy_mode=_optional_text(data.get("requested_autonomy_mode")),
            effective_autonomy_mode=_optional_text(data.get("effective_autonomy_mode")),
            submission_status=_optional_text(data.get("submission_status")),
            inspection_status=_optional_text(data.get("inspection_status")),
            recommended_action=_optional_text(data.get("recommended_action")),
            design_review_recommendation=_optional_text(data.get("design_review_recommendation")),
            design_review_path=_optional_text(data.get("design_review_path")),
            tranche_status=_optional_text(data.get("tranche_status")),
            pr_urls=_string_list(data.get("pr_urls")),
            findings=_string_list(data.get("findings")),
            events=[dict(item) for item in data.get("events", []) if isinstance(item, dict)],
            stop_reason=_optional_text(data.get("stop_reason")),
            started_at=_coerce_optional_datetime(data.get("started_at")),
            updated_at=_coerce_optional_datetime(data.get("updated_at")),
            finished_at=_coerce_optional_datetime(data.get("finished_at")),
            result=dict(data.get("result") or {}),
        )


@dataclass(slots=True)
class TrancheQueueRunState:
    queue_id: str
    status: str = QUEUE_STATUS_PENDING
    current_item_id: str | None = None
    consecutive_failures: int = 0
    stop_reason: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    item_states: dict[str, TrancheQueueItemRunState] = field(default_factory=dict)

    def ensure_manifest(self, manifest: TrancheQueueManifest) -> None:
        for item in manifest.items:
            self.item_states.setdefault(
                item.item_id, TrancheQueueItemRunState(item_id=item.item_id)
            )
        self.updated_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "status": self.status,
            "current_item_id": self.current_item_id,
            "consecutive_failures": self.consecutive_failures,
            "stop_reason": self.stop_reason,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "item_states": {
                item_id: item_state.to_dict()
                for item_id, item_state in sorted(self.item_states.items())
            },
        }

    def save(self, path: str | Path) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with locked_manifest_path(target):
            target.write_text(_dump_structured_object(self.to_dict()), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrancheQueueRunState:
        data = data or {}
        item_states_raw = data.get("item_states") or {}
        return cls(
            queue_id=str(data.get("queue_id", "")).strip(),
            status=str(data.get("status", QUEUE_STATUS_PENDING)).strip() or QUEUE_STATUS_PENDING,
            current_item_id=_optional_text(data.get("current_item_id")),
            consecutive_failures=max(0, int(data.get("consecutive_failures", 0) or 0)),
            stop_reason=_optional_text(data.get("stop_reason")),
            created_at=_coerce_datetime(data.get("created_at")),
            started_at=_coerce_optional_datetime(data.get("started_at")),
            updated_at=_coerce_datetime(data.get("updated_at")),
            finished_at=_coerce_optional_datetime(data.get("finished_at")),
            item_states={
                str(item_id): TrancheQueueItemRunState.from_dict(item_state)
                for item_id, item_state in item_states_raw.items()
                if str(item_id).strip() and isinstance(item_state, dict)
            },
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        manifest: TrancheQueueManifest,
    ) -> TrancheQueueRunState:
        target = Path(path).resolve()
        if not target.exists():
            state = cls(queue_id=manifest.queue_id)
            state.ensure_manifest(manifest)
            return state
        with locked_manifest_path(target):
            payload = _load_structured_object(target)
        state = cls.from_dict(payload)
        if not state.queue_id:
            state.queue_id = manifest.queue_id
        state.ensure_manifest(manifest)
        return state


@dataclass(slots=True)
class _IssueSource:
    number: int
    repo: str | None
    source_url: str | None


def _default_queue_source_id(kind: str, source: str) -> str:
    if kind == "issue_query":
        return f"query-{_slugify_text(source)[:40] or 'issues'}"
    return f"{kind}-{_slugify_text(source)[:40] or 'source'}"


def _parse_issue_source(value: str) -> _IssueSource:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Issue source is required.")
    if text.startswith("https://github.com/"):
        parts = text.rstrip("/").split("/")
        if len(parts) >= 7 and parts[-2] == "issues" and parts[-1].isdigit():
            repo = "/".join(parts[-4:-2])
            return _IssueSource(number=int(parts[-1]), repo=repo, source_url=text)
    if "#" in text:
        repo, _, number = text.partition("#")
        repo = repo.strip()
        number = number.strip()
        if repo and number.isdigit():
            return _IssueSource(number=int(number), repo=repo, source_url=None)
    if text.isdigit():
        return _IssueSource(number=int(text), repo=None, source_url=None)
    raise ValueError(f"Unsupported issue source: {value!r}")


def _fetch_issue_payload(source: str) -> dict[str, Any]:
    parsed = _parse_issue_source(source)
    cmd = [
        "gh",
        "issue",
        "view",
        str(parsed.number),
        "--json",
        "number,title,body,url",
    ]
    if parsed.repo:
        cmd.extend(["--repo", parsed.repo])
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh issue view failed")
    payload = json.loads(proc.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("gh issue view did not return a JSON object")
    return payload


def _query_issue_payloads(*, query: str, repo: str | None, limit: int) -> list[dict[str, Any]]:
    cmd = [
        "gh",
        "issue",
        "list",
        "--json",
        "number,title,url",
        "--limit",
        str(max(1, int(limit))),
        "--search",
        str(query),
    ]
    if repo:
        cmd.extend(["--repo", repo])
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh issue list failed")
    payload = json.loads(proc.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("gh issue list did not return a JSON array")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _compile_bundle_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_compiled"


def _slugify_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "item"


def _proposal_record(
    source: TrancheQueueSource,
    *,
    reason: str,
    locator: str | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "kind": source.kind,
        "mode": source.mode,
        "merge_class": source.merge_class,
        "status": "needs_human",
        "reason": str(reason or "").strip() or "Queue source requires human clarification.",
        "source": locator or source.source,
    }


def _resolve_doc_source(source: TrancheQueueSource, *, sources_path: Path) -> str:
    candidate = Path(source.source)
    if candidate.exists() or "://" not in source.source:
        if not candidate.is_absolute():
            candidate = (sources_path.parent / candidate).resolve()
        return str(candidate)
    return source.source


def _compile_doc_source_to_bundle(
    source: TrancheQueueSource,
    *,
    sources_path: Path,
    bundle_dir: Path,
    queue_output_path: Path,
    seen_item_ids: set[str],
) -> tuple[TrancheQueueItem | None, dict[str, Any] | None]:
    source_ref = _resolve_doc_source(source, sources_path=sources_path)
    if source.mode == "synthesize":
        return None, _proposal_record(
            source,
            reason=(
                "Synthesize doc sources are not executable yet; provide an explicit objective or "
                "curate this source into a bounded intake bundle."
            ),
            locator=source_ref,
        )
    if "://" not in source_ref and not Path(source_ref).exists():
        return None, _proposal_record(
            source,
            reason="Execute doc source path was not found.",
            locator=source_ref,
        )
    if not source.objective:
        return None, _proposal_record(
            source,
            reason="Execute doc sources require an explicit objective.",
            locator=source_ref,
        )

    item_id = _unique_compiled_item_id(source.source_id, seen_item_ids)
    bundle: dict[str, Any] = {
        "objective": source.objective,
        "autonomy_mode": source.autonomy_mode or "adaptive",
        "source_refs": [
            {
                "url": source_ref,
                "meaning": f"Compiled queue source {source.source_id}",
            }
        ],
    }
    lane: dict[str, Any] = {
        "lane_id": item_id,
        "title": source.objective,
        "prompt": source.objective,
        "owner_role": "implementation_engineer",
        "merge_class": source.merge_class,
        "merge_policy": _queue_merge_policy(merge_class=source.merge_class),
        "queue_item_id": item_id,
        "source_refs": [source_ref],
    }
    if source.allowed_write_scope:
        lane["allowed_write_scope"] = list(source.allowed_write_scope)
    if source.verification_commands:
        lane["verification_commands"] = list(source.verification_commands)
    bundle["candidate_lanes"] = [lane]

    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"{item_id}.intake.yaml"
    bundle_path.write_text(_dump_structured_object(bundle), encoding="utf-8")
    relative_bundle = str(bundle_path.relative_to(queue_output_path.parent))
    return (
        TrancheQueueItem(
            item_id=item_id,
            kind="intake",
            source=relative_bundle,
            merge_class=source.merge_class,
            max_lanes=source.max_lanes,
            allowed_write_scope=list(source.allowed_write_scope),
            autonomy_mode=source.autonomy_mode,
        ),
        None,
    )


def _compile_issue_source(
    source: TrancheQueueSource,
    *,
    seen_item_ids: set[str],
) -> tuple[list[TrancheQueueItem], list[dict[str, Any]]]:
    item_id = _unique_compiled_item_id(source.source_id, seen_item_ids)
    return (
        [
            TrancheQueueItem(
                item_id=item_id,
                kind="issue",
                source=source.source,
                objective_override=source.objective,
                merge_class=source.merge_class,
                max_lanes=source.max_lanes,
                allowed_write_scope=list(source.allowed_write_scope),
                autonomy_mode=source.autonomy_mode,
                verification_commands=list(source.verification_commands),
            )
        ],
        [],
    )


def _compile_issue_query_source(
    source: TrancheQueueSource,
    *,
    seen_item_ids: set[str],
) -> tuple[list[TrancheQueueItem], list[dict[str, Any]]]:
    issues = _query_issue_payloads(query=source.source, repo=source.repo, limit=source.limit)
    items: list[TrancheQueueItem] = []
    for issue in issues:
        number = int(issue.get("number", 0) or 0)
        issue_url = _optional_text(issue.get("url")) or str(number)
        base_item_id = f"{source.source_id}-issue-{number}" if number else source.source_id
        items.append(
            TrancheQueueItem(
                item_id=_unique_compiled_item_id(base_item_id, seen_item_ids),
                kind="issue",
                source=issue_url,
                objective_override=source.objective,
                merge_class=source.merge_class,
                max_lanes=source.max_lanes,
                allowed_write_scope=list(source.allowed_write_scope),
                autonomy_mode=source.autonomy_mode,
                verification_commands=list(source.verification_commands),
            )
        )
    if items:
        return items, []
    return [], [
        _proposal_record(source, reason="Issue query returned no issues.", locator=source.source)
    ]


def _unique_compiled_item_id(candidate: str, seen: set[str]) -> str:
    base = _slugify_text(candidate)
    item_id = base
    suffix = 2
    while item_id in seen:
        item_id = f"{base}-{suffix}"
        suffix += 1
    seen.add(item_id)
    return item_id


def _fallback_issue_lane(item: TrancheQueueItem, *, objective: str, context: str) -> dict[str, Any]:
    scope = item.allowed_write_scope or _normalize_scope(
        SwarmSpec.infer_file_scope_hints(objective)
    )
    return {
        "lane_id": item.item_id,
        "title": objective,
        "prompt": context,
        "owner_role": "implementation_engineer",
        "allowed_write_scope": list(scope),
        "verification_commands": list(item.verification_commands)
        if item.verification_commands
        else [],
        "merge_class": item.merge_class,
        "merge_policy": "manual",
        "queue_item_id": item.item_id,
    }


def _queue_safe_issue_lanes(
    item: TrancheQueueItem,
    *,
    objective: str,
    context: str,
    candidate_lanes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if item.max_lanes != 1:
        return [dict(lane) for lane in candidate_lanes[: item.max_lanes]]
    if len(candidate_lanes) == 1:
        dependencies = [
            str(dep).strip()
            for dep in candidate_lanes[0].get("dependencies", [])
            if str(dep).strip()
        ]
        if not dependencies:
            return [dict(candidate_lanes[0])]
        logger.info(
            "queue issue %s planner produced a dependent lane; collapsing to single fallback lane",
            item.item_id,
        )
    elif candidate_lanes:
        logger.info(
            "queue issue %s planner produced %d lanes; collapsing to single fallback lane",
            item.item_id,
            len(candidate_lanes),
        )
    return [_fallback_issue_lane(item, objective=objective, context=context)]


def _limit_bundle_candidate_lanes(
    *,
    item: TrancheQueueItem,
    candidate_lanes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if item.max_lanes < 1 or len(candidate_lanes) <= item.max_lanes:
        return [dict(lane) for lane in candidate_lanes]
    retained = [dict(lane) for lane in candidate_lanes[: item.max_lanes]]
    retained_ids = {
        str(lane.get("lane_id", "")).strip()
        for lane in retained
        if str(lane.get("lane_id", "")).strip()
    }
    for lane in retained:
        lane["dependencies"] = [
            dep for dep in _string_list(lane.get("dependencies")) if dep in retained_ids
        ]
    logger.info(
        "queue item %s limited bundle candidate lanes from %d to %d",
        item.item_id,
        len(candidate_lanes),
        len(retained),
    )
    return retained


def _event_summary(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if event_type == "run":
        return {
            "type": "run",
            "results": [
                {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "run_id": item.get("run_id"),
                }
                for item in payload.get("results", [])
                if isinstance(item, dict)
            ],
        }
    if event_type == "review":
        return {
            "type": "review",
            "lane_id": payload.get("lane_id"),
            "status": payload.get("status"),
            "findings": _string_list(payload.get("findings")),
        }
    if event_type == "integrate":
        publish_result = payload.get("publish_result")
        summary: dict[str, Any] = {
            "type": "integrate",
            "lane_id": payload.get("lane_id"),
            "recommendation": payload.get("recommendation"),
            "checks": payload.get("checks"),
            "review_status": payload.get("review_status"),
            "pr_url": payload.get("pr_url"),
        }
        if isinstance(publish_result, dict):
            summary["publish_result"] = {
                "action": publish_result.get("action"),
                "detail": publish_result.get("detail"),
            }
        return summary
    return {"type": event_type}


def _truncate_events(events: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    if len(events) <= limit:
        return [dict(item) for item in events]
    return [dict(item) for item in events[-limit:]]


def compile_tranche_queue(
    *,
    sources_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    resolved_sources_path = Path(sources_path).resolve()
    resolved_output_path = Path(output_path).resolve()
    source_manifest = TrancheQueueSourceManifest.load(resolved_sources_path)
    ordered_sources = sorted(
        enumerate(source_manifest.sources),
        key=lambda item: (item[1].priority, item[0]),
    )
    bundle_dir = _compile_bundle_dir(resolved_output_path)
    seen_item_ids: set[str] = set()
    items: list[TrancheQueueItem] = []
    proposals: list[dict[str, Any]] = []

    for _, source in ordered_sources:
        try:
            if source.kind == "issue":
                compiled_items, source_proposals = _compile_issue_source(
                    source,
                    seen_item_ids=seen_item_ids,
                )
            elif source.kind == "issue_query":
                compiled_items, source_proposals = _compile_issue_query_source(
                    source,
                    seen_item_ids=seen_item_ids,
                )
            else:
                compiled_item, proposal = _compile_doc_source_to_bundle(
                    source,
                    sources_path=resolved_sources_path,
                    bundle_dir=bundle_dir,
                    queue_output_path=resolved_output_path,
                    seen_item_ids=seen_item_ids,
                )
                compiled_items = [compiled_item] if compiled_item is not None else []
                source_proposals = [proposal] if proposal is not None else []
        except Exception as exc:
            logger.warning("compile-queue source %s failed: %s", source.source_id, exc)
            compiled_items = []
            source_proposals = [
                _proposal_record(
                    source,
                    reason="Queue source compilation failed. Check logs for detail.",
                )
            ]
        items.extend(compiled_items)
        proposals.extend(source_proposals)

    if not items:
        try:
            resolved_output_path.unlink()
        except FileNotFoundError:
            pass
        return {
            "mode": "tranche-queue-compile",
            "queue_id": source_manifest.queue_id,
            "status": "needs_human",
            "detail": "All sources produced proposals. Review proposals before running.",
            "sources_path": str(resolved_sources_path),
            "output_path": str(resolved_output_path),
            "compiled_bundle_dir": str(bundle_dir) if bundle_dir.exists() else None,
            "item_count": 0,
            "proposal_count": len(proposals),
            "items": [],
            "proposals": proposals,
            "repo_root": str(repo),
            "wrote_queue": False,
        }

    queue_manifest = TrancheQueueManifest(
        queue_id=source_manifest.queue_id,
        items=items,
    )
    compiled_payload = queue_manifest.to_dict()
    compiled_payload.update(
        {
            "compiled_at": _utcnow().isoformat(),
            "sources_manifest": str(resolved_sources_path),
            "compiled_bundle_dir": (str(bundle_dir) if bundle_dir.exists() else None),
            "proposals": proposals,
        }
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        _dump_structured_object(compiled_payload),
        encoding="utf-8",
    )
    return {
        "mode": "tranche-queue-compile",
        "queue_id": source_manifest.queue_id,
        "status": "compiled",
        "sources_path": str(resolved_sources_path),
        "output_path": str(resolved_output_path),
        "compiled_bundle_dir": str(bundle_dir) if bundle_dir.exists() else None,
        "item_count": len(items),
        "proposal_count": len(proposals),
        "items": [item.to_dict() for item in items],
        "proposals": proposals,
        "repo_root": str(repo),
        "wrote_queue": True,
    }


class TrancheQueueExecutor:
    def __init__(
        self,
        *,
        queue_path: str | Path,
        repo_root: str | Path,
        target_branch: str = "main",
        interval_seconds: float = 5.0,
        max_hours: float = 12.0,
        max_consecutive_failures: int = 3,
        planner_model: str = "claude",
        planner_strategy: str = "heuristic",
        worker_model: str = "codex",
        review_model: str = "claude",
        enforce_cross_model_review: bool = True,
        max_parallel_lanes: int = 1,
    ) -> None:
        self.queue_path = Path(queue_path).resolve()
        self.state_path = queue_state_path_for_queue(self.queue_path)
        self.repo_root = Path(repo_root).resolve()
        self.target_branch = str(target_branch or "main").strip() or "main"
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.max_hours = max(0.1, float(max_hours))
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.planner_model = planner_model
        self.planner_strategy = planner_strategy
        self.worker_model = worker_model
        self.review_model = review_model
        self.enforce_cross_model_review = bool(enforce_cross_model_review)
        self.max_parallel_lanes = _resolve_queue_max_parallel_lanes(max_parallel_lanes)
        self._github: GitHubControl | None = None
        self._registry_client_obj: PullRequestRegistry | None = None
        self._supervisor = None

    async def run(self) -> dict[str, Any]:
        manifest = TrancheQueueManifest.load(self.queue_path)
        state = TrancheQueueRunState.load(self.state_path, manifest=manifest)
        state.status = QUEUE_STATUS_RUNNING
        state.started_at = state.started_at or _utcnow()
        state.finished_at = None
        state.stop_reason = None
        state.ensure_manifest(manifest)
        state.save(self.state_path)

        deadline = _utcnow() + timedelta(hours=self.max_hours)
        for item in manifest.items:
            item_state = state.item_states[item.item_id]
            if item_state.status in _QUEUE_ITEM_TERMINAL_STATUSES:
                continue
            if _utcnow() >= deadline:
                state.status = QUEUE_STATUS_STOPPED
                state.stop_reason = "time_limit_exceeded"
                state.current_item_id = item.item_id
                state.finished_at = _utcnow()
                state.updated_at = _utcnow()
                state.save(self.state_path)
                return self.reconcile()

            state.current_item_id = item.item_id
            state.updated_at = _utcnow()
            state.save(self.state_path)

            try:
                systemic_reason = await self._process_item(
                    manifest=manifest,
                    item=item,
                    item_state=item_state,
                    deadline=deadline,
                )
            except Exception as exc:
                logger.exception("tranche queue item %s crashed", item.item_id)
                systemic_reason = None
                item_state.status = QUEUE_ITEM_STATUS_NEEDS_HUMAN
                item_state.stop_reason = "processing_error"
                item_state.finished_at = _utcnow()
                item_state.updated_at = _utcnow()
                item_state.result["processing_error"] = {"error_type": type(exc).__name__}
                self._append_finding(
                    item_state,
                    "Queue item processing failed. Check logs for detail.",
                )

            if item_state.status in {
                QUEUE_ITEM_STATUS_COMPLETED,
                QUEUE_ITEM_STATUS_NEEDS_HUMAN,
            }:
                state.consecutive_failures = 0
            else:
                state.consecutive_failures += 1
            state.updated_at = _utcnow()
            state.current_item_id = None

            if systemic_reason:
                state.status = QUEUE_STATUS_STOPPED
                state.stop_reason = systemic_reason
                state.finished_at = _utcnow()
                state.save(self.state_path)
                return self.reconcile()

            if state.consecutive_failures >= self.max_consecutive_failures:
                state.status = QUEUE_STATUS_STOPPED
                state.stop_reason = "max_consecutive_failures"
                state.finished_at = _utcnow()
                state.save(self.state_path)
                return self.reconcile()

            state.save(self.state_path)

        state.status = QUEUE_STATUS_COMPLETED
        state.stop_reason = None
        state.finished_at = _utcnow()
        state.updated_at = _utcnow()
        state.current_item_id = None
        state.save(self.state_path)
        return self.reconcile()

    def reconcile(self) -> dict[str, Any]:
        manifest = TrancheQueueManifest.load(self.queue_path)
        state = TrancheQueueRunState.load(self.state_path, manifest=manifest)
        artifact_store = TrancheArtifactStore(repo_root=self.repo_root)
        try:
            store: DevCoordinationStore | None = DevCoordinationStore(repo_root=self.repo_root)
        except RuntimeError:
            store = None
        changed = False
        counts: dict[str, int] = {}
        items: list[dict[str, Any]] = []
        for item in manifest.items:
            item_state = state.item_states[item.item_id]
            status = item_state.status
            counts[status] = counts.get(status, 0) + 1
            tranche_status = item_state.tranche_status
            if item_state.manifest_path:
                try:
                    current = load_tranche_run_state(item_state.manifest_path)
                    refreshed = refresh_tranche_state(
                        current,
                        artifact_store=artifact_store,
                        store=store,
                        repo_root=self.repo_root,
                    )
                    if refreshed.to_dict() != current.to_dict():
                        refreshed.save(run_state_path_for_manifest(item_state.manifest_path))
                        changed = True
                    tranche_status = refreshed.status
                    lane_pr_urls = [
                        url
                        for lane_state in refreshed.lane_states.values()
                        if (url := _optional_text(getattr(lane_state, "pr_url", None)))
                    ]
                    if lane_pr_urls != item_state.pr_urls:
                        item_state.pr_urls = lane_pr_urls
                        changed = True
                except (OSError, ValueError):
                    tranche_status = item_state.tranche_status
            if tranche_status != item_state.tranche_status:
                item_state.tranche_status = tranche_status
                changed = True
            if item_state.status not in _QUEUE_ITEM_TERMINAL_STATUSES and tranche_status in {
                TRANCHE_STATUS_COMPLETED,
                TRANCHE_STATUS_NEEDS_HUMAN,
            }:
                item_state.status = (
                    QUEUE_ITEM_STATUS_COMPLETED
                    if tranche_status == TRANCHE_STATUS_COMPLETED
                    else QUEUE_ITEM_STATUS_NEEDS_HUMAN
                )
                if tranche_status == TRANCHE_STATUS_NEEDS_HUMAN and not item_state.stop_reason:
                    item_state.stop_reason = tranche_status
                item_state.finished_at = item_state.finished_at or _utcnow()
                item_state.updated_at = _utcnow()
                status = item_state.status
                changed = True
            else:
                status = item_state.status
            items.append(
                {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "source": item.source,
                    "status": status,
                    "merge_class": item.merge_class,
                    "requested_autonomy_mode": item_state.requested_autonomy_mode,
                    "effective_autonomy_mode": item_state.effective_autonomy_mode,
                    "manifest_id": item_state.manifest_id,
                    "manifest_path": item_state.manifest_path,
                    "submission_status": item_state.submission_status,
                    "inspection_status": item_state.inspection_status,
                    "design_review_recommendation": item_state.design_review_recommendation,
                    "tranche_status": tranche_status,
                    "pr_urls": list(item_state.pr_urls),
                    "findings": list(item_state.findings),
                    "stop_reason": item_state.stop_reason,
                    "started_at": (
                        item_state.started_at.isoformat() if item_state.started_at else None
                    ),
                    "finished_at": (
                        item_state.finished_at.isoformat() if item_state.finished_at else None
                    ),
                }
            )
        running_items = [
            item_id
            for item_id, item_state in state.item_states.items()
            if item_state.status == QUEUE_ITEM_STATUS_RUNNING
        ]
        pending_items = [
            item_id
            for item_id, item_state in state.item_states.items()
            if item_state.status == QUEUE_ITEM_STATUS_PENDING
        ]
        if running_items:
            next_current = running_items[0]
            if state.current_item_id != next_current:
                state.current_item_id = next_current
                changed = True
            if state.status != QUEUE_STATUS_RUNNING:
                state.status = QUEUE_STATUS_RUNNING
                state.stop_reason = None
                state.finished_at = None
                changed = True
        elif pending_items:
            if state.current_item_id is not None:
                state.current_item_id = None
                changed = True
            if state.status == QUEUE_STATUS_RUNNING:
                state.status = QUEUE_STATUS_STOPPED
                state.stop_reason = state.stop_reason or "driver_stopped"
                state.finished_at = state.finished_at or _utcnow()
                changed = True
        else:
            if state.current_item_id is not None:
                state.current_item_id = None
                changed = True
            if state.status != QUEUE_STATUS_COMPLETED:
                state.status = QUEUE_STATUS_COMPLETED
                state.stop_reason = None
                state.finished_at = state.finished_at or _utcnow()
                changed = True
        if changed:
            state.updated_at = _utcnow()
            state.save(self.state_path)
        counts = {}
        for item_state in state.item_states.values():
            counts[item_state.status] = counts.get(item_state.status, 0) + 1
        return {
            "mode": "tranche-queue",
            "queue_id": manifest.queue_id,
            "queue_path": str(self.queue_path),
            "state_path": str(self.state_path),
            "status": state.status,
            "stop_reason": state.stop_reason,
            "current_item_id": state.current_item_id,
            "consecutive_failures": state.consecutive_failures,
            "created_at": state.created_at.isoformat(),
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "updated_at": state.updated_at.isoformat(),
            "finished_at": state.finished_at.isoformat() if state.finished_at else None,
            "counts": counts,
            "items": items,
        }

    def _planner(self) -> CampaignPlanner:
        return CampaignPlanner(
            repo_root=self.repo_root,
            planner_model=self.planner_model,
            planner_strategy=self.planner_strategy,
            worker_model=self.worker_model,
            review_model=self.review_model,
            enforce_cross_model_review=self.enforce_cross_model_review,
            max_parallel_ready_projects=self.max_parallel_lanes,
        )

    def _github_client(self) -> GitHubControl:
        if self._github is None:
            self._github = GitHubControl(repo_root=self.repo_root)
        return self._github

    def _registry_client(self) -> PullRequestRegistry:
        if self._registry_client_obj is None:
            self._registry_client_obj = PullRequestRegistry()
        return self._registry_client_obj

    def _supervisor_store(self):
        if self._supervisor is None:
            from aragora.swarm.supervisor import SwarmSupervisor

            self._supervisor = SwarmSupervisor(repo_root=self.repo_root)
        return self._supervisor

    def _persist_item_progress(
        self,
        *,
        manifest: TrancheQueueManifest,
        item_state: TrancheQueueItemRunState,
        current_item_id: str | None,
    ) -> None:
        state = TrancheQueueRunState.load(self.state_path, manifest=manifest)
        state.status = QUEUE_STATUS_RUNNING
        state.current_item_id = current_item_id
        state.item_states[item_state.item_id] = TrancheQueueItemRunState.from_dict(
            item_state.to_dict()
        )
        state.updated_at = _utcnow()
        state.save(self.state_path)

    async def _process_item(
        self,
        *,
        manifest: TrancheQueueManifest,
        item: TrancheQueueItem,
        item_state: TrancheQueueItemRunState,
        deadline: datetime,
    ) -> str | None:
        item_state.status = QUEUE_ITEM_STATUS_RUNNING
        item_state.attempts += 1
        item_state.started_at = item_state.started_at or _utcnow()
        item_state.updated_at = _utcnow()
        item_state.finished_at = None
        item_state.stop_reason = None
        item_state.events = _truncate_events(item_state.events)

        requested_autonomy = _optional_text(item.autonomy_mode) or "adaptive"
        effective_autonomy, downgraded_autonomy = _resolve_queue_autonomy_mode(
            requested_autonomy,
            merge_class=item.merge_class,
        )
        item_state.requested_autonomy_mode = requested_autonomy
        item_state.effective_autonomy_mode = effective_autonomy
        if downgraded_autonomy:
            self._append_finding(
                item_state,
                "Queue auto-merge policy is not enabled yet; requested fire_and_forget was downgraded to adaptive.",
            )
        self._persist_item_progress(
            manifest=manifest,
            item_state=item_state,
            current_item_id=item.item_id,
        )

        if not item_state.manifest_path:
            bundle = self._bundle_for_item(item, effective_autonomy_mode=effective_autonomy)
            submit_payload = submit_intake_bundle(
                bundle,
                repo_root=self.repo_root,
                autonomy_mode=effective_autonomy,
                planner=self._planner(),
            )
            item_state.manifest_id = _optional_text(submit_payload.get("manifest_id"))
            item_state.manifest_path = _optional_text(submit_payload.get("manifest_path"))
            item_state.tranche_dir = _optional_text(submit_payload.get("tranche_dir"))
            item_state.submission_status = _optional_text(submit_payload.get("submission_status"))
            item_state.inspection_status = _optional_text(submit_payload.get("inspection_status"))
            item_state.recommended_action = _optional_text(submit_payload.get("recommended_action"))
            item_state.result["submit"] = dict(submit_payload)
            item_state.updated_at = _utcnow()
            self._persist_item_progress(
                manifest=manifest,
                item_state=item_state,
                current_item_id=item.item_id,
            )

        if not item_state.manifest_path:
            item_state.status = QUEUE_ITEM_STATUS_NEEDS_HUMAN
            item_state.stop_reason = "manifest_missing_after_submit"
            item_state.finished_at = _utcnow()
            self._append_finding(item_state, "Queue item did not produce a tranche manifest path.")
            return None

        manifest_path = Path(item_state.manifest_path).resolve()
        tranche_manifest = load_tranche_manifest(manifest_path)
        inspection = TrancheInspector(repo_root=self.repo_root).inspect(tranche_manifest)
        item_state.inspection_status = _optional_text(inspection.get("preflight_status"))
        if str(inspection.get("preflight_status", "")).strip() == "blocked":
            item_state.status = QUEUE_ITEM_STATUS_NEEDS_HUMAN
            item_state.stop_reason = "preflight_blocked"
            item_state.finished_at = _utcnow()
            for finding in _string_list(inspection.get("preflight_blockers")):
                self._append_finding(item_state, finding)
            item_state.result["inspection"] = dict(inspection)
            return None

        design_review_path = manifest_path.with_name("design_review.yaml")
        item_state.design_review_path = str(design_review_path)
        design_review_recommendation = item_state.design_review_recommendation
        if not design_review_recommendation:
            if design_review_path.exists():
                record = load_design_review(design_review_path)
                design_review_recommendation = record.recommendation
            else:
                normalized_path = manifest_path.with_name("normalized_bundle.yaml")
                normalized_bundle = _load_structured_object(normalized_path)
                design_review_payload = await run_design_review(
                    manifest=tranche_manifest,
                    normalized_bundle=normalized_bundle,
                    inspection=inspection,
                )
                save_design_review(
                    design_review_path,
                    DesignReviewRecord.from_dict(design_review_payload.get("record")),
                )
                design_review_recommendation = _optional_text(
                    design_review_payload.get("recommendation")
                )
                item_state.result["design_review"] = dict(design_review_payload)
        item_state.design_review_recommendation = design_review_recommendation
        if design_review_recommendation != "approved":
            item_state.status = QUEUE_ITEM_STATUS_NEEDS_HUMAN
            item_state.stop_reason = (
                design_review_recommendation or "design_review_awaiting_confirmation"
            )
            item_state.finished_at = _utcnow()
            self._append_finding(
                item_state,
                f"Design review did not approve execution: {design_review_recommendation or 'unknown'}.",
            )
            return None
        self._persist_item_progress(
            manifest=manifest,
            item_state=item_state,
            current_item_id=item.item_id,
        )

        systemic_reason = await self._drive_manifest(
            item=item,
            item_state=item_state,
            manifest_path=manifest_path,
            tranche_manifest=tranche_manifest,
            deadline=deadline,
        )
        item_state.updated_at = _utcnow()
        return systemic_reason

    def _bundle_for_item(
        self,
        item: TrancheQueueItem,
        *,
        effective_autonomy_mode: str,
    ) -> dict[str, Any]:
        effective_item = replace(item, max_lanes=self._effective_max_lanes(item))
        if effective_item.kind == "intake":
            return self._bundle_from_intake_item(
                effective_item,
                effective_autonomy_mode=effective_autonomy_mode,
            )
        return self._bundle_from_issue_item(
            effective_item,
            effective_autonomy_mode=effective_autonomy_mode,
        )

    def _effective_max_lanes(self, item: TrancheQueueItem) -> int:
        return min(max(1, int(item.max_lanes or 1)), self.max_parallel_lanes)

    def _bundle_from_intake_item(
        self,
        item: TrancheQueueItem,
        *,
        effective_autonomy_mode: str,
    ) -> dict[str, Any]:
        source_path = Path(item.source)
        if not source_path.is_absolute():
            source_path = (self.queue_path.parent / source_path).resolve()
        bundle = _load_structured_object(source_path)
        if item.objective_override:
            bundle["objective"] = item.objective_override
        bundle.setdefault("autonomy_mode", effective_autonomy_mode)
        bundle["max_lanes"] = item.max_lanes
        lanes = bundle.get("candidate_lanes")
        if isinstance(lanes, list):
            normalized_lanes: list[dict[str, Any]] = []
            for lane in lanes:
                if not isinstance(lane, dict):
                    continue
                updated = dict(lane)
                if item.allowed_write_scope:
                    updated["allowed_write_scope"] = list(item.allowed_write_scope)
                updated["merge_class"] = item.merge_class
                updated["merge_policy"] = _queue_merge_policy(merge_class=item.merge_class)
                updated["queue_item_id"] = item.item_id
                normalized_lanes.append(updated)
            if normalized_lanes:
                bundle["candidate_lanes"] = _limit_bundle_candidate_lanes(
                    item=item,
                    candidate_lanes=normalized_lanes,
                )
        return bundle

    def _bundle_from_issue_item(
        self,
        item: TrancheQueueItem,
        *,
        effective_autonomy_mode: str,
    ) -> dict[str, Any]:
        issue = _fetch_issue_payload(item.source)
        number = int(issue.get("number", 0) or 0)
        title = str(issue.get("title", "")).strip()
        body = str(issue.get("body", "")).strip()
        url = _optional_text(issue.get("url"))
        objective = (
            item.objective_override
            or f"Implement issue #{number}: {title}".strip()
            or f"Implement queue issue {item.item_id}"
        )
        issue_text = f"[Issue #{number}] {title}".strip()
        if body:
            issue_text = f"{issue_text}\n\n{body}".strip()
        planning_text = (
            f"{item.objective_override}\n\nSource issue context:\n{issue_text}".strip()
            if item.objective_override
            else issue_text
        )
        planner = self._planner()
        campaign_manifest = planner.plan_from_items(
            [planning_text],
            source_kind="queue_issue",
            source_ref=url or item.source,
        )
        lanes = _queue_safe_issue_lanes(
            item,
            objective=objective,
            context=planning_text,
            candidate_lanes=campaign_projects_to_candidate_lanes(
                campaign_manifest.projects,
                planner=planner,
            ),
        )
        normalized_lanes: list[dict[str, Any]] = []
        for lane in lanes:
            updated = dict(lane)
            if item.allowed_write_scope:
                updated["allowed_write_scope"] = list(item.allowed_write_scope)
            updated["merge_class"] = item.merge_class
            updated["merge_policy"] = _queue_merge_policy(merge_class=item.merge_class)
            updated["queue_item_id"] = item.item_id
            if item.verification_commands and not updated.get("verification_commands"):
                updated["verification_commands"] = list(item.verification_commands)
            normalized_lanes.append(updated)
        return {
            "objective": objective,
            "autonomy_mode": effective_autonomy_mode,
            "max_lanes": item.max_lanes,
            "source_refs": (
                [{"url": url, "meaning": f"Canonical issue for queue item {item.item_id}"}]
                if url
                else []
            ),
            "candidate_lanes": normalized_lanes,
        }

    async def _drive_manifest(
        self,
        *,
        item: TrancheQueueItem,
        item_state: TrancheQueueItemRunState,
        manifest_path: Path,
        tranche_manifest: Any,
        deadline: datetime,
    ) -> str | None:
        state_path = run_state_path_for_manifest(manifest_path)
        current = load_tranche_run_state(manifest_path)
        artifact_store = TrancheArtifactStore(repo_root=self.repo_root)
        executor = TrancheExecutor(repo_root=self.repo_root)
        session_id = f"tranche-queue-{manifest_path.stem}-{os.getpid()}"
        github = self._github_client()
        registry = self._registry_client()
        events = list(item_state.events)

        async def _watch_run_fn(*, manifest):
            try:
                payload = await executor.run(
                    manifest,
                    all_ready=self._effective_max_lanes(item) > 1,
                    owner_session_id=session_id,
                    target_branch=self.target_branch,
                    max_ticks=360,
                    wait_for_completion=False,
                    skip_review=True,
                )
            except ValueError as exc:
                detail = str(exc or "").strip()
                if (
                    "No ready claimable lanes found" in detail
                    or "Tranche is not ready to run." in detail
                    or detail.endswith("is not ready.")
                ):
                    return None
                raise
            if isinstance(payload, dict):
                events.append(_event_summary("run", payload))
            return payload

        async def _watch_review_fn(*, manifest, lane_id, artifact):
            if artifact is None:
                payload = {
                    "lane_id": lane_id,
                    "status": "blocked_nonreviewable",
                    "findings": ["Missing tranche artifact."],
                }
                events.append(_event_summary("review", payload))
                return payload
            run_id = str(getattr(artifact, "run_id", None) or "").strip()
            if not run_id:
                payload = {
                    "lane_id": lane_id,
                    "status": "blocked_nonreviewable",
                    "findings": ["Artifact has no run_id."],
                }
                events.append(_event_summary("review", payload))
                return payload
            supervisor = self._supervisor_store()
            try:
                run_dict = supervisor.refresh_run(run_id).to_dict()
            except Exception:
                record = supervisor.store.get_supervisor_run(run_id)
                if not isinstance(record, dict):
                    payload = {
                        "lane_id": lane_id,
                        "status": "blocked_nonreviewable",
                        "findings": [f"Supervisor run {run_id} is not available."],
                    }
                    events.append(_event_summary("review", payload))
                    return payload
                run_dict = dict(record)
            lane = manifest.lane(lane_id)
            tier = select_review_tier(
                write_scope=list(getattr(lane, "allowed_write_scope", [])),
                diff_lines=int(getattr(artifact, "metadata", {}).get("diff_lines", 0) or 0),
                verification_passed=bool(getattr(artifact, "commands", [])),
                risk_tolerance=str(
                    getattr(artifact, "metadata", {}).get("risk_tolerance", "") or ""
                ).strip()
                or None,
            )
            payload = await review_lane(
                manifest=manifest,
                lane_id=lane_id,
                artifact=artifact,
                run_dict=run_dict,
                tier=tier,
                repo_root=self.repo_root,
            )
            if isinstance(payload, dict):
                payload = {"lane_id": lane_id, **payload}
                events.append(_event_summary("review", payload))
            return payload

        async def _watch_integrate_fn(*, manifest, lane_id, artifact, approve, run_state=None):
            if artifact is None:
                payload = {
                    "lane_id": lane_id,
                    "recommendation": "needs_human",
                    "executed": False,
                    "rationale": "Missing tranche artifact.",
                }
                events.append(_event_summary("integrate", payload))
                return payload
            payload = await integrate_lane(
                artifact=artifact,
                manifest=manifest,
                approve=bool(approve),
                repo_root=self.repo_root,
                github=github,
                registry=registry,
                store=DevCoordinationStore(repo_root=self.repo_root),
                target_branch=self.target_branch,
                decided_by="tranche-queue",
                rationale="Tranche queue approved merge after green checks and review.",
                run_state=run_state,
                autonomy_mode=str(item_state.effective_autonomy_mode or "adaptive"),
            )
            if isinstance(payload, dict):
                payload = {"lane_id": lane_id, **payload}
                events.append(_event_summary("integrate", payload))
            return payload

        try:
            current = claim_driver(current, session_id=session_id)
            current.save(state_path)
        except DriverAlreadyClaimedError as exc:
            item_state.status = QUEUE_ITEM_STATUS_STOPPED
            item_state.stop_reason = "driver_already_claimed"
            item_state.finished_at = _utcnow()
            self._append_finding(item_state, str(exc))
            item_state.tranche_status = current.status
            return "driver_already_claimed"

        systemic_reason: str | None = None
        try:
            while True:
                if _utcnow() >= deadline:
                    systemic_reason = "time_limit_exceeded"
                    item_state.status = QUEUE_ITEM_STATUS_STOPPED
                    item_state.stop_reason = systemic_reason
                    item_state.tranche_status = current.status
                    item_state.finished_at = _utcnow()
                    break
                current = await watch_loop(
                    current,
                    manifest=tranche_manifest,
                    interval_seconds=self.interval_seconds,
                    max_ticks=1,
                    state_path=state_path,
                    driver_session_id=session_id,
                    artifact_store=artifact_store,
                    repo_root=self.repo_root,
                    run_fn=_watch_run_fn,
                    review_fn=_watch_review_fn,
                    integrate_fn=_watch_integrate_fn,
                )
                item_state.tranche_status = current.status
                if current.status in {
                    TRANCHE_STATUS_ABORTED,
                    TRANCHE_STATUS_COMPLETED,
                    TRANCHE_STATUS_NEEDS_HUMAN,
                }:
                    break
                if self.interval_seconds > 0:
                    await asyncio.sleep(self.interval_seconds)
        except Exception as exc:
            logger.warning("tranche queue watch failed for %s: %s", item.item_id, exc)
            item_state.status = QUEUE_ITEM_STATUS_STOPPED
            item_state.stop_reason = "watch_failure"
            item_state.finished_at = _utcnow()
            item_state.tranche_status = current.status
            self._append_finding(item_state, "Queue watch driver failed. Check logs for detail.")
            item_state.result["watch_error"] = {"error_type": type(exc).__name__}
            systemic_reason = "watch_failure"
        finally:
            try:
                released = release_driver(current, session_id=session_id)
                released.save(state_path)
                current = released
            except Exception as exc:
                logger.warning("tranche queue release driver failed for %s: %s", item.item_id, exc)

        if current.status in {
            TRANCHE_STATUS_ABORTED,
            TRANCHE_STATUS_COMPLETED,
            TRANCHE_STATUS_NEEDS_HUMAN,
        }:
            await self._publish_terminal_lane_deliverables(
                current=current,
                tranche_manifest=tranche_manifest,
                artifact_store=artifact_store,
                github=github,
                registry=registry,
                state_path=state_path,
                item_state=item_state,
                events=events,
            )

        if item_state.status not in _QUEUE_ITEM_TERMINAL_STATUSES:
            if current.status == TRANCHE_STATUS_COMPLETED:
                item_state.status = QUEUE_ITEM_STATUS_COMPLETED
            else:
                item_state.status = QUEUE_ITEM_STATUS_NEEDS_HUMAN
                item_state.stop_reason = current.status
        item_state.tranche_status = current.status
        item_state.finished_at = item_state.finished_at or _utcnow()
        item_state.updated_at = _utcnow()
        item_state.events = _truncate_events(events)

        pr_urls = [
            url
            for lane_state in current.lane_states.values()
            if (url := _optional_text(getattr(lane_state, "pr_url", None)))
        ]
        item_state.pr_urls = list(dict.fromkeys(pr_urls))
        for finding in self._findings_from_events(events):
            self._append_finding(item_state, finding)
        item_state.result["tranche"] = {
            "status": current.status,
            "lane_states": {
                lane_id: lane_state.to_dict()
                for lane_id, lane_state in sorted(current.lane_states.items())
            },
        }

        return systemic_reason or self._classify_systemic_reason(events, item_state)

    async def _publish_terminal_lane_deliverables(
        self,
        *,
        current: Any,
        tranche_manifest: Any,
        artifact_store: TrancheArtifactStore | Any,
        github: Any,
        registry: Any,
        state_path: Path,
        item_state: TrancheQueueItemRunState,
        events: list[dict[str, Any]],
    ) -> None:
        updated = False
        for lane_id, lane_state in current.lane_states.items():
            if _optional_text(getattr(lane_state, "pr_url", None)):
                continue
            artifact = self._load_artifact_for_lane(artifact_store, current.manifest_id, lane_id)
            if artifact is None or not self._artifact_has_publishable_deliverable(artifact):
                continue
            payload = await integrate_lane(
                artifact=artifact,
                manifest=tranche_manifest,
                approve=False,
                repo_root=self.repo_root,
                github=github,
                registry=registry,
                store=DevCoordinationStore(repo_root=self.repo_root),
                target_branch=self.target_branch,
                decided_by="tranche-queue-terminal-publish",
                rationale=(
                    "Tranche queue published a completed lane deliverable after watch "
                    "terminated without recording a PR."
                ),
                run_state=current,
                autonomy_mode=str(item_state.effective_autonomy_mode or "adaptive"),
            )
            if not isinstance(payload, dict):
                continue
            payload = {"lane_id": lane_id, **payload}
            events.append(_event_summary("integrate", payload))
            pr_url = _optional_text(payload.get("pr_url"))
            if pr_url:
                lane_state.pr_url = pr_url
                updated = True
        if updated:
            current.updated_at = _utcnow()
            current.save(state_path)

    @staticmethod
    def _artifact_has_publishable_deliverable(artifact: Any) -> bool:
        metadata = getattr(artifact, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        deliverable = metadata.get("deliverable", {})
        if not isinstance(deliverable, dict):
            deliverable = {}
        branch = _optional_text(deliverable.get("branch")) or _optional_text(metadata.get("branch"))
        commit_shas = [
            str(item).strip() for item in deliverable.get("commit_shas", []) if str(item).strip()
        ]
        return bool(branch and commit_shas)

    @staticmethod
    def _load_artifact_for_lane(
        artifact_store: TrancheArtifactStore | Any,
        manifest_id: str,
        lane_id: str,
    ) -> Any | None:
        if hasattr(artifact_store, "load"):
            artifact = artifact_store.load(manifest_id, lane_id)
            if artifact is not None:
                return artifact
        if hasattr(artifact_store, "list"):
            for artifact in artifact_store.list(manifest_id):
                if getattr(artifact, "lane_id", None) == lane_id:
                    return artifact
        return None

    def _classify_systemic_reason(
        self,
        events: list[dict[str, Any]],
        item_state: TrancheQueueItemRunState,
    ) -> str | None:
        for event in events:
            if event.get("type") == "review":
                findings = [
                    str(item).strip().lower()
                    for item in event.get("findings", [])
                    if str(item).strip()
                ]
                if any(
                    marker in finding for finding in findings for marker in _SYSTEMIC_REVIEW_MARKERS
                ):
                    item_state.stop_reason = "reviewer_routing_unavailable"
                    return "reviewer_routing_unavailable"
            if event.get("type") == "integrate":
                publish_result = event.get("publish_result")
                if isinstance(publish_result, dict):
                    action = str(publish_result.get("action", "")).strip().lower()
                    if action in _SYSTEMIC_PUBLISH_ACTIONS:
                        item_state.stop_reason = "controller_publication_unavailable"
                        return "controller_publication_unavailable"
        return None

    def _findings_from_events(self, events: list[dict[str, Any]]) -> list[str]:
        findings: list[str] = []
        for event in events:
            if event.get("type") == "review":
                findings.extend(_string_list(event.get("findings")))
            if event.get("type") == "integrate":
                publish_result = event.get("publish_result")
                if isinstance(publish_result, dict):
                    detail = _optional_text(publish_result.get("detail"))
                    if detail:
                        findings.append(detail)
        return list(dict.fromkeys(findings))

    @staticmethod
    def _append_finding(item_state: TrancheQueueItemRunState, finding: str) -> None:
        text = str(finding or "").strip()
        if not text:
            return
        if text not in item_state.findings:
            item_state.findings.append(text)


async def run_tranche_queue(
    *,
    queue_path: str | Path,
    repo_root: str | Path,
    target_branch: str = "main",
    interval_seconds: float = 5.0,
    max_hours: float = 12.0,
    max_consecutive_failures: int = 3,
    planner_model: str = "claude",
    planner_strategy: str = "heuristic",
    worker_model: str = "codex",
    review_model: str = "claude",
    enforce_cross_model_review: bool = True,
    max_parallel_lanes: int = 1,
) -> dict[str, Any]:
    executor = TrancheQueueExecutor(
        queue_path=queue_path,
        repo_root=repo_root,
        target_branch=target_branch,
        interval_seconds=interval_seconds,
        max_hours=max_hours,
        max_consecutive_failures=max_consecutive_failures,
        planner_model=planner_model,
        planner_strategy=planner_strategy,
        worker_model=worker_model,
        review_model=review_model,
        enforce_cross_model_review=enforce_cross_model_review,
        max_parallel_lanes=max_parallel_lanes,
    )
    return await executor.run()


def reconcile_tranche_queue(
    *,
    queue_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    executor = TrancheQueueExecutor(
        queue_path=queue_path,
        repo_root=repo_root,
    )
    return executor.reconcile()


def harvest_tranche_queue(
    *,
    queue_path: str | Path,
    repo_root: str | Path,
    execute_merge: bool = False,
    allow_admin: bool = False,
) -> dict[str, Any]:
    resolved_queue_path = Path(queue_path).resolve()
    resolved_repo_root = Path(repo_root).resolve()
    reconciled = reconcile_tranche_queue(
        queue_path=resolved_queue_path, repo_root=resolved_repo_root
    )
    manifest = TrancheQueueManifest.load(resolved_queue_path)
    state_path = queue_state_path_for_queue(resolved_queue_path)
    state = TrancheQueueRunState.load(state_path, manifest=manifest)
    github = GitHubControl(repo_root=resolved_repo_root)

    pr_counts: dict[str, int] = {}
    executed_merges: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    for item in manifest.items:
        item_state = state.item_states[item.item_id]
        pr_records: list[dict[str, Any]] = []
        for pr_url in item_state.pr_urls:
            normalized_pr_url = str(pr_url or "").strip()
            if not normalized_pr_url:
                continue
            try:
                snapshot = github.fetch_gate_snapshot(normalized_pr_url)
                pr_record = snapshot.to_dict()
                pr_record["snapshot_disposition"] = snapshot.disposition
                disposition = snapshot.disposition
                if execute_merge and snapshot.disposition == "merge_now":
                    merge_result = github.merge_pr(
                        normalized_pr_url,
                        required_checks_green=snapshot.required_checks_green,
                        allow_admin=allow_admin,
                    )
                    pr_record["merge_result"] = merge_result.to_dict()
                    disposition = "merged" if merge_result.merged else "merge_failed"
                    pr_record["disposition"] = disposition
                    if merge_result.merged:
                        executed_merges.append(
                            {
                                "item_id": item.item_id,
                                "pr_url": normalized_pr_url,
                                "branch": snapshot.head_branch,
                                "action": merge_result.action,
                                "used_admin": merge_result.used_admin,
                            }
                        )
                pr_counts[disposition] = pr_counts.get(disposition, 0) + 1
            except Exception as exc:
                pr_record = {
                    "pr_url": normalized_pr_url,
                    "disposition": "error",
                    "error_type": type(exc).__name__,
                    "detail": str(exc).strip() or type(exc).__name__,
                }
                pr_counts["error"] = pr_counts.get("error", 0) + 1
            pr_records.append(pr_record)

        items.append(
            {
                "item_id": item.item_id,
                "kind": item.kind,
                "source": item.source,
                "status": item_state.status,
                "merge_class": item.merge_class,
                "tranche_status": item_state.tranche_status,
                "pr_urls": list(item_state.pr_urls),
                "findings": list(item_state.findings),
                "stop_reason": item_state.stop_reason,
                "prs": pr_records,
            }
        )

    return {
        "mode": "tranche-queue-harvest",
        "queue_id": manifest.queue_id,
        "queue_path": str(resolved_queue_path),
        "state_path": str(state_path),
        "status": state.status,
        "stop_reason": state.stop_reason,
        "current_item_id": state.current_item_id,
        "counts": dict(reconciled.get("counts") or {}),
        "pr_counts": pr_counts,
        "execute_merge": bool(execute_merge),
        "allow_admin": bool(allow_admin),
        "executed_merges": executed_merges,
        "items": items,
    }
