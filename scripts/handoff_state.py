#!/usr/bin/env python3
"""Read-only handoff state classifier for Aragora automation outbox items."""

from __future__ import annotations

import dataclasses
import ast
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:  # Works both as ``python scripts/handoff_state.py`` and as ``scripts.handoff_state``.
    from scripts.github_cli_health import github_cli_env
except Exception:  # pragma: no cover - script-path fallback
    try:
        from github_cli_health import github_cli_env  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - partially bootstrapped fallback

        def github_cli_env(
            base_env: Mapping[str, str] | None = None,
            *,
            prefer_app: bool = True,
        ) -> dict[str, str]:
            return dict(os.environ if base_env is None else base_env)


UTC = timezone.utc
SCHEMA_VERSION = "aragora-handoff-state/1.0"

DEFAULT_REPO_ROOT = Path(".")
DEFAULT_OUTBOX_DIR = Path(".aragora/automation-outbox")
DEFAULT_RECEIPT_DIR = Path(".aragora/automation-receipts")
DEFAULT_STATUS_CACHE = Path(".aragora/automation-github-status/latest.json")
DEFAULT_LANE_REGISTRY = Path(".aragora/agent-bridge/lanes.json")
DEFAULT_STEERING_ROOT = Path(".aragora/operator-steering")
DEFAULT_HEARTBEATS = Path(".aragora/agent-bridge/heartbeats.json")
DEFAULT_QUEUE_CAP_CACHE_MAX_AGE_SECONDS = 1800

TERMINAL_RECEIPT_STATUSES = {"published", "already_satisfied", "completed", "skipped"}
TERMINAL_STEERING_OUTCOMES = {"completed", "stale", "superseded"}
PR_PUBLICATION_ACTIONS = {
    "open_pr",
    "open_pull_request",
    "open_or_update_pr",
    "open_or_update_pull_request",
    "push_branch_and_open_pr",
    "push_branch_and_open_pull_request",
    "push_branch_and_open_or_update_pr",
    "push_branch_and_open_or_update_pull_request",
}
PR_PUBLICATION_IDEMPOTENCY_PREFIXES = ("open-pr-", "update-pr-")


class HandoffState(str, Enum):
    PUBLICATION_REQUESTED = "publication_requested"
    REPRESENTED_BY_EXACT_OPEN_PR = "represented_by_exact_open_pr"
    REPRESENTED_BY_EXACT_REMOTE_BRANCH = "represented_by_exact_remote_branch"
    BLOCKED_BY_OWNER = "blocked_by_owner"
    BLOCKED_BY_HUMAN = "blocked_by_human"
    BLOCKED_BY_POSSIBLE_UNPUSHED_WORK = "blocked_by_possible_unpushed_work"
    BLOCKED_BY_LIVE_QUEUE_CAP = "blocked_by_live_queue_cap"
    PRESERVED_NOT_ACTIONABLE = "preserved_not_actionable"
    UNKNOWN = "unknown"


@dataclass
class GitHubEvidence:
    mode: str
    error: str | None = None
    open_prs: list[dict[str, Any]] = field(default_factory=list)
    exact_open_pr: dict[str, Any] | None = None
    remote_ref: dict[str, Any] | None = None


@dataclass
class ReceiptEvidence:
    status: str | None = None
    reason: str | None = None
    has_pr_reference: bool = False
    has_issue_reference: bool = False
    issue_only_pr_receipt: bool = False
    path: str | None = None
    target_pr: int | None = None


@dataclass
class OwnerEvidence:
    available: bool = False
    matched: bool = False
    error: str | None = None
    lane_id: str | None = None
    owner_session: str | None = None
    source: str | None = None
    status: str | None = None
    owner_state: str | None = None
    owner_blocking_state: str | None = None
    owner_blocking_state_reason: str | None = None
    advisory_withheld: str | None = None
    stale_claim_available: bool | None = None
    payload: dict[str, Any] | None = None


@dataclass
class SteeringEvidence:
    pending_message_count: int = 0
    blocking_message_count: int = 0
    human_message_count: int = 0
    resolved_message_count: int = 0
    latest_message: dict[str, Any] | None = None
    latest_read_receipt: dict[str, Any] | None = None


@dataclass
class QueueCapEvidence:
    available: bool = False
    degraded: bool | None = None
    degraded_reason: str | None = None
    open_pr_cap_reached: bool | None = None
    raw_open_pr_cap_reached: bool | None = None
    open_codex_pr_count: int | None = None
    max_open_prs: int | None = None
    generated_at: str | None = None
    cache_age_seconds: float | None = None
    cache_stale: bool | None = None
    cache_stale_threshold_seconds: int | None = None
    decision_source: str | None = None


@dataclass
class HandoffClassification:
    outbox_file: str
    idempotency_key: str | None
    branch: str | None
    desired_head_sha: str | None
    state: HandoffState
    reason: str
    evidence: dict[str, Any]
    next_mutation_candidate: str = "none"
    safe_to_mutate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "outbox_file": self.outbox_file,
            "idempotency_key": self.idempotency_key,
            "branch": self.branch,
            "desired_head_sha": self.desired_head_sha,
            "state": self.state.value,
            "reason": self.reason,
            "evidence": self.evidence,
            "next_mutation_candidate": self.next_mutation_candidate,
            "safe_to_mutate": self.safe_to_mutate,
        }


class NarrowGitHubClient:
    """Narrow REST-only GitHub reader.

    This intentionally uses exact branch/ref endpoints and never calls GraphQL
    or mutating gh commands.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        github_repo: str,
        disabled: bool = False,
        timeout_seconds: int = 20,
        known_open_pr_heads: set[str] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.github_repo = github_repo
        self.disabled = disabled
        self.timeout_seconds = timeout_seconds
        self.known_open_pr_heads = known_open_pr_heads
        self._pr_cache: dict[str, tuple[list[dict[str, Any]] | None, str | None]] = {}
        self._ref_cache: dict[str, tuple[dict[str, Any] | None, str | None]] = {}

    @property
    def mode(self) -> str:
        if self.disabled:
            return "disabled"
        return "ready"

    def open_prs_for_branch(self, branch: str) -> tuple[list[dict[str, Any]] | None, str | None]:
        if self.disabled:
            return None, "github disabled"
        if branch in self._pr_cache:
            return self._pr_cache[branch]
        owner = self.github_repo.split("/", 1)[0]
        head = f"{owner}:{quote(branch, safe='')}"
        endpoint = f"repos/{self.github_repo}/pulls?state=open&head={head}&per_page=5"
        payload, error = self._api(endpoint)
        if error is not None:
            result = (None, error)
        elif not isinstance(payload, list):
            result = (None, "open PR REST response was not a list")
        else:
            result = ([item for item in payload if isinstance(item, dict)], None)
        self._pr_cache[branch] = result
        return result

    def remote_ref(self, branch: str) -> tuple[dict[str, Any] | None, str | None]:
        if self.disabled:
            return None, "github disabled"
        if branch in self._ref_cache:
            return self._ref_cache[branch]
        endpoint = f"repos/{self.github_repo}/git/ref/heads/{quote(branch, safe='/')}"
        payload, error = self._api(endpoint)
        if error is not None and "404" in error:
            result = (None, None)
        elif error is not None:
            result = (None, error)
        elif not isinstance(payload, dict):
            result = (None, "ref REST response was not a mapping")
        else:
            result = (payload, None)
        self._ref_cache[branch] = result
        return result

    def _api(self, endpoint: str) -> tuple[Any | None, str | None]:
        try:
            proc = subprocess.run(
                ["gh", "api", endpoint],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                env=github_cli_env(os.environ),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"gh api failed ({exc.__class__.__name__})"
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip().splitlines()
            return None, f"gh api exited {proc.returncode}: {detail[0] if detail else ''}"
        try:
            return json.loads(proc.stdout), None
        except json.JSONDecodeError:
            return None, "gh api returned unparseable JSON"


class OwnerProbe:
    """Read-only owner/liveness probe via the existing supported helper."""

    def __init__(self, *, repo_root: Path, state_root: Path, timeout_seconds: int = 20) -> None:
        self.repo_root = repo_root
        self.state_root = _as_aragora_root(state_root)
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, OwnerEvidence] = {}

    def probe(self, branch: str) -> OwnerEvidence:
        if branch in self._cache:
            return self._cache[branch]
        registry = self.state_root / "agent-bridge" / "lanes.json"
        steering_root = self.state_root / "operator-steering"
        heartbeat_path = self.state_root / "agent-bridge" / "heartbeats.json"
        script = self.repo_root / "scripts" / "identify_lane_owner.py"
        cmd = [
            sys.executable,
            str(script),
            "--branch",
            branch,
            "--liveness",
            "--json",
            "--registry-path",
            str(registry),
            "--steering-inbox-root",
            str(steering_root),
            "--heartbeat-path",
            str(heartbeat_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = OwnerEvidence(
                available=False, error=f"owner probe failed ({exc.__class__.__name__})"
            )
            self._cache[branch] = result
            return result
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or "").strip().splitlines()
            text = message[0] if message else f"identify_lane_owner exited {proc.returncode}"
            matched = "no lane matched" not in text.lower()
            result = OwnerEvidence(
                available=True,
                matched=matched,
                error=text,
            )
            self._cache[branch] = result
            return result
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            result = OwnerEvidence(available=False, error="owner probe returned unparseable JSON")
            self._cache[branch] = result
            return result
        if not isinstance(payload, dict):
            result = OwnerEvidence(available=False, error="owner probe returned non-mapping JSON")
            self._cache[branch] = result
            return result
        result = owner_evidence_from_payload(payload)
        self._cache[branch] = result
        return result


class LaneRegistryOwnerProbe:
    """Fast read-only owner probe from the shared lane registry.

    This is intentionally less deep than identify_lane_owner.py --liveness, but
    it is enough to keep whole-outbox classification bounded and deterministic.
    Focused runs can still opt into the supported liveness helper.
    """

    def __init__(self, *, state_root: Path) -> None:
        self.state_root = _as_aragora_root(state_root)
        self.records = _load_lane_records(self.state_root / "agent-bridge" / "lanes.json")

    def probe(self, branch: str) -> OwnerEvidence:
        candidates = [row for row in self.records if str(row.get("branch") or "") == branch]
        if not candidates:
            return OwnerEvidence(available=True, matched=False, error="no lane matched")
        row = _best_lane_record(candidates)
        status = str(row.get("status") or "").strip().lower()
        blocking_state = None
        if status in {"active", "blocked", "blocked_on_publication", "claimed"}:
            blocking_state = "live_owner"
        evidence = owner_evidence_from_payload(row)
        evidence.status = status or evidence.status
        evidence.owner_blocking_state = evidence.owner_blocking_state or blocking_state
        evidence.owner_blocking_state_reason = evidence.owner_blocking_state_reason or (
            "lane registry status is active/blocking" if blocking_state else None
        )
        evidence.advisory_withheld = evidence.advisory_withheld or _possible_unpushed_marker(row)
        return evidence


def owner_evidence_from_payload(payload: Mapping[str, Any]) -> OwnerEvidence:
    advisory = payload.get("stale_claim_advisory")
    available = None
    if isinstance(advisory, Mapping):
        available = bool(advisory.get("available"))
    owner_session = _first_text(payload, "owner_session", "session", "to_session")
    lane_id = _first_text(payload, "lane_id", "lane")
    return OwnerEvidence(
        available=True,
        matched=True,
        lane_id=lane_id,
        owner_session=owner_session,
        source=_first_text(payload, "source"),
        status=_first_text(payload, "status", "lane_status"),
        owner_state=_first_text(payload, "owner_state", "assessed"),
        owner_blocking_state=_first_text(payload, "owner_blocking_state"),
        owner_blocking_state_reason=_first_text(payload, "owner_blocking_state_reason"),
        advisory_withheld=_first_text(payload, "advisory_withheld"),
        stale_claim_available=available,
        payload=dict(payload),
    )


def _possible_unpushed_marker(payload: Mapping[str, Any]) -> str | None:
    text = json.dumps(payload, sort_keys=True).lower()
    return "possible_unpushed_work" if "possible_unpushed_work" in text else None


def classify_handoffs(
    *,
    repo_root: Path,
    state_root: Path | None = None,
    github_repo: str | None = None,
    outbox_file: str | Path | None = None,
    no_github: bool = False,
    owner_timeout_seconds: int = 20,
    with_liveness_helper: bool = False,
    queue_cache_max_age_seconds: int = DEFAULT_QUEUE_CAP_CACHE_MAX_AGE_SECONDS,
    github_client: Any | None = None,
    owner_probe: Any | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    state_root = _as_aragora_root(state_root or repo_root)
    github_repo = github_repo or _github_repo_from_origin(repo_root) or "synaptent/aragora"
    outbox_dir = _state_path(state_root, DEFAULT_OUTBOX_DIR)
    receipt_dir = _state_path(state_root, DEFAULT_RECEIPT_DIR)
    status_cache_path = _state_path(state_root, DEFAULT_STATUS_CACHE)

    queue_cap = load_queue_cap_evidence(
        status_cache_path,
        max_age_seconds=queue_cache_max_age_seconds,
    )
    known_open_pr_heads = None
    if not no_github:
        known_open_pr_heads = load_fresh_open_pr_head_cache(
            status_cache_path,
            max_age_seconds=queue_cache_max_age_seconds,
        )
    receipts = load_terminal_receipts(receipt_dir)
    outbox_files = _selected_outbox_files(outbox_dir, outbox_file)
    gh = github_client or NarrowGitHubClient(
        repo_root=repo_root,
        github_repo=github_repo,
        disabled=no_github,
        known_open_pr_heads=known_open_pr_heads,
    )
    if owner_probe is not None:
        owner = owner_probe
    elif with_liveness_helper:
        owner = OwnerProbe(
            repo_root=repo_root,
            state_root=state_root,
            timeout_seconds=owner_timeout_seconds,
        )
    else:
        owner = LaneRegistryOwnerProbe(state_root=state_root)

    items: list[HandoffClassification] = []
    github_errors: list[str] = []
    for path in outbox_files:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            items.append(
                HandoffClassification(
                    outbox_file=path.name,
                    idempotency_key=path.stem,
                    branch=None,
                    desired_head_sha=None,
                    state=HandoffState.UNKNOWN,
                    reason="outbox file is missing or unparseable",
                    evidence={},
                )
            )
            continue
        item = classify_handoff_item(
            path=path,
            payload=payload,
            receipt=receipts.get(str(payload.get("idempotency_key") or path.stem).strip()),
            queue_cap=queue_cap,
            github_client=gh,
            owner_probe=owner,
            state_root=state_root,
        )
        gh_error = item.evidence.get("github", {}).get("error")
        if isinstance(gh_error, str) and gh_error:
            github_errors.append(gh_error)
        items.append(item)

    counts = Counter(item.state.value for item in items)
    github_mode = "disabled" if no_github else ("degraded" if github_errors else "ready")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "repo": str(repo_root),
        "state_root": str(state_root),
        "github_repo": github_repo,
        "outbox_dir": str(outbox_dir),
        "receipt_dir": str(receipt_dir),
        "outbox_count": len(items),
        "counts": dict(sorted(counts.items())),
        "github": {
            "mode": github_mode,
            "error": github_errors[0] if github_errors else None,
        },
        "queue_cap": dataclasses.asdict(queue_cap),
        "items": [item.to_dict() for item in items],
    }
    return payload


def classify_handoff_item(
    *,
    path: Path,
    payload: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
    queue_cap: QueueCapEvidence,
    github_client: Any,
    owner_probe: Any,
    state_root: Path,
) -> HandoffClassification:
    idem = str(payload.get("idempotency_key") or path.stem).strip() or path.stem
    branch = branch_from_payload(payload)
    desired_head = desired_head_from_payload(payload)
    receipt_evidence = receipt_evidence_from_payload(receipt, payload)

    evidence: dict[str, Any] = {
        "receipt": dataclasses.asdict(receipt_evidence),
        "queue_cap": dataclasses.asdict(queue_cap),
    }
    if not branch:
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=None,
            desired_head_sha=desired_head or None,
            state=HandoffState.UNKNOWN,
            reason="outbox payload has no branch",
            evidence=evidence,
        )

    github = github_evidence_for_branch(github_client, branch, desired_head)
    evidence["github"] = dataclasses.asdict(github)
    owner = owner_probe.probe(branch) if branch else OwnerEvidence()
    evidence["owner"] = dataclasses.asdict(owner)
    steering = steering_evidence_for_branch(
        state_root=state_root,
        branch=branch,
        owner_session=owner.owner_session,
        lane_id=owner.lane_id,
    )
    evidence["steering"] = dataclasses.asdict(steering)
    if github.exact_open_pr is not None:
        number = github.exact_open_pr.get("number")
        mutation_safe = not (
            _possible_unpushed(owner)
            or _human_blocked(owner, steering)
            or _owner_blocked(owner, steering)
        )
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.REPRESENTED_BY_EXACT_OPEN_PR,
            reason=f"branch has exact-head open PR #{number}",
            evidence=evidence,
            next_mutation_candidate="write_representation_receipt_then_archive",
            safe_to_mutate=mutation_safe,
        )

    if _remote_ref_matches(github.remote_ref, desired_head) and not _possible_unpushed(owner):
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.REPRESENTED_BY_EXACT_REMOTE_BRANCH,
            reason="desired head is preserved by exact remote branch",
            evidence=evidence,
            next_mutation_candidate="represent_or_publish_remote_branch",
        )

    if _possible_unpushed(owner):
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.BLOCKED_BY_POSSIBLE_UNPUSHED_WORK,
            reason="owner liveness withholds advisory because possible unpushed work exists",
            evidence=evidence,
            next_mutation_candidate="owner_preservation_request",
        )

    if _human_blocked(owner, steering):
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.BLOCKED_BY_HUMAN,
            reason="handoff is gated by a human owner or human-directed steering",
            evidence=evidence,
            next_mutation_candidate="human_gate",
        )

    if _owner_blocked(owner, steering):
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.BLOCKED_BY_OWNER,
            reason="handoff has an owner/lane blocker and no exact representation proof",
            evidence=evidence,
            next_mutation_candidate="owner_followup",
        )

    if receipt_evidence.issue_only_pr_receipt:
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.PUBLICATION_REQUESTED,
            reason="PR-intended handoff has issue-only receipt; PR representation still requested",
            evidence=evidence,
            next_mutation_candidate="publish_or_represent_pr",
        )

    if queue_cap.open_pr_cap_reached and is_pr_publication_request(payload):
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP,
            reason="publication requested but live cache reports open PR cap reached",
            evidence=evidence,
            next_mutation_candidate="queue_drain",
        )

    if is_pr_publication_request(payload):
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.PUBLICATION_REQUESTED,
            reason="PR publication remains requested and no stronger representation/blocker matched",
            evidence=evidence,
            next_mutation_candidate="publish_or_represent_pr",
        )

    if github.remote_ref is not None:
        return HandoffClassification(
            outbox_file=path.name,
            idempotency_key=idem,
            branch=branch,
            desired_head_sha=desired_head or None,
            state=HandoffState.PRESERVED_NOT_ACTIONABLE,
            reason="remote branch exists but exact desired-head representation is not proven",
            evidence=evidence,
        )

    return HandoffClassification(
        outbox_file=path.name,
        idempotency_key=idem,
        branch=branch,
        desired_head_sha=desired_head or None,
        state=HandoffState.UNKNOWN,
        reason="no representation, owner, cap, or publication state could be proven",
        evidence=evidence,
    )


def github_evidence_for_branch(
    github_client: Any,
    branch: str,
    desired_head: str,
) -> GitHubEvidence:
    open_prs, pr_error = github_client.open_prs_for_branch(branch)
    ref, ref_error = github_client.remote_ref(branch)
    exact_open_pr = None
    open_pr_items = open_prs or []
    if desired_head:
        for item in open_pr_items:
            head = item.get("head") if isinstance(item.get("head"), Mapping) else {}
            head_sha = str(head.get("sha") or item.get("head_sha") or item.get("headRefOid") or "")
            if heads_match(desired_head, head_sha):
                exact_open_pr = {
                    "number": item.get("number"),
                    "state": item.get("state"),
                    "draft": item.get("draft"),
                    "head": head.get("ref") or item.get("head_ref"),
                    "head_sha": head_sha,
                    "html_url": item.get("html_url") or item.get("url"),
                }
                break
    remote_ref = None
    if ref is not None:
        obj = ref.get("object") if isinstance(ref.get("object"), Mapping) else {}
        remote_ref = {
            "ref": ref.get("ref"),
            "sha": obj.get("sha") or ref.get("sha"),
        }
    errors = "; ".join(error for error in (pr_error, ref_error) if error)
    if getattr(github_client, "disabled", False):
        mode = "disabled"
    elif errors:
        mode = "degraded"
    else:
        mode = "ready"
    return GitHubEvidence(
        mode=mode,
        error=errors or None,
        open_prs=open_pr_items,
        exact_open_pr=exact_open_pr,
        remote_ref=remote_ref,
    )


def load_terminal_receipts(receipt_dir: Path) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    if not receipt_dir.exists():
        return receipts
    for path in sorted(receipt_dir.glob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status not in TERMINAL_RECEIPT_STATUSES:
            continue
        key = str(payload.get("idempotency_key") or path.stem).strip()
        if key:
            payload = dict(payload)
            payload["__receipt_path"] = str(path)
            receipts[key] = payload
    return receipts


def load_queue_cap_evidence(
    path: Path,
    *,
    max_age_seconds: int = DEFAULT_QUEUE_CAP_CACHE_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> QueueCapEvidence:
    payload = _load_json(path)
    if isinstance(payload, list):
        payload = next((item for item in reversed(payload) if isinstance(item, dict)), None)
    if not isinstance(payload, dict):
        return QueueCapEvidence(available=False)
    github_queue = (
        payload.get("github_queue") if isinstance(payload.get("github_queue"), Mapping) else {}
    )
    limits = payload.get("limits") if isinstance(payload.get("limits"), Mapping) else {}
    pressure = (
        github_queue.get("pressure") if isinstance(github_queue.get("pressure"), Mapping) else {}
    )
    generated_at = str(payload.get("generated_at") or "") or None
    generated_dt = _parse_datetime(generated_at)
    cache_age_seconds = None
    cache_stale = None
    decision_source = "fresh_cache"
    if generated_dt is None:
        cache_stale = True
        decision_source = "cache_missing_or_invalid_generated_at"
    else:
        current = now or datetime.now(UTC)
        cache_age_seconds = max(0.0, (current - generated_dt).total_seconds())
        cache_stale = cache_age_seconds > max_age_seconds
        if cache_stale:
            decision_source = "expired_cache"
    raw_cap = _bool_or_none(pressure.get("open_pr_cap_reached"))
    effective_cap = None if cache_stale else raw_cap
    degraded = _bool_or_none(github_queue.get("degraded"))
    if not cache_stale and degraded:
        decision_source = "fresh_degraded_cache_rest_fallback"
    return QueueCapEvidence(
        available=True,
        degraded=degraded,
        degraded_reason=str(github_queue.get("degraded_reason") or "") or None,
        open_pr_cap_reached=effective_cap,
        raw_open_pr_cap_reached=raw_cap,
        open_codex_pr_count=_int_or_none(github_queue.get("open_codex_pr_count")),
        max_open_prs=_int_or_none(limits.get("max_open_prs")),
        generated_at=generated_at,
        cache_age_seconds=cache_age_seconds,
        cache_stale=cache_stale,
        cache_stale_threshold_seconds=max_age_seconds,
        decision_source=decision_source,
    )


def load_fresh_open_pr_head_cache(
    path: Path,
    *,
    max_age_seconds: int = DEFAULT_QUEUE_CAP_CACHE_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> set[str] | None:
    """Return fresh cached open PR head refs, or None when cache cannot be trusted."""
    payload = _load_json(path)
    if isinstance(payload, list):
        payload = next((item for item in reversed(payload) if isinstance(item, dict)), None)
    if not isinstance(payload, dict):
        return None
    github_queue = (
        payload.get("github_queue") if isinstance(payload.get("github_queue"), Mapping) else {}
    )
    raw_heads = github_queue.get("open_pr_heads")
    if not isinstance(raw_heads, Sequence) or isinstance(raw_heads, (str, bytes, bytearray)):
        return None
    heads = {str(head).strip() for head in raw_heads if str(head).strip()}
    open_count = _int_or_none(github_queue.get("open_codex_pr_count"))
    if open_count is not None and open_count != len(heads):
        return None
    generated_at = str(
        github_queue.get("open_pr_heads_cached_at") or payload.get("generated_at") or ""
    )
    generated_dt = _parse_datetime(generated_at)
    if generated_dt is None:
        return None
    current = now or datetime.now(UTC)
    if max(0.0, (current - generated_dt).total_seconds()) > max_age_seconds:
        return None
    return heads


def receipt_evidence_from_payload(
    receipt: Mapping[str, Any] | None,
    outbox_payload: Mapping[str, Any],
) -> ReceiptEvidence:
    if receipt is None:
        return ReceiptEvidence()
    status = str(receipt.get("status") or "").strip().lower() or None
    reason = str(receipt.get("reason") or "").strip().lower() or None
    has_pr = receipt_has_pr_reference(receipt)
    has_issue = receipt_has_issue_reference(receipt)
    issue_only = (
        is_pr_publication_request(outbox_payload)
        and status in {"already_satisfied", "published"}
        and not has_pr
        and (reason in {"published", "existing_issue", "created_issue"} or has_issue)
    )
    return ReceiptEvidence(
        status=status,
        reason=reason,
        has_pr_reference=has_pr,
        has_issue_reference=has_issue,
        issue_only_pr_receipt=issue_only,
        path=str(receipt.get("__receipt_path") or "") or None,
        target_pr=target_pr_number_from_receipt(receipt),
    )


def steering_evidence_for_branch(
    *,
    state_root: Path,
    branch: str,
    owner_session: str | None,
    lane_id: str | None,
) -> SteeringEvidence:
    root = _state_path(state_root, DEFAULT_STEERING_ROOT)
    if not root.exists():
        return SteeringEvidence()
    matches: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        raw = json.dumps(payload, sort_keys=True)
        to_session = str(payload.get("to_session") or "")
        lane_hint = str(payload.get("lane_id_hint") or "")
        if owner_session and to_session == owner_session:
            pass
        elif lane_id and lane_id in lane_hint:
            pass
        elif branch and branch in raw:
            pass
        else:
            continue
        receipt = latest_read_receipt_for_message(path)
        resolved = _steering_message_resolved(receipt)
        matches.append(
            {
                "path": str(path),
                "to_session": to_session or None,
                "lane_id_hint": lane_hint or None,
                "priority": str(payload.get("priority") or "").lower() or None,
                "subject": payload.get("subject"),
                "sent_at_utc": payload.get("sent_at_utc"),
                "from": payload.get("from"),
                "latest_read_receipt": receipt,
                "resolved_by_read_receipt": resolved,
            }
        )
    unresolved = [m for m in matches if not m.get("resolved_by_read_receipt")]
    blocking = [m for m in unresolved if m.get("priority") == "blocking"]
    human = [m for m in unresolved if _looks_human(m)]
    latest = (
        sorted(unresolved or matches, key=lambda m: str(m.get("sent_at_utc") or ""))[-1]
        if matches
        else None
    )
    receipts = [
        receipt
        for receipt in (m.get("latest_read_receipt") for m in matches)
        if isinstance(receipt, dict)
    ]
    latest_receipt = (
        sorted(receipts, key=lambda r: str(r.get("read_at_utc") or ""))[-1] if receipts else None
    )
    return SteeringEvidence(
        pending_message_count=len(matches),
        blocking_message_count=len(blocking),
        human_message_count=len(human),
        resolved_message_count=len(matches) - len(unresolved),
        latest_message=latest,
        latest_read_receipt=latest_receipt,
    )


def branch_from_payload(payload: Mapping[str, Any]) -> str:
    for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
        branch = str(local_evidence.get("branch") or "").strip()
        if branch:
            return branch
    branch = str(payload.get("branch") or "").strip()
    if branch:
        return branch
    requested_action = _mapping_from_action(payload.get("requested_action"))
    if requested_action is not None:
        return str(requested_action.get("branch") or "").strip()
    return ""


def desired_head_from_payload(payload: Mapping[str, Any]) -> str:
    for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
        head = str(
            local_evidence.get("desired_head_sha")
            or local_evidence.get("head_sha")
            or local_evidence.get("head")
            or local_evidence.get("commit")
            or ""
        ).strip()
        if head:
            return head
    for key in ("desired_head_sha", "head_sha", "head", "commit"):
        head = str(payload.get(key) or "").strip()
        if head:
            return head
    requested_action = _mapping_from_action(payload.get("requested_action"))
    if requested_action is not None:
        for key in ("desired_head_sha", "head_sha", "head", "commit"):
            head = str(requested_action.get(key) or "").strip()
            if head:
                return head
    return ""


def is_pr_publication_request(payload: Mapping[str, Any]) -> bool:
    action = requested_action_type(payload)
    idempotency_key = str(payload.get("idempotency_key") or "")
    return action in PR_PUBLICATION_ACTIONS or idempotency_key.startswith(
        PR_PUBLICATION_IDEMPOTENCY_PREFIXES
    )


def requested_action_type(payload: Mapping[str, Any]) -> str:
    requested_action = payload.get("requested_action")
    mapping = _mapping_from_action(requested_action)
    if mapping is not None:
        return str(mapping.get("type") or mapping.get("action") or "").strip().lower()
    if isinstance(requested_action, str):
        return requested_action.strip().lower()
    return ""


def receipt_has_pr_reference(receipt: Mapping[str, Any]) -> bool:
    for key in (
        "created_pr_url",
        "existing_pr_url",
        "pr_url",
        "pull_request_url",
        "created_pull_request_url",
        "existing_pull_request_url",
    ):
        if str(receipt.get(key) or "").strip():
            return True
    return False


def receipt_has_issue_reference(receipt: Mapping[str, Any]) -> bool:
    for key in ("created_issue_url", "existing_issue_url", "issue_url"):
        if str(receipt.get(key) or "").strip():
            return True
    return False


def target_pr_number_from_receipt(receipt: Mapping[str, Any]) -> int | None:
    for key in ("target_pr", "pr_number", "pull_request_number"):
        number = _pr_number_from_value(receipt.get(key))
        if number is not None:
            return number
    for key in (
        "created_pr_url",
        "existing_pr_url",
        "pr_url",
        "pull_request_url",
        "created_pull_request_url",
        "existing_pull_request_url",
    ):
        number = _pr_number_from_value(receipt.get(key))
        if number is not None:
            return number
    return None


def heads_match(expected: str, actual: str) -> bool:
    expected_value = str(expected or "").strip().lower()
    actual_value = str(actual or "").strip().lower()
    if len(expected_value) < 7 or len(actual_value) < 7:
        return False
    return actual_value.startswith(expected_value) or expected_value.startswith(actual_value)


def _remote_ref_matches(remote_ref: Mapping[str, Any] | None, desired_head: str) -> bool:
    if not remote_ref or not desired_head:
        return False
    return heads_match(desired_head, str(remote_ref.get("sha") or ""))


def _possible_unpushed(owner: OwnerEvidence) -> bool:
    value = str(owner.advisory_withheld or "").strip().lower()
    return value == "possible_unpushed_work"


def _human_blocked(owner: OwnerEvidence, steering: SteeringEvidence) -> bool:
    tokens = [
        owner.owner_session,
        owner.lane_id,
        owner.source,
        owner.status,
        owner.owner_blocking_state,
    ]
    if any("human" in str(token or "").lower() for token in tokens):
        return True
    return steering.human_message_count > 0


def _owner_blocked(owner: OwnerEvidence, steering: SteeringEvidence) -> bool:
    if steering.blocking_message_count > 0:
        return True
    status = str(owner.status or "").strip().lower()
    if status in {"active", "blocked", "blocked_on_publication", "claimed"}:
        return True
    blocking = str(owner.owner_blocking_state or "").strip().lower()
    if blocking in {"live_owner", "stale_owner", "unknown_owner", "stale_terminal_owner"}:
        return True
    return False


def _looks_human(mapping: Mapping[str, Any]) -> bool:
    text = json.dumps(mapping, sort_keys=True).lower()
    return "human" in text or "operator" in text


def latest_read_receipt_for_message(message_path: Path) -> dict[str, Any] | None:
    receipt_dir = message_path.parent / "_read_receipts"
    if not receipt_dir.is_dir():
        return None
    receipts: list[dict[str, Any]] = []
    data = _load_json(message_path)
    if not isinstance(data, dict):
        data = {}
    message_sha = str(data.get("message_sha256") or "")
    for receipt_path in sorted(receipt_dir.glob("*.json")):
        payload = _load_json(receipt_path)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("message_filename") or "") != message_path.name:
            continue
        receipt_sha = str(payload.get("message_sha256") or "")
        if message_sha and receipt_sha and receipt_sha != message_sha:
            continue
        receipts.append(
            {
                "path": str(receipt_path),
                "read_at_utc": payload.get("read_at_utc"),
                "read_by_session": payload.get("read_by_session"),
                "outcome": payload.get("outcome"),
                "outcome_note": payload.get("outcome_note"),
            }
        )
    if not receipts:
        return None
    return sorted(receipts, key=lambda r: str(r.get("read_at_utc") or ""))[-1]


def _steering_message_resolved(receipt: Mapping[str, Any] | None) -> bool:
    if not receipt:
        return False
    outcome = str(receipt.get("outcome") or "").strip().lower()
    return outcome in TERMINAL_STEERING_OUTCOMES


def _selected_outbox_files(outbox_dir: Path, outbox_file: str | Path | None) -> list[Path]:
    if outbox_file is None:
        if not outbox_dir.exists():
            return []
        return sorted(p for p in outbox_dir.iterdir() if p.is_file() and p.suffix == ".json")
    value = Path(outbox_file).expanduser()
    path = value if value.is_absolute() else outbox_dir / value
    return [path.resolve()]


def _load_lane_records(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, Mapping):
        records = payload.get("lanes") or payload.get("records")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


def _best_lane_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    active_statuses = {"active", "blocked", "blocked_on_publication", "claimed"}
    active = [
        row for row in records if str(row.get("status") or "").strip().lower() in active_statuses
    ]
    candidates = active or records
    return sorted(candidates, key=lambda row: str(row.get("updated_at") or ""), reverse=True)[0]


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mapping_from_action(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
    if isinstance(parsed, Mapping):
        return parsed
    return None


def _local_evidence_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _state_path(state_root: Path, default_relative: Path) -> Path:
    state_root = _as_aragora_root(state_root)
    if default_relative.parts[:1] == (".aragora",):
        return state_root.joinpath(*default_relative.parts[1:])
    return state_root / default_relative


def _as_aragora_root(path: Path) -> Path:
    expanded = path.expanduser().resolve()
    if expanded.name == ".aragora":
        return expanded
    return expanded / ".aragora"


def _github_repo_from_origin(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    if not value:
        return None
    if value.startswith("git@github.com:"):
        value = value.removeprefix("git@github.com:")
    elif "github.com/" in value:
        value = value.split("github.com/", 1)[1]
    value = value.removesuffix(".git").strip("/")
    parts = value.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return None


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = str(mapping.get(key) or "").strip()
        if value:
            return value
    return None


def _pr_number_from_value(value: Any) -> int | None:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return None
    if text.isdigit():
        return int(text)
    marker = "/pull/"
    if marker not in text:
        return None
    candidate = text.rsplit(marker, 1)[1].split("/", 1)[0]
    return int(candidate) if candidate.isdigit() else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def compact_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "repo": payload.get("repo"),
        "state_root": payload.get("state_root"),
        "github_repo": payload.get("github_repo"),
        "outbox_count": payload.get("outbox_count"),
        "counts": payload.get("counts", {}),
        "github": payload.get("github", {}),
        "queue_cap": payload.get("queue_cap", {}),
    }


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
