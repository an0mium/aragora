"""SpecUpgrader: convert weak GitHub-issue specs into dispatchable SwarmSpecs.

Public entry point: ``upgrade_spec()``. See
``docs/plans/2026-04-17-spec-upgrader-design.md`` for the architecture.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from aragora.swarm.spec import SwarmSpec

UpgradePath = Literal["deterministic", "llm", "deterministic+llm"]
UpgradeStatus = Literal["upgraded", "escalated"]


class SpecUpgraderUnavailable(Exception):
    """Raised for transient infrastructure failure (LLM 5xx, timeout, etc.).

    Callers should treat this as 'skip for this tick, retry next tick'.
    Does NOT consume an attempt in the durable counter.
    """


@dataclass(frozen=True)
class UpgradeFailureContext:
    """Structured input to the upgrader, explaining why the spec needs upgrading."""

    missing_bounds: list[str]
    preflight_diff: dict | None
    prior_attempts: int
    original_issue_body: str
    issue_title: str
    track_tag: str | None


@dataclass(frozen=True)
class UpgradeResult:
    """Outcome of an upgrade attempt. Tagged union via ``status`` field."""

    status: UpgradeStatus
    upgraded_spec: SwarmSpec | None
    audit_markdown: str
    attempt_count: int
    upgrade_path: UpgradePath | None
    failure_context: UpgradeFailureContext
    unresolved_questions: list[str] = field(default_factory=list)


# Labels emitted by ``SwarmSpec.missing_dispatch_bounds()`` mapped to actionable
# enrichment flags. Keep the keys in sync with ``aragora/swarm/spec.py`` -- in
# particular ``"explicit work order"`` matches the label returned by the spec
# (not ``"work order"``).
_BOUND_LABELS = {
    "acceptance criterion": "needs_acceptance",
    "file-scope hint": "needs_file_scope",
    "constraint": "needs_constraint",
    "explicit work order": "needs_work_order",
}


def _classify_missing_bounds(missing_bounds: list[str]) -> dict[str, bool]:
    """Map ``missing_dispatch_bounds()`` labels to actionable flags for enrichment."""
    classified = {flag: False for flag in _BOUND_LABELS.values()}
    for label in missing_bounds:
        flag = _BOUND_LABELS.get(label)
        if flag is not None:
            classified[flag] = True
    return classified


# Matches common Python/TS/MD file references. Intentionally narrow to avoid false
# positives.
_PATH_RE = re.compile(r"(?P<path>[a-zA-Z0-9_\-./]+\.(?:py|ts|tsx|js|jsx|md|yaml|yml|json|sh))")


def _extract_file_paths(issue_body: str, *, repo_root: Path) -> list[str]:
    """Extract file paths mentioned in the issue body and validate existence.

    Only paths that actually exist (relative to ``repo_root``) are returned. Paths
    that are hallucinated or merely aspirational are dropped.
    """
    candidates: set[str] = set()
    for match in _PATH_RE.finditer(issue_body):
        candidate = match.group("path").strip("./")
        if "/" in candidate and (repo_root / candidate).is_file():
            candidates.add(candidate)
    return sorted(candidates)


# Low-confidence candidate scopes per track-tag prefix. Must be validated against
# the current repo before merging into a spec.
_TRACK_SCOPE_CANDIDATES: dict[str, list[str]] = {
    "TW": ["aragora/swarm/"],
    "CS": ["aragora/swarm/", "docs/status/"],
    "RS": ["aragora/swarm/"],
}

# Design-heavy tracks must NOT use path inference; fall through to LLM or escalate.
_DESIGN_HEAVY_PREFIXES = frozenset({"AGT", "DIC"})


def _infer_track_scope(track_tag: str | None, *, issue_body: str, repo_root: Path) -> list[str]:
    """Return validated candidate scope hints for ``track_tag``, or ``[]`` to fall through."""
    if not track_tag:
        return []
    prefix = track_tag.split("-", 1)[0].upper()
    if prefix in _DESIGN_HEAVY_PREFIXES:
        return []
    candidates = _TRACK_SCOPE_CANDIDATES.get(prefix)
    if not candidates:
        return []
    validated = [c for c in candidates if (repo_root / c.rstrip("/")).is_dir()]
    return validated


def _drift_to_acceptance_criterion(drift: dict | None) -> str | None:
    """Translate preflight contract drift into an actionable acceptance criterion.

    Returns ``None`` if drift is absent or the expected and actual files match.
    """
    if not drift:
        return None
    expected = drift.get("expected", {}) or {}
    actual = drift.get("actual", {}) or {}
    expected_files = list(expected.get("files", []))
    actual_files = set(actual.get("files", []))
    if not expected_files or set(expected_files) == actual_files:
        return None
    files_str = ", ".join(f"`{f}`" for f in expected_files)
    return (
        f"Worker must scope changes strictly to: {files_str}. "
        "Reject any edits to files outside this list during preflight."
    )


def _tier1_enrich(
    spec: SwarmSpec,
    ctx: UpgradeFailureContext,
    *,
    repo_root: Path,
) -> SwarmSpec | None:
    """Deterministic enrichment from static signals (no LLM).

    Returns the upgraded spec if the enrichment bounds it, otherwise ``None``
    to signal that Tier 2 (LLM) is needed.
    """
    flags = _classify_missing_bounds(ctx.missing_bounds)
    extracted_paths = _extract_file_paths(ctx.original_issue_body, repo_root=repo_root)
    track_hints = _infer_track_scope(
        ctx.track_tag, issue_body=ctx.original_issue_body, repo_root=repo_root
    )
    drift_crit = _drift_to_acceptance_criterion(ctx.preflight_diff)

    new_file_scope = list(spec.file_scope_hints)
    if flags["needs_file_scope"]:
        for path in extracted_paths:
            if path not in new_file_scope:
                new_file_scope.append(path)
        for hint in track_hints:
            if hint not in new_file_scope:
                new_file_scope.append(hint)

    # Always add drift criterion when drift is present -- it conveys a
    # scoping constraint beyond whatever ``missing_bounds`` flags imply.
    new_acceptance = list(spec.acceptance_criteria)
    if drift_crit and drift_crit not in new_acceptance:
        new_acceptance.append(drift_crit)
    if flags["needs_acceptance"] and not new_acceptance and ctx.issue_title and new_file_scope:
        new_acceptance.append(f"Implement the behavior described by: {ctx.issue_title.strip()}")

    new_constraints = list(spec.constraints)
    if flags["needs_constraint"] and new_file_scope:
        constraint = (
            f"Limit modifications to the listed file-scope hints: {', '.join(new_file_scope)}."
        )
        if constraint not in new_constraints:
            new_constraints.append(constraint)

    new_work_orders: list[dict[str, Any]] = list(spec.work_orders)
    if flags["needs_work_order"] and new_acceptance:
        seed_order = {"description": f"Satisfy: {new_acceptance[0]}"}
        if seed_order not in new_work_orders:
            new_work_orders.append(seed_order)

    candidate = replace(
        spec,
        file_scope_hints=new_file_scope,
        acceptance_criteria=new_acceptance,
        constraints=new_constraints,
        work_orders=new_work_orders,
    )
    if candidate.is_dispatch_bounded():
        return candidate
    return None


class _LLMLogicFailure(Exception):
    """Internal: LLM returned malformed / ungrounded output after local retry."""


def _build_tier2_prompt(spec: SwarmSpec, ctx: UpgradeFailureContext, repo_root: Path) -> str:
    """Build the Tier 2 LLM prompt from the current spec + failure context."""
    del repo_root  # reserved for future use (e.g., injecting repo tree)
    return f"""You are upgrading an underspecified GitHub issue into a dispatchable SwarmSpec.

Issue title: {ctx.issue_title}
Issue body:
{ctx.original_issue_body}

Missing bounds: {ctx.missing_bounds}
Preflight drift: {json.dumps(ctx.preflight_diff) if ctx.preflight_diff else "none"}

Current spec state:
- acceptance_criteria: {spec.acceptance_criteria}
- file_scope_hints: {spec.file_scope_hints}
- constraints: {spec.constraints}
- work_orders: {spec.work_orders}

Respond with ONLY a JSON object containing fields to ADD (not replace) to the spec:
{{
  "acceptance_criteria": [...],
  "file_scope_hints": [...],
  "constraints": [...],
  "work_orders": [...]
}}

Rules:
- File paths MUST exist in the repo. Do not invent paths.
- Acceptance criteria must be specific and verifiable.
- Constraints must be enforceable (e.g., "no changes outside listed files").
- Omit any field you cannot responsibly fill.
"""


def _tier2_enrich(
    spec: SwarmSpec,
    ctx: UpgradeFailureContext,
    *,
    client: Any,
    repo_root: Path,
) -> SwarmSpec | None:
    """LLM-backed enrichment.

    Raises :class:`SpecUpgraderUnavailable` on transient infrastructure errors
    (timeouts, connection errors) so the caller can skip-for-this-tick without
    consuming an attempt. Raises :class:`_LLMLogicFailure` on malformed or
    ungrounded output after one local retry.

    Returns the upgraded ``SwarmSpec`` on success, or ``None`` if the upgrade
    still isn't dispatch-bounded (caller escalates).
    """
    prompt = _build_tier2_prompt(spec, ctx, repo_root)
    last_err: Exception | None = None

    for attempt in range(2):
        try:
            raw = client.complete(prompt)
        except (ConnectionError, TimeoutError) as exc:
            raise SpecUpgraderUnavailable(str(exc)) from exc
        except Exception as exc:  # transient infra error surfaced by client
            last_err = exc
            if attempt == 0:
                time.sleep(1)
                continue
            raise SpecUpgraderUnavailable(str(exc)) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_err = exc
            if attempt == 0:
                continue
            raise _LLMLogicFailure(f"LLM output not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise _LLMLogicFailure("LLM output not a JSON object")

        added_acceptance = [str(item) for item in parsed.get("acceptance_criteria", []) if item]
        added_file_scope = [str(item) for item in parsed.get("file_scope_hints", []) if item]
        added_constraints = [str(item) for item in parsed.get("constraints", []) if item]
        added_work_orders: list[dict[str, Any]] = []
        for item in parsed.get("work_orders", []) or []:
            if isinstance(item, dict):
                added_work_orders.append(item)
            elif isinstance(item, str) and item.strip():
                added_work_orders.append({"description": item.strip()})

        candidate = replace(
            spec,
            acceptance_criteria=[*spec.acceptance_criteria, *added_acceptance],
            file_scope_hints=[*spec.file_scope_hints, *added_file_scope],
            constraints=[*spec.constraints, *added_constraints],
            work_orders=[*spec.work_orders, *added_work_orders],
        )
        if candidate.is_dispatch_bounded():
            return candidate
        # Still unbounded even after LLM enrichment -- caller escalates.
        return None

    raise _LLMLogicFailure(f"Exhausted LLM attempts: {last_err}")


_AUDIT_MARKER_RE = re.compile(
    r"<!--\s*spec-upgraded:v(?P<version>\d+)\s+attempt=(?P<attempt>\d+)\s*-->"
)
_AUDIT_MARKER_PRESENT_RE = re.compile(r"<!--\s*spec-upgraded:")
MAX_ATTEMPTS = 2


def _parse_audit_marker(comment_body: str) -> tuple[int, bool]:
    """Parse the attempt count from an audit comment.

    Returns ``(attempt_count, valid)``. When a marker is present but unparseable
    (corrupted or unknown version), returns ``(MAX_ATTEMPTS, False)`` to
    conservatively trigger escalation rather than reset the counter.
    """
    match = _AUDIT_MARKER_RE.search(comment_body)
    if match is not None and match.group("version") == "1":
        try:
            return int(match.group("attempt")), True
        except ValueError:  # pragma: no cover - regex guarantees digits
            return MAX_ATTEMPTS, False
    if _AUDIT_MARKER_PRESENT_RE.search(comment_body):
        # Marker-ish present but didn't parse -- treat as corrupted.
        return MAX_ATTEMPTS, False
    return 0, True


class AuditPersistence:
    """Idempotent upsert of the ``[spec-upgraded]`` audit comment on a GitHub issue."""

    MARKER_PREFIX = "<!-- spec-upgraded:v1"

    def __init__(self, issue_number: int, *, repo: str = "synaptent/aragora") -> None:
        self.issue_number = issue_number
        self.repo = repo

    def read_attempt_count(self) -> tuple[int, bool]:
        """Scan comments for the marker and return ``(attempt_count, marker_valid)``."""
        comments = self._gh_list_comments()
        for comment in comments:
            body = comment.get("body") or ""
            if self.MARKER_PREFIX in body:
                return _parse_audit_marker(body)
        return 0, True

    def upsert(self, *, attempt: int, audit_markdown: str) -> bool:
        """Upsert the audit comment. Returns ``True`` on success, ``False`` on ``gh`` failure."""
        marker = f"<!-- spec-upgraded:v1 attempt={attempt} -->"
        body = f"{marker}\n\n{audit_markdown}"
        try:
            existing = self._find_existing_comment()
            if existing is None:
                self._gh_create_comment(body=body)
            else:
                self._gh_update_comment(comment_id=existing["id"], body=body)
            return True
        except subprocess.CalledProcessError:
            return False

    def _find_existing_comment(self) -> dict | None:
        for comment in self._gh_list_comments():
            if self.MARKER_PREFIX in (comment.get("body") or ""):
                return comment
        return None

    # --- gh wrappers (seams for test mocking) ---

    def _gh_list_comments(self) -> list[dict]:
        out = subprocess.check_output(
            [
                "gh",
                "issue",
                "view",
                str(self.issue_number),
                "--repo",
                self.repo,
                "--json",
                "comments",
                "--jq",
                ".comments",
            ],
            text=True,
        )
        return json.loads(out or "[]")

    def _gh_create_comment(self, *, body: str) -> None:
        subprocess.check_call(
            [
                "gh",
                "issue",
                "comment",
                str(self.issue_number),
                "--repo",
                self.repo,
                "--body",
                body,
            ]
        )

    def _gh_update_comment(self, *, comment_id: int, body: str) -> None:
        # gh does not expose direct comment edit; use gh api
        subprocess.check_call(
            [
                "gh",
                "api",
                "--method",
                "PATCH",
                f"/repos/{self.repo}/issues/comments/{comment_id}",
                "-f",
                f"body={body}",
            ]
        )


def emit_upgrade_telemetry(
    *,
    metrics_path: Path,
    issue_number: int,
    seam: Literal["A", "B"],
    attempt_count: int,
    status: UpgradeStatus,
    upgrade_path: UpgradePath | None,
    wall_clock_ms: int,
    audit_failed: bool,
    escalation_failed: bool,
    llm_tokens_in: int,
    llm_tokens_out: int,
    failure_reasons: list[str],
) -> str:
    """Append a per-upgrade row to ``boss_metrics.jsonl``.

    Returns the generated ``upgrade_id`` so callers can reference it from
    dispatch records (see design doc ``upgrade_refs``).
    """
    upgrade_id = str(uuid.uuid4())
    record = {
        "event": "spec_upgrade",
        "upgrade_id": upgrade_id,
        "issue_number": issue_number,
        "seam": seam,
        "attempt_count": attempt_count,
        "status": status,
        "upgrade_path": upgrade_path,
        "wall_clock_ms": wall_clock_ms,
        "audit_failed": audit_failed,
        "escalation_failed": escalation_failed,
        "llm_tokens_in": llm_tokens_in,
        "llm_tokens_out": llm_tokens_out,
        "failure_reasons": failure_reasons,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return upgrade_id


class Escalator:
    """Apply ``needs-clarification`` label + post unresolved-questions comment.

    Fail-closed: returns ``False`` if either label or comment mutation fails;
    caller must NOT dispatch the issue in that case.
    """

    LABEL = "needs-clarification"

    def __init__(self, issue_number: int, *, repo: str = "synaptent/aragora") -> None:
        self.issue_number = issue_number
        self.repo = repo

    def escalate(
        self,
        *,
        unresolved_questions: list[str],
        failure_context_summary: str,
    ) -> bool:
        try:
            self._gh_add_label()
        except subprocess.CalledProcessError:
            return False
        body = self._render_comment(unresolved_questions, failure_context_summary)
        try:
            self._gh_create_comment(body=body)
        except subprocess.CalledProcessError:
            return False
        return True

    def _render_comment(self, questions: list[str], summary: str) -> str:
        q_block = "\n".join(f"- {q}" for q in questions) if questions else "- (none specified)"
        return (
            "## Needs clarification\n\n"
            "The autonomous spec upgrader could not bound this issue after the maximum "
            "attempts. Human review required.\n\n"
            f"**Failure summary:** {summary}\n\n"
            f"**Unresolved questions:**\n{q_block}\n\n"
            "_Posted by SpecUpgrader._"
        )

    def _gh_add_label(self) -> None:
        subprocess.check_call(
            [
                "gh",
                "issue",
                "edit",
                str(self.issue_number),
                "--repo",
                self.repo,
                "--add-label",
                self.LABEL,
            ]
        )

    def _gh_create_comment(self, *, body: str) -> None:
        subprocess.check_call(
            [
                "gh",
                "issue",
                "comment",
                str(self.issue_number),
                "--repo",
                self.repo,
                "--body",
                body,
            ]
        )


def _derive_questions(ctx: UpgradeFailureContext) -> list[str]:
    """Convert ``missing_bounds`` into reviewer-facing clarifying questions."""
    questions: list[str] = []
    if "acceptance criterion" in ctx.missing_bounds:
        questions.append("What observable behaviour proves this issue is resolved?")
    if "file-scope hint" in ctx.missing_bounds:
        questions.append("Which files (exact paths) should be modified?")
    if "constraint" in ctx.missing_bounds:
        questions.append("Are there files, APIs, or behaviours that must NOT change?")
    if "explicit work order" in ctx.missing_bounds:
        questions.append("What concrete steps should an implementer take?")
    return questions


def _summarise_failure(ctx: UpgradeFailureContext) -> str:
    parts = [f"missing: {', '.join(ctx.missing_bounds)}"] if ctx.missing_bounds else []
    if ctx.preflight_diff:
        parts.append("preflight contract drift detected")
    return "; ".join(parts) or "underspecified"


def _render_audit(
    attempt: int,
    path: UpgradePath | None,
    ctx: UpgradeFailureContext,
    *,
    escalated: bool,
) -> str:
    verdict = "ESCALATED" if escalated else "UPGRADED"
    return (
        "## Upgrade audit\n\n"
        f"- **Attempt:** {attempt}\n"
        f"- **Path:** {path or 'n/a'}\n"
        f"- **Verdict:** {verdict}\n"
        f"- **Missing bounds on entry:** {', '.join(ctx.missing_bounds) or 'none'}\n"
        f"- **Preflight drift:** {'yes' if ctx.preflight_diff else 'no'}\n"
    )


def upgrade_spec(
    spec: SwarmSpec,
    failure_context: UpgradeFailureContext,
    *,
    issue_number: int,
    seam: Literal["A", "B"],
    repo_root: Path,
    metrics_path: Path,
    llm_client: Any = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> UpgradeResult:
    """Upgrade a weak ``SwarmSpec`` into a dispatchable one.

    See ``docs/plans/2026-04-17-spec-upgrader-design.md`` for the full
    architecture. Raises :class:`SpecUpgraderUnavailable` on transient
    infrastructure failure; the caller should skip-for-this-tick without
    consuming an attempt.
    """
    start = time.monotonic()

    audit = AuditPersistence(issue_number=issue_number)
    prior_attempts, marker_valid = audit.read_attempt_count()

    # Marker corrupted OR budget exhausted -> escalate immediately.
    if not marker_valid or prior_attempts >= max_attempts:
        questions = _derive_questions(failure_context)
        summary = _summarise_failure(failure_context)
        esc = Escalator(issue_number=issue_number)
        escalated_ok = esc.escalate(
            unresolved_questions=questions,
            failure_context_summary=summary,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        emit_upgrade_telemetry(
            metrics_path=metrics_path,
            issue_number=issue_number,
            seam=seam,
            attempt_count=prior_attempts,
            status="escalated",
            upgrade_path=None,
            wall_clock_ms=elapsed,
            audit_failed=False,
            escalation_failed=not escalated_ok,
            llm_tokens_in=0,
            llm_tokens_out=0,
            failure_reasons=list(failure_context.missing_bounds),
        )
        return UpgradeResult(
            status="escalated",
            upgraded_spec=None,
            audit_markdown="Budget exhausted or marker corrupted.",
            attempt_count=prior_attempts,
            upgrade_path=None,
            failure_context=failure_context,
            unresolved_questions=questions,
        )

    attempt = prior_attempts + 1
    path_taken: UpgradePath = "deterministic"

    # Tier 1 deterministic enrichment.
    upgraded = _tier1_enrich(spec, failure_context, repo_root=repo_root)

    # Tier 2 LLM fallback if Tier 1 was insufficient.
    if upgraded is None and llm_client is not None:
        try:
            upgraded = _tier2_enrich(spec, failure_context, client=llm_client, repo_root=repo_root)
            path_taken = "deterministic+llm"
        except _LLMLogicFailure:
            upgraded = None

    elapsed = int((time.monotonic() - start) * 1000)

    if upgraded is None:
        questions = _derive_questions(failure_context)
        summary = _summarise_failure(failure_context)
        esc = Escalator(issue_number=issue_number)
        escalated_ok = esc.escalate(
            unresolved_questions=questions,
            failure_context_summary=summary,
        )
        emit_upgrade_telemetry(
            metrics_path=metrics_path,
            issue_number=issue_number,
            seam=seam,
            attempt_count=attempt,
            status="escalated",
            upgrade_path=path_taken,
            wall_clock_ms=elapsed,
            audit_failed=False,
            escalation_failed=not escalated_ok,
            llm_tokens_in=0,
            llm_tokens_out=0,
            failure_reasons=list(failure_context.missing_bounds),
        )
        return UpgradeResult(
            status="escalated",
            upgraded_spec=None,
            audit_markdown=_render_audit(attempt, path_taken, failure_context, escalated=True),
            attempt_count=attempt,
            upgrade_path=path_taken,
            failure_context=failure_context,
            unresolved_questions=questions,
        )

    audit_md = _render_audit(attempt, path_taken, failure_context, escalated=False)
    audit_ok = audit.upsert(attempt=attempt, audit_markdown=audit_md)
    emit_upgrade_telemetry(
        metrics_path=metrics_path,
        issue_number=issue_number,
        seam=seam,
        attempt_count=attempt,
        status="upgraded",
        upgrade_path=path_taken,
        wall_clock_ms=elapsed,
        audit_failed=not audit_ok,
        escalation_failed=False,
        llm_tokens_in=0,
        llm_tokens_out=0,
        failure_reasons=list(failure_context.missing_bounds),
    )
    return UpgradeResult(
        status="upgraded",
        upgraded_spec=upgraded,
        audit_markdown=audit_md,
        attempt_count=attempt,
        upgrade_path=path_taken,
        failure_context=failure_context,
        unresolved_questions=[],
    )
