"""Helpers for serialized boss-loop follow-up actions."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.swarm.acceptance_gate import (
    AcceptanceGateResult,
    evaluate_acceptance,
    inject_closes_into_pr_body,
    pr_body_already_closes,
)
from aragora.swarm.issue_scanner import infer_issue_category_from_title
from aragora.swarm.issue_upgrader import upgrade_issue_heuristic
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.spec_upgrader import (
    SpecUpgraderUnavailable,
    UpgradeFailureContext,
    extract_drift_diagnostic,
    upgrade_spec,
)

logger = logging.getLogger(__name__)

_MARKDOWN_BULLET_RE = re.compile(r"^[-*]\s+(?:\[[ xX]\]\s+)?(?P<text>.+)$")
_ACCEPTANCE_SECTION_HEADINGS = {"acceptance", "acceptance criteria"}

# ---------------------------------------------------------------------------
# PR-1 of #7209 — corpus-aware dispatch upgrade (flag-gated, default OFF)
#
# When ``ARAGORA_CORPUS_AWARE_DISPATCH`` is truthy and the issue under
# dispatch is a member of ``docs/benchmarks/corpus.json::issues`` whose
# ``execution_class`` and terminal history are in the PR-1 whitelists,
# augment under-specified specs with the corpus row's ``scope_hint`` +
# ``known_constraints`` and an ``execution_class``-specific
# acceptance-criteria template so the dispatch contract gate no longer
# rejects the spec as ``blocked_not_dispatch_bounded``.
#
# Default behaviour is unchanged.  See docs/plans/
# 2026-05-16-a2-admission-class-productization.md for the full plan.
# ---------------------------------------------------------------------------

_CORPUS_AWARE_DISPATCH_ENV = "ARAGORA_CORPUS_AWARE_DISPATCH"
_CORPUS_AWARE_DISPATCH_TRUTHY = frozenset({"1", "true", "yes", "on"})
_CORPUS_AWARE_DISPATCH_WHITELIST = frozenset(
    {
        "missing_test_coverage",
        "small_refactor",
        "validation_tightening",
        "exception_narrowing",
    }
)
_CORPUS_AWARE_DISPATCH_TERMINAL_WHITELIST = frozenset(
    {
        "blocked_not_dispatch_bounded",
        "blocked_auth_failure",
    }
)
_CORPUS_AWARE_DISPATCH_CORPUS_RELPATH = Path("docs/benchmarks/corpus.json")

# Per ``execution_class`` rendered acceptance-criteria templates.  Each
# class lists the bounded-target shape the worker should hit.  ``{scope}``
# is rendered as the comma-joined list of ``scope_hint`` paths to keep the
# bullet self-contained.  Templates intentionally restate the constraint
# in operator-readable form because the corpus row's ``known_constraints``
# already lands as ``spec.constraints`` separately.
_EXECUTION_CLASS_ACCEPTANCE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "missing_test_coverage": (
        "Add unit tests covering happy path and at least one edge case for the modules listed under File Scope ({scope}).",
        "`pytest` over the new test file exits 0.",
        "Do not modify production code under `aragora/` unless strictly required by the test scaffold.",
    ),
    "small_refactor": (
        "Refactor is scoped to the files listed under File Scope ({scope}); no other production files are modified.",
        "`pytest` for tests covering the scoped files continues to exit 0.",
        "Public API of the refactored modules is unchanged.",
    ),
    "validation_tightening": (
        "Tighten input validation in the files listed under File Scope ({scope}) so the constraints under ``known_constraints`` are enforced.",
        "Existing valid inputs continue to be accepted; new tests exercise at least one rejected-input case.",
        "`pytest` for tests covering the scoped files exits 0.",
    ),
    "exception_narrowing": (
        "Replace broad ``except Exception`` / bare ``except:`` in the files listed under File Scope ({scope}) with explicit exception classes per ``known_constraints``.",
        "`ruff check` over the scoped files passes.",
        "`pytest` for tests covering the scoped files exits 0.",
    ),
}


def _corpus_aware_dispatch_enabled() -> bool:
    """True iff ``ARAGORA_CORPUS_AWARE_DISPATCH`` resolves to a truthy value."""
    raw = os.environ.get(_CORPUS_AWARE_DISPATCH_ENV, "")
    return str(raw or "").strip().lower() in _CORPUS_AWARE_DISPATCH_TRUTHY


def _corpus_path_for(repo_root: Path | None) -> Path:
    """Resolve the canonical corpus path under ``repo_root``."""
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    return base / _CORPUS_AWARE_DISPATCH_CORPUS_RELPATH


def _load_corpus_entry_for_issue(
    issue_number: int,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    """Look up an issue's corpus row.

    Returns ``None`` when the corpus file is missing, unparseable, or the
    issue is not a recorded member.  Read-on-each-call (no cache): the
    corpus file is small and dispatch hits this path at most once per
    bounded admission attempt.
    """
    if not issue_number or int(issue_number) <= 0:
        return None
    corpus_path = _corpus_path_for(repo_root)
    try:
        raw = corpus_path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    issues = data.get("issues") if isinstance(data, dict) else None
    if not isinstance(issues, list):
        return None
    target = int(issue_number)
    for entry in issues:
        if not isinstance(entry, dict):
            continue
        try:
            if int(entry.get("issue_id", 0) or 0) == target:
                return entry
        except (TypeError, ValueError):
            continue
    return None


def _augment_spec_from_corpus(spec: SwarmSpec, corpus_entry: dict[str, Any]) -> None:
    """Mutate ``spec`` in place with corpus row's file scope + constraints + acceptance criteria."""
    scope_hint = [
        str(path).strip()
        for path in (corpus_entry.get("scope_hint") or [])
        if str(path or "").strip()
    ]
    known_constraints = [
        str(item).strip()
        for item in (corpus_entry.get("known_constraints") or [])
        if str(item or "").strip()
    ]
    execution_class = str(corpus_entry.get("execution_class", "") or "").strip()

    if scope_hint:
        spec.file_scope_hints = _ordered_unique([*spec.file_scope_hints, *scope_hint])

    if known_constraints:
        spec.constraints = _ordered_unique([*spec.constraints, *known_constraints])

    templates = _EXECUTION_CLASS_ACCEPTANCE_TEMPLATES.get(execution_class, ())
    if templates:
        scope_text = ", ".join(scope_hint) if scope_hint else "see scope_hint"
        rendered: list[str] = []
        for template in templates:
            try:
                rendered.append(template.format(scope=scope_text))
            except (KeyError, IndexError):
                rendered.append(template)
        spec.acceptance_criteria = _ordered_unique([*spec.acceptance_criteria, *rendered])


def _corpus_terminal_classes(corpus_entry: dict[str, Any]) -> set[str]:
    """Return normalized terminal classes recorded for a corpus row."""
    provenance = corpus_entry.get("dispatch_provenance")
    if not isinstance(provenance, dict):
        return set()
    classes = provenance.get("terminal_classes")
    if not isinstance(classes, list):
        return set()
    return {str(item).strip() for item in classes if str(item or "").strip()}


def maybe_upgrade_dispatch_spec_from_corpus(
    *,
    issue: Any,
    spec: SwarmSpec,
    repo_root: Path,
) -> SwarmSpec:
    """Augment an under-specified spec from the bounded-execution corpus.

    No-op (returns ``spec`` unchanged) when any of the following hold:

    * ``ARAGORA_CORPUS_AWARE_DISPATCH`` is OFF (default).
    * ``spec`` is already ``is_dispatch_bounded()``.
    * The issue is not a member of ``docs/benchmarks/corpus.json::issues``.
    * The corpus row's ``execution_class`` is not in the PR-1 whitelist
      (``missing_test_coverage``, ``small_refactor``, ``validation_tightening``,
      ``exception_narrowing``).
    * The corpus row's terminal history does not include a PR-1 admission class
      (``blocked_not_dispatch_bounded`` or ``blocked_auth_failure``).

    Otherwise mutates ``spec`` in place with the corpus row's
    ``scope_hint`` (→ ``file_scope_hints``), ``known_constraints``
    (→ ``constraints``), and an ``execution_class``-specific acceptance
    template (→ ``acceptance_criteria``) and returns it.

    Identity-preserving: the returned spec is always the same object
    that was passed in.  Callers may chain into the existing
    title-category heuristic without surprise.
    """
    if not _corpus_aware_dispatch_enabled():
        return spec
    if spec.is_dispatch_bounded():
        return spec
    issue_number = getattr(issue, "number", None)
    if not issue_number:
        return spec
    try:
        issue_number_int = int(issue_number)
    except (TypeError, ValueError):
        return spec
    corpus_entry = _load_corpus_entry_for_issue(issue_number_int, repo_root=repo_root)
    if corpus_entry is None:
        return spec
    execution_class = str(corpus_entry.get("execution_class", "") or "").strip()
    if execution_class not in _CORPUS_AWARE_DISPATCH_WHITELIST:
        return spec
    terminal_classes = _corpus_terminal_classes(corpus_entry)
    if not (terminal_classes & _CORPUS_AWARE_DISPATCH_TERMINAL_WHITELIST):
        return spec

    _augment_spec_from_corpus(spec, corpus_entry)
    logger.info(
        "corpus_aware_dispatch_augmented issue=#%s execution_class=%s terminal_classes=%s scope_hint_count=%s",
        issue_number_int,
        execution_class,
        ",".join(sorted(terminal_classes)),
        len(corpus_entry.get("scope_hint") or []),
    )
    return spec


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _extract_acceptance_criteria(markdown: str) -> list[str]:
    criteria: list[str] = []
    in_acceptance_section = False
    for raw_line in str(markdown or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().rstrip(":").lower()
            in_acceptance_section = heading in _ACCEPTANCE_SECTION_HEADINGS
            continue
        if not in_acceptance_section or not stripped:
            continue
        bullet_match = _MARKDOWN_BULLET_RE.match(stripped)
        if bullet_match:
            criteria.append(bullet_match.group("text"))
            continue
        criteria.append(stripped)
    return _ordered_unique(criteria)


def maybe_upgrade_dispatch_spec(
    *,
    issue: Any,
    spec: SwarmSpec,
    sanitized_issue_body: str,
    repo_root: Path,
) -> SwarmSpec:
    """Try upgrading an under-specified issue before blocking dispatch."""
    if spec.is_dispatch_bounded():
        return spec

    # PR-1 of #7209: corpus-aware augmentation runs first when the flag is
    # ON, so rev-4 corpus members with whitelisted execution classes can be
    # bounded even when the title-category heuristic doesn't fire.  Flag is
    # OFF by default; this is a strict no-op otherwise.
    spec = maybe_upgrade_dispatch_spec_from_corpus(
        issue=issue,
        spec=spec,
        repo_root=repo_root,
    )
    if spec.is_dispatch_bounded():
        return spec

    category = infer_issue_category_from_title(getattr(issue, "title", None))
    if category is None:
        return spec

    upgraded = upgrade_issue_heuristic(
        str(getattr(issue, "title", "") or ""),
        sanitized_issue_body,
        repo_root=repo_root,
        category=category,
        acceptance_criteria=list(getattr(spec, "acceptance_criteria", []) or []),
    )
    if upgraded is None:
        return spec

    upgraded_spec = SwarmSpec.from_direct_goal(
        f"[Issue #{issue.number}] {issue.title}\n\n{upgraded.upgraded_body}",
        budget_limit_usd=spec.budget_limit_usd,
        requires_approval=spec.requires_approval,
        user_expertise=spec.user_expertise,
        use_llm=False,
    )
    inferred_scope = SwarmSpec.infer_file_scope_hints(upgraded.upgraded_body)
    spec.raw_goal = upgraded_spec.raw_goal
    spec.refined_goal = upgraded_spec.refined_goal or spec.refined_goal
    spec.acceptance_criteria = _ordered_unique(
        [*spec.acceptance_criteria, *_extract_acceptance_criteria(upgraded.upgraded_body)]
    )
    spec.constraints = _ordered_unique([*spec.constraints, *upgraded_spec.constraints])
    spec.track_hints = _ordered_unique([*spec.track_hints, *upgraded_spec.track_hints])
    spec.file_scope_hints = _ordered_unique(
        [*spec.file_scope_hints, *upgraded_spec.file_scope_hints, *inferred_scope]
    )
    spec.estimated_complexity = upgraded_spec.estimated_complexity or spec.estimated_complexity
    return spec


_TRACK_TAG_RE = re.compile(r"^\s*\[([A-Z]+-\d+)\]")


def _extract_track_tag(issue_title: str) -> str | None:
    """Extract ``[TW-02]``-style prefix from an issue title."""
    match = _TRACK_TAG_RE.match(issue_title or "")
    return match.group(1) if match else None


def upgrade_unbounded_spec(
    spec: SwarmSpec,
    *,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    repo_root: Path,
    metrics_path: Path,
    llm_client: Any = None,
) -> SwarmSpec | None:
    """Seam A: upgrade an unbounded spec before the contract-gate dispatch.

    Returns the upgraded :class:`SwarmSpec` if dispatch should proceed, or
    ``None`` if the upgrader escalated to ``needs-clarification`` (caller must
    skip dispatch).

    Raises :class:`SpecUpgraderUnavailable` on transient infrastructure
    failure -- caller treats as skip-for-this-tick.
    """
    if spec.is_dispatch_bounded():
        return spec
    ctx = UpgradeFailureContext(
        missing_bounds=spec.missing_dispatch_bounds(),
        preflight_diff=None,
        prior_attempts=0,  # read durably inside ``upgrade_spec``
        original_issue_body=issue_body,
        issue_title=issue_title,
        track_tag=_extract_track_tag(issue_title),
    )
    result = upgrade_spec(
        spec,
        ctx,
        issue_number=issue_number,
        seam="A",
        repo_root=repo_root,
        metrics_path=metrics_path,
        llm_client=llm_client,
    )
    if result.status == "upgraded":
        return result.upgraded_spec
    return None


def upgrade_on_contract_drift(
    spec: SwarmSpec,
    *,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    preflight_diff: dict,
    repo_root: Path,
    metrics_path: Path,
    llm_client: Any = None,
) -> SwarmSpec | None:
    """Seam B: upgrade a spec after contract-gate reported drift.

    Returns the upgraded spec to retry dispatch, or ``None`` to escalate
    (caller skips). Raises :class:`SpecUpgraderUnavailable` on transient infra
    failure.
    """
    ctx = UpgradeFailureContext(
        missing_bounds=list(spec.missing_dispatch_bounds()),
        preflight_diff=preflight_diff,
        prior_attempts=0,  # read durably inside ``upgrade_spec``
        original_issue_body=issue_body,
        issue_title=issue_title,
        track_tag=_extract_track_tag(issue_title),
    )
    result = upgrade_spec(
        spec,
        ctx,
        issue_number=issue_number,
        seam="B",
        repo_root=repo_root,
        metrics_path=metrics_path,
        llm_client=llm_client,
    )
    if result.status == "upgraded":
        return result.upgraded_spec
    return None


def maybe_upgrade_on_contract_drift(
    *,
    gate_result: Any,
    spec: SwarmSpec,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    repo_root: Path,
    metrics_path: Path,
    llm_client: Any = None,
) -> SwarmSpec | None:
    """Seam B wiring helper -- extract drift from a failed contract-gate result.

    Given the dict returned by
    :func:`aragora.swarm.dispatch_contract_gate.dispatch_contract_gate` when
    admission fails, pull out a drift diagnostic (if any), seed
    ``expected.files`` from the spec's scope hints so the drift translator can
    emit a concrete scoping criterion, and invoke
    :func:`upgrade_on_contract_drift`.

    Returns the upgraded :class:`SwarmSpec` when the upgrade produced a
    dispatch-bounded spec; ``None`` when there was no drift to act on, the
    upgrader escalated, or the LLM infrastructure was transiently unavailable.
    """
    drift = extract_drift_diagnostic(gate_result)
    if drift is None:
        return None
    scope = [str(path).strip() for path in (spec.file_scope_hints or []) if str(path).strip()]
    if scope:
        expected = drift.setdefault("expected", {})
        if not expected.get("files"):
            expected["files"] = scope
    try:
        return upgrade_on_contract_drift(
            spec,
            issue_number=issue_number,
            issue_title=issue_title,
            issue_body=issue_body,
            preflight_diff=drift,
            repo_root=repo_root,
            metrics_path=metrics_path,
            llm_client=llm_client,
        )
    except SpecUpgraderUnavailable:
        logger.warning(
            "spec_upgrader_unavailable_on_contract_drift issue=#%s",
            issue_number,
        )
        return None


__all__ = [
    "SpecUpgraderUnavailable",
    "annotate_result_with_conductor",
    "collect_worker_changed_paths",
    "enforce_acceptance_binding",
    "inject_closes_into_published_pr",
    "maybe_upgrade_dispatch_spec",
    "maybe_upgrade_dispatch_spec_from_corpus",
    "maybe_upgrade_on_contract_drift",
    "upgrade_on_contract_drift",
    "upgrade_unbounded_spec",
]


# ---------------------------------------------------------------------------
# v1.3 — post-delivery acceptance-criteria binding
# ---------------------------------------------------------------------------


def collect_worker_changed_paths(worker_result: dict[str, Any]) -> list[str]:
    """Union of ``changed_paths`` across a worker result's run + deliverable."""
    paths: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        text = str(candidate or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        paths.append(text)

    run = worker_result.get("run") if isinstance(worker_result, dict) else None
    if isinstance(run, dict):
        for work_order in run.get("work_orders", []) or []:
            if not isinstance(work_order, dict):
                continue
            for path in work_order.get("changed_paths", []) or []:
                _add(path)

    deliverable = worker_result.get("deliverable") if isinstance(worker_result, dict) else None
    if isinstance(deliverable, dict):
        for path in deliverable.get("changed_paths", []) or []:
            _add(path)

    return paths


def _append_spec_upgrade_telemetry(
    metrics_path: Path | None,
    *,
    issue_number: int,
    gate_result: AcceptanceGateResult,
    changed_paths: list[str],
) -> None:
    """Emit a ``spec_upgrade``-style JSONL row for the Stage-Gate Conductor.

    This is best-effort: any I/O exception is swallowed so the gate outcome
    takes precedence over telemetry failures.
    """
    if metrics_path is None:
        return
    try:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": "acceptance_gate",
            "issue_number": int(issue_number),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": bool(gate_result.passed),
            "failure_classes": list(gate_result.failure_classes),
            "reasons": list(gate_result.reasons),
            "checks_run": list(gate_result.checks_run),
            "changed_paths": list(changed_paths),
            "out_of_scope_paths": list(gate_result.out_of_scope_paths),
            "missing_expected_files": list(gate_result.missing_expected_files),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    except (OSError, ValueError):
        logger.debug("acceptance_gate_telemetry_write_failed path=%s", metrics_path, exc_info=True)


def enforce_acceptance_binding(
    *,
    issue_number: int,
    issue_body: str,
    spec: SwarmSpec,
    worker_result: dict[str, Any],
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Run the post-delivery acceptance-criteria gate on a worker result.

    Mutates ``worker_result`` in-place with one of:

    * ``acceptance_gate`` → serialized :class:`AcceptanceGateResult`
    * ``acceptance_gate_passed`` → True / False
    * ``closes_issue_number`` → ``issue_number`` when the gate passes and an
      explicit originating issue is available.  The publish path reads this
      to inject ``Closes #<issue_number>`` into the PR body.

    On gate **failure**, the result's ``status`` is transformed to
    ``"needs_human"`` with a structured ``outcome`` = ``"acceptance_gate_failed"``
    and the failure reasons are merged into the existing ``reasons`` list.
    The deliverable is NOT removed from the worker result — downstream
    tooling may still need it for forensic inspection — but auto-publish
    is expected to skip on ``needs_human`` results without a confirmed
    deliverable type, and the Closes #N injection will not fire.

    The function is a no-op when:

    * The worker result is not in a terminal successful state
      (``completed`` or ``needs_human`` with a deliverable).
    * The spec has no acceptance criteria, no scope hints, and no
      pytest targets to seed expected files from.
    """
    if not isinstance(worker_result, dict):
        return worker_result

    status = str(worker_result.get("status", "")).strip().lower()
    # Only run the gate when the worker produced something publishable.
    if status not in {"completed", "needs_human"}:
        return worker_result

    deliverable = worker_result.get("deliverable")
    # No deliverable → nothing to bind.  Let existing pathways handle it.
    if not isinstance(deliverable, dict):
        return worker_result

    changed_paths = collect_worker_changed_paths(worker_result)
    # Also surface deliverable changed_paths when the run didn't include them.
    if not changed_paths:
        # An adopted_pr or type=pr deliverable has no concrete path list.
        # In that case we can't evaluate file-scope or test-presence, so we
        # skip the gate rather than raise a false alarm.
        return worker_result

    acceptance_criteria = list(getattr(spec, "acceptance_criteria", []) or [])
    if not acceptance_criteria:
        acceptance_criteria = _extract_acceptance_criteria(issue_body)
    file_scope_hints = list(getattr(spec, "file_scope_hints", []) or [])

    gate_result = evaluate_acceptance(
        acceptance_criteria=acceptance_criteria,
        file_scope_hints=file_scope_hints,
        changed_paths=changed_paths,
        issue_body=issue_body,
    )

    worker_result["acceptance_gate"] = gate_result.to_dict()
    worker_result["acceptance_gate_passed"] = bool(gate_result.passed)

    # Telemetry — best-effort, never blocks the gate decision.
    _append_spec_upgrade_telemetry(
        metrics_path,
        issue_number=issue_number,
        gate_result=gate_result,
        changed_paths=changed_paths,
    )

    if gate_result.passed:
        # Signal to the publish path that ``Closes #<issue_number>`` may be
        # injected on this deliverable's PR body (idempotent downstream).
        if int(issue_number) > 0:
            worker_result["closes_issue_number"] = int(issue_number)
        return worker_result

    # Gate failed — mark the run as needing human review and merge the
    # failure reasons into the worker result.  We do NOT strip the
    # deliverable or PR URL so forensic tooling can still inspect it.
    existing_reasons = worker_result.get("reasons")
    reasons: list[str] = (
        [str(item).strip() for item in existing_reasons if str(item).strip()]
        if isinstance(existing_reasons, list)
        else []
    )
    for reason in gate_result.reasons:
        if reason and reason not in reasons:
            reasons.append(reason)
    worker_result["reasons"] = reasons
    worker_result["status"] = "needs_human"
    worker_result["outcome"] = "acceptance_gate_failed"
    existing_failure_classes = worker_result.get("failure_classes")
    failure_classes: list[str] = (
        [str(item).strip() for item in existing_failure_classes if str(item).strip()]
        if isinstance(existing_failure_classes, list)
        else []
    )
    for fc in gate_result.failure_classes:
        if fc and fc not in failure_classes:
            failure_classes.append(fc)
    worker_result["failure_classes"] = failure_classes

    logger.info(
        "acceptance_gate_failed issue=#%s failure_classes=%s checks_run=%s",
        issue_number,
        ",".join(gate_result.failure_classes),
        ",".join(gate_result.checks_run),
    )

    return worker_result


def _gh_pr_view_body(pr_url: str, *, repo_root: Path | None = None) -> str | None:
    """Fetch the current PR body via ``gh pr view --json body``.

    Returns ``None`` on any failure so callers can safely abort injection.
    """
    if not pr_url:
        return None
    cwd = str(repo_root) if repo_root is not None else None
    try:
        proc = subprocess.run(
            ["gh", "pr", "view", pr_url, "--json", "body"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "")
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    body = payload.get("body")
    return str(body or "")


def _gh_pr_edit_body(pr_url: str, new_body: str, *, repo_root: Path | None = None) -> bool:
    """Update the PR body via ``gh pr edit --body``.

    Returns True on success.  Never raises.
    """
    if not pr_url:
        return False
    cwd = str(repo_root) if repo_root is not None else None
    try:
        proc = subprocess.run(
            ["gh", "pr", "edit", pr_url, "--body", new_body],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def inject_closes_into_published_pr(
    *,
    pr_url: str,
    issue_number: int,
    repo_root: Path | None = None,
    body_fetcher: Any = None,
    body_setter: Any = None,
) -> dict[str, Any]:
    """Inject ``Closes #<issue_number>`` into an already-published PR body.

    This is the Phase 3 auto-close step.  It runs only after the Phase 2
    acceptance gate has *passed*; callers are expected to check
    ``worker_result["acceptance_gate_passed"]`` before invoking.

    Idempotent: if the PR body already contains ``Closes/Fixes/Resolves
    #<issue_number>`` (or any variant), the body is left unchanged.

    ``body_fetcher`` and ``body_setter`` are injectable for testing.
    Their production defaults call the ``gh`` CLI.

    Returns a status dict with keys ``{"action", "injected", "detail"}``.
    """
    fetcher = body_fetcher or (lambda: _gh_pr_view_body(pr_url, repo_root=repo_root))
    setter = body_setter or (
        lambda new_body: _gh_pr_edit_body(pr_url, new_body, repo_root=repo_root)
    )

    if not pr_url or int(issue_number) <= 0:
        return {
            "action": "skipped",
            "injected": False,
            "detail": "pr_url and positive issue_number required",
        }

    current_body = fetcher()
    if current_body is None:
        return {
            "action": "fetch_failed",
            "injected": False,
            "detail": "could not read current PR body via gh pr view",
        }

    if pr_body_already_closes(current_body, issue_number=issue_number):
        return {
            "action": "already_closes",
            "injected": False,
            "detail": "PR body already references Closes/Fixes/Resolves for this issue",
        }

    new_body = inject_closes_into_pr_body(current_body, issue_number=issue_number)
    if new_body == current_body:
        return {
            "action": "already_closes",
            "injected": False,
            "detail": "no change required",
        }

    updated = setter(new_body)
    if not updated:
        return {
            "action": "edit_failed",
            "injected": False,
            "detail": "gh pr edit failed; PR body not updated",
        }
    return {
        "action": "injected",
        "injected": True,
        "detail": f"Prepended 'Closes #{int(issue_number)}' to PR body",
    }


def annotate_result_with_conductor(
    *,
    issue_number: int,
    result: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Attach conductor follow-up hints to non-success dispatch results."""
    if result.get("status") not in {"needs_human", "failed"}:
        return result

    try:
        from aragora.swarm.conductor import Conductor

        step = Conductor(repo_root=repo_root).evaluate_worker_output(issue_number, result)
    except Exception:
        return result

    annotated = dict(result)
    annotated.update(
        {
            "conductor_next_action": step.next_action,
            "conductor_next_prompt": (step.next_prompt or "")[:500],
            "conductor_terminal_class": step.terminal_class.value
            if hasattr(step.terminal_class, "value")
            else str(step.terminal_class),
        }
    )
    return annotated
