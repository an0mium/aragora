# SpecUpgrader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v1 SpecUpgrader that converts the boss loop's dispatch path from a gate (rejects weak specs) to a repair loop (upgrades weak specs, dispatches the upgraded result, audits the transformation). Targets the `blocked_not_dispatch_bounded` failure class blocking #5898 and #5903.

**Architecture:** Single public library module `aragora/swarm/spec_upgrader.py` exposing `upgrade_spec()`. Two integration seams in `boss_worker_lifecycle.py` — Seam A replaces the existing pre-contract-gate heuristic call; Seam B adds drift-feedback after contract-gate failure. Persistence via idempotent `[spec-upgraded]` GitHub comment (durable source of truth for attempt counts). Two enrichment tiers: deterministic first (no LLM), LLM fallback on miss. Hard cap of 2 attempts per issue; escalation via `needs-clarification` label on exhaustion.

**Tech Stack:** Python 3.13, pytest, `subprocess` for `gh` CLI, existing `aragora.swarm` types (`SwarmSpec`, `missing_dispatch_bounds()`), existing `aragora.agents` factory for LLM client, `boss_metrics.jsonl` for telemetry.

**Design reference:** [`docs/plans/2026-04-17-spec-upgrader-design.md`](./2026-04-17-spec-upgrader-design.md)

---

## File structure

**Created:**
- `aragora/swarm/spec_upgrader.py` — public library module; if it exceeds ~500 LOC during implementation, split into a package (v1.1 refactor, not blocking).
- `tests/swarm/test_spec_upgrader.py` — unit tests.
- `tests/swarm/test_spec_upgrader_integration.py` — integration tests (real preflight, mocked LLM).
- `tests/swarm/fixtures/spec_upgrader/issue_5898.json` — frozen issue snapshot.
- `tests/swarm/fixtures/spec_upgrader/issue_5903.json` — frozen issue snapshot.

**Modified:**
- `aragora/swarm/boss_worker_lifecycle.py` — Seam A integration at line ~801, Seam B integration at line ~876. **LOC ratchet applies** (hard limit enforced by CI); keep diff tight.
- `aragora/swarm/dispatch_followups.py` — minimal wrapper adjustment if needed to pass `UpgradeFailureContext` through.

---

## Task 0: Create feature branch

**Files:** none yet (branch setup only).

- [ ] **Step 1: Verify current tree is clean and on main**

Run: `git status -sb`
Expected: `## main...origin/main` with no uncommitted changes in tracked files (untracked `.aragora_coordination/` is OK).

- [ ] **Step 2: Pull latest main**

Run: `git fetch origin && git checkout main && git pull --ff-only origin main`
Expected: `Already up to date` or fast-forward merge.

- [ ] **Step 3: Create feature branch**

Run: `git checkout -b feature/spec-upgrader-v1`
Expected: `Switched to a new branch 'feature/spec-upgrader-v1'`

- [ ] **Step 4: Commit the design doc + plan together**

Run:
```bash
git add docs/plans/2026-04-17-spec-upgrader-design.md docs/plans/2026-04-17-spec-upgrader-plan.md
git commit -m "docs(plans): SpecUpgrader v1 design + implementation plan"
```
Expected: clean commit, no hook failures.

- [ ] **Step 5: Push the branch**

Run: `git push -u origin feature/spec-upgrader-v1`
Expected: branch created upstream.

---

## Task 1: Core types

**Files:**
- Create: `aragora/swarm/spec_upgrader.py`
- Test: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write the failing test for `UpgradeFailureContext` construction**

Create `tests/swarm/test_spec_upgrader.py` with:

```python
"""Unit tests for SpecUpgrader."""
from __future__ import annotations

import pytest

from aragora.swarm.spec_upgrader import (
    SpecUpgraderUnavailable,
    UpgradeFailureContext,
    UpgradeResult,
)


def test_upgrade_failure_context_construction():
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion", "file-scope hint"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="Do the thing.",
        issue_title="[TW-02] Improve X",
        track_tag="TW-02",
    )
    assert ctx.missing_bounds == ["acceptance criterion", "file-scope hint"]
    assert ctx.prior_attempts == 0
    assert ctx.track_tag == "TW-02"


def test_upgrade_failure_context_frozen():
    ctx = UpgradeFailureContext(
        missing_bounds=[],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="",
        issue_title="",
        track_tag=None,
    )
    with pytest.raises(Exception):  # dataclass(frozen=True) raises FrozenInstanceError
        ctx.prior_attempts = 1  # type: ignore[misc]


def test_upgrade_result_upgraded_shape():
    from aragora.swarm.spec import SwarmSpec  # existing type
    spec = SwarmSpec.empty() if hasattr(SwarmSpec, "empty") else SwarmSpec()  # adapt to constructor
    res = UpgradeResult(
        status="upgraded",
        upgraded_spec=spec,
        audit_markdown="stub",
        attempt_count=1,
        upgrade_path="deterministic",
        failure_context=UpgradeFailureContext(
            missing_bounds=[], preflight_diff=None, prior_attempts=0,
            original_issue_body="", issue_title="", track_tag=None,
        ),
        unresolved_questions=[],
    )
    assert res.status == "upgraded"
    assert res.upgraded_spec is spec
    assert res.unresolved_questions == []


def test_upgrade_result_escalated_shape():
    res = UpgradeResult(
        status="escalated",
        upgraded_spec=None,
        audit_markdown="stub",
        attempt_count=2,
        upgrade_path="deterministic+llm",
        failure_context=UpgradeFailureContext(
            missing_bounds=["acceptance criterion"],
            preflight_diff=None, prior_attempts=2,
            original_issue_body="", issue_title="", track_tag=None,
        ),
        unresolved_questions=["What is the acceptance criterion?"],
    )
    assert res.status == "escalated"
    assert res.upgraded_spec is None
    assert len(res.unresolved_questions) == 1


def test_spec_upgrader_unavailable_is_exception():
    with pytest.raises(SpecUpgraderUnavailable):
        raise SpecUpgraderUnavailable("LLM client timed out")
```

- [ ] **Step 2: Run tests to verify import failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v`
Expected: `ImportError: cannot import name 'SpecUpgraderUnavailable' from 'aragora.swarm.spec_upgrader'` or module not found.

- [ ] **Step 3: Create the module with types**

Create `aragora/swarm/spec_upgrader.py` with:

```python
"""SpecUpgrader: convert weak GitHub-issue specs into dispatchable SwarmSpecs.

Public entry point: `upgrade_spec()`. See docs/plans/2026-04-17-spec-upgrader-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
    """Outcome of an upgrade attempt. Tagged union via `status` field."""

    status: UpgradeStatus
    upgraded_spec: SwarmSpec | None
    audit_markdown: str
    attempt_count: int
    upgrade_path: UpgradePath | None
    failure_context: UpgradeFailureContext
    unresolved_questions: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): core types (UpgradeFailureContext, UpgradeResult, SpecUpgraderUnavailable)"
```

---

## Task 2: Tier 1 deterministic — parse `missing_bounds` into target fields

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from aragora.swarm.spec_upgrader import _classify_missing_bounds


def test_classify_missing_bounds_all_categories():
    bounds = [
        "acceptance criterion",
        "file-scope hint",
        "constraint",
        "work order",
    ]
    result = _classify_missing_bounds(bounds)
    assert result == {
        "needs_acceptance": True,
        "needs_file_scope": True,
        "needs_constraint": True,
        "needs_work_order": True,
    }


def test_classify_missing_bounds_partial():
    bounds = ["acceptance criterion"]
    result = _classify_missing_bounds(bounds)
    assert result["needs_acceptance"] is True
    assert result["needs_file_scope"] is False


def test_classify_missing_bounds_empty():
    result = _classify_missing_bounds([])
    assert all(v is False for v in result.values())
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py::test_classify_missing_bounds_all_categories -v`
Expected: `ImportError: cannot import name '_classify_missing_bounds'`

- [ ] **Step 3: Implement `_classify_missing_bounds`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
_BOUND_LABELS = {
    "acceptance criterion": "needs_acceptance",
    "file-scope hint": "needs_file_scope",
    "constraint": "needs_constraint",
    "work order": "needs_work_order",
}


def _classify_missing_bounds(missing_bounds: list[str]) -> dict[str, bool]:
    """Map `missing_dispatch_bounds()` labels to actionable flags for enrichment."""
    classified = {flag: False for flag in _BOUND_LABELS.values()}
    for label in missing_bounds:
        flag = _BOUND_LABELS.get(label)
        if flag is not None:
            classified[flag] = True
    return classified
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k classify`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): classify missing_bounds into enrichment flags"
```

---

## Task 3: Tier 1 deterministic — extract file paths from issue body

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from pathlib import Path
from aragora.swarm.spec_upgrader import _extract_file_paths


def test_extract_file_paths_from_body(tmp_path, monkeypatch):
    # Create fake repo files
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    (tmp_path / "aragora" / "swarm" / "spec.py").write_text("")
    monkeypatch.chdir(tmp_path)

    body = (
        "Fix the thing in `aragora/swarm/boss_loop.py` and also "
        "the parser at aragora/swarm/spec.py. This imaginary/path.py does not exist."
    )
    paths = _extract_file_paths(body, repo_root=Path(tmp_path))
    assert "aragora/swarm/boss_loop.py" in paths
    assert "aragora/swarm/spec.py" in paths
    assert "imaginary/path.py" not in paths


def test_extract_file_paths_empty_body(tmp_path):
    assert _extract_file_paths("", repo_root=Path(tmp_path)) == []


def test_extract_file_paths_no_matches(tmp_path):
    body = "This issue has no file references, just prose."
    assert _extract_file_paths(body, repo_root=Path(tmp_path)) == []
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py::test_extract_file_paths_from_body -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `_extract_file_paths`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
import re
from pathlib import Path


# Matches common Python/TS/MD file references. Intentionally narrow to avoid false positives.
_PATH_RE = re.compile(r"(?P<path>[a-zA-Z0-9_\-./]+\.(?:py|ts|tsx|js|jsx|md|yaml|yml|json|sh))")


def _extract_file_paths(issue_body: str, *, repo_root: Path) -> list[str]:
    """Extract file paths mentioned in issue body and validate they exist in the repo.

    Only paths that actually exist (relative to repo_root) are returned. Hallucinated
    paths from the issue author are dropped.
    """
    candidates = set()
    for match in _PATH_RE.finditer(issue_body):
        candidate = match.group("path").strip("./")
        if "/" in candidate and (repo_root / candidate).is_file():
            candidates.add(candidate)
    return sorted(candidates)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k extract_file_paths`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): extract and validate file paths from issue body"
```

---

## Task 4: Tier 1 deterministic — track-tag scope inference with repo validation

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from aragora.swarm.spec_upgrader import _infer_track_scope


def test_infer_track_scope_tw_validates_repo(tmp_path, monkeypatch):
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "__init__.py").write_text("")

    hints = _infer_track_scope("TW-02", issue_body="refactor boss_loop logic", repo_root=Path(tmp_path))
    assert hints == ["aragora/swarm/"]


def test_infer_track_scope_unknown_tag_returns_empty(tmp_path):
    hints = _infer_track_scope("XYZ-99", issue_body="", repo_root=Path(tmp_path))
    assert hints == []


def test_infer_track_scope_design_heavy_returns_empty(tmp_path):
    # AGT-*/DIC-* are vision-layer; must not guess paths
    assert _infer_track_scope("AGT-01", issue_body="", repo_root=Path(tmp_path)) == []
    assert _infer_track_scope("DIC-15", issue_body="", repo_root=Path(tmp_path)) == []


def test_infer_track_scope_missing_directory_drops_hint(tmp_path):
    # Repo doesn't have aragora/swarm/ - hint is not validated, returns empty
    hints = _infer_track_scope("TW-02", issue_body="", repo_root=Path(tmp_path))
    assert hints == []
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k infer_track_scope`
Expected: `ImportError`.

- [ ] **Step 3: Implement `_infer_track_scope`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
# Low-confidence candidate scopes per track-tag prefix.
# Must be validated against repo before merging into spec.
_TRACK_SCOPE_CANDIDATES: dict[str, list[str]] = {
    "TW": ["aragora/swarm/"],
    "CS": ["aragora/swarm/", "docs/status/"],
    "RS": ["aragora/swarm/"],
}
# Design-heavy tracks that must NOT use path inference - fall through to LLM/escalation.
_DESIGN_HEAVY_PREFIXES = frozenset({"AGT", "DIC"})


def _infer_track_scope(track_tag: str | None, *, issue_body: str, repo_root: Path) -> list[str]:
    """Return validated candidate scope hints for the given track tag, or [] to fall through."""
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k infer_track_scope`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): validated track-tag scope inference (design-heavy tracks skipped)"
```

---

## Task 5: Tier 1 deterministic — translate preflight drift into acceptance criteria

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from aragora.swarm.spec_upgrader import _drift_to_acceptance_criterion


def test_drift_files_mismatch_generates_scoping_criterion():
    drift = {
        "expected": {"files": ["aragora/swarm/a.py"]},
        "actual": {"files": ["aragora/swarm/a.py", "unrelated/b.py"]},
    }
    crit = _drift_to_acceptance_criterion(drift)
    assert crit is not None
    assert "aragora/swarm/a.py" in crit
    assert "unrelated/b.py" not in crit  # Don't name disallowed paths positively
    assert "scope" in crit.lower() or "restrict" in crit.lower()


def test_drift_none_returns_none():
    assert _drift_to_acceptance_criterion(None) is None


def test_drift_identical_returns_none():
    drift = {"expected": {"files": ["a"]}, "actual": {"files": ["a"]}}
    assert _drift_to_acceptance_criterion(drift) is None
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k drift`
Expected: `ImportError`.

- [ ] **Step 3: Implement `_drift_to_acceptance_criterion`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
def _drift_to_acceptance_criterion(drift: dict | None) -> str | None:
    """Translate preflight contract drift into an actionable acceptance criterion.

    Returns None if drift is absent or matches (no drift to correct).
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k drift`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): translate contract drift into scoping acceptance criterion"
```

---

## Task 6: Tier 1 orchestrator — combine all enrichments

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from aragora.swarm.spec_upgrader import _tier1_enrich
from aragora.swarm.spec import SwarmSpec


def _make_unbounded_spec():
    """Build a minimally-underspecified SwarmSpec for testing."""
    # Adapt fields to match actual SwarmSpec constructor in aragora/swarm/spec.py.
    return SwarmSpec(
        goal="Improve boss_loop",
        acceptance_criteria=[],
        constraints=[],
        file_scope_hints=[],
        work_orders=[],
    )


def test_tier1_enriches_from_body_and_track_tag(tmp_path, monkeypatch):
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    (tmp_path / "aragora" / "swarm" / "__init__.py").write_text("")

    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion", "file-scope hint"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="Fix bugs in `aragora/swarm/boss_loop.py`.",
        issue_title="[TW-02] Fix boss loop bugs",
        track_tag="TW-02",
    )
    upgraded = _tier1_enrich(spec, ctx, repo_root=Path(tmp_path))
    assert upgraded is not None
    assert "aragora/swarm/boss_loop.py" in upgraded.file_scope_hints
    assert upgraded.acceptance_criteria  # non-empty after enrichment


def test_tier1_returns_none_when_cannot_bound(tmp_path):
    # No body content, no track tag, no drift - nothing to enrich from
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion", "file-scope hint", "constraint", "work order"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="",
        issue_title="[AGT-01] Design-heavy ambiguous",
        track_tag="AGT-01",
    )
    result = _tier1_enrich(spec, ctx, repo_root=Path(tmp_path))
    assert result is None
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k tier1`
Expected: `ImportError`.

- [ ] **Step 3: Implement `_tier1_enrich`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
from dataclasses import replace


def _tier1_enrich(
    spec: SwarmSpec,
    ctx: UpgradeFailureContext,
    *,
    repo_root: Path,
) -> SwarmSpec | None:
    """Deterministic enrichment: fill missing_bounds from body, track-tag, and drift.

    Returns an upgraded SwarmSpec if the enrichment bounds it, else None to signal
    that Tier 2 (LLM) is needed.
    """
    flags = _classify_missing_bounds(ctx.missing_bounds)
    extracted_paths = _extract_file_paths(ctx.original_issue_body, repo_root=repo_root)
    track_hints = _infer_track_scope(
        ctx.track_tag, issue_body=ctx.original_issue_body, repo_root=repo_root,
    )
    drift_crit = _drift_to_acceptance_criterion(ctx.preflight_diff)

    # Build enrichments
    new_file_scope = list(spec.file_scope_hints)
    if flags["needs_file_scope"]:
        new_file_scope.extend(p for p in extracted_paths if p not in new_file_scope)
        new_file_scope.extend(p for p in track_hints if p not in new_file_scope)

    new_acceptance = list(spec.acceptance_criteria)
    if flags["needs_acceptance"]:
        if drift_crit and drift_crit not in new_acceptance:
            new_acceptance.append(drift_crit)
        # Synthesise a generic acceptance from the title if still empty
        if not new_acceptance and ctx.issue_title:
            new_acceptance.append(
                f"Implement the behavior described by: {ctx.issue_title.strip()}"
            )

    new_constraints = list(spec.constraints)
    if flags["needs_constraint"] and new_file_scope:
        constraint = (
            f"Limit modifications to the listed file-scope hints: "
            f"{', '.join(new_file_scope)}."
        )
        if constraint not in new_constraints:
            new_constraints.append(constraint)

    new_work_orders = list(spec.work_orders)
    if flags["needs_work_order"] and new_acceptance:
        # Seed a minimal work order from acceptance criteria if no other source
        new_work_orders.append(f"Satisfy: {new_acceptance[0]}")

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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k tier1`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): tier-1 deterministic enrichment orchestrator"
```

---

## Task 7: Tier 2 LLM enrichment

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from unittest.mock import MagicMock
from aragora.swarm.spec_upgrader import _tier2_enrich


def test_tier2_enrich_success(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="Ambiguous task.",
        issue_title="[CS-01] Stuff",
        track_tag="CS-01",
    )
    mock_client = MagicMock()
    mock_client.complete.return_value = (
        '{"acceptance_criteria": ["The code produces output matching docs/examples/X.md"], '
        '"file_scope_hints": ["aragora/swarm/boss_loop.py"], '
        '"constraints": ["No changes outside listed files"], '
        '"work_orders": ["Add regression test for X"]}'
    )
    result = _tier2_enrich(spec, ctx, client=mock_client, repo_root=Path(tmp_path))
    assert result is not None
    assert result.acceptance_criteria


def test_tier2_enrich_malformed_json_raises(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"], preflight_diff=None, prior_attempts=0,
        original_issue_body="", issue_title="", track_tag=None,
    )
    mock_client = MagicMock()
    mock_client.complete.return_value = "this is not json"
    from aragora.swarm.spec_upgrader import _LLMLogicFailure
    with pytest.raises(_LLMLogicFailure):
        _tier2_enrich(spec, ctx, client=mock_client, repo_root=Path(tmp_path))


def test_tier2_enrich_transient_raises_unavailable(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"], preflight_diff=None, prior_attempts=0,
        original_issue_body="", issue_title="", track_tag=None,
    )
    mock_client = MagicMock()
    mock_client.complete.side_effect = ConnectionError("api 503")
    with pytest.raises(SpecUpgraderUnavailable):
        _tier2_enrich(spec, ctx, client=mock_client, repo_root=Path(tmp_path))
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k tier2`
Expected: `ImportError`.

- [ ] **Step 3: Implement `_tier2_enrich`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
import json
import time


class _LLMLogicFailure(Exception):
    """Internal: LLM returned malformed/ungrounded output after local retry."""


def _tier2_enrich(
    spec: SwarmSpec,
    ctx: UpgradeFailureContext,
    *,
    client,
    repo_root: Path,
) -> SwarmSpec | None:
    """LLM-backed enrichment. Raises SpecUpgraderUnavailable on transient infra errors,
    raises _LLMLogicFailure on malformed/ungrounded output after one local retry.
    Returns upgraded SwarmSpec on success, or None if the upgrade still isn't bounded
    (caller treats as logic failure → escalate)."""

    prompt = _build_tier2_prompt(spec, ctx, repo_root)
    last_err: Exception | None = None

    for attempt in range(2):
        try:
            raw = client.complete(prompt)
        except (ConnectionError, TimeoutError) as exc:
            raise SpecUpgraderUnavailable(str(exc)) from exc
        except Exception as exc:  # anything else during client call
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

        # Validate shape
        if not isinstance(parsed, dict):
            raise _LLMLogicFailure("LLM output not a JSON object")

        candidate = replace(
            spec,
            acceptance_criteria=list(spec.acceptance_criteria) + parsed.get("acceptance_criteria", []),
            file_scope_hints=list(spec.file_scope_hints) + parsed.get("file_scope_hints", []),
            constraints=list(spec.constraints) + parsed.get("constraints", []),
            work_orders=list(spec.work_orders) + parsed.get("work_orders", []),
        )
        if candidate.is_dispatch_bounded():
            return candidate
        # Still unbounded even after LLM enrichment - caller escalates
        return None

    raise _LLMLogicFailure(f"Exhausted LLM attempts: {last_err}")


def _build_tier2_prompt(spec: SwarmSpec, ctx: UpgradeFailureContext, repo_root: Path) -> str:
    """Build an LLM prompt from spec + failure context. Kept simple for v1; iterate later."""
    return f"""You are upgrading an underspecified GitHub issue into a dispatchable SwarmSpec.

Issue title: {ctx.issue_title}
Issue body:
{ctx.original_issue_body}

Missing bounds: {ctx.missing_bounds}
Preflight drift: {json.dumps(ctx.preflight_diff) if ctx.preflight_diff else 'none'}

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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k tier2`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): tier-2 LLM enrichment with transient/logic failure split"
```

---

## Task 8: Audit marker parsing

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from aragora.swarm.spec_upgrader import _parse_audit_marker


def test_parse_audit_marker_valid():
    comment = "<!-- spec-upgraded:v1 attempt=1 -->\n\n## Upgrade audit\nblah blah"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 1
    assert valid is True


def test_parse_audit_marker_attempt_2():
    comment = "<!-- spec-upgraded:v1 attempt=2 -->\ncontent"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 2
    assert valid is True


def test_parse_audit_marker_corrupted_returns_max():
    # Marker present but unparseable → conservative: treat as max attempts reached
    comment = "<!-- spec-upgraded:v1 attempt=garbage -->\ncontent"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 2  # max_attempts sentinel
    assert valid is False


def test_parse_audit_marker_wrong_version():
    comment = "<!-- spec-upgraded:v2 attempt=1 -->\ncontent"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 2  # treat unknown version as corrupted
    assert valid is False


def test_parse_audit_marker_no_marker():
    comment = "Some unrelated comment"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 0
    assert valid is True
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k audit_marker`
Expected: `ImportError`.

- [ ] **Step 3: Implement `_parse_audit_marker`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
_AUDIT_MARKER_RE = re.compile(
    r"<!--\s*spec-upgraded:v(?P<version>\d+)\s+attempt=(?P<attempt>\d+)\s*-->"
)
_AUDIT_MARKER_PRESENT_RE = re.compile(r"<!--\s*spec-upgraded:")
MAX_ATTEMPTS = 2


def _parse_audit_marker(comment_body: str) -> tuple[int, bool]:
    """Parse the attempt count from an audit comment.

    Returns (attempt_count, valid). If a marker is present but unparseable
    (corrupted or unknown version), returns (MAX_ATTEMPTS, False) to conservatively
    trigger escalation rather than reset the counter.
    """
    match = _AUDIT_MARKER_RE.search(comment_body)
    if match is not None and match.group("version") == "1":
        try:
            return int(match.group("attempt")), True
        except ValueError:
            return MAX_ATTEMPTS, False
    if _AUDIT_MARKER_PRESENT_RE.search(comment_body):
        # Marker-ish present but didn't parse - treat as corrupted.
        return MAX_ATTEMPTS, False
    return 0, True
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k audit_marker`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): durable audit marker parser with corruption handling"
```

---

## Task 9: Audit comment read/upsert via `gh`

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from unittest.mock import patch
from aragora.swarm.spec_upgrader import AuditPersistence


def test_audit_read_attempt_count_no_prior_marker():
    ap = AuditPersistence(issue_number=5898)
    with patch.object(ap, "_gh_list_comments", return_value=[
        {"id": 1, "body": "unrelated"},
        {"id": 2, "body": "also unrelated"},
    ]):
        count, valid = ap.read_attempt_count()
    assert count == 0
    assert valid is True


def test_audit_read_attempt_count_existing_marker():
    ap = AuditPersistence(issue_number=5898)
    with patch.object(ap, "_gh_list_comments", return_value=[
        {"id": 1, "body": "<!-- spec-upgraded:v1 attempt=1 -->\n## Upgrade audit"},
    ]):
        count, valid = ap.read_attempt_count()
    assert count == 1
    assert valid is True


def test_audit_upsert_creates_when_missing():
    ap = AuditPersistence(issue_number=5898)
    with patch.object(ap, "_gh_list_comments", return_value=[]) as _lc, \
         patch.object(ap, "_gh_create_comment") as cc, \
         patch.object(ap, "_gh_update_comment") as uc:
        ap.upsert(attempt=1, audit_markdown="## body")
        cc.assert_called_once()
        uc.assert_not_called()
        # verify marker is prepended
        args, kwargs = cc.call_args
        posted_body = kwargs.get("body") or args[-1]
        assert "<!-- spec-upgraded:v1 attempt=1 -->" in posted_body


def test_audit_upsert_updates_when_present():
    existing_comment = {"id": 42, "body": "<!-- spec-upgraded:v1 attempt=1 -->\nold"}
    ap = AuditPersistence(issue_number=5898)
    with patch.object(ap, "_gh_list_comments", return_value=[existing_comment]), \
         patch.object(ap, "_gh_create_comment") as cc, \
         patch.object(ap, "_gh_update_comment") as uc:
        ap.upsert(attempt=2, audit_markdown="## new body")
        uc.assert_called_once()
        cc.assert_not_called()
        args, kwargs = uc.call_args
        assert kwargs.get("comment_id") == 42 or args[1] == 42
        posted_body = kwargs.get("body") or args[-1]
        assert "attempt=2" in posted_body
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k audit`
Expected: `ImportError`.

- [ ] **Step 3: Implement `AuditPersistence`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
import subprocess


class AuditPersistence:
    """Idempotent upsert of the [spec-upgraded] audit comment on a GitHub issue."""

    MARKER_PREFIX = "<!-- spec-upgraded:v1"

    def __init__(self, issue_number: int, *, repo: str = "synaptent/aragora"):
        self.issue_number = issue_number
        self.repo = repo

    def read_attempt_count(self) -> tuple[int, bool]:
        """Scan comments for marker; return (attempt_count, marker_valid)."""
        comments = self._gh_list_comments()
        for c in comments:
            body = c.get("body") or ""
            if self.MARKER_PREFIX in body:
                return _parse_audit_marker(body)
        return 0, True

    def upsert(self, *, attempt: int, audit_markdown: str) -> bool:
        """Upsert the audit comment. Returns True on success, False on gh failure."""
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
        for c in self._gh_list_comments():
            if self.MARKER_PREFIX in (c.get("body") or ""):
                return c
        return None

    # --- gh wrappers (seams for test mocking) ---

    def _gh_list_comments(self) -> list[dict]:
        out = subprocess.check_output(
            ["gh", "issue", "view", str(self.issue_number),
             "--repo", self.repo, "--json", "comments",
             "--jq", ".comments"],
            text=True,
        )
        return json.loads(out or "[]")

    def _gh_create_comment(self, *, body: str) -> None:
        subprocess.check_call(
            ["gh", "issue", "comment", str(self.issue_number),
             "--repo", self.repo, "--body", body]
        )

    def _gh_update_comment(self, *, comment_id: int, body: str) -> None:
        # gh does not expose direct comment edit; use gh api
        subprocess.check_call(
            ["gh", "api", "--method", "PATCH",
             f"/repos/{self.repo}/issues/comments/{comment_id}",
             "-f", f"body={body}"]
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k audit`
Expected: 5 passed (includes the marker-parsing tests from Task 8).

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): AuditPersistence with gh-backed idempotent upsert"
```

---

## Task 10: Escalator (C-path)

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from aragora.swarm.spec_upgrader import Escalator


def test_escalator_success():
    esc = Escalator(issue_number=5898)
    with patch.object(esc, "_gh_add_label") as al, \
         patch.object(esc, "_gh_create_comment") as cc:
        success = esc.escalate(
            unresolved_questions=["What file scope?", "What acceptance criterion?"],
            failure_context_summary="Missing all bounds",
        )
    assert success is True
    al.assert_called_once()
    cc.assert_called_once()


def test_escalator_label_failure_is_fail_closed():
    esc = Escalator(issue_number=5898)
    with patch.object(esc, "_gh_add_label", side_effect=subprocess.CalledProcessError(1, "gh")), \
         patch.object(esc, "_gh_create_comment") as cc:
        success = esc.escalate(
            unresolved_questions=["Q1"],
            failure_context_summary="summary",
        )
    assert success is False
    cc.assert_not_called()  # label failed first, don't proceed


def test_escalator_comment_failure_is_fail_closed():
    esc = Escalator(issue_number=5898)
    with patch.object(esc, "_gh_add_label"), \
         patch.object(esc, "_gh_create_comment", side_effect=subprocess.CalledProcessError(1, "gh")):
        success = esc.escalate(
            unresolved_questions=["Q1"],
            failure_context_summary="summary",
        )
    assert success is False
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k escalator`
Expected: `ImportError`.

- [ ] **Step 3: Implement `Escalator`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
class Escalator:
    """Apply `needs-clarification` label + post unresolved-questions comment.

    Fail-closed: returns False if either label or comment mutation fails; caller
    must NOT dispatch the issue in that case.
    """

    LABEL = "needs-clarification"

    def __init__(self, issue_number: int, *, repo: str = "synaptent/aragora"):
        self.issue_number = issue_number
        self.repo = repo

    def escalate(self, *, unresolved_questions: list[str], failure_context_summary: str) -> bool:
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
            f"attempts. Human review required.\n\n"
            f"**Failure summary:** {summary}\n\n"
            f"**Unresolved questions:**\n{q_block}\n\n"
            "_Posted by SpecUpgrader._"
        )

    def _gh_add_label(self) -> None:
        subprocess.check_call(
            ["gh", "issue", "edit", str(self.issue_number),
             "--repo", self.repo, "--add-label", self.LABEL]
        )

    def _gh_create_comment(self, *, body: str) -> None:
        subprocess.check_call(
            ["gh", "issue", "comment", str(self.issue_number),
             "--repo", self.repo, "--body", body]
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k escalator`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): Escalator C-path with fail-closed gh mutations"
```

---

## Task 11: Telemetry helper

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from aragora.swarm.spec_upgrader import emit_upgrade_telemetry


def test_emit_upgrade_telemetry_writes_jsonl(tmp_path):
    metrics_path = tmp_path / "boss_metrics.jsonl"
    upgrade_id = emit_upgrade_telemetry(
        metrics_path=metrics_path,
        issue_number=5898,
        seam="A",
        attempt_count=1,
        status="upgraded",
        upgrade_path="deterministic",
        wall_clock_ms=432,
        audit_failed=False,
        escalation_failed=False,
        llm_tokens_in=0,
        llm_tokens_out=0,
        failure_reasons=["acceptance criterion"],
    )
    assert metrics_path.exists()
    line = metrics_path.read_text().strip()
    record = json.loads(line)
    assert record["event"] == "spec_upgrade"
    assert record["upgrade_id"] == upgrade_id
    assert record["issue_number"] == 5898
    assert record["seam"] == "A"
    assert record["status"] == "upgraded"


def test_emit_upgrade_telemetry_appends(tmp_path):
    metrics_path = tmp_path / "boss_metrics.jsonl"
    emit_upgrade_telemetry(
        metrics_path=metrics_path, issue_number=1, seam="A", attempt_count=1,
        status="upgraded", upgrade_path="deterministic", wall_clock_ms=1,
        audit_failed=False, escalation_failed=False,
        llm_tokens_in=0, llm_tokens_out=0, failure_reasons=[],
    )
    emit_upgrade_telemetry(
        metrics_path=metrics_path, issue_number=2, seam="B", attempt_count=2,
        status="escalated", upgrade_path="deterministic+llm", wall_clock_ms=2,
        audit_failed=False, escalation_failed=False,
        llm_tokens_in=10, llm_tokens_out=20, failure_reasons=["constraint"],
    )
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(l) for l in lines]
    assert recs[0]["issue_number"] == 1
    assert recs[1]["issue_number"] == 2
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k telemetry`
Expected: `ImportError`.

- [ ] **Step 3: Implement `emit_upgrade_telemetry`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
import uuid


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
    """Append a per-upgrade row to boss_metrics.jsonl. Returns the generated upgrade_id."""
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k telemetry`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): emit per-upgrade telemetry rows to boss_metrics.jsonl"
```

---

## Task 12: `upgrade_spec()` public entry point

**Files:**
- Modify: `aragora/swarm/spec_upgrader.py`
- Modify: `tests/swarm/test_spec_upgrader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/swarm/test_spec_upgrader.py`:

```python
from aragora.swarm.spec_upgrader import upgrade_spec


def test_upgrade_spec_tier1_success(tmp_path, monkeypatch):
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    metrics = tmp_path / "boss_metrics.jsonl"

    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion", "file-scope hint"],
        preflight_diff=None, prior_attempts=0,
        original_issue_body="Fix `aragora/swarm/boss_loop.py` behaviour.",
        issue_title="[TW-02] Fix boss loop",
        track_tag="TW-02",
    )

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = upgrade_spec(
            spec, ctx,
            issue_number=5898,
            seam="A",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,  # tier-1 should succeed without LLM
        )

    assert result.status == "upgraded"
    assert result.attempt_count == 1
    assert result.upgrade_path == "deterministic"
    assert metrics.exists()


def test_upgrade_spec_escalates_on_max_attempts(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"], preflight_diff=None, prior_attempts=0,
        original_issue_body="", issue_title="[CS-01] stuck", track_tag="CS-01",
    )
    metrics = tmp_path / "boss_metrics.jsonl"

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP, \
         patch("aragora.swarm.spec_upgrader.Escalator") as ESC:
        AP.return_value.read_attempt_count.return_value = (MAX_ATTEMPTS, True)
        ESC.return_value.escalate.return_value = True
        result = upgrade_spec(
            spec, ctx,
            issue_number=5903,
            seam="A",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )

    assert result.status == "escalated"
    ESC.return_value.escalate.assert_called_once()


def test_upgrade_spec_llm_unavailable_bubbles(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"], preflight_diff=None, prior_attempts=0,
        original_issue_body="ambiguous", issue_title="[CS-01] x", track_tag="CS-01",
    )
    metrics = tmp_path / "boss_metrics.jsonl"
    mock_client = MagicMock()
    mock_client.complete.side_effect = ConnectionError("timeout")

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        with pytest.raises(SpecUpgraderUnavailable):
            upgrade_spec(
                spec, ctx,
                issue_number=5903,
                seam="A",
                repo_root=Path(tmp_path),
                metrics_path=metrics,
                llm_client=mock_client,
            )
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader.py -v -k upgrade_spec`
Expected: `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement `upgrade_spec`**

Append to `aragora/swarm/spec_upgrader.py`:

```python
def upgrade_spec(
    spec: SwarmSpec,
    failure_context: UpgradeFailureContext,
    *,
    issue_number: int,
    seam: Literal["A", "B"],
    repo_root: Path,
    metrics_path: Path,
    llm_client=None,
    max_attempts: int = MAX_ATTEMPTS,
) -> UpgradeResult:
    """Public entry point. See docs/plans/2026-04-17-spec-upgrader-design.md."""
    start = time.monotonic()

    audit = AuditPersistence(issue_number=issue_number)
    prior_attempts, marker_valid = audit.read_attempt_count()

    # Marker corrupted OR budget exhausted → escalate
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
            metrics_path=metrics_path, issue_number=issue_number, seam=seam,
            attempt_count=prior_attempts, status="escalated", upgrade_path=None,
            wall_clock_ms=elapsed, audit_failed=False,
            escalation_failed=not escalated_ok,
            llm_tokens_in=0, llm_tokens_out=0,
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

    # Tier 1
    upgraded = _tier1_enrich(spec, failure_context, repo_root=repo_root)

    # Tier 2 if Tier 1 insufficient
    if upgraded is None and llm_client is not None:
        try:
            upgraded = _tier2_enrich(spec, failure_context, client=llm_client, repo_root=repo_root)
            path_taken = "deterministic+llm"
        except _LLMLogicFailure:
            upgraded = None

    elapsed = int((time.monotonic() - start) * 1000)

    if upgraded is None:
        # Escalate — Tier 1 and Tier 2 both failed
        questions = _derive_questions(failure_context)
        summary = _summarise_failure(failure_context)
        esc = Escalator(issue_number=issue_number)
        escalated_ok = esc.escalate(unresolved_questions=questions, failure_context_summary=summary)
        emit_upgrade_telemetry(
            metrics_path=metrics_path, issue_number=issue_number, seam=seam,
            attempt_count=attempt, status="escalated", upgrade_path=path_taken,
            wall_clock_ms=elapsed, audit_failed=False,
            escalation_failed=not escalated_ok,
            llm_tokens_in=0, llm_tokens_out=0,
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

    # Success path - persist audit
    audit_md = _render_audit(attempt, path_taken, failure_context, escalated=False)
    audit_ok = audit.upsert(attempt=attempt, audit_markdown=audit_md)
    emit_upgrade_telemetry(
        metrics_path=metrics_path, issue_number=issue_number, seam=seam,
        attempt_count=attempt, status="upgraded", upgrade_path=path_taken,
        wall_clock_ms=elapsed, audit_failed=not audit_ok,
        escalation_failed=False,
        llm_tokens_in=0, llm_tokens_out=0,
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


def _derive_questions(ctx: UpgradeFailureContext) -> list[str]:
    """Convert missing_bounds into reviewer-facing clarifying questions."""
    q = []
    if "acceptance criterion" in ctx.missing_bounds:
        q.append("What observable behaviour proves this issue is resolved?")
    if "file-scope hint" in ctx.missing_bounds:
        q.append("Which files (exact paths) should be modified?")
    if "constraint" in ctx.missing_bounds:
        q.append("Are there files, APIs, or behaviours that must NOT change?")
    if "work order" in ctx.missing_bounds:
        q.append("What concrete steps should an implementer take?")
    return q


def _summarise_failure(ctx: UpgradeFailureContext) -> str:
    parts = [f"missing: {', '.join(ctx.missing_bounds)}"] if ctx.missing_bounds else []
    if ctx.preflight_diff:
        parts.append("preflight contract drift detected")
    return "; ".join(parts) or "underspecified"


def _render_audit(
    attempt: int,
    path: UpgradePath,
    ctx: UpgradeFailureContext,
    *,
    escalated: bool,
) -> str:
    verdict = "ESCALATED" if escalated else "UPGRADED"
    return (
        f"## Upgrade audit\n\n"
        f"- **Attempt:** {attempt}\n"
        f"- **Path:** {path}\n"
        f"- **Verdict:** {verdict}\n"
        f"- **Missing bounds on entry:** {', '.join(ctx.missing_bounds) or 'none'}\n"
        f"- **Preflight drift:** {'yes' if ctx.preflight_diff else 'no'}\n"
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/spec_upgrader.py tests/swarm/test_spec_upgrader.py
git commit -m "feat(spec-upgrader): upgrade_spec() entry point with full orchestration"
```

---

## Task 13: Seam A integration — replace `maybe_upgrade_dispatch_spec`

**Files:**
- Modify: `aragora/swarm/boss_worker_lifecycle.py` (near line 801)
- Modify: `aragora/swarm/dispatch_followups.py`
- Test: `tests/swarm/test_spec_upgrader_integration.py` (create)

- [ ] **Step 1: Read the existing code to confirm line numbers**

Run: `grep -n 'maybe_upgrade_dispatch_spec' aragora/swarm/boss_worker_lifecycle.py aragora/swarm/dispatch_followups.py`

Verify the current call path. The `dispatch_followups.maybe_upgrade_dispatch_spec` function currently delegates to `issue_upgrader.upgrade_issue_heuristic`. Record the exact line number for later replacement.

- [ ] **Step 2: Write integration test**

Create `tests/swarm/test_spec_upgrader_integration.py`:

```python
"""Integration tests: SpecUpgrader wired into dispatch_followups."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.swarm.dispatch_followups import maybe_upgrade_dispatch_spec
from aragora.swarm.spec import SwarmSpec


def _make_unbounded_spec():
    return SwarmSpec(
        goal="Improve thing",
        acceptance_criteria=[],
        constraints=[],
        file_scope_hints=[],
        work_orders=[],
    )


def test_followup_routes_through_spec_upgrader(tmp_path, monkeypatch):
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    monkeypatch.chdir(tmp_path)
    metrics = tmp_path / "boss_metrics.jsonl"

    spec = _make_unbounded_spec()
    issue_body = "Improve `aragora/swarm/boss_loop.py`."
    issue_title = "[TW-02] Improve boss_loop"

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = maybe_upgrade_dispatch_spec(
            spec,
            issue_number=5898,
            issue_title=issue_title,
            issue_body=issue_body,
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )
    assert result is not None
    assert result.is_dispatch_bounded()
```

- [ ] **Step 3: Run integration test to verify it fails**

Run: `pytest tests/swarm/test_spec_upgrader_integration.py -v`
Expected: Fails (either `maybe_upgrade_dispatch_spec` signature mismatch or the new params don't exist).

- [ ] **Step 4: Update `dispatch_followups.maybe_upgrade_dispatch_spec`**

Open `aragora/swarm/dispatch_followups.py`. Replace the body of `maybe_upgrade_dispatch_spec()` with:

```python
from __future__ import annotations

from pathlib import Path

from aragora.swarm.spec import SwarmSpec
from aragora.swarm.spec_upgrader import (
    SpecUpgraderUnavailable,
    UpgradeFailureContext,
    upgrade_spec,
)


def _extract_track_tag(issue_title: str) -> str | None:
    """Extract [TW-02] style prefix from an issue title."""
    import re
    m = re.match(r"\s*\[([A-Z]+-\d+)\]", issue_title)
    return m.group(1) if m else None


def maybe_upgrade_dispatch_spec(
    spec: SwarmSpec,
    *,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    repo_root: Path,
    metrics_path: Path,
    llm_client=None,
) -> SwarmSpec | None:
    """Seam A: upgrade an unbounded spec before contract-gate dispatch.

    Returns the upgraded SwarmSpec if dispatch should proceed, None if the upgrader
    escalated to `needs-clarification` (caller must skip dispatch).

    Raises SpecUpgraderUnavailable on transient infrastructure failure - caller treats
    as skip-for-this-tick.
    """
    if spec.is_dispatch_bounded():
        return spec  # nothing to do

    ctx = UpgradeFailureContext(
        missing_bounds=spec.missing_dispatch_bounds(),
        preflight_diff=None,
        prior_attempts=0,  # read durably inside upgrade_spec
        original_issue_body=issue_body,
        issue_title=issue_title,
        track_tag=_extract_track_tag(issue_title),
    )

    try:
        result = upgrade_spec(
            spec, ctx,
            issue_number=issue_number,
            seam="A",
            repo_root=repo_root,
            metrics_path=metrics_path,
            llm_client=llm_client,
        )
    except SpecUpgraderUnavailable:
        raise  # bubble up; caller skips this tick

    if result.status == "upgraded":
        return result.upgraded_spec
    return None  # escalated
```

- [ ] **Step 5: Update the call site in `boss_worker_lifecycle.py`**

Open `aragora/swarm/boss_worker_lifecycle.py` around line 801. Update the call to `maybe_upgrade_dispatch_spec` to pass the new arguments (`issue_number`, `issue_title`, `issue_body`, `repo_root`, `metrics_path`, `llm_client`). The exact before/after depends on current signature; read the surrounding 20 lines first and thread the existing variables (they already exist in that scope — issue object, repo path, metrics path from config).

After edit, verify the LOC delta is small. Keep the diff under ~30 lines in this file to preserve the ratchet.

- [ ] **Step 6: Run integration tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader_integration.py -v && pytest tests/swarm/test_spec_upgrader.py -v`
Expected: all green.

- [ ] **Step 7: Run the full swarm test suite for regression check**

Run: `pytest tests/swarm/ -v --timeout=60`
Expected: no new failures vs main baseline.

- [ ] **Step 8: Commit**

```bash
git add aragora/swarm/dispatch_followups.py aragora/swarm/boss_worker_lifecycle.py \
        tests/swarm/test_spec_upgrader_integration.py
git commit -m "feat(spec-upgrader): Seam A integration replacing heuristic-only follow-up"
```

---

## Task 14: Seam B integration — drift feedback into upgrade

**Files:**
- Modify: `aragora/swarm/boss_worker_lifecycle.py` (near line 876, contract-gate failure path)
- Modify: `tests/swarm/test_spec_upgrader_integration.py`

- [ ] **Step 1: Read the existing contract-gate failure handling**

Run: `grep -n 'dispatch_contract_gate' aragora/swarm/boss_worker_lifecycle.py`

Identify the block that handles the failure return (around line 876 per the design doc reference). Read 40 lines of surrounding context to understand what variables are in scope.

- [ ] **Step 2: Write Seam-B integration test**

Append to `tests/swarm/test_spec_upgrader_integration.py`:

```python
def test_seam_b_upgrades_on_contract_drift(tmp_path):
    """When contract gate fails with drift, SpecUpgrader is called with preflight_diff."""
    # This is a focused test of the Seam B helper function (added in this task).
    from aragora.swarm.dispatch_followups import upgrade_on_contract_drift

    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    metrics = tmp_path / "boss_metrics.jsonl"

    spec = SwarmSpec(
        goal="Improve",
        acceptance_criteria=["Do the thing"],
        constraints=[],
        file_scope_hints=["aragora/swarm/boss_loop.py"],
        work_orders=["Satisfy: Do the thing"],
    )  # already bounded; drift is the only reason to upgrade
    drift = {
        "expected": {"files": ["aragora/swarm/boss_loop.py"]},
        "actual": {"files": ["aragora/swarm/boss_loop.py", "unrelated.py"]},
    }

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = upgrade_on_contract_drift(
            spec,
            issue_number=5898,
            issue_title="[TW-02] Fix",
            issue_body="Fix `aragora/swarm/boss_loop.py`.",
            preflight_diff=drift,
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )
    assert result is not None
    # Scoping criterion from drift should be present
    assert any("scope" in c.lower() for c in result.acceptance_criteria)
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/swarm/test_spec_upgrader_integration.py -v -k seam_b`
Expected: `ImportError: cannot import name 'upgrade_on_contract_drift'`.

- [ ] **Step 4: Add `upgrade_on_contract_drift` helper in `dispatch_followups.py`**

Append to `aragora/swarm/dispatch_followups.py`:

```python
def upgrade_on_contract_drift(
    spec: SwarmSpec,
    *,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    preflight_diff: dict,
    repo_root: Path,
    metrics_path: Path,
    llm_client=None,
) -> SwarmSpec | None:
    """Seam B: upgrade a spec after contract-gate reported drift.

    Returns upgraded spec to retry dispatch, or None to escalate (caller skips).
    Raises SpecUpgraderUnavailable on transient infra failure.
    """
    missing = list(spec.missing_dispatch_bounds())
    ctx = UpgradeFailureContext(
        missing_bounds=missing,
        preflight_diff=preflight_diff,
        prior_attempts=0,  # read durably inside upgrade_spec
        original_issue_body=issue_body,
        issue_title=issue_title,
        track_tag=_extract_track_tag(issue_title),
    )
    result = upgrade_spec(
        spec, ctx,
        issue_number=issue_number,
        seam="B",
        repo_root=repo_root,
        metrics_path=metrics_path,
        llm_client=llm_client,
    )
    if result.status == "upgraded":
        return result.upgraded_spec
    return None
```

- [ ] **Step 5: Wire the call site in `boss_worker_lifecycle.py`**

In the contract-gate failure handler near line 876, where a dispatch currently terminates with `blocked_not_dispatch_bounded` due to contract drift, insert a call to `upgrade_on_contract_drift`. If it returns a spec, re-enter the contract gate once. If None, fall through to the existing failure termination.

Pseudocode for the site:

```python
# Existing failure detection (preserve exactly):
gate_result = dispatch_contract_gate(...)
if not gate_result.ok and gate_result.drift_detected:
    # NEW: Seam B
    try:
        upgraded = upgrade_on_contract_drift(
            spec,
            issue_number=issue.number,
            issue_title=issue.title,
            issue_body=issue.body,
            preflight_diff=gate_result.drift,
            repo_root=repo_root,
            metrics_path=metrics_path,
            llm_client=llm_client,
        )
    except SpecUpgraderUnavailable:
        return existing_skip_this_tick_path(...)
    if upgraded is not None:
        spec = upgraded
        gate_result = dispatch_contract_gate(...)  # one retry with upgraded spec
        if not gate_result.ok:
            return existing_blocked_result(...)  # final escalation
    else:
        return existing_blocked_result(...)  # escalated inside upgrader
```

Keep the net LOC delta small (<~30 lines). Respect the ratchet.

- [ ] **Step 6: Run tests to verify pass**

Run: `pytest tests/swarm/test_spec_upgrader_integration.py -v && pytest tests/swarm/ -v --timeout=60`
Expected: all green, no regressions.

- [ ] **Step 7: Commit**

```bash
git add aragora/swarm/dispatch_followups.py aragora/swarm/boss_worker_lifecycle.py \
        tests/swarm/test_spec_upgrader_integration.py
git commit -m "feat(spec-upgrader): Seam B integration - upgrade on contract-gate drift"
```

---

## Task 15: Frozen fixtures for #5898 and #5903

**Files:**
- Create: `tests/swarm/fixtures/spec_upgrader/issue_5898.json`
- Create: `tests/swarm/fixtures/spec_upgrader/issue_5903.json`
- Create: `tests/swarm/fixtures/spec_upgrader/__init__.py` (empty, for pytest discovery if needed)

- [ ] **Step 1: Capture live state of #5898 and #5903 into fixtures**

Run:
```bash
mkdir -p tests/swarm/fixtures/spec_upgrader
gh issue view 5898 --repo synaptent/aragora --json number,title,body,labels,comments > tests/swarm/fixtures/spec_upgrader/issue_5898.json
gh issue view 5903 --repo synaptent/aragora --json number,title,body,labels,comments > tests/swarm/fixtures/spec_upgrader/issue_5903.json
```

Expected: both files exist and contain non-empty JSON bodies.

- [ ] **Step 2: Verify content**

Run: `jq '.title' tests/swarm/fixtures/spec_upgrader/issue_5898.json tests/swarm/fixtures/spec_upgrader/issue_5903.json`
Expected: non-null titles (TW-02 and CS-01 respectively).

- [ ] **Step 3: Commit fixtures**

```bash
git add tests/swarm/fixtures/spec_upgrader/issue_5898.json \
        tests/swarm/fixtures/spec_upgrader/issue_5903.json
git commit -m "test(spec-upgrader): frozen fixtures for #5898 and #5903"
```

---

## Task 16: E2E regression test using fixtures

**Files:**
- Modify: `tests/swarm/test_spec_upgrader_integration.py`

- [ ] **Step 1: Write failing E2E test**

Append to `tests/swarm/test_spec_upgrader_integration.py`:

```python
def test_e2e_issue_5898_gets_bounded(tmp_path, monkeypatch):
    """Frozen-fixture regression: #5898 can be upgraded to a bounded spec."""
    fixture = json.loads(
        Path("tests/swarm/fixtures/spec_upgrader/issue_5898.json").read_text()
    )

    # Realistic repo layout subset needed for path validation
    for p in ["aragora/swarm/", "tests/swarm/"]:
        (tmp_path / p).mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(tmp_path)
    metrics = tmp_path / "boss_metrics.jsonl"
    spec = SwarmSpec(
        goal=fixture["title"],
        acceptance_criteria=[], constraints=[], file_scope_hints=[], work_orders=[],
    )

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = maybe_upgrade_dispatch_spec(
            spec,
            issue_number=fixture["number"],
            issue_title=fixture["title"],
            issue_body=fixture.get("body") or "",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )

    assert result is not None, "Tier 1 should succeed for #5898 given its bounded title"
    assert result.is_dispatch_bounded()


def test_e2e_issue_5903_gets_bounded(tmp_path, monkeypatch):
    fixture = json.loads(
        Path("tests/swarm/fixtures/spec_upgrader/issue_5903.json").read_text()
    )
    for p in ["aragora/swarm/", "docs/status/"]:
        (tmp_path / p).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    metrics = tmp_path / "boss_metrics.jsonl"
    spec = SwarmSpec(
        goal=fixture["title"],
        acceptance_criteria=[], constraints=[], file_scope_hints=[], work_orders=[],
    )

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = maybe_upgrade_dispatch_spec(
            spec,
            issue_number=fixture["number"],
            issue_title=fixture["title"],
            issue_body=fixture.get("body") or "",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )
    assert result is not None, "Tier 1 should bound #5903 via CS-01 track hints"
    assert result.is_dispatch_bounded()
```

Note: Add `import json` at the top of the test file if not already present.

- [ ] **Step 2: Run E2E tests**

Run: `pytest tests/swarm/test_spec_upgrader_integration.py::test_e2e_issue_5898_gets_bounded tests/swarm/test_spec_upgrader_integration.py::test_e2e_issue_5903_gets_bounded -v`

Expected outcomes:
- If Tier 1 is sufficient for both: both pass. Done.
- If one requires Tier 2: test will fail with the assertion message. Investigate — if the fixture issue genuinely needs LLM enrichment, adjust the test to provide a mocked LLM client returning a plausible structured response.

- [ ] **Step 3: If a fixture requires LLM, add a mocked-LLM variant**

For any fixture where Tier 1 alone cannot bound, extend the test:

```python
mock_client = MagicMock()
mock_client.complete.return_value = json.dumps({
    "acceptance_criteria": ["<concrete criterion inferred from fixture body>"],
    "file_scope_hints": ["aragora/swarm/<inferred-file>.py"],
    "constraints": ["No changes outside listed files"],
    "work_orders": ["Add regression test for the acceptance criterion"],
})
# ... then pass llm_client=mock_client to maybe_upgrade_dispatch_spec
```

- [ ] **Step 4: Commit**

```bash
git add tests/swarm/test_spec_upgrader_integration.py
git commit -m "test(spec-upgrader): E2E regression for #5898 and #5903 fixtures"
```

---

## Task 17: Final validation — run full test suite and check ratchet

**Files:** none modified; validation only.

- [ ] **Step 1: Run the full swarm test suite**

Run: `pytest tests/swarm/ -v --timeout=120`
Expected: no new failures vs main baseline.

- [ ] **Step 2: Run ruff on new/modified files**

Run: `ruff check aragora/swarm/spec_upgrader.py aragora/swarm/dispatch_followups.py tests/swarm/test_spec_upgrader.py tests/swarm/test_spec_upgrader_integration.py`
Expected: no lint errors.

- [ ] **Step 3: Run mypy on the touched modules**

Run: `mypy aragora/swarm/spec_upgrader.py aragora/swarm/dispatch_followups.py`
Expected: no errors.

- [ ] **Step 4: Verify LOC ratchet on `boss_worker_lifecycle.py`**

Run: `wc -l aragora/swarm/boss_worker_lifecycle.py`

Expected: under the hard limit enforced by CI (see `.github/workflows/` for the exact threshold). If the diff pushed the file over the limit, factor the Seam B integration into a helper in `dispatch_followups.py` and shrink the call site.

- [ ] **Step 5: Run the automation preflight**

Run: `bash scripts/automation_pr_preflight.sh origin/main HEAD`
Expected: all checks pass.

- [ ] **Step 6: Push branch**

Run: `git push`
Expected: push to `feature/spec-upgrader-v1` on origin.

- [ ] **Step 7: Open draft PR**

Run:
```bash
gh pr create --draft --title "feat(swarm): SpecUpgrader v1 — upgrade weak specs instead of rejecting" \
  --body "$(cat <<'EOF'
## Summary
- Introduces `aragora/swarm/spec_upgrader.py` with `upgrade_spec()` entry point.
- Two integration seams in `boss_worker_lifecycle.py`: Seam A (pre-contract-gate) replaces the heuristic-only follow-up; Seam B (post-contract-gate) adds drift feedback into a targeted re-upgrade.
- Durable attempt counting via idempotent `[spec-upgraded]` comment; hard cap of 2 attempts, then `needs-clarification` escalation.
- Telemetry via per-upgrade rows in `boss_metrics.jsonl`.

## Design
See [docs/plans/2026-04-17-spec-upgrader-design.md](docs/plans/2026-04-17-spec-upgrader-design.md) and [docs/plans/2026-04-17-spec-upgrader-plan.md](docs/plans/2026-04-17-spec-upgrader-plan.md).

## Test plan
- [x] Unit tests: `pytest tests/swarm/test_spec_upgrader.py`
- [x] Integration tests: `pytest tests/swarm/test_spec_upgrader_integration.py`
- [x] Frozen fixture E2E for #5898 and #5903
- [x] Full swarm suite passes
- [x] ruff + mypy clean
- [x] Automation preflight passes
EOF
)"
```
Expected: PR URL returned.

---

## Self-review

**Spec coverage:** Every section of the design doc maps to a task:

- Architecture (single library boundary, `upgrade_spec()`) → Tasks 1, 12
- Tier 1 deterministic enrichment → Tasks 2–6
- Tier 2 LLM enrichment → Task 7
- PreflightGate (re-enter existing contract gate) → Task 14 (Seam B wires the retry)
- AuditPersistence → Tasks 8, 9
- Escalator (C-path) → Task 10
- Integration points (Seam A at line 801, Seam B at line 876) → Tasks 13, 14
- Return type (`UpgradeResult` with `upgraded | escalated`) → Task 1, verified in Task 12
- Error handling precedence → exercised in Task 12 test cases
- Telemetry (per-upgrade rows + `upgrade_refs`) → Task 11, emitted in Task 12. NOTE: `upgrade_refs` on dispatch records is a follow-up (v1.0.1) — boss_worker_lifecycle dispatch-record writes are out of scope for the ratchet-friendly v1 diff.
- Testing (unit/integration/E2E fixtures) → Tasks 2–12 (unit), 13–14 (integration), 15–16 (fixtures + E2E)
- Acceptance criteria → validated in Task 17

**Gap flagged:** dispatch-record `upgrade_refs` is not wired in v1. If blocking, add as Task 14.5; otherwise leave as an explicit follow-up.

**Placeholder scan:** Task 5, 13, 14 mention "around line 801" and "around line 876" because the exact line numbers may drift by the time an implementer reads this plan. Every such reference is backed by a `grep` step to locate the current line, and the implementation step names the function/symbol rather than relying on the line number alone.

**Type consistency:** `upgrade_spec()` signature matches across Tasks 12–14. `UpgradeFailureContext` fields are defined in Task 1 and consumed in Tasks 6, 7, 12, 13, 14. `UpgradeResult.status` is `"upgraded" | "escalated"` everywhere. `AuditPersistence` and `Escalator` class APIs are stable across Tasks 9, 10, 12.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-04-17-spec-upgrader-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
