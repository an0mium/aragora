---
title: Mission Cadence — M0 + M1 Implementation Plan
description: Mission Cadence — M0 + M1 Implementation Plan
---

# Mission Cadence — M0 + M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the durable Mission Queue (M0) and promote the Open Decision Receipt to a v1.0 governance contract with independent verification (M1) — entirely within Tier 0-1, no Tier 3-4 surfaces.

**Architecture:** M0 appends a `## Strategy Mission Queue` section to the existing Intake Register (single-register rule) and opens a tracking epic — this is the cross-session durable state. M1 authors the ODR **Versioning & Stability** contract and a **native↔ODR mapping**, then locks the emitter↔verifier contract with a checked-in example receipt that the standalone `aragora-verify` package verifies independently. No format-version bump, no signing, no production wiring (all parked Tier 3).

**Tech Stack:** Python 3, pytest, ruff/mypy (pre-commit), the in-repo `aragora/gauntlet/odr_export.py` emitter, the standalone `aragora-verify/` package (PyPI v0.1.0, zero Aragora dependency).

## Global Constraints

- **No Tier 3-4 surfaces in this plan.** Do NOT edit: `README.md`, `action.yml`, `.github/workflows/`, server verify endpoints, auto-signing-in-gate, `aragora_verify` published version/release, secrets, `CLAUDE.md`, `aragora/__init__.py`, `.env`, `scripts/nomic_loop.py`.
- **Do NOT bump the ODR format version.** `ODR_VERSION` stays `"0.1"` in both `aragora/gauntlet/odr_export.py` and `aragora-verify/src/aragora_verify/schema.py`. The v1.0 *contract* is authored as docs; the actual version flip + coordinated PyPI re-release is a parked Tier 3 release step.
- **Single-register rule.** Extend `docs/status/ROADMAP_INTAKE_REGISTER.md`; never create a parallel register file.
- **`aragora-verify` stays zero-dependency on `aragora`.** Its tests may not import `aragora.*`. The committed example receipt JSON is the only contract crossing the boundary.
- **DRY/YAGNI/TDD, frequent commits.** Reuse the `DecisionReceipt` construction already in `tests/gauntlet/test_odr_export.py`; do not hand-rebuild it.
- **Receipts in this plan are unsigned.** `decision_receipt_to_odr` emits `signatures: []`; `verify()` reports an absent-signature *warning*, not a failure. That is acceptable for M1.
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Current state (verified — re-confirm at execution time)

- **Worktree:** `$(git rev-parse --show-toplevel)` — `.claude/worktrees/strategy-mission-cadence`
- **Branch:** `worktree-strategy-mission-cadence`
- **HEAD:** `7a16a827a8`
- **Divergence:** `origin/main...HEAD` = **2 behind, 3 ahead**; merge-base `2489ea3f52`; `origin/main` tip `0e1bb9ba42` (local `origin/main` has advanced since the branch was cut).
- **Status:** clean working tree.
- **Ahead of merge-base (3 commits):**
  - `7a16a827a8 docs(plan): M0+M1 implementation plan (ODR v1.0 docs+verification)`
  - `50a6a9650b docs(spec): scope M1 to ODR v1.0 docs+verification only (no README/Tier3-4)`
  - `397b3107ff docs(spec): strategy as a bounded-mission cadence design`
- **Spec this plan implements:** [`docs/superpowers/specs/2026-06-26-strategy-as-bounded-mission-cadence-design.md`](./strategy-as-bounded-mission-cadence-design)

> These SHAs are a snapshot. **Step 0 of execution (below) re-confirms live state and reconciles the 2-commit divergence before any task runs.**

### Task 0: Refresh / reconcile before execution

- [ ] **Step 1:** `git fetch origin` and re-read divergence: `git rev-list --left-right --count origin/main...HEAD`.
- [ ] **Step 2:** Reconcile the branch onto current `origin/main` (rebase preferred for this docs-only branch):

```bash
git fetch origin
git rebase origin/main      # branch is docs-only so far; conflicts unlikely
# if a conflict appears in a forbidden surface, STOP and surface it (see Stop conditions)
```

- [ ] **Step 3:** Re-confirm the touch-targets still exist post-rebase:

```bash
test -f tests/gauntlet/test_odr_export.py && grep -n "_full_receipt" tests/gauntlet/test_odr_export.py
test -f aragora/gauntlet/odr_export.py && test -f docs/specs/OPEN_DECISION_RECEIPT.md
test -d aragora-verify/tests
```

Expected: `_full_receipt` present; all paths exist. If any moved, pause and re-locate before proceeding.

## Exact M0 / M1 scope split

| Phase | Tier | In scope (autonomous) | Out of scope (parked) |
|---|---|---|---|
| **M0** | 0 | Add `## Strategy Mission Queue` section to the Intake Register with rows M1/M2/M3 and statuses; a structural parser test enforcing the at-most-one-active invariant. | Opening the tracking epic (Task M0.2 — outward-facing GitHub write, **parked for operator**); the `MissionMetronome` service + `GateEvaluation` extension (separate plan — see "Subsequent plans"). |
| **M1** | 0-1 | ODR **Versioning & Stability** contract (spec doc); **native↔ODR mapping** doc + a drift-guard test; a committed **example receipt** generated by `odr_export` + a main-repo conformance/digest test; an `aragora-verify` test that independently verifies the example. | Format-version bump to 1.0; signing-in-gate (#8225); server verify endpoint (#8226); any PyPI re-release. |

## Files expected to change

**M0**
- Modify: `docs/status/ROADMAP_INTAKE_REGISTER.md` (append one section)
- Create: `tests/docs/test_mission_queue_register.py`

**M1**
- Modify: `docs/specs/OPEN_DECISION_RECEIPT.md` (add Versioning & Stability section)
- Create: `docs/specs/odr-native-mapping.md`
- Create: `docs/specs/examples/example-decision-receipt.odr.json` (committed fixture; the emitter↔verifier contract)
- Create: `tests/gauntlet/test_odr_native_mapping.py` (mapping drift-guard)
- Create: `tests/gauntlet/test_odr_example_receipt.py` (main-repo conformance + regeneration guard)
- Create: `aragora-verify/tests/test_example_live_receipt.py` (independent verification)

---

## Phase M0 — Mission Queue in the Intake Register

### Task M0.1: Add the Strategy Mission Queue section

**Files:**
- Modify: `docs/status/ROADMAP_INTAKE_REGISTER.md`
- Test: `tests/docs/test_mission_queue_register.py`

**Interfaces:**
- Produces: a Markdown section `## Strategy Mission Queue` containing a table with header columns exactly `id | title | tier | status | external-proof gate | tracking`, and a status vocabulary `queued | active | blocked-on-proof | blocked-on-human | done`.

- [ ] **Step 1: Write the failing test**

```python
# tests/docs/test_mission_queue_register.py
"""The Intake Register carries a single Strategy Mission Queue with a valid,
single-active invariant. This is the durable cross-session state for the
bounded-mission cadence (see docs/superpowers/specs/2026-06-26-strategy-as-
bounded-mission-cadence-design.md)."""
from __future__ import annotations

import re
from pathlib import Path

REGISTER = Path("docs/status/ROADMAP_INTAKE_REGISTER.md")
REQUIRED_COLUMNS = ["id", "title", "tier", "status", "external-proof gate", "tracking"]
VALID_STATUS = {"queued", "active", "blocked-on-proof", "blocked-on-human", "done"}


def _section(text: str, header: str) -> str:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _rows(section: str) -> list[list[str]]:
    rows = []
    for ln in section.splitlines():
        if ln.startswith("|") and "---" not in ln:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(cells)
    return rows


def test_mission_queue_section_exists_and_is_valid():
    text = REGISTER.read_text(encoding="utf-8")
    section = _section(text, "## Strategy Mission Queue")
    rows = _rows(section)
    assert rows, "Strategy Mission Queue table is empty"
    header = [c.lower() for c in rows[0]]
    assert header == REQUIRED_COLUMNS, f"unexpected columns: {header}"
    data = rows[1:]
    assert data, "no mission rows"
    statuses = [r[3] for r in data]
    for s in statuses:
        assert s in VALID_STATUS, f"invalid status {s!r}"
    assert statuses.count("active") <= 1, "at most one mission may be active"
    ids = [r[0] for r in data]
    assert {"M1", "M2", "M3"}.issubset(set(ids)), f"missing mission rows: {ids}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_mission_queue_register.py -v`
Expected: FAIL — `StopIteration` / section not found (the section does not exist yet).

- [ ] **Step 3: Add the section to the register**

Append to `docs/status/ROADMAP_INTAKE_REGISTER.md` (use these exact rows):

```markdown
## Strategy Mission Queue

Bounded, exit-gated sub-missions for the "useful + unique" strategy mission.
Advanced one at a time on a standing cadence; row N+1 stays `queued` until row
N's external-proof gate verifies. See the design spec for the cadence mechanism.

| id | title | tier | status | external-proof gate | tracking |
|---|---|---|---|---|---|
| M1 | ODR v1.0 GA (docs + verification only) | 0-1 | queued | `aragora-verify` verifies a committed example receipt produced by `odr_export` against the published ODR profile | epic TBD-link |
| M2 | Action wedge (quorum review → verifiable receipt artifact + PR comment) | 2-3 | queued | the Action runs green on a real PR here and uploads a receipt `aragora-verify` passes (workflow change parks for founder) | epic TBD-link |
| M3 | Proof corpus + legibility (README narrative + sprawl quarantine) | mixed | queued | public artifact with the receipt corpus live, plus a one-sentence README + ≤5 documented core modules on main (narrative parks for founder) | epic TBD-link |
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_mission_queue_register.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/status/ROADMAP_INTAKE_REGISTER.md tests/docs/test_mission_queue_register.py
git commit -m "docs(register): add Strategy Mission Queue section (M0)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task M0.2: Open the tracking epic (operator step)

- [ ] **Step 1:** Open a GitHub epic "Strategy as a bounded-mission cadence (M0-M3)" linking the spec and this plan.

```bash
gh issue create --title "Epic: Strategy as a bounded-mission cadence (M0-M3)" \
  --body "Durable home: docs/status/ROADMAP_INTAKE_REGISTER.md#strategy-mission-queue
Spec: docs/superpowers/specs/2026-06-26-strategy-as-bounded-mission-cadence-design.md
Plan: docs/superpowers/plans/2026-06-26-mission-cadence-m0-m1.md
M1 docs+verification only; M2/M3 park for founder."
```

- [ ] **Step 2:** Replace the three `epic TBD-link` cells in the register with the epic URL; re-run `pytest tests/docs/test_mission_queue_register.py -v` (PASS); commit.

> Note: epic creation writes to GitHub (outward-facing). If running autonomously, **park this task for the operator** and proceed to M1 — M1 does not depend on the epic existing.

---

## Phase M1 — ODR v1.0 contract + independent verification

### Task M1.1: Author the Versioning & Stability contract

**Files:**
- Modify: `docs/specs/OPEN_DECISION_RECEIPT.md`

**Interfaces:**
- Produces: a section `## Versioning and Stability` in the ODR spec defining: (a) field-stability tiers (stable / provisional / reserved), (b) backward/forward-compat guarantees for a v1.0 profile, (c) deprecation policy, (d) the native `DecisionReceipt.schema_version` ↔ `odr_version` relationship, (e) an explicit statement that the on-wire `odr_version` remains `"0.1"` until a coordinated GA release bumps emitter + bundled schema + published verifier together.

- [ ] **Step 1: Add the section.** Append `## Versioning and Stability` to `docs/specs/OPEN_DECISION_RECEIPT.md` covering (a)-(e) above. State that `aragora-verify` is the conformance authority and that breaking changes require a major `odr_version` bump + a new published verifier.

- [ ] **Step 2: Validate presence**

Run: `rg -n "^## Versioning and Stability$" docs/specs/OPEN_DECISION_RECEIPT.md`
Expected: one match.

- [ ] **Step 3: Commit**

```bash
git add docs/specs/OPEN_DECISION_RECEIPT.md
git commit -m "docs(odr): add Versioning and Stability contract (M1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task M1.2: native↔ODR mapping doc + drift-guard test

**Files:**
- Create: `docs/specs/odr-native-mapping.md`
- Test: `tests/gauntlet/test_odr_native_mapping.py`

**Interfaces:**
- Consumes: `aragora.gauntlet.odr_export.decision_receipt_to_odr` (returns a dict whose top-level keys are the ODR fields).
- Produces: `docs/specs/odr-native-mapping.md` documenting every ODR top-level field and its native `DecisionReceipt` source (or "absent" rationale).

- [ ] **Step 1: Write the failing test**

```python
# tests/gauntlet/test_odr_native_mapping.py
"""The native<->ODR mapping doc must document every ODR top-level field the
emitter produces, so the mapping cannot silently drift from odr_export."""
from __future__ import annotations

from pathlib import Path

from tests.gauntlet.test_odr_export import _full_receipt  # existing factory (verified)
from aragora.gauntlet.odr_export import decision_receipt_to_odr

MAPPING_DOC = Path("docs/specs/odr-native-mapping.md")


def test_mapping_doc_covers_every_odr_field():
    odr = decision_receipt_to_odr(_full_receipt())
    doc = MAPPING_DOC.read_text(encoding="utf-8")
    missing = [k for k in odr.keys() if f"`{k}`" not in doc]
    assert not missing, f"mapping doc missing ODR fields: {missing}"
```

> `_full_receipt()` is the existing factory in `tests/gauntlet/test_odr_export.py`
> (verified at lines 139-180; a sibling `_minimal_receipt()` also exists). Use
> `_full_receipt` so the mapping is checked against the richest emitter output.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gauntlet/test_odr_native_mapping.py -v`
Expected: FAIL — `FileNotFoundError` (doc missing) or assertion listing all fields.

- [ ] **Step 3: Write the mapping doc.** Create `docs/specs/odr-native-mapping.md` with a table mapping each ODR top-level field (`odr_version`, `profile`, `receipt_id`, `issued_at`, `subject`, `claim`, `reasoning`, `quorum`, `confidence`, `cruxes`, `attestation`, `routing`, `signatures`, `source`) to its `DecisionReceipt` origin, citing the `_map_*` function in `aragora/gauntlet/odr_export.py`. Wrap each field name in backticks.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gauntlet/test_odr_native_mapping.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/specs/odr-native-mapping.md tests/gauntlet/test_odr_native_mapping.py
git commit -m "docs(odr): native<->ODR field mapping with drift-guard test (M1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task M1.3: Committed example receipt + conformance/regeneration guard

**Files:**
- Create: `docs/specs/examples/example-decision-receipt.odr.json`
- Test: `tests/gauntlet/test_odr_example_receipt.py`

**Interfaces:**
- Consumes: `decision_receipt_to_odr`, `load_odr_schema`, `odr_content_digest` from `aragora.gauntlet.odr_export`; `_full_receipt` from `tests/gauntlet/test_odr_export.py`.
- Produces: a committed example ODR JSON used by `aragora-verify` (Task M1.4) — the emitter↔verifier contract artifact.

- [ ] **Step 1: Write the failing test**

```python
# tests/gauntlet/test_odr_example_receipt.py
"""The committed example receipt is the emitter<->verifier contract. It must
be (a) exactly what odr_export emits today (regeneration guard) and (b)
schema-conformant with a recomputable JCS digest."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from tests.gauntlet.test_odr_export import _full_receipt
from aragora.gauntlet.odr_export import (
    decision_receipt_to_odr,
    load_odr_schema,
    odr_content_digest,
)

EXAMPLE = Path("docs/specs/examples/example-decision-receipt.odr.json")


def test_example_matches_current_emitter_output():
    expected = decision_receipt_to_odr(_full_receipt())
    actual = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert actual == expected, "example receipt is stale; regenerate it"


def test_example_is_schema_conformant_and_digestible():
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    jsonschema.validate(doc, load_odr_schema())
    digest = odr_content_digest(doc)
    assert len(digest) == 64  # sha-256 hex
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gauntlet/test_odr_example_receipt.py -v`
Expected: FAIL — `FileNotFoundError` (example not written yet).

- [ ] **Step 3: Generate and commit the example**

```bash
mkdir -p docs/specs/examples
python3 -c "
import json
from tests.gauntlet.test_odr_export import _full_receipt
from aragora.gauntlet.odr_export import decision_receipt_to_odr
odr = decision_receipt_to_odr(_full_receipt())
open('docs/specs/examples/example-decision-receipt.odr.json','w').write(
    json.dumps(odr, indent=2, sort_keys=True) + '\n')
print('wrote example')
"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gauntlet/test_odr_example_receipt.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add docs/specs/examples/example-decision-receipt.odr.json tests/gauntlet/test_odr_example_receipt.py
git commit -m "test(odr): committed example receipt + regeneration guard (M1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task M1.4: Independent verification by `aragora-verify`

**Files:**
- Create: `aragora-verify/tests/test_example_live_receipt.py`

**Interfaces:**
- Consumes: `aragora_verify.verify`; the committed example at `docs/specs/examples/example-decision-receipt.odr.json` (read via repo-root-relative path; NO import of `aragora.*`).
- Produces: the M1 external-proof: the standalone verifier confirms the emitter's receipt with zero Aragora dependency.

- [ ] **Step 1: Write the failing test**

```python
# aragora-verify/tests/test_example_live_receipt.py
"""Independent verification of an emitter-produced ODR receipt. This is M1's
external proof: aragora-verify (zero Aragora dependency) confirms a receipt
that aragora/gauntlet/odr_export.py produced. The example JSON is the only
artifact crossing the package boundary."""
from __future__ import annotations

import json
from pathlib import Path

from aragora_verify import verify
from aragora_verify.verifier import FAIL, PASS  # status constants (verified)

# repo root = three parents up from aragora-verify/tests/<file>
EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "docs" / "specs" / "examples" / "example-decision-receipt.odr.json"
)


def test_emitter_receipt_verifies_independently():
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    result = verify(doc)  # unsigned: the "signature" check is WARN, not FAIL
    failed = [c for c in result.checks if c.status == FAIL]
    assert result.ok, failed
    assert not failed
    statuses = {c.name: c.status for c in result.checks}
    # schema + digest must affirmatively pass; quorum is PASS or SKIP, never FAIL.
    assert statuses["schema_conformance"] == PASS
    assert statuses["canonical_digest"] == PASS
    assert statuses.get("quorum_consistency") != FAIL
```

> Verified against `aragora-verify/src/aragora_verify/verifier.py`: `Check(name,
> status, detail)`; constants `PASS/FAIL/WARN/SKIP`; check names `signature`
> (WARN when unsigned), `schema_conformance`, `canonical_digest`,
> `quorum_consistency` (SKIP when no present quorum block), `chain_link` (SKIP
> without `--chain`).

- [ ] **Step 2: Run test to verify it fails (or errors on missing example)**

Run: `cd aragora-verify && python -m pytest tests/test_example_live_receipt.py -v`
Expected: initially FAIL if run before M1.3; PASS once the example exists.

- [ ] **Step 3: Run to verify it passes**

Run: `cd aragora-verify && python -m pytest tests/test_example_live_receipt.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add aragora-verify/tests/test_example_live_receipt.py
git commit -m "test(aragora-verify): independently verify an emitter receipt (M1 proof)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task M1.5: Flip the M1 register row to done

- [ ] **Step 1:** In `docs/status/ROADMAP_INTAKE_REGISTER.md`, set the M1 row `status` to `done`. Run `pytest tests/docs/test_mission_queue_register.py -v` (PASS — at-most-one-active still holds). Commit.

---

## Validation commands (full M0+M1 gate)

```bash
# scoped test slice
pytest tests/docs/test_mission_queue_register.py \
       tests/gauntlet/test_odr_native_mapping.py \
       tests/gauntlet/test_odr_example_receipt.py -v
cd aragora-verify && python -m pytest tests/ -v && cd ..
# local truth (elves gate step 3)
pre-commit run --all-files
mypy aragora/gauntlet/odr_export.py
# spec presence checks
rg -n "^## Versioning and Stability$" docs/specs/OPEN_DECISION_RECEIPT.md
rg -n "^## Strategy Mission Queue$" docs/status/ROADMAP_INTAKE_REGISTER.md
```

Expected: all green; no new mypy errors vs baseline; test count does not decrease.

## Stop conditions (halt and surface; do not push through)

- Any task would require editing a Global-Constraints forbidden surface (README, `action.yml`, `.github/workflows/`, server verify endpoint, signing-in-gate, `aragora_verify` version/release, `aragora/__init__.py`, `CLAUDE.md`, `.env`).
- A rebase conflict (Task 0) lands in a forbidden surface → STOP and surface it.
- The `_full_receipt` factory has moved or been renamed in `tests/gauntlet/test_odr_export.py` post-rebase → pause and re-locate (do not invent a parallel factory).
- `jsonschema` is unavailable in the env → pause (do not silently swap validators).
- The example receipt cannot be made schema-conformant unsigned (verify() FAILs on schema/digest/quorum, not just the signature warning) → pause; this signals an emitter/schema mismatch worth a real diagnosis.
- Any required check on `origin/main` goes red → main-red incident mode: stop roadmap work, fix first.
- A single diff exceeds ~800 LOC or two consecutive tasks fail CI for distinct reasons → stop the wave, ask.

## Parked for founder / operator approval

- **M0.2 epic creation** — writes to GitHub (outward-facing); operator runs it (M1 does not block on it).
- **ODR format-version bump to 1.0** + coordinated PyPI re-release of `aragora-verify` — Tier 3 release.
- **Auto-signing-in-gate (#8225)** and **server verify endpoint (#8226)** — Tier 3, hard-stop.
- **M2** (the Action / `.github/workflows/` change) — approval-required surface.
- **M3** (README single-narrative decision; sprawl quarantine) — Tier 3 founder call.
- **The `MissionMetronome` service + `GateEvaluation` extension** — the cadence automation; a separate plan (below). Until it exists, the queue is advanced by hand or via the `elves-aragora` skill.

## First autonomous slice (recommended after plan approval)

**Phase M0, Task M0.1** — add the `## Strategy Mission Queue` section + its parser test. Rationale: pure docs+test, Tier 0, zero forbidden surfaces, and it materializes the durable state every later step reads. Then proceed straight into **M1.1 → M1.4** (all Tier 0-1, no parked surfaces) in one bounded run, ending at the M1 external-proof gate (Task M1.4 green). Park M0.2 (epic) and Task M1.5's "done" flip for the operator if running unattended.

## Subsequent plans (not in this plan — scope isolation)

1. **Mission cadence automation** — `MissionMetronome` service + `GateEvaluation` mission-proof gate + launchd unit (default-OFF). Independent subsystem; its own spec section already exists. Write after M1 proves the gate pattern by hand.
2. **M2 — Action wedge** — `CollectOutcome → DecisionReceipt` bridge (Tier 1-2, autonomous) then the `action.yml` rewrite (parks for founder).
3. **M3 — Proof corpus + legibility** — gated on the founder's single-narrative decision.

---

## Self-review

- **Spec coverage:** M0 (register queue) ✓ Task M0.1; M1 versioning contract ✓ M1.1; native↔ODR mapping ✓ M1.2; example-receipt verification ✓ M1.3/M1.4; "no Tier 3-4 surfaces" ✓ Global Constraints + Stop conditions; cadence/metronome + M2 + M3 explicitly deferred to subsequent plans ✓.
- **Placeholder scan:** the only `TBD` is `epic TBD-link`, intentionally replaced in Task M0.2; no "add error handling"/"write tests for the above" placeholders; all code steps carry real code.
- **Type/name consistency:** `_full_receipt` (verified at `tests/gauntlet/test_odr_export.py:139`) reused across M1.2/M1.3; `decision_receipt_to_odr`, `load_odr_schema`, `odr_content_digest`, `verify`, and `Check.name`/`Check.status` with constants `PASS`/`FAIL` all match the verified source; example path identical in M1.3 (writer) and M1.4 (reader).
- **Both prior soft spots resolved:** the factory is confirmed `_full_receipt`; the verifier status constants are confirmed `PASS`/`FAIL` and now imported rather than string-compared. Remaining external dependency: `jsonschema` availability (gated by a stop condition).
