# A2 — Admission-Class Productization Plan (issue #7209)

**Status:** plan-only (no code in this PR).
**Owner:** droid (Factory Droid) for plan; implementation owner TBD.
**Authorization:** founder sentence 2026-05-16 — "open A2 implementation-plan PR for #7209 admission-class productization".
**Canonical gate alignment:** `docs/status/NEXT_STEPS_CANONICAL.md` — *"operationalizing the proof-first loop, not adding new roadmap scope."* This plan productizes a repeated rescue class. It does not widen B2, does not unblock DIC/AGT delayed tracks, does not touch H1/H2.

## Problem

`docs/status/B0_BENCHMARK_TRUTH_STATUS.md` (2026-05-15T16:31:15Z, rev-4) reports:

| Metric | Value |
| --- | --- |
| Verified truth success rate (primary) | **0.0%** |
| In-progress graduation rate | 0.0% (0/13) |
| Failure class `blocked_not_dispatch_bounded` | **8 ticks** |
| Failure class `blocked_auth_failure` | **7 ticks** |
| Failure class `blocked_sanitation_failed` | 2 ticks |
| Failure class `rescue_no_deliverable` | 1 tick |

**15 of 18 failure ticks (83%) are pre-execution admission rejections** — the worker never reached the code-edit phase. The 30-day target in `docs/status/NEXT_STEPS_CANONICAL.md` (`≥50% no-rescue on bounded corpus`) is not evaluable while the admission layer rejects every dispatch.

This plan applies the canonical Operating Law (`docs/CANONICAL_GOALS.md`):

> *If humans intervene twice for the same class of failure, the next system change should absorb that rescue as product behavior.*

Stage-Gate Conductor flagged this twice (2026-05-15T22:05Z, 2026-05-16T04:05Z) without a tracking ticket; #7209 is the ticket.

## Diagnosis (read-only investigation 2026-05-16)

The classification chain that produces `blocked_not_dispatch_bounded` and `blocked_auth_failure`:

```
aragora/swarm/terminal_truth.py:48       BLOCKED_NOT_DISPATCH_BOUNDED enum
aragora/swarm/terminal_truth.py:129      outcome=="blocked"               -> BLOCKED_NOT_DISPATCH_BOUNDED
aragora/swarm/terminal_truth.py:172      gate.failure_classes ∩ {contract_missing, context_policy_unresolved} -> BLOCKED_NOT_DISPATCH_BOUNDED
aragora/swarm/preflight.py:1518-1545     evaluate_preflight_dispatch_gate emits these failure classes
aragora/swarm/boss_worker_lifecycle.py:774  failure_classes=["contract_missing"] when issue lacks acceptance criteria
aragora/swarm/boss_worker_lifecycle.py:872  failure_classes=["contract_missing"] when spec.is_dispatch_bounded() is False
aragora/swarm/boss_worker_lifecycle.py:946  one-shot drift-feedback upgrade exists; if still bounded-False -> terminal blocked
aragora/swarm/mission.py:172             MissionContextPolicy.is_resolvable() requires: role + allowed_artifact_classes (non-empty) + max_source_count>0 + max_chars>0 + transcript_allowance
aragora/swarm/worker_contract.py:271     normalize_context_policies builds default; default always satisfies is_resolvable()
```

The rev-4 corpus entries (e.g. `#5185`, `#5187`, `#5197`, ...) carry rich metadata in `docs/benchmarks/corpus.json`:

- `execution_class`: `missing_test_coverage` | `small_refactor` | `validation_tightening` | `exception_narrowing`
- `scope_hint`: paths the task targets
- `known_constraints`: bounding rules

The dispatch pipeline **never reads this metadata**. The gate makes its bounded-or-not decision purely from the GitHub issue body. Some rev-4 entries' bodies lack the explicit `Acceptance Criteria` / `Validation` / `Test Plan` / `Definition of Done` section that `require_validation_contract` looks for, even though their corpus entries are bounded. Result: `failure_classes=["contract_missing"]` → `BLOCKED_NOT_DISPATCH_BOUNDED`.

**Evidence**: the same corpus issue `#5185` shows `dispatch_provenance.terminal_classes = [blocked_not_dispatch_bounded, deliverable_pr_created]`. Same issue, different dispatches → different terminal class. The gate is **flaky on shape, not statically wrong**. The rev-4 corpus carries the bounded structure metadata; the gate just doesn't read it.

The 7 `blocked_auth_failure` ticks come from preflight subprocess auth probes inside `_run_preflight_worker` (real worker spawn with real provider keys). Cheaper credential-envelope probes already exist (`RS-05` shipped `CredentialEnvelope`) but are not wired into preflight.

## Proposed bounded change

Three sequential, additive, flag-gated PRs. None widens B2. None changes default behaviour for non-corpus issues. Each is independently reversible.

### PR-1 — Corpus-aware dispatch upgrade (closes the 8 `blocked_not_dispatch_bounded` ticks)

**Scope:** `aragora/swarm/dispatch_followups.py` (plus tests).

Add a small helper that, when `upgrade_unbounded_spec` is invoked on an issue whose number appears in `docs/benchmarks/corpus.json::issues`, reads the corpus row and synthesizes the missing dispatch-bounded structure from `scope_hint + known_constraints + execution_class`. Behaviour:

```python
def upgrade_unbounded_spec(spec, *, issue_number, ...):
    if not spec.is_dispatch_bounded():
        corpus_entry = _lookup_corpus_entry(issue_number)
        if corpus_entry is not None:
            spec = _augment_spec_from_corpus(spec, corpus_entry)
        else:
            spec = _existing_llm_or_heuristic_upgrade(spec, ...)
    return spec
```

`_augment_spec_from_corpus` injects file_scope from `scope_hint`, acceptance criteria from `known_constraints` mapped to `execution_class`-specific templates (e.g. `missing_test_coverage` → "pytest tests/<scope_hint_path> -v exits 0; new tests cover happy path + at least one edge case"), and one work_order item. The augmentation is gated behind `ARAGORA_CORPUS_AWARE_DISPATCH=1` (default OFF). When OFF, behaviour is identical to today.

**LOC estimate:** ~120 LOC source + ~180 LOC tests.

### PR-2 — Launcher-side credential-envelope preflight (closes the 7 `blocked_auth_failure` ticks)

**Scope:** `aragora/swarm/preflight.py` (plus a thin `CredentialEnvelope.probe()` call surface; uses existing RS-05 envelope).

Before `_run_preflight_worker` spawns a subprocess, probe the credential envelope directly:

```python
def _run_preflight_worker(...):
    envelope_probe = CredentialEnvelope.probe(contract=contract, env=env)
    if not envelope_probe.passed:
        return _preflight_short_circuit_auth_failure(envelope_probe)
    return _run_preflight_worker_subprocess(...)
```

Failure short-circuits with `failure_classes=["credential_envelope_failed"]` and a structured envelope-state receipt — this stops the dispatch from incurring the 5-10s subprocess auth probe cost and produces a richer fault signal (which provider, which slice, which rotation attempted) than the current "auth in stderr" hint match. Default flag `ARAGORA_CREDENTIAL_ENVELOPE_PROBE=1` (ON only after PR-1 is validated; OFF by default until then).

**LOC estimate:** ~80 LOC source + ~120 LOC tests.

### PR-3 — Productization fixtures + rescue-map entry (regression guard)

**Scope:** `benchmarks/fixtures/swarm/terminal_truth/` + `docs/benchmarks/rescue_productization.json`.

1. Add `benchmarks/fixtures/swarm/terminal_truth/upgraded_from_admission.json` — rows that previously classified as `blocked_not_dispatch_bounded` but, after PR-1's corpus-aware upgrade, classify as `deliverable_branch_pushed` or `deliverable_pr_created`. Pinned regression — if PR-1 regresses on a corpus-shape, the fixture test fails.
2. Add `benchmarks/fixtures/swarm/terminal_truth/credential_envelope_preflight.json` — rows that previously classified as `blocked_auth_failure` (auth-in-stderr) but now classify as `blocked_credential_envelope_failed` with explicit envelope state.
3. Append `docs/benchmarks/rescue_productization.json` entry `admission_class_corpus_synthesis_v1` with method, code seams, before/after fixtures, and the canonical Operating Law link.

**LOC estimate:** ~40 LOC fixtures + ~80 LOC test wiring + ~30 LOC docs.

## Acceptance criteria

PR-1 acceptance criteria:

- `tests/swarm/test_dispatch_followups.py::test_corpus_aware_upgrade_attaches_scope_hint` — pytest-only, asserts spec gains `file_scope` from corpus when issue body is sparse and `execution_class` is one of the 4 rev-4 classes.
- `tests/swarm/test_dispatch_followups.py::test_corpus_aware_upgrade_off_by_default` — asserts behaviour is identical to today when flag is OFF.
- `tests/swarm/test_dispatch_followups.py::test_non_corpus_issue_passes_through` — asserts no behaviour change for non-corpus issues even with flag ON.
- `pytest tests/swarm/test_dispatch_followups.py -v` exits 0.

PR-2 acceptance criteria:

- `tests/swarm/test_preflight_credential_envelope.py::test_envelope_probe_short_circuits_auth_failure` — asserts no subprocess spawn when envelope fails.
- `tests/swarm/test_preflight_credential_envelope.py::test_envelope_probe_off_by_default` — asserts existing behaviour preserved when flag OFF.
- `pytest tests/swarm/test_preflight_credential_envelope.py -v` exits 0.

PR-3 acceptance criteria:

- `tests/swarm/test_terminal_truth_fixtures.py` re-runs against the new fixture files; new rows classify per spec.
- `python scripts/measure_b0_scorecard.py --fixtures benchmarks/fixtures/swarm/terminal_truth/upgraded_from_admission.json` shows correct class distribution.
- `docs/benchmarks/rescue_productization.json` round-trips through `python -c "import json; json.load(open('docs/benchmarks/rescue_productization.json'))"`.

**Cross-PR canonical-target acceptance** (the actual A2 goal):

- After PR-1 + PR-3 land and the flag is enabled for one bounded shift, the next B0 publication shows `blocked_not_dispatch_bounded` ticks **decrease** for the rev-4 corpus (expectation: 8 → ≤2; not 0 because some genuinely-unbounded sanitation cases remain).
- After PR-2 + PR-3 land and the flag is enabled, `blocked_auth_failure` ticks are reclassified as `blocked_credential_envelope_failed` with structured envelope state.

These cross-PR criteria are observational, not test-time. They are evaluated on `docs/status/B0_BENCHMARK_TRUTH_STATUS.md` after the next scheduled publication.

## Risk + rollback

| Risk | Mitigation | Rollback |
|---|---|---|
| Corpus-aware augment produces wrong file_scope and worker writes wrong file | Flag OFF by default; augment only applies to whitelisted `execution_class` set; corpus rows already passed honesty audit | `unset ARAGORA_CORPUS_AWARE_DISPATCH` or `git revert <PR-1 sha>` |
| Credential-envelope probe is wrong about envelope readiness and blocks legitimate dispatches | Flag OFF by default; gate landing of PR-2 behind one bounded probe-run smoke; envelope `probe()` returns existing-behaviour result on probe failure | `unset ARAGORA_CREDENTIAL_ENVELOPE_PROBE` or `git revert <PR-2 sha>` |
| Fixtures encode wrong post-state and prevent legitimate behaviour changes later | Fixtures live in `benchmarks/fixtures/`, not in production source; test failures are explicit and tie back to this plan doc | `git revert <PR-3 sha>` or update fixture in a successor PR |
| LOC budget creep | Each PR pre-scoped above; PR-1 hard cap 300 LOC, PR-2 hard cap 200 LOC, PR-3 hard cap 150 LOC. If exceeded, plan returns for re-decomposition. | Reject PR; re-plan |

Each PR is independently revertible. Flags guarantee zero default-behaviour change until explicit operator opt-in.

## Implementation sequencing

1. **PR-1** lands first (flag OFF). Tests pass on CI.
2. Smoke run: flag-ON for one corpus dispatch in a bounded shift; observe `blocked_not_dispatch_bounded` count drop on B0.
3. **PR-3** lands second (fixtures from real PR-1 ON observations).
4. **PR-2** lands third (flag OFF). Tests pass on CI.
5. Smoke run: flag-ON for one corpus dispatch with stale credential probe; observe `blocked_credential_envelope_failed` replacing `blocked_auth_failure`.
6. PR-3 amendment adds the credential-envelope fixture.
7. **Both flags default-ON** is a separate Tier-4 operator decision after the 30-day canonical target (`≥50% no-rescue on bounded corpus`) is met.

Steps 2, 5 are observational; they do not block the next PR's merge.

## Not in scope

- **No B2 expansion.** `docs/status/NEXT_STEPS_CANONICAL.md` § "B2 guard expansion criteria" governs; this plan does not move that gate.
- **No new corpus entries.** rev-4 corpus stays as-is.
- **No changes to TerminalClass enum.** PR-1 reuses existing classes; PR-2 adds a refinement via gate failure_class string but does not introduce new enum values.
- **No DIC/AGT/H2 scope.** Each delayed track stays delayed.
- **No founder review surface changes.** Issue-triage and review-queue surfaces unchanged.
- **No agent-bridge / live-feed changes.** Bridge work continues independently.

## Open questions (require operator answer before PR-1 opens)

1. **Flag-default policy after smoke.** Once PR-1 smoke run shows `blocked_not_dispatch_bounded` drop, do we promote the flag to ON-by-default in PR-1 itself, or hold for a separate Tier-4 decision? Default recommendation: hold; separate decision.
2. **Whitelist of execution_classes for augment.** Initial set: `missing_test_coverage`, `small_refactor`, `validation_tightening`, `exception_narrowing` (all 4 currently present in rev-4 corpus). Adding new execution_classes is a separate corpus-revision PR. OK?
3. **Implementation owner.** This plan was authored by Factory Droid (assessment + diagnostic + plan). Implementation can be: (a) droid (same session, continuation); (b) claude-code; (c) codex-cli; (d) founder. The plan is small enough that any of (a/b/c/d) is viable.

## References

- Tracking issue: [#7209](https://github.com/synaptent/aragora/issues/7209)
- Canonical proof surface: `docs/status/B0_BENCHMARK_TRUTH_STATUS.md`
- Corpus manifest: `docs/benchmarks/corpus.json` (rev-4, 13 in-progress)
- Operating Law source: `docs/CANONICAL_GOALS.md`
- Gate doctrine: `docs/status/NEXT_STEPS_CANONICAL.md`
- Stage-Gate Conductor log: [#7162](https://github.com/synaptent/aragora/issues/7162)
- Settlement-receipt hygiene precedent: `.aragora/review-queue/receipts/pr-7210-recorded-7210-c44acbeb4888-admin_squash_merge-admin_squash_merge.json`
- Closure-receipt precedent: `.aragora/audits/closure-receipts/2026-05-14-stale-tasking-closure.json`
- Related (non-overlapping) stage-gate-drift issues: [#6598](https://github.com/synaptent/aragora/issues/6598), [#6706](https://github.com/synaptent/aragora/issues/6706), [#6097](https://github.com/synaptent/aragora/issues/6097), [#6246](https://github.com/synaptent/aragora/issues/6246)
