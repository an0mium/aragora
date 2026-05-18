# P20 — model_pins frontier-aligned receipt

- Session: `droid-F473CDBF`
- Lane: `P20-model-pins-frontier-aligned`
- Branch: `droid/P20-model-pins-frontier-aligned-20260518-041438`
- PR: [#7306](https://github.com/synaptent/aragora/pull/7306) (open, ready, MERGEABLE, 49 SUCCESS / 69 SKIPPED / 1 PENDING at wait-window close, 0 FAILURE / 0 CANCELLED)
- Started: 2026-05-18T04:14:38Z
- Completed: 2026-05-18T04:33:00Z (approximate)
- Outcome: **shipped**

## Acceptance against P20 spec

| Spec | Implementation |
|---|---|
| Add `OPUS_4_7` module-level constant | Line 64 of `aragora/config/model_pins.py` |
| Add `GPT_5_4` module-level constant | Line 65 |
| Add `GEMINI_3_1_PRO` module-level constant | Line 66 |
| Each alias re-exports the canonical `*_DIRECT` value | Verified by `TestAliasesMatchFrontier` (3 tests) |
| All three appear in `__all__` | Verified by `TestAliasesInAll` |
| `check_canonical_metrics.py` regex matches | Verified by `TestCanonicalMetricsRegex` (3 tests) |
| `security.model_pins.frontier_aligned` passes | **Verified via live run**: before=fail, after=pass |

## Canonical-metrics receipt diff

**Before** (`docs/status/generated/canonical_metrics/latest.json` snapshot at 04:14:38Z):

```json
{"summary": {"fail": 1, "pass": 8, "warn": 1}, "results": [
  {"status": "fail", "claim_id": "security.model_pins.frontier_aligned",
   "observed": "missing: OPUS_4_7, GPT_5_4, GEMINI_3_1_PRO", ...},
  ...
]}
```

**After** (live re-run from worktree post-edit):

```json
{"summary": {"fail": 0, "pass": 9, "warn": 1}, "results": [
  {"status": "pass", "claim_id": "security.model_pins.frontier_aligned", ...},
  ...
]}
```

`security.model_pins.frontier_aligned` moves from fail to pass.
`canonical.test_definitions.count` remains a documented warn (P24
scope).

## Tests (10 / 10 pass)

- `TestUnderscoredAliasesExist::test_opus_4_7_is_module_attribute`
- `TestUnderscoredAliasesExist::test_gpt_5_4_is_module_attribute`
- `TestUnderscoredAliasesExist::test_gemini_3_1_pro_is_module_attribute`
- `TestAliasesMatchFrontier::test_opus_4_7_matches_direct`
- `TestAliasesMatchFrontier::test_gpt_5_4_matches_direct`
- `TestAliasesMatchFrontier::test_gemini_3_1_pro_matches_direct`
- `TestAliasesInAll::test_all_includes_three_aliases`
- `TestCanonicalMetricsRegex::test_check_regex_matches_opus_4_7`
- `TestCanonicalMetricsRegex::test_check_regex_matches_gpt_5_4`
- `TestCanonicalMetricsRegex::test_check_regex_matches_gemini_3_1_pro`

Ruff check + format clean.

## CI

- Draft phase: 16 SUCCESS / 52 SKIPPED / 0 failure (lane-aware suite).
- Ready phase (full suite): 49 SUCCESS / 69 SKIPPED / 1 PENDING after
  10-minute wait window. 0 FAILURE / 0 CANCELLED.
- Per v8 R10 (wait-window expired with checks still pending, CI green
  so far), proceed to Phase 4 and leave the note.

## Defense-in-depth observations

- No semantic change to the runtime frontier. The aliases re-export
  the same direct-provider IDs that the `*_DIRECT` constants already
  pinned (claude-opus-4-7 / gpt-5.5 / gemini-3.1-pro).
- No `__getattr__` magic — explicit module-level bindings so
  static-analysis treats them as constants.
- `__all__` extended in the same PR to keep wildcard imports coherent.
- Pattern follows the existing `GPT54_DIRECT = GPT55_DIRECT` legacy
  alias precedent already present in the file; no new naming
  convention introduced.

## Scope notes for v9

1. **Naming-convention freeze**: this PR pins the **underscored**
   form as the canonical-metrics contract. A future provider bump
   (GPT-5.5 → 5.6, Opus 4.7 → 4.8) will need both the `*_DIRECT`
   constant and the underscored alias updated. A v9 phase could
   audit `check_canonical_metrics.py` to look for the canonical
   `*_DIRECT` names directly, eliminating the alias requirement —
   that would be a one-time cleanup with no ongoing drift surface.

2. **Test gate**: the new `TestCanonicalMetricsRegex` suite reads the
   exact regex pattern from a literal string copy. If
   `check_canonical_metrics.py` changes its detection logic, the
   test won't follow automatically. A v9 phase could refactor to
   import the regex string from a shared constant — minor.

## Lane

`P20-model-pins-frontier-aligned` released at session close (status
`completed`, branch + pr captured in registry).
