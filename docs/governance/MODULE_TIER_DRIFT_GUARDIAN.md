# Module Tier Drift Guardian — Design Plan & Finding

Status: **proposal (plan-only)** — no code or CI/pre-commit changes are made by this
document. It records a correction to a previously-assumed problem and lays out
guardian options with a risk analysis so a human can choose a direction.

Owner surface: `scripts/regenerate_module_tiers.py`,
`.github/workflows/module-tier-drift.yml`, `aragora/module_tiers.yaml`.

## TL;DR

- The "Module Tier Drift" CI check **does not fail on metric-count drift.** Its
  verification step runs `regenerate_module_tiers.py --check`, and `check_drift()`
  only reports **tier changes, new modules, and removed modules** — it never
  compares `importer_count` / `py_files` / `test_file_count`.
- Therefore a **blanket "auto-regenerate on every `aragora/**` change" pre-commit
  hook is the wrong fix**: it would prevent *zero* real CI failures (counts never
  fail the gate) and would **silently apply tier promotions/demotions**, defeating
  the exact tripwire the workflow exists to enforce ("Silent promotion/demotion is
  as harmful as silent metric drift", per the workflow header).
- The recurring red on the `check` job observed across PRs (#7844, #7873) was
  **`actions/checkout` cancellation** (`##[error]The operation was canceled.` at
  the Checkout step), i.e. concurrency/infra — **not** drift. The drift logic never
  ran to completion in those failures.
- **Recommendation:** do not ship the blanket auto-regen hook. Pursue the two
  genuinely useful, lower-risk items instead (B and D below), and keep the one-time
  cosmetic count refresh (PR #7873) as-is. Human decides which option(s) to fund.

## Evidence (reproducible on current `origin/main`)

Run from a clean checkout at `origin/main` (captured at `1cb0a4144c`):

```
$ python3 scripts/regenerate_module_tiers.py --check
No tier drift.
[--check exit: 0]

$ cp aragora/module_tiers.yaml /tmp/before.yaml
$ python3 scripts/regenerate_module_tiers.py >/dev/null
$ git diff --stat aragora/module_tiers.yaml
 aragora/module_tiers.yaml | 54 +++++++++++++++++++++++------------------------
 1 file changed, 27 insertions(+), 27 deletions(-)
$ git diff aragora/module_tiers.yaml | grep -E '^[+-].*(name:|tier:|override_reason:)'
(none — counts only)
```

So: the committed yaml is **count-stale** (a full regen rewrites 27 count lines),
yet `--check` reports **no drift and exits 0**. The CI gate is green on count drift.
A second regen is idempotent (no further diff), confirming the generator is stable.

The `check` CI failures were cancellations, not drift — confirmed from the job log:

```
check  Checkout    ##[error]The operation was canceled.
check  Emit summary  ## Module Tier Drift Check
check  Emit summary    - Status: failed — `aragora/module_tiers.yaml` is stale
```

The "stale" line is **conditional boilerplate in the `Emit summary` step** (it runs
`if: always()` and prints the failure branch whenever `job.status != success`, which
includes cancellation). It does **not** indicate the drift comparison ran.

## What the check actually guards (by design)

From `scripts/regenerate_module_tiers.py::check_drift()`:

- `NEW module: X -> tier` — a top-level `aragora/` package appeared.
- `TIER CHANGE: X 'old' -> 'new'` — a module crossed a maturity boundary
  (core/integrated/experimental/deprecated).
- `REMOVED module: X` — a package disappeared.

Tier classification is evidence-based (importer count + test coverage, with a small
`MANUAL_TIER_OVERRIDES` map). The drift gate is intentionally a **human-review
tripwire for tier movement**, not a freshness check for counts. Counts are recorded
for the cold-auditor truth surface but are not gated.

## Guardian options (with risk)

### A. Blanket auto-regen pre-commit hook on `aragora/**` — **REJECTED**
Run `regenerate_module_tiers.py` and `git add aragora/module_tiers.yaml` whenever
`aragora/**` or `tests/**` change.
- Prevents **no** real CI failure (count drift is not gated).
- **Silently rewrites tier classifications**, defeating the drift tripwire — the
  workflow explicitly calls this out as harmful.
- Edits an approval-required surface (`.pre-commit-config.yaml`).
- Verdict: do not do this.

### B. Pre-push **tier-only** mirror hook — **LOW RISK, OPT-IN VALUE**
Add a `stages: [pre-push]` local hook that runs `regenerate_module_tiers.py --check`
(the same tier-only logic CI uses) and **fails the push** with the regenerate
instructions when a tier actually drifts. It never mutates the yaml; it only
surfaces tier movement earlier than CI.
- Pros: catches the *one* condition that actually fails CI, before push; no silent
  mutation; mirrors CI exactly.
- Cons: duplicates a check CI already runs; still edits the approval-required
  `.pre-commit-config.yaml`; runs a `git ls-files`-based scan on every push.
- Verdict: reasonable if the team wants earlier local signal; needs human approval.

### C. CI autofix-commit on drift — **REJECTED**
Have the workflow regenerate and commit the yaml on drift. Same silent-promotion
hazard as (A), plus a bot writing to PR branches. Do not do this.

### D. Fix the real problem: checkout-cancellation robustness — **RECOMMENDED (separate)**
The observed reds are `actions/checkout` cancellations. Options:
- Audit the `concurrency: cancel-in-progress: true` interaction for this workflow
  (and the sibling `portability` job) to confirm whether self-cancellation or
  runner preemption is the cause.
- Add a re-trigger guardian (boss-loop step or a small workflow) that re-runs
  **checkout-canceled** required/advisory jobs once, to cut false-red noise.
- This is workflow/automation work (approval-required) and is independent of tier
  drift; it addresses the actual recurring failure.

### E. Scheduled count refresh — **OPTIONAL, LOW VALUE**
The workflow already runs weekly (`cron: '0 6 * * 1'`). A scheduled job could open a
`chore(tiers)` PR with the regenerated counts so the truth surface never lags long.
Keeps counts fresh without a tripwire conflict (PR is human-reviewed), but the
payoff is purely cosmetic. PR #7873 is the manual one-shot equivalent.

## Recommendation

1. **Do not** implement (A) or (C).
2. Treat the recurring `check`/`portability` red as an **infra/cancellation** issue
   and pursue (D) as the real fix.
3. Optionally adopt (B) if earlier local tier-drift signal is wanted.
4. Keep **PR #7873** (one-time count refresh) — harmless and improves the truth
   surface — but understand it does **not** fix a failing gate.

## Decision needed

Which of {B, D, E} (if any) to fund, and whether checkout-cancellation robustness
(D) should be owned here or by the broader CI-resilience lane
(`docs/governance/ci-main-guardrails.md`).
