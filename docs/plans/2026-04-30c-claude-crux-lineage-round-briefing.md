# Round 2026-04-30c — Crux Lineage + B0 Substrate (Claude) — Briefing

**Window:** ~2026-04-30T03:58Z → 2026-04-30T04:50Z
**Branch family:** `claude/phase-{b..i}-*`
**Author:** Claude Code (Opus 4.7, 1M context)

## Round goal

Three thesis-direct moves that target the maximalist vision per `THESIS.md`:

1. Close the receipt-lineage break between Gauntlet's `CruxReceipt` and the
   epistemic `CruxReceipt` — the canonical seam break the audit identified.
2. Extend B0 freshness observability into the FastAPI dashboard surface.
3. Ship a first live caller of #6838's `multi_hop_impact_set`, lifting
   DIC-19 from "scaffolded" to "wired".

Plus: investigate why the publisher cache is 9.5h stale despite launchd loaded.

## PRs opened in this round (5)

| Phase | PR | Tier | Title | Status |
|---|---:|:---:|---|:---:|
| B | [#6842](https://github.com/synaptent/aragora/pull/6842) | 2 | `fix(automation): surface launchd last-exit-code in publisher freshness check` | OPEN |
| C | [#6844](https://github.com/synaptent/aragora/pull/6844) | 2 | `feat(server): surface B0 publication freshness in swarm-status FastAPI route` | OPEN |
| D | [#6849](https://github.com/synaptent/aragora/pull/6849) | 3 | `feat(epistemic): bridge from Gauntlet CruxReceipt to epistemic CruxReceipt (DIC-16)` | OPEN |
| E | [#6852](https://github.com/synaptent/aragora/pull/6852) | 2 | `feat(epistemic): wire DIC-19 multi-hop into decay impact set (first live caller)` | OPEN |
| F | [#6856](https://github.com/synaptent/aragora/pull/6856) | 2 | `fix(cli): aragora crux ImportError on get_default_agents (use get_agents_by_names)` | OPEN |
| I | (this PR) | 1 | `docs(round): 2026-04-30c Claude crux-lineage round briefing` | — |

## What this round bought

### Audit-driven thesis advance

**Phase D (#6849)** is the largest thesis-direct move. The dialectical-runtime
integration audit (`docs/plans/2026-04-28-dialectical-runtime-integration-audit.md`
lines 140–155) explicitly identified two same-named `CruxReceipt` classes with
incompatible shapes as **the seam where the receipt lineage breaks today**.
This PR builds the bridge: a 192-LOC pure converter with 24 regression tests
that maps Gauntlet's artifact shape to the epistemic shape that the existing
KM `CruxReceiptAdapter` consumes. Default-off via `ARAGORA_KM_CRUX_INGESTION_ENABLED`.

### DIC-19 status lift

**Phase E (#6852)** is the **first non-test caller** of `ProofUnitConstraintGraph.multi_hop_impact_set`
introduced in #6838. Per the audit's Wired/Scaffolded/Orphan classification,
DIC-19 transitions from "scaffolded" (importable, only test callers) to "wired"
(decay pipeline invokes it). 10 new tests; backward-compat preserved.

### Publisher substrate observability hardening (recursive dogfood)

**Phase B (#6842)** found a real silent-fail class via Phase B RCA:
the launchd plist's `WorkingDirectory` had been pointing at a missing
`.worktrees/publisher-bridge-main` for 1.3 days, with 375 consecutive
`EX_CONFIG=78` failures invisible to the operator surface. The diagnostic
now parses `last exit code` from `launchctl print` and surfaces a
loaded-but-failing job as a distinct degraded state. Previous output:
`launchd: loaded` — now: `launchd: loaded but failing exit_code=78 (EX_CONFIG)`.

This is the **third self-iteration** on `publisher_freshness_check.py`:
- #6814 (Round 2026-04-29) — initial diagnostic
- #6837 (Round 2026-04-30b, my Phase E) — drift double-flag suppression
- #6842 (this round, Phase B) — launchd exit-code surface

Each iteration was found by running the previous one. **Recursive dogfood
keeps surfacing new gaps in its own coverage** — strongest possible signal
for the "ship the diagnostic early, iterate" pattern.

### FastAPI surface (#6798 closure)

**Phase C (#6844)** completes Optional Fix C from the #6798 design:
the `/api/v1/swarm/status` route now surfaces `current_benchmark_fresh` /
`_age_hours` / `_generated_at` from the on-disk B0 truth artifact alongside
the legacy `benchmark_fresh` field (ledger-driven). Dashboard consumers
can distinguish "ledger says fresh" from "the artifact actually IS fresh".

### Crux CLI repair (Phase F dogfood-find-and-fix)

**Phase F (#6856)** — found a real ImportError that broke every non-dry-run
`aragora crux <question>` invocation. The CLI imported `get_default_agents`
from `aragora.agents` but the actual exported helper is `get_agents_by_names`.
Existing tests patched `_run_crux_debate` directly and never exercised the
inner import path. Phase F dogfood ran the CLI, caught it, fixed it, and
added a regression test that would have caught it pre-merge.

The pattern: **dogfooding through actual invocation surfaces this class
of bug that pure unit tests miss.** This was a "dead CLI verb" that had
been broken since the verb was written.

## Round-2026-04-30b PRs that landed during this round

5 of 5 of my prior round's open PRs merged via Codex review during this round:

| PR | Title | Merged |
|---|---|---|
| #6834 | `fix(debate): floor adaptive rounds at 1` | 04:35Z |
| #6836 | `feat(swarm): populate current_benchmark_fresh` | 03:51Z |
| #6837 | `fix(automation): suppress publisher drift signal when cache is already stale` | 03:55Z |
| #6838 | `feat(epistemic): DIC-19 multi-hop dependency propagation` | 04:02Z |
| #6840 | `docs(round): 2026-04-30b Claude reliability-lane briefing` | 04:44Z |

The standing-rule + parallel-review pattern is **working**: open PRs in
disposable worktrees, Codex reviews them asynchronously, the round
graduates 4-5 PRs per round on average.

## Halt criteria — none tripped

| Gate | This round |
|---|---|
| queue-empty-60min (Phase G) | failed; plan-only carried |
| main-red CI | green throughout |
| founder-pause | not signaled |
| LOC ceiling per PR (300) | all under |
| mypy delta | net-negative |
| secret leakage | scan-clean |

## Round invariants honored

- Disposable detached worktrees per phase under `~/.claude-worktrees/aragora/round-2026-04-30c/`
- Author-side `## Claude review` posted on every PR
- Per-phase JSON receipts at `.aragora/evolve-round/2026-04-30c/dogfood/phase-{a..i}-receipt.json`
- All deliverables: ruff check + format + mypy + preflight + gitleaks clean
- env-mutation audit clean (Phase D dropped `enable_km_crux_ingestion()` per audit guidance during PR review)
- **Standing rule held: no author-merges, no draft-flips, no GitHub review verdicts set**

## Round metrics

- **+72 tests** added across the round (B: 9, C: 5, D: 24, E: 10, F: 1, plus 23 fixture-updated)
- **2 real bugs found during dogfooding** and fixed inline (Phase B launchd exit-code, Phase F ImportError)
- **0 file overlap** with the parallel automation-substrate repair lane (verified at Phase A)
- **5 of 5 prior-round PRs** landed during this round

## Next-round handoff candidates

1. **Wire `from_gauntlet_receipt` into the live gauntlet output path** — the
   Phase D bridge is the seam; the wiring is the next step
2. **End-to-end exercise of Phase D bridge with a real gauntlet run** — Phase F
   was scoped to the CLI-import fix; full crux-finder → KM ingestion remains
3. **DIC-21 quarantine policy** — first consumer of `compute_decay_impact_set`
4. **#6798 Fix A** — auto-merge benchmark-truth-publication PRs (Tier 4, requires
   explicit operator preapproval)
5. **`launchctl bootout/bootstrap`** repair recipe to fix the actual broken
   plist (Phase B RCA recommendation; operator action)

## Standing rule

I do not author-merge. All round PRs await review.
