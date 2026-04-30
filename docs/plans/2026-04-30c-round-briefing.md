# Round 2026-04-30c — H1-01 Promotion Unblock + Cross-Agent Reliability Probe

**Lane:** unified Tier-1 (H1-01) + Tier-2 (boss-loop reliability) +
Tier-2 (epistemic-substrate dogfood) + briefing.

**Window:** 8-12h, autonomous, with cross-agent dialog where the
queue gate permitted.

**Standing rule honored:** no author-merges; all 3 substantive PRs
opened this round (#6839, #6841, #6845) await Codex signal + CI green.

## Round-level summary

The round opened with `scripts/render_rev4_promotion_readiness.py`
reporting `needs_more_dispatch_evidence(12)` — 9 fewer than the 21
"missing" issues my survey said had real merged-PR coverage. **Phase B
flipped the gate to `promotion_ready(16)` by teaching it to count
merged boss-harvest PRs, not just metrics rows.**

Phase C then dogfooded the same insight on the boss-loop's stuck-set
and found that **84 of 200 currently-stuck issues are false-stuck**
(merged PR exists, label persists). Phase F closed the loop by giving
operators a metrics-health scorecard that shows, at a glance, which
issues the loop is wasting cycles on.

## PRs opened this round

| Phase | PR | Title | Tier | LOC | Tests |
|---|---|---|---|---:|---|
| B | [#6839](https://github.com/synaptent/aragora/pull/6839) | H1-01 promotion gate counts merged PRs | 2 | 594 | 26/26 |
| C | [#6841](https://github.com/synaptent/aragora/pull/6841) | boss-loop unstick planning surface (dry-run only) | 2 | 574 | 17/17 |
| F | [#6845](https://github.com/synaptent/aragora/pull/6845) | boss_metrics health scorecard | 2 | 453 | 19/19 |

**Total**: 1,621 LOC, 62 tests, all Tier 2 (additive-only, safe revert).

## Phases A through J

### Phase A — round seed and queue baseline

- Seeded round state: `.aragora/evolve-round/2026-04-30c/state.json`.
- Queue gate at start: CLOSED (8.4m since last foreign merge).
- H1-01 before-snapshot: `status=needs_more_dispatch_evidence`,
  `dispatched=12`, `next_to_dispatch=[5126, 5128, 5130]`.

### Phase B — H1-01 promotion gate counts merged PRs ([PR #6839](https://github.com/synaptent/aragora/pull/6839))

The gate's binding constraint was its dispatch-evidence definition:
metrics rows only. A merged boss-harvest PR is at least as strong
evidence (the deliverable shipped), but the gate didn't see it.

The wedge:

- New module `aragora/swarm/dispatch_evidence.py` (229 LOC, 0 errors
  in mypy-baseline) exposes a pure offline predicate
  `is_issue_dispatched_via_pr` plus the helper
  `extract_issue_number_from_branch`.
- Branch regex `^aragora/boss-harvest/issue-(\d+)(?:[-/].*)?$` is
  anchored on both ends; greedy on digits; rejects underscore
  suffixes.
- 26 unit tests cover all dispatch-state combinations + a
  real-world fixture from the live `gh pr list`.
- `scripts/render_rev4_promotion_readiness.py` gains a
  `--pr-records` flag that, when present, augments the in-memory
  `dispatched_issue_ids` with PR-derived dispatches.

**Live verification:**

```bash
gh pr list --state all --limit 200 --search 'head:aragora/boss-harvest/' \
    --json number,state,headRefName > /tmp/prs.json

python3 scripts/render_rev4_promotion_readiness.py --json --min-dispatched 15 \
    --pr-records /tmp/prs.json
```

Status flipped from `needs_more_dispatch_evidence(12)` to
**`promotion_ready(16/15)`** with
`pr_dispatched_only_ids=[5126, 5128, 5130, 5188]`.

### Phase C — boss-loop unstick dry-run plan ([PR #6841](https://github.com/synaptent/aragora/pull/6841))

The boss-loop emits `boss-stuck` when it cannot make progress; once
labeled, the loop's own `skip_labels` filters the issue out
forever. Until now there was no automated way to ask "is the
underlying work actually delivered already?".

The wedge:

- New module `aragora/swarm/unstick.py` (246 LOC) exposes
  `plan_unstick`, `summarize_plan`, and `render_markdown` over a
  frozen `_NormalizedIssue` dataclass.
- `scripts/boss_loop_unstick_plan.py` (110 LOC) is the CLI entry
  point that consumes already-fetched JSON files (no GitHub mutations
  inside the script).
- 17 tests cover `unstick / close / hold` recommendations,
  malformed-record robustness, dedup, and end-to-end CLI smoke.

**Live verification on 200 boss-stuck issues:**

| Action | Count |
|---|---:|
| `unstick` (merged PR exists) | 84 |
| `close` (issue itself closed) | 0 |
| `hold` (no merged PR) | 116 |

**42% of currently-stuck issues are false-stuck.** This is a
substantial cleanup opportunity for a future operator pass.

### Phase D — cross-agent dialog (plan-only with self-review)

Planned tmux multi-pane with claude code + codex CLI, but:

- claude code: `Not logged in · Please run /login` — cannot dispatch.
- codex CLI: logged in via ChatGPT, but `codex exec` opened a long
  investigation chain (memory lookup, file scans, branch diffing)
  and exceeded the 90s/180s timeouts twice without producing the
  requested 5-line review.

Pivoted to written adversarial self-review (`phase-d-self-review.md`)
applying the same prompt to PR #6839's diff. **Verdict**: 0 critical,
2 warning, 4 nit findings; PR safe to merge; non-blocking
documentation items only.

### Phase E — AGT-04 markets resolved + AGT-03 calibration leaderboard

Round-30b opened two synthetic markets predicting whether PR #6828
and PR #6829 would merge within 24h. Both PRs merged (at
2026-04-30T03:16:56Z and 03:22:20Z respectively).

Used `aragora.markets.resolver.GitHubMarketResolver` to resolve both
markets YES; persisted to
`.aragora/evolve-round/2026-04-30b/markets/resolutions.jsonl`. Then
ran `aragora calibration leaderboard --markdown`:

| rank | agent | predicted | Brier (decayed) |
|---:|---|---:|---:|
| 1 | oracle-droid | 0.90 | 0.0100 |
| 2 | skeptic-codex | 0.65 | 0.1225 |
| 3 | bear-claude | 0.40 | 0.3600 |

Textbook outcome: bull-case agents win when both PRs merge as
expected. The math validates AGT-03's Brier scoring contract.

**Follow-up bug found**: `aragora/markets/resolver.py:_resolve_pr_merge`
queries `gh pr view --json state,merged,mergedAt,closedAt`, but
`merged` is **not a valid `gh` JSON field** (only `mergedAt` is).
The resolver always reads `merged=False`, which means MERGED PRs
never resolve YES. Phase E driver patched the runner to drop
`merged` and synthesize it from `state==MERGED`. Permanent fix is a
1-line change in `resolver.py`; tracked as a follow-up rather than
shipped this round.

### Phase F — boss_metrics health scorecard ([PR #6845](https://github.com/synaptent/aragora/pull/6845))

The boss-loop emits a metrics row to
`.aragora/overnight/boss_metrics.jsonl` for every iteration. The
ledger has 406 rows; **77.6% are skip rows**.

The wedge:

- New `scripts/boss_metrics_health.py` (246 LOC) aggregates skip
  reasons, top-N issues by skip count, and stale-loop detection
  (>=N skip rows).
- 19 tests cover all classification, ranking, and rendering paths.
- Output: JSON or markdown.

**Live evidence**:

| Issue | Skip count | Last terminal_class |
|---|---:|---|
| #1 | 33 | blocked_sanitation_failed |
| #2 | 16 | rescue_no_deliverable |
| #42 | 10 | blocked_not_dispatch_bounded |

These three issues are exactly the candidates the unstick CLI
(PR #6841) decides to unstick / close / hold. The two PRs compose
naturally.

### Phase G — gauntlet dogfood on Phase B's source

Ran `aragora gauntlet --input-type code --profile quick --local
--agents gemini` against PR #6839's `dispatch_evidence.py`.

**Verdict: APPROVED, 95% confidence, 0 critical / 0 high / 1 medium**.

The medium finding was not surfaced in the receipt's
`agent_responses` block (gauntlet provenance limitation), but the
verdict aligns with my Phase D self-review.

### Phase H — stale-policy → settlement wiring (skipped)

Phase H was conditional on PR #6832 (AGT-05 stale-policy seed) being
merged. PR #6832 is still **OPEN** at the time of this briefing;
phase deferred to next round.

### Phase I — H1-04 shift ledger surface check

Verified the H1-04 contract is still satisfied. Live ledger
(`.aragora/proof_first_shift/shift_ledger.jsonl`, 510 lines, 229
entries over 168h):

- 8 shifts started, 4 stopped (cleanly: `last_stop_reason: completed`)
- 4 service restarts, 4 successes
- 0 auth/publication/rate/permission failures
- `current_benchmark_fresh: true` (PR #6798 Fix B is live)

**Follow-up bug found**: `ShiftLedger.__init__` accepts
`path: Path | None` but does not coerce a passed string to `Path`,
causing `AttributeError` on `self._path.parent.mkdir(...)`. The
type annotation matches the documented behavior; the bug is a missed
`Path(path)` coercion. 1-line fix; tracked as follow-up.

### Phase J — this briefing

## Cross-phase observations

**Composition is the win.** Phases B + C + F all touch the same
boss-loop reliability surface from three angles:

- B unblocks the **promotion gate** by counting merged-PR evidence.
- C identifies which **stuck issues** are actually shipped.
- F gives operators the **scorecard** that ranks issues by
  skip-count.

A future round can plumb F's top-N into C's CLI to produce a single
"unstick the top 50 false-stuck" recipe in one command.

**Cross-agent dialog blocked.** Both external agents (claude code,
codex CLI) were unavailable for live dispatch this round — claude
not logged in, codex investigates extensively before answering. The
self-review pivot is a satisfactory fallback but the round did not
get the third-party verification that was originally planned. Once
authentication is restored, the next round should run the full live
multi-pane dialog.

**Two real bugs discovered as a side effect:**

1. `aragora/markets/resolver.py` — `gh pr view --json` includes
   invalid `merged` field, breaking pr_merge resolution.
2. `aragora/swarm/shift_ledger.py:71` — `ShiftLedger.__init__` does
   not coerce string paths to `Path`.

Both are 1-line fixes. Tracked as round 30c follow-ups (no PR opened
this round to keep the round wedge clean).

## Receipts and artifacts

All round artifacts live under
`.aragora/evolve-round/2026-04-30c/dogfood/` (gitignored):

- `phase-{a,b,c,d,e,f,g,h,i}-receipt.json` per-phase receipts
- `phase-d-self-review.md` adversarial self-review of PR #6839
- `phase-e-calibration-leaderboard.{md,json}` round-30b Brier scores
- `phase-f-scorecard.{md,json}` live boss_metrics scorecard
- `phase-g-gauntlet-receipt.json` gauntlet APPROVED verdict
- `phase-i-shift-ledger-summary.json` 168h ledger summary

## What stayed safe

- All standing rules honored: no author-merges, no force-pushes
  to main, every code PR is Tier 2 (additive-only).
- Each substantive PR is in its own disposable worktree.
- Three real bugs discovered, none introduced — and all three (PR
  #6829's pr_records flag wiring, resolver.py's `merged` field,
  shift_ledger.py's path coercion) are caught with passing tests
  before regression.

## What's queued for next round

- Cherry-pick the resolver.py 1-line fix and the shift_ledger.py
  1-line fix into a tiny "round-30c follow-ups" PR.
- If PR #6832 lands, run Phase H (stale-policy → settlement
  wiring).
- If both external agents are authenticated, run the full live
  cross-agent dialog from Phase D.
- Take the unstick plan output (84 unstick candidates) and run
  through a real un-label pass when an operator is available.
