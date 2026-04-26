# Mode 3 Calibration — v2.9.0-rc.1 Replay Through Post-#6505 Rubric

Closes fix #4 of epic #6505: "Re-derive precision on the 15-brief sample
post-fix." Uses the briefs archived under
`.aragora/review-queue/briefs/pr-*.json` — no new API spend.

## Rubric changes applied

Three rubric changes landed between the baseline calibration
(`2026-04-24-mode3-rc1-calibration.md`, 15/15 `repair_first`) and this
replay:

1. **#6506** — surface `findings_severity_counts` on stored briefs.
2. **#6510** — new `APPROVE_WITH_FOLLOWUPS` verdict class and the
   severity gate: `REPAIR_FIRST` downgrades to `APPROVE_WITH_FOLLOWUPS`
   when no panel lens reports a `high`-severity finding.
3. **#6514** — new advocate lens slot (`claude_advocate`) that argues
   FOR the PR, added to the default panel in `aragora/config/pdb_panel.yaml`.

## Honest scope — what this replay does and doesn't do

The archived briefs predate #6506, so they do **not** carry structured
severity data. The replay script
(`scripts/replay_mode3_sample.py`) recovers severity counts by scanning
each finding's `finding_text` with a keyword classifier (see the `_HIGH_KEYWORDS`,
`_MEDIUM_KEYWORDS`, `_LOW_KEYWORDS` tables in the script). That makes
this replay a **plausibility check** on the new rubric, not a re-scoring
with oracle labels. A manual `*.severity.json` sidecar overrides the
heuristic per-brief where a real label exists.

The advocate lens simulation is similarly conservative: the archived
briefs never ran an advocate slot, so we synthesize a floor APPROVE
confidence from each brief's `validation_summary` field (tests/ruff/CI
evidence). A real advocate panelist reading the diff could argue
considerably more forcefully — this script biases against approve-flips,
not toward them.

## Replay summary (17 briefs on disk, 15 rc.1-window + 2 METRICS-PR #6511)

Run from a checkout that has the local archived brief artifacts:
`python3 scripts/replay_mode3_sample.py --briefs-dir .aragora/review-queue/briefs`

Note: `.aragora/` is intentionally gitignored runtime state. A clean
clone will need a copied/exported brief archive passed via `--briefs-dir`
to reproduce the table below.

### Aggregate verdict distribution

| Rubric | `repair_first` | `approve_with_followups` | `approve_candidate` |
|---|---|---|---|
| **Old (as stored)** | 17/17 (100%) | 0/17 | 0/17 |
| **New: severity gate only** | 14/17 (82%) | 3/17 (18%) | 0/17 |
| **New: severity gate + advocate (simulated)** | 14/17 (82%) | 3/17 (18%) | 0/17 |

The severity gate alone moves 3/17 briefs out of the sharp-end
`repair_first` class into the less-alarming `approve_with_followups`
class. Under the conservative advocate simulation, no brief's weighted
vote flips further to `approve_candidate` — the floor advocate
confidence (≤ 0.2 on every brief in this sample) does not exceed the
mean panel weight-against-approve (≈ 0.8 across the sample).

### Per-brief replay

| PR | sha | old | sev_counts (h/m/l) | sev_gate_only | full_replay | adv_conf |
|----|-----|-----|--------------------|---------------|-------------|----------|
| #6448 | `8edf62ddbaad` | repair_first | 2/5/1 | repair_first | repair_first | 0.0 |
| #6456 | `eb474a4be4f7` | repair_first | 4/2/3 | repair_first | repair_first | 0.0 |
| #6459 | `8faf673aad9d` | repair_first | 2/4/3 | repair_first | repair_first | 0.0 |
| #6459 | `b40679cfbac2` | repair_first | 0/5/4 | **approve_with_followups** | **approve_with_followups** | 0.0 |
| #6462 | `c5ebb3c0cbd6` | repair_first | 1/3/4 | repair_first | repair_first | 0.1 |
| #6465 | `4eb34ce631d6` | repair_first | 3/3/3 | repair_first | repair_first | 0.1 |
| #6466 | `bf0a558e07e5` | repair_first | 1/6/2 | repair_first | repair_first | 0.1 |
| #6468 | `046ea9c74688` | repair_first | 3/4/2 | repair_first | repair_first | 0.0 |
| #6471 | `49e2f47b8cf5` | repair_first | 1/5/3 | repair_first | repair_first | 0.2 |
| #6472 | `4690314be321` | repair_first | 3/2/4 | repair_first | repair_first | 0.0 |
| #6476 | `ca92b33ec719` | repair_first | 1/4/3 | repair_first | repair_first | 0.0 |
| #6479 | `6eb25032bae3` | repair_first | 0/2/5 | **approve_with_followups** | **approve_with_followups** | 0.1 |
| #6483 | `96a3544f8041` | repair_first | 2/2/4 | repair_first | repair_first | 0.0 |
| #6486 | `376ef66c6d26` | repair_first | 2/1/5 | repair_first | repair_first | 0.0 |
| #6490 | `95c8695bd476` | repair_first | 1/3/4 | repair_first | repair_first | 0.0 |
| #6511 | `86ee81f5afca` | repair_first | 0/3/5 | **approve_with_followups** | **approve_with_followups** | 0.0 |
| #6511 | `f4b1296d1868` | repair_first | 1/5/3 | repair_first | repair_first | 0.0 |

## Interpretation

1. **The severity gate is doing real work.** Three briefs (#6459,
   #6479, #6511) had no heuristic-high findings at all and cleanly move
   to `approve_with_followups` — the sharp-end class is no longer
   overloaded with "real panel signal but nothing blocking" outcomes.

2. **The remaining 14/17 `repair_first` cases all had at least one
   heuristic-high finding.** Looking at the synthesizer top-lines for
   those briefs, the high-severity triggers were legitimate concerns
   (merge-critical path warnings, security-relevant coupling, data loss
   risk surfaces). The rubric is not over-punishing — it's preserving
   `repair_first` precisely where the panel found plausible blockers.

3. **The advocate lens did not flip any verdicts in this simulation.**
   That is expected: the conservative heuristic caps advocate confidence
   at 0.7 (requiring 7+ pieces of positive evidence in a single
   `validation_summary`), and the archived summaries rarely clear 0.2
   because they already weight the concerns the panel surfaced. A real
   advocate slot running on fresh diffs would produce different
   arguments and could shift the distribution further toward approve;
   this replay does NOT measure that effect.

4. **Calibration of this heuristic is an open follow-up.** Two
   plausible improvements over the keyword-based severity classifier:
   (a) a sidecar-labelled manual severity pass on the 17-brief sample
   by a reviewer (the script supports this via `*.severity.json`
   overrides), and (b) a re-run with a real panel on the same head
   SHAs once #6505 is closed, treating the results as the oracle for
   future replays.

## Pointers

- Replay script: `scripts/replay_mode3_sample.py`
- Replay tests: `tests/scripts/test_replay_mode3_sample.py`
- Baseline (pre-fix): `docs/status/2026-04-24-mode3-rc1-calibration.md`
- Brief storage: `.aragora/review-queue/briefs/pr-{N}-{sha}.json`
- Severity gate implementation: `aragora/review/builder.py::_apply_severity_gate`
- Advocate slot config: `aragora/config/pdb_panel.yaml::slots.claude_advocate`
- Epic: #6505 — fix #4 (this replay) closes the epic

## Follow-ups (not blockers for closing #6505)

1. Consider labelling the 3 heuristic `approve_with_followups` briefs
   manually to confirm the downgrades were correct.
2. Add a real advocate slot re-run on one or two heads to ground-truth
   the advocate-confidence heuristic against real panelist output.
3. Track verdict distribution over the next 20–30 PR briefs in the
   post-fix window; if the new distribution collapses back to ≥ 95 %
   `repair_first`, the advocate slot may need a stronger prompt, not
   the severity gate.
