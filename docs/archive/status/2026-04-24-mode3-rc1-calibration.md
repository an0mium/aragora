# Mode 3 Calibration — v2.9.0-rc.1 Window

Second wave of heterogeneous-panel briefs, run on 2026-04-24 against the PRs
merged into `main` during the rc.1-preconditioning window. Supplements the
four-sample baseline in `docs/plans/2026-04-22-mode3-dogfood-findings.md`.

## Sample

15 briefs on disk under `.aragora/review-queue/briefs/`:

| Earlier | This wave |
|---|---|
| #6459 (×2), #6465, #6468, #6472 | #6448, #6456, #6462, #6466, #6471, #6476, #6479, #6483, #6486, #6490 |

All targets are merged commits; the panel was run post-merge, against the
PR head SHA at merge time. No brief was used to gate a merge — required
checks + review were the authoritative gate per current release contract.

## Aggregate numbers

| Metric | Value |
|---|---|
| Briefs | 15 |
| Verdict distribution | `repair_first`: 15/15 (100%) |
| Disagreement score | mean 0.02, max 0.12 |
| Overall confidence | mean 0.827, median 0.823, range 0.80–0.87 |
| Active panel slots | mean 8.5 / 8 target, range 7–9 |
| Cost per brief | mean $0.181, max $0.297 |
| Total cost | $2.713 |
| Wall clock | median 426 s (~7 min) |

## Reading the 100 % `repair_first`

Three honest interpretations, not mutually exclusive:

1. **Panel is calibrated strictly.** The rubric treats the presence of any
   plausible repair recommendation as sufficient to return `repair_first`,
   even when required checks + review are green. Useful for surfacing
   latent debt, but will not produce `approve` on real merged code at any
   useful rate.
2. **Every merged PR in this window genuinely has latent debt worth
   surfacing.** Spot-checking #6448 confirms the panel caught a real
   multi-path SQLite divergence issue that required-check coverage missed.
   That is a successful surfacing of legitimate repair signal.
3. **Ground truth is not in the dataset.** We do not have an oracle that
   labels each merged PR as "should have been blocked" vs "should have
   been approved." The ~95 % precision figure quoted in earlier
   communications is pre-this-sample and was anchored on a different set
   of briefs with different scoring. The number should be re-derived once
   we have labelled outcomes.

None of this is a regression against the rc.1 cut. It is a calibration
observation to feed into v2.9.0 stable planning.

## Implications for rc.1 → stable

- The N≥20 gate is at **15/20 = 75 %**. The remaining 5 can come from
  normal PR flow during the soak window.
- The calibration quality gate needs a rubric change, a label source, or
  both, before a "precision" number means anything. This is a stable-gate
  item, not an rc-blocker.
- Cost posture is clean: $0.18/brief real-world at median 7 min.
  A week of 10 briefs/day runs $12.60; a month $37.80. Sustainable.
- Panel slot fill is above spec (mean 8.5 / 8); extra slots indicate
  fallbacks occasionally filling multiple heterodox positions without
  dedup. Minor — track as gardening, not an rc-blocker.

## Follow-up items

1. **Rubric calibration pass**: reread the prompts that drive `verdict`
   selection and find why `approve` is effectively unreachable on
   post-merge code. Intended or rubric drift?
2. **Label source**: decide how ground truth gets attached to each brief.
   Options: (a) author self-label at PR-close, (b) post-hoc panel of
   reviewers, (c) regression correlation (did the flagged issue cause a
   later incident?).
3. **Per-slot fallback dedup**: investigate why some briefs show 9 active
   slots instead of 8. Likely an OpenRouter fallback filling both
   heterodox and regulatory positions.

## Pointers

- Brief storage: `.aragora/review-queue/briefs/pr-{N}-{sha}.json`
- Brief index: `.aragora/review-queue/briefs/index.jsonl`
- Generator: `scripts/generate_one_brief.py`
- Earlier findings: `docs/plans/2026-04-22-mode3-dogfood-findings.md`
- Release checklist: #6492
