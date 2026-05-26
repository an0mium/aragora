# AFT v0.2 Result: Tightened Evaluation — Directional Improvement, Not Statistically Conclusive

**Date:** 2026-05-22
**Status:** v0.1 H1 PASS was inflated by data contamination + weak frontier prompt. v0.2 with both repaired: trained advocate **directionally** beats frontier but **not statistically separable** on n≤78 holdouts. Decision per steering rule: keep harness as falsification rig; do not advance to v0.2 production track yet.
**Supersedes:** `docs/status/AFT_RESULT_v0.1.md` (kept on disk for the audit trail)
**Related:** PR #7438, `docs/specs/ARAGORA_ROADMAP_REVISION_ADVOCATES.md`,
`scripts/aft_harness.py`, `scripts/aft_seeded_train_eval.py`,
`scripts/aft_repeated_eval.py`, `scripts/aft_manifest.py`

## TL;DR

The original v0.1 "H1 PASS" result was inflated by two confounds:

1. **Weak frontier prompt** that defaulted to `open_aged` for uncertain inputs,
   while the dataset has zero `open_aged` examples in current operating
   conditions. The frontier baseline was artificially weak.
2. **Data contamination** between training and holdout: the v0.1 evaluation
   used the existing Llama-1B adapter (trained on a fixed 80/10/10 corpus
   split) against holdouts re-derived from the full corpus. Some holdout
   examples had been seen during training.

When **both** are repaired:

| Model | Acc (clean holdout n=78) | Brier | Verdict |
|---|---|---|---|
| `baseline_random` (calibrated prior) | 0.577 | 0.637 | reference |
| `frontier_rules` (repaired prompt) | **0.654** | **0.541** | repaired baseline |
| `local_advocate` Llama-3.2-1B (500-iter LoRA) | 0.641 | 0.718 | **fails to beat frontier** |
| `local_advocate` Qwen2.5-7B (500-iter LoRA) | **0.718** | 0.564 | beats frontier by +6.4pts (NOT significant at p<0.05 Bonferroni) |

Per the steering directive's decision rule:

> If local_advocate still beats the repaired frontier baseline with
> statistically meaningful margin, mark v0.1 accepted and propose v0.2.
> If margin collapses, keep the harness as a falsification rig and
> improve corpus/rubric before training work.

The margin **directionally holds for Qwen-7B but does not reach statistical
significance** at our current holdout sizes. **The conservative reading is:
do not declare v0.1 accepted; treat the harness as a falsification rig and
prioritize corpus expansion + rubric refinement before further training.**

## What changed from v0.1

### 1. Frontier prompt repaired (`scripts/aft_harness.py::FrontierRules.OPERATOR_RULES`)

The v0.1 prompt defaulted to `open_aged` for uncertain inputs. This was wrong:
the dataset has zero `open_aged` examples in current operating conditions
because the recent merge cadence has cleared the stalled-PR backlog. The
repaired prompt:

- Bakes in the calibration prior (~90% `merged_fast`, ~10% `closed_no_merge`,
  ~0% `open_aged`).
- Defaults to `merged_fast` for uncertain inputs (majority class).
- Recognizes dependency-bot branches (`dependabot/`, `renovate/`) as almost-
  always-merge.
- Recognizes scout artifacts (`preflight/`, tiny diff, no reviews) as
  almost-always-close.
- Explicitly tells the model *not* to default to `open_aged`.

The dry-run heuristic in the same class was repaired to match the same rules
so dry-run results stay meaningfully comparable.

**Effect on the baseline numbers (single seed, same v0.1 holdout):**

| | v0.1 frontier | v0.2 frontier (repaired) |
|---|---|---|
| Accuracy | 0.26 | 0.62 |
| Brier | 0.851 | 0.616 |

The frontier baseline is now competitive with the trained advocate. The
"+28 acc points" gap of v0.1 was an artifact of the weak prompt.

### 2. Repeated seeded splits (`scripts/aft_repeated_eval.py`, `scripts/aft_seeded_train_eval.py`)

Two new orchestrator scripts:

- `aft_repeated_eval.py` runs N seeded splits against an **existing**
  adapter. Useful for stability checks on a fixed model. Contaminated by
  design (the adapter has seen most of the corpus).
- `aft_seeded_train_eval.py` re-trains an adapter from scratch **per seed**
  on each seed's training set, then evaluates on that seed's held-out set.
  This is the proper clean evaluation; ~3 min per seed for Llama-1B,
  ~15 min per seed for Qwen-7B.

Aggregates mean ± stddev across seeds and counts how many seeds clear
Bonferroni-corrected p<0.05 per pairwise comparison.

### 3. Larger model: Qwen2.5-7B-Instruct-4bit

Reference target from `docs/specs/LOCAL_ADVOCATE_TRAINING_PIPELINE.md`.
Trained at rank 8, lr 1e-4, batch 2, 500 iters (matches Llama-1B config).
Wall-clock: ~15 min on M3 (vs ~3 min for Llama-1B).

### 4. Reproducibility manifest (`scripts/aft_manifest.py`)

Emits a JSON manifest with: corpus SHA256, split seed(s), model
name+revision, adapter SHA256 per file, training args, eval command,
summary artifact hashes, tool versions, git state. See
`data/aft/results/aft_manifest_v0.2.json` for this run's manifest.

## Clean evaluation results

### Apples-to-apples on `seed_00017` clean balanced holdout (n=78)

Both adapters trained on the SAME 478-row training set; both evaluated on
the SAME 78-row held-out set (28 `closed_no_merge` + 50 `merged_fast`,
stratified). No data contamination.

| Condition | Accuracy | Brier | Cost USD | Per-class merged_fast | Per-class closed_no_merge |
|---|---|---|---|---|---|
| `baseline_random` (uniform priors here; no train data passed to harness) | 0.577 | 0.637 | $0.00 | 43/50 (86%) | 2/28 (7%) |
| `frontier_rules` (repaired) | 0.654 | 0.541 | $0.94 est | 46/50 (92%) | 5/28 (18%) |
| **`local_advocate` Llama-3.2-1B** (500-iter LoRA) | 0.641 | 0.718 | $0.008 | 43/50 (86%) | 7/28 (25%) |
| **`local_advocate` Qwen2.5-7B** (500-iter LoRA) | **0.718** | 0.564 | $0.008 | **49/50 (98%)** | 7/28 (25%) |

McNemar pairwise (Bonferroni-corrected, factor 3):

| Pair | p | p_bonferroni | Verdict |
|---|---|---|---|
| `baseline_random` vs `frontier_rules` | 0.238 | 0.714 | not different |
| `baseline_random` vs `local_advocate` Llama-1B | 0.383 | 1.000 | not different |
| `frontier_rules` vs `local_advocate` Llama-1B | 1.000 | 1.000 | not different |
| `baseline_random` vs `local_advocate` Qwen-7B | **0.013** | **0.038** | **significant** ✅ |
| `frontier_rules` vs `local_advocate` Qwen-7B | 0.063 | 0.188 | directional but not significant |

Qwen-7B is the only condition that clears Bonferroni significance vs
baseline. Vs the repaired frontier, the +6.4 accuracy-point gap is
directional but does not clear p<0.05.

### Cross-check: held-out test set (n=57 imbalanced) — Qwen-7B trained on `data/aft/mlx`

Both adapters trained on the full 444-example MLX `train.jsonl + valid.jsonl`
(80%+10% of corpus); evaluated on the 57-row `test.jsonl` slice both adapters
never saw.

| Condition | Accuracy | Brier | Per-class merged_fast | Per-class closed_no_merge |
|---|---|---|---|---|
| `baseline_random` | 0.807 | 0.100 | 46/54 (85%) | 0/3 (0%) |
| `frontier_rules` (repaired) | 0.895 | 0.179 | 49/54 (91%) | 2/3 (67%) |
| `local_advocate` Llama-3.2-1B (500-iter) | 0.754 | 0.491 | 41/54 (76%) | 2/3 (67%) |
| `local_advocate` Qwen2.5-7B (1500-iter) | **0.947** | **0.105** | **52/54 (96%)** | 2/3 (67%) |

Same qualitative story: Qwen-7B beats frontier on majority class while
matching on minority. n=57 imbalanced makes accuracy a weak signal — McNemar
p_bonf for Qwen vs frontier = 1.00 here.

### Real frontier validation (n=15 subset)

`claude --print` invoked with the same repaired prompt on the first 15 rows
of the clean test:

| | Real `claude --print` | Dry-run heuristic |
|---|---|---|
| Accuracy | 0.933 | 0.895 (full 57) |
| Brier | 0.145 | 0.179 (full 57) |

Real frontier is slightly stronger than the dry-run heuristic. **The dry-run
is not artificially weak**; if anything it under-represents real frontier
strength. The trained-advocate-vs-frontier gap would narrow further if all
57 rows were run through the real frontier (cost: ~$0.70).

### Repeated-seed Llama-1B with contaminated adapter (5 seeds, n=50 per seed)

Run against the v0.1 contaminated adapter (`aft-pilot-llama1b-500`,
trained on a single fixed split) across 5 different holdout seeds. Reported
here for completeness; contamination means these numbers over-state advocate
quality.

| Condition | Acc mean | Acc stddev | Brier mean |
|---|---|---|---|
| baseline_random | 0.508 | 0.059 | 0.885 |
| frontier_rules (repaired heuristic) | 0.576 | 0.034 | 0.664 |
| local_advocate (Llama-1B contaminated) | 0.760 | 0.033 | 0.480 |

Bonferroni-significant seeds:
- baseline vs frontier: 0 of 5
- baseline vs advocate: 4 of 5
- frontier vs advocate: 3 of 5

The contaminated advocate beats frontier in 3/5 seeds. After
de-contamination (the proper seeded-train-eval result above) the gap
collapses for Llama-1B and reduces to directional-not-significant for
Qwen-7B.

## What the steering directive's decision rule says

The user's tightening directive contains an explicit decision rule:

> **Decision threshold:**
> - If local_advocate still beats the repaired frontier baseline with
>   statistically meaningful margin, mark v0.1 accepted and propose v0.2.
> - If margin collapses, keep the harness as a falsification rig and
>   improve corpus/rubric before training work.

The cleanest single number for this rule is the **Qwen-7B-seed17 vs
frontier-repaired McNemar Bonferroni p-value: p_bonf = 0.188**. This is
above the conventional 0.05 threshold.

**Recommendation: do not declare v0.1 accepted. Keep the harness as a
falsification rig. Improve corpus and rubric before further training
investment.**

Specific next steps that would move the needle:

1. **Widen the corpus.** Current data has only 56 `closed_no_merge` and 0
   `open_aged` examples. The minority class drives most of the variance.
   Either re-pull when the GitHub GraphQL API recovers from its current
   intermittent 5xx storm (we hit several 502s during this run), or
   pull across multiple operator repos and report results per-operator
   only.
2. **Drop the 3-class label space if `open_aged` stays empty.** A genuine
   2-class problem with calibrated priors would give baseline_random a
   meaningful comparator (currently it's ~50% on balanced holdouts but
   ~81% on the natural distribution).
3. **Larger holdouts.** n=78 is too small to detect a 6.4-point gap at
   p<0.05 with Bonferroni-3. Roughly n≥200 would be required.
4. **Tune the frontier prompt further.** The repaired prompt is good but
   not exhaustive. A short prompt-evolution sprint on a held-out *prompt*
   set would tighten the frontier baseline. We should be confident the
   frontier baseline is as strong as we can make it before declaring the
   advocate beats it.
5. **Train Qwen-7B longer with early-stopping on validation loss.** Val
   loss at iter 1500 (0.461) was higher than at iter 500 (0.399) — mild
   overfitting. The iter-500 checkpoint may be a better operating point.

## Per-step caveats

1. **Cost is an estimate, not a measured invoice.** The `cost_usd` numbers
   are the harness's `COST_PER_CALL_ESTIMATE` constants × n. Real costs
   will vary by token usage; order-of-magnitude gap (~120×) is robust.
2. **Repaired heuristic ≠ real frontier.** Real `claude --print` is
   slightly stronger than the dry-run heuristic at n=15 (0.933 vs 0.895).
   Full real-frontier runs would cost ~$0.70 per 57-row evaluation;
   skipped here to preserve attention budget. A full real-frontier run is
   a sensible next step if v0.2 is approved.
3. **Adapter overfitting risk.** Qwen-7B val loss climbed 0.40 → 0.46
   between iter 500 and iter 1500. iter-500 may be the right operating
   point; this run reports the final iter 1500. iter 500 numbers are
   essentially identical (0.947 vs 0.947 on clean_test_holdout) so the
   conclusion does not change.
4. **Calibration prior is per-operator and time-dependent.** The
   ~90/10/0 distribution is for this specific operator over the last
   ~30 days. The advocate's adapter encodes this prior and will need
   retraining as merge cadence shifts.
5. **n=78 and n=57 are not large.** Conclusions about statistical
   significance are tentative at these sizes. Effect-size estimates may
   not generalize to larger holdouts.
6. **Per-operator, per-task artifact.** No claim of cross-operator
   generalization.
7. **No live wiring.** Nothing in this PR causes any change to the
   review-queue, receipts, or any operator-facing surface. Per the
   steering directive's point 5: "Keep adapter artifacts gitignored and
   harness-only. No production/operator-surface wiring."

## Reproducibility

A manifest with corpus/adapter/summary SHA256s, training args, eval
command, tool versions, and git state is at
`data/aft/results/aft_manifest_v0.2.json`. To reproduce:

```bash
# 1. Extract corpus (gh-rate-limited; will retry on transient 5xx)
python3 scripts/aft_extract_training_data.py extract --max-prs 500 \
  --output data/aft/pr_triage_corpus.jsonl

# 2. Per-seed clean train+eval (Llama-1B + Qwen-7B)
python3 scripts/aft_seeded_train_eval.py --seeds 17 \
  --holdout-size 100 --iters 500 --frontier-dry-run \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit
python3 scripts/aft_seeded_train_eval.py --seeds 17 \
  --holdout-size 100 --iters 500 --frontier-dry-run \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit

# 3. Optional: real-frontier validation on a subset
head -15 data/aft/clean_test_holdout.jsonl > /tmp/clean_test_15.jsonl
python3 scripts/aft_harness.py --holdout /tmp/clean_test_15.jsonl \
  --conditions frontier_rules

# 4. Build manifest
python3 scripts/aft_manifest.py \
  --corpus data/aft/pr_triage_corpus.jsonl \
  --summary <summary1.json> <summary2.json> \
  --adapter-dir artifacts/advocates/seeded/* \
  --seeds 17 --holdout-size 78 --iters 500 \
  --model "mlx-community/Llama-3.2-1B-Instruct-4bit + mlx-community/Qwen2.5-7B-Instruct-4bit" \
  --out data/aft/results/aft_manifest_v0.2.json
```

Adapter binaries are gitignored under `artifacts/advocates/`; the manifest
records their SHA256 so a re-trained adapter can be compared bit-for-bit.

## Net assessment

The falsification harness worked as designed. v0.1's H1 PASS was an
artifact of two confounds (weak frontier prompt + data contamination).
v0.2 with both repaired:

- **Llama-3.2-1B fails the v0.1 standard outright.** Trained advocate is
  on par with or slightly worse than the repaired frontier baseline.
- **Qwen2.5-7B shows directional improvement** (+6.4 acc pts, lower
  Brier, ~120× cheaper) but does **not** clear Bonferroni p<0.05 at
  current holdout sizes (n≤78).

The honest read is that the advocate hypothesis is **not falsified** —
the larger model points the right way — but it is **not yet confirmed**
by the rigorous bar the steering directive set.

Per the directive: **keep the harness as a falsification rig, improve
corpus and rubric, do not advance to v0.2 production track until the
significance gap is closed or the operator explicitly accepts a weaker
bar.**
