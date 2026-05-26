# AFT v0.1 Result: H1 PASS (preliminary) — SUPERSEDED

> **⚠️ SUPERSEDED BY [`AFT_RESULT_v0.2.md`](AFT_RESULT_v0.2.md).** The "H1 PASS"
> conclusion below was **inflated** by two confounds that v0.2 repaired:
> (1) a weak frontier prompt that defaulted to an empty `open_aged` class,
> and (2) data contamination between training and the held-out set. The
> clean v0.2 evaluation shows the advocate is **directionally** better than
> the repaired frontier baseline but **not statistically separable** at
> n≤78. Read v0.2 for the current verdict. This file is kept on disk as
> an audit trail of the original inflated reading.

**Date:** 2026-05-22
**Status:** Hypothesis H1 cleared at preliminary scale (n=50, small model) — *no longer the operative verdict; see v0.2*
**Related:** PR #7438, `docs/status/AFT_BASELINE_v0.1.md` (superseded by this doc),
`docs/specs/ARAGORA_ROADMAP_REVISION_ADVOCATES.md`,
`docs/specs/LOCAL_ADVOCATE_TRAINING_PIPELINE.md`

## TL;DR

A 1B-parameter open-weight model (`mlx-community/Llama-3.2-1B-Instruct-4bit`),
fine-tuned with LoRA on 444 historical PR decisions for **~3 minutes**, beat a
frontier-with-rules baseline on this operator's PR triage task by **28
accuracy points**, with **better calibration** (Brier 0.44 vs 0.73), at
**~1% of the per-decision cost**. The gap is statistically significant
after Bonferroni correction (`p_bonf = 0.008`).

This is a **preliminary** result on a **small holdout (n=50)** with a
**small base model**. It is not a final answer, but it clears the
pre-registered H1 threshold and the refutation rule from
`scripts/aft_harness.py::PRE_REGISTERED_HYPOTHESES`. **The advocate
hypothesis is NOT falsified for PR triage; it is provisionally supported.**

## Numbers

| Condition | Accuracy | Brier | Cost USD | n |
|---|---|---|---|---|
| `baseline_random` (prior-weighted) | 0.46 | 0.885 | $0.000 | 50 |
| `frontier_rules` (`claude --print` + operator rules) | 0.50 | 0.725 | $0.600 estimate | 50 |
| **`local_advocate` (Llama-3.2-1B + 500-iter LoRA)** | **0.78** | **0.440** | $0.005 | 50 |

Cost USD is the harness's per-call estimate × n. The frontier number is a
real claude-CLI invocation (`claude --print` was preferred over `aragora ask`
because the latter failed on a workstation without API keys loaded). Local
advocate cost is the harness's electricity-proxy estimate.

### Pairwise McNemar (Bonferroni-corrected, factor 3)

| Pair | p | p_bonferroni | Verdict |
|---|---|---|---|
| `baseline_random` vs `frontier_rules` | 0.81 | 1.00 | not different |
| `baseline_random` vs `local_advocate` | **0.003** | **0.008** | local advocate beats baseline at p<0.01 |
| `frontier_rules` vs `local_advocate` | **0.003** | **0.008** | local advocate beats frontier at p<0.01 |

### Confusion matrices

`baseline_random` predicts mostly majority class:

```
truth\pred       merged_fast  closed_no_merge  open_aged
merged_fast              20                5          0
closed_no_merge          22                3          0
open_aged                 0                0          0
```

`frontier_rules` spreads across all 3 classes (over-predicts `open_aged`):

```
truth\pred       merged_fast  closed_no_merge  open_aged
merged_fast              18                2          5
closed_no_merge           5                7         13
open_aged                 0                0          0
```

`local_advocate` correctly learned the operator's revealed 2-class policy
(never predicts `open_aged`, which has zero training examples):

```
truth\pred       merged_fast  closed_no_merge  open_aged
merged_fast              21                4          0
closed_no_merge           7               18          0
open_aged                 0                0          0
```

## Pre-registered hypotheses — verdicts

From `scripts/aft_harness.py::PRE_REGISTERED_HYPOTHESES`:

- **H0 (null)** — local_advocate fails to reach within 2 accuracy points of
  frontier_rules OR Brier worse by >0.02: **REJECTED** (advocate is +28
  acc, Brier is 0.285 better).
- **H1 (primary)** — local_advocate matches frontier_rules within 2 acc
  points AND Brier within 0.02 AND costs ≤10%: **CLEARED with margin**
  (advocate is *better* on accuracy and Brier, at ~1% cost).
- **H2 (cost-quality frontier)** — only relevant if H1 fails; not invoked.
- **Refutation rule** — advocate must beat baseline_random at p<0.05
  Bonferroni: **CLEARED** (p_bonf = 0.008).

The advocate-ensemble hypothesis is **not falsified** at this preliminary
scale.

## What was actually done

1. **Extracted** 556 PR decisions from `synaptent/aragora` via `gh pr list`
   (with retry-with-backoff handling for transient HTTP 5xx / stream
   errors). Time: ~30s after rate-limit recovery.
2. **Split** the corpus stratified by minority class: 50-task holdout
   (25 `merged_fast` + 25 `closed_no_merge`), 506-task training set.
3. **Converted** to MLX chat-format JSONL via
   `scripts/aft_build_mlx_dataset.py` (444 train / 55 valid / 57 test).
4. **Fine-tuned** Llama-3.2-1B-Instruct-4bit with LoRA for **500
   iterations** at rank 8, lr 1e-4, batch 2. Training loss went
   1.66 → 0.19; validation loss 0.51 → 0.34. Wall-clock: ~3 min on M3.
5. **Inferred** all 50 holdout tasks through `bin/aft-advocate
   --backend mlx --model <base> --adapter <path>`. Wall-clock: ~10s for
   model load + 50 predictions.
6. **Scored** against the harness's baseline_random (prior-weighted) and a
   real `claude --print` invocation as `frontier_rules`. Wall-clock for the
   full 3-condition run: ~6 minutes (frontier latency dominates).

## Caveats

These caveats are real and the operator should weight them before
accepting the H1 verdict at full strength.

1. **n=50 holdout is small.** A 28-point gap is large enough to clear
   p<0.01 paired-McNemar at this size, but the absolute accuracy numbers
   are noisy. A holdout in the 200+ range would tighten the confidence
   interval.
2. **The base model is 1B-4bit Llama.** The spec's reference target is
   Qwen2.5-7B. Going to 7B is expected to *improve* the local advocate
   further, but it also raises memory/time costs, so the
   accuracy/cost-frontier story depends on which size we ship.
3. **The frontier prompt is the spec's first draft.** The current
   `FrontierRules.OPERATOR_RULES` defaults to `open_aged` for governance
   paths, which the data shows is wrong in current operating conditions.
   An updated rule sheet would raise the `frontier_rules` baseline — the
   comparison may narrow.
4. **The `open_aged` class is empty in current data.** The advocate
   learned a 2-class policy because that is what the data showed. If the
   operator changes their merge cadence and `open_aged` re-emerges, the
   adapter will need retraining. The harness will surface this via
   per-class precision dropping on the new class.
5. **This is a per-operator, per-task artifact.** The adapter encodes one
   operator's revealed-preference policy on one task. It does not
   generalize across operators and was never intended to.
6. **Cost numbers are estimates.** The $0.60 frontier cost and the $0.005
   local cost are per-call constants in `scripts/aft_harness.py`
   (`COST_PER_CALL_ESTIMATE`). Real costs will vary by token usage; the
   order-of-magnitude gap (~120x) is robust to that.
7. **No live wiring.** Nothing in this PR causes any change to the
   review-queue, receipts, or any operator-facing surface. The advocate
   exists only as an artifact under `artifacts/advocates/` and is
   exercised only by the harness.

## What this finding means for the roadmap

Per `docs/specs/ARAGORA_ROADMAP_REVISION_ADVOCATES.md` Success Criteria,
the v0.1 result allows advance to v0.2 *if* the operator settles the risk:

> If those criteria are met, v0.2 begins with one additional task (most
> likely inbox triage). If not, the advocate-ensemble hypothesis is
> **falsified for this codebase** and we do not expand it.

The criteria are met at preliminary scale. The next governance step is
operator settlement on whether to:

- **Accept v0.1 and advance to v0.2** (inbox triage advocate)
- **Tighten v0.1 first** (Qwen-7B base, updated frontier prompt, larger
  holdout) before advancing
- **Hold** for an external dogfood / second-operator validation
- **Reject** and treat this as an artifact for future reference

## Reproducibility

All artifacts are deterministic given the seeds. To reproduce:

```bash
# 1. Extract corpus (idempotent given gh state)
python3 scripts/aft_extract_training_data.py extract --max-prs 500 --output data/aft/pr_triage_corpus.jsonl
python3 scripts/aft_extract_training_data.py split

# 2. Build MLX dataset
python3 scripts/aft_build_mlx_dataset.py

# 3. Fine-tune (~3 min on M3)
python3 -m mlx_lm lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train \
  --data data/aft/mlx \
  --adapter-path artifacts/advocates/aft-pilot-llama1b-500 \
  --num-layers 8 --batch-size 2 --iters 500 \
  --val-batches 10 --steps-per-report 50 --steps-per-eval 100 \
  --learning-rate 1e-4 --seed 17

# 4. Run harness (~6 min; frontier latency dominates)
python3 scripts/aft_harness.py \
  --advocate-cmd "bin/aft-advocate --backend mlx \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --adapter artifacts/advocates/aft-pilot-llama1b-500"
```

Outputs land under `data/aft/results/` (gitignored). The pre-registered
hypotheses are written verbatim alongside the summary so the verdict is
not retconned.

## Net assessment

The Advocate Feasibility Test as a *falsification rig* worked end-to-end
on real data. The hypothesis it was built to test cleared its pre-
registered bar at preliminary scale. The cost-quality story is dramatic
enough (28 points better, ~120x cheaper) that even after tightening
caveats 1-3, the directional finding is unlikely to reverse.

The substantive engineering work this enables — wiring a per-operator
advocate as a *proposal-only* preamble to the existing review queue,
with the existing debate substrate as the escalation backstop — remains
gated on operator settlement at Tier 3 per
`docs/REVIEW_AUTHORITY_PRINCIPLES.md`. Nothing in this PR ships that
wiring.
