# Local Advocate Training Pipeline (Draft Spec v0.1)

**Status:** draft, untested
**Owner:** Armand
**Date:** 2026-05-22
**Related:** `scripts/aft_extract_training_data.py`, `scripts/aft_harness.py`,
`docs/specs/ARAGORA_ROADMAP_REVISION_ADVOCATES.md`,
`memory/feedback_substrate_freeze_external_proof.md`

## Purpose

Define the smallest credible end-to-end pipeline for training and serving a
local, open-weight, operator-finetunable "advocate" model whose role is to
proxy a specific operator's revealed-preference decision policy on bounded,
routine tasks (initially: PR triage). The pipeline must run end-to-end on a
single Apple-Silicon workstation with no inbound network dependency at
inference time.

This is *not* a production roadmap. It is a falsification harness: if the
local advocate, after this pipeline, cannot match a frontier-with-rules
baseline within the bounds described in
`scripts/aft_harness.py::PRE_REGISTERED_HYPOTHESES`, the advocate-ensemble
hypothesis is falsified for this task and we do not expand it.

## Non-goals

- Replacing the frontier debate substrate. Advocates feed into it, never
  around it.
- A general-purpose finetuning service. This pipeline targets one decision
  type at a time.
- Cross-tenant or multi-operator training. Each advocate is one operator's
  policy artifact; merging them is out of scope for v0.1.
- Inference SLAs that beat frontier latency. We expect local advocates to be
  faster *and* cheaper but not necessarily smarter on novel inputs.

## Hardware envelope

The pipeline targets a workstation in the M3 Max / M4 Max / 64 GB+ class with
~200 GB free SSD. Concrete points in the envelope:

| Tier | Hardware | Base model | Training mode | Comment |
|---|---|---|---|---|
| A | MacBook Pro M3 Max, 64 GB | Qwen2.5-7B or Llama-3.1-8B | MLX-LoRA | Reference target. ~6-12 h per advocate. |
| B | MacBook Pro M4 Max, 128 GB | Qwen2.5-14B | MLX-LoRA, rank 16 | Stretch local target. |
| C | Tinker (Thinking Machines) | Qwen2.5-7B/14B, Llama-3.1-8B | Hosted LoRA-SFT | Used only to check whether local results are training-method-limited or data-limited. |

We do not target GPU servers in v0.1. Tier C is a control, not a production
path. If the local pipeline cannot reach within 2 accuracy points of Tier C
on the same data, the bottleneck is the training method, not the data, and
we re-evaluate before investing further.

## Stack

| Layer | Choice (v0.1) | Why |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct (sanity check) | Strong open weights, permissive licenses, MLX support. Qwen tends to follow structured-output instructions more reliably at small sizes. |
| Local finetuning | `mlx-lm lora` | First-party Apple-Silicon path; LoRA fits in 64 GB unified memory at 7B. |
| Portable finetuning | `peft` + `transformers` on CUDA / MPS | Used only when reproducing a Tinker run locally for parity checking. |
| Hosted finetuning | Tinker (Thinking Machines) | SFT/RL/DPO/distillation on the same base weights; used as upper-bound control. |
| Serving | `mlx-lm` (chat completion), `llama.cpp` (GGUF for laptops without MLX) | Both fit the `aft-advocate` shim contract. |
| Shim | `aft-advocate` script: JSONL on stdin → JSONL on stdout, one prediction per task | Decouples the harness from any specific runtime. |

The shim contract is intentionally tiny because we want to swap inference
backends without changing the harness.

## Data

The corpus comes from `scripts/aft_extract_training_data.py`:

- Source: `gh pr list` over the operator's own repo, across merged/closed/
  open states.
- Label space: `merged_fast | closed_no_merge | open_aged` (defined in the
  extractor's `classify_decision`).
- Features: low-information observable cues only — branch namespace, title
  tokens, label count, presence/absence of reviews, comment count bucket,
  diff size bucket, file count bucket, tier hint heuristic.
- **No diffs, no comment bodies, no PII** beyond what `gh` already exposes
  publicly. This is the privacy boundary: the advocate must succeed without
  reading the content of a PR. If it cannot, that is a finding, not a bug.
- Split: stratified 80/20 train/holdout with a fixed seed
  (`scripts/aft_extract_training_data.py split`).
- Storage: `data/aft/` (gitignored if it contains corpus rows; the schema
  itself is committed via the extractor source).

### Sample size sanity

If `gh pr list` returns fewer than ~250 examples per class after
stratification, the holdout is too small to detect a 2-point accuracy
difference at p<0.05 even with paired-McNemar. The pipeline must emit a
warning in that case, and we either:

1. Widen the corpus across multiple operator repos (each repo trains a
   *separate* advocate — never merge labels across operators), or
2. Switch to a different bounded task with more historical data
   (e.g. inbox triage, calendar acceptance), or
3. Declare the task data-limited and shelve until enough history accrues.

We do not paper over a small holdout with cross-validation tricks; the test
is whether the advocate works in real operating conditions, not whether we
can extract a statistic from a small sample.

## Training procedure (Tier A reference target)

The procedure is pinned, with deviations recorded in the experiment journal.

```text
1. Materialize the corpus
   python3 scripts/aft_extract_training_data.py extract --limit 2000
   python3 scripts/aft_extract_training_data.py split --seed 17

2. Convert JSONL to the chat-format training file
   (script TODO: scripts/aft_to_mlx_chat.py)
   Each row becomes:
     <|im_start|>system\n<operator policy preamble>\n<|im_end|>
     <|im_start|>user\n<title_redacted + rationale_seeds JSON>\n<|im_end|>
     <|im_start|>assistant\n{"label": "<label>", "confidence": <0..1>}<|im_end|>

3. Local LoRA finetune (MLX)
   mlx_lm.lora \
     --model Qwen/Qwen2.5-7B-Instruct \
     --train --data data/aft/pr_triage_train.mlx.jsonl \
     --adapter-path artifacts/advocates/pr-triage-qwen7b-v0.1 \
     --rank 8 --batch-size 4 --learning-rate 1e-4 \
     --iters 2000 --val-batches 25 --seed 17

4. Serve via the shim
   bin/aft-advocate \
     --backend mlx \
     --model Qwen/Qwen2.5-7B-Instruct \
     --adapter artifacts/advocates/pr-triage-qwen7b-v0.1 \
     --temperature 0.0 --max-tokens 32

5. Evaluate
   python3 scripts/aft_harness.py \
     --advocate-cmd bin/aft-advocate ... \
     --frontier-dry-run
```

Tier C (Tinker) substitutes step 3 with a hosted run on the same base
weights and downloads the adapter back. The shim and harness do not change.

### Why LoRA and not full finetuning

The decision policy we are encoding is small: a few hundred rules of thumb
about which PRs the operator merges fast versus parks. LoRA at rank 8 has
enough capacity for that, fits in unified memory at 7B, and re-trains in
hours instead of days. If LoRA underfits noticeably (training loss plateau
above a calibrated threshold), we raise rank to 16 *before* considering
full-parameter finetuning.

## Calibration and refusal

The advocate must emit a confidence in `[0, 1]` per decision. We use that
confidence in two ways:

1. **Brier scoring in the harness.** The
   `scripts/aft_harness.py::brier_score` function penalizes overconfident
   wrong answers more than underconfident wrong ones. A well-calibrated
   advocate that knows when it does not know is more useful than a confident
   one that is right slightly more often.
2. **Refusal threshold for downstream routing.** In production usage the
   advocate would not actually decide; it would *propose* with a confidence,
   and below a configurable threshold the request escalates to the frontier
   debate substrate. The threshold is a hyperparameter set per operator and
   per task.

We deliberately do not train the advocate to refuse. Training a small model
to abstain reliably is hard; instead the runtime layer makes the abstain
decision based on the emitted confidence.

## Adversarial review

Before any advocate is wired into a live path, the same heterogeneous
model-review quorum that gates other merge-authority self-modifications
reviews the trained adapter as a Tier 3 change (per
`docs/REVIEW_AUTHORITY_PRINCIPLES.md`):

- Independent model review of the *training corpus distribution* against
  potential label leakage and demographic skew.
- Independent model review of the *decision rules implied by the LoRA*
  (sampled by red-team prompts) to catch obvious failure modes.
- Operator settlement that explicitly accepts the residual risk.

Wiring the advocate into a path that *initiates* GitHub writes or external
side effects, rather than only proposing, is Tier 4 and requires human
preapproval before implementation and before merge.

## Failure modes we expect and watch for

| Failure | Detection | Response |
|---|---|---|
| Advocate matches baseline_random | McNemar p < 0.05 fails | Falsified; do not expand. |
| Advocate matches frontier_rules only because both are wrong about the same things | Per-class precision/recall in summary | Re-bin the label space (split `open_aged` into stalled vs in-flight) and re-run. |
| Holdout too small | Extractor warning | Widen corpus or shelve task. |
| Operator policy drifted since training data | Brier score climbs over time on production traffic | Re-extract corpus and retrain; advocate adapters are commodities. |
| Advocate hallucinates new labels | Harness sees `prediction not in CLASSES` | Treat as `open_aged` and flag the rate as a quality metric. |
| Local model crashes mid-run | Shim returns no JSON | Harness logs a stubbed prediction, marks condition `_stubbed`; results are not reported as empirical. |

## Privacy boundary, repeated

- The corpus extraction reads only what `gh` reads — public-by-default PR
  metadata plus repo-local diff sizes and file counts (counts, not content).
- The training data is stored under `data/aft/` and is *not* shipped with
  the repo; the schema is committed via the extractor source so it is
  reproducible.
- The trained LoRA adapter encodes the operator's revealed-preference
  policy. It is treated as PII-equivalent for that operator and is not
  uploaded to shared infrastructure without explicit operator action.
- The inference shim never logs full input rows to stdout. It logs only
  task IDs and predicted labels.

## Open questions (v0.1)

1. Should the advocate emit a structured *explanation* alongside its
   prediction, or only the label+confidence? Explanations are useful for
   audit but the model has to learn them, which inflates the corpus. We
   deferred this for v0.1 and reserve it for v0.2 if v0.1 falsifies.
2. Is there a meaningful distinction between an advocate (one operator,
   bounded task) and a *role* (privacy, preference, risk, memory,
   contrarian, delegation) as proposed in the May 21 codex thread? We
   suspect the advocate is the substrate and the roles are downstream
   compositions; v0.1 tests only the substrate.
3. Tinker-vs-MLX delta. If local training plateaus and hosted training
   does not, the bottleneck is method; if both plateau at the same point,
   the bottleneck is data or model size. v0.1 measures this as a side
   effect of running Tier C as a control.

## Out of scope for v0.1

- Multi-task advocates (one adapter for triage *and* inbox).
- DPO / RL on operator feedback. SFT first, only escalate if SFT is not
  enough.
- Distillation from frontier-on-policy traces. We considered this and held
  it back because it conflates "advocate matches frontier" with "advocate
  is a small frontier copy"; the experiment we want is whether the
  *operator's revealed policy* is learnable, not whether the frontier's
  is. Distillation is fair game for v0.2 if v0.1 motivates it.
- Cross-operator generalization. Each advocate is one operator.

## Next steps (in dependency order)

1. Land this spec and `scripts/aft_harness.py` as a draft PR. No live wiring.
2. Add `scripts/aft_to_mlx_chat.py` (corpus → MLX chat format).
3. Add `bin/aft-advocate` shim contract + a stubbed implementation that
   round-trips the harness.
4. Materialize the corpus and run baselines (`baseline_random`,
   `frontier_rules`) with the harness in `--frontier-dry-run` mode to
   sanity-check plumbing.
5. Train the first Tier A adapter, run the harness with all three
   conditions, and write the result up — pass or fail — in
   `docs/status/AFT_RESULT_v0.1.md`.
6. Only on a positive v0.1 result, scope v0.2 to one additional task
   (inbox triage looks most tractable per
   `memory/feedback_inbox_wedge_ui.md`).
