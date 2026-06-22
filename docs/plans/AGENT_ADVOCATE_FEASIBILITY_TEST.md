# Agent Advocate Feasibility Test

## Purpose

The Advocate Feasibility Test (AFT) asks one narrow question before Aragora invests in
local fine-tuning or Tinker-backed training:

Can a local user-interest advocate improve PR-triage decisions over simple rules and a
frontier-model prompt baseline?

This is not a production admission gate. It is a benchmark-first experiment for the
local advocate layer proposed as part of Aragora's agent-native decision integrity
roadmap.

## Pre-Registered Hypotheses

H1: The local advocate arm beats the rules/classifier baseline by at least 25 percent
relative accuracy on held-out PR-decision examples.

H2: The local advocate arm matches or beats the frontier-prompted baseline by at least
10 percent relative accuracy, or by calibrated challenge/block usefulness when exact
labels are ambiguous.

H3: The local advocate arm keeps median latency under 500 ms for mock/local inference
and remains materially cheaper than frontier calls.

H4: The local advocate arm preserves more operator-specific caution signals than generic
frontier prompting, especially around Tier 3/4, human-risk settlement, dirty/conflicting
state, workflow/security paths, and active-owner overlap.

## Corpus Rules

The first domain is PR triage because the repo already has ground truth:

- merged PRs map to `accept`;
- closed unmerged PRs map to `block`;
- open drafts map to `ask_user`;
- dirty, blocked, or ambiguous open PRs map to `challenge`.

Corpus extraction must use sanitized PR metadata, review-queue packet facts, settlement
receipt summaries, and bounded queue-drain log hints. It must not include raw private
transcripts, API keys, secrets, unredacted comments, or arbitrary local logs.

The minimum AFT run is 200 labeled examples with a deterministic 50-example holdout, or
the nearest available 80/20 deterministic split when fewer examples exist during early
development.

## Arms

1. `rules`: deterministic rules/classifier baseline.
2. `frontier_prompt`: frontier-model prediction under an explicit operator-preference
   prompt, recorded through a fixture or later provider client.
3. `local_advocate`: local advocate adapter. The initial implementation is a mock local
   model with the same interface as a future LoRA/QLoRA-backed model.

Future AAVT runs may add:

4. direct frontier;
5. Aragora debate;
6. frontier plus advocate;
7. full Aragora debate plus advocate.

## Metrics

Each arm records:

- decision: `accept`, `challenge`, `ask_user`, or `block`;
- rationale;
- confidence;
- cited features;
- latency;
- estimated cost;
- exact-label accuracy;
- challenge/block usefulness for cases where the operator's label is conservative.

The initial pass/fail threshold is:

- local advocate accuracy beats rules by at least 25 percent relative, and
- local advocate matches or beats frontier prompt by at least 10 percent relative, or
- local advocate has equal accuracy but higher challenge/block usefulness on high-risk
  mistakes.

If those thresholds fail, do not train a real model. Improve the corpus, rubric, or
product framing first.

## Privacy Posture

The advocate layer represents durable user and organization interests. Its training data
is sensitive by default.

Default path:

- extract sanitized metadata;
- train and evaluate locally;
- avoid raw comments/transcripts;
- avoid cloud training for raw user-interest examples.

Optional cloud path:

- Tinker may be used only for redacted, synthetic, distilled, or non-private benchmark
  datasets behind an explicit privacy gate;
- raw user-interest data must not be sent to Tinker by default;
- any Tinker run must record dataset class, redaction method, and operator authorization.

Reference docs:

- [Tinker overview](https://tinker-docs.thinkingmachines.ai/)
- [Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
- [Tinker LoRA primer](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/)

## Implementation Hooks

- `scripts/extract_pr_decision_corpus.py` creates redacted JSONL examples.
- `scripts/aft_harness.py` evaluates the three initial arms and writes stable artifacts.
- `aragora/advocates/user_interest.py` defines the advocate interface.
- `aragora/verticals/models/finetuning.py` remains the local-first training reference
  for future LoRA/QLoRA work.

## Non-Goals

- No production admission gate changes.
- No merge policy changes.
- No queue-drain behavior changes.
- No real model fine-tuning until the corpus and harness demonstrate value.
