# Local Advocate Ensemble Roadmap

## Positioning

Local advocates are a user-interest mediation layer between generic frontier reasoners
and Aragora's debate, receipt, and compliance systems.

They are not replacements for debate. They are witnesses that represent durable local
preferences, privacy boundaries, risk budgets, institutional memory, and contrarian
regret checks before a frontier model or automated agent acts.

## Why This Belongs In Aragora

Aragora already models multi-agent disagreement, evidence receipts, and risk-tiered
settlement. A local advocate layer adds the missing user-specific context:

- privacy advocate: decides what context should stay local;
- preference advocate: checks whether an answer matches durable operator priorities;
- risk advocate: challenges actions above the user's risk budget;
- memory advocate: catches contradictions against known history;
- contrarian advocate: names plausible future regret;
- delegation advocate: recommends automate, escalate, pause, or ask.

The first implementation should be advisory only. Advocates can challenge or block a
proposed action in a benchmark, but they do not become production rulers.

## Architecture

```text
Frontier model proposal
  -> local user-interest advocates
  -> Aragora debate / dissent preservation
  -> decision receipt
  -> vertical compliance artifacts
```

The initial interface is:

```text
input:
  task_type
  artifact_summary
  proposed_action
  context_features

output:
  decision: accept | challenge | ask_user | block
  confidence
  rationale
  cited_features
```

## Training Path

Stage 0: deterministic rules and mock-local advocate baselines.

Stage 1: build a redacted PR-decision corpus from repo history, review packets,
settlement receipts, and queue-drain logs.

Stage 2: run AFT. Do not train real models unless the advocate arm shows measurable
value against rules and frontier prompting.

Stage 3: local-first LoRA/QLoRA using the existing fine-tuning support in
`aragora/verticals/models/finetuning.py`.

Stage 4: optional Tinker experiments only for redacted, synthetic, distilled, or
non-private benchmark datasets. Raw user-interest data stays local by default.

## Tinker Boundary

Tinker can accelerate LoRA experimentation for open-weight models, but it is cloud
training. That conflicts with the strongest privacy version of local advocates unless
the dataset is redacted or synthetic.

Allowed by default:

- synthetic operator-policy tasks;
- public PR metadata;
- redacted benchmark examples;
- non-private organizational evals.

Not allowed by default:

- raw private transcripts;
- raw user memory;
- secrets or credentials;
- unredacted customer decisions;
- private preference traces without explicit operator authorization.

## Success Criteria

The local advocate primitive graduates from experiment to roadmap pillar only if:

- it improves held-out PR-triage decisions over rules;
- it matches or beats frontier-prompting on at least one calibrated metric;
- it preserves privacy-sensitive user-interest data locally by default;
- it composes cleanly with debate and receipts;
- it does not become another unfalsified governance layer.

If AFT fails, keep the interface as a test seam and defer model training.
