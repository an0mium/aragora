# Model Family Bench (Spec v0.1)

**Status:** draft, scaffold only — no live provider runs in this PR
**Owner:** Armand
**Date:** 2026-05-24
**Related:** `docs/specs/MODEL_QUORUM_FAMILY_EXPANSION.md` (PR-A1),
`docs/specs/LOCAL_ADVOCATE_TRAINING_PIPELINE.md` (AFT pattern reused),
`scripts/aft_harness.py` (statistical primitives reused),
`bin/aft-advocate` (shim contract reused)

## Why this exists

PR-A1 (#7450) proposed adding 6+ reviewer families (Yi, GLM, MiniMax,
Hermes, Gemini-3.5-Flash, Grok-4.3) to the model-quorum recognizer.
The operator preapproval questions in PR-A1 include "**should all of
these be wired, or a subset?**" Answering that responsibly requires
evidence, not vibes. This spec defines the harness that produces the
evidence.

The harness follows the same falsification discipline that turned the
AFT advocate result from "H1 PASS (inflated)" into "v0.2 directional
(honest)": pre-registered hypotheses, head-grounded eval, McNemar
significance with Bonferroni correction, explicit caveats on small-n
results. The goal is not to anoint a family; it is to refuse to anoint
one without evidence.

## Pre-registered hypotheses (written before any data is collected)

**H0 (null) — no Pareto displacement.** No new family added in PR-A2
sits on the cost/quality Pareto frontier for *any* aragora-relevant
task; the four cores (Claude Opus, OpenAI GPT, Gemini Pro, Grok 4)
already cover the frontier.

**H1 — DeepSeek + Qwen cost dominance.** On Tier 0-2 PR review tasks,
at least one of {DeepSeek V3.2, Qwen3-Max} reaches within 5 accuracy
points of the best frontier model at ≤5% of the per-decision cost. If
this holds we accept those families for Tier 0-2 quorum counting.

**H2 — Gemini 3.5 Flash for agentic.** On the agentic-task subset
(multi-step PR analysis with tool-call rationale), Gemini 3.5 Flash
matches or exceeds Gemini 3.1 Pro at ≤25% the cost. If this holds we
wire Flash as a routing alias for agentic-typed work.

**H3 — Grok 4.3 for governance lens.** On the policy/governance task
subset (PRs that touch `aragora/cli/commands/review_queue.py`,
`docs/REVIEW_AUTHORITY_PRINCIPLES.md`, or `.github/workflows/`), Grok
4.3 produces dissent flags at a rate ≥1.5× the rate of the frontier
models, AND those flags are judged "useful" by manual review of a
held-back sample.

**H4 — GLM-4.6 differential reasoning.** On reasoning-heavy tasks
(debate critique, contradiction detection), GLM-4.6 produces
non-redundant findings (Jaccard distance from DeepSeek findings >0.4)
at ≥10% the cost of frontier reasoning models.

**H5 — Mistral demotion.** Mistral Large does NOT reach within 10
accuracy points of any frontier model on any task class, confirming
that demotion to "EU-jurisdiction routing only" (per the privacy
contract in `docs/REVIEW_AUTHORITY_PRINCIPLES.md`) is justified.

### Refutation rules

- If H1 fails: do not add DeepSeek / Qwen to Tier 0-2 quorum
  counting in PR-A2 even if the recognizer patches them in.
- If H2 fails: do not wire Gemini 3.5 Flash; stay on 3.1 Pro
  exclusively.
- If H3 fails: do not wire Grok 4.3 as a quorum reviewer; the
  CaseLaw/CorpFin benchmark wins do not translate to aragora's
  domain.
- If H4 fails: add GLM-4.6 to recognizer for completeness, but mark
  as advisory-only regardless of Tier (no incremental signal).
- If H5 fails (Mistral is competitive): retract the spec's
  demotion language; treat as equal-eligibility family.
- If H0 holds for all candidates: PR-A2 is **not justified** — the
  recognizer patch is fine to land (it's a correctness fix), but no
  new families should be added beyond the already-routed-and-paid-for
  set (OpenAI, Mistral, DeepSeek, Qwen, Kimi).

## Task categories

Three task classes, drawn from aragora's actual operational surfaces.

> **Scaffold-vs-target scope (PR-B, this PR).** The full preregistered
> corpus is **25 tasks** (10 PR-review + 5 debate-critique + 10
> inbox-triage). The corpus checked in by *this PR* (PR-B) is a
> **9-task scaffold** (3 + 3 + 3) that proves the plumbing
> — orchestrator, scoring, Pareto frontier, pairwise McNemar, small-n
> warnings, summary JSON shape — round-trips cleanly. The remaining
> 16 tasks land in a follow-on PR alongside the per-task-class
> per-family-class statistical summaries the H1-H5 hypotheses
> actually need. Treat the per-class task counts below as the
> **target** for the full corpus, not the count present in this PR's
> fixtures; the harness reports the actual loaded count
> (`summary["n_tasks"]`) and a `stub_only_run: true` flag at the top
> of every output JSON.

### 1. PR review (10 tasks — target; this PR ships 3 as scaffold)

Synthetic PRs of varying tier and shape:
- 4 Tier 0-1 docs/tests/scoped-code PRs
- 4 Tier 2 live-automation PRs
- 2 Tier 3 security/persistence-touch PRs (no Tier 4 — reviewer
  families don't count them anyway per the principles doc)

Each task is a `{title, diff_excerpt, tier_hint, ground_truth_decision}`
tuple. Ground truth is `merge_recommend | request_changes | defer`.
Scoring: agreement with ground truth (accuracy) + Brier on per-class
confidence + cost per decision.

**Privacy posture:** all synthetic. No real PR diffs from any private
repo. The task corpus is curated to exhibit the patterns we care
about (small/scoped vs sprawling, additive vs surgical, with-tests vs
without, etc.) without containing any operator-private content. Same
discipline as the AFT extractor.

### 2. Debate critique (5 tasks — target; this PR ships 3 as scaffold)

Synthetic debate prompts with one valid-looking-but-flawed argument
each (logical fallacy, hidden assumption, irrelevant evidence, etc.).
Reviewer must surface the flaw. Scoring: did the reviewer name the
specific flaw (per a curated answer key) vs produce a generic critique
vs miss it entirely.

### 3. Inbox triage proxy (10 tasks — target; this PR ships 3 as scaffold)

AFT-style low-information features (`{subject_token_count,
sender_domain_class, label_count, has_reviews}`) → ground-truth
action (`archive | reply | escalate`). Reuses AFT harness types
directly. No raw email content — same privacy boundary the AFT
extractor enforces.

Target total: **25 tasks** (preregistered). Small by ML-benchmark
standards even at the target size; appropriate for **plumbing +
Pareto-direction sensing** rather than statistical publication. The
harness prints the small-n caveat in its summary at any task count.

In this PR (PR-B) the checked-in fixture corpus is **9 tasks** (3 per
class) — a scaffold sized to validate the plumbing without
overstating benchmark readiness. The 16-task gap to the
preregistered target lands in a follow-on PR that also adds
per-task-class summaries.

## Scoring

For each `(task, family)` cell:
- `prediction`: family's chosen label
- `confidence`: family's stated confidence
- `latency_ms`: wall-clock
- `cost_usd`: per-call estimate × tokens (constants per family)
- `correct`: prediction == ground_truth

For each family aggregated across tasks:
- `accuracy`: mean(correct)
- `brier`: AFT `brier_score` (already implemented)
- `cost_usd_total`: sum
- `latency_ms_p50 / p95`

For each (family_A, family_B) pair:
- `mcnemar_p`: AFT `mcnemar_p` on paired binary correctness
- `p_bonferroni`: corrected for number of pairs

Pareto report: for each task class, list families on the frontier
(no other family strictly dominates on both accuracy and cost).

## Architecture (reuses AFT primitives)

```
scripts/aft_family_bench.py        (orchestrator; this PR)
  ├── reuses scripts/aft_harness.py: brier_score, accuracy, mcnemar_p
  ├── reuses bin/aft-advocate shim contract for each family
  ├── reads tests/fixtures/family_bench/*.jsonl (small synthetic corpus)
  └── emits data/family_bench/results/<ts>.json

bin/aft-bench-family                (shim per family; future PR)
  same JSONL stdin/stdout contract as bin/aft-advocate;
  one process per family, with stub/mlx/ollama/http backends
  → for this PR, only the stub backend is wired
```

## What's in this PR (PR-B, Tier 1)

- `docs/specs/MODEL_FAMILY_BENCH_SPEC.md` (this doc)
- `scripts/aft_family_bench.py` — orchestrator skeleton; runs against
  the stub backend; produces a real summary JSON; refuses to run
  against any non-stub backend without `--allow-live` AND a per-family
  cost-cap flag (a guardrail; not enforceable in this PR but the flag
  is wired).
- `scripts/aft_family_bench_scoring.py` — pure-function scoring
  helpers (pareto frontier identification, Jaccard distance on flag
  sets for H4, cost/quality table builder). Importable for tests.
- `tests/fixtures/family_bench/` — a **9-task scaffold** of the
  synthetic corpus (3 PR-review + 3 debate-critique + 3 inbox-triage,
  checked in; ~5KB total). The remaining 16 tasks needed to reach the
  preregistered 25-task corpus land in a follow-on PR — see
  "Scaffold-vs-target scope" under §Corpus above.
- `tests/scripts/test_aft_family_bench.py` — pure-function unit tests
  for the scoring helpers (~15-20 cases)

## What's NOT in this PR

- No live provider calls. The orchestrator runs only against the
  stub backend in this PR. `--allow-live` is reserved for a future
  PR and **fails closed** today — passing it exits non-zero with a
  clear error rather than silently running the stub. The flag and
  `--max-cost-usd` remain in the CLI surface as the forward-compat
  hooks the live-wiring PR will populate (credentials + cost-cap
  guardrail enforced there, not here).
- No `bin/aft-bench-family` shim with non-stub backends — that's a
  follow-on PR-C once H1-H5 indicate which families are worth wiring.
- No wiring into the quorum gate (that's PR-A2, Tier 4, blocked on
  PR-A1 preapproval).
- No PII / private-repo content in the corpus.

## Operator action needed before live runs

The harness will run end-to-end against the stub backend after this
PR lands (producing the report shape that operators can review). For
the actual H1-H5 evidence collection, the operator needs to:

1. Run the orchestrator in an environment where provider credentials
   are loaded (the same env that worked for the AFT v0.2 bench).
2. Pass `--allow-live` and `--max-cost-usd 5.00` (cost cap).
3. Cite the resulting summary JSON SHA in the answer to PR-A1's
   preapproval question #2 ("which families to wire").

That separates the **scaffolding-and-scoring-correctness-and-design
discipline** (which I can do unilaterally) from the **actual evidence
collection** (which uses operator credentials and money).

## Caveats baked into the spec

1. **n=25 is small.** Power analysis: detecting a 10-accuracy-point
   gap at p<0.05 paired-McNemar needs ~15 disagreement pairs. 25
   tasks gives us roughly that many on most pairs. The summary report
   will explicitly print "small-n; treat as directional, not
   conclusive" if any pair has <15 disagreement count.
2. **Synthetic corpus.** The PR-review tasks are synthetic patterns,
   not real PRs. Helps with privacy and reproducibility, hurts
   external validity. Acceptable for *family-direction* sensing;
   not acceptable for an academic publication.
3. **Cost constants are estimates.** Per-token prices are model-page
   list prices. Real costs vary with prompt length, output length, and
   provider markup. The Pareto comparison is meant to surface
   order-of-magnitude differences, not 5% deltas.
4. **No tool-use tasks.** The agentic-tool-use subset of H2 is
   abstracted to multi-step "rationale" outputs the family must emit.
   A real tool-use bench is out of scope.
5. **Single-shot scoring.** No multi-attempt, no chain-of-thought
   scaffolding. Sampling temperature held at 0.0 throughout. This
   intentionally penalizes "thinks better with reasoning tokens"
   models like R1 / Grok 4.3-thinking; their results should be read
   as a lower bound.

## Net assessment

If the harness lands and the operator runs it with credentials, we
get a defensible answer to PR-A1's question #2 ("which families to
wire") in 1-2 hours of operator-attended work. If H1-H4 hold, we
have positive evidence for the expansion; if they fail, we have
positive evidence for *not* expanding beyond the gate-correctness
fix; if H0 holds, we conclude the gate-fix is sufficient and no
new family wiring is warranted in PR-A2.

Either outcome is a substantive answer. The bench's job is to make
the wrong answer harder to reach by accident.
