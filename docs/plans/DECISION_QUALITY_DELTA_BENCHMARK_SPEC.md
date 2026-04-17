# Decision Quality Delta Benchmark Spec

Last updated: 2026-04-17
Status: planning-truth artifact; benchmark contract only, not yet a published proof surface

Related:
- `docs/strategy/PROOF_AND_EVIDENCE.md`
- `docs/plans/PMF_DOGFOOD_EXECUTION_PLAN.md`
- `docs/status/ACTIVE_EXECUTION_ISSUES.md`
- `docs/status/NEXT_STEPS_CANONICAL.md`
- `docs/examples/decision-quality-delta-benchmark-prompt-pack.yaml`
- `docs/examples/decision-quality-delta-benchmark-sources.yaml`
- `docs/templates/DECISION_QUALITY_DELTA_BENCHMARK_REPORT_TEMPLATE.md`
- `docs/CLI_REFERENCE.md`
- `aragora/cli/review.py`
- `aragora/cli/commands/review_pr.py`
- `aragora/cli/gauntlet.py`

## Purpose

Turn the repo's hardest unproven claim into an auditable benchmark contract:

- one stable benchmark question
- one bounded corpus shape
- one fair baseline policy
- one adjudication method
- one go / no-go threshold set
- one automation-ready prompt pack and source manifest
- one compact proof-pack template for publishing the result truthfully

This artifact exists because the repo already proves process integrity better
than most single-model workflows, but does not yet prove the most valuable
commercial claim: quality lift over the strongest single-model alternative on
real bounded work.

## Benchmark Question

On the same bounded task, with the same input artifact and a fixed evaluation
contract:

1. does Aragora's multi-agent review path catch more real issues than the
   strongest single-model baseline
2. does it avoid materially worse false-positive blocker behavior
3. is any gain large enough to justify its cost and latency

If the answer is no, the commercial thesis should narrow to process integrity
and truthful stopping. If the answer is yes, the repo has earned the right to
invest in cleanup, packaging, and partner proof around that wedge.

## Why This Comes Before Refactor

This benchmark is existential. Architectural cleanup is conditional.

- if the multi-agent path does not beat the strongest single-model baseline,
  refactoring the debate orchestration is not the best use of time
- if the multi-agent path does beat the baseline, cleanup becomes justified
  investment rather than speculative polish

Do a small bench-readiness pass first. Do not do a broad orchestrator collapse
before the result exists.

## What This Benchmark Is

This benchmark is:

- a decision-quality delta benchmark
- focused first on code review, then on gauntlet-style spec review
- bounded to repo-native CLI paths that already exist on `main`
- designed to produce a publishable proof pack or a publishable falsification

This benchmark is not:

- the `TW-01..03` autonomous execution benchmark
- a generic agent leaderboard
- a design-partner proof substitute
- a substitute for inbox trust wedge proof
- a permission to widen `AGT-*`

## Benchmark Order

### Phase 0: Bench-readiness

Before any result counts:

1. fix deterministic environment issues that would make reruns ambiguous
2. freeze the corpus revision and benchmark git SHA
3. freeze the model roster and prompt contract for the entire run window
4. verify that the CLI paths named below complete end to end
5. define artifact paths for raw outputs, adjudication sheets, and summary JSON

This phase may change scripts or docs. It must not widen product scope.

### Phase 1: Review delta benchmark (primary gate)

This is the first and most important benchmark.

Why first:

- `aragora review` can express both the single-model and multi-agent legs
- code review is a direct fit for the claim in `docs/strategy/PROOF_AND_EVIDENCE.md`
- it isolates the debate value better than inbox workflows or full autonomous
  execution

### Phase 2: Gauntlet delta benchmark (secondary gate)

After the review benchmark is stable, run a smaller secondary benchmark on
`aragora gauntlet` for specs, architectures, policies, or contracts.

Why second:

- it measures the same epistemic thesis on non-code artifacts
- it broadens the evidence without mixing in inbox credentials or execution
  substrate failures

### Phase 3: Operational follow-through

If and only if Phase 1 is green:

- use `aragora review-pr` as the operational follow-up wedge on live PR heads
- feed the result into partner proof packs and before / after case studies
- use benchmark movement to justify selective refactor or packaging work

## Exact CLI Surfaces

Primary code-review benchmark surface:

```bash
python3 -m aragora.cli.main review --diff-file <diff.patch> --agents anthropic-api --output-format json --output-dir <dir>
python3 -m aragora.cli.main review --diff-file <diff.patch> --agents openai-api --output-format json --output-dir <dir>
python3 -m aragora.cli.main review --diff-file <diff.patch> --agents gemini --output-format json --output-dir <dir>
python3 -m aragora.cli.main review --diff-file <diff.patch> --agents anthropic-api,openai-api,gemini,grok --output-format json --output-dir <dir>
```

Optional PR-URL equivalent:

```bash
python3 -m aragora.cli.main review https://github.com/<owner>/<repo>/pull/<n> --agents anthropic-api --output-format json --output-dir <dir>
python3 -m aragora.cli.main review https://github.com/<owner>/<repo>/pull/<n> --agents anthropic-api,openai-api,gemini,grok --output-format json --output-dir <dir>
```

Secondary gauntlet benchmark surface:

```bash
python3 -m aragora.cli.main gauntlet <input.md> --input-type spec --profile quick --agents anthropic-api --format json --output <path>
python3 -m aragora.cli.main gauntlet <input.md> --input-type spec --profile quick --agents openai-api --format json --output <path>
python3 -m aragora.cli.main gauntlet <input.md> --input-type spec --profile quick --agents gemini --format json --output <path>
python3 -m aragora.cli.main gauntlet <input.md> --input-type spec --profile quick --agents anthropic-api,openai-api,gemini,grok --format json --output <path>
```

Operational follow-up path after benchmark proof:

```bash
python3 -m aragora.cli.main review-pr <pr_number> --reviewer claude --json
```

`review-pr` is not the primary benchmark harness because it does not express the
single-model and multi-agent legs symmetrically. It is the live workflow that
can benefit from benchmark proof later.

## Corpus Contract

### Review corpus (primary)

Start with `12` benchmark tasks.

Required mix:

- `4` tasks where a materially important security or correctness issue should be
  caught
- `4` tasks where maintainability, test coverage, or edge-case reliability
  issues should be caught
- `4` tasks that are intentionally clean or near-clean and should not trigger a
  blocking verdict

Every review task must record:

- `task_id`
- immutable source: PR URL or diff path plus commit SHA / head SHA
- repo and file scope
- task class: `security`, `correctness`, `reliability`, `maintainability`,
  `tests`, or `clean`
- expected finding inventory with severity and short answer key
- adjudication notes for borderline matches
- clean-task expectation when blocking should not happen

Prefer real PRs or diffs over synthetic samples whenever feasible.

### Gauntlet corpus (secondary)

Start with `8` benchmark tasks.

Required mix:

- `2` specs
- `2` architectures
- `2` policies or contracts
- `2` clean-or-low-risk inputs where blocker inflation would be visible

Every gauntlet task must record:

- `task_id`
- immutable input path and benchmark SHA
- input type and profile
- expected hazard inventory with severity and short answer key
- adjudication notes for partial matches
- clean-task expectation when blocking should not happen

### Corpus quality rules

- no task may be added after reviewing its outcome artifact
- no task may be dropped because a result is embarrassing
- corpus changes require a new corpus revision ID
- every result bundle must record the exact corpus revision

## Baseline Policy

The benchmark must compare Aragora to the strongest realistic single-model
alternative, not to a weak strawman.

Use three benchmark legs for each primary review task:

1. `single_claude` or current strongest frontier single-model candidate
2. `single_openai` or current strongest OpenAI single-model candidate
3. `single_gemini` or another strong independent single-model candidate
4. `aragora_team` with a fixed multi-agent roster

Primary comparator:

- `best_single` = the highest-scoring single-model leg under the same corpus,
  prompt contract, and adjudication rules

Secondary comparator:

- `single_default_operational` = the default single-reviewer path the team
  would actually use if Aragora did not exist

Why this rule:

- beating the average single model is not strong enough
- beating the worst single model is meaningless
- beating the best-single leg is the closest honest test of the thesis

## Fixed Roster and Prompt Rules

For any one benchmark window:

- freeze the exact single-model roster before the first run
- freeze the exact Aragora multi-agent roster before the first run
- keep prompt framing constant across all legs except the agent roster change
- record the roster, provider, and model identifiers in every artifact

If a provider is unavailable and the run proceeds in degraded mode:

- record `incomplete_roster=true`
- do not count that run toward published quality deltas

## Scoring Rubric

### Primary metrics

#### 1. Weighted catch-rate

For tasks with expected issues:

- `critical = 5`
- `high = 3`
- `medium = 1`
- `low = 0.5`

A finding counts as matched when the adjudicator confirms it identifies the
same underlying issue, file or artifact region, and materially similar concern.

Weighted catch-rate:

`sum(weights of matched expected findings) / sum(weights of all expected findings)`

#### 2. False-positive blocker rate

For clean or near-clean tasks:

- blocking verdict on a clean task counts as a false-positive blocker
- unsupported critical or high-severity claim also counts as a false-positive
  blocker

#### 3. Clean-task pass rate

The percent of clean tasks where the tool returns a correct non-blocking or
low-severity result.

### Secondary metrics

- useful unique findings per task
- duplicate / spam finding rate
- adjudicator usefulness score on a `1-5` scale
- wall-clock latency
- token usage when available
- total cost per task

### Metric publication rule

Publish primary metrics even if they are bad. Secondary metrics explain why.

## Blind Adjudication Contract

Every benchmark leg must be adjudicated blind to model identity.

Required process:

1. strip agent and provider names from outputs before adjudication
2. have `2` adjudicators score each task independently against the answer key
3. mark each candidate finding as one of:
   - `match`
   - `partial_match`
   - `duplicate`
   - `unsupported`
   - `false_positive_blocker`
4. resolve disagreements through a third adjudicator or explicit tie-break pass
5. publish raw agreement and the resolved score sheet

No benchmark claim counts without adjudication artifacts.

## Cost and Latency Capture

Every run bundle must record at minimum:

- `benchmark_id`
- `task_id`
- `corpus_revision`
- `git_sha`
- `run_leg` (`single_claude`, `single_openai`, `single_gemini`, `aragora_team`)
- exact CLI command
- started / completed timestamps
- exit status
- wall-clock seconds
- token counts if available
- estimated USD cost if available
- raw output path
- normalized output path
- adjudication sheet path

Suggested artifact root:

`.aragora/benchmarks/decision-quality-delta/<benchmark_id>/`

Suggested subdirectories:

- `raw/`
- `normalized/`
- `adjudication/`
- `reports/`

## Go / No-Go Thresholds

### Go

Aragora earns a `go` on Phase 1 only if all of the following are true versus
`best_single`:

1. weighted catch-rate delta is `>= +15 percentage points`
2. false-positive blocker rate is not worse by more than `10 percentage points`
3. clean-task pass rate is not worse by more than `5 percentage points`
4. artifact completeness is `100%`
5. blind adjudication coverage is `100%`

Latency and cost do not block `go` on their own, but they must be published.

### Conditional go

Aragora earns `conditional_go` when:

- weighted catch-rate delta is positive and materially useful
- but cost or latency is high enough that the wedge must stay narrow

This permits narrow operational use and optimization work. It does not justify
broad platform claims.

### No-go

Aragora earns `no_go` when any of the following is true:

- weighted catch-rate delta versus `best_single` is `<= 0`
- false-positive blocker behavior is materially worse
- clean-task pass rate regresses materially
- adjudication or artifact completeness is missing

`no_go` means:

- do not widen the commercial claim
- do not justify broad refactor on benchmark optimism
- narrow the story to governed process integrity until the delta improves

## Proof-Pack Contract

Every benchmark window should leave behind one compact report containing:

- corpus revision and git SHA
- exact rosters and commands
- weighted catch-rate by leg
- false-positive blocker rate by leg
- clean-task pass rate by leg
- latency and cost summaries
- adjudication method and agreement rate
- `go`, `conditional_go`, or `no_go`
- three short interpretations:
  - what Aragora did better
  - what it still did worse
  - what should happen next

Use `docs/templates/DECISION_QUALITY_DELTA_BENCHMARK_REPORT_TEMPLATE.md` as
the canonical report shape.

## Automation-Ready Artifacts

Use these machine-readable artifacts for bounded long-running execution:

- prompt pack: `docs/examples/decision-quality-delta-benchmark-prompt-pack.yaml`
- source manifest: `docs/examples/decision-quality-delta-benchmark-sources.yaml`

Suggested tranche workflow:

```bash
python3 -m aragora.cli.main swarm tranche plan \
  --from-prompts docs/examples/decision-quality-delta-benchmark-prompt-pack.yaml \
  --output .aragora/tranches/decision-quality-delta-benchmark/tranche.yaml \
  --json

python3 -m aragora.cli.main swarm tranche inspect \
  --manifest .aragora/tranches/decision-quality-delta-benchmark/tranche.yaml \
  --json

python3 -m aragora.cli.main swarm tranche design-review \
  --manifest .aragora/tranches/decision-quality-delta-benchmark/tranche.yaml \
  --json
```

Operational rule:

- use tranche automation for corpus curation, harness, adjudication, and report
  assembly
- stop rather than bluff when a lane needs live credentials or model access that
  is unavailable
- do not present an automated pre-score as final truth without blind
  adjudication

## Stop Conditions

Stop and replan if:

- the benchmark drifts into a generic agent leaderboard
- the corpus is changed after seeing model outputs
- the single-model baseline is weakened for convenience
- the benchmark mixes debate quality with unrelated substrate failures
- the benchmark publishes a quality-lift claim without adjudication
- the benchmark expands into inbox, design-partner, or full execution proof
  before Phase 1 is complete
