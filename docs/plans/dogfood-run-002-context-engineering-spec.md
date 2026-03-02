# Dogfood Run 002: Context Engineering Spec

## Objective
Turn the run-001 finding ("well-structured but wrong because context was incomplete") into a repeatable, measurable execution path that grounds debate plans in real codebase state before agents propose changes.

## Problem Statement
Run 001 proved debate mechanics worked, but plan quality failed on implementation reality:
- Proposed new files that duplicated existing modules.
- Used incorrect path conventions (`/src/...`) instead of actual repo layout.
- Lacked a reliable "what already exists" gate before "what to change."

## Hypothesis
If we inject a layered context pack before debate:
1. Deterministic repository inventory (file-grounded)
2. RLM-aware codebase map for large-context navigation
3. Multi-harness explorer synthesis (Claude Code, Codex, optional KiloCode Gemini/Grok)

Then plan output quality will improve on path correctness and non-duplication without sacrificing structure.

## Implementation (Completed in This Branch)
`aragora ask` now supports context-engineering flags:
- `--codebase-context`
- `--codebase-context-path`
- `--codebase-context-harnesses`
- `--codebase-context-kilocode`
- `--codebase-context-rlm`
- `--codebase-context-max-chars`
- `--codebase-context-timeout`
- `--codebase-context-out`
- `--codebase-context-exclude-tests`

New builder module:
- `aragora/debate/context_engineering.py`
- Produces a single "Codebase Reality Check" block with:
  - deterministic index + anchor table
  - optional harness explorer synthesis
  - anti-recreation guardrails

## Harness Order (Best-Effort, Cost-Aware)
1. Deterministic inventory first (always)
2. Claude + Codex explorers in parallel (if enabled)
3. KiloCode explorers (Gemini + Grok) in parallel when CLI is available
4. Inject merged context into debate input
5. Also enable built-in Arena codebase grounding (`enable_codebase_grounding=true`)

Rationale: deterministic map provides hard ground truth; harnesses provide semantic cross-checks and gap detection.

## Run Plan (Dogfood)
### Step 1: Build grounded context + run debate
```bash
source .env 2>/dev/null || export $(grep -v '^#' .env | xargs) 2>/dev/null

python -m aragora.cli.main ask \
  "Generate a prioritized implementation plan to improve Aragora dogfood quality without recreating existing components." \
  --local \
  --agents "openrouter|anthropic/claude-sonnet-4.6||proposer,openrouter|openai/gpt-4o||critic,openrouter|google/gemini-2.0-flash-001||synthesizer" \
  --rounds 2 \
  --consensus majority \
  --codebase-context \
  --codebase-context-path . \
  --codebase-context-harnesses \
  --codebase-context-kilocode \
  --codebase-context-rlm \
  --codebase-context-max-chars 80000 \
  --codebase-context-timeout 240 \
  --codebase-context-out /tmp/dogfood_run_002_context.md \
  --required-sections "Task List,Task Details,Dissenting Positions,Risks and Rollback"
```

### Step 2: Evaluate against grounding metrics
Use the same scorecard style as run 001, adding:
- `path_validity_rate`: fraction of proposed file paths that exist or are justified as truly new.
- `duplicate_recreation_ratio`: fraction of tasks proposing components that already exist.
- `existing_first_compliance`: tasks that include explicit "what already exists" references.

## Acceptance Criteria (Run 002)
1. `path_validity_rate >= 0.95`
2. `duplicate_recreation_ratio <= 0.10`
3. At least 5 ranked tasks with owner paths and measurable acceptance criteria
4. At least 2 dissenting positions preserved
5. No generic `/src/...` path proposals unless repo actually contains that layout

## Failure Modes / Fallback
- If harness explorers fail or time out, deterministic inventory still injects and debate proceeds.
- If KiloCode CLI is missing, run with Claude + Codex explorers only.
- If RLM full-corpus summary is too slow, rerun with `--codebase-context-rlm` removed.

## Artifacts
- Engineered context file: `/tmp/dogfood_run_002_context.md`
- Debate output: CLI stdout / chosen output artifact path
- Scorecard update target: `docs/plans/dogfood-run-001-spec.md` (append run-002 section) or a dedicated run-002 results file

## Run 002 Results (March 2, 2026)

### Executed Command Profile (Successful Scored Run)
- Mode: local debate
- Agents: OpenRouter Claude Sonnet 4.6 (proposer), OpenRouter GPT-4o (critic), OpenRouter Gemini 2.0 Flash (synthesizer)
- Rounds: 2
- Context engineering: enabled (`--codebase-context`, `--codebase-context-harnesses`)
- Output contract: `docs/plans/dogfood_output_contract_v1.json`
- Artifacts:
  - Output: `/tmp/dogfood_run_002_output.txt`
  - Engineered context: `/tmp/dogfood_run_002_context.md`

### Scorecard

| Criterion | Pass? | Notes |
|---|---|---|
| `path_validity_rate >= 0.95` | **PASS** | Raw: 13/14 existing (0.9286). Adjusted: 14/14 because the only missing path (`aragora/knowledge/mound/adapters/obsidian.py`) is explicitly identified as a confirmed gap to implement. |
| `duplicate_recreation_ratio <= 0.10` | **PASS** | No proposals to recreate existing core modules; plan targets existing files plus one justified missing adapter. |
| At least 5 ranked tasks | **FAIL** | Output contains only 3 unique ranked tasks (`P0/P1`, `P1`, `P2`). |
| Owner paths + measurable criteria | **PASS (partial quality)** | Owner path section present and mostly valid; gate criteria include quantitative thresholds. |
| At least 2 dissenting positions preserved | **FAIL** | Only one explicit dissent/pushback position is preserved. |
| No generic `/src/...` paths | **PASS** | Zero `/src/...` proposals detected. |

### Outcome
Run 002 shows clear improvement in codebase grounding and path realism, but still fails the two structural acceptance requirements:
1. minimum ranked-task count (needs >=5),
2. minimum dissent count (needs >=2).

### Practical Next Adjustment for Run 003
Add explicit deterministic output constraints to force:
1. exactly 5-7 ranked tasks,
2. a dedicated `Dissenting Positions` section with at least two numbered dissent entries.
