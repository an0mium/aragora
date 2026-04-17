# Decision Quality Delta Benchmark Report

Use this template for the compact artifact that proves or falsifies the claim
that Aragora's multi-agent review path beats the strongest realistic
single-model baseline on bounded review tasks.

## 1. Benchmark Window

- Benchmark ID:
- Window start:
- Window end:
- Corpus revision:
- Git SHA:
- Operators / adjudicators:
- Related issues or tranche IDs:

## 2. Benchmark Statement

Describe the exact question answered in one sentence:

`same task -> same corpus -> single-model baselines vs Aragora-team -> blind adjudication -> delta decision`

Primary benchmark surfaces:

- `python3 -m aragora.cli.main review`
- `python3 -m aragora.cli.main gauntlet`

Explicitly excluded from this proof window:

- inbox wedge proof
- design-partner proof
- full autonomous execution benchmark claims

## 3. Roster and Prompt Contract

| Leg | Agents / models | Prompt contract notes |
|---|---|---|
| `single_claude` |  |  |
| `single_openai` |  |  |
| `single_gemini` |  |  |
| `aragora_team` |  |  |

Record any degraded or incomplete roster runs explicitly:

- `incomplete_roster=true/false`
- If true, why:

## 4. Corpus Summary

### Review corpus

| Task class | Count | Notes |
|---|---|---|
| Security / correctness |  |  |
| Reliability / maintainability / tests |  |  |
| Clean / near-clean |  |  |

### Gauntlet corpus

| Input class | Count | Notes |
|---|---|---|
| Spec |  |  |
| Architecture |  |  |
| Policy / contract |  |  |
| Clean / low-risk |  |  |

## 5. Exact Commands

List the exact benchmark commands used.

```bash
python3 -m aragora.cli.main review --diff-file <diff.patch> --agents anthropic-api --output-format json --output-dir <dir>
python3 -m aragora.cli.main review --diff-file <diff.patch> --agents anthropic-api,openai-api,gemini,grok --output-format json --output-dir <dir>
python3 -m aragora.cli.main gauntlet <input.md> --input-type spec --profile quick --agents anthropic-api --format json --output <path>
python3 -m aragora.cli.main gauntlet <input.md> --input-type spec --profile quick --agents anthropic-api,openai-api,gemini,grok --format json --output <path>
```

## 6. Primary Scoreboard

| Metric | Target | `best_single` | `aragora_team` | Delta | Pass/Fail |
|---|---|---|---|---|---|
| Weighted catch-rate | `>= +15pp` vs `best_single` |  |  |  |  |
| False-positive blocker rate | `<= +10pp` regression |  |  |  |  |
| Clean-task pass rate | `<= +5pp` regression |  |  |  |  |
| Artifact completeness | `100%` |  |  |  |  |
| Blind adjudication coverage | `100%` |  |  |  |  |

## 7. Secondary Metrics

| Metric | `best_single` | `aragora_team` | Notes |
|---|---|---|---|
| Useful unique findings / task |  |  |  |
| Duplicate finding rate |  |  |  |
| Adjudicator usefulness score |  |  |  |
| p50 latency (s) |  |  |  |
| p95 latency (s) |  |  |  |
| Mean cost / task (USD) |  |  |  |

## 8. Adjudication Method

- Identity stripping method:
- Number of adjudicators:
- Disagreement resolution path:
- Agreement rate:
- Answer-key revision:
- Adjudication sheet path:

## 9. Representative Task Results

### Review tasks

| Task ID | `best_single` outcome | `aragora_team` outcome | Key difference |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

### Clean-task behavior

| Task ID | `best_single` blocker? | `aragora_team` blocker? | Correct? |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |

### Gauntlet tasks

| Task ID | `best_single` outcome | `aragora_team` outcome | Key difference |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |

## 10. Interpretation

### What Aragora did better

-

### What Aragora did worse

-

### What remains ambiguous

-

## 11. Gate Decision

Choose exactly one:

- `go`
- `conditional_go`
- `no_go`

Decision rationale:

## 12. Next Bounded Actions

- Action 1:
- Action 2:
- Action 3:
