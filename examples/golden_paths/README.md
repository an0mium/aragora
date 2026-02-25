# Golden Path Examples

Five canonical examples for developer onboarding. Each is self-contained, runs without API keys, and completes in under 30 seconds.

## Quick Start

```bash
# Run any example directly
python examples/golden_paths/basic_debate/main.py
python examples/golden_paths/slack_triage/main.py
python examples/golden_paths/code_review/main.py
python examples/golden_paths/compliance_check/main.py
python examples/golden_paths/knowledge_query/main.py
```

## The Five Golden Paths

| # | Path | What it shows | Key APIs |
|---|------|---------------|----------|
| 1 | [basic_debate/](basic_debate/) | 3 agents debate a question, reach consensus, produce a receipt | `Arena`, `StyledMockAgent`, `DebateConfig` |
| 2 | [slack_triage/](slack_triage/) | Auto-triage Slack messages, spawn debates for critical alerts | `Arena`, priority classification, `InboxDebateTrigger` pattern |
| 3 | [code_review/](code_review/) | Adversarial code review with structured findings and severity ratings | `Arena`, custom proposals, SARIF/receipt export |
| 4 | [compliance_check/](compliance_check/) | EU AI Act risk classification and conformity report generation | `RiskClassifier`, `ConformityReportGenerator` |
| 5 | [knowledge_query/](knowledge_query/) | Query Knowledge Mound, inject context into debate for grounded decisions | `Arena(context=...)`, `KnowledgeMound` pattern |

## Progression

The examples build on each other conceptually:

1. **basic_debate** -- Learn the core debate loop (propose, critique, vote, receipt)
2. **slack_triage** -- Apply debates to real-world triggers (incoming messages)
3. **code_review** -- Customize agent proposals for domain-specific review
4. **compliance_check** -- Use Aragora's compliance framework (no debate needed)
5. **knowledge_query** -- Enrich debates with organizational knowledge

## Requirements

- Python 3.10+
- `aragora` package (or run from the repository root)
- No API keys, databases, or network access required

## Demo Artifacts

Pre-generated golden-path harness outputs are in the `demo/` directory:

```bash
python scripts/golden_paths.py --mode full --output-dir examples/golden_paths/demo
```

Artifacts: `ask_result.json`, `gauntlet_result.json`, `review_result.json`, `summary.json`
