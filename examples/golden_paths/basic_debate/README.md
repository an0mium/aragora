# Golden Path 1: Basic Multi-Agent Debate

Three AI agents with different perspectives debate a technical decision, then produce a consensus verdict with a cryptographic decision receipt.

## What it demonstrates

- Creating agents with distinct debate styles (supportive, critical, balanced)
- Configuring a debate with round count and consensus method
- Running the Arena orchestrator (propose -> critique -> vote)
- Reading proposals, critiques, votes, and dissenting views from the result
- Generating and inspecting a decision receipt

## Run it

```bash
python examples/golden_paths/basic_debate/main.py
```

No API keys required. Uses `StyledMockAgent` for fully offline execution.

## Expected output

```
================================================================
  Aragora Golden Path: Basic Multi-Agent Debate
================================================================

Question: Should we migrate our monolithic API to microservices? ...
Agents:   architect, security-reviewer, tech-lead
Rounds:   2

--- Debate Result ---
Status:    consensus_reached
Rounds:    1
Consensus: Reached
Confidence: 73%
Duration:  0.01s

--- Proposals ---
  [architect] After careful analysis, I strongly endorse this approach. ...
  [security-reviewer] I have significant concerns about this approach. ...
  [tech-lead] There are valid arguments on both sides. ...

--- Critiques (4) ---
  [architect -> security-reviewer] Could benefit from more quantitative evidence; ...
  [architect -> tech-lead] Could benefit from more quantitative evidence; ...
  [security-reviewer -> architect] Missing cost analysis for migration and ongoing operations; ...
  [security-reviewer -> tech-lead] Missing cost analysis for migration and ongoing operations; ...

--- Votes ---
  [architect] voted for security-reviewer (confidence: 85%) -- ...
  [security-reviewer] voted for tech-lead (confidence: 60%) -- ...
  [tech-lead] voted for architect (confidence: 70%) -- ...

--- Decision Receipt ---
# Decision Receipt DR-20260224-...

**Question:** Should we migrate our monolithic API to microservices? ...
**Verdict:** Approved With Conditions
**Confidence:** 73%
...
```

(Exact output varies due to random style-template selection.)

## Key APIs used

| Import | Purpose |
|--------|---------|
| `aragora_debate.Arena` | Debate orchestrator |
| `aragora_debate.DebateConfig` | Rounds, consensus method, early stopping |
| `aragora_debate.StyledMockAgent` | Offline agents with realistic debate styles |
| `result.receipt.to_markdown()` | Human-readable decision receipt |

## Next steps

- Try `StyledMockAgent("devil", style="contrarian")` for a devil's advocate
- Set `enable_trickster=True` in `DebateConfig` to detect hollow consensus
- Swap mock agents for real LLM agents with `create_agent("anthropic")`
