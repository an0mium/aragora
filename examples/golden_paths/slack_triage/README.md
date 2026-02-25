# Golden Path 2: Slack Inbox Auto-Triage

Simulates how Aragora triages incoming Slack messages. Critical alerts trigger a multi-agent debate that produces a triage recommendation and audit receipt.

## What it demonstrates

- Classifying message priority from content and channel
- Triggering a debate only for critical-priority messages
- Using custom agent proposals tailored to incident response roles
- Generating a triage recommendation with consensus and receipt
- Rate limiting and cooldown (explained, not simulated)

## Run it

```bash
python examples/golden_paths/slack_triage/main.py
```

No API keys or Slack workspace required.

## Expected output

```
================================================================
  Aragora Golden Path: Slack Inbox Auto-Triage
================================================================

[#incidents] CRITICAL: Database connection pool exhausted
  From: monitoring-bot
  Priority: CRITICAL
  -> Triggering triage debate...
  -> Consensus: Yes (73% confidence)
  -> Verdict: approved_with_conditions
  -> Receipt: DR-20260224-...
  -> Agents: incident-responder, risk-analyst, triage-coordinator
  -> Recommendation: Triage recommendation for: CRITICAL: Database...

[#team-updates] Sprint planning moved to Thursday
  From: project-manager
  Priority: LOW
  -> Skipped (below critical threshold)

[#security] URGENT: Exposed API key detected in public repository
  From: security-scanner
  Priority: CRITICAL
  -> Triggering triage debate...
  -> Consensus: Yes (73% confidence)
  -> Verdict: approved_with_conditions
  -> Receipt: DR-20260224-...
  ...
```

## How it maps to production

| Demo step | Production equivalent |
|-----------|----------------------|
| `classify_priority()` | `InboxDebateTrigger.should_trigger()` with rate limits |
| `run_triage_debate()` | `InboxDebateTrigger.trigger_debate()` via playground API |
| Mock Slack payloads | Real Slack webhook via `aragora.connectors.slack` |
| Console output | Slack thread reply + Knowledge Mound persistence |

## Key APIs used

| Import | Purpose |
|--------|---------|
| `aragora_debate.Arena` | Debate orchestrator |
| `aragora_debate.StyledMockAgent` | Role-specific triage agents |
| `result.receipt` | Audit trail for the triage decision |

## Next steps

- Connect to a real Slack workspace with `aragora.connectors.slack.SlackIntegration`
- Use `InboxDebateTrigger` from `aragora.server.handlers.inbox.auto_debate` for rate-limited production triage
- Store triage results in Knowledge Mound for cross-incident learning
