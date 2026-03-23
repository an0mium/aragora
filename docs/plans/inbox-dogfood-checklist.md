# Inbox Trust Wedge — Dogfood Checklist

Run the narrow proving path on founder's real Gmail inbox.

## Pre-flight (one-time setup, ~10 min)

- [ ] **Gmail OAuth**: `python scripts/gmail_oauth_setup.py`
  - Creates `~/.aragora/gmail_credentials.json`
  - Grants `gmail.readonly` + `gmail.modify` scopes
  - Test: `python -c "from aragora.connectors.email.gmail_sync import GmailSync; print('OK')"`

- [ ] **API keys**: Ensure at least 2 provider keys in `.env`
  - Required: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
  - Recommended: both + `OPENROUTER_API_KEY` for fallback
  - Test: `aragora doctor` should show green for providers

- [ ] **Signing key**: Auto-created on first run at `~/.aragora/signing.key`

## Dogfood run (~5 min)

```bash
# Step 1: Triage 5 emails with manual approval
aragora triage run --batch 5

# What happens:
# - Fetches 5 unread emails from Gmail
# - Runs adversarial debate on each (should I act on this?)
# - Generates signed receipt per email
# - Presents CLI review for human approval
# - Approved actions execute via Gmail API (archive/star/label)

# Step 2: Check status
aragora triage status
```

## What to verify

- [ ] Emails are fetched from real inbox (not mocked)
- [ ] Each email triggers a real multi-model debate
- [ ] Debate uses 2+ different LLM providers (check agent names in output)
- [ ] Receipt is generated with SHA-256 hash
- [ ] CLI review shows verdict + confidence + dissent
- [ ] Approved actions actually execute (email archived/starred/labeled)
- [ ] Rejected actions are NOT executed
- [ ] Total latency per email < 30 seconds

## Auto-approval mode (after manual validation)

```bash
# Run with auto-approval for low-risk actions
aragora triage run --batch 10 --auto-approve
```

## If something breaks

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "No Gmail credentials" | OAuth not set up | `python scripts/gmail_oauth_setup.py` |
| "At least 2 agents required" | Missing API keys | Add keys to `.env` |
| "Signing key not found" | First run | Will auto-create; run again |
| "Debate timed out" | Provider issue | Check rate limits; add `OPENROUTER_API_KEY` |
| "Action rejected" | Policy gate | Expected for high-risk actions |

## Key files

| Component | Path | Lines |
|-----------|------|-------|
| OAuth setup | `scripts/gmail_oauth_setup.py` | 490 |
| Trust wedge | `aragora/inbox/trust_wedge.py` | 1,216 |
| Triage runner | `aragora/inbox/triage_runner.py` | 475 |
| CLI review | `aragora/inbox/cli_review.py` | 259 |
| Auto-approval | `aragora/inbox/auto_approval.py` | 136 |
| Receipt executor | `aragora/inbox/receipt_gated_executor.py` | 149 |
| CLI command | `aragora/cli/commands/triage.py` | ~200 |
