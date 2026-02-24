# Slack Code Review Bot

A Slack bot powered by Aragora's multi-agent debate engine that performs comprehensive code reviews on pull requests. Multiple AI agents review code from different perspectives (security, performance, best practices) and reach consensus on findings.

## How It Works

```
Slack Command (/review owner/repo#42)
        |
        v
+------------------+
|  Fetch PR Diff   |  <-- GitHub API / gh CLI
+------------------+
        |
        v
+------------------+
| Multi-Agent      |  Security Reviewer (Claude)
| Debate           |  Performance Reviewer (GPT)
| (Aragora Arena)  |  Best Practices Reviewer (Gemini)
+------------------+
        |
        v
+------------------+
| Consensus Engine |  <-- Majority/Unanimous voting
+------------------+
        |
        v
+------------------+
| Post to Slack    |  <-- Formatted findings + receipt hash
+------------------+
```

## Quick Start (Demo Mode)

No API keys required -- uses a sample PR diff with intentional vulnerabilities:

```bash
python examples/slack-review-bot/main.py --demo
```

## Full Setup

### 1. Install Dependencies

```bash
pip install -r examples/slack-review-bot/requirements.txt
```

### 2. Set Environment Variables

```bash
# At least one AI provider key
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Slack webhook for posting results
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../..."

# Optional: GitHub token for private repos
export GITHUB_TOKEN="ghp_..."
```

### 3. Run a Review

```bash
# Review a specific PR
python examples/slack-review-bot/main.py --repo myorg/myrepo --pr 42

# Review and post to a specific Slack channel
python examples/slack-review-bot/main.py \
    --repo myorg/myrepo --pr 42 \
    --webhook https://hooks.slack.com/services/T.../B.../... \
    --channel "#security-reviews"

# Use more debate rounds for thorough review
python examples/slack-review-bot/main.py --repo myorg/myrepo --pr 42 --rounds 3

# JSON output for CI/CD pipelines
python examples/slack-review-bot/main.py --demo --json
```

## Features

- **Multi-agent consensus**: Agents debate code issues and converge on findings
- **Severity ratings**: Critical / High / Medium / Low classification
- **Category tagging**: Security, Performance, Best Practices
- **Audit trail**: SHA-256 receipt hash for every review
- **Slack formatting**: Rich mrkdwn messages with issue breakdowns
- **JSON output**: Use `--json` for machine-readable output (CI/CD integration)
- **Graceful fallback**: Demo mode when API keys are unavailable
- **Rate limiting**: Built-in Slack rate limiting via Aragora's SlackIntegration

## Architecture

The bot uses three key Aragora components:

1. **`Arena`** -- Orchestrates the multi-agent debate with configurable rounds and consensus rules
2. **`SlackIntegration`** -- Posts formatted results with rate limiting and circuit breaker protection
3. **`GitHubConnector`** -- Fetches PR diffs (optional; falls back to demo data)

## Output Example

```
======================================================================
  Code Review: PR #42 in example/repo
======================================================================

  Consensus: YES (confidence 92%, 2 rounds)
  Reviewers: security_reviewer, performance_reviewer, best_practices_reviewer

  --- 6 Issues Found ---

  1. [Critical] SQL Injection in authenticate_user (Security)
     The query uses f-string interpolation with user input...
     Fix: Use parameterized queries

  2. [Critical] SQL Injection in delete_all_users (Security)
     User-supplied IDs are interpolated directly into DELETE query...

  ...

  Receipt: a3f8b2c1d4e5f6a7...
  Reviewed at: 2026-02-23T10:30:00+00:00
======================================================================
```
