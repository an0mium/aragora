# Slack Code Review

Multi-agent code review powered by Aragora's debate engine, with optional
Slack integration for posting results to your team channel.

## How it works

1. Reads a code diff (file or stdin)
2. Runs a 3-round multi-agent debate where agents review the code for:
   - Bugs and logic errors
   - Security vulnerabilities
   - Performance concerns
   - Style and maintainability
   - Missing tests or docs
3. Outputs a structured review with severity ratings
4. Optionally posts the review to a Slack channel

## Setup

```bash
# Install aragora
pip install aragora

# Set at least one API key
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Optional: set Slack webhook for posting results
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../..."
```

## Usage

```bash
# Review a diff file
python examples/slack_code_review/main.py --diff changes.patch

# Pipe from git
git diff HEAD~1 | python examples/slack_code_review/main.py --diff -

# Output as JSON
git diff --staged | python examples/slack_code_review/main.py --diff - --json
```

## Integration with CI/CD

Add to your GitHub Actions workflow:

```yaml
- name: AI Code Review
  run: |
    git diff ${{ github.event.pull_request.base.sha }} HEAD \
      | python examples/slack_code_review/main.py --diff - --json \
      > review.json
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Customization

Edit `main.py` to adjust:
- `build_review_task()` -- change the review prompt or focus areas
- `DebateProtocol(rounds=3)` -- increase rounds for more thorough review
- `SlackConfig(channel=...)` -- change the target Slack channel
