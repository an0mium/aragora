# Aragora Review -- Beta User Guide

Multi-agent AI code review that debates your diffs.

## Quick Start (2 minutes)

```bash
# Install
pip install aragora

# Set at least one API key
export ANTHROPIC_API_KEY=sk-ant-...

# Try the demo (no API key needed)
aragora review --demo

# Review your code
git diff main | aragora review
```

## What It Does

`aragora review` runs a **multi-round debate** between AI models (Claude, GPT, Gemini, etc.) on your code changes. Each model reviews independently, then they critique each other's findings, revise positions, and vote on final issues.

You get:
- **Unanimous findings** -- issues all models agree on (high confidence)
- **Split opinions** -- where models disagree (transparent uncertainty)
- **Severity levels** -- CRITICAL, HIGH, MEDIUM, LOW
- **Agreement score** -- overall confidence metric

## Usage

### Review a diff from stdin
```bash
git diff main | aragora review
```

### Review a GitHub PR
```bash
aragora review https://github.com/owner/repo/pull/123
```
Requires `gh` CLI installed.

### Review a diff file
```bash
git diff main > changes.diff
aragora review --diff-file changes.diff
```

## Key Options

| Option | Default | Purpose |
|--------|---------|---------|
| `--agents` | auto-detected | Comma-separated agent list |
| `--rounds` | `3` | Debate rounds (1-10) |
| `--focus` | `security,performance,quality` | Review focus areas |
| `--output-format` | `github` | `github`, `json`, or `html` |
| `--ci` | false | Exit 1 on critical, 2 on high |
| `--sarif FILE` | none | Export SARIF 2.1.0 for GitHub Security |
| `--demo` | false | Run without API keys |

## CI/CD Integration

### GitHub Actions
```yaml
name: Code Review
on: pull_request

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Aragora Review
        uses: an0mium/aragora/.github/actions/aragora-review@main
        with:
          api-key: ${{ secrets.ARAGORA_API_KEY }}
          personas: security,performance
```

### Generic CI (exit code gating)
```bash
git diff $BASE_SHA | aragora review --ci --output-format json
# Exit 0 = clean, 1 = critical, 2 = high severity
```

## Cost Estimates

| Config | ~Cost |
|--------|-------|
| 2 agents, 2 rounds | $0.03 |
| 2 agents, 3 rounds | $0.08 |
| 3 agents, 3 rounds | $0.12 |
| 3 agents, 5 rounds | $0.25 |

Bring your own API keys -- Aragora never marks up LLM costs.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No API keys configured" | Export `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| "gh: command not found" | `brew install gh` (macOS) |
| Empty output | Ensure diff has changes: `git diff main \| wc -l` |
| Timeout | Increase with `--timeout 600` |

## Feedback

We're looking for:
1. Does the review catch real issues in your PRs?
2. Are there false positives? Which ones?
3. How does it compare to your current review process?
4. What's missing?

Report issues: https://github.com/synaptent/aragora/issues
