# Aragora Code Review Action

Multi-agent AI code review for pull requests. Uses adversarial debate across LLM providers to catch issues that single reviewers miss.

## Quick Start

```yaml
name: Code Review
on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/aragora-code-review
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
```

## How It Works

1. Extracts the PR diff automatically
2. Multiple AI agents independently review the code
3. Agents debate and critique each other's findings
4. Consensus findings are highlighted; disagreements are flagged
5. Results posted as a PR comment with severity ratings

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `anthropic-api-key` | At least 2 keys | - | Anthropic API key |
| `openai-api-key` | required | - | OpenAI API key |
| `openrouter-api-key` | | - | OpenRouter API key (fallback) |
| `agents` | | auto | Comma-separated agent list |
| `rounds` | | `2` | Debate rounds (1-5) |
| `focus` | | `security,performance,quality` | Focus areas |
| `post-comment` | | `true` | Post results as PR comment |
| `fail-on-critical` | | `true` | Fail if critical issues found |
| `fail-on-high` | | `false` | Fail if high issues found |
| `sarif-upload` | | `false` | Upload SARIF to Security tab |

## Outputs

| Output | Description |
|--------|-------------|
| `critical-count` | Critical issues found |
| `high-count` | High-severity issues |
| `medium-count` | Medium-severity issues |
| `low-count` | Low-severity issues |
| `total-count` | Total issues |
| `agreement-score` | Agent consensus (0.0-1.0) |
| `review-path` | Path to JSON results |
| `sarif-path` | Path to SARIF output |

## Advanced Usage

### Gate PRs on Review Quality

```yaml
- uses: ./.github/actions/aragora-code-review
  id: review
  with:
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    fail-on-critical: 'true'
    fail-on-high: 'true'
    sarif-upload: 'true'

- name: Check agreement
  if: steps.review.outputs.agreement-score < 0.5
  run: echo "::warning::Low agent agreement -- review manually"
```

### Security-Focused Review

```yaml
- uses: ./.github/actions/aragora-code-review
  with:
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    focus: 'security'
    rounds: '3'
    sarif-upload: 'true'
```
