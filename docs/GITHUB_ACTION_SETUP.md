# GitHub Action Setup Guide

Add multi-agent AI code review to your pull requests in under 5 minutes.

## Quick Start

### 1. Add API Keys as GitHub Secrets

Go to your repo's **Settings > Secrets and variables > Actions** and add at least one:

| Secret | Required | Provider |
|--------|----------|----------|
| `ANTHROPIC_API_KEY` | Yes (or OpenAI) | [Anthropic Console](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | Yes (or Anthropic) | [OpenAI Platform](https://platform.openai.com/) |
| `OPENROUTER_API_KEY` | No | Fallback provider |

For best results, add both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` -- multi-model consensus produces higher-quality reviews.

### 2. Add the Workflow File

Create `.github/workflows/aragora-review.yml` in your repository:

```yaml
name: Aragora AI Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

concurrency:
  group: aragora-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    name: AI Code Review
    runs-on: ubuntu-latest
    if: github.event.pull_request.draft == false && github.actor != 'dependabot[bot]'

    steps:
      - name: Run Aragora Review
        id: review
        uses: an0mium/aragora@main
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          post-comment: 'true'
          fail-on-critical: 'false'
```

### 3. Open a Pull Request

That's it. The next PR will get an AI code review comment.

## Configuration

### Action Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `agents` | `anthropic-api,openai-api` | Comma-separated agent list |
| `rounds` | `2` | Number of debate rounds (1-5) |
| `focus` | `security,performance,quality` | Review focus areas |
| `post-comment` | `true` | Post review as PR comment |
| `fail-on-critical` | `false` | Fail CI if critical issues found |
| `max-diff-size` | `50000` | Max diff size in bytes |

### Action Outputs

| Output | Description |
|--------|-------------|
| `review-path` | Path to generated review file |
| `review-generated` | Whether a PR comment was generated |
| `review-json-path` | Path to the generated `review.json` (structured output) |
| `review-log-path` | Path to the `review.log` file |
| `unanimous-count` | Issues all agents agree on |
| `critical-count` | Critical severity issues |
| `high-count` | High severity issues |
| `medium-count` | Medium severity issues |
| `low-count` | Low severity issues |
| `total-count` | Total severity issues (`critical+high+medium+low`) |
| `risk-areas-count` | Risk areas noted (lower confidence items) |
| `split-opinions-count` | Split opinions (agent disagreement items) |
| `agreement-score` | Agent agreement score (0-1) |

### Strict Mode (Block PRs on Critical Issues)

Set `fail-on-critical: 'true'` and add the review as a required status check:

1. Set `fail-on-critical: 'true'` in the workflow
2. Go to **Settings > Branches > Branch protection rules**
3. Enable "Require status checks to pass" and add "AI Code Review"

See `examples/github-action/aragora-review-strict.yml` for a complete example.
See also `examples/github-action/basic.yml` and `examples/github-action/advanced.yml`.

## How It Works

1. The action fetches the PR diff using `gh pr diff`
2. Multiple AI agents independently review the code
3. Agents debate findings over multiple rounds to reduce false positives
4. A consensus report is posted as a PR comment

### Review Comment Structure

The review comment includes:

- **Unanimous Issues** -- All agents agree these need attention (highest confidence)
- **Critical & High Severity** -- Security vulnerabilities, data loss risks
- **Split Opinions** -- Agents disagree, presented as tradeoffs for your judgment
- **Risk Areas** -- Lower confidence findings for manual review
- **Agreement Score** -- How much the agents agreed overall

## Customization

### Review Only Specific Files

Use GitHub Actions path filters to only trigger reviews on certain file types:

```yaml
on:
  pull_request:
    paths:
      - '**.py'
      - '**.ts'
      - '**.js'
```

### Security-Only Reviews

Focus the review on security concerns:

```yaml
- uses: an0mium/aragora@main
  with:
    focus: 'security'
    rounds: '3'
    fail-on-critical: 'true'
```

### Skip Large PRs

The `max-diff-size` input prevents excessive API costs on large PRs. The default of 50KB handles most PRs. For monorepo or generated code, increase it:

```yaml
- uses: an0mium/aragora@main
  with:
    max-diff-size: '200000'
```

## Troubleshooting

### "No API keys configured"

Ensure your GitHub Secrets are named exactly `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and are passed to the action via `anthropic-api-key` / `openai-api-key` inputs.

### Review comment is empty

Check the workflow logs. Common causes:
- Diff is empty (no file changes)
- Diff exceeds `max-diff-size` (increase the limit)
- API key is invalid or expired

### Rate limiting

If you have many PRs, consider:
- Using `concurrency` groups to limit parallel reviews
- Reducing `rounds` from 3 to 2
- Using only one agent instead of two

### Costs

Approximate costs per review (2 agents, 2 rounds, typical PR):
- Anthropic Claude: ~$0.05-0.15
- OpenAI GPT-4: ~$0.10-0.30
- With OpenRouter fallback: ~$0.02-0.10
