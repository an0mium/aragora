# Developer Quickstart: Review a Real PR in 5 Minutes

The fastest truthful path to a real result is to run `aragora review-pr` against a live GitHub pull request head. Aragora fetches the current remote diff, routes it to an available reviewer, prints the verdict in the terminal, and saves the structured artifacts to disk so you can inspect or automate against them.

> **Want to try without API keys?** Run the full platform locally with Docker:
> ```bash
> docker compose -f deploy/demo/docker-compose.yml up --build
> ```
> Backend at `localhost:8080`, frontend at `localhost:3000`. See [Docker Quickstart](guides/QUICKSTART_DOCKER.md) for details.

---

## Prerequisites

- Python 3.11 or later
- A local clone of the repository that owns the PR you want to review
- GitHub CLI installed and authenticated so `gh auth status` succeeds
- At least one available reviewer: authenticated `claude` CLI, authenticated `codex` CLI, or `OPENROUTER_API_KEY`

---

## Step 1: Install Aragora

```bash
pip install aragora
aragora --version
```

## Step 2: Run a Live `review-pr`

From the checkout of the repository that owns the PR:

```bash
cd /path/to/repo
gh auth status
aragora review-pr 123
```

The shortest path is to pass the PR number while you are inside that repository clone. If you want to be explicit, you can also pass the full PR URL:

```bash
aragora review-pr https://github.com/owner/repo/pull/123
```

Expected terminal output looks like this:

```text
PR #123 final status: changes_requested
Artifact dir: /path/to/repo/.aragora/review-pr/pr-123/20260322T154500Z
Latest review: changes_requested via codex
Findings:
  - [P1] Missing null guard on webhook payload
```

Every run writes durable files under the printed artifact directory:

- `run.json` -- final summary for the whole run
- `review-1.json` -- structured reviewer output
- `review-1.diff` -- the exact diff snapshot Aragora reviewed

Exit codes are stable for scripting:

- `0` -- review passed
- `2` -- reviewer requested changes
- `1` -- review was blocked or non-reviewable

`review-pr` is intentionally a live GitHub path. It does not have a demo mode. If you only want to preview the review format locally, use `aragora review --demo`.

## Step 3: Optional Fix-and-Rerun Loop

If the first pass returns `changes_requested`, Aragora can hand the findings to a fixer, push the branch, and re-review the updated head:

```bash
aragora review-pr 123 --fixer codex --auto-rerun
```

When you use the fix loop, Aragora also writes:

- `fix.json` -- fixer status, pushed commit SHAs, and worktree details
- `review-2.json` -- the follow-up review after the fixer push

Use `--keep-worktree` if you want to inspect the detached fixer worktree afterward.

## Step 4: Automate the Same Review on Every PR

Once you have a real local run working, add the GitHub Action so every PR gets the same review flow automatically.

### 4a. Add API keys as GitHub Secrets

Go to your repo: **Settings > Secrets and variables > Actions > New repository secret**

Add at least one:

| Secret Name | Provider |
|-------------|----------|
| `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | [OpenAI Platform](https://platform.openai.com/) |

### 4b. Create the workflow file

Create `.github/workflows/aragora-review.yml` in your repository:

```yaml
name: Aragora AI Review
on:
  pull_request:
    types: [opened, synchronize]

permissions:
  pull-requests: write
  contents: read

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: synaptent/aragora@main
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
```

Commit and push. Every new PR will now get an AI code review.

---

## Step 5: Read the PR Comment

When the action runs, it posts a comment on your PR with these sections:

### Unanimous Issues

Findings that all AI models agree on. These have the highest confidence and almost always warrant action.

```
## Unanimous Issues (2)
1. SQL injection vulnerability in user search -- query built with string concatenation
2. Missing input validation on file upload endpoint
```

### Split Opinions

Findings where models disagree. These are presented as tradeoffs for your judgment, not as directives.

```
## Split Opinions (2)
- Add request rate limiting
  Majority: anthropic-api, openai-api | Minority: gemini-api
- Cache database queries
  Majority: anthropic-api | Minority: openai-api, gemini-api
```

### Risk Areas

Lower-confidence findings flagged for manual review.

### Agreement Score

A 0-1 score indicating how much the models agreed overall. Higher scores mean more consensus across the review. A score of 0.75+ generally indicates strong agreement on the key findings.

---

## Step 6: Customize Diff-Based Review

`aragora review` is the diff-driven sibling of `review-pr`. Use it when you want to review local changes, run a demo, or export SARIF without fetching a live PR head.

### Focus areas

Narrow the review to specific concerns:

```bash
aragora review --focus security,performance
```

In the GitHub Action:

```yaml
- uses: synaptent/aragora@main
  with:
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    focus: 'security'
```

### SARIF export

Export findings as SARIF 2.1.0 for IDE integration and the GitHub Security tab:

```bash
aragora review --sarif
```

This creates `review-results.sarif` by default. Specify a custom path:

```bash
aragora review --sarif findings.sarif
```

In the GitHub Action, enable SARIF output:

```yaml
- uses: synaptent/aragora@main
  with:
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    output-format: 'sarif'
    sarif-upload: 'true'
```

### Fail builds on critical issues

Block PRs that have critical security findings:

```yaml
- uses: synaptent/aragora@main
  with:
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    fail-on-critical: 'true'
```

Then in your repo's **Settings > Branches > Branch protection rules**, enable "Require status checks to pass" and add the review job as a required check.

### Gauntlet mode

Run an adversarial stress-test after the standard review. The gauntlet uses attack/defend cycles to probe deeper for vulnerabilities:

```bash
aragora review --gauntlet
```

### Adjust debate depth

More rounds means more thorough review (and higher API cost):

```yaml
- uses: synaptent/aragora@main
  with:
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    rounds: '3'
```

### Filter by file type

Only trigger reviews on relevant file changes:

```yaml
on:
  pull_request:
    paths:
      - '**.py'
      - '**.ts'
      - '**.js'
```

---

## Advanced: Self-Hosted

Run the Aragora server locally or on your own infrastructure for full API access:

```bash
aragora serve --api-port 8080 --ws-port 8765
```

This gives you the REST API (3,000+ operations), WebSocket streaming, and programmatic access to debates, receipts, and analytics.

Start with the [Curated API Reference](/docs/api-reference) for the base URL, auth flow, and the essential Debates, Agents, Knowledge, and Workflow endpoints. Use the [Generated API Reference](./api/API_REFERENCE.md) for full endpoint-level schemas and response details. See [Documentation Index](INDEX.md) for architecture navigation.

---

## What Makes This Different

**Multi-model consensus, not a single opinion.** Standard AI code review tools run one model and present its output as truth. Aragora runs multiple models independently, then has them debate. You see where they agree (act on these) and where they disagree (use your judgment).

**Disagreement is a feature.** Split opinions are explicitly surfaced. When Claude flags a security issue but GPT-4 does not, that tells you something different than when both flag it. The disagreement itself is informative.

**Cryptographic decision receipts.** Every review produces a SHA-256 hashed audit trail -- which models participated, what they found, how they voted, and what the consensus was. This is not a log file; it is a verifiable receipt.

**SARIF integration.** Findings export as standard SARIF 2.1.0, which means they show up in the GitHub Security tab alongside your other code scanning tools, and in any IDE that supports SARIF.

---

## Cost Estimate

Approximate cost per review (2 agents, 2 rounds, typical PR):

| Provider | Cost per Review |
|----------|----------------|
| Anthropic Claude | ~$0.05-0.15 |
| OpenAI GPT-4 | ~$0.10-0.30 |
| OpenRouter fallback | ~$0.02-0.10 |

---

## Next Steps

- [Curated API Reference](/docs/api-reference) -- essential endpoints, auth, and section-by-section API navigation
- [Generated API Reference](./api/API_REFERENCE.md) -- full REST API schemas and endpoint details
- [Documentation Index](INDEX.md) -- architecture, memory tiers, and reference entry points
- Example workflows: `examples/github-action/` in the repository
