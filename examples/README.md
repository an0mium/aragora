# Aragora Examples

Practical examples demonstrating Aragora's multi-agent debate engine, workflow automation, and integration patterns.

## Example App Pack

Three self-contained example applications, each with its own `main.py`, `README.md`, and `requirements.txt`. All three support `--demo` mode so you can run them without API keys.

### 1. Slack Code Review Bot

**Directory:** [`slack-review-bot/`](./slack-review-bot/)

A Slack bot that uses multi-agent debate to review GitHub pull requests. Security, performance, and best-practices reviewers debate code changes and post a consensus review to Slack with a tamper-evident receipt hash.

```bash
# Demo mode -- no API keys or Slack webhook needed
python examples/slack-review-bot/main.py --demo

# Review a real PR and post to Slack
python examples/slack-review-bot/main.py \
    --repo myorg/myrepo --pr 42 \
    --webhook https://hooks.slack.com/services/T.../B.../...

# JSON output for CI/CD pipelines
python examples/slack-review-bot/main.py --demo --json
```

**Uses:** `Arena`, `DebateProtocol`, `SlackIntegration`, receipt generation

See [slack-review-bot/README.md](./slack-review-bot/README.md) for full setup and architecture.

---

### 2. Document Analysis Pipeline

**Directory:** [`document-analysis/`](./document-analysis/)

A document Q&A system where multiple AI agents debate answers grounded in document evidence. Ingest a directory of markdown, text, code, or config files, then ask questions and get cited, consensus-backed answers.

```bash
# Demo mode -- uses built-in sample architecture docs
python examples/document-analysis/main.py --demo

# Analyze your own docs
python examples/document-analysis/main.py \
    --docs /path/to/architecture/docs \
    --question "What is the authentication strategy?"

# Interactive Q&A (ask multiple questions without reloading)
python examples/document-analysis/main.py --docs /path/to/docs --interactive

# JSON output for piping
python examples/document-analysis/main.py --demo --json | jq '.answer'
```

**Uses:** `Arena`, `DebateProtocol`, document ingestion, evidence-grounded debate

See [document-analysis/README.md](./document-analysis/README.md) for full setup and architecture.

---

### 3. Workflow Automation

**Directory:** [`workflow-automation/`](./workflow-automation/)

A content publishing pipeline built on Aragora's `WorkflowEngine`. Demonstrates DAG-based orchestration with sequential steps, parallel branches, conditional routing, checkpointing, and event tracking.

```bash
# Demo mode -- no API keys needed
python examples/workflow-automation/main.py --demo

# Custom topic
python examples/workflow-automation/main.py --topic "API security guidelines"

# View the workflow DAG without executing
python examples/workflow-automation/main.py --show-dag

# JSON output for programmatic use
python examples/workflow-automation/main.py --demo --json
```

**Uses:** `WorkflowEngine`, `WorkflowDefinition`, `StepDefinition`, `TransitionRule`, custom `BaseStep` subclasses

See [workflow-automation/README.md](./workflow-automation/README.md) for full setup and architecture.

---

## Quick-Start Examples

Standalone scripts for learning the core APIs. No project structure needed.

### Simple Multi-Agent Debate (2-5 min)

```bash
python examples/01_simple_debate.py
```

Multiple AI agents debate a topic, critique each other, and reach consensus. Demonstrates the core propose-critique-revise workflow.

### Tournament & Leaderboard (10-30 min)

```bash
python examples/02_tournament.py
```

Agents compete across multiple topics with ELO ranking. Shows how Aragora tracks agent performance over time.

### Nomic Loop - Self-Improvement (5-10 min)

```bash
python examples/03_nomic_loop.py
```

Demonstrates Aragora's self-improvement capability where agents debate improvements, design solutions, implement code, and verify changes (dry-run mode).

### Gauntlet Showcase - Decision Assurance (1-2 min)

```bash
python examples/04_gauntlet_showcase.py
```

Adversarial validation and compliance stress-testing with decision receipt generation. No API keys required -- views pre-computed results from 12 AI models.

### TypeScript SDK Integration (5 min)

```bash
aragora serve --api-port 8080 --ws-port 8765  # in one terminal
npx ts-node examples/05_typescript_sdk.ts       # in another
```

Demonstrates the TypeScript SDK for programmatic debate creation, polling, and error handling.

## SDK Demo Apps

| App | Language | Directory |
|-----|----------|-----------|
| Python Debate CLI | Python | [`python-debate/`](./python-debate/) |
| TypeScript Web App | TypeScript | [`typescript-web/`](./typescript-web/) |
| Node.js Slack Bot | TypeScript | [`nodejs-slack-bot/`](./nodejs-slack-bot/) |
| Next.js App Router | TypeScript | [`nextjs-app-router/`](./nextjs-app-router/) |
| Remix | TypeScript | [`remix/`](./remix/) |
| SvelteKit | TypeScript | [`sveltekit/`](./sveltekit/) |

## Using Real Agents

Set at least one API key for real LLM-powered debates:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."  # for Grok
```

For best results, set 2-3 keys to enable multi-agent debates with diverse models.

## Troubleshooting

**"Need at least 2 agents"** -- Set more API keys (at least 2 providers needed for debate).

**Timeout errors** -- Some API providers may be slow. Try setting `ARAGORA_AGENT_TIMEOUT` to increase the timeout.

**Rate limit errors** -- Set `OPENROUTER_API_KEY` for automatic fallback on 429 errors.

## Full Documentation

- [Python SDK Guide](../docs/SDK_GUIDE.md)
- [API Reference](../docs/api/API_REFERENCE.md)
- [Architecture Overview](../CLAUDE.md)
- [Feature Status](../docs/STATUS.md)
