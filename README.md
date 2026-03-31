# Aragora

Aragora is an auditable execution control plane for AI-assisted work. It
governs consequential decisions and execution with multi-model review,
receipts, provenance, and truthful gates.

### Govern AI-Assisted Work With Receipts, Review, and Truthful Gates

[![PyPI](https://img.shields.io/pypi/v/aragora)](https://pypi.org/project/aragora/)
[![Tests](https://github.com/an0mium/aragora/actions/workflows/test.yml/badge.svg)](https://github.com/an0mium/aragora/actions/workflows/test.yml)
[![Smoke Tests](https://github.com/an0mium/aragora/actions/workflows/smoke.yml/badge.svg)](https://github.com/an0mium/aragora/actions/workflows/smoke.yml)
[![Docker Build](https://github.com/an0mium/aragora/actions/workflows/docker.yml/badge.svg)](https://github.com/an0mium/aragora/actions/workflows/docker.yml)
[![Deploy](https://github.com/an0mium/aragora/actions/workflows/deploy-lightsail.yml/badge.svg)](https://github.com/an0mium/aragora/actions/workflows/deploy-lightsail.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**New here?** Start with the [Quickstart Guide](docs/quickstart.md) -- you'll have a working debate in under a minute. For a deeper overview, see [Start Here](docs/START_HERE.md). For strategic framing, see [Competitive Positioning (March 2026)](docs/strategy/COMPETITIVE_POSITIONING_2026_03.md), [When To Use Aragora Vs Execution Substrates](docs/strategy/WHEN_TO_USE_ARAGORA_VS_EXECUTION_SUBSTRATES.md), and the [Terminology And Phrase Glossary](docs/strategy/TERMINOLOGY_GLOSSARY.md).

| I want to... | Install |
|--------------|---------|
| Try a debate in 30 seconds | `pip install aragora-debate` |
| Call the Aragora API from Python | `pip install aragora-sdk` |
| Self-host the full platform | `docker compose -f deploy/demo/docker-compose.yml up` |

**Individual LLMs are unreliable. Their personas shift with context, their
confidence does not correlate with accuracy, and they often optimize for
plausible agreement instead of truth.**

Aragora treats that as a systems problem. It coordinates heterogeneous models
through structured debate and review, preserves receipts and provenance, and
stops truthfully when evidence is insufficient. The goal is not just faster AI
output, but governed AI-assisted execution you can actually inspect.

Why teams adopt Aragora:

- **Disagreement becomes useful evidence.** Models challenge each other before work advances.
- **Every consequential action has a receipt.** Review, provenance, confidence, and next steps stay inspectable.
- **Execution is bounded and truthful.** Consequential actions are denied by default unless an admin-scoped approval artifact exists.
- **Sandboxed effectors are mandatory.** Browser automation and similar effectors require an approved sandbox backend instead of host-side execution.
- **It fits above existing tools.** Use Aragora when review and governance matter; keep direct runtimes when raw speed is enough.

## Product Boundary

Aragora is not a generic autonomous-agent platform or a replacement for worker
runtimes like Codex, Claude Code, OpenCode, or Pi. It is the control plane used
when AI-assisted work becomes consequential enough to require receipts, review,
provenance, bounded delegation, and truthful stopping behavior.

The explicit product non-goals are tracked in the
[Non-Goals Ledger](docs/strategy/NON_GOALS_LEDGER.md). In practice that means:

- we do not try to win on raw agent breadth or generic orchestration alone
- we do not sell lights-out autonomy as the default story
- we do not replace execution substrates; we govern work that runs through them
- we do not treat bigger swarms as a product goal unless they improve truthfulness
- we do not advance work without evidence, review, and clear terminal states

## Try It Now

**Option A -- One command, no API keys:**

```bash
pip install aragora && aragora demo
```

This runs a full adversarial debate with mock agents and opens a decision receipt in your browser.

**Option B -- Docker (includes dashboard UI):**

```bash
docker compose -f deploy/demo/docker-compose.yml up
# Open http://localhost:3000 — try any question in the playground
```

**Option C -- Live playground:**

Visit the deployed instance and type any question. Three AI agents will debate it, critique each other, vote, and produce a shareable decision receipt.

<details>
<summary>What you'll see (click to expand)</summary>

```
================================================================
  ARAGORA DEMO -- Adversarial Decision Stress-Test
================================================================

  Topic:  Should we adopt microservices?
  Agents: Analyst, Critic, Synthesizer, Devil's Advocate
  Rounds: 2

  --- Round 1 --------------------------------------------------

  [ANALYST] (supportive)
    This is a sound strategy. The evidence points toward
    significant gains in maintainability and team productivity.

  [CRITIC] (critical)
    The claimed benefits are overstated. Most organizations
    underestimated the operational burden by 3-5x. I recommend
    a modular monolith as the safer path.

  [SYNTHESIZER] (balanced)
    The tradeoffs here are real. On one hand, the current
    architecture limits independent scaling. On the other,
    the migration carries execution risk.

  --- Decision Receipt -----------------------------------------

  Verdict:    CONDITIONAL APPROVAL
  Confidence: 72%
  Consensus:  Partial (3 of 4 agents)
  Dissent:    Devil's Advocate flagged migration risk
```

</details>

```bash
# Review your current changes against main
git diff main | aragora review --demo

# Or review a GitHub PR
aragora review --pr https://github.com/org/repo/pull/123 --demo
```

```bash
# Stress-test a specification
aragora gauntlet spec.md --profile thorough --output receipt.html

# Run a multi-agent debate
aragora ask "Design a rate limiter for 1M req/sec" --agents anthropic-api,openai-api,gemini

# Start the API server
aragora serve
```

### Self-Improving Pipeline

Aragora can improve itself under explicit supervision -- decompose a vague goal,
assign subtasks to isolated worktrees, execute with gauntlet validation, and
merge passing branches:

```bash
# Preview what the pipeline will do
aragora self-improve "Maximize utility for SME businesses" --dry-run

# Run with human approval gates and budget cap
aragora self-improve "Harden security" --require-approval --budget-limit 20 --receipt
```

Each subtask gets an isolated git worktree, cross-agent review, sandbox
validation, and a receipt trail before merge or truthful escalation. This is a
governed execution path, not a claim that Aragora is a generic autonomous-agent
runtime.

Current default boundary:

- public and issue-driven paths can draft plans and receipts, but cannot execute code, drive browsers, mutate git, emit consequential webhooks, or write on-chain without an explicit approval record
- browser execution requires a sandbox backend; host-side browser automation is disabled by default
- blockchain write requests queue durable chain actions for an admin signer lane instead of signing in the request path

### Add to Your CI Pipeline (1 minute)

```yaml
# .github/workflows/aragora-review.yml
name: Aragora Review
on:
  pull_request:
    types: [opened, synchronize]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: an0mium/aragora@main
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
```

Or generate it automatically: `aragora init --ci github`

---

## Five Pillars

Aragora is built on five architectural commitments designed for a world where individual AI agents cannot be trusted with consequential decisions alone.

### 1. SMB-Ready, Enterprise-Grade

Aragora is useful to a 5-person startup on day one and scales to regulated enterprise without rearchitecting. Enterprise features -- OIDC/SAML SSO, MFA, AES-256-GCM encryption, multi-tenant isolation, RBAC with 7 roles and 450+ permission combinations, SOC 2 / GDPR / HIPAA compliance frameworks -- are built in, not bolted on. Security hardening (rate limiting, SSRF protection, path traversal guards, input validation, audit trails) is the default, not a premium tier.

### 2. Leading-Edge Memory and Context

Single agents lose context. Aragora's 4-tier Continuum Memory (fast / medium / slow / glacial) and Knowledge Mound with 42 registered adapter specs give every debate access to institutional history, cross-session learning, and evidence provenance. The RLM (Recursive Language Models) system compresses and structures context to reduce prompt bloat, enabling debates that sustain coherence across long multi-round sessions and large document sets where individual models would degrade.

### 3. Control Plane Above Worker Runtimes

Codex, Claude Code, OpenCode, Pi, and similar tools are useful execution
substrates. Aragora sits above them when the work needs governance: bounded
delegation, explicit review, receipt generation, merge and publish gates, and
truthful blocker handling. Provider breadth, connectors, SDKs, and workflow
surfaces matter because they let Aragora fit into your stack, but they are
supporting infrastructure, not the primary moat.

### 4. Multi-Agent Robustness

Individual LLMs exhibit persona instability -- their outputs shift based on framing, context, and even prompt ordering. Aragora treats this as a feature: by running Claude, GPT, Gemini, Grok, Mistral, DeepSeek, Qwen, Kimi, and local models in structured Propose / Critique / Revise debates, the system surfaces disagreements that reveal genuine uncertainty. ELO rankings track agent performance. Calibration scoring (Brier scores) measures prediction accuracy. The Trickster detects hollow consensus where models agree without genuine reasoning. The result: when models with different training data independently converge on an answer, that convergence is meaningful -- and when they disagree, the dissent trail tells you exactly where human judgment is needed.

### 5. Self-Healing and Self-Extending

The Nomic Loop is Aragora's autonomous self-improvement system: agents debate improvements to the codebase, design solutions, implement code, run tests, and verify changes -- with human approval gates and automatic rollback on failure. This is how Aragora grew from a debate engine to 3,200+ modules. Red-team mode stress-tests the platform's own specs. The Gauntlet runs adversarial attacks against proposed changes. The system hardens itself.

---

## Why Aragora?

A single LLM will confidently give you a wrong answer and you won't know it. Research shows that LLM personas are context-dependent, fragile under adversarial pressure, and prone to sycophantic agreement with whoever is asking. [Stanford's taxonomy of LLM reasoning failures](https://arxiv.org/abs/2602.06176) documents systematic breakdowns in formal logic, unfaithful chain-of-thought, and robustness failures under minor prompt variations -- exactly the failure modes that structured adversarial debate is designed to surface. When the decision matters -- hiring, architecture, compliance, strategy -- one model's opinion is insufficient.

Aragora treats each model as an **unreliable witness** and uses structured debate, review, and receipts to extract signal from disagreement instead of hiding it behind a single routed answer:

| What you get | How it works |
|---|---|
| **Adversarial Validation** | Models with different training data and blind spots challenge each other's reasoning |
| **Decision Receipts** | Cryptographic audit trails with evidence chains, dissent tracking, and confidence calibration |
| **Gauntlet Mode** | Red-team stress-tests for specs, policies, and architectures using adversarial personas |
| **Calibrated Trust** | ELO rankings and Brier scores track which models are actually reliable on which domains |
| **Institutional Memory** | Decisions persist across sessions with 4-tier memory and Knowledge Mound (<!-- adpt-count -->42<!-- /adpt-count --> adapters) |
| **Channel Delivery** | Results route to Slack, Teams, Discord, Telegram, WhatsApp, email, or voice |

### If You're New: How To Read Aragora

Aragora is designed for the moment after an AI recommendation is made, when a buyer, reviewer, or operator asks: "Why should I trust this enough to act?"

Every consequential run is explained through three plain-language elements:

| Term | What it means in practice |
|---|---|
| **Receipt** | The record of what was asked, what Aragora recommended, and what should happen next |
| **Evidence** | The documents, policies, tests, prior decisions, or artifacts Aragora relied on |
| **Dissent** | The disagreements, residual risks, or open questions that still need human judgment |

The goal is not to make AI sound more certain. The goal is to make the decision legible. A first-time operator should be able to answer four questions in under a minute:

1. What happened?
2. Why did the system reach that recommendation?
3. What could still be wrong?
4. What is the next human action?

That is the purpose of Aragora receipts: not just a log, but a reviewable handoff.

---

## Quick Start

### 1. Install and Try It (30 seconds to demo, ~5 minutes for a real PR review)

```bash
pip install aragora

# Run a zero-config demo debate — opens receipt in your browser
aragora quickstart --demo

# From a clone of the target repo, review the live GitHub PR head
aragora review-pr 123
```

`aragora review-pr` fetches the current remote PR head from GitHub, prints a pass or changes-requested summary, and writes artifacts under `.aragora/review-pr/`. For the first real run, make sure `gh auth status` succeeds and at least one reviewer is available (`claude`, `codex`, or `OPENROUTER_API_KEY`).

See [docs/QUICKSTART_DEVELOPER.md](docs/QUICKSTART_DEVELOPER.md) for the 5-minute `review-pr` path, artifact layout, and optional fix loop.

### 2. Run Debates and Start the Server

```bash
# Set at least one API key
export ANTHROPIC_API_KEY=your-key  # or OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY

# Run a multi-agent debate
aragora ask "Should we adopt microservices?" --agents anthropic-api,openai-api --rounds 3

# Start the API server
aragora serve
```

See [docs/guides/GETTING_STARTED.md](docs/guides/GETTING_STARTED.md) for the complete 5-minute setup.

### 3. Deploy with Docker

```bash
# Clone and deploy
git clone https://github.com/an0mium/aragora && cd aragora

# Production deployment (secrets from AWS Secrets Manager)
cd deploy/liftmode && ./setup.sh

# Or run directly with Docker Compose
docker compose -f deploy/docker-compose.yml up -d
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment options (Docker, Kubernetes, offline mode).

### 4. Develop with the SDK

| Package | Install | Purpose | PyPI |
|---|---|---|---|
| `aragora` | `pip install aragora` | Full platform (server, CLI, debate engine) | [v2.8.0](https://pypi.org/project/aragora/) |
| `aragora-debate` | `pip install aragora-debate` | Standalone debate engine (no server needed) | [v0.2.0](https://pypi.org/project/aragora-debate/) |
| `aragora-sdk` | `pip install aragora-sdk` | Python client SDK for connecting to aragora | [v2.8.0](https://pypi.org/project/aragora-sdk/) |
| `@aragora/sdk` | `npm install @aragora/sdk` | TypeScript/Node.js client SDK | — |

---

## Core Workflows

### 1. Gauntlet Mode -- Adversarial Stress Testing

Stress-test specs, architectures, and policies before they ship:

```bash
aragora gauntlet spec.md --input-type spec --profile quick
aragora gauntlet policy.yaml --input-type policy --persona gdpr
aragora gauntlet architecture.md --profile thorough --output report.html
```

| Attack Type | What It Tests |
|---|---|
| **Red Team** | Security holes, injection points, auth bypasses |
| **Devil's Advocate** | Logic flaws, hidden assumptions, edge cases |
| **Scaling Critic** | Performance bottlenecks, SPOF, thundering herd |
| **Compliance** | GDPR, HIPAA, SOC 2, AI Act violations |

Decision receipts provide cryptographic audit trails for every finding.

### 2. AI Code Review

Get **multi-model consensus** on your pull requests:

```bash
git diff main | aragora review
aragora review https://github.com/owner/repo/pull/123
aragora review --demo  # try without API keys
```

When 3+ independent models with different training data agree on an issue, that convergence is meaningful. Split opinions show where human judgment is needed -- the disagreement is the signal.

### 3. Structured Debates

The debate protocol follows thesis > antithesis > synthesis:

1. **Propose** -- Agents generate initial responses from different perspectives
2. **Critique** -- Agents challenge each other's proposals with severity scores
3. **Revise** -- Proposers incorporate valid critiques
4. **Synthesize** -- Judge combines best elements into a final answer

Configurable consensus: majority, unanimous, judge-based, or none.

---

## Architecture

```
aragora/
├── debate/         # Core debate engine (210+ modules)
│   ├── orchestrator.py   # Arena -- main debate loop
│   ├── consensus.py      # Consensus detection and proofs
│   ├── convergence.py    # Semantic similarity detection
│   └── phases/           # Propose, critique, revise, vote, judge
├── agents/         # 43 registered agent types (CLI, direct API, OpenRouter, local)
│   ├── api_agents/       # Anthropic, OpenAI, Gemini, Grok, Mistral, OpenRouter
│   ├── cli_agents.py     # Claude Code, Codex, Gemini CLI, Grok CLI
│   └── fallback.py       # OpenRouter fallback on quota errors
├── gauntlet/       # Adversarial stress testing
├── knowledge/      # Knowledge Mound with 42 registered adapter specs
├── memory/         # 4-tier memory (fast/medium/slow/glacial)
├── server/         # 3,000+ API operations, 270+ WebSocket event types
├── pipeline/       # Decision-to-PR generation
├── genesis/        # Fractal debates, agent evolution
├── sandbox/        # Docker-based safe execution
├── rbac/           # Role-based access control (7 roles, 360+ permissions)
├── compliance/     # SOC 2, GDPR, HIPAA frameworks
└── workflow/       # DAG-based automation engine
```

**Scale:** 3,000+ Python modules | 152,000+ tests across 5,000+ test files

### Performance and Costs

| Metric | Typical Value |
|---|---|
| Debate latency (3 agents, 2 rounds) | 30-90 seconds |
| Token usage per debate | 8,000-25,000 tokens |
| Estimated cost per debate | $0.05-$0.30 (depends on models) |
| Concurrent debates supported | 50+ (configurable) |
| API response time (cached) | < 200ms |
| Memory tier lookup (fast tier) | < 10ms |

Costs vary by model mix. Use `aragora decide --dry-run` to preview costs before execution.

| Model Mix | Agents | Rounds | Typical Cost |
|-----------|--------|--------|--------------|
| Haiku + GPT-4o-mini | 3 | 2 | ~$0.05 |
| Sonnet + GPT-4o | 3 | 3 | ~$0.15 |
| Opus + GPT-4 | 5 | 3 | ~$0.30 |
| Mock agents (demo mode) | Any | Any | $0.00 |

### How Aragora Compares

| Capability | Aragora | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Adversarial debate protocol | Built-in (propose/critique/revise) | Manual | No | No |
| Decision receipts with audit trail | Cryptographic, SHA-256 | No | No | No |
| Agent calibration (ELO + Brier) | Built-in | No | No | No |
| Multi-model consensus | Heterogeneous by design, with adversarial review and receipts | Single-provider | Single-provider | Multi-provider |
| Gauntlet stress testing | Built-in CLI | No | No | No |
| Enterprise security (SSO, RBAC, encryption) | Production-ready | No | No | No |
| Self-improvement (Nomic Loop) | Autonomous with safety gates | No | No | No |
| Knowledge persistence (42 adapter specs) | 4-tier memory + Knowledge Mound | Custom | Custom | Custom |
| Channel delivery (Slack, Teams, etc.) | 8 channels built-in | No | No | No |

---

## Programmatic Usage

```python
from aragora import Arena, Environment, DebateProtocol
from aragora.agents import create_agent

agents = [
    create_agent("anthropic-api", name="claude", role="proposer"),
    create_agent("openai-api", name="gpt", role="critic"),
    create_agent("gemini", name="gemini", role="synthesizer"),
]

env = Environment(task="Design a distributed cache with LRU eviction")
protocol = DebateProtocol(rounds=3, consensus="majority")
arena = Arena(env, agents, protocol)
result = await arena.run()

print(result.final_answer)
print(f"Consensus: {result.consensus_reached} ({result.confidence:.0%})")
```

### Python SDK

```python
from aragora.client import AragoraClient

client = AragoraClient(base_url="http://localhost:8080")
debate = client.debates.run(task="Should we adopt microservices?")
receipt = await client.gauntlet.run_and_wait(input_content="spec.md")
```

See [docs/SDK_GUIDE.md](docs/SDK_GUIDE.md) for the full API.

---

## Channels and Integrations

Aragora delivers debate results to wherever your team works:

| Channel | Status |
|---|---|
| Slack | Bot + OAuth |
| Microsoft Teams | Bot + OAuth |
| Discord | Interactions API |
| Telegram | Bot API |
| WhatsApp | Business API |
| Email | SMTP + Gmail + Outlook |
| Voice | TTS integration |
| Webhooks | Custom delivery |

Results automatically route to the originating channel via bidirectional chat routing.

See [docs/integrations/INTEGRATIONS.md](docs/integrations/INTEGRATIONS.md) for setup.

---

## Enterprise Features

| Category | Capabilities |
|---|---|
| **Authentication** | OIDC/SAML SSO, MFA (TOTP/HOTP), API key management, SCIM 2.0 |
| **Multi-Tenancy** | Tenant isolation, resource quotas, usage metering |
| **Security** | AES-256-GCM encryption, rate limiting, SSRF protection, key rotation |
| **Compliance** | SOC 2 controls, GDPR support, HIPAA, audit trails |
| **Observability** | Prometheus metrics, Grafana dashboards, OpenTelemetry tracing |
| **RBAC** | 7 roles, 360+ permissions, decorator-based enforcement |
| **Backup** | Incremental backups, retention policies, disaster recovery |
| **Control Plane** | Agent registry, task scheduler, health monitoring, policy governance |

See [docs/enterprise/ENTERPRISE_FEATURES.md](docs/enterprise/ENTERPRISE_FEATURES.md) for details.

---

## Self-Improvement (Nomic Loop)

Aragora includes an autonomous self-improvement system where agents debate and implement improvements to the codebase itself. **Experimental** -- always run in a sandbox with human review.

```bash
python scripts/run_nomic_with_stream.py run --cycles 3
python scripts/self_develop.py --goal "Improve test coverage" --require-approval
```

Safety: automatic backups, protected file checksums, rollback on failure, explicit approval gates for consequential actions, sandbox-required browser execution, and asynchronous chain settlement.

---

## Deployment

| Goal | Command | Requirements |
|------|---------|-------------|
| **Try it** | `docker compose -f deploy/demo/docker-compose.yml up` | Docker only |
| **Self-hosted** | `cd deploy/self-hosted && docker compose up -d` | Docker + API key |
| **Local dev** | `aragora serve --api-port 8080 --ws-port 8765` | Python + API key |

See [deploy/README.md](deploy/README.md) for the full deployment guide.

**API:** REST at `/api/v2/*` | WebSocket at `/ws` | OpenAPI at `/api/openapi`

---

## Security

- Ed25519 signature verification for webhooks (Discord, Slack)
- Rate limiting (IP, token, and endpoint-based)
- Input validation and content-length enforcement
- CORS allowlists, security headers, error message sanitization
- Path traversal protection, upload validation with magic byte checking
- WebSocket message limits (64KB), debate timeouts, backpressure control

See [docs/enterprise/SECURITY.md](docs/enterprise/SECURITY.md) and [docs/enterprise/COMPLIANCE.md](docs/enterprise/COMPLIANCE.md).

---

## Documentation

| Need | Where |
|---|---|
| Developer quickstart | [QUICKSTART_DEVELOPER.md](docs/QUICKSTART_DEVELOPER.md) |
| First-time setup | [GETTING_STARTED.md](docs/guides/GETTING_STARTED.md) |
| API reference | [API_REFERENCE.md](docs/api/API_REFERENCE.md) |
| SDK guide | [SDK_GUIDE.md](docs/SDK_GUIDE.md) |
| Enterprise features | [ENTERPRISE_FEATURES.md](docs/enterprise/ENTERPRISE_FEATURES.md) |
| Gauntlet guide | [GAUNTLET.md](docs/debate/GAUNTLET.md) |
| Agent catalog | [AGENTS.md](docs/debate/AGENTS.md) |
| Feature discovery | [FEATURE_DISCOVERY.md](docs/FEATURE_DISCOVERY.md) |
| Extended README | [EXTENDED_README.md](docs/EXTENDED_README.md) |
| Full index | [INDEX.md](docs/reference/INDEX.md) |

---

## Inspiration and Citations

Aragora synthesizes ideas from these open-source projects:

- **[Stanford Generative Agents](https://github.com/joonspk-research/generative_agents)** -- Memory + reflection architecture
- **[ChatArena](https://github.com/chatarena/chatarena)** -- Multi-agent interaction environments
- **[LLM Multi-Agent Debate](https://github.com/composable-models/llm_multiagent_debate)** -- ICML 2024 consensus mechanisms
- **[ai-counsel](https://github.com/AI-Counsel/ai-counsel)** -- Semantic convergence detection (MIT)
- **[DebateLLM](https://github.com/Tsinghua-MARS-Lab/DebateLLM)** -- Agreement intensity modulation (Apache 2.0)
- **[claude-flow](https://github.com/ruvnet/claude-flow)** -- Adaptive topology switching (MIT)
- **[LLM Reasoning Failures](https://arxiv.org/abs/2602.06176)** -- Stanford taxonomy of systematic reasoning breakdowns (Song et al. 2026)

See the full attribution table in [docs/reference/CREDITS.md](docs/reference/CREDITS.md).

---

## Contributing

Contributions welcome. Areas of interest:

- Additional agent backends
- Debate visualization
- Benchmark datasets for agent evaluation
- Lean 4 theorem proving integration

## License

MIT

---

*The name "aragora" evokes the Greek agora -- the public assembly where citizens debated and reached collective decisions through reasoned discourse.*
