# Aragora

### The Decision Integrity Platform

[![Tests](https://github.com/an0mium/aragora/actions/workflows/test.yml/badge.svg)](https://github.com/an0mium/aragora/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Individual LLMs are unreliable. Their personas shift with context, their confidence doesn't correlate with accuracy, and they say what you want to hear. For consequential decisions, you need infrastructure that treats this as a feature to be engineered around, not a problem to be ignored.**

Aragora orchestrates 30+ agent types in structured adversarial debates -- forcing models to challenge each other's reasoning, surface blind spots, and produce decisions with complete audit trails showing where they agreed, where they disagreed, and why.

```bash
# Stress-test a specification
aragora gauntlet spec.md --profile thorough --output receipt.html

# Run a multi-agent debate
aragora ask "Design a rate limiter for 1M req/sec" --agents anthropic-api,openai-api,gemini

# Start the API server
aragora serve
```

---

## Five Pillars

Aragora is built on five architectural commitments designed for a world where individual AI agents cannot be trusted with consequential decisions alone.

### 1. SMB-Ready, Enterprise-Grade

Aragora is useful to a 5-person startup on day one and scales to regulated enterprise without rearchitecting. Enterprise features -- OIDC/SAML SSO, MFA, AES-256-GCM encryption, multi-tenant isolation, RBAC with 7 roles and 50+ permissions, SOC 2 / GDPR / HIPAA compliance frameworks -- are built in, not bolted on. Security hardening (rate limiting, SSRF protection, path traversal guards, input validation, audit trails) is the default, not a premium tier.

### 2. Leading-Edge Memory and Context

Single agents lose context. Aragora's 4-tier Continuum Memory (fast / medium / slow / glacial) and Knowledge Mound with 28 registered adapters give every debate access to institutional history, cross-session learning, and evidence provenance. The RLM (Recursive Language Models) system compresses and structures context to reduce prompt bloat, enabling debates that sustain coherence across long multi-round sessions and large document sets where individual models would degrade.

### 3. Extensible and Modular

Connectors for Slack, Teams, Discord, Telegram, WhatsApp, email, voice, Kafka, RabbitMQ, GitHub, Jira, Salesforce, healthcare HL7/FHIR, and dozens more. SDKs in Python and TypeScript (140 namespaces in the TypeScript SDK). 2,000+ API operations across 1,800+ paths and 190+ WebSocket event types. OpenClaw integration for portable agent governance. A workflow engine with DAG execution and 50+ templates. A marketplace for agent personas, debate templates, and workflow patterns. Aragora adapts to your stack -- not the other way around.

### 4. Multi-Agent Robustness

Individual LLMs exhibit persona instability -- their outputs shift based on framing, context, and even prompt ordering. Aragora treats this as a feature: by running Claude, GPT, Gemini, Grok, Mistral, DeepSeek, Qwen, Kimi, and local models in structured Propose / Critique / Revise debates, the system surfaces disagreements that reveal genuine uncertainty. ELO rankings track agent performance. Calibration scoring (Brier scores) measures prediction accuracy. The Trickster detects hollow consensus where models agree without genuine reasoning. The result: when models with different training data independently converge on an answer, that convergence is meaningful -- and when they disagree, the dissent trail tells you exactly where human judgment is needed.

### 5. Self-Healing and Self-Extending

The Nomic Loop is Aragora's autonomous self-improvement system: agents debate improvements to the codebase, design solutions, implement code, run tests, and verify changes -- with human approval gates and automatic rollback on failure. This is how Aragora grew from a debate engine to 3,000+ modules. Red-team mode stress-tests the platform's own specs. The Gauntlet runs adversarial attacks against proposed changes. The system hardens itself.

---

## Why Aragora?

A single LLM will confidently give you a wrong answer and you won't know it. Research shows that LLM personas are context-dependent, fragile under adversarial pressure, and prone to sycophantic agreement with whoever is asking. [Stanford's taxonomy of LLM reasoning failures](https://arxiv.org/abs/2602.06176) documents systematic breakdowns in formal logic, unfaithful chain-of-thought, and robustness failures under minor prompt variations -- exactly the failure modes that structured adversarial debate is designed to surface. When the decision matters -- hiring, architecture, compliance, strategy -- one model's opinion is insufficient.

Aragora treats each model as an **unreliable witness** and uses structured debate protocols to extract signal from their disagreements:

| What you get | How it works |
|---|---|
| **Adversarial Validation** | Models with different training data and blind spots challenge each other's reasoning |
| **Decision Receipts** | Cryptographic audit trails with evidence chains, dissent tracking, and confidence calibration |
| **Gauntlet Mode** | Red-team stress-tests for specs, policies, and architectures using adversarial personas |
| **Calibrated Trust** | ELO rankings and Brier scores track which models are actually reliable on which domains |
| **Institutional Memory** | Decisions persist across sessions with 4-tier memory and Knowledge Mound (28 adapters) |
| **Channel Delivery** | Results route to Slack, Teams, Discord, Telegram, WhatsApp, email, or voice |

---

## Quick Start

```bash
git clone https://github.com/an0mium/aragora.git && cd aragora
pip install -e .

# Set at least one API key
export ANTHROPIC_API_KEY=your-key  # or OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY

# Run a debate
aragora ask "Should we adopt microservices?" --agents anthropic-api,openai-api --rounds 3
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for the complete 5-minute setup.

### SDK Packages

| Package | Purpose | Install |
|---|---|---|
| `aragora` | Full control plane | `pip install aragora` |
| `aragora-sdk` | Python SDK client | `pip install aragora-sdk` |
| `@aragora/sdk` | TypeScript/Node.js SDK | `npm install @aragora/sdk` |

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
├── agents/         # 30+ registered agent types (CLI, direct API, OpenRouter, local)
│   ├── api_agents/       # Anthropic, OpenAI, Gemini, Grok, Mistral, OpenRouter
│   ├── cli_agents.py     # Claude Code, Codex, Gemini CLI, Grok CLI
│   └── fallback.py       # OpenRouter fallback on quota errors
├── gauntlet/       # Adversarial stress testing
├── knowledge/      # Knowledge Mound with 28 registered adapters
├── memory/         # 4-tier memory (fast/medium/slow/glacial)
├── server/         # 2,000+ API operations, 190+ WebSocket event types
├── pipeline/       # Decision-to-PR generation
├── genesis/        # Fractal debates, agent evolution
├── sandbox/        # Docker-based safe execution
├── rbac/           # Role-based access control (7 roles, 50+ permissions)
├── compliance/     # SOC 2, GDPR, HIPAA frameworks
└── workflow/       # DAG-based automation engine
```

**Scale:** 3,000+ Python modules | 131,000+ tests

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

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for setup.

---

## Enterprise Features

| Category | Capabilities |
|---|---|
| **Authentication** | OIDC/SAML SSO, MFA (TOTP/HOTP), API key management, SCIM 2.0 |
| **Multi-Tenancy** | Tenant isolation, resource quotas, usage metering |
| **Security** | AES-256-GCM encryption, rate limiting, SSRF protection, key rotation |
| **Compliance** | SOC 2 controls, GDPR support, HIPAA, audit trails |
| **Observability** | Prometheus metrics, Grafana dashboards, OpenTelemetry tracing |
| **RBAC** | 7 roles, 50+ permissions, decorator-based enforcement |
| **Backup** | Incremental backups, retention policies, disaster recovery |
| **Control Plane** | Agent registry, task scheduler, health monitoring, policy governance |

See [docs/ENTERPRISE_FEATURES.md](docs/ENTERPRISE_FEATURES.md) for details.

---

## Self-Improvement (Nomic Loop)

Aragora includes an autonomous self-improvement system where agents debate and implement improvements to the codebase itself. **Experimental** -- always run in a sandbox with human review.

```bash
python scripts/run_nomic_with_stream.py run --cycles 3
python scripts/self_develop.py --goal "Improve test coverage" --require-approval
```

Safety: automatic backups, protected file checksums, rollback on failure, human approval gates.

---

## Deployment

```bash
# Local development
aragora serve --api-port 8080 --ws-port 8765

# Production (AWS Lightsail + Cloudflare Tunnel)
./deploy/lightsail-setup.sh
```

**API:** REST endpoints at `/api/v2/*` | WebSocket streaming at `/ws`
**Docs:** OpenAPI at `/api/openapi` | Swagger UI at `/api/docs`

---

## Security

- Ed25519 signature verification for webhooks (Discord, Slack)
- Rate limiting (IP, token, and endpoint-based)
- Input validation and content-length enforcement
- CORS allowlists, security headers, error message sanitization
- Path traversal protection, upload validation with magic byte checking
- WebSocket message limits (64KB), debate timeouts, backpressure control

See [docs/SECURITY.md](docs/SECURITY.md) and [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

---

## Documentation

| Need | Where |
|---|---|
| First-time setup | [GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| API reference | [API_REFERENCE.md](docs/API_REFERENCE.md) |
| SDK guide | [SDK_GUIDE.md](docs/SDK_GUIDE.md) |
| Enterprise features | [ENTERPRISE_FEATURES.md](docs/ENTERPRISE_FEATURES.md) |
| Gauntlet guide | [GAUNTLET.md](docs/GAUNTLET.md) |
| Agent catalog | [AGENTS.md](docs/AGENTS.md) |
| Feature discovery | [FEATURE_DISCOVERY.md](docs/FEATURE_DISCOVERY.md) |
| Extended README | [EXTENDED_README.md](docs/EXTENDED_README.md) |
| Full index | [INDEX.md](docs/INDEX.md) |

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

See the full attribution table in [docs/CREDITS.md](docs/CREDITS.md).

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
