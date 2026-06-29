# Aragora

**Aragora is an auditable execution control plane for AI-assisted decisions:
multi-model review in, a verifiable Decision Receipt out.**

It coordinates heterogeneous models to adversarially review a change or a
decision, preserves the dissent and provenance, stops truthfully when evidence
is thin, and emits a portable receipt anyone can verify offline with the
standalone verifier. PyPI publishing for the verifier is pending.

[![PyPI](https://img.shields.io/pypi/v/aragora)](https://pypi.org/project/aragora/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **New here?** The [Quickstart](docs/quickstart.md) gets you a working debate in
> under a minute. Auditors should start with the [Cold Reviewer Guide](docs/COLD_REVIEWER_GUIDE.md).

| I want to… | Command |
|------------|---------|
| Run the standalone debate engine | `pip install aragora-debate` |
| Verify a Decision Receipt with the standalone verifier | `PYTHONPATH=src python -m aragora_verify <receipt>` from `aragora-verify/`; PyPI publish pending |
| Call the Aragora API from Python | `pip install aragora-sdk` |
| Self-host the full platform | `docker compose -f deploy/demo/docker-compose.yml up` |

## The problem

Individual LLMs are unreliable. Their personas shift with context, their
confidence does not correlate with accuracy, and they often optimize for
plausible agreement instead of truth. Aragora treats that as a systems problem:
it makes consequential AI-assisted decisions **inspectable and verifiable**
instead of asking you to trust one model's say-so.

- **Disagreement becomes evidence.** Heterogeneous models challenge each other before work advances; dissent is preserved, not averaged away.
- **Every decision has a receipt.** Verdict, the reviewing models and their independence, dissent, confidence, and provenance stay inspectable.
- **It stops truthfully.** When the quorum can't be formed or evidence is thin, the receipt says so — it never fabricates a consensus.
- **Receipts are portable and verifiable.** A receipt is a schema-conformant artifact (the [Open Decision Receipt](docs/specs/OPEN_DECISION_RECEIPT.md)) that `aragora-verify` checks offline, with no dependency on Aragora.

## The wedge: a governance gate for AI-written code

Drop Aragora into CI. A multi-model quorum reviews each PR and posts a grounded
PR comment — your second opinion, with the same review surface that feeds the
Decision Receipt path.

```yaml
# .github/workflows/aragora-review.yml
name: Aragora Review
on:
  pull_request:
    types: [opened, synchronize, reopened]
permissions:
  contents: read
  pull-requests: write
  issues: write
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: synaptent/aragora@v2.9.0
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          post-comment: 'true'
```

The action posts a PR review comment and uploads the machine-readable review
artifact. When a Decision Receipt artifact exists, anyone — a teammate, an
auditor, a customer — can verify it independently with the standalone
`aragora-verify` package (no Aragora dependency):

```bash
# PyPI publish pending; today it lives in this repo under aragora-verify/:
cd aragora-verify
PYTHONPATH=src python -m aragora_verify ../decision-receipt.odr.json

# After PyPI publish:
# aragora-verify decision-receipt.odr.json
```

## Try it now

```bash
pip install aragora
aragora demo --offline              # zero-key debate, writes a local receipt

export ANTHROPIC_API_KEY=...        # provider credential for live model review
aragora review-pr 123               # multi-agent review of a GitHub PR
aragora receipt export <id> --format odr -o receipt.odr.json   # portable receipt
```

## Core workflows

- **AI code review** — heterogeneous-model review of a diff or PR, with severity-tagged findings and a receipt. See [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md).
- **Gauntlet** — adversarial stress-testing of a claim or spec; attack/defend cycles produce a cryptographic receipt.
- **Structured debates** — multi-round debate with consensus detection and convergence tracking (`aragora ask`).

## The load-bearing core

Aragora is large, but five modules carry the product. Start here:

| Module | Responsibility |
|--------|----------------|
| `aragora/debate/` | The Arena orchestrator — runs rounds, detects consensus/convergence. |
| `aragora/agents/` | Agent implementations (API + CLI), heterogeneous model transport, fallback. |
| `aragora/gauntlet/` | Decision Receipts: the native record, the portable [ODR](docs/specs/OPEN_DECISION_RECEIPT.md), export and signing. |
| `aragora/swarm/` | The merge-quorum gate — collects model-review evidence and tiers settlement. |
| `aragora/server/` | The HTTP/WebSocket API and handlers. |

`aragora-verify/` is a separate verifier package with no Aragora dependency:
the public verifier for receipts. Everything else under `aragora/` is
supporting or experimental surface — treat it as such until it's documented
here.

## Product boundary

Aragora is a **governance and review layer**, not an execution runtime. It is
not a replacement for worker runtimes like Codex, Claude Code, or OpenCode. Use
it when review, provenance, and a verifiable decision record matter; keep your
existing runtimes when raw speed is all you need.

- We do not sell lights-out autonomy as the default story.
- We do not advance work without evidence, review, and clear terminal states.
- Consequential effectors are denied by default unless an admin-scoped approval artifact exists; sandboxed backends are mandatory for browser/host effectors.

See [Boundaries and Scope](docs/strategy/BOUNDARIES_AND_SCOPE.md) for the full non-goals ledger.

## Documentation

- [Quickstart](docs/quickstart.md) · [Cold Reviewer Guide](docs/COLD_REVIEWER_GUIDE.md) · [CLI Reference](docs/CLI_REFERENCE.md)
- [Open Decision Receipt spec](docs/specs/OPEN_DECISION_RECEIPT.md) · [SDK Guide](docs/SDK_GUIDE.md) · [API Reference](docs/api/API_REFERENCE.md)
- [Feature status](docs/STATUS.md) · [Enterprise features](docs/enterprise/ENTERPRISE_FEATURES.md) · [Architecture deep-dive](docs/EXTENDED_README.md)
- [Inspiration and credits](docs/reference/CREDITS.md)

## Security

Secrets load from AWS Secrets Manager in production (never standing env keys);
local development uses a gitignored `.env`. See the
[security overview](docs/enterprise/SECURITY.md),
[compliance overview](docs/enterprise/COMPLIANCE.md), and
[deployment guide](docs/deployment/DEPLOYMENT.md).

## Contributing & License

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed
(see [LICENSE](LICENSE)).
