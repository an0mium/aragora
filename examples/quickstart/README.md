# Quickstart Examples

Beginner-friendly examples using `aragora-debate`. All examples run offline with zero API keys.

## Setup

```bash
pip install aragora-debate
```

## Examples

Run them in order -- each builds on the previous one.

### 01 - Simple Debate

Three mock agents debate whether to migrate from a monolith to microservices.

```bash
python examples/quickstart/01_simple_debate.py
```

**What you learn:** Creating a `Debate`, adding agents with `create_agent("mock", ...)`, running the debate, and reading results.

### 02 - Decision Receipt

Run a debate with `StyledMockAgent` (realistic varied responses) and produce a signed decision receipt.

```bash
python examples/quickstart/02_with_receipt.py
```

**What you learn:** Using `Arena` for more configuration, generating Markdown receipts, signing with HMAC-SHA256, and verifying signatures for tamper detection.

### 03 - Evidence Quality

Score agent responses for evidence quality and detect hollow consensus (agents agreeing without substantive evidence).

```bash
python examples/quickstart/03_evidence_quality.py
```

**What you learn:** Using `EvidenceQualityAnalyzer` to score citation density, specificity, and reasoning chains. Using `HollowConsensusDetector` to flag weak agreement.

## Using Real LLM Providers

To use real models instead of mock agents, install the provider extra and set your API key:

```bash
pip install aragora-debate[anthropic]
export ANTHROPIC_API_KEY=sk-ant-...
```

Then replace mock agents with real ones:

```python
from aragora_debate import create_agent

# Instead of: create_agent("mock", name="analyst", proposal="...")
# Use:        create_agent("anthropic", name="analyst")
```

Supported providers: `anthropic`, `openai`, `mistral`, `gemini`.

## More Examples

The existing examples in this directory also work without API keys:

- `basic_debate.py` -- Another debate example using the `Debate` and `ReceiptBuilder` APIs
- `code_review.py` -- Multi-model code review with SARIF output
- `healthcare_review.py` -- Clinical decision review with HIPAA-compliant receipt

## Next Steps

- [docs/START_HERE.md](../../docs/START_HERE.md) -- Unified onboarding guide
- [aragora-debate README](../../aragora-debate/README.md) -- Full API reference
- [examples/](../) -- Advanced examples (tournaments, TypeScript SDK, Slack bot, etc.)
