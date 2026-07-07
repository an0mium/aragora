---
title: Quickstart
description: Run your first Aragora debate and receipt path in under a minute.
---

# Quickstart

Get from zero to a working adversarial debate in under a minute.

## 1. Install

```bash
pip install aragora-debate
```

## 2. Run the Zero-Key Demo

No API keys are required. The offline demo runs a complete adversarial debate
with mock agents:

```bash
python3 -m aragora_debate
```

You will see three agents propose, critique each other, vote, reach consensus,
and produce an audit-ready decision receipt with a SHA-256 verdict hash.

If you installed the full platform instead with `pip install aragora`, the
equivalent zero-key demo is:

```bash
aragora demo
```

## 3. Run a Three-Line Debate

```python
import asyncio

from aragora_debate.arena import Arena
from aragora_debate.styled_mock import StyledMockAgent

agents = [
    StyledMockAgent("analyst", style="supportive"),
    StyledMockAgent("critic", style="critical"),
    StyledMockAgent("pm", style="balanced"),
]
arena = Arena(question="Should we adopt GraphQL?", agents=agents)
result = asyncio.run(arena.run())
print(result.receipt.to_markdown())
```

## 4. Add Real Models

Set at least one provider key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

Then run a real debate:

```python
import asyncio

from aragora import Arena, DebateProtocol, Environment

env = Environment(task="Design a rate limiter for our API")
protocol = DebateProtocol(rounds=3, consensus="majority")

arena = Arena(env, protocol=protocol)
result = asyncio.run(arena.run())
print(result.summary)
```

## 5. Wire the Public Utility Path

The core loop is: run a debate, get a receipt, verify it independently, then
wire review receipts into CI.

| Next step | Guide |
|-----------|-------|
| Understand the receipt model | [Receipt Lineage Reconciliation](../specs/receipt-lineage-reconciliation) |
| Verify a receipt without installing Aragora | [Independent Verifier Guide](../specs/independent-verifier-guide) |
| Add multi-model CI review and receipts | [GitHub Action Setup](../guides/github-actions-review) |
| See every CLI command | [CLI Reference](../api/cli) |
