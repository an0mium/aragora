#!/usr/bin/env python3
"""Integration test: Run a real multi-agent debate and generate a receipt.

Usage:
    # With real API keys:
    ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-... python scripts/integration_test.py

    # Demo mode (no API keys needed):
    python scripts/integration_test.py --demo
"""

from __future__ import annotations

# Suppress noisy deprecation warnings *before* any library imports.
import warnings  # isort: skip

warnings.simplefilter("ignore")

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# Ensure the project root is on sys.path so `aragora` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Keep library logs quiet so the script output stays clean.
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logging.getLogger("aragora").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

QUESTION = "What are the three most important principles for designing a reliable API?"


def _has_api_keys() -> dict[str, str]:
    """Return dict of available provider -> key."""
    keys: dict[str, str] = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        keys["anthropic"] = os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        keys["openai"] = os.environ["OPENAI_API_KEY"]
    return keys


def _build_agents(mode: str, keys: dict[str, str]) -> list:
    """Create two agents based on available mode/keys."""
    if mode == "demo":
        from aragora.agents.demo_agent import DemoAgent

        return [
            DemoAgent(name="demo-alpha", role="proposer"),
            DemoAgent(name="demo-beta", role="critic"),
        ]

    agents = []

    if "anthropic" in keys:
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agents.append(
            AnthropicAPIAgent(
                name="claude",
                model="claude-sonnet-4-20250514",
                role="proposer",
                api_key=keys["anthropic"],
                enable_fallback=False,
            )
        )

    if "openai" in keys:
        from aragora.agents.api_agents.openai import OpenAIAPIAgent

        agents.append(
            OpenAIAPIAgent(
                name="gpt",
                model="gpt-4o-mini",
                role="critic",
                api_key=keys["openai"],
                enable_fallback=False,
            )
        )

    # If only one provider key, add a demo agent as the second participant.
    if len(agents) == 1:
        from aragora.agents.demo_agent import DemoAgent

        role = "critic" if agents[0].role == "proposer" else "proposer"
        agents.append(DemoAgent(name="demo-fallback", role=role))

    return agents


async def run_debate(agents: list) -> "DebateResult":  # noqa: F821
    """Run a minimal 2-round debate and return the result."""
    from aragora.core_types import Environment
    from aragora.debate.orchestrator import Arena
    from aragora.debate.protocol import DebateProtocol

    env = Environment(task=QUESTION, max_rounds=2)
    protocol = DebateProtocol(
        rounds=2,
        consensus="majority",
        use_structured_phases=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            auto_create_knowledge_mound=False,
            enable_knowledge_retrieval=False,
            enable_knowledge_ingestion=False,
            enable_cross_debate_memory=False,
            enable_ml_delegation=False,
            enable_quality_gates=False,
            enable_consensus_estimation=False,
            enable_performance_monitor=False,
            enable_agent_hierarchy=False,
        )
        return await arena.run()


def generate_receipt(result) -> dict:
    """Build a DecisionReceipt from the debate result and return its dict."""
    from aragora.export.decision_receipt import DecisionReceipt

    receipt = DecisionReceipt.from_debate_result(result)
    return receipt.to_dict()


def save_receipt(receipt_dict: dict) -> Path:
    """Save receipt JSON to examples/sample_receipt.json."""
    out = PROJECT_ROOT / "examples" / "sample_receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt_dict, indent=2, default=str) + "\n")
    return out


def print_summary(result, receipt_dict: dict, elapsed: float, agents: list) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("  ARAGORA INTEGRATION TEST - RESULTS")
    print("=" * 60)
    print(f"  Question : {QUESTION}")
    print(f"  Agents   : {', '.join(a.name for a in agents)}")
    print(f"  Rounds   : {result.rounds_used}")
    agreement = min(getattr(result, "confidence", 0.0), 1.0)
    print(f"  Agreement: {agreement:.0%}")
    print(f"  Verdict  : {receipt_dict['verdict']}")
    print(f"  Receipt  : {receipt_dict['receipt_id']}")
    print(f"  Checksum : {receipt_dict['checksum']}")
    print(f"  Time     : {elapsed:.1f}s")
    print("=" * 60)
    if result.final_answer:
        preview = result.final_answer[:200]
        if len(result.final_answer) > 200:
            preview += "..."
        print(f"\n  Answer preview:\n  {preview}\n")


async def main() -> int:
    demo_mode = "--demo" in sys.argv
    keys = _has_api_keys()

    if not demo_mode and not keys:
        print("No API keys found (ANTHROPIC_API_KEY / OPENAI_API_KEY).")
        print("Run with --demo for offline mode, or export at least one key.")
        return 0  # Not an error -- just nothing to test.

    mode = "demo" if demo_mode else "live"
    agents = _build_agents(mode, keys)
    agent_names = ", ".join(a.name for a in agents)
    print(f"[integration_test] mode={mode}  agents=[{agent_names}]")

    t0 = time.monotonic()
    try:
        result = await run_debate(agents)
    except Exception as exc:
        print(f"[FAIL] Debate raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - t0

    try:
        receipt_dict = generate_receipt(result)
    except Exception as exc:
        print(f"[FAIL] Receipt generation raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out_path = save_receipt(receipt_dict)
    print_summary(result, receipt_dict, elapsed, agents)
    print(f"  Receipt saved to {out_path.relative_to(PROJECT_ROOT)}")
    print("\n[PASS] Integration test succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
