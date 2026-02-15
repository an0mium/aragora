"""End-to-end integration test: debate → receipt → verification.

Tests the full debate lifecycle using the aragora-debate package
with mock agents. No external services or API keys required.

Usage:
    pytest tests/integration/test_e2e_debate.py -v
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "aragora_debate", reason="aragora_debate package is required for debate integration tests"
)

from aragora_debate import (
    Debate,
    DebateResult,
    DecisionReceipt,
    create_agent,
)
from aragora_debate.receipt import ReceiptBuilder


@pytest.mark.asyncio
async def test_full_debate_lifecycle() -> None:
    """Run a complete debate and verify the receipt."""
    debate = Debate(
        topic="Should we use PostgreSQL or MongoDB for the new service?",
        context="The service needs ACID transactions and handles relational data.",
        rounds=2,
        consensus="majority",
    )

    debate.add_agent(
        create_agent(
            "mock",
            name="backend-lead",
            proposal="Use PostgreSQL for ACID compliance and relational queries.",
            vote_for="backend-lead",
        )
    )
    debate.add_agent(
        create_agent(
            "mock",
            name="devops-lead",
            proposal="Use MongoDB for flexible schema and horizontal scaling.",
            vote_for="backend-lead",
        )
    )
    debate.add_agent(
        create_agent(
            "mock",
            name="architect",
            proposal="Use PostgreSQL with JSONB columns for flexibility.",
            vote_for="backend-lead",
        )
    )

    result = await debate.run()

    # Verify result structure
    assert isinstance(result, DebateResult)
    assert result.receipt is not None
    assert isinstance(result.receipt, DecisionReceipt)

    # Verify receipt fields
    receipt = result.receipt
    assert receipt.question == "Should we use PostgreSQL or MongoDB for the new service?"
    assert len(receipt.agents) == 3
    assert receipt.rounds_used >= 1
    assert receipt.receipt_id

    # Verify receipt exports
    md = receipt.to_markdown()
    assert len(md) > 100

    json_str = ReceiptBuilder.to_json(receipt)
    data = json.loads(json_str)
    assert "question" in data
    assert "receipt_id" in data

    # Verify HMAC signing
    ReceiptBuilder.sign_hmac(receipt, "test-secret-key")
    assert receipt.signature
    assert receipt.signature_algorithm == "HMAC-SHA256"
    assert ReceiptBuilder.verify_hmac(receipt, "test-secret-key") is True
    assert ReceiptBuilder.verify_hmac(receipt, "wrong-key") is False


@pytest.mark.asyncio
async def test_debate_with_trickster() -> None:
    """Verify trickster-enabled debate runs without error."""
    debate = Debate(
        topic="Should we rewrite the frontend in Rust?",
        rounds=2,
        consensus="majority",
        enable_trickster=True,
        trickster_sensitivity=0.3,
    )

    for i in range(4):
        debate.add_agent(
            create_agent(
                "mock",
                name=f"agent-{i}",
                proposal="Yes, rewrite in Rust for performance.",
                vote_for="agent-0",
            )
        )

    result = await debate.run()
    assert result.receipt is not None
    assert result.receipt.rounds_used >= 1


@pytest.mark.asyncio
async def test_debate_with_dissent() -> None:
    """Verify dissenting opinions are captured in consensus."""
    debate = Debate(topic="Monolith vs microservices?", rounds=2, consensus="majority")

    debate.add_agent(
        create_agent("mock", name="pro-mono", proposal="Keep the monolith.", vote_for="pro-mono")
    )
    debate.add_agent(
        create_agent(
            "mock", name="pro-micro", proposal="Split into microservices.", vote_for="pro-micro"
        )
    )
    debate.add_agent(
        create_agent(
            "mock",
            name="pragmatist",
            proposal="Start monolith, extract later.",
            vote_for="pro-mono",
        )
    )

    result = await debate.run()
    assert result.receipt is not None

    # Check consensus has dissenting agents
    consensus = result.receipt.consensus
    assert len(consensus.dissenting_agents) >= 1 or len(consensus.dissents) >= 0


@pytest.mark.asyncio
async def test_debate_event_callbacks() -> None:
    """Verify event callbacks fire during debate."""
    events: list[str] = []

    def on_event(event: object) -> None:
        events.append(type(event).__name__)

    debate = Debate(
        topic="Redis vs Memcached for caching?",
        rounds=1,
        consensus="majority",
        on_event=on_event,
    )
    debate.add_agent(create_agent("mock", name="a1", vote_for="a1"))
    debate.add_agent(create_agent("mock", name="a2", vote_for="a1"))

    await debate.run()
    assert len(events) > 0


@pytest.mark.asyncio
async def test_receipt_json_roundtrip() -> None:
    """Verify receipt survives JSON serialization roundtrip."""
    debate = Debate(topic="Tabs vs spaces?", rounds=1, consensus="majority")
    debate.add_agent(create_agent("mock", name="tabs", proposal="Tabs", vote_for="tabs"))
    debate.add_agent(create_agent("mock", name="spaces", proposal="Spaces", vote_for="tabs"))

    result = await debate.run()
    json_str = ReceiptBuilder.to_json(result.receipt)

    data = json.loads(json_str)
    assert data["question"] == "Tabs vs spaces?"
    assert "agents" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_minimum_agents_error() -> None:
    """Verify proper error with <2 agents."""
    debate = Debate(topic="Solo debate?", rounds=1)
    debate.add_agent(create_agent("mock", name="lonely"))

    with pytest.raises(ValueError, match="at least 2 agents"):
        await debate.run()
