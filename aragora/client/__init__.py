"""
Aragora Python SDK.

A type-safe client library for the Aragora API.

Quick Start:
    from aragora.client import AragoraClient

    # Connect to local server
    client = AragoraClient(base_url="http://localhost:8080")

    # Run a debate synchronously
    debate = client.debates.run(
        task="Should we use microservices or monolith?",
        agents=["anthropic-api", "openai-api"],
        rounds=3,
    )

    print(f"Consensus reached: {debate.consensus.reached}")
    print(f"Final answer: {debate.consensus.final_answer}")

Async Usage:
    import asyncio
    from aragora.client import AragoraClient

    async def main():
        async with AragoraClient() as client:
            debate = await client.debates.run_async(
                task="React vs Vue vs Svelte?",
                agents=["anthropic-api", "gemini"],
            )
            print(debate.consensus.final_answer)

    asyncio.run(main())

Gauntlet Usage:
    # Stress-test a policy document
    receipt = client.gauntlet.run_and_wait(
        input_content=open("policy.md").read(),
        input_type="policy",
        persona="gdpr",
        profile="thorough",
    )

    print(f"Verdict: {receipt.verdict}")
    for finding in receipt.findings:
        print(f"  [{finding.severity}] {finding.title}")
"""

from .client import (
    AragoraClient,
    AragoraAPIError,
    DebatesAPI,
    AgentsAPI,
    LeaderboardAPI,
    GauntletAPI,
)
from .models import (
    # Enums
    DebateStatus,
    ConsensusType,
    GauntletVerdict,
    # Debate models
    Debate,
    DebateRound,
    DebateCreateRequest,
    DebateCreateResponse,
    AgentMessage,
    Vote,
    ConsensusResult,
    # Agent models
    AgentProfile,
    LeaderboardEntry,
    # Gauntlet models
    GauntletReceipt,
    GauntletRunRequest,
    GauntletRunResponse,
    Finding,
    # System models
    HealthCheck,
    APIError,
)

__all__ = [
    # Client
    "AragoraClient",
    "AragoraAPIError",
    # API interfaces
    "DebatesAPI",
    "AgentsAPI",
    "LeaderboardAPI",
    "GauntletAPI",
    # Enums
    "DebateStatus",
    "ConsensusType",
    "GauntletVerdict",
    # Debate models
    "Debate",
    "DebateRound",
    "DebateCreateRequest",
    "DebateCreateResponse",
    "AgentMessage",
    "Vote",
    "ConsensusResult",
    # Agent models
    "AgentProfile",
    "LeaderboardEntry",
    # Gauntlet models
    "GauntletReceipt",
    "GauntletRunRequest",
    "GauntletRunResponse",
    "Finding",
    # System models
    "HealthCheck",
    "APIError",
]
