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
    from pathlib import Path

    # Stress-test a policy document
    receipt = client.gauntlet.run_and_wait(
        input_content=Path("policy.md").read_text(),
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
    GraphDebatesAPI,
    MatrixDebatesAPI,
    VerificationAPI,
    MemoryAPI,
    ReplayAPI,
)
from .websocket import (
    DebateStream,
    DebateEvent,
    DebateEventType,
    WebSocketOptions,
    stream_debate,
)
from .models import (
    # Enums
    DebateStatus,
    ConsensusType,
    GauntletVerdict,
    VerificationStatus,
    VerificationBackend,
    # Debate models
    Debate,
    DebateRound,
    DebateCreateRequest,
    DebateCreateResponse,
    AgentMessage,
    Vote,
    ConsensusResult,
    # Graph debate models
    GraphDebate,
    GraphDebateNode,
    GraphDebateBranch,
    GraphDebateCreateRequest,
    GraphDebateCreateResponse,
    # Matrix debate models
    MatrixDebate,
    MatrixScenario,
    MatrixScenarioResult,
    MatrixConclusion,
    MatrixDebateCreateRequest,
    MatrixDebateCreateResponse,
    # Verification models
    VerifyClaimRequest,
    VerifyClaimResponse,
    VerifyStatusResponse,
    VerificationBackendStatus,
    # Memory models
    MemoryTierStats,
    MemoryRecommendation,
    MemoryAnalyticsResponse,
    MemorySnapshotResponse,
    # Replay models
    Replay,
    ReplaySummary,
    ReplayEvent,
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
    "GraphDebatesAPI",
    "MatrixDebatesAPI",
    "VerificationAPI",
    "MemoryAPI",
    "ReplayAPI",
    # WebSocket
    "DebateStream",
    "DebateEvent",
    "DebateEventType",
    "WebSocketOptions",
    "stream_debate",
    # Enums
    "DebateStatus",
    "ConsensusType",
    "GauntletVerdict",
    "VerificationStatus",
    "VerificationBackend",
    # Debate models
    "Debate",
    "DebateRound",
    "DebateCreateRequest",
    "DebateCreateResponse",
    "AgentMessage",
    "Vote",
    "ConsensusResult",
    # Graph debate models
    "GraphDebate",
    "GraphDebateNode",
    "GraphDebateBranch",
    "GraphDebateCreateRequest",
    "GraphDebateCreateResponse",
    # Matrix debate models
    "MatrixDebate",
    "MatrixScenario",
    "MatrixScenarioResult",
    "MatrixConclusion",
    "MatrixDebateCreateRequest",
    "MatrixDebateCreateResponse",
    # Verification models
    "VerifyClaimRequest",
    "VerifyClaimResponse",
    "VerifyStatusResponse",
    "VerificationBackendStatus",
    # Memory models
    "MemoryTierStats",
    "MemoryRecommendation",
    "MemoryAnalyticsResponse",
    "MemorySnapshotResponse",
    # Replay models
    "Replay",
    "ReplaySummary",
    "ReplayEvent",
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
