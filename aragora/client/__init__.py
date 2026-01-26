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
    AgentsAPI,
    AragoraClient,
    DebatesAPI,
    GauntletAPI,
    GraphDebatesAPI,
    LeaderboardAPI,
    MatrixDebatesAPI,
    MemoryAPI,
    OrganizationsAPI,
    ReplayAPI,
    VerificationAPI,
)
from .errors import (
    AragoraAPIError,
    AuthenticationError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
)
from .transport import RateLimiter, RetryConfig
from .models import (
    AgentMessage,
    # Agent models
    AgentProfile,
    APIError,
    ConsensusResult,
    ConsensusType,
    # Debate models
    Debate,
    DebateCreateRequest,
    DebateCreateResponse,
    DebateRound,
    # Enums
    DebateStatus,
    Finding,
    # Gauntlet models
    GauntletReceipt,
    GauntletRunRequest,
    GauntletRunResponse,
    GauntletVerdict,
    # Graph debate models
    GraphDebate,
    GraphDebateBranch,
    GraphDebateCreateRequest,
    GraphDebateCreateResponse,
    GraphDebateNode,
    # System models
    HealthCheck,
    LeaderboardEntry,
    MatrixConclusion,
    # Matrix debate models
    MatrixDebate,
    MatrixDebateCreateRequest,
    MatrixDebateCreateResponse,
    MatrixScenario,
    MatrixScenarioResult,
    MemoryAnalyticsResponse,
    MemoryRecommendation,
    MemorySnapshotResponse,
    # Memory models
    MemoryTierStats,
    # Replay models
    Replay,
    ReplayEvent,
    ReplaySummary,
    VerificationBackend,
    VerificationBackendStatus,
    VerificationStatus,
    # Verification models
    VerifyClaimRequest,
    VerifyClaimResponse,
    VerifyStatusResponse,
    Vote,
)
from .websocket import (
    DebateEvent,
    DebateEventType,
    DebateStream,
    WebSocketOptions,
    stream_debate,
    stream_debate_by_id,
)

__all__ = [
    # Client
    "AragoraClient",
    "AragoraAPIError",
    "RetryConfig",
    "RateLimiter",
    # Errors
    "AuthenticationError",
    "NotFoundError",
    "QuotaExceededError",
    "RateLimitError",
    "ValidationError",
    # API interfaces
    "DebatesAPI",
    "AgentsAPI",
    "LeaderboardAPI",
    "GauntletAPI",
    "GraphDebatesAPI",
    "MatrixDebatesAPI",
    "VerificationAPI",
    "MemoryAPI",
    "OrganizationsAPI",
    "ReplayAPI",
    # WebSocket
    "DebateStream",
    "DebateEvent",
    "DebateEventType",
    "WebSocketOptions",
    "stream_debate",
    "stream_debate_by_id",
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
