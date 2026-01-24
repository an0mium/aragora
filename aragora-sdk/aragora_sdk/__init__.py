"""
Aragora SDK - Python client for the Aragora multi-agent debate platform.

.. deprecated::
    This package is deprecated. Use `aragora-client` instead:
    ``pip install aragora-client``

Basic Usage:
    from aragora_sdk import AragoraClient

    async with AragoraClient(api_key="ara_...") as client:
        result = await client.review(
            spec="Your design document here...",
            personas=["security", "sox", "hipaa"],
            rounds=3,
        )
        print(result.consensus.status)
        print(result.dissenting_opinions)

One-off Review:
    from aragora_sdk import review

    result = await review(
        spec="Your design document...",
        personas=["sox", "pci_dss"],
    )

Streaming:
    async for event in client.review_stream(spec, personas=["security"]):
        print(event)

Available Compliance Personas:
    - sox: Sarbanes-Oxley financial controls
    - pci_dss: Payment Card Industry Data Security Standard
    - hipaa: Health Insurance Portability and Accountability Act
    - gdpr: General Data Protection Regulation
    - fda_21_cfr: FDA 21 CFR Part 11 (electronic records)
    - fisma: Federal Information Security Management Act
    - finra: Financial Industry Regulatory Authority

Technical Personas:
    - security: General security review
    - performance: Performance and scalability
    - architecture: System architecture
    - testing: Test coverage and quality
"""

import warnings

warnings.warn(
    "aragora-sdk is deprecated. Please use aragora-client instead: " "pip install aragora-client",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = "2.1.13"

from .client import AragoraClient, review
from .exceptions import (
    AragoraError,
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from .models import (
    Agent,
    ConsensusResult,
    ConsensusStatus,
    Critique,
    DebateStatus,
    DecisionReceipt,
    DissentingOpinion,
    HealthStatus,
    Position,
    ReviewRequest,
    ReviewResult,
    UsageInfo,
    Vote,
)

__all__ = [
    # Version
    "__version__",
    # Client
    "AragoraClient",
    "review",
    # Models
    "Agent",
    "ConsensusResult",
    "ConsensusStatus",
    "Critique",
    "DebateStatus",
    "DecisionReceipt",
    "DissentingOpinion",
    "HealthStatus",
    "Position",
    "ReviewRequest",
    "ReviewResult",
    "UsageInfo",
    "Vote",
    # Exceptions
    "AragoraError",
    "AuthenticationError",
    "AuthorizationError",
    "ConnectionError",
    "NotFoundError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ValidationError",
]
