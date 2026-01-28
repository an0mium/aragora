"""
Aragora Python SDK

Official Python client for the Aragora multi-agent debate platform.
"""

from .client import AragoraClient, AragoraAsyncClient
from .exceptions import AragoraError, AuthenticationError, RateLimitError, ValidationError
from .generated_types import (
    # Core types
    Debate,
    DebateCreateRequest,
    DebateCreateResponse,
    Message,
    Agent,
    # Workflow types
    Workflow,
    WorkflowTemplate,
    # Gauntlet types
    DecisionReceipt,
    GauntletResult,
    # Export all generated types
)

__version__ = "0.1.0"
__all__ = [
    # Client
    "AragoraClient",
    "AragoraAsyncClient",
    # Exceptions
    "AragoraError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    # Core types
    "Debate",
    "DebateCreateRequest",
    "DebateCreateResponse",
    "Message",
    "Agent",
    "Workflow",
    "WorkflowTemplate",
    "DecisionReceipt",
    "GauntletResult",
]
