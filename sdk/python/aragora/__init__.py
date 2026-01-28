"""
Aragora Python SDK

Official Python client for the Aragora multi-agent debate platform.
"""

from .client import AragoraAsyncClient, AragoraClient
from .exceptions import AragoraError, AuthenticationError, RateLimitError, ValidationError
from .generated_types import (
    Agent,
    # Core types
    Debate,
    DebateCreateRequest,
    DebateCreateResponse,
    # Gauntlet types
    DecisionReceipt,
    Message,
    # Workflow types
    Workflow,
    WorkflowTemplate,
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
]
