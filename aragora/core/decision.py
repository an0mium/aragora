"""
Unified Decision Request Schema and Router.

Provides a single entry point for all decision-making requests across
Aragora, normalizing inputs from HTTP, WebSocket, chat, voice, and email
channels before routing to the appropriate decision engine.

Usage:
    from aragora.core.decision import DecisionRequest, DecisionRouter

    # Create a unified request
    request = DecisionRequest(
        content="Should we use microservices?",
        decision_type=DecisionType.DEBATE,
        source=InputSource.SLACK,
        response_channel=ResponseChannel(platform="slack", channel_id="C123"),
    )

    # Route to appropriate engine
    router = DecisionRouter()
    result = await router.route(request)

This module has been decomposed for maintainability:
- decision_types.py: Enums (DecisionType, InputSource, Priority, ResponseFormat)
- decision_models.py: Dataclasses (DecisionRequest, DecisionResult, etc.)
- decision_router.py: DecisionRouter class and singleton management
"""

from __future__ import annotations

# Re-export all types for backward compatibility
from .decision_types import (
    DecisionType,
    InputSource,
    Priority,
    ResponseFormat,
    _DEFAULT_DECISION_CONSENSUS,
    _DEFAULT_DECISION_MAX_AGENTS,
    _DEFAULT_DECISION_ROUNDS,
    _default_decision_agents,
)

from .decision_models import (
    DecisionConfig,
    DecisionRequest,
    DecisionResult,
    RequestContext,
    ResponseChannel,
)

from .decision_router import (
    DecisionRouter,
    get_decision_router,
    reset_decision_router,
)


__all__ = [
    # Enums
    "DecisionType",
    "InputSource",
    "Priority",
    "ResponseFormat",
    # Dataclasses
    "ResponseChannel",
    "RequestContext",
    "DecisionConfig",
    "DecisionRequest",
    "DecisionResult",
    # Router
    "DecisionRouter",
    "get_decision_router",
    "reset_decision_router",
    # Default settings (for internal use)
    "_DEFAULT_DECISION_ROUNDS",
    "_DEFAULT_DECISION_CONSENSUS",
    "_DEFAULT_DECISION_MAX_AGENTS",
    "_default_decision_agents",
]
