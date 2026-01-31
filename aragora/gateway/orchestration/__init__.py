"""
Gateway Orchestration - Task routing and fallback logic.

Provides intelligent routing of tasks to appropriate agents:
- Capability-based routing (match task needs to agent capabilities)
- Cost-based routing (optimize for cost efficiency)
- Latency-based routing (optimize for speed)
- Fallback chains (graceful degradation)
"""

from aragora.gateway.orchestration.router import (
    TaskRouter,
    RoutingStrategy,
    RoutingDecision,
)
from aragora.gateway.orchestration.fallback import (
    FallbackChain,
    FallbackResult,
)

__all__ = [
    "TaskRouter",
    "RoutingStrategy",
    "RoutingDecision",
    "FallbackChain",
    "FallbackResult",
]
