"""
Modular HTTP request handlers for the unified server.

Each module handles a specific domain of endpoints:
- debates: Debate history and management
- agents: Agent profiles, rankings, and metrics
- system: Health checks, nomic state, modes

Usage:
    from aragora.server.handlers import DebatesHandler, AgentsHandler, SystemHandler

    # Create handlers with server context
    ctx = {"storage": storage, "elo_system": elo, "nomic_dir": nomic_dir}
    debates = DebatesHandler(ctx)
    agents = AgentsHandler(ctx)
    system = SystemHandler(ctx)

    # Handle requests
    if debates.can_handle(path):
        result = debates.handle(path, query_params, handler)
"""

from .base import HandlerResult, BaseHandler, json_response, error_response
from .debates import DebatesHandler
from .agents import AgentsHandler
from .system import SystemHandler

__all__ = [
    # Base utilities
    "HandlerResult",
    "BaseHandler",
    "json_response",
    "error_response",
    # Handlers
    "DebatesHandler",
    "AgentsHandler",
    "SystemHandler",
]
