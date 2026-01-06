"""
Modular HTTP request handlers for the unified server.

Each module handles a specific domain of endpoints:
- debates: Debate history and management
- agents: Agent profiles, rankings, and metrics
- system: Health checks, nomic state, modes
- pulse: Trending topics from multiple sources
- analytics: Aggregated metrics and statistics
- consensus: Consensus memory and dissent tracking

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
from .pulse import PulseHandler
from .analytics import AnalyticsHandler
from .metrics import MetricsHandler
from .consensus import ConsensusHandler
from .belief import BeliefHandler
from .critique import CritiqueHandler
from .genesis import GenesisHandler
from .replays import ReplaysHandler
from .tournaments import TournamentHandler
from .memory import MemoryHandler
from .leaderboard import LeaderboardViewHandler
from .relationship import RelationshipHandler
from .moments import MomentsHandler
from .documents import DocumentHandler
from .verification import VerificationHandler
from .auditing import AuditingHandler
from .dashboard import DashboardHandler

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
    "PulseHandler",
    "AnalyticsHandler",
    "MetricsHandler",
    "ConsensusHandler",
    "BeliefHandler",
    "CritiqueHandler",
    "GenesisHandler",
    "ReplaysHandler",
    "TournamentHandler",
    "MemoryHandler",
    "LeaderboardViewHandler",
    "RelationshipHandler",
    "MomentsHandler",
    "DocumentHandler",
    "VerificationHandler",
    "AuditingHandler",
    "DashboardHandler",
]
