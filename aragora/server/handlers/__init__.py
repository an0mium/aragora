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

from __future__ import annotations

from aragora.config.stability import Stability

from .admin import AdminHandler
from .agents import AgentsHandler
from .analytics import AnalyticsHandler
from .auditing import AuditingHandler
from .auth import AuthHandler
from .base import BaseHandler, HandlerResult, error_response, json_response
from .belief import BeliefHandler
from .admin import BillingHandler  # Moved to admin/
from .breakpoints import BreakpointsHandler
from .features import AudioHandler  # Moved to features/
from .features import BroadcastHandler  # Moved to features/
from .agents import CalibrationHandler  # Moved to agents/
from .checkpoints import CheckpointHandler
from .consensus import ConsensusHandler
from .critique import CritiqueHandler
from .admin import DashboardHandler  # Moved to admin/
from .debates import DebatesHandler
from .docs import DocsHandler
from .features import DocumentHandler  # Moved to features/
from .features import DocumentBatchHandler  # Batch document upload
from .features import EvidenceHandler  # Moved to features/
from .features import FolderUploadHandler  # Folder upload support
from .evolution import EvolutionABTestingHandler  # Moved to evolution/
from .evolution import EvolutionHandler  # Moved to evolution/
from .features import FeaturesHandler  # Moved to features/
from .verification import FormalVerificationHandler  # Moved to verification/
from .gallery import GalleryHandler
from .gauntlet import GauntletHandler
from .genesis import GenesisHandler
from .debates import GraphDebatesHandler  # Moved to debates/
from .admin import HealthHandler  # Moved to admin/
from .memory import InsightsHandler  # Moved to memory/
from .introspection import IntrospectionHandler
from .laboratory import LaboratoryHandler
from .agents import LeaderboardViewHandler  # Moved to agents/
from .memory import LearningHandler  # Moved to memory/
from .debates import MatrixDebatesHandler  # Moved to debates/
from .memory import MemoryHandler  # memory/ subdirectory
from .memory import MemoryAnalyticsHandler  # Moved to memory/
from .metrics import MetricsHandler
from .moments import MomentsHandler
from .nomic import NomicHandler
from .oauth import OAuthHandler
from .organizations import OrganizationsHandler
from .persona import PersonaHandler
from .privacy import PrivacyHandler
from .features import PluginsHandler  # Moved to features/
from .agents import ProbesHandler  # Moved to agents/
from .features import PulseHandler  # Moved to features/
from .social import RelationshipHandler  # Moved to social/
from .replays import ReplaysHandler
from .reviews import ReviewsHandler
from .routing import RoutingHandler
from .selection import SelectionHandler
from .social import SlackHandler  # Moved to social/
from .social import SocialMediaHandler
from .admin import SystemHandler  # Moved to admin/
from .tournaments import TournamentHandler
from .verification import VerificationHandler
from .webhooks import WebhookHandler
from .social import CollaborationHandlers, get_collaboration_handlers  # Moved to social/

# List of all handler classes for automatic dispatch registration
# Order matters: more specific handlers should come first
ALL_HANDLERS = [
    GraphDebatesHandler,  # More specific path: /api/debates/graph
    MatrixDebatesHandler,  # More specific path: /api/debates/matrix
    DebatesHandler,
    AgentsHandler,
    HealthHandler,  # More specific: /healthz, /readyz, /api/health/*
    NomicHandler,  # More specific: /api/nomic/*
    DocsHandler,  # More specific: /api/openapi*, /api/docs*, /api/redoc*
    SystemHandler,
    PulseHandler,
    AnalyticsHandler,
    MetricsHandler,
    ConsensusHandler,
    BeliefHandler,
    CritiqueHandler,
    GenesisHandler,
    ReplaysHandler,
    TournamentHandler,
    MemoryHandler,
    LeaderboardViewHandler,
    RelationshipHandler,
    MomentsHandler,
    DocumentHandler,
    DocumentBatchHandler,  # Batch document upload
    FolderUploadHandler,  # Folder upload support
    VerificationHandler,
    AuditingHandler,
    DashboardHandler,
    PersonaHandler,
    IntrospectionHandler,
    CalibrationHandler,
    CheckpointHandler,
    RoutingHandler,
    SelectionHandler,  # Selection plugin API
    EvolutionABTestingHandler,  # More specific: /api/evolution/ab-tests
    EvolutionHandler,
    PluginsHandler,
    AudioHandler,
    SocialMediaHandler,
    BroadcastHandler,
    LaboratoryHandler,
    ProbesHandler,
    InsightsHandler,
    GalleryHandler,
    BreakpointsHandler,
    LearningHandler,
    AuthHandler,
    BillingHandler,
    OrganizationsHandler,
    OAuthHandler,
    FeaturesHandler,
    MemoryAnalyticsHandler,
    GauntletHandler,
    ReviewsHandler,
    FormalVerificationHandler,
    SlackHandler,
    EvidenceHandler,
    WebhookHandler,
    AdminHandler,
    PrivacyHandler,
]

# Handler stability classifications
# - STABLE: Production-ready, extensively tested, API stable
# - EXPERIMENTAL: Works but may change, use with awareness
# - PREVIEW: Early access, expect changes and potential issues
# - DEPRECATED: Being phased out, use alternative
HANDLER_STABILITY: dict[str, Stability] = {
    # Core - Stable
    "DebatesHandler": Stability.STABLE,
    "AgentsHandler": Stability.STABLE,
    "SystemHandler": Stability.STABLE,
    "HealthHandler": Stability.STABLE,  # Extracted from SystemHandler
    "NomicHandler": Stability.STABLE,  # Extracted from SystemHandler
    "DocsHandler": Stability.STABLE,  # Extracted from SystemHandler
    "AnalyticsHandler": Stability.STABLE,
    "ConsensusHandler": Stability.STABLE,
    "MetricsHandler": Stability.STABLE,
    "MemoryHandler": Stability.STABLE,
    "LeaderboardViewHandler": Stability.STABLE,
    "ReplaysHandler": Stability.STABLE,
    "FeaturesHandler": Stability.STABLE,
    "AuthHandler": Stability.STABLE,
    # Extended - Stable
    "TournamentHandler": Stability.STABLE,
    "CritiqueHandler": Stability.STABLE,
    "RelationshipHandler": Stability.STABLE,
    "DashboardHandler": Stability.STABLE,
    "RoutingHandler": Stability.STABLE,
    "SelectionHandler": Stability.STABLE,  # Selection plugin API
    # Promoted to Stable (Jan 2026) - tested in production
    "BillingHandler": Stability.STABLE,  # Transaction tests, Stripe webhooks
    "OAuthHandler": Stability.STABLE,  # OAuth flow tests, Google integration
    "AudioHandler": Stability.STABLE,  # Podcast generation, TTS
    "VerificationHandler": Stability.STABLE,  # Z3 formal verification
    "PulseHandler": Stability.STABLE,  # Trending topics API
    "GalleryHandler": Stability.STABLE,  # Consensus gallery
    "GauntletHandler": Stability.STABLE,  # Adversarial validation - 6+ test files
    "BeliefHandler": Stability.STABLE,  # Belief networks - 4 test files
    "CalibrationHandler": Stability.STABLE,  # Agent calibration - 4 test files
    "PersonaHandler": Stability.STABLE,  # Agent personas - 2 test files
    # Promoted to Stable (Jan 2026) - comprehensive test coverage
    "GraphDebatesHandler": Stability.STABLE,  # 7 test files, 95+ tests
    "MatrixDebatesHandler": Stability.STABLE,  # Handler tests + integration
    "EvolutionHandler": Stability.STABLE,  # 7 test files, 66+ tests
    "EvolutionABTestingHandler": Stability.STABLE,  # AB testing with evolution
    "LaboratoryHandler": Stability.STABLE,  # 3 test files, 70+ tests
    "IntrospectionHandler": Stability.STABLE,  # 2 test files, 53+ tests
    "LearningHandler": Stability.STABLE,  # 2 test files, 66+ tests
    "MemoryAnalyticsHandler": Stability.STABLE,  # Handler tests, 23+ tests
    "ProbesHandler": Stability.STABLE,  # 16 tests, capability probing
    "InsightsHandler": Stability.STABLE,  # 3 test files, 110+ tests
    "ReviewsHandler": Stability.STABLE,  # 18 tests, shareable code reviews
    "FormalVerificationHandler": Stability.STABLE,  # 18 tests, Z3/Lean backends
    # Promoted to Stable (Jan 2026) - from Preview
    "OrganizationsHandler": Stability.STABLE,  # 49 tests, team management
    "SocialMediaHandler": Stability.STABLE,  # 31 tests, OAuth flows
    "MomentsHandler": Stability.STABLE,  # 33 tests, moment detection
    "AuditingHandler": Stability.STABLE,  # 55 tests, audit trails
    "PluginsHandler": Stability.STABLE,  # 23 tests, plugin system
    "BroadcastHandler": Stability.STABLE,  # 65 tests, podcast generation
    "GenesisHandler": Stability.STABLE,  # 26 tests, evolution visibility
    "DocumentHandler": Stability.STABLE,  # 36 tests, document management
    "DocumentBatchHandler": Stability.STABLE,  # Batch document upload/processing
    "FolderUploadHandler": Stability.EXPERIMENTAL,  # Folder upload support - new
    "BreakpointsHandler": Stability.STABLE,  # 34 tests, debate breakpoints
    "SlackHandler": Stability.EXPERIMENTAL,  # Slack integration - new
    "EvidenceHandler": Stability.STABLE,  # Evidence collection and storage
    "WebhookHandler": Stability.STABLE,  # Webhook registration and delivery
    "AdminHandler": Stability.STABLE,  # Admin panel backend API
    "PrivacyHandler": Stability.STABLE,  # GDPR/CCPA data export and deletion
}


def get_handler_stability(handler_name: str) -> Stability:
    """Get the stability level for a handler.

    Args:
        handler_name: Handler class name (e.g., 'DebatesHandler')

    Returns:
        Stability level, defaults to EXPERIMENTAL if not classified
    """
    return HANDLER_STABILITY.get(handler_name, Stability.EXPERIMENTAL)


def get_all_handler_stability() -> dict[str, str]:
    """Get all handler stability levels as strings for API response."""
    return {name: stability.value for name, stability in HANDLER_STABILITY.items()}


__all__ = [
    # Base utilities
    "HandlerResult",
    "BaseHandler",
    "json_response",
    "error_response",
    # Handler registry
    "ALL_HANDLERS",
    # Individual handlers
    "DebatesHandler",
    "AgentsHandler",
    "SystemHandler",
    "HealthHandler",
    "NomicHandler",
    "DocsHandler",
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
    "DocumentBatchHandler",
    "FolderUploadHandler",
    "VerificationHandler",
    "AuditingHandler",
    "DashboardHandler",
    "PersonaHandler",
    "IntrospectionHandler",
    "CalibrationHandler",
    "RoutingHandler",
    "EvolutionHandler",
    "EvolutionABTestingHandler",
    "PluginsHandler",
    "AudioHandler",
    "SocialMediaHandler",
    "BroadcastHandler",
    "LaboratoryHandler",
    "ProbesHandler",
    "InsightsHandler",
    "GalleryHandler",
    "BreakpointsHandler",
    "LearningHandler",
    "AuthHandler",
    "BillingHandler",
    "OrganizationsHandler",
    "OAuthHandler",
    "GraphDebatesHandler",
    "MatrixDebatesHandler",
    "FeaturesHandler",
    "MemoryAnalyticsHandler",
    "GauntletHandler",
    "ReviewsHandler",
    "FormalVerificationHandler",
    "SlackHandler",
    "EvidenceHandler",
    "WebhookHandler",
    "AdminHandler",
    "PrivacyHandler",
    # Collaboration handlers
    "CollaborationHandlers",
    "get_collaboration_handlers",
    # Stability utilities
    "HANDLER_STABILITY",
    "get_handler_stability",
    "get_all_handler_stability",
]
