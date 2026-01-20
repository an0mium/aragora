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
from .agents import AgentConfigHandler
from .agents import AgentsHandler
from .analytics import AnalyticsHandler
from .analytics_dashboard import AnalyticsDashboardHandler
from .auditing import AuditingHandler
from .auth import AuthHandler
from .base import BaseHandler, HandlerResult, error_response, json_response

# Handler interfaces for type checking and contract definition
from .interface import (
    HandlerInterface,
    AuthenticatedHandlerInterface,
    PaginatedHandlerInterface,
    CachedHandlerInterface,
    StorageAccessInterface,
    MinimalServerContext,
    RouteConfig,
    HandlerRegistration,
    is_handler,
    is_authenticated_handler,
)

# Standalone utilities that don't require full server infrastructure
from .utilities import (
    get_host_header,
    get_agent_name,
    agent_to_dict,
    normalize_agent_names,
    extract_path_segment,
    build_api_url,
    is_json_content_type,
    get_media_type,
    get_request_id,
    get_content_length,
)
from .belief import BeliefHandler
from .admin import BillingHandler  # Moved to admin/
from .breakpoints import BreakpointsHandler
from .features import AudioHandler  # Moved to features/
from .transcription import TranscriptionHandler
from .features import BroadcastHandler  # Moved to features/
from .agents import CalibrationHandler  # Moved to agents/
from .checkpoints import CheckpointHandler
from .composite import CompositeHandler
from .consensus import ConsensusHandler
from .control_plane import ControlPlaneHandler
from .critique import CritiqueHandler
from .cross_pollination import (
    CrossPollinationStatsHandler,
    CrossPollinationSubscribersHandler,
    CrossPollinationBridgeHandler,
    CrossPollinationMetricsHandler,
    CrossPollinationResetHandler,
    CrossPollinationKMHandler,
    CrossPollinationKMSyncHandler,
    CrossPollinationKMStalenessHandler,
    CrossPollinationKMCultureHandler,
)
from .admin import DashboardHandler  # Moved to admin/
from .debates import DebatesHandler
from .docs import DocsHandler
from .features import DocumentHandler  # Moved to features/
from .features import DocumentBatchHandler  # Batch document upload
from .features import DocumentQueryHandler  # NL document querying
from .features import EvidenceHandler  # Moved to features/
from .features import EvidenceEnrichmentHandler  # Evidence enrichment for findings
from .features import FolderUploadHandler  # Folder upload support
from .features import FindingWorkflowHandler  # Finding workflow management
from .features import SchedulerHandler  # Audit scheduling
from .evaluation import EvaluationHandler
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
from .knowledge_base import KnowledgeHandler, KnowledgeMoundHandler
from .laboratory import LaboratoryHandler
from .agents import LeaderboardViewHandler  # Moved to agents/
from .memory import LearningHandler  # Moved to memory/
from .debates import MatrixDebatesHandler  # Moved to debates/
from .memory import MemoryHandler  # memory/ subdirectory
from .memory import MemoryAnalyticsHandler  # Moved to memory/
from .memory import CoordinatorHandler  # Memory coordinator API
from .metrics import MetricsHandler
from .moments import MomentsHandler
from .nomic import NomicHandler
from .oauth import OAuthHandler
from .organizations import OrganizationsHandler
from .persona import PersonaHandler
from .policy import PolicyHandler
from .privacy import PrivacyHandler
from .queue import QueueHandler
from .repository import RepositoryHandler
from .uncertainty import UncertaintyHandler
from .verticals import VerticalsHandler
from .workspace import WorkspaceHandler
from .features import PluginsHandler  # Moved to features/
from .agents import ProbesHandler  # Moved to agents/
from .features import PulseHandler  # Moved to features/
from .social import RelationshipHandler  # Moved to social/
from .replays import ReplaysHandler
from .reviews import ReviewsHandler
from .routing import RoutingHandler
from .ml import MLHandler
from .rlm import RLMContextHandler
from .selection import SelectionHandler
from .social import SlackHandler  # Moved to social/
from .social import SocialMediaHandler
from .admin import SystemHandler  # Moved to admin/
from .tournaments import TournamentHandler
from .training import TrainingHandler
from .verification import VerificationHandler
from .webhooks import WebhookHandler
from .workflows import WorkflowHandler
from .social import CollaborationHandlers, get_collaboration_handlers  # Moved to social/
from .bots import DiscordHandler, TeamsHandler, TelegramHandler, WhatsAppHandler, ZoomHandler  # Bot platform handlers
from .autonomous import (  # Autonomous operations handlers (Phase 5)
    ApprovalHandler,
    AlertHandler,
    TriggerHandler,
    MonitoringHandler,
    LearningHandler as AutonomousLearningHandler,  # Renamed to avoid conflict with memory/LearningHandler
)

# List of all handler classes for automatic dispatch registration
# Order matters: more specific handlers should come first
ALL_HANDLERS = [
    GraphDebatesHandler,  # More specific path: /api/debates/graph
    MatrixDebatesHandler,  # More specific path: /api/debates/matrix
    CompositeHandler,  # More specific: /api/debates/*/full-context, /api/agents/*/reliability
    DebatesHandler,
    AgentConfigHandler,  # More specific: /api/agents/configs/*
    AgentsHandler,
    HealthHandler,  # More specific: /healthz, /readyz, /api/health/*
    NomicHandler,  # More specific: /api/nomic/*
    DocsHandler,  # More specific: /api/openapi*, /api/docs*, /api/redoc*
    SystemHandler,
    PulseHandler,
    AnalyticsHandler,
    AnalyticsDashboardHandler,  # Enterprise analytics dashboard
    MetricsHandler,
    CrossPollinationStatsHandler,  # Cross-subsystem event observability
    CrossPollinationSubscribersHandler,
    CrossPollinationBridgeHandler,
    CrossPollinationMetricsHandler,
    CrossPollinationResetHandler,
    CrossPollinationKMHandler,
    CrossPollinationKMSyncHandler,  # Manual KM adapter sync
    CrossPollinationKMStalenessHandler,  # Manual staleness check
    CrossPollinationKMCultureHandler,  # Culture patterns query
    ConsensusHandler,
    BeliefHandler,
    ControlPlaneHandler,  # Enterprise control plane API
    CritiqueHandler,
    GenesisHandler,
    ReplaysHandler,
    TournamentHandler,
    MemoryHandler,
    CoordinatorHandler,
    LeaderboardViewHandler,
    RelationshipHandler,
    MomentsHandler,
    DocumentQueryHandler,  # NL document querying (more specific paths)
    DocumentHandler,
    DocumentBatchHandler,  # Batch document upload
    FolderUploadHandler,  # Folder upload support
    FindingWorkflowHandler,  # Finding workflow management
    EvidenceEnrichmentHandler,  # Evidence enrichment for findings
    SchedulerHandler,  # Audit scheduling
    VerificationHandler,
    AuditingHandler,
    DashboardHandler,
    PersonaHandler,
    IntrospectionHandler,
    CalibrationHandler,
    CheckpointHandler,
    RoutingHandler,
    MLHandler,  # ML capabilities API (routing, scoring, consensus)
    RLMContextHandler,  # RLM context compression and query API
    SelectionHandler,  # Selection plugin API
    EvaluationHandler,  # LLM-as-Judge evaluation endpoints
    EvolutionABTestingHandler,  # More specific: /api/evolution/ab-tests
    EvolutionHandler,
    PluginsHandler,
    AudioHandler,
    TranscriptionHandler,  # Speech-to-text transcription API
    SocialMediaHandler,
    BroadcastHandler,
    LaboratoryHandler,
    ProbesHandler,
    InsightsHandler,
    KnowledgeHandler,
    KnowledgeMoundHandler,  # Extended Knowledge Mound API - Phase A1
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
    PolicyHandler,  # Policy and compliance management API
    PrivacyHandler,
    QueueHandler,  # Job queue management API
    RepositoryHandler,  # Repository indexing API - Phase A3
    UncertaintyHandler,  # Uncertainty estimation API - Phase A1
    VerticalsHandler,  # Vertical specialist API
    WorkspaceHandler,  # Enterprise workspace/privacy management
    WorkflowHandler,  # Enterprise workflow engine API
    TrainingHandler,  # RLM training data collection API
    # Bot platform handlers
    DiscordHandler,  # Discord Interactions API
    TeamsHandler,  # Microsoft Teams Bot Framework
    TelegramHandler,  # Telegram Bot API webhooks
    WhatsAppHandler,  # WhatsApp Cloud API
    ZoomHandler,  # Zoom webhooks and chat
    # Autonomous operations handlers (Phase 5)
    ApprovalHandler,  # Human-in-the-loop approval flows
    AlertHandler,  # Alert management and thresholds
    TriggerHandler,  # Scheduled debate triggers
    MonitoringHandler,  # Trend and anomaly monitoring
    AutonomousLearningHandler,  # Continuous learning (ELO, patterns, calibration)
]

# Handler stability classifications
# - STABLE: Production-ready, extensively tested, API stable
# - EXPERIMENTAL: Works but may change, use with awareness
# - PREVIEW: Early access, expect changes and potential issues
# - DEPRECATED: Being phased out, use alternative
HANDLER_STABILITY: dict[str, Stability] = {
    # Core - Stable
    "DebatesHandler": Stability.STABLE,
    "AgentConfigHandler": Stability.STABLE,  # YAML agent config endpoints
    "AgentsHandler": Stability.STABLE,
    "SystemHandler": Stability.STABLE,
    "HealthHandler": Stability.STABLE,  # Extracted from SystemHandler
    "NomicHandler": Stability.STABLE,  # Extracted from SystemHandler
    "DocsHandler": Stability.STABLE,  # Extracted from SystemHandler
    "AnalyticsHandler": Stability.STABLE,
    "AnalyticsDashboardHandler": Stability.EXPERIMENTAL,  # Enterprise analytics dashboard
    "ConsensusHandler": Stability.STABLE,
    "MetricsHandler": Stability.STABLE,
    "MemoryHandler": Stability.STABLE,
    "CoordinatorHandler": Stability.STABLE,
    "LeaderboardViewHandler": Stability.STABLE,
    "ReplaysHandler": Stability.STABLE,
    "FeaturesHandler": Stability.STABLE,
    "AuthHandler": Stability.STABLE,
    # Extended - Stable
    "TournamentHandler": Stability.STABLE,
    "ControlPlaneHandler": Stability.EXPERIMENTAL,  # Enterprise control plane - Phase 0
    "CritiqueHandler": Stability.STABLE,
    "RelationshipHandler": Stability.STABLE,
    "DashboardHandler": Stability.STABLE,
    "RoutingHandler": Stability.STABLE,
    "CompositeHandler": Stability.EXPERIMENTAL,  # Composite API endpoints - new
    "MLHandler": Stability.EXPERIMENTAL,  # ML capabilities API - new
    "RLMContextHandler": Stability.EXPERIMENTAL,  # RLM context compression and query API - new
    "SelectionHandler": Stability.STABLE,  # Selection plugin API
    # Promoted to Stable (Jan 2026) - tested in production
    "BillingHandler": Stability.STABLE,  # Transaction tests, Stripe webhooks
    "OAuthHandler": Stability.STABLE,  # OAuth flow tests, Google integration
    "AudioHandler": Stability.STABLE,  # Podcast generation, TTS
    "TranscriptionHandler": Stability.EXPERIMENTAL,  # Speech-to-text transcription - new
    "TrainingHandler": Stability.EXPERIMENTAL,  # RLM training data collection - new
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
    "EvaluationHandler": Stability.EXPERIMENTAL,  # LLM-as-Judge evaluation - new
    "EvolutionHandler": Stability.STABLE,  # 7 test files, 66+ tests
    "EvolutionABTestingHandler": Stability.STABLE,  # AB testing with evolution
    "LaboratoryHandler": Stability.STABLE,  # 3 test files, 70+ tests
    "IntrospectionHandler": Stability.STABLE,  # 2 test files, 53+ tests
    "LearningHandler": Stability.STABLE,  # 2 test files, 66+ tests
    "MemoryAnalyticsHandler": Stability.STABLE,  # Handler tests, 23+ tests
    "ProbesHandler": Stability.STABLE,  # 16 tests, capability probing
    "InsightsHandler": Stability.STABLE,  # 3 test files, 110+ tests
    "KnowledgeHandler": Stability.EXPERIMENTAL,  # Knowledge base API - new
    "KnowledgeMoundHandler": Stability.EXPERIMENTAL,  # Knowledge Mound API - Phase A1
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
    "DocumentQueryHandler": Stability.EXPERIMENTAL,  # NL document querying - new
    "FolderUploadHandler": Stability.EXPERIMENTAL,  # Folder upload support - new
    "FindingWorkflowHandler": Stability.EXPERIMENTAL,  # Finding workflow - new
    "EvidenceEnrichmentHandler": Stability.EXPERIMENTAL,  # Evidence enrichment - new
    "SchedulerHandler": Stability.EXPERIMENTAL,  # Audit scheduling - new
    "BreakpointsHandler": Stability.STABLE,  # 34 tests, debate breakpoints
    "SlackHandler": Stability.EXPERIMENTAL,  # Slack integration - new
    "EvidenceHandler": Stability.STABLE,  # Evidence collection and storage
    "WebhookHandler": Stability.STABLE,  # Webhook registration and delivery
    "AdminHandler": Stability.STABLE,  # Admin panel backend API
    "PolicyHandler": Stability.EXPERIMENTAL,  # Policy and compliance management - new
    "PrivacyHandler": Stability.STABLE,  # GDPR/CCPA data export and deletion
    "WorkspaceHandler": Stability.EXPERIMENTAL,  # Enterprise workspace/privacy management
    "WorkflowHandler": Stability.EXPERIMENTAL,  # Enterprise workflow engine API - Phase 2
    "QueueHandler": Stability.EXPERIMENTAL,  # Job queue management API - Phase A1
    "RepositoryHandler": Stability.EXPERIMENTAL,  # Repository indexing API - Phase A3
    "UncertaintyHandler": Stability.EXPERIMENTAL,  # Uncertainty estimation API - Phase A1
    "VerticalsHandler": Stability.EXPERIMENTAL,  # Vertical specialist API - Phase A1
    # Bot platform handlers
    "DiscordHandler": Stability.EXPERIMENTAL,  # Discord Interactions API - new
    "TeamsHandler": Stability.EXPERIMENTAL,  # Microsoft Teams Bot Framework - new
    "TelegramHandler": Stability.EXPERIMENTAL,  # Telegram Bot API webhooks - new
    "WhatsAppHandler": Stability.EXPERIMENTAL,  # WhatsApp Cloud API - new
    "ZoomHandler": Stability.EXPERIMENTAL,  # Zoom webhooks and chat - new
    # Autonomous operations handlers (Phase 5)
    "ApprovalHandler": Stability.EXPERIMENTAL,  # Human-in-the-loop approval flows - Phase 5.1
    "AlertHandler": Stability.EXPERIMENTAL,  # Alert management and thresholds - Phase 5.3
    "TriggerHandler": Stability.EXPERIMENTAL,  # Scheduled debate triggers - Phase 5.3
    "MonitoringHandler": Stability.EXPERIMENTAL,  # Trend and anomaly monitoring - Phase 5.3
    "AutonomousLearningHandler": Stability.EXPERIMENTAL,  # Continuous learning - Phase 5.2
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
    # Handler interfaces (from interface.py)
    "HandlerInterface",
    "AuthenticatedHandlerInterface",
    "PaginatedHandlerInterface",
    "CachedHandlerInterface",
    "StorageAccessInterface",
    "MinimalServerContext",
    "RouteConfig",
    "HandlerRegistration",
    "is_handler",
    "is_authenticated_handler",
    # Standalone utilities (from utilities.py)
    "get_host_header",
    "get_agent_name",
    "agent_to_dict",
    "normalize_agent_names",
    "extract_path_segment",
    "build_api_url",
    "is_json_content_type",
    "get_media_type",
    "get_request_id",
    "get_content_length",
    # Handler registry
    "ALL_HANDLERS",
    # Individual handlers
    "DebatesHandler",
    "AgentConfigHandler",
    "AgentsHandler",
    "SystemHandler",
    "HealthHandler",
    "NomicHandler",
    "DocsHandler",
    "PulseHandler",
    "AnalyticsHandler",
    "AnalyticsDashboardHandler",
    "MetricsHandler",
    "ConsensusHandler",
    "BeliefHandler",
    "ControlPlaneHandler",
    "CritiqueHandler",
    "GenesisHandler",
    "ReplaysHandler",
    "TournamentHandler",
    "MemoryHandler",
    "CoordinatorHandler",
    "LeaderboardViewHandler",
    "RelationshipHandler",
    "MomentsHandler",
    "DocumentHandler",
    "DocumentBatchHandler",
    "DocumentQueryHandler",
    "FolderUploadHandler",
    "FindingWorkflowHandler",
    "EvidenceEnrichmentHandler",
    "SchedulerHandler",
    "VerificationHandler",
    "AuditingHandler",
    "DashboardHandler",
    "PersonaHandler",
    "IntrospectionHandler",
    "CalibrationHandler",
    "CompositeHandler",
    "RoutingHandler",
    "MLHandler",
    "RLMContextHandler",
    "EvolutionHandler",
    "EvolutionABTestingHandler",
    "PluginsHandler",
    "AudioHandler",
    "TranscriptionHandler",
    "SocialMediaHandler",
    "BroadcastHandler",
    "LaboratoryHandler",
    "ProbesHandler",
    "InsightsHandler",
    "KnowledgeHandler",
    "KnowledgeMoundHandler",
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
    "PolicyHandler",
    "PrivacyHandler",
    "QueueHandler",
    "RepositoryHandler",
    "UncertaintyHandler",
    "VerticalsHandler",
    "WorkspaceHandler",
    "WorkflowHandler",
    "TrainingHandler",
    # Collaboration handlers
    "CollaborationHandlers",
    "get_collaboration_handlers",
    # Bot platform handlers
    "DiscordHandler",
    "TeamsHandler",
    "TelegramHandler",
    "WhatsAppHandler",
    "ZoomHandler",
    # Autonomous operations handlers (Phase 5)
    "ApprovalHandler",
    "AlertHandler",
    "TriggerHandler",
    "MonitoringHandler",
    "AutonomousLearningHandler",
    # Stability utilities
    "HANDLER_STABILITY",
    "get_handler_stability",
    "get_all_handler_stability",
]
