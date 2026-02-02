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

import importlib
from typing import TYPE_CHECKING, Any

from aragora.config.stability import Stability

# Lazy loading infrastructure - load early, contains only string mappings
from ._lazy_imports import ALL_HANDLER_NAMES, HANDLER_MODULES

# IMPORTANT: Import order matters to avoid circular imports.
# The admin.cache module must be loaded before base.py because:
# 1. base.py imports from admin.cache
# 2. admin/__init__.py imports from handler.py, which imports from base.py
# By pre-loading admin.cache, we break the circular dependency.
from .admin import cache as _admin_cache  # noqa: F401

# Expose utils submodule for tests
from . import utils as utils  # noqa: PLC0414

# Base utilities - always loaded (small and frequently needed)
from .base import BaseHandler, HandlerResult, error_response, json_response

# Handler mixins (extracted to separate module)
from .mixins import (
    AuthenticatedHandlerMixin,
    CachedHandlerMixin,
    PaginatedHandlerMixin,
)

# API decorators (extracted to separate module)
from .api_decorators import (
    api_endpoint,
    rate_limit,
    require_quota,
    validate_body,
)

# Typed handler base classes (extracted to separate module)
from .typed_handlers import (
    AdminHandler as TypedAdminHandler,
    AsyncTypedHandler,
    AuthenticatedHandler as TypedAuthenticatedHandler,
    MaybeAsyncHandlerResult,
    PermissionHandler,
    ResourceHandler,
    TypedHandler,
)

# Handler interfaces for type checking and contract definition
from .interface import (
    AuthenticatedHandlerInterface,
    CachedHandlerInterface,
    HandlerInterface,
    HandlerRegistration,
    MinimalServerContext,
    PaginatedHandlerInterface,
    RouteConfig,
    StorageAccessInterface,
    is_authenticated_handler,
    is_handler,
)

# Shared types for handlers (protocols, type aliases, common parameters)
from .types import (
    AsyncHandlerFunction,
    AsyncMiddlewareFunction,
    FilterParams,
    HandlerFunction,
    HandlerProtocol,
    MaybeAsyncHandlerFunction,
    MaybeAsyncMiddlewareFunction,
    MiddlewareFactory,
    MiddlewareFunction,
    PaginationParams,
    QueryParams,
    RequestContext,
    ResponseType,
    SortParams,
)

# Standalone utilities that don't require full server infrastructure
from .utilities import (
    agent_to_dict,
    build_api_url,
    extract_path_segment,
    get_agent_name,
    get_content_length,
    get_host_header,
    get_media_type,
    get_request_id,
    is_json_content_type,
    normalize_agent_names,
)

# Type checking imports - these are not executed at runtime
if TYPE_CHECKING:
    from .a2a import A2AHandler
    from .admin import (
        AdminHandler,
        BillingHandler,
        DashboardHandler,
        HealthHandler,
        SecurityHandler,
        SystemHandler,
    )
    from .agents import (
        AgentConfigHandler,
        AgentsHandler,
        CalibrationHandler,
        LeaderboardViewHandler,
        ProbesHandler,
    )
    from ._analytics_impl import AnalyticsHandler
    from .analytics_dashboard import AnalyticsDashboardHandler
    from ._analytics_metrics_impl import AnalyticsMetricsHandler
    from .ap_automation import APAutomationHandler
    from .ar_automation import ARAutomationHandler
    from .auditing import AuditingHandler
    from .auth import AuthHandler
    from .autonomous import (
        AlertHandler,
        ApprovalHandler,
        LearningHandler as AutonomousLearningHandler,
        MonitoringHandler,
        TriggerHandler,
    )
    from .backup_handler import BackupHandler
    from .belief import BeliefHandler
    from .bindings import BindingsHandler
    from .bots import (
        DiscordHandler,
        GoogleChatHandler,
        TeamsHandler,
        TelegramHandler,
        WhatsAppHandler,
        ZoomHandler,
    )
    from .breakpoints import BreakpointsHandler
    from .budgets import BudgetHandler
    from .canvas import CanvasHandler
    from .checkpoints import CheckpointHandler
    from .code_review import CodeReviewHandler
    from .codebase import IntelligenceHandler
    from .composite import CompositeHandler
    from .computer_use_handler import ComputerUseHandler
    from .consensus import ConsensusHandler
    from .control_plane import ControlPlaneHandler
    from .critique import CritiqueHandler
    from .cross_pollination import (
        CrossPollinationBridgeHandler,
        CrossPollinationKMCultureHandler,
        CrossPollinationKMHandler,
        CrossPollinationKMStalenessHandler,
        CrossPollinationKMSyncHandler,
        CrossPollinationMetricsHandler,
        CrossPollinationResetHandler,
        CrossPollinationStatsHandler,
        CrossPollinationSubscribersHandler,
    )
    from .debates import DebatesHandler, GraphDebatesHandler, MatrixDebatesHandler
    from .decision import DecisionHandler
    from .decisions import DecisionExplainHandler
    from .deliberations import DeliberationsHandler
    from .dependency_analysis import DependencyAnalysisHandler
    from .devices import DeviceHandler
    from .docs import DocsHandler
    from .email import EmailHandler
    from .email_services import EmailServicesHandler
    from .endpoint_analytics import EndpointAnalyticsHandler
    from .erc8004 import ERC8004Handler
    from .evaluation import EvaluationHandler
    from .evolution import EvolutionABTestingHandler, EvolutionHandler
    from .expenses import ExpenseHandler
    from .explainability import ExplainabilityHandler
    from .external_agents import ExternalAgentsHandler
    from .external_integrations import ExternalIntegrationsHandler
    from .features import (
        AdvertisingHandler,
        AnalyticsPlatformsHandler,
        AuditSessionsHandler,
        AudioHandler,
        BroadcastHandler,
        CloudStorageHandler,
        CodebaseAuditHandler,
        ConnectorsHandler,
        CRMHandler,
        CrossPlatformAnalyticsHandler,
        DevOpsHandler,
        DocumentBatchHandler,
        DocumentHandler,
        DocumentQueryHandler,
        EcommerceHandler,
        EmailWebhooksHandler,
        EvidenceEnrichmentHandler,
        EvidenceHandler,
        FeaturesHandler,
        FindingWorkflowHandler,
        FolderUploadHandler,
        GmailIngestHandler,
        GmailQueryHandler,
        IntegrationsHandler,
        LegalHandler,
        MarketplaceHandler,
        PluginsHandler,
        PulseHandler,
        ReconciliationHandler,
        RLMHandler,
        RoutingRulesHandler,
        SchedulerHandler,
        SmartUploadHandler,
        SupportHandler,
        UnifiedInboxHandler,
    )
    from .features.gmail_labels import GmailLabelsHandler
    from .features.gmail_threads import GmailThreadsHandler
    from .gallery import GalleryHandler
    from .gateway_agents_handler import GatewayAgentsHandler
    from .gateway_credentials_handler import GatewayCredentialsHandler
    from .gateway_handler import GatewayHandler
    from .gateway_health_handler import GatewayHealthHandler
    from .gauntlet import GauntletHandler
    from .gauntlet_v1 import (
        GAUNTLET_V1_HANDLERS,
        GauntletAllSchemasHandler,
        GauntletHeatmapExportHandler,
        GauntletReceiptExportHandler,
        GauntletSchemaHandler,
        GauntletTemplateHandler,
        GauntletTemplatesListHandler,
        GauntletValidateReceiptHandler,
    )
    from .genesis import GenesisHandler
    from .hybrid_debate_handler import HybridDebateHandler
    from .integration_management import (
        IntegrationsHandler as IntegrationManagementHandler,
    )
    from .introspection import IntrospectionHandler
    from .invoices import InvoiceHandler
    from .knowledge.checkpoints import KMCheckpointHandler
    from .knowledge_base import KnowledgeHandler, KnowledgeMoundHandler
    from .knowledge_chat import KnowledgeChatHandler
    from .laboratory import LaboratoryHandler
    from .memory import (
        CoordinatorHandler,
        InsightsHandler,
        LearningHandler,
        MemoryAnalyticsHandler,
        MemoryHandler,
    )
    from .metrics import MetricsHandler
    from .ml import MLHandler
    from .moments import MomentsHandler
    from .nomic import NomicHandler
    from .oauth import OAuthHandler
    from .oauth_wizard import OAuthWizardHandler
    from .onboarding import (
        OnboardingHandler,
        get_onboarding_handlers,
        handle_analytics,
        handle_first_debate,
        handle_get_flow,
        handle_get_templates,
        handle_init_flow,
        handle_quick_start,
        handle_update_step,
    )
    from .openclaw_gateway import OpenClawGatewayHandler
    from .orchestration import OrchestrationHandler
    from .organizations import OrganizationsHandler
    from .persona import PersonaHandler
    from .policy import PolicyHandler
    from .privacy import PrivacyHandler
    from .public import StatusPageHandler
    from .queue import QueueHandler
    from .replays import ReplaysHandler
    from .repository import RepositoryHandler
    from .reviews import ReviewsHandler
    from .rlm import RLMContextHandler
    from .routing import RoutingHandler
    from .scim_handler import SCIMHandler
    from .selection import SelectionHandler
    from .skills import SkillsHandler
    from .slo import SLOHandler
    from .sme_usage_dashboard import SMEUsageDashboardHandler
    from .social import (
        CollaborationHandlers,
        RelationshipHandler,
        SlackHandler,
        SocialMediaHandler,
        get_collaboration_handlers,
    )
    from .social.teams import TeamsIntegrationHandler
    from .template_marketplace import TemplateMarketplaceHandler
    from .tournaments import TournamentHandler
    from .training import TrainingHandler
    from .transcription import TranscriptionHandler
    from .uncertainty import UncertaintyHandler
    from .usage_metering import UsageMeteringHandler
    from .verification import FormalVerificationHandler, VerificationHandler
    from .verticals import VerticalsHandler
    from .webhooks import WebhookHandler
    from .workflow_templates import (
        TemplateRecommendationsHandler,
        WorkflowCategoriesHandler,
        WorkflowPatternTemplatesHandler,
        WorkflowPatternsHandler,
        WorkflowTemplatesHandler,
    )
    from .workflows import WorkflowHandler
    from .workspace import WorkspaceHandler


# Cache for lazily loaded handlers
_handler_cache: dict[str, Any] = {}

# Cached ALL_HANDLERS list
_all_handlers_cache: list[type] | None = None


def _get_all_handlers() -> list[type]:
    """Lazily load and return all handler classes."""
    global _all_handlers_cache
    if _all_handlers_cache is not None:
        return _all_handlers_cache

    handlers = []
    for name in ALL_HANDLER_NAMES:
        try:
            handler = _lazy_import(name)
            if handler is not None:
                handlers.append(handler)
        except (ImportError, AttributeError):
            # Skip handlers that fail to import
            pass
    _all_handlers_cache = handlers
    return handlers


def _lazy_import(name: str) -> Any:
    """Lazily import a handler by name."""
    if name in _handler_cache:
        return _handler_cache[name]

    if name not in HANDLER_MODULES:
        return None

    module_path = HANDLER_MODULES[name]
    module = importlib.import_module(module_path)
    attr = getattr(module, name)
    _handler_cache[name] = attr
    return attr


def __getattr__(name: str) -> Any:
    """Lazy loading via module __getattr__."""
    # Handle ALL_HANDLERS specially
    if name == "ALL_HANDLERS":
        return _get_all_handlers()

    # Handle GAUNTLET_V1_HANDLERS specially
    if name == "GAUNTLET_V1_HANDLERS":
        return _lazy_import("GAUNTLET_V1_HANDLERS")

    # Check if this is a lazily-loaded handler
    if name in HANDLER_MODULES:
        return _lazy_import(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Handler stability classifications
# - STABLE: Production-ready, extensively tested, API stable
# - EXPERIMENTAL: Works but may change, use with awareness
# - PREVIEW: Early access, expect changes and potential issues
# - DEPRECATED: Being phased out, use alternative
HANDLER_STABILITY: dict[str, Stability] = {
    # Core - Stable
    "DebatesHandler": Stability.STABLE,
    "AgentConfigHandler": Stability.STABLE,
    "AgentsHandler": Stability.STABLE,
    "SystemHandler": Stability.STABLE,
    "HealthHandler": Stability.STABLE,
    "StatusPageHandler": Stability.STABLE,
    "NomicHandler": Stability.STABLE,
    "DocsHandler": Stability.STABLE,
    "AnalyticsHandler": Stability.STABLE,
    "AnalyticsDashboardHandler": Stability.STABLE,
    "AnalyticsMetricsHandler": Stability.STABLE,
    "EndpointAnalyticsHandler": Stability.STABLE,
    "CrossPlatformAnalyticsHandler": Stability.STABLE,
    "ConsensusHandler": Stability.STABLE,
    "MetricsHandler": Stability.STABLE,
    "SLOHandler": Stability.STABLE,
    "MemoryHandler": Stability.STABLE,
    "CoordinatorHandler": Stability.STABLE,
    "LeaderboardViewHandler": Stability.STABLE,
    "ReplaysHandler": Stability.STABLE,
    "FeaturesHandler": Stability.STABLE,
    "ConnectorsHandler": Stability.STABLE,
    "IntegrationsHandler": Stability.STABLE,
    "ExternalIntegrationsHandler": Stability.STABLE,
    "IntegrationManagementHandler": Stability.STABLE,
    "OAuthWizardHandler": Stability.STABLE,
    "TeamsIntegrationHandler": Stability.STABLE,
    "AuthHandler": Stability.STABLE,
    "TournamentHandler": Stability.STABLE,
    "DecisionHandler": Stability.STABLE,
    "ControlPlaneHandler": Stability.STABLE,
    "CritiqueHandler": Stability.STABLE,
    "RelationshipHandler": Stability.STABLE,
    "DashboardHandler": Stability.STABLE,
    "RoutingHandler": Stability.STABLE,
    "RoutingRulesHandler": Stability.STABLE,
    "CompositeHandler": Stability.STABLE,
    "MLHandler": Stability.STABLE,
    "RLMContextHandler": Stability.STABLE,
    "RLMHandler": Stability.STABLE,
    "SelectionHandler": Stability.STABLE,
    "BillingHandler": Stability.STABLE,
    "BudgetHandler": Stability.EXPERIMENTAL,
    "OAuthHandler": Stability.STABLE,
    "AudioHandler": Stability.STABLE,
    "DeviceHandler": Stability.STABLE,
    "TranscriptionHandler": Stability.STABLE,
    "TrainingHandler": Stability.STABLE,
    "VerificationHandler": Stability.STABLE,
    "PulseHandler": Stability.STABLE,
    "GalleryHandler": Stability.STABLE,
    "GauntletHandler": Stability.STABLE,
    "GauntletSchemaHandler": Stability.STABLE,
    "GauntletAllSchemasHandler": Stability.STABLE,
    "GauntletTemplatesListHandler": Stability.STABLE,
    "GauntletTemplateHandler": Stability.STABLE,
    "GauntletReceiptExportHandler": Stability.STABLE,
    "GauntletHeatmapExportHandler": Stability.STABLE,
    "GauntletValidateReceiptHandler": Stability.STABLE,
    "BeliefHandler": Stability.STABLE,
    "SkillsHandler": Stability.STABLE,
    "BindingsHandler": Stability.STABLE,
    "CalibrationHandler": Stability.STABLE,
    "PersonaHandler": Stability.STABLE,
    "GraphDebatesHandler": Stability.STABLE,
    "MatrixDebatesHandler": Stability.STABLE,
    "EvaluationHandler": Stability.STABLE,
    "EvolutionHandler": Stability.STABLE,
    "EvolutionABTestingHandler": Stability.STABLE,
    "LaboratoryHandler": Stability.STABLE,
    "IntrospectionHandler": Stability.STABLE,
    "LearningHandler": Stability.STABLE,
    "MemoryAnalyticsHandler": Stability.STABLE,
    "ProbesHandler": Stability.STABLE,
    "InsightsHandler": Stability.STABLE,
    "KnowledgeHandler": Stability.STABLE,
    "KnowledgeMoundHandler": Stability.STABLE,
    "KnowledgeChatHandler": Stability.STABLE,
    "ReviewsHandler": Stability.STABLE,
    "FormalVerificationHandler": Stability.STABLE,
    "OrganizationsHandler": Stability.STABLE,
    "SocialMediaHandler": Stability.STABLE,
    "MomentsHandler": Stability.STABLE,
    "AuditingHandler": Stability.STABLE,
    "PluginsHandler": Stability.STABLE,
    "BroadcastHandler": Stability.STABLE,
    "GenesisHandler": Stability.STABLE,
    "DocumentHandler": Stability.STABLE,
    "DocumentBatchHandler": Stability.STABLE,
    "DocumentQueryHandler": Stability.STABLE,
    "FolderUploadHandler": Stability.STABLE,
    "SmartUploadHandler": Stability.STABLE,
    "CloudStorageHandler": Stability.EXPERIMENTAL,
    "FindingWorkflowHandler": Stability.EXPERIMENTAL,
    "EvidenceEnrichmentHandler": Stability.EXPERIMENTAL,
    "SchedulerHandler": Stability.EXPERIMENTAL,
    "AuditSessionsHandler": Stability.EXPERIMENTAL,
    "BreakpointsHandler": Stability.STABLE,
    "SlackHandler": Stability.STABLE,
    "EvidenceHandler": Stability.STABLE,
    "WebhookHandler": Stability.STABLE,
    "AdminHandler": Stability.STABLE,
    "SecurityHandler": Stability.STABLE,
    "PolicyHandler": Stability.STABLE,
    "PrivacyHandler": Stability.STABLE,
    "WorkspaceHandler": Stability.STABLE,
    "WorkflowHandler": Stability.STABLE,
    "WorkflowTemplatesHandler": Stability.STABLE,
    "WorkflowCategoriesHandler": Stability.STABLE,
    "WorkflowPatternsHandler": Stability.STABLE,
    "WorkflowPatternTemplatesHandler": Stability.STABLE,
    "TemplateRecommendationsHandler": Stability.STABLE,
    "TemplateMarketplaceHandler": Stability.STABLE,
    "MarketplaceHandler": Stability.STABLE,
    "QueueHandler": Stability.EXPERIMENTAL,
    "RepositoryHandler": Stability.STABLE,
    "UncertaintyHandler": Stability.STABLE,
    "VerticalsHandler": Stability.STABLE,
    "DiscordHandler": Stability.STABLE,
    "GoogleChatHandler": Stability.STABLE,
    "TeamsHandler": Stability.STABLE,
    "TelegramHandler": Stability.STABLE,
    "WhatsAppHandler": Stability.STABLE,
    "ZoomHandler": Stability.STABLE,
    "ExplainabilityHandler": Stability.STABLE,
    "SCIMHandler": Stability.STABLE,
    "A2AHandler": Stability.EXPERIMENTAL,
    "ApprovalHandler": Stability.STABLE,
    "AlertHandler": Stability.EXPERIMENTAL,
    "TriggerHandler": Stability.STABLE,
    "MonitoringHandler": Stability.STABLE,
    "AutonomousLearningHandler": Stability.EXPERIMENTAL,
    "EmailHandler": Stability.STABLE,
    "EmailServicesHandler": Stability.STABLE,
    "GmailIngestHandler": Stability.STABLE,
    "GmailQueryHandler": Stability.STABLE,
    "UnifiedInboxHandler": Stability.STABLE,
    "EmailWebhooksHandler": Stability.STABLE,
    "DependencyAnalysisHandler": Stability.EXPERIMENTAL,
    "CodebaseAuditHandler": Stability.EXPERIMENTAL,
    "ExpenseHandler": Stability.EXPERIMENTAL,
    "InvoiceHandler": Stability.EXPERIMENTAL,
    "ARAutomationHandler": Stability.EXPERIMENTAL,
    "APAutomationHandler": Stability.EXPERIMENTAL,
    "ReconciliationHandler": Stability.EXPERIMENTAL,
    "CodeReviewHandler": Stability.STABLE,
    "LegalHandler": Stability.STABLE,
    "DevOpsHandler": Stability.STABLE,
    "AdvertisingHandler": Stability.EXPERIMENTAL,
    "AnalyticsPlatformsHandler": Stability.EXPERIMENTAL,
    "CRMHandler": Stability.STABLE,
    "SupportHandler": Stability.STABLE,
    "EcommerceHandler": Stability.STABLE,
    "ExternalAgentsHandler": Stability.STABLE,
    "OpenClawGatewayHandler": Stability.EXPERIMENTAL,
    "GatewayHealthHandler": Stability.EXPERIMENTAL,
    "GatewayAgentsHandler": Stability.EXPERIMENTAL,
    "GatewayCredentialsHandler": Stability.EXPERIMENTAL,
    "HybridDebateHandler": Stability.STABLE,
    "ERC8004Handler": Stability.STABLE,
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


# Populate the registry for modules that need to avoid circular imports
# (e.g., features.py needs to enumerate handlers)
# This is deferred to avoid importing all handlers
def _populate_registry() -> None:
    """Populate the handler registry with lazily loaded handlers."""
    from aragora.server.handlers import _registry

    _registry.ALL_HANDLERS[:] = _get_all_handlers()
    _registry.HANDLER_STABILITY.update(HANDLER_STABILITY)


__all__ = [
    # Base utilities
    "HandlerResult",
    "BaseHandler",
    "json_response",
    "error_response",
    # Handler mixins (from mixins.py)
    "PaginatedHandlerMixin",
    "CachedHandlerMixin",
    "AuthenticatedHandlerMixin",
    # API decorators (from api_decorators.py)
    "api_endpoint",
    "rate_limit",
    "validate_body",
    "require_quota",
    # Typed handler base classes (from typed_handlers.py)
    "TypedHandler",
    "TypedAuthenticatedHandler",
    "PermissionHandler",
    "TypedAdminHandler",
    "AsyncTypedHandler",
    "ResourceHandler",
    "MaybeAsyncHandlerResult",
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
    # Shared types (from types.py)
    "HandlerProtocol",
    "RequestContext",
    "ResponseType",
    "HandlerFunction",
    "AsyncHandlerFunction",
    "MaybeAsyncHandlerFunction",
    "MiddlewareFunction",
    "AsyncMiddlewareFunction",
    "MaybeAsyncMiddlewareFunction",
    "MiddlewareFactory",
    "PaginationParams",
    "FilterParams",
    "SortParams",
    "QueryParams",
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
    # Individual handlers (lazily loaded)
    "DebatesHandler",
    "AgentConfigHandler",
    "AgentsHandler",
    "SystemHandler",
    "HealthHandler",
    "StatusPageHandler",
    "NomicHandler",
    "DocsHandler",
    "PulseHandler",
    "AnalyticsHandler",
    "AnalyticsDashboardHandler",
    "AnalyticsMetricsHandler",
    "EndpointAnalyticsHandler",
    "CrossPlatformAnalyticsHandler",
    "MetricsHandler",
    "SLOHandler",
    "ConsensusHandler",
    "BeliefHandler",
    "SkillsHandler",
    "BindingsHandler",
    "ControlPlaneHandler",
    "OrchestrationHandler",
    "DecisionExplainHandler",
    "DecisionHandler",
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
    "SmartUploadHandler",
    "CloudStorageHandler",
    "FindingWorkflowHandler",
    "EvidenceEnrichmentHandler",
    "SchedulerHandler",
    "AuditSessionsHandler",
    "VerificationHandler",
    "AuditingHandler",
    "DashboardHandler",
    "PersonaHandler",
    "IntrospectionHandler",
    "CalibrationHandler",
    "CanvasHandler",
    "CompositeHandler",
    "RoutingHandler",
    "RoutingRulesHandler",
    "MLHandler",
    "RLMContextHandler",
    "RLMHandler",
    "EvolutionHandler",
    "EvolutionABTestingHandler",
    "PluginsHandler",
    "AudioHandler",
    "DeviceHandler",
    "TranscriptionHandler",
    "SocialMediaHandler",
    "BroadcastHandler",
    "LaboratoryHandler",
    "ProbesHandler",
    "InsightsHandler",
    "KnowledgeHandler",
    "KnowledgeMoundHandler",
    "KnowledgeChatHandler",
    "GalleryHandler",
    "BreakpointsHandler",
    "LearningHandler",
    "AuthHandler",
    "BillingHandler",
    "BudgetHandler",
    "UsageMeteringHandler",
    "SMEUsageDashboardHandler",
    "OrganizationsHandler",
    # Onboarding handlers
    "handle_get_flow",
    "handle_init_flow",
    "handle_update_step",
    "handle_get_templates",
    "handle_first_debate",
    "handle_quick_start",
    "handle_analytics",
    "get_onboarding_handlers",
    "OAuthHandler",
    "GraphDebatesHandler",
    "MatrixDebatesHandler",
    "FeaturesHandler",
    "ConnectorsHandler",
    "IntegrationsHandler",
    "ExternalIntegrationsHandler",
    "IntegrationManagementHandler",
    "OAuthWizardHandler",
    "TeamsIntegrationHandler",
    "MemoryAnalyticsHandler",
    "GauntletHandler",
    # Gauntlet v1 API
    "GauntletSchemaHandler",
    "GauntletAllSchemasHandler",
    "GauntletTemplatesListHandler",
    "GauntletTemplateHandler",
    "GauntletReceiptExportHandler",
    "GauntletHeatmapExportHandler",
    "GauntletValidateReceiptHandler",
    "GAUNTLET_V1_HANDLERS",
    "ReviewsHandler",
    "FormalVerificationHandler",
    "SlackHandler",
    "EvidenceHandler",
    "WebhookHandler",
    "AdminHandler",
    "SecurityHandler",
    "PolicyHandler",
    "PrivacyHandler",
    "QueueHandler",
    "RepositoryHandler",
    "UncertaintyHandler",
    "VerticalsHandler",
    "WorkspaceHandler",
    "WorkflowHandler",
    "WorkflowTemplatesHandler",
    "WorkflowCategoriesHandler",
    "WorkflowPatternsHandler",
    "WorkflowPatternTemplatesHandler",
    "TemplateRecommendationsHandler",
    "TemplateMarketplaceHandler",
    "MarketplaceHandler",
    "TrainingHandler",
    "EmailHandler",
    "EmailServicesHandler",
    "GmailIngestHandler",
    "GmailQueryHandler",
    "UnifiedInboxHandler",
    "EmailWebhooksHandler",
    "DependencyAnalysisHandler",
    "CodebaseAuditHandler",
    "IntelligenceHandler",
    # Collaboration handlers
    "CollaborationHandlers",
    "get_collaboration_handlers",
    # Bot platform handlers
    "DiscordHandler",
    "GoogleChatHandler",
    "TeamsHandler",
    "TelegramHandler",
    "WhatsAppHandler",
    "ZoomHandler",
    # Explainability
    "ExplainabilityHandler",
    # Enterprise provisioning
    "SCIMHandler",
    # Protocols
    "A2AHandler",
    # Autonomous operations handlers (Phase 5)
    "ApprovalHandler",
    "AlertHandler",
    "TriggerHandler",
    "MonitoringHandler",
    "AutonomousLearningHandler",
    # Accounting handlers (Phase 4 - SME Vertical)
    "ExpenseHandler",
    "InvoiceHandler",
    "ARAutomationHandler",
    "APAutomationHandler",
    "ReconciliationHandler",
    # Code review handler (Phase 5 - SME Vertical)
    "CodeReviewHandler",
    "LegalHandler",
    "DevOpsHandler",
    # Connector platform handlers
    "AdvertisingHandler",
    "AnalyticsPlatformsHandler",
    "CRMHandler",
    "SupportHandler",
    "EcommerceHandler",
    # OpenClaw enterprise gateway
    "OpenClawGatewayHandler",
    # Secure Gateway handlers (Batch 5)
    "GatewayHealthHandler",
    "GatewayAgentsHandler",
    "GatewayCredentialsHandler",
    "HybridDebateHandler",
    # Blockchain handlers (ERC-8004)
    "ERC8004Handler",
    # Cross-pollination handlers
    "CrossPollinationStatsHandler",
    "CrossPollinationSubscribersHandler",
    "CrossPollinationBridgeHandler",
    "CrossPollinationMetricsHandler",
    "CrossPollinationResetHandler",
    "CrossPollinationKMHandler",
    "CrossPollinationKMSyncHandler",
    "CrossPollinationKMStalenessHandler",
    "CrossPollinationKMCultureHandler",
    # Onboarding
    "OnboardingHandler",
    "BackupHandler",
    "GmailLabelsHandler",
    "GmailThreadsHandler",
    # Additional handlers (TYPE_CHECKING exports)
    "CheckpointHandler",
    "ComputerUseHandler",
    "DeliberationsHandler",
    "EvaluationHandler",
    "ExternalAgentsHandler",
    "GatewayHandler",
    "KMCheckpointHandler",
    "SelectionHandler",
    # Stability utilities
    "HANDLER_STABILITY",
    "get_handler_stability",
    "get_all_handler_stability",
]
