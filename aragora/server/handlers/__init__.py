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
from .admin import cache as cache  # noqa: F401, PLC0414  -- public for patch paths

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
    from .action_canvas import ActionCanvasHandler
    from .admin import (
        AdminHandler,
        BillingHandler,
        DashboardHandler,
        HealthHandler,
        SecurityHandler,
        SystemHandler,
    )
    from .admin.credits import CreditsAdminHandler
    from .admin.emergency_access import EmergencyAccessHandler
    from .admin.feature_flags import FeatureFlagAdminHandler
    from .admin.health.liveness import LivenessHandler
    from .admin.health.readiness import ReadinessHandler
    from .admin.health.storage_health import StorageHealthHandler
    from .agents import (
        AgentConfigHandler,
        AgentsHandler,
        CalibrationHandler,
        LeaderboardViewHandler,
        ProbesHandler,
    )
    from .agents.feedback import FeedbackHandler
    from .agents.recommendations import AgentRecommendationHandler
    from ._analytics_impl import AnalyticsHandler
    from .analytics_dashboard import AnalyticsDashboardHandler
    from ._analytics_metrics_impl import AnalyticsMetricsHandler
    from .ap_automation import APAutomationHandler
    from .ar_automation import ARAutomationHandler
    from .audit_trail import AuditTrailHandler
    from .audience_suggestions import AudienceSuggestionsHandler
    from .auditing import AuditingHandler
    from .security_debate import SecurityDebateHandler
    from .auth import AuthHandler
    from .autonomous import (
        AlertHandler,
        ApprovalHandler,
        # Unified approvals inbox
        # (separate handler module)
        LearningHandler as AutonomousLearningHandler,
        MonitoringHandler,
        TriggerHandler,
    )
    from .approvals_inbox import UnifiedApprovalsHandler
    from .backup_handler import BackupHandler
    from .belief import BeliefHandler
    from .benchmarking import BenchmarkingHandler
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
    from .compliance_reports import ComplianceReportHandler
    from .composite import CompositeHandler
    from .connectors.management import ConnectorManagementHandler
    from .context_budget import ContextBudgetHandler
    from .computer_use_handler import ComputerUseHandler
    from .consensus import ConsensusHandler
    from .control_plane import ControlPlaneHandler
    from .billing.cost_dashboard import CostDashboardHandler
    from .costs import CostHandler
    from .critique import CritiqueHandler
    from .debate_stats import DebateStatsHandler
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
    from .debates.decision_package import DecisionPackageHandler
    from .debates.share import DebateShareHandler
    from .decision import DecisionHandler
    from .decisions import DecisionExplainHandler
    from .deliberations import DeliberationsHandler
    from .dependency_analysis import DependencyAnalysisHandler
    from .devices import DeviceHandler
    from .docs import DocsHandler
    from .dr_handler import DRHandler
    from .email import EmailHandler
    from .email_debate import EmailDebateHandler
    from .email_services import EmailServicesHandler
    from .email_triage import EmailTriageHandler
    from .endpoint_analytics import EndpointAnalyticsHandler
    from .erc8004 import ERC8004Handler
    from .evaluation import EvaluationHandler
    from .evolution import EvolutionABTestingHandler, EvolutionHandler
    from .expenses import ExpenseHandler
    from .explainability import ExplainabilityHandler
    from .external_agents import ExternalAgentsHandler
    from .external_integrations import ExternalIntegrationsHandler
    from .feature_flags import FeatureFlagsHandler
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
    from .features.control_plane import AgentDashboardHandler
    from .features.gmail_labels import GmailLabelsHandler
    from .features.gmail_threads import GmailThreadsHandler
    from .features.outlook import OutlookHandler
    from .feedback import FeedbackRoutesHandler
    from .gallery import GalleryHandler
    from .gastown_dashboard import GasTownDashboardHandler
    from .gateway_agents_handler import GatewayAgentsHandler
    from .gateway_config_handler import GatewayConfigHandler
    from .gateway_credentials_handler import GatewayCredentialsHandler
    from .gateway_handler import GatewayHandler
    from .gateway_health_handler import GatewayHealthHandler
    from .gauntlet import GauntletHandler
    from .gdpr_deletion import GDPRDeletionHandler
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
    from .github.audit_bridge import AuditGitHubBridgeHandler
    from .github.pr_review import PRReviewHandler
    from .goal_canvas import GoalCanvasHandler
    from .hybrid_debate_handler import HybridDebateHandler
    from .idea_canvas import IdeaCanvasHandler
    from .integrations.automation import AutomationHandler
    from .integrations.health import IntegrationHealthHandler
    from .integration_management import (
        IntegrationsHandler as IntegrationManagementHandler,
    )
    from .introspection import IntrospectionHandler
    from .invoices import InvoiceHandler
    from .knowledge.adapters import KMAdapterStatusHandler
    from .knowledge.checkpoints import KMCheckpointHandler
    from .knowledge.sharing_notifications import SharingNotificationsHandler
    from .knowledge_base import KnowledgeHandler, KnowledgeMoundHandler
    from .knowledge_chat import KnowledgeChatHandler
    from .laboratory import LaboratoryHandler
    from .marketplace_browse import MarketplaceBrowseHandler
    from .memory import (
        CoordinatorHandler,
        InsightsHandler,
        LearningHandler,
        MemoryAnalyticsHandler,
        MemoryHandler,
    )
    from .memory.unified_handler import UnifiedMemoryHandler
    from .metrics import MetricsHandler
    from .metrics_endpoint import UnifiedMetricsHandler
    from .ml import MLHandler
    from .moderation import ModerationHandler
    from .moderation_analytics import ModerationAnalyticsHandler
    from .moments import MomentsHandler
    from .nomic import NomicHandler
    from .notifications.history import NotificationHistoryHandler
    from .notifications.preferences import NotificationPreferencesHandler
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
    from .orchestration_canvas import OrchestrationCanvasHandler
    from .organizations import OrganizationsHandler
    from .partner import PartnerHandler
    from .payments.handler import PaymentRoutesHandler
    from .persona import PersonaHandler
    from .pipeline_graph import PipelineGraphHandler
    from .pipeline.plans import PlanManagementHandler
    from .pipeline.provenance_explorer import ProvenanceExplorerHandler
    from .pipeline.transitions import PipelineTransitionsHandler
    from .pipeline.universal_graph import UniversalGraphHandler
    from .plans import PlansHandler
    from .playbooks import PlaybookHandler
    from .playground import PlaygroundHandler
    from .policy import PolicyHandler
    from .privacy import PrivacyHandler
    from .public import StatusPageHandler
    from .queue import QueueHandler
    from .receipt_export import ReceiptExportHandler
    from .receipts import ReceiptsHandler
    from .replays import ReplaysHandler
    from .repository import RepositoryHandler
    from .reviews import ReviewsHandler
    from .rlm import RLMContextHandler
    from .routing import RoutingHandler
    from .scim_handler import SCIMHandler
    from .selection import SelectionHandler
    from .skill_marketplace import SkillMarketplaceHandler
    from .skills import SkillsHandler
    from .slo import SLOHandler
    from .sme.budget_controls import BudgetControlsHandler
    from .sme.receipt_delivery import ReceiptDeliveryHandler
    from .sme.slack_workspace import SlackWorkspaceHandler
    from .sme.teams_workspace import TeamsWorkspaceHandler
    from .sme_success_dashboard import SMESuccessDashboardHandler
    from .sme_usage_dashboard import SMEUsageDashboardHandler
    from .social import (
        CollaborationHandlers,
        RelationshipHandler,
        SlackHandler,
        SocialMediaHandler,
        get_collaboration_handlers,
    )
    from .social.channel_health import ChannelHealthHandler
    from .social.discord_oauth import DiscordOAuthHandler
    from .social.notifications import NotificationsHandler
    from .social.sharing import SharingHandler
    from .social.slack_oauth import SlackOAuthHandler
    from .social.teams import TeamsIntegrationHandler
    from .social.teams_oauth import TeamsOAuthHandler
    from .sso import SSOHandler
    from .streaming.handler import StreamingConnectorHandler
    from .tasks.execution import TaskExecutionHandler
    from .template_discovery import TemplateDiscoveryHandler
    from .template_marketplace import TemplateMarketplaceHandler
    from .threat_intel import ThreatIntelHandler
    from .tournaments import TournamentHandler
    from .training import TrainingHandler
    from .transcription import TranscriptionHandler
    from .uncertainty import UncertaintyHandler
    from .usage_metering import UsageMeteringHandler
    from .verification import FormalVerificationHandler, VerificationHandler
    from .verticals import VerticalsHandler
    from .webhooks import WebhookHandler
    from .workflow_templates import (
        SMEWorkflowsHandler,
        TemplateRecommendationsHandler,
        WorkflowCategoriesHandler,
        WorkflowPatternTemplatesHandler,
        WorkflowPatternsHandler,
        WorkflowTemplatesHandler,
    )
    from .workflows import WorkflowHandler
    from .workflows.builder import WorkflowBuilderHandler
    from .workflows.registry import TemplateRegistryHandler
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
    "CostDashboardHandler": Stability.STABLE,
    "CostHandler": Stability.STABLE,
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
    "BudgetHandler": Stability.STABLE,
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
    "SecurityDebateHandler": Stability.STABLE,
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
    "QueueHandler": Stability.STABLE,
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
    "ExpenseHandler": Stability.STABLE,
    "InvoiceHandler": Stability.STABLE,
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
    "OpenClawGatewayHandler": Stability.STABLE,
    "GatewayHealthHandler": Stability.STABLE,
    "GatewayAgentsHandler": Stability.STABLE,
    "GatewayCredentialsHandler": Stability.STABLE,
    "HybridDebateHandler": Stability.STABLE,
    "ERC8004Handler": Stability.STABLE,
    "AudienceSuggestionsHandler": Stability.EXPERIMENTAL,
    # Governance
    "OutcomeHandler": Stability.STABLE,
    # --- Newly registered handlers ---
    # admin/ sub-handlers
    "CreditsAdminHandler": Stability.EXPERIMENTAL,
    "EmergencyAccessHandler": Stability.EXPERIMENTAL,
    "FeatureFlagAdminHandler": Stability.EXPERIMENTAL,
    "LivenessHandler": Stability.STABLE,
    "ReadinessHandler": Stability.STABLE,
    "StorageHealthHandler": Stability.EXPERIMENTAL,
    # agents/ sub-handlers
    "AgentRecommendationHandler": Stability.EXPERIMENTAL,
    "FeedbackHandler": Stability.EXPERIMENTAL,
    # canvas pipeline stages
    "ActionCanvasHandler": Stability.EXPERIMENTAL,
    "GoalCanvasHandler": Stability.EXPERIMENTAL,
    "IdeaCanvasHandler": Stability.EXPERIMENTAL,
    "OrchestrationCanvasHandler": Stability.EXPERIMENTAL,
    # connectors
    "ConnectorManagementHandler": Stability.EXPERIMENTAL,
    # debates/ sub-handlers
    "DebateShareHandler": Stability.EXPERIMENTAL,
    "DebateStatsHandler": Stability.STABLE,
    "DecisionPackageHandler": Stability.EXPERIMENTAL,
    # email-related
    "EmailDebateHandler": Stability.EXPERIMENTAL,
    "EmailTriageHandler": Stability.EXPERIMENTAL,
    # features/ sub-handlers
    "AgentDashboardHandler": Stability.EXPERIMENTAL,
    "OutlookHandler": Stability.EXPERIMENTAL,
    # gateway
    "GatewayConfigHandler": Stability.EXPERIMENTAL,
    # github
    "AuditGitHubBridgeHandler": Stability.EXPERIMENTAL,
    "PRReviewHandler": Stability.EXPERIMENTAL,
    # integrations
    "AutomationHandler": Stability.EXPERIMENTAL,
    "IntegrationHealthHandler": Stability.EXPERIMENTAL,
    # knowledge/ sub-handlers
    "KMAdapterStatusHandler": Stability.EXPERIMENTAL,
    "SharingNotificationsHandler": Stability.EXPERIMENTAL,
    # memory/ sub-handlers
    "UnifiedMemoryHandler": Stability.EXPERIMENTAL,
    # notifications
    "NotificationHistoryHandler": Stability.EXPERIMENTAL,
    "NotificationPreferencesHandler": Stability.EXPERIMENTAL,
    # payments
    "PaymentRoutesHandler": Stability.EXPERIMENTAL,
    # pipeline
    "PipelineGraphHandler": Stability.EXPERIMENTAL,
    "PipelineTransitionsHandler": Stability.EXPERIMENTAL,
    "PlanManagementHandler": Stability.EXPERIMENTAL,
    "ProvenanceExplorerHandler": Stability.EXPERIMENTAL,
    "UniversalGraphHandler": Stability.EXPERIMENTAL,
    # sme/ sub-handlers
    "BudgetControlsHandler": Stability.EXPERIMENTAL,
    "ReceiptDeliveryHandler": Stability.EXPERIMENTAL,
    "SlackWorkspaceHandler": Stability.EXPERIMENTAL,
    "TeamsWorkspaceHandler": Stability.EXPERIMENTAL,
    # social/ sub-handlers
    "ChannelHealthHandler": Stability.EXPERIMENTAL,
    "DiscordOAuthHandler": Stability.EXPERIMENTAL,
    "NotificationsHandler": Stability.EXPERIMENTAL,
    "SharingHandler": Stability.EXPERIMENTAL,
    "SlackOAuthHandler": Stability.EXPERIMENTAL,
    "TeamsOAuthHandler": Stability.EXPERIMENTAL,
    # streaming
    "StreamingConnectorHandler": Stability.EXPERIMENTAL,
    # tasks
    "TaskExecutionHandler": Stability.EXPERIMENTAL,
    # top-level handlers
    "AuditTrailHandler": Stability.STABLE,
    "BenchmarkingHandler": Stability.EXPERIMENTAL,
    "ComplianceReportHandler": Stability.EXPERIMENTAL,
    "ContextBudgetHandler": Stability.EXPERIMENTAL,
    "DRHandler": Stability.EXPERIMENTAL,
    "FeatureFlagsHandler": Stability.EXPERIMENTAL,
    "FeedbackRoutesHandler": Stability.EXPERIMENTAL,
    "GasTownDashboardHandler": Stability.EXPERIMENTAL,
    "GDPRDeletionHandler": Stability.STABLE,
    "MarketplaceBrowseHandler": Stability.EXPERIMENTAL,
    "ModerationHandler": Stability.EXPERIMENTAL,
    "ModerationAnalyticsHandler": Stability.EXPERIMENTAL,
    "PartnerHandler": Stability.EXPERIMENTAL,
    "PlansHandler": Stability.EXPERIMENTAL,
    "PlaybookHandler": Stability.EXPERIMENTAL,
    "PlaygroundHandler": Stability.EXPERIMENTAL,
    "ReceiptExportHandler": Stability.STABLE,
    "ReceiptsHandler": Stability.STABLE,
    "SkillMarketplaceHandler": Stability.EXPERIMENTAL,
    "SMESuccessDashboardHandler": Stability.EXPERIMENTAL,
    "SMEWorkflowsHandler": Stability.EXPERIMENTAL,
    "SSOHandler": Stability.STABLE,
    "TemplateDiscoveryHandler": Stability.EXPERIMENTAL,
    "TemplateRegistryHandler": Stability.EXPERIMENTAL,
    "ThreatIntelHandler": Stability.EXPERIMENTAL,
    "UnifiedMetricsHandler": Stability.EXPERIMENTAL,
    # workflows/ sub-handlers
    "WorkflowBuilderHandler": Stability.EXPERIMENTAL,
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
    "DecisionPipelineHandler",
    "DecisionHandler",
    "CostDashboardHandler",
    "CostHandler",
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
    "SecurityDebateHandler",
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
    "UnifiedApprovalsHandler",
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
    # Audience suggestions
    "AudienceSuggestionsHandler",
    # Spectate (real-time debate observation bridge)
    "SpectateStreamHandler",
    # --- Newly registered handlers ---
    # admin/ sub-handlers
    "CreditsAdminHandler",
    "EmergencyAccessHandler",
    "FeatureFlagAdminHandler",
    "LivenessHandler",
    "ReadinessHandler",
    "StorageHealthHandler",
    # agents/ sub-handlers
    "AgentRecommendationHandler",
    "FeedbackHandler",
    # canvas pipeline stages
    "ActionCanvasHandler",
    "GoalCanvasHandler",
    "IdeaCanvasHandler",
    "OrchestrationCanvasHandler",
    # connectors
    "ConnectorManagementHandler",
    # debates/ sub-handlers
    "DebateShareHandler",
    "DebateStatsHandler",
    "DecisionPackageHandler",
    # email-related
    "EmailDebateHandler",
    "EmailTriageHandler",
    # features/ sub-handlers
    "AgentDashboardHandler",
    "OutlookHandler",
    # gateway
    "GatewayConfigHandler",
    # github
    "AuditGitHubBridgeHandler",
    "PRReviewHandler",
    # integrations
    "AutomationHandler",
    "IntegrationHealthHandler",
    # knowledge/ sub-handlers
    "KMAdapterStatusHandler",
    "SharingNotificationsHandler",
    # memory/ sub-handlers
    "UnifiedMemoryHandler",
    # notifications
    "NotificationHistoryHandler",
    "NotificationPreferencesHandler",
    # payments
    "PaymentRoutesHandler",
    # pipeline
    "PipelineGraphHandler",
    "PipelineTransitionsHandler",
    "PlanManagementHandler",
    "ProvenanceExplorerHandler",
    "UniversalGraphHandler",
    # sme/ sub-handlers
    "BudgetControlsHandler",
    "ReceiptDeliveryHandler",
    "SlackWorkspaceHandler",
    "TeamsWorkspaceHandler",
    # social/ sub-handlers
    "ChannelHealthHandler",
    "DiscordOAuthHandler",
    "NotificationsHandler",
    "SharingHandler",
    "SlackOAuthHandler",
    "TeamsOAuthHandler",
    # streaming
    "StreamingConnectorHandler",
    # tasks
    "TaskExecutionHandler",
    # top-level handlers
    "AuditTrailHandler",
    "BenchmarkingHandler",
    "ComplianceReportHandler",
    "ContextBudgetHandler",
    "DRHandler",
    "FeatureFlagsHandler",
    "FeedbackRoutesHandler",
    "GasTownDashboardHandler",
    "GDPRDeletionHandler",
    "MarketplaceBrowseHandler",
    "ModerationHandler",
    "ModerationAnalyticsHandler",
    "PartnerHandler",
    "PlansHandler",
    "PlaybookHandler",
    "PlaygroundHandler",
    "ReceiptExportHandler",
    "ReceiptsHandler",
    "SkillMarketplaceHandler",
    "SMESuccessDashboardHandler",
    "SMEWorkflowsHandler",
    "SSOHandler",
    "TemplateDiscoveryHandler",
    "TemplateRegistryHandler",
    "ThreatIntelHandler",
    "UnifiedMetricsHandler",
    # workflows/ sub-handlers
    "WorkflowBuilderHandler",
    # Stability utilities
    "HANDLER_STABILITY",
    "get_handler_stability",
    "get_all_handler_stability",
]
