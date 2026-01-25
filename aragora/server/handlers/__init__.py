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
from .analytics_metrics import AnalyticsMetricsHandler
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
from .budgets import BudgetHandler
from .usage_metering import UsageMeteringHandler  # Token-level usage metering
from .breakpoints import BreakpointsHandler
from .features import AudioHandler  # Moved to features/
from .transcription import TranscriptionHandler
from .features import BroadcastHandler  # Moved to features/
from .agents import CalibrationHandler  # Moved to agents/
from .checkpoints import CheckpointHandler
from .composite import CompositeHandler
from .consensus import ConsensusHandler
from .control_plane import ControlPlaneHandler
from .orchestration import OrchestrationHandler
from .decisions import DecisionExplainHandler
from .decision import DecisionHandler
from .deliberations import DeliberationsHandler
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
from .features import AuditSessionsHandler  # Audit sessions API
from .features import CloudStorageHandler  # Cloud storage integration API
from .features import CodebaseAuditHandler  # Codebase audit API
from .features import ConnectorsHandler  # Unified connectors registry API
from .features import CrossPlatformAnalyticsHandler  # Cross-platform analytics API
from .features import DevOpsHandler  # DevOps integrations API
from .features import DocumentHandler  # Moved to features/
from .features import DocumentBatchHandler  # Batch document upload
from .features import DocumentQueryHandler  # NL document querying
from .features import EmailWebhooksHandler  # Email webhook endpoints
from .features import EvidenceHandler  # Moved to features/
from .features import EvidenceEnrichmentHandler  # Evidence enrichment for findings
from .features import FindingWorkflowHandler  # Finding workflow management
from .features import FolderUploadHandler  # Folder upload support
from .features import GmailIngestHandler  # Gmail inbox ingestion API
from .features import GmailQueryHandler  # Gmail querying API
from .features import IntegrationsHandler  # Integration config API
from .features import LegalHandler  # Legal integrations API
from .features import MarketplaceHandler  # Template marketplace API
from .features import ReconciliationHandler  # Financial reconciliation API
from .features import RoutingRulesHandler  # Routing rules API
from .features import RLMHandler  # RLM operations API
from .features import SchedulerHandler  # Audit scheduling
from .features import SmartUploadHandler  # Smart upload and classification API
from .features import UnifiedInboxHandler  # Unified inbox API
from .evaluation import EvaluationHandler
from .evolution import EvolutionABTestingHandler  # Moved to evolution/
from .evolution import EvolutionHandler  # Moved to evolution/
from .features import FeaturesHandler  # Moved to features/
from .verification import FormalVerificationHandler  # Moved to verification/
from .gallery import GalleryHandler
from .gauntlet import GauntletHandler
from .gauntlet_v1 import (
    GauntletSchemaHandler,
    GauntletAllSchemasHandler,
    GauntletTemplatesListHandler,
    GauntletTemplateHandler,
    GauntletReceiptExportHandler,
    GauntletHeatmapExportHandler,
    GauntletValidateReceiptHandler,
    GAUNTLET_V1_HANDLERS,
)
from .genesis import GenesisHandler
from .debates import GraphDebatesHandler  # Moved to debates/
from .admin import HealthHandler  # Moved to admin/
from .memory import InsightsHandler  # Moved to memory/
from .introspection import IntrospectionHandler
from .knowledge_base import KnowledgeHandler, KnowledgeMoundHandler
from .knowledge.checkpoints import KMCheckpointHandler
from .laboratory import LaboratoryHandler
from .agents import LeaderboardViewHandler  # Moved to agents/
from .memory import LearningHandler  # Moved to memory/
from .debates import MatrixDebatesHandler  # Moved to debates/
from .memory import MemoryHandler  # memory/ subdirectory
from .memory import MemoryAnalyticsHandler  # Moved to memory/
from .memory import CoordinatorHandler  # Memory coordinator API
from .metrics import MetricsHandler
from .slo import SLOHandler
from .moments import MomentsHandler
from .nomic import NomicHandler
from .oauth import OAuthHandler
from .onboarding import (
    handle_get_flow,
    handle_init_flow,
    handle_update_step,
    handle_get_templates,
    handle_first_debate,
    handle_quick_start,
    handle_analytics,
    get_onboarding_handlers,
)
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
from .features import AdvertisingHandler  # Unified advertising platforms API
from .features import AnalyticsPlatformsHandler  # Unified analytics platforms API
from .features import CRMHandler  # Unified CRM platforms API
from .features import SupportHandler  # Unified support platforms API
from .features import EcommerceHandler  # Unified ecommerce platforms API
from .social import RelationshipHandler  # Moved to social/
from .replays import ReplaysHandler
from .reviews import ReviewsHandler
from .routing import RoutingHandler
from .ml import MLHandler
from .rlm import RLMContextHandler
from .selection import SelectionHandler
from .social import SlackHandler  # Moved to social/
from .social.teams import TeamsIntegrationHandler
from .social import SocialMediaHandler
from .admin import SystemHandler  # Moved to admin/
from .tournaments import TournamentHandler
from .training import TrainingHandler
from .verification import VerificationHandler
from .webhooks import WebhookHandler
from .workflows import WorkflowHandler
from .workflow_templates import (
    WorkflowTemplatesHandler,
    WorkflowCategoriesHandler,
    WorkflowPatternsHandler,
    WorkflowPatternTemplatesHandler,
    TemplateRecommendationsHandler,
)
from .template_marketplace import TemplateMarketplaceHandler
from .email import EmailHandler  # Email prioritization API
from .email_services import EmailServicesHandler  # Email services (follow-up, snooze, categories)
from .dependency_analysis import DependencyAnalysisHandler  # Dependency analysis API

# Accounting handlers
from .expenses import ExpenseHandler  # Expense tracking API
from .invoices import InvoiceHandler  # Invoice processing API
from .ar_automation import ARAutomationHandler  # AR automation API
from .ap_automation import APAutomationHandler  # AP automation API

# Code review handler
from .code_review import CodeReviewHandler  # Multi-agent code review API
from .codebase import IntelligenceHandler  # Code intelligence (AST, call graphs, dead code)
from .social import CollaborationHandlers, get_collaboration_handlers  # Moved to social/
from .bots import (
    DiscordHandler,
    GoogleChatHandler,
    TeamsHandler,
    TelegramHandler,
    WhatsAppHandler,
    ZoomHandler,
)  # Bot platform handlers
from .explainability import ExplainabilityHandler  # Decision explainability API
from .a2a import A2AHandler  # A2A protocol handler
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
    AnalyticsMetricsHandler,  # Debate metrics and agent performance analytics
    CrossPlatformAnalyticsHandler,  # Cross-platform analytics aggregation
    MetricsHandler,
    SLOHandler,  # SLO tracking and monitoring
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
    DecisionExplainHandler,  # Decision explainability API
    DecisionHandler,  # Unified decision routing API
    ControlPlaneHandler,  # Enterprise control plane API
    DeliberationsHandler,  # Multi-deliberation dashboard API
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
    SmartUploadHandler,  # Smart upload with file classification
    CloudStorageHandler,  # Cloud storage integration API
    FindingWorkflowHandler,  # Finding workflow management
    EvidenceEnrichmentHandler,  # Evidence enrichment for findings
    SchedulerHandler,  # Audit scheduling
    AuditSessionsHandler,  # Audit session tracking API
    VerificationHandler,
    AuditingHandler,
    DashboardHandler,
    PersonaHandler,
    IntrospectionHandler,
    CalibrationHandler,
    CheckpointHandler,
    RoutingHandler,
    RoutingRulesHandler,  # Routing rules management API
    MLHandler,  # ML capabilities API (routing, scoring, consensus)
    RLMContextHandler,  # RLM context compression and query API
    RLMHandler,  # RLM operations API
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
    KnowledgeMoundHandler,  # Extended Knowledge Mound API - STABLE
    KMCheckpointHandler,  # KM checkpoint backup/restore API
    GalleryHandler,
    BreakpointsHandler,
    LearningHandler,
    AuthHandler,
    BillingHandler,
    BudgetHandler,
    UsageMeteringHandler,  # Token-level usage metering for ENTERPRISE_PLUS
    OrganizationsHandler,
    OAuthHandler,
    FeaturesHandler,
    ConnectorsHandler,  # Unified connectors registry
    TeamsIntegrationHandler,  # Microsoft Teams integration endpoints
    IntegrationsHandler,  # Integration config API
    GmailIngestHandler,  # Gmail OAuth + sync ingestion API
    GmailQueryHandler,  # Gmail search/query API
    UnifiedInboxHandler,  # Unified inbox API
    EmailWebhooksHandler,  # Unified inbox email webhooks
    MemoryAnalyticsHandler,
    # Gauntlet v1 API (versioned endpoints - more specific paths)
    GauntletSchemaHandler,
    GauntletAllSchemasHandler,
    GauntletTemplatesListHandler,
    GauntletTemplateHandler,
    GauntletReceiptExportHandler,
    GauntletHeatmapExportHandler,
    GauntletValidateReceiptHandler,
    GauntletHandler,  # Legacy endpoints
    ReviewsHandler,
    FormalVerificationHandler,
    SlackHandler,
    EvidenceHandler,
    WebhookHandler,
    CodebaseAuditHandler,  # Codebase audit API
    AdminHandler,
    PolicyHandler,  # Policy and compliance management API
    PrivacyHandler,
    QueueHandler,  # Job queue management API
    RepositoryHandler,  # Repository indexing API - STABLE
    UncertaintyHandler,  # Uncertainty estimation API - STABLE
    VerticalsHandler,  # Vertical specialist API
    WorkspaceHandler,  # Enterprise workspace/privacy management
    WorkflowHandler,  # Enterprise workflow engine API
    WorkflowTemplatesHandler,  # Workflow template marketplace API
    WorkflowCategoriesHandler,  # Workflow template categories
    WorkflowPatternsHandler,  # Workflow patterns listing
    WorkflowPatternTemplatesHandler,  # Pattern-based workflow templates
    TemplateRecommendationsHandler,  # Template recommendations for onboarding
    TemplateMarketplaceHandler,  # Community template marketplace
    MarketplaceHandler,  # Marketplace API
    TrainingHandler,  # RLM training data collection API
    EmailHandler,  # Email prioritization API
    EmailServicesHandler,  # Email services (follow-up, snooze, categories)
    DependencyAnalysisHandler,  # Codebase dependency analysis API
    IntelligenceHandler,  # Code intelligence (AST, call graphs, dead code)
    # Bot platform handlers
    DiscordHandler,  # Discord Interactions API
    GoogleChatHandler,  # Google Chat Cards API
    TeamsHandler,  # Microsoft Teams Bot Framework
    TelegramHandler,  # Telegram Bot API webhooks
    WhatsAppHandler,  # WhatsApp Cloud API
    ZoomHandler,  # Zoom webhooks and chat
    # Explainability
    ExplainabilityHandler,  # Decision explainability API
    # Protocols
    A2AHandler,  # A2A protocol handler
    # Autonomous operations handlers (Phase 5)
    ApprovalHandler,  # Human-in-the-loop approval flows
    AlertHandler,  # Alert management and thresholds
    TriggerHandler,  # Scheduled debate triggers
    MonitoringHandler,  # Trend and anomaly monitoring
    AutonomousLearningHandler,  # Continuous learning (ELO, patterns, calibration)
    # Accounting handlers (Phase 4 - SME Vertical)
    ExpenseHandler,  # Expense tracking and receipt processing
    InvoiceHandler,  # Invoice processing and PO matching
    ARAutomationHandler,  # Accounts receivable automation
    APAutomationHandler,  # Accounts payable automation
    ReconciliationHandler,  # Accounting reconciliation API
    # Code review handler (Phase 5 - SME Vertical)
    CodeReviewHandler,  # Multi-agent code review
    LegalHandler,  # Legal integrations API
    DevOpsHandler,  # DevOps integrations API
    # Connector platform handlers (unified APIs)
    AdvertisingHandler,  # Google Ads, Meta, LinkedIn, Microsoft, Twitter, TikTok
    AnalyticsPlatformsHandler,  # Metabase, GA4, Mixpanel, Segment
    CRMHandler,  # HubSpot, Salesforce
    SupportHandler,  # Zendesk, Freshdesk, Intercom, HelpScout
    EcommerceHandler,  # Shopify, ShipStation, Walmart
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
    "AnalyticsMetricsHandler": Stability.EXPERIMENTAL,  # Debate metrics and agent performance analytics
    "CrossPlatformAnalyticsHandler": Stability.EXPERIMENTAL,  # Cross-platform analytics aggregation
    "ConsensusHandler": Stability.STABLE,
    "MetricsHandler": Stability.STABLE,
    "SLOHandler": Stability.EXPERIMENTAL,  # SLO tracking and monitoring
    "MemoryHandler": Stability.STABLE,
    "CoordinatorHandler": Stability.STABLE,
    "LeaderboardViewHandler": Stability.STABLE,
    "ReplaysHandler": Stability.STABLE,
    "FeaturesHandler": Stability.STABLE,
    "ConnectorsHandler": Stability.EXPERIMENTAL,  # Unified connectors registry
    "IntegrationsHandler": Stability.EXPERIMENTAL,  # Integration config API
    "TeamsIntegrationHandler": Stability.EXPERIMENTAL,  # Teams bot integration endpoints
    "AuthHandler": Stability.STABLE,
    # Extended - Stable
    "TournamentHandler": Stability.STABLE,
    "DecisionHandler": Stability.STABLE,  # Unified decision routing API - 13 tests
    "ControlPlaneHandler": Stability.STABLE,  # Enterprise control plane - 122 tests
    "CritiqueHandler": Stability.STABLE,
    "RelationshipHandler": Stability.STABLE,
    "DashboardHandler": Stability.STABLE,
    "RoutingHandler": Stability.STABLE,
    "RoutingRulesHandler": Stability.EXPERIMENTAL,  # Routing rules management
    "CompositeHandler": Stability.EXPERIMENTAL,  # Composite API endpoints - new
    "MLHandler": Stability.EXPERIMENTAL,  # ML capabilities API - new
    "RLMContextHandler": Stability.STABLE,  # RLM context compression and query API - 86 tests
    "RLMHandler": Stability.EXPERIMENTAL,  # RLM operations API
    "SelectionHandler": Stability.STABLE,  # Selection plugin API
    # Promoted to Stable (Jan 2026) - tested in production
    "BillingHandler": Stability.STABLE,  # Transaction tests, Stripe webhooks
    "BudgetHandler": Stability.EXPERIMENTAL,  # Budget management API
    "OAuthHandler": Stability.STABLE,  # OAuth flow tests, Google integration
    "AudioHandler": Stability.STABLE,  # Podcast generation, TTS
    "TranscriptionHandler": Stability.EXPERIMENTAL,  # Speech-to-text transcription - new
    "TrainingHandler": Stability.EXPERIMENTAL,  # RLM training data collection - new
    "VerificationHandler": Stability.STABLE,  # Z3 formal verification
    "PulseHandler": Stability.STABLE,  # Trending topics API
    "GalleryHandler": Stability.STABLE,  # Consensus gallery
    "GauntletHandler": Stability.STABLE,  # Adversarial validation - 6+ test files
    # Gauntlet v1 API (versioned, documented endpoints)
    "GauntletSchemaHandler": Stability.STABLE,
    "GauntletAllSchemasHandler": Stability.STABLE,
    "GauntletTemplatesListHandler": Stability.STABLE,
    "GauntletTemplateHandler": Stability.STABLE,
    "GauntletReceiptExportHandler": Stability.STABLE,
    "GauntletHeatmapExportHandler": Stability.STABLE,
    "GauntletValidateReceiptHandler": Stability.STABLE,
    "BeliefHandler": Stability.STABLE,  # Belief networks - 4 test files
    "CalibrationHandler": Stability.STABLE,  # Agent calibration - 4 test files
    "PersonaHandler": Stability.STABLE,  # Agent personas - 2 test files
    # Promoted to Stable (Jan 2026) - comprehensive test coverage
    "GraphDebatesHandler": Stability.STABLE,  # 7 test files, 95+ tests
    "MatrixDebatesHandler": Stability.STABLE,  # Handler tests + integration
    "EvaluationHandler": Stability.STABLE,  # LLM-as-Judge evaluation - 11 tests
    "EvolutionHandler": Stability.STABLE,  # 7 test files, 66+ tests
    "EvolutionABTestingHandler": Stability.STABLE,  # AB testing with evolution
    "LaboratoryHandler": Stability.STABLE,  # 3 test files, 70+ tests
    "IntrospectionHandler": Stability.STABLE,  # 2 test files, 53+ tests
    "LearningHandler": Stability.STABLE,  # 2 test files, 66+ tests
    "MemoryAnalyticsHandler": Stability.STABLE,  # Handler tests, 23+ tests
    "ProbesHandler": Stability.STABLE,  # 16 tests, capability probing
    "InsightsHandler": Stability.STABLE,  # 3 test files, 110+ tests
    "KnowledgeHandler": Stability.EXPERIMENTAL,  # Knowledge base API - new
    "KnowledgeMoundHandler": Stability.STABLE,  # Knowledge Mound API - Graduated from Phase A1
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
    "SmartUploadHandler": Stability.EXPERIMENTAL,  # Smart upload classification - new
    "CloudStorageHandler": Stability.EXPERIMENTAL,  # Cloud storage integration - new
    "FindingWorkflowHandler": Stability.EXPERIMENTAL,  # Finding workflow - new
    "EvidenceEnrichmentHandler": Stability.EXPERIMENTAL,  # Evidence enrichment - new
    "SchedulerHandler": Stability.EXPERIMENTAL,  # Audit scheduling - new
    "AuditSessionsHandler": Stability.EXPERIMENTAL,  # Audit session tracking - new
    "BreakpointsHandler": Stability.STABLE,  # 34 tests, debate breakpoints
    "SlackHandler": Stability.EXPERIMENTAL,  # Slack integration - new
    "EvidenceHandler": Stability.STABLE,  # Evidence collection and storage
    "WebhookHandler": Stability.STABLE,  # Webhook registration and delivery
    "AdminHandler": Stability.STABLE,  # Admin panel backend API
    "PolicyHandler": Stability.EXPERIMENTAL,  # Policy and compliance management - new
    "PrivacyHandler": Stability.STABLE,  # GDPR/CCPA data export and deletion
    "WorkspaceHandler": Stability.EXPERIMENTAL,  # Enterprise workspace/privacy management
    "WorkflowHandler": Stability.STABLE,  # Enterprise workflow engine API - 48 tests
    "WorkflowTemplatesHandler": Stability.STABLE,  # Workflow template marketplace API - new
    "WorkflowCategoriesHandler": Stability.STABLE,  # Workflow template categories - new
    "WorkflowPatternsHandler": Stability.STABLE,  # Workflow patterns listing - new
    "WorkflowPatternTemplatesHandler": Stability.STABLE,  # Pattern-based workflow templates - new
    "TemplateRecommendationsHandler": Stability.STABLE,  # Template recommendations for onboarding - new
    "TemplateMarketplaceHandler": Stability.EXPERIMENTAL,  # Community template marketplace - new
    "MarketplaceHandler": Stability.EXPERIMENTAL,  # Marketplace API - new
    "QueueHandler": Stability.EXPERIMENTAL,  # Job queue management API - Phase A1
    "RepositoryHandler": Stability.STABLE,  # Repository indexing API - Graduated from Phase A3
    "UncertaintyHandler": Stability.STABLE,  # Uncertainty estimation API - Graduated from Phase A1
    "VerticalsHandler": Stability.EXPERIMENTAL,  # Vertical specialist API - Phase A1
    # Bot platform handlers - Graduated Jan 2026
    "DiscordHandler": Stability.STABLE,  # Discord Interactions API - 14 tests
    "GoogleChatHandler": Stability.STABLE,  # Google Chat Cards API - follows Discord pattern
    "TeamsHandler": Stability.STABLE,  # Microsoft Teams Bot Framework - follows Discord pattern
    "TelegramHandler": Stability.STABLE,  # Telegram Bot API webhooks - 47 tests
    "WhatsAppHandler": Stability.STABLE,  # WhatsApp Cloud API - 48 tests
    "ZoomHandler": Stability.STABLE,  # Zoom webhooks and chat - 19 tests
    # Explainability
    "ExplainabilityHandler": Stability.EXPERIMENTAL,  # Decision explainability API - new
    # Protocols
    "A2AHandler": Stability.EXPERIMENTAL,  # A2A protocol handler - new
    # Autonomous operations handlers (Phase 5)
    "ApprovalHandler": Stability.EXPERIMENTAL,  # Human-in-the-loop approval flows - Phase 5.1
    "AlertHandler": Stability.EXPERIMENTAL,  # Alert management and thresholds - Phase 5.3
    "TriggerHandler": Stability.EXPERIMENTAL,  # Scheduled debate triggers - Phase 5.3
    "MonitoringHandler": Stability.EXPERIMENTAL,  # Trend and anomaly monitoring - Phase 5.3
    "AutonomousLearningHandler": Stability.EXPERIMENTAL,  # Continuous learning - Phase 5.2
    "EmailHandler": Stability.EXPERIMENTAL,  # Email prioritization API - new
    "EmailServicesHandler": Stability.EXPERIMENTAL,  # Email services (follow-up, snooze) - new
    "GmailIngestHandler": Stability.EXPERIMENTAL,  # Gmail OAuth + sync ingestion
    "GmailQueryHandler": Stability.EXPERIMENTAL,  # Gmail search/query API
    "UnifiedInboxHandler": Stability.PREVIEW,  # Unified inbox API - in-memory by default
    "EmailWebhooksHandler": Stability.EXPERIMENTAL,  # Unified inbox email webhooks
    "DependencyAnalysisHandler": Stability.EXPERIMENTAL,  # Dependency analysis API - new
    "CodebaseAuditHandler": Stability.EXPERIMENTAL,  # Codebase audit API - new
    # Accounting handlers (Phase 4 - SME Vertical)
    "ExpenseHandler": Stability.EXPERIMENTAL,  # Expense tracking and receipt processing - new
    "InvoiceHandler": Stability.EXPERIMENTAL,  # Invoice processing and PO matching - new
    "ARAutomationHandler": Stability.EXPERIMENTAL,  # Accounts receivable automation - new
    "APAutomationHandler": Stability.EXPERIMENTAL,  # Accounts payable automation - new
    "ReconciliationHandler": Stability.EXPERIMENTAL,  # Accounting reconciliation - new
    # Code review handler (Phase 5 - SME Vertical)
    "CodeReviewHandler": Stability.EXPERIMENTAL,  # Multi-agent code review - new
    "LegalHandler": Stability.EXPERIMENTAL,  # Legal integrations API - new
    "DevOpsHandler": Stability.EXPERIMENTAL,  # DevOps integrations API - new
    # Connector platform handlers (unified APIs)
    "AdvertisingHandler": Stability.EXPERIMENTAL,  # Unified advertising platforms API - new
    "AnalyticsPlatformsHandler": Stability.EXPERIMENTAL,  # Unified analytics platforms API - new
    "CRMHandler": Stability.EXPERIMENTAL,  # Unified CRM platforms API - new
    "SupportHandler": Stability.EXPERIMENTAL,  # Unified support platforms API - new
    "EcommerceHandler": Stability.EXPERIMENTAL,  # Unified ecommerce platforms API - new
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
    "AnalyticsMetricsHandler",
    "CrossPlatformAnalyticsHandler",
    "MetricsHandler",
    "SLOHandler",
    "ConsensusHandler",
    "BeliefHandler",
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
    "BudgetHandler",
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
    # Stability utilities
    "HANDLER_STABILITY",
    "get_handler_stability",
    "get_all_handler_stability",
]
