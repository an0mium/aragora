"""
Handler registry for modular HTTP endpoint routing.

This module provides centralized initialization and routing for all modular
HTTP handlers. The HandlerRegistryMixin can be mixed into request handler
classes to add modular routing capabilities.

Features:
- O(1) exact path lookup via route index
- LRU cached prefix matching for dynamic routes
- Lazy handler initialization
- API versioning support (/api/v1/... paths)

Usage:
    class MyHandler(HandlerRegistryMixin, BaseHTTPRequestHandler):
        pass
"""

import asyncio
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Dict, List, Optional, Tuple, Type

from aragora.server.versioning import (
    extract_version,
    strip_version_prefix,
    version_response_headers,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aragora.agents.personas import PersonaManager
    from aragora.agents.positions import PositionLedger
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    from aragora.memory.store import CritiqueStore
    from aragora.ranking.elo import EloSystem
    from aragora.server.handlers.base import BaseHandler
    from aragora.server.storage import DebateStorage

logger = logging.getLogger(__name__)

# Type alias for handler classes that may be None when handlers are unavailable
# This allows proper type hints without requiring type: ignore comments
HandlerType = Optional[Type[Any]]

# Handler class placeholders - set to actual classes on successful import
SystemHandler: HandlerType = None
HealthHandler: HandlerType = None
NomicHandler: HandlerType = None
DocsHandler: HandlerType = None
DebatesHandler: HandlerType = None
AgentsHandler: HandlerType = None
PulseHandler: HandlerType = None
AnalyticsHandler: HandlerType = None
MetricsHandler: HandlerType = None
ConsensusHandler: HandlerType = None
BeliefHandler: HandlerType = None
DecisionExplainHandler: HandlerType = None
DecisionHandler: HandlerType = None
CritiqueHandler: HandlerType = None
GenesisHandler: HandlerType = None
ReplaysHandler: HandlerType = None
TournamentHandler: HandlerType = None
MemoryHandler: HandlerType = None
LeaderboardViewHandler: HandlerType = None
DocumentHandler: HandlerType = None
DocumentBatchHandler: HandlerType = None
VerificationHandler: HandlerType = None
AuditingHandler: HandlerType = None
RelationshipHandler: HandlerType = None
MomentsHandler: HandlerType = None
PersonaHandler: HandlerType = None
DashboardHandler: HandlerType = None
IntrospectionHandler: HandlerType = None
CalibrationHandler: HandlerType = None
RoutingHandler: HandlerType = None
EvolutionHandler: HandlerType = None
EvolutionABTestingHandler: HandlerType = None
PluginsHandler: HandlerType = None
BroadcastHandler: HandlerType = None
AudioHandler: HandlerType = None
SocialMediaHandler: HandlerType = None
LaboratoryHandler: HandlerType = None
ProbesHandler: HandlerType = None
InsightsHandler: HandlerType = None
BreakpointsHandler: HandlerType = None
LearningHandler: HandlerType = None
GalleryHandler: HandlerType = None
AuthHandler: HandlerType = None
BillingHandler: HandlerType = None
BudgetHandler: HandlerType = None
CheckpointHandler: HandlerType = None
GraphDebatesHandler: HandlerType = None
MatrixDebatesHandler: HandlerType = None
FeaturesHandler: HandlerType = None
MemoryAnalyticsHandler: HandlerType = None
GauntletHandler: HandlerType = None
SlackHandler: HandlerType = None
SlackOAuthHandler: HandlerType = None
TeamsIntegrationHandler: HandlerType = None
TeamsOAuthHandler: HandlerType = None
DiscordOAuthHandler: HandlerType = None
OrganizationsHandler: HandlerType = None
OAuthHandler: HandlerType = None
ReviewsHandler: HandlerType = None
FormalVerificationHandler: HandlerType = None
EvaluationHandler: HandlerType = None
EvidenceHandler: HandlerType = None
FolderUploadHandler: HandlerType = None
WebhookHandler: HandlerType = None
WorkflowHandler: HandlerType = None
AdminHandler: HandlerType = None
SecurityHandler: HandlerType = None
ControlPlaneHandler: HandlerType = None
OrchestrationHandler: HandlerType = None
DeliberationsHandler: HandlerType = None
KnowledgeHandler: HandlerType = None
KnowledgeMoundHandler: HandlerType = None
PolicyHandler: HandlerType = None
QueueHandler: HandlerType = None
RLMContextHandler: HandlerType = None
TrainingHandler: HandlerType = None
TranscriptionHandler: HandlerType = None
UncertaintyHandler: HandlerType = None
VerticalsHandler: HandlerType = None
WorkspaceHandler: HandlerType = None
EmailHandler: HandlerType = None
GmailIngestHandler: HandlerType = None
GmailQueryHandler: HandlerType = None
GoogleChatHandler: HandlerType = None
ExplainabilityHandler: HandlerType = None
A2AHandler: HandlerType = None
CodeIntelligenceHandler: HandlerType = None
AdvertisingHandler: HandlerType = None
AnalyticsPlatformsHandler: HandlerType = None
CRMHandler: HandlerType = None
SupportHandler: HandlerType = None
EcommerceHandler: HandlerType = None
ReconciliationHandler: HandlerType = None
UnifiedInboxHandler: HandlerType = None
CodebaseAuditHandler: HandlerType = None
LegalHandler: HandlerType = None
DevOpsHandler: HandlerType = None
ReceiptsHandler: HandlerType = None
HandlerResult: HandlerType = None

# Import handlers with graceful fallback
try:
    from aragora.server.handlers import (
        AdminHandler as _AdminHandler,
    )
    from aragora.server.handlers.admin import (
        SecurityHandler as _SecurityHandler,
    )
    from aragora.server.handlers import (
        ControlPlaneHandler as _ControlPlaneHandler,
    )
    from aragora.server.handlers import (
        OrchestrationHandler as _OrchestrationHandler,
    )
    from aragora.server.handlers import (
        DeliberationsHandler as _DeliberationsHandler,
    )
    from aragora.server.handlers import (
        KnowledgeHandler as _KnowledgeHandler,
    )
    from aragora.server.handlers import (
        KnowledgeMoundHandler as _KnowledgeMoundHandler,
    )
    from aragora.server.handlers import (
        AgentsHandler as _AgentsHandler,
    )
    from aragora.server.handlers import (
        AnalyticsHandler as _AnalyticsHandler,
    )
    from aragora.server.handlers import (
        AudioHandler as _AudioHandler,
    )
    from aragora.server.handlers import (
        AuditingHandler as _AuditingHandler,
    )
    from aragora.server.handlers import (
        AuthHandler as _AuthHandler,
    )
    from aragora.server.handlers import (
        BeliefHandler as _BeliefHandler,
    )
    from aragora.server.handlers import (
        BillingHandler as _BillingHandler,
    )
    from aragora.server.handlers import (
        BudgetHandler as _BudgetHandler,
    )
    from aragora.server.handlers import (
        BreakpointsHandler as _BreakpointsHandler,
    )
    from aragora.server.handlers import (
        BroadcastHandler as _BroadcastHandler,
    )
    from aragora.server.handlers import (
        CalibrationHandler as _CalibrationHandler,
    )
    from aragora.server.handlers import (
        CheckpointHandler as _CheckpointHandler,
    )
    from aragora.server.handlers import (
        ConsensusHandler as _ConsensusHandler,
    )
    from aragora.server.handlers import (
        DecisionExplainHandler as _DecisionExplainHandler,
    )
    from aragora.server.handlers import (
        DecisionHandler as _DecisionHandler,
    )
    from aragora.server.handlers import (
        CritiqueHandler as _CritiqueHandler,
    )
    from aragora.server.handlers import (
        DashboardHandler as _DashboardHandler,
    )
    from aragora.server.handlers import (
        DebatesHandler as _DebatesHandler,
    )
    from aragora.server.handlers import (
        DocumentHandler as _DocumentHandler,
    )
    from aragora.server.handlers import (
        DocumentBatchHandler as _DocumentBatchHandler,
    )
    from aragora.server.handlers import (
        EvaluationHandler as _EvaluationHandler,
    )
    from aragora.server.handlers import (
        EvidenceHandler as _EvidenceHandler,
    )
    from aragora.server.handlers import (
        FolderUploadHandler as _FolderUploadHandler,
    )
    from aragora.server.handlers import (
        EvolutionABTestingHandler as _EvolutionABTestingHandler,
    )
    from aragora.server.handlers import (
        EvolutionHandler as _EvolutionHandler,
    )
    from aragora.server.handlers import (
        FeaturesHandler as _FeaturesHandler,
    )
    from aragora.server.handlers import (
        FormalVerificationHandler as _FormalVerificationHandler,
    )
    from aragora.server.handlers import (
        GalleryHandler as _GalleryHandler,
    )
    from aragora.server.handlers import (
        GauntletHandler as _GauntletHandler,
    )
    from aragora.server.handlers import (
        GenesisHandler as _GenesisHandler,
    )
    from aragora.server.handlers import (
        GraphDebatesHandler as _GraphDebatesHandler,
    )
    from aragora.server.handlers import (
        HandlerResult as _HandlerResult,
    )
    from aragora.server.handlers import (
        InsightsHandler as _InsightsHandler,
    )
    from aragora.server.handlers import (
        IntrospectionHandler as _IntrospectionHandler,
    )
    from aragora.server.handlers import (
        LaboratoryHandler as _LaboratoryHandler,
    )
    from aragora.server.handlers import (
        LeaderboardViewHandler as _LeaderboardViewHandler,
    )
    from aragora.server.handlers import (
        LearningHandler as _LearningHandler,
    )
    from aragora.server.handlers import (
        MatrixDebatesHandler as _MatrixDebatesHandler,
    )
    from aragora.server.handlers import (
        MemoryAnalyticsHandler as _MemoryAnalyticsHandler,
    )
    from aragora.server.handlers import (
        MemoryHandler as _MemoryHandler,
    )
    from aragora.server.handlers import (
        MetricsHandler as _MetricsHandler,
    )
    from aragora.server.handlers import (
        MomentsHandler as _MomentsHandler,
    )
    from aragora.server.handlers import (
        OAuthHandler as _OAuthHandler,
    )
    from aragora.server.handlers import (
        OrganizationsHandler as _OrganizationsHandler,
    )
    from aragora.server.handlers import (
        PersonaHandler as _PersonaHandler,
    )
    from aragora.server.handlers import (
        PluginsHandler as _PluginsHandler,
    )
    from aragora.server.handlers import (
        ProbesHandler as _ProbesHandler,
    )
    from aragora.server.handlers import (
        PulseHandler as _PulseHandler,
    )
    from aragora.server.handlers import (
        RelationshipHandler as _RelationshipHandler,
    )
    from aragora.server.handlers import (
        ReplaysHandler as _ReplaysHandler,
    )
    from aragora.server.handlers import (
        ReviewsHandler as _ReviewsHandler,
    )
    from aragora.server.handlers import (
        RoutingHandler as _RoutingHandler,
    )
    from aragora.server.handlers import (
        SlackHandler as _SlackHandler,
    )
    from aragora.server.handlers.social.slack_oauth import (
        SlackOAuthHandler as _SlackOAuthHandler,
    )
    from aragora.server.handlers.social.teams import (
        TeamsIntegrationHandler as _TeamsIntegrationHandler,
    )
    from aragora.server.handlers.social.teams_oauth import (
        TeamsOAuthHandler as _TeamsOAuthHandler,
    )
    from aragora.server.handlers.social.discord_oauth import (
        DiscordOAuthHandler as _DiscordOAuthHandler,
    )
    from aragora.server.handlers import (
        SocialMediaHandler as _SocialMediaHandler,
    )
    from aragora.server.handlers import (
        SystemHandler as _SystemHandler,
    )
    from aragora.server.handlers import (
        HealthHandler as _HealthHandler,
    )
    from aragora.server.handlers import (
        NomicHandler as _NomicHandler,
    )
    from aragora.server.handlers import (
        DocsHandler as _DocsHandler,
    )
    from aragora.server.handlers import (
        TournamentHandler as _TournamentHandler,
    )
    from aragora.server.handlers import (
        VerificationHandler as _VerificationHandler,
    )
    from aragora.server.handlers import (
        WebhookHandler as _WebhookHandler,
    )
    from aragora.server.handlers import (
        WorkflowHandler as _WorkflowHandler,
    )
    from aragora.server.handlers import (
        PolicyHandler as _PolicyHandler,
    )
    from aragora.server.handlers import (
        QueueHandler as _QueueHandler,
    )
    from aragora.server.handlers import (
        RLMContextHandler as _RLMContextHandler,
    )
    from aragora.server.handlers import (
        TrainingHandler as _TrainingHandler,
    )
    from aragora.server.handlers import (
        TranscriptionHandler as _TranscriptionHandler,
    )
    from aragora.server.handlers import (
        UncertaintyHandler as _UncertaintyHandler,
    )
    from aragora.server.handlers import (
        VerticalsHandler as _VerticalsHandler,
    )
    from aragora.server.handlers import (
        WorkspaceHandler as _WorkspaceHandler,
    )
    from aragora.server.handlers import (
        EmailHandler as _EmailHandler,
    )
    from aragora.server.handlers.features import (
        GmailIngestHandler as _GmailIngestHandler,
    )
    from aragora.server.handlers.features import (
        GmailQueryHandler as _GmailQueryHandler,
    )
    from aragora.server.handlers import (
        GoogleChatHandler as _GoogleChatHandler,
    )
    from aragora.server.handlers import (
        ExplainabilityHandler as _ExplainabilityHandler,
    )
    from aragora.server.handlers import (
        A2AHandler as _A2AHandler,
    )
    from aragora.server.handlers.codebase import (
        IntelligenceHandler as _CodeIntelligenceHandler,
    )
    from aragora.server.handlers.features import (
        AdvertisingHandler as _AdvertisingHandler,
    )
    from aragora.server.handlers.features import (
        AnalyticsPlatformsHandler as _AnalyticsPlatformsHandler,
    )
    from aragora.server.handlers.features import (
        CRMHandler as _CRMHandler,
    )
    from aragora.server.handlers.features import (
        SupportHandler as _SupportHandler,
    )
    from aragora.server.handlers.features import (
        EcommerceHandler as _EcommerceHandler,
    )
    from aragora.server.handlers.features import (
        ReconciliationHandler as _ReconciliationHandler,
    )
    from aragora.server.handlers.features import (
        UnifiedInboxHandler as _UnifiedInboxHandler,
    )
    from aragora.server.handlers.features import (
        CodebaseAuditHandler as _CodebaseAuditHandler,
    )
    from aragora.server.handlers.features import (
        LegalHandler as _LegalHandler,
    )
    from aragora.server.handlers.features import (
        DevOpsHandler as _DevOpsHandler,
    )
    from aragora.server.handlers.receipts import (
        ReceiptsHandler as _ReceiptsHandler,
    )

    # Assign imported classes to module-level variables
    SystemHandler = _SystemHandler
    HealthHandler = _HealthHandler
    NomicHandler = _NomicHandler
    DocsHandler = _DocsHandler
    DebatesHandler = _DebatesHandler
    AgentsHandler = _AgentsHandler
    PulseHandler = _PulseHandler
    AnalyticsHandler = _AnalyticsHandler
    MetricsHandler = _MetricsHandler
    ConsensusHandler = _ConsensusHandler
    BeliefHandler = _BeliefHandler
    DecisionExplainHandler = _DecisionExplainHandler
    DecisionHandler = _DecisionHandler
    CritiqueHandler = _CritiqueHandler
    GenesisHandler = _GenesisHandler
    ReplaysHandler = _ReplaysHandler
    TournamentHandler = _TournamentHandler
    MemoryHandler = _MemoryHandler
    LeaderboardViewHandler = _LeaderboardViewHandler
    DocumentHandler = _DocumentHandler
    DocumentBatchHandler = _DocumentBatchHandler
    VerificationHandler = _VerificationHandler
    AuditingHandler = _AuditingHandler
    RelationshipHandler = _RelationshipHandler
    MomentsHandler = _MomentsHandler
    PersonaHandler = _PersonaHandler
    DashboardHandler = _DashboardHandler
    IntrospectionHandler = _IntrospectionHandler
    CalibrationHandler = _CalibrationHandler
    RoutingHandler = _RoutingHandler
    EvolutionHandler = _EvolutionHandler
    EvolutionABTestingHandler = _EvolutionABTestingHandler
    PluginsHandler = _PluginsHandler
    BroadcastHandler = _BroadcastHandler
    AudioHandler = _AudioHandler
    SocialMediaHandler = _SocialMediaHandler
    LaboratoryHandler = _LaboratoryHandler
    ProbesHandler = _ProbesHandler
    InsightsHandler = _InsightsHandler
    BreakpointsHandler = _BreakpointsHandler
    LearningHandler = _LearningHandler
    GalleryHandler = _GalleryHandler
    AuthHandler = _AuthHandler
    BillingHandler = _BillingHandler
    BudgetHandler = _BudgetHandler
    CheckpointHandler = _CheckpointHandler
    GraphDebatesHandler = _GraphDebatesHandler
    MatrixDebatesHandler = _MatrixDebatesHandler
    FeaturesHandler = _FeaturesHandler
    MemoryAnalyticsHandler = _MemoryAnalyticsHandler
    GauntletHandler = _GauntletHandler
    SlackHandler = _SlackHandler
    SlackOAuthHandler = _SlackOAuthHandler
    TeamsIntegrationHandler = _TeamsIntegrationHandler
    TeamsOAuthHandler = _TeamsOAuthHandler
    DiscordOAuthHandler = _DiscordOAuthHandler
    OrganizationsHandler = _OrganizationsHandler
    OAuthHandler = _OAuthHandler
    ReviewsHandler = _ReviewsHandler
    FormalVerificationHandler = _FormalVerificationHandler
    EvaluationHandler = _EvaluationHandler
    EvidenceHandler = _EvidenceHandler
    FolderUploadHandler = _FolderUploadHandler
    WebhookHandler = _WebhookHandler
    WorkflowHandler = _WorkflowHandler
    AdminHandler = _AdminHandler
    SecurityHandler = _SecurityHandler
    ControlPlaneHandler = _ControlPlaneHandler
    OrchestrationHandler = _OrchestrationHandler
    DeliberationsHandler = _DeliberationsHandler
    KnowledgeHandler = _KnowledgeHandler
    KnowledgeMoundHandler = _KnowledgeMoundHandler
    PolicyHandler = _PolicyHandler
    QueueHandler = _QueueHandler
    RLMContextHandler = _RLMContextHandler
    TrainingHandler = _TrainingHandler
    TranscriptionHandler = _TranscriptionHandler
    UncertaintyHandler = _UncertaintyHandler
    VerticalsHandler = _VerticalsHandler
    WorkspaceHandler = _WorkspaceHandler
    EmailHandler = _EmailHandler
    GmailIngestHandler = _GmailIngestHandler
    GmailQueryHandler = _GmailQueryHandler
    GoogleChatHandler = _GoogleChatHandler
    ExplainabilityHandler = _ExplainabilityHandler
    A2AHandler = _A2AHandler
    CodeIntelligenceHandler = _CodeIntelligenceHandler
    AdvertisingHandler = _AdvertisingHandler
    AnalyticsPlatformsHandler = _AnalyticsPlatformsHandler
    CRMHandler = _CRMHandler
    SupportHandler = _SupportHandler
    EcommerceHandler = _EcommerceHandler
    ReconciliationHandler = _ReconciliationHandler
    UnifiedInboxHandler = _UnifiedInboxHandler
    CodebaseAuditHandler = _CodebaseAuditHandler
    LegalHandler = _LegalHandler
    DevOpsHandler = _DevOpsHandler
    ReceiptsHandler = _ReceiptsHandler
    HandlerResult = _HandlerResult

    HANDLERS_AVAILABLE = True
except ImportError as e:
    HANDLERS_AVAILABLE = False
    # Log the import error for debugging - this should not be silently swallowed
    logger.error(
        f"Failed to import handlers: {e}. "
        "This may indicate a broken module or missing dependency. "
        "Handler functionality will be degraded."
    )
    # Handler class placeholders remain None for graceful degradation


# Handler class registry - ordered list of (attr_name, handler_class) pairs
# Handlers are tried in this order during routing
HANDLER_REGISTRY: List[Tuple[str, Any]] = [
    ("_health_handler", HealthHandler),
    ("_nomic_handler", NomicHandler),
    ("_docs_handler", DocsHandler),
    ("_system_handler", SystemHandler),
    ("_debates_handler", DebatesHandler),
    ("_agents_handler", AgentsHandler),
    ("_pulse_handler", PulseHandler),
    ("_analytics_handler", AnalyticsHandler),
    ("_metrics_handler", MetricsHandler),
    ("_consensus_handler", ConsensusHandler),
    ("_belief_handler", BeliefHandler),
    ("_decision_explain_handler", DecisionExplainHandler),
    ("_decision_handler", DecisionHandler),
    ("_critique_handler", CritiqueHandler),
    ("_genesis_handler", GenesisHandler),
    ("_replays_handler", ReplaysHandler),
    ("_tournament_handler", TournamentHandler),
    ("_memory_handler", MemoryHandler),
    ("_leaderboard_handler", LeaderboardViewHandler),
    ("_document_handler", DocumentHandler),
    ("_document_batch_handler", DocumentBatchHandler),
    ("_verification_handler", VerificationHandler),
    ("_auditing_handler", AuditingHandler),
    ("_relationship_handler", RelationshipHandler),
    ("_moments_handler", MomentsHandler),
    ("_persona_handler", PersonaHandler),
    ("_dashboard_handler", DashboardHandler),
    ("_introspection_handler", IntrospectionHandler),
    ("_calibration_handler", CalibrationHandler),
    ("_routing_handler", RoutingHandler),
    ("_evolution_ab_testing_handler", EvolutionABTestingHandler),
    ("_evolution_handler", EvolutionHandler),
    ("_plugins_handler", PluginsHandler),
    ("_audio_handler", AudioHandler),
    ("_social_handler", SocialMediaHandler),
    ("_broadcast_handler", BroadcastHandler),
    ("_laboratory_handler", LaboratoryHandler),
    ("_probes_handler", ProbesHandler),
    ("_insights_handler", InsightsHandler),
    ("_breakpoints_handler", BreakpointsHandler),
    ("_learning_handler", LearningHandler),
    ("_gallery_handler", GalleryHandler),
    ("_auth_handler", AuthHandler),
    ("_billing_handler", BillingHandler),
    ("_budget_handler", BudgetHandler),
    ("_checkpoint_handler", CheckpointHandler),
    ("_graph_debates_handler", GraphDebatesHandler),
    ("_matrix_debates_handler", MatrixDebatesHandler),
    ("_features_handler", FeaturesHandler),
    ("_memory_analytics_handler", MemoryAnalyticsHandler),
    ("_gauntlet_handler", GauntletHandler),
    ("_slack_handler", SlackHandler),
    ("_slack_oauth_handler", SlackOAuthHandler),
    ("_teams_oauth_handler", TeamsOAuthHandler),
    ("_discord_oauth_handler", DiscordOAuthHandler),
    ("_teams_integration_handler", TeamsIntegrationHandler),
    ("_organizations_handler", OrganizationsHandler),
    ("_oauth_handler", OAuthHandler),
    ("_reviews_handler", ReviewsHandler),
    ("_formal_verification_handler", FormalVerificationHandler),
    ("_evaluation_handler", EvaluationHandler),
    ("_evidence_handler", EvidenceHandler),
    ("_folder_upload_handler", FolderUploadHandler),
    ("_webhook_handler", WebhookHandler),
    ("_workflow_handler", WorkflowHandler),
    ("_admin_handler", AdminHandler),
    ("_security_handler", SecurityHandler),
    ("_control_plane_handler", ControlPlaneHandler),
    ("_knowledge_handler", KnowledgeHandler),
    ("_knowledge_mound_handler", KnowledgeMoundHandler),
    ("_policy_handler", PolicyHandler),
    ("_queue_handler", QueueHandler),
    ("_rlm_context_handler", RLMContextHandler),
    ("_training_handler", TrainingHandler),
    ("_transcription_handler", TranscriptionHandler),
    ("_uncertainty_handler", UncertaintyHandler),
    ("_verticals_handler", VerticalsHandler),
    ("_workspace_handler", WorkspaceHandler),
    ("_email_handler", EmailHandler),
    ("_gmail_ingest_handler", GmailIngestHandler),
    ("_gmail_query_handler", GmailQueryHandler),
    ("_google_chat_handler", GoogleChatHandler),
    ("_explainability_handler", ExplainabilityHandler),
    ("_a2a_handler", A2AHandler),
    ("_code_intelligence_handler", CodeIntelligenceHandler),
    ("_advertising_handler", AdvertisingHandler),
    ("_analytics_platforms_handler", AnalyticsPlatformsHandler),
    ("_crm_handler", CRMHandler),
    ("_support_handler", SupportHandler),
    ("_ecommerce_handler", EcommerceHandler),
    ("_reconciliation_handler", ReconciliationHandler),
    ("_unified_inbox_handler", UnifiedInboxHandler),
    ("_codebase_audit_handler", CodebaseAuditHandler),
    ("_legal_handler", LegalHandler),
    ("_devops_handler", DevOpsHandler),
    ("_receipts_handler", ReceiptsHandler),
]


class RouteIndex:
    """O(1) route lookup index for handler dispatch.

    Builds an index of exact paths and prefix patterns at initialization,
    enabling fast route resolution without iterating through all handlers.

    Performance:
    - Exact paths: O(1) dict lookup
    - Dynamic paths: O(1) LRU cache hit, O(n) cache miss with prefix scan
    """

    def __init__(self) -> None:
        # Exact path → (attr_name, handler) mapping
        self._exact_routes: Dict[str, Tuple[str, Any]] = {}
        # Prefix patterns for dynamic routes: [(prefix, attr_name, handler)]
        self._prefix_routes: List[Tuple[str, str, Any]] = []
        # Cache for resolved dynamic routes
        self._cache_size: int = 500

    def build(self, registry_mixin: Any) -> None:
        """Build route index from initialized handlers.

        Extracts ROUTES from each handler for exact matching,
        and identifies prefix patterns from can_handle logic.
        """
        self._exact_routes.clear()
        self._prefix_routes.clear()

        # Known prefix patterns by handler (extracted from can_handle implementations)
        PREFIX_PATTERNS = {
            "_health_handler": ["/healthz", "/readyz", "/api/health"],
            "_nomic_handler": ["/api/nomic/", "/api/modes"],
            "_docs_handler": ["/api/openapi", "/api/docs", "/api/redoc", "/api/postman"],
            "_debates_handler": ["/api/debate", "/api/debates", "/api/debates/", "/api/search"],
            "_agents_handler": [
                "/api/agent/",
                "/api/agents",
                "/api/leaderboard",
                "/api/rankings",
                "/api/calibration/leaderboard",
                "/api/matches/recent",
            ],
            "_pulse_handler": ["/api/pulse/"],
            "_consensus_handler": ["/api/consensus/"],
            "_belief_handler": ["/api/belief-network/", "/api/laboratory/"],
            "_decision_handler": ["/api/decisions"],
            "_genesis_handler": ["/api/genesis/"],
            "_replays_handler": ["/api/replays/"],
            "_tournament_handler": ["/api/tournaments/"],
            "_memory_handler": ["/api/memory/"],
            "_document_handler": ["/api/documents/"],
            "_document_batch_handler": ["/api/documents/batch", "/api/documents/processing/"],
            "_auditing_handler": [
                "/api/debates/capability-probe",
                "/api/debates/deep-audit",
                "/api/redteam/",
            ],
            "_relationship_handler": ["/api/relationship/"],
            "_moments_handler": ["/api/moments/"],
            "_persona_handler": ["/api/personas", "/api/agent/"],
            "_introspection_handler": ["/api/introspection/"],
            "_calibration_handler": ["/api/agent/"],
            "_evolution_handler": ["/api/evolution/"],
            "_plugins_handler": ["/api/plugins/"],
            "_audio_handler": ["/audio/", "/api/podcast/"],
            "_social_handler": ["/api/youtube/"],
            "_broadcast_handler": ["/api/podcast/"],
            "_insights_handler": ["/api/insights/"],
            "_learning_handler": ["/api/learning/"],
            "_gallery_handler": ["/api/gallery/"],
            "_auth_handler": ["/api/auth/", "/api/v1/auth/"],
            "_billing_handler": ["/api/billing/", "/api/v1/billing/"],
            "_budget_handler": ["/api/v1/budgets"],
            "_checkpoint_handler": ["/api/checkpoints"],
            "_graph_debates_handler": ["/api/debates/graph"],
            "_matrix_debates_handler": ["/api/debates/matrix"],
            "_gauntlet_handler": ["/api/gauntlet/"],
            "_organizations_handler": [
                "/api/org/",
                "/api/user/organizations",
                "/api/invitations/",
            ],
            "_oauth_handler": ["/api/auth/oauth/", "/api/v1/auth/oauth/"],
            "_reviews_handler": ["/api/reviews/"],
            "_formal_verification_handler": ["/api/verify/"],
            "_evidence_handler": ["/api/evidence"],
            "_folder_upload_handler": ["/api/documents/folder", "/api/documents/folders"],
            "_webhook_handler": ["/api/webhooks"],
            "_admin_handler": ["/api/admin"],
            "_control_plane_handler": ["/api/control-plane/"],
            "_knowledge_handler": ["/api/knowledge/"],
            "_knowledge_mound_handler": ["/api/knowledge/mound/"],
            "_policy_handler": ["/api/policies", "/api/compliance/"],
            "_queue_handler": ["/api/queue/"],
            "_rlm_context_handler": ["/api/rlm/"],
            "_training_handler": ["/api/training/"],
            "_transcription_handler": ["/api/transcription/", "/api/transcribe/"],
            "_uncertainty_handler": ["/api/uncertainty/"],
            "_verticals_handler": ["/api/verticals"],
            "_workspace_handler": [
                "/api/workspaces",
                "/api/retention/",
                "/api/classify",
                "/api/audit/",
            ],
            "_email_handler": [
                "/api/email/",
            ],
            "_teams_oauth_handler": [
                "/api/integrations/teams/install",
                "/api/integrations/teams/callback",
                "/api/integrations/teams/refresh",
            ],
            "_discord_oauth_handler": [
                "/api/integrations/discord/install",
                "/api/integrations/discord/callback",
                "/api/integrations/discord/uninstall",
            ],
            "_teams_integration_handler": [
                "/api/v1/integrations/teams",
            ],
            "_google_chat_handler": [
                "/api/bots/google-chat/",
            ],
            "_explainability_handler": [
                "/api/v1/debates/",
                "/api/v1/explain/",
                "/api/debates/",
                "/api/explain/",
            ],
            "_a2a_handler": [
                "/api/a2a/",
                "/.well-known/agent.json",
            ],
            "_code_intelligence_handler": [
                "/api/codebase/",
                "/api/v1/codebase/",
            ],
            "_advertising_handler": [
                "/api/advertising/",
                "/api/v1/advertising/",
            ],
            "_analytics_platforms_handler": [
                "/api/analytics-platforms/",
                "/api/v1/analytics-platforms/",
            ],
            "_crm_handler": [
                "/api/crm/",
                "/api/v1/crm/",
            ],
            "_support_handler": [
                "/api/support/",
                "/api/v1/support/",
            ],
            "_ecommerce_handler": [
                "/api/ecommerce/",
                "/api/v1/ecommerce/",
            ],
            "_receipts_handler": [
                "/api/v2/receipts",
                "/api/v2/receipts/",
            ],
        }

        for attr_name, _ in HANDLER_REGISTRY:
            handler = getattr(registry_mixin, attr_name, None)
            if handler is None:
                continue

            # Extract exact routes from ROUTES attribute
            routes = getattr(handler, "ROUTES", [])
            for path in routes:
                if path not in self._exact_routes:
                    self._exact_routes[path] = (attr_name, handler)

            # Add prefix patterns
            prefixes = PREFIX_PATTERNS.get(attr_name, [])
            for prefix in prefixes:
                self._prefix_routes.append((prefix, attr_name, handler))

        # Clear the LRU cache when index is rebuilt
        self._get_handler_cached.cache_clear()

        logger.debug(
            f"[route-index] Built index: {len(self._exact_routes)} exact, "
            f"{len(self._prefix_routes)} prefix patterns"
        )

    def get_handler(self, path: str) -> Optional[Tuple[str, Any]]:
        """Get handler for path with O(1) lookup for known routes.

        Supports both versioned (/api/v1/debates) and legacy (/api/debates) paths.
        Versioned paths are normalized by stripping the version prefix before matching.

        Args:
            path: URL path to match

        Returns:
            Tuple of (attr_name, handler) or None if no match
        """
        # Fast path: exact match (for legacy paths)
        if path in self._exact_routes:
            return self._exact_routes[path]

        # Try matching with version stripped (for /api/v1/* paths)
        normalized_path = strip_version_prefix(path)
        if normalized_path != path and normalized_path in self._exact_routes:
            return self._exact_routes[normalized_path]

        # Cached prefix lookup for dynamic routes
        return self._get_handler_cached(path, normalized_path)

    @lru_cache(maxsize=500)
    def _get_handler_cached(self, path: str, normalized_path: str) -> Optional[Tuple[str, Any]]:
        """Cached prefix matching for dynamic routes.

        Tries matching both the original path and the normalized (version-stripped) path.
        """
        # Try original path first
        for prefix, attr_name, handler in self._prefix_routes:
            if path.startswith(prefix):
                # Verify with handler's can_handle for complex patterns
                if handler.can_handle(path):
                    return (attr_name, handler)

        # Try normalized path for versioned routes (/api/v1/debates -> /api/debates)
        if normalized_path != path:
            for prefix, attr_name, handler in self._prefix_routes:
                if normalized_path.startswith(prefix):
                    # Check if handler can handle the normalized path
                    if handler.can_handle(normalized_path):
                        return (attr_name, handler)

        return None


# Global route index instance
_route_index: Optional[RouteIndex] = None


def get_route_index() -> RouteIndex:
    """Get or create the global route index."""
    global _route_index
    if _route_index is None:
        _route_index = RouteIndex()
    return _route_index


# =============================================================================
# Handler Validation
# =============================================================================


class HandlerValidationError(Exception):
    """Raised when a handler fails validation."""

    pass


def validate_handler_class(handler_class: Any, handler_name: str) -> List[str]:
    """
    Validate that a handler class has required methods and attributes.

    Args:
        handler_class: The handler class to validate
        handler_name: Name for error messages

    Returns:
        List of validation errors (empty if valid)
    """
    errors: List[str] = []

    if handler_class is None:
        errors.append(f"{handler_name}: Handler class is None")
        return errors

    # Required method: can_handle(path: str) -> bool
    if not hasattr(handler_class, "can_handle"):
        errors.append(f"{handler_name}: Missing required method 'can_handle'")
    elif not callable(getattr(handler_class, "can_handle")):
        errors.append(f"{handler_name}: 'can_handle' is not callable")

    # Required method: handle(path: str, query: dict, request_handler) -> HandlerResult
    if not hasattr(handler_class, "handle"):
        errors.append(f"{handler_name}: Missing required method 'handle'")
    elif not callable(getattr(handler_class, "handle")):
        errors.append(f"{handler_name}: 'handle' is not callable")

    # Optional but recommended: ROUTES attribute for exact path matching
    if not hasattr(handler_class, "ROUTES"):
        logger.debug(f"{handler_name}: No ROUTES attribute (will use prefix matching only)")

    return errors


def validate_handler_instance(handler: Any, handler_name: str) -> List[str]:
    """
    Validate an instantiated handler works correctly.

    Args:
        handler: The handler instance
        handler_name: Name for error messages

    Returns:
        List of validation errors (empty if valid)
    """
    errors: List[str] = []

    if handler is None:
        errors.append(f"{handler_name}: Handler instance is None")
        return errors

    # Verify can_handle doesn't crash with a test path
    try:
        result = handler.can_handle("/api/test-path-validation")
        if not isinstance(result, bool):
            errors.append(f"{handler_name}: can_handle() returned non-bool: {type(result)}")
    except Exception as e:
        errors.append(f"{handler_name}: can_handle() raised exception: {e}")

    return errors


def validate_all_handlers(raise_on_error: bool = False) -> Dict[str, Any]:
    """
    Validate all registered handler classes.

    This should be called at startup to catch configuration issues early.

    Args:
        raise_on_error: If True, raise exception on validation failures

    Returns:
        Dict with validation results:
        - valid: List of valid handler names
        - invalid: Dict of handler name -> error messages
        - missing: List of handlers that couldn't be imported
    """
    if not HANDLERS_AVAILABLE:
        logger.warning("[handler-validation] Handler imports failed, skipping validation")
        return {
            "valid": [],
            "invalid": {},
            "missing": [name for name, _ in HANDLER_REGISTRY],
            "status": "imports_failed",
        }

    results: Dict[str, Any] = {
        "valid": [],
        "invalid": {},
        "missing": [],
        "status": "ok",
    }

    for attr_name, handler_class in HANDLER_REGISTRY:
        handler_name = attr_name.replace("_handler", "").replace("_", " ").title()

        if handler_class is None:
            results["missing"].append(handler_name)
            continue

        errors = validate_handler_class(handler_class, handler_name)
        if errors:
            results["invalid"][handler_name] = errors
        else:
            results["valid"].append(handler_name)

    # Log summary
    valid_count = len(results["valid"])
    invalid_count = len(results["invalid"])
    missing_count = len(results["missing"])
    total = valid_count + invalid_count + missing_count

    if invalid_count > 0 or missing_count > 0:
        logger.warning(
            f"[handler-validation] {valid_count}/{total} handlers valid, "
            f"{invalid_count} invalid, {missing_count} missing"
        )
        for name, errors in results["invalid"].items():
            for error in errors:
                logger.warning(f"[handler-validation] {error}")
        results["status"] = "validation_errors"
    else:
        logger.info(f"[handler-validation] All {valid_count} handlers validated successfully")

    if raise_on_error and (invalid_count > 0 or missing_count > 0):
        raise HandlerValidationError(
            f"Handler validation failed: {invalid_count} invalid, {missing_count} missing"
        )

    return results


def validate_handlers_on_init(registry_mixin: Any) -> Dict[str, Any]:
    """
    Validate instantiated handlers after initialization.

    Called from _init_handlers to verify all handlers work correctly.

    Args:
        registry_mixin: The HandlerRegistryMixin instance with initialized handlers

    Returns:
        Dict with validation results
    """
    results: Dict[str, Any] = {
        "valid": [],
        "invalid": {},
        "not_initialized": [],
    }

    for attr_name, handler_class in HANDLER_REGISTRY:
        handler_name = attr_name.replace("_handler", "").replace("_", " ").title()
        handler = getattr(registry_mixin, attr_name, None)

        if handler is None:
            results["not_initialized"].append(handler_name)
            continue

        errors = validate_handler_instance(handler, handler_name)
        if errors:
            results["invalid"][handler_name] = errors
        else:
            results["valid"].append(handler_name)

    if results["invalid"]:
        for name, errors in results["invalid"].items():
            for error in errors:
                logger.warning(f"[handler-instance-validation] {error}")

    return results


class HandlerRegistryMixin:
    """
    Mixin providing modular HTTP handler initialization and routing.

    This mixin expects the following class attributes from the parent:
    - storage: Optional[DebateStorage]
    - elo_system: Optional[EloSystem]
    - debate_embeddings: Optional[DebateEmbeddingsDatabase]
    - document_store: Optional[DocumentStore]
    - nomic_state_file: Optional[Path] (for deriving nomic_dir)
    - critique_store: Optional[CritiqueStore]
    - persona_manager: Optional[PersonaManager]
    - position_ledger: Optional[PositionLedger]

    And these methods:
    - _add_cors_headers()
    - _add_security_headers()
    - send_response(status)
    - send_header(name, value)
    - end_headers()
    - wfile.write(data)
    """

    # Type stubs for attributes expected from parent class
    storage: Optional["DebateStorage"]
    elo_system: Optional["EloSystem"]
    debate_embeddings: Optional["DebateEmbeddingsDatabase"]
    document_store: Optional[Any]
    nomic_state_file: Optional["Path"]
    critique_store: Optional["CritiqueStore"]
    persona_manager: Optional["PersonaManager"]
    position_ledger: Optional["PositionLedger"]
    wfile: BinaryIO

    # Type stubs for methods expected from parent class
    _add_cors_headers: Callable[[], None]
    _add_security_headers: Callable[[], None]
    send_response: Callable[[int], None]
    send_header: Callable[[str, str], None]
    end_headers: Callable[[], None]

    # Handler instances (initialized lazily)
    _health_handler: Optional["BaseHandler"] = None
    _nomic_handler: Optional["BaseHandler"] = None
    _docs_handler: Optional["BaseHandler"] = None
    _system_handler: Optional["BaseHandler"] = None
    _debates_handler: Optional["BaseHandler"] = None
    _agents_handler: Optional["BaseHandler"] = None
    _pulse_handler: Optional["BaseHandler"] = None
    _analytics_handler: Optional["BaseHandler"] = None
    _metrics_handler: Optional["BaseHandler"] = None
    _consensus_handler: Optional["BaseHandler"] = None
    _belief_handler: Optional["BaseHandler"] = None
    _critique_handler: Optional["BaseHandler"] = None
    _decision_handler: Optional["BaseHandler"] = None
    _genesis_handler: Optional["BaseHandler"] = None
    _replays_handler: Optional["BaseHandler"] = None
    _tournament_handler: Optional["BaseHandler"] = None
    _memory_handler: Optional["BaseHandler"] = None
    _leaderboard_handler: Optional["BaseHandler"] = None
    _document_handler: Optional["BaseHandler"] = None
    _document_batch_handler: Optional["BaseHandler"] = None
    _verification_handler: Optional["BaseHandler"] = None
    _auditing_handler: Optional["BaseHandler"] = None
    _relationship_handler: Optional["BaseHandler"] = None
    _moments_handler: Optional["BaseHandler"] = None
    _persona_handler: Optional["BaseHandler"] = None
    _dashboard_handler: Optional["BaseHandler"] = None
    _introspection_handler: Optional["BaseHandler"] = None
    _calibration_handler: Optional["BaseHandler"] = None
    _routing_handler: Optional["BaseHandler"] = None
    _evolution_handler: Optional["BaseHandler"] = None
    _plugins_handler: Optional["BaseHandler"] = None
    _broadcast_handler: Optional["BaseHandler"] = None
    _audio_handler: Optional["BaseHandler"] = None
    _social_handler: Optional["BaseHandler"] = None
    _laboratory_handler: Optional["BaseHandler"] = None
    _probes_handler: Optional["BaseHandler"] = None
    _insights_handler: Optional["BaseHandler"] = None
    _breakpoints_handler: Optional["BaseHandler"] = None
    _learning_handler: Optional["BaseHandler"] = None
    _gallery_handler: Optional["BaseHandler"] = None
    _auth_handler: Optional["BaseHandler"] = None
    _billing_handler: Optional["BaseHandler"] = None
    _budget_handler: Optional["BaseHandler"] = None
    _graph_debates_handler: Optional["BaseHandler"] = None
    _matrix_debates_handler: Optional["BaseHandler"] = None
    _features_handler: Optional["BaseHandler"] = None
    _memory_analytics_handler: Optional["BaseHandler"] = None
    _gauntlet_handler: Optional["BaseHandler"] = None
    _slack_handler: Optional["BaseHandler"] = None
    _slack_oauth_handler: Optional["BaseHandler"] = None
    _teams_oauth_handler: Optional["BaseHandler"] = None
    _discord_oauth_handler: Optional["BaseHandler"] = None
    _teams_integration_handler: Optional["BaseHandler"] = None
    _organizations_handler: Optional["BaseHandler"] = None
    _oauth_handler: Optional["BaseHandler"] = None
    _reviews_handler: Optional["BaseHandler"] = None
    _formal_verification_handler: Optional["BaseHandler"] = None
    _evolution_ab_testing_handler: Optional["BaseHandler"] = None
    _evidence_handler: Optional["BaseHandler"] = None
    _folder_upload_handler: Optional["BaseHandler"] = None
    _webhook_handler: Optional["BaseHandler"] = None
    _admin_handler: Optional["BaseHandler"] = None
    _control_plane_handler: Optional["BaseHandler"] = None
    _knowledge_handler: Optional["BaseHandler"] = None
    _knowledge_mound_handler: Optional["BaseHandler"] = None
    _email_handler: Optional["BaseHandler"] = None
    _gmail_ingest_handler: Optional["BaseHandler"] = None
    _gmail_query_handler: Optional["BaseHandler"] = None
    _google_chat_handler: Optional["BaseHandler"] = None
    _explainability_handler: Optional["BaseHandler"] = None
    _a2a_handler: Optional["BaseHandler"] = None
    _advertising_handler: Optional["BaseHandler"] = None
    _analytics_platforms_handler: Optional["BaseHandler"] = None
    _crm_handler: Optional["BaseHandler"] = None
    _support_handler: Optional["BaseHandler"] = None
    _ecommerce_handler: Optional["BaseHandler"] = None
    _handlers_initialized: bool = False

    @classmethod
    def _init_handlers(cls) -> None:
        """Initialize modular HTTP handlers with server context.

        Called lazily on first request. Creates handler instances with
        references to storage, ELO system, and other shared resources.
        """
        if cls._handlers_initialized or not HANDLERS_AVAILABLE:
            return

        # Build server context for handlers
        nomic_dir = None
        if hasattr(cls, "nomic_state_file") and cls.nomic_state_file:
            nomic_dir = cls.nomic_state_file.parent

        ctx = {
            "storage": getattr(cls, "storage", None),
            "stream_emitter": getattr(cls, "stream_emitter", None),
            "control_plane_stream": getattr(cls, "control_plane_stream", None),
            "nomic_loop_stream": getattr(cls, "nomic_loop_stream", None),
            "elo_system": getattr(cls, "elo_system", None),
            "nomic_dir": nomic_dir,
            "debate_embeddings": getattr(cls, "debate_embeddings", None),
            "critique_store": getattr(cls, "critique_store", None),
            "document_store": getattr(cls, "document_store", None),
            "persona_manager": getattr(cls, "persona_manager", None),
            "position_ledger": getattr(cls, "position_ledger", None),
            "user_store": getattr(cls, "user_store", None),
        }

        # Initialize all handlers from registry
        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is not None:
                setattr(cls, attr_name, handler_class(ctx))

        cls._handlers_initialized = True
        logger.info(f"[handlers] Modular handlers initialized ({len(HANDLER_REGISTRY)} handlers)")

        # Validate instantiated handlers
        validation_results = validate_handlers_on_init(cls)
        if validation_results["invalid"]:
            logger.warning(
                f"[handlers] {len(validation_results['invalid'])} handlers have validation issues"
            )

        # Build route index for O(1) dispatch
        route_index = get_route_index()
        route_index.build(cls)

        # Log resource availability for observability
        cls._log_resource_availability(nomic_dir)

    @classmethod
    def _log_resource_availability(cls, nomic_dir) -> None:
        """Log which optional resources are available at startup."""
        from aragora.persistence.db_config import LEGACY_DB_NAMES, DatabaseType

        resources = {
            "storage": getattr(cls, "storage", None) is not None,
            "elo_system": getattr(cls, "elo_system", None) is not None,
            "debate_embeddings": getattr(cls, "debate_embeddings", None) is not None,
            "document_store": getattr(cls, "document_store", None) is not None,
            "nomic_dir": nomic_dir is not None,
        }

        # Check database files if nomic_dir exists
        if nomic_dir:
            db_files = [
                ("positions_db", "aragora_positions.db"),
                ("personas_db", LEGACY_DB_NAMES[DatabaseType.PERSONAS]),
                ("grounded_db", "grounded_positions.db"),
                ("insights_db", "insights.db"),
                ("calibration_db", "agent_calibration.db"),
                ("embeddings_db", "debate_embeddings.db"),
            ]
            for name, filename in db_files:
                resources[name] = (nomic_dir / filename).exists()

        available = [k for k, v in resources.items() if v]
        unavailable = [k for k, v in resources.items() if not v]

        if unavailable:
            logger.info(f"[resources] Available: {', '.join(available)}")
            logger.warning(f"[resources] Unavailable: {', '.join(unavailable)}")
        else:
            logger.info(f"[resources] All resources available: {', '.join(available)}")

    def _try_modular_handler(self, path: str, query: dict) -> bool:
        """Try to handle request via modular handlers.

        Uses O(1) route index for fast handler lookup instead of iterating
        through all handlers. Supports API versioning with automatic
        version header injection.

        Returns True if handled, False if should fall through to legacy routes.
        """
        if not HANDLERS_AVAILABLE:
            return False

        # Ensure handlers are initialized
        self._init_handlers()

        # Extract API version from path/headers
        request_headers = {}
        if hasattr(self, "headers"):
            request_headers = {k: v for k, v in self.headers.items()}
        api_version, is_legacy = extract_version(path, request_headers)

        # Normalize path for handler matching (strip version prefix)
        normalized_path = strip_version_prefix(path)

        # Convert query params from {key: [val]} to {key: val}
        query_dict = {
            k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in query.items()
        }

        # Determine HTTP method for routing
        method = getattr(self, "command", "GET")

        # O(1) route lookup via index (uses both original and normalized paths)
        route_index = get_route_index()
        route_match = route_index.get_handler(path)

        if route_match is None:
            # Fallback: iterate through handlers for edge cases not in index
            # Try normalized path first for versioned routes
            for attr_name, _ in HANDLER_REGISTRY:
                handler = getattr(self, attr_name, None)
                if handler:
                    if handler.can_handle(normalized_path):
                        route_match = (attr_name, handler)
                        break
                    elif normalized_path != path and handler.can_handle(path):
                        route_match = (attr_name, handler)
                        break

        if route_match is None:
            return False

        attr_name, handler = route_match

        try:
            # Use normalized path for handler dispatch
            dispatch_path = normalized_path

            # Dispatch to appropriate handler method based on HTTP method
            if method == "POST" and hasattr(handler, "handle_post"):
                result = handler.handle_post(dispatch_path, query_dict, self)
            elif method == "DELETE" and hasattr(handler, "handle_delete"):
                result = handler.handle_delete(dispatch_path, query_dict, self)
            elif method == "PATCH" and hasattr(handler, "handle_patch"):
                result = handler.handle_patch(dispatch_path, query_dict, self)
            elif method == "PUT" and hasattr(handler, "handle_put"):
                result = handler.handle_put(dispatch_path, query_dict, self)
            else:
                result = handler.handle(dispatch_path, query_dict, self)

            # Handle async handlers - await coroutines
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                result = loop.run_until_complete(result)

            if result:
                self.send_response(result.status_code)
                self.send_header("Content-Type", result.content_type)

                # Add API version headers
                version_headers = version_response_headers(api_version, is_legacy)
                for h_name, h_val in version_headers.items():
                    self.send_header(h_name, h_val)

                # Add handler-specific headers
                for h_name, h_val in result.headers.items():
                    self.send_header(h_name, h_val)

                # Add CORS and security headers for modular handlers
                self._add_cors_headers()
                self._add_security_headers()
                self.end_headers()
                self.wfile.write(result.body)
                return True
        except Exception as e:
            logger.error(f"[handlers] Error in {handler.__class__.__name__}: {e}")
            # Fall through to legacy handler on error
            return False

        return False

    def _get_handler_stats(self) -> Dict[str, Any]:
        """Get statistics about initialized handlers.

        Returns:
            Dict with handler counts and names
        """
        if not self._handlers_initialized:
            return {"initialized": False, "count": 0, "handlers": []}

        initialized_handlers = []
        for attr_name, _ in HANDLER_REGISTRY:
            handler = getattr(self, attr_name, None)
            if handler is not None:
                initialized_handlers.append(handler.__class__.__name__)

        return {
            "initialized": True,
            "count": len(initialized_handlers),
            "handlers": initialized_handlers,
        }


__all__ = [
    "HandlerRegistryMixin",
    "HANDLER_REGISTRY",
    "HANDLERS_AVAILABLE",
    "RouteIndex",
    "get_route_index",
    # Validation functions
    "HandlerValidationError",
    "validate_handler_class",
    "validate_handler_instance",
    "validate_all_handlers",
    "validate_handlers_on_init",
]
