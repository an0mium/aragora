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
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

from aragora.server.versioning import (
    extract_version,
    strip_version_prefix,
    version_response_headers,
    APIVersion,
)

if TYPE_CHECKING:
    from aragora.server.handlers.base import BaseHandler
    from aragora.server.storage import DebateStorage
    from aragora.ranking.elo import EloSystem
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    from aragora.memory.store import CritiqueStore
    from aragora.agents.personas import PersonaManager
    from aragora.agents.positions import PositionLedger
    from pathlib import Path

logger = logging.getLogger(__name__)

# Type alias for handler classes that may be None when handlers are unavailable
# This allows proper type hints without requiring type: ignore comments
HandlerType = Optional[Type[Any]]

# Handler class placeholders - set to actual classes on successful import
SystemHandler: HandlerType = None
DebatesHandler: HandlerType = None
AgentsHandler: HandlerType = None
PulseHandler: HandlerType = None
AnalyticsHandler: HandlerType = None
MetricsHandler: HandlerType = None
ConsensusHandler: HandlerType = None
BeliefHandler: HandlerType = None
CritiqueHandler: HandlerType = None
GenesisHandler: HandlerType = None
ReplaysHandler: HandlerType = None
TournamentHandler: HandlerType = None
MemoryHandler: HandlerType = None
LeaderboardViewHandler: HandlerType = None
DocumentHandler: HandlerType = None
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
GraphDebatesHandler: HandlerType = None
MatrixDebatesHandler: HandlerType = None
FeaturesHandler: HandlerType = None
MemoryAnalyticsHandler: HandlerType = None
GauntletHandler: HandlerType = None
SlackHandler: HandlerType = None
OrganizationsHandler: HandlerType = None
OAuthHandler: HandlerType = None
ReviewsHandler: HandlerType = None
FormalVerificationHandler: HandlerType = None
EvidenceHandler: HandlerType = None
WebhookHandler: HandlerType = None
AdminHandler: HandlerType = None
HandlerResult: HandlerType = None

# Import handlers with graceful fallback
try:
    from aragora.server.handlers import (
        SystemHandler as _SystemHandler,
        DebatesHandler as _DebatesHandler,
        AgentsHandler as _AgentsHandler,
        PulseHandler as _PulseHandler,
        AnalyticsHandler as _AnalyticsHandler,
        MetricsHandler as _MetricsHandler,
        ConsensusHandler as _ConsensusHandler,
        BeliefHandler as _BeliefHandler,
        CritiqueHandler as _CritiqueHandler,
        GenesisHandler as _GenesisHandler,
        ReplaysHandler as _ReplaysHandler,
        TournamentHandler as _TournamentHandler,
        MemoryHandler as _MemoryHandler,
        LeaderboardViewHandler as _LeaderboardViewHandler,
        DocumentHandler as _DocumentHandler,
        VerificationHandler as _VerificationHandler,
        AuditingHandler as _AuditingHandler,
        RelationshipHandler as _RelationshipHandler,
        MomentsHandler as _MomentsHandler,
        PersonaHandler as _PersonaHandler,
        DashboardHandler as _DashboardHandler,
        IntrospectionHandler as _IntrospectionHandler,
        CalibrationHandler as _CalibrationHandler,
        RoutingHandler as _RoutingHandler,
        EvolutionHandler as _EvolutionHandler,
        EvolutionABTestingHandler as _EvolutionABTestingHandler,
        PluginsHandler as _PluginsHandler,
        BroadcastHandler as _BroadcastHandler,
        AudioHandler as _AudioHandler,
        SocialMediaHandler as _SocialMediaHandler,
        LaboratoryHandler as _LaboratoryHandler,
        ProbesHandler as _ProbesHandler,
        InsightsHandler as _InsightsHandler,
        BreakpointsHandler as _BreakpointsHandler,
        LearningHandler as _LearningHandler,
        GalleryHandler as _GalleryHandler,
        AuthHandler as _AuthHandler,
        BillingHandler as _BillingHandler,
        GraphDebatesHandler as _GraphDebatesHandler,
        MatrixDebatesHandler as _MatrixDebatesHandler,
        FeaturesHandler as _FeaturesHandler,
        MemoryAnalyticsHandler as _MemoryAnalyticsHandler,
        GauntletHandler as _GauntletHandler,
        SlackHandler as _SlackHandler,
        OrganizationsHandler as _OrganizationsHandler,
        OAuthHandler as _OAuthHandler,
        ReviewsHandler as _ReviewsHandler,
        FormalVerificationHandler as _FormalVerificationHandler,
        EvidenceHandler as _EvidenceHandler,
        WebhookHandler as _WebhookHandler,
        AdminHandler as _AdminHandler,
        HandlerResult as _HandlerResult,
    )

    # Assign imported classes to module-level variables
    SystemHandler = _SystemHandler
    DebatesHandler = _DebatesHandler
    AgentsHandler = _AgentsHandler
    PulseHandler = _PulseHandler
    AnalyticsHandler = _AnalyticsHandler
    MetricsHandler = _MetricsHandler
    ConsensusHandler = _ConsensusHandler
    BeliefHandler = _BeliefHandler
    CritiqueHandler = _CritiqueHandler
    GenesisHandler = _GenesisHandler
    ReplaysHandler = _ReplaysHandler
    TournamentHandler = _TournamentHandler
    MemoryHandler = _MemoryHandler
    LeaderboardViewHandler = _LeaderboardViewHandler
    DocumentHandler = _DocumentHandler
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
    GraphDebatesHandler = _GraphDebatesHandler
    MatrixDebatesHandler = _MatrixDebatesHandler
    FeaturesHandler = _FeaturesHandler
    MemoryAnalyticsHandler = _MemoryAnalyticsHandler
    GauntletHandler = _GauntletHandler
    SlackHandler = _SlackHandler
    OrganizationsHandler = _OrganizationsHandler
    OAuthHandler = _OAuthHandler
    ReviewsHandler = _ReviewsHandler
    FormalVerificationHandler = _FormalVerificationHandler
    EvidenceHandler = _EvidenceHandler
    WebhookHandler = _WebhookHandler
    AdminHandler = _AdminHandler
    HandlerResult = _HandlerResult

    HANDLERS_AVAILABLE = True
except ImportError:
    HANDLERS_AVAILABLE = False
    # Handler class placeholders remain None for graceful degradation


# Handler class registry - ordered list of (attr_name, handler_class) pairs
# Handlers are tried in this order during routing
HANDLER_REGISTRY: List[Tuple[str, Any]] = [
    ("_system_handler", SystemHandler),
    ("_debates_handler", DebatesHandler),
    ("_agents_handler", AgentsHandler),
    ("_pulse_handler", PulseHandler),
    ("_analytics_handler", AnalyticsHandler),
    ("_metrics_handler", MetricsHandler),
    ("_consensus_handler", ConsensusHandler),
    ("_belief_handler", BeliefHandler),
    ("_critique_handler", CritiqueHandler),
    ("_genesis_handler", GenesisHandler),
    ("_replays_handler", ReplaysHandler),
    ("_tournament_handler", TournamentHandler),
    ("_memory_handler", MemoryHandler),
    ("_leaderboard_handler", LeaderboardViewHandler),
    ("_document_handler", DocumentHandler),
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
    ("_graph_debates_handler", GraphDebatesHandler),
    ("_matrix_debates_handler", MatrixDebatesHandler),
    ("_features_handler", FeaturesHandler),
    ("_memory_analytics_handler", MemoryAnalyticsHandler),
    ("_gauntlet_handler", GauntletHandler),
    ("_slack_handler", SlackHandler),
    ("_organizations_handler", OrganizationsHandler),
    ("_oauth_handler", OAuthHandler),
    ("_reviews_handler", ReviewsHandler),
    ("_formal_verification_handler", FormalVerificationHandler),
    ("_evidence_handler", EvidenceHandler),
    ("_webhook_handler", WebhookHandler),
    ("_admin_handler", AdminHandler),
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
            "_debates_handler": ["/api/debates/", "/api/search"],
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
            "_genesis_handler": ["/api/genesis/"],
            "_replays_handler": ["/api/replays/"],
            "_tournament_handler": ["/api/tournaments/"],
            "_memory_handler": ["/api/memory/"],
            "_document_handler": ["/api/documents/"],
            "_auditing_handler": ["/api/debates/"],  # for /red-team
            "_relationship_handler": ["/api/relationship/"],
            "_moments_handler": ["/api/moments/"],
            "_persona_handler": ["/api/personas", "/api/agent/"],
            "_introspection_handler": ["/api/introspection/"],
            "_calibration_handler": ["/api/agent/"],
            "_evolution_handler": ["/api/evolution/"],
            "_plugins_handler": ["/api/plugins/"],
            "_audio_handler": ["/audio/", "/api/podcast/"],
            "_social_handler": ["/api/youtube/", "/api/debates/"],
            "_broadcast_handler": ["/api/debates/"],
            "_insights_handler": ["/api/insights/"],
            "_learning_handler": ["/api/learning/"],
            "_gallery_handler": ["/api/gallery/"],
            "_auth_handler": ["/api/auth/"],
            "_billing_handler": ["/api/billing/"],
            "_graph_debates_handler": ["/api/debates/graph"],
            "_matrix_debates_handler": ["/api/debates/matrix"],
            "_gauntlet_handler": ["/api/gauntlet/"],
            "_organizations_handler": ["/api/organizations/"],
            "_oauth_handler": ["/api/auth/oauth/"],
            "_reviews_handler": ["/api/reviews/"],
            "_formal_verification_handler": ["/api/verify/"],
            "_evidence_handler": ["/api/evidence"],
            "_webhook_handler": ["/api/webhooks"],
            "_admin_handler": ["/api/admin"],
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
    _system_handler: Optional["BaseHandler"] = None
    _debates_handler: Optional["BaseHandler"] = None
    _agents_handler: Optional["BaseHandler"] = None
    _pulse_handler: Optional["BaseHandler"] = None
    _analytics_handler: Optional["BaseHandler"] = None
    _metrics_handler: Optional["BaseHandler"] = None
    _consensus_handler: Optional["BaseHandler"] = None
    _belief_handler: Optional["BaseHandler"] = None
    _critique_handler: Optional["BaseHandler"] = None
    _genesis_handler: Optional["BaseHandler"] = None
    _replays_handler: Optional["BaseHandler"] = None
    _tournament_handler: Optional["BaseHandler"] = None
    _memory_handler: Optional["BaseHandler"] = None
    _leaderboard_handler: Optional["BaseHandler"] = None
    _document_handler: Optional["BaseHandler"] = None
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
    _graph_debates_handler: Optional["BaseHandler"] = None
    _matrix_debates_handler: Optional["BaseHandler"] = None
    _features_handler: Optional["BaseHandler"] = None
    _memory_analytics_handler: Optional["BaseHandler"] = None
    _gauntlet_handler: Optional["BaseHandler"] = None
    _slack_handler: Optional["BaseHandler"] = None
    _organizations_handler: Optional["BaseHandler"] = None
    _oauth_handler: Optional["BaseHandler"] = None
    _reviews_handler: Optional["BaseHandler"] = None
    _formal_verification_handler: Optional["BaseHandler"] = None
    _evolution_ab_testing_handler: Optional["BaseHandler"] = None
    _evidence_handler: Optional["BaseHandler"] = None
    _webhook_handler: Optional["BaseHandler"] = None
    _admin_handler: Optional["BaseHandler"] = None
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

        # Build route index for O(1) dispatch
        route_index = get_route_index()
        route_index.build(cls)

        # Log resource availability for observability
        cls._log_resource_availability(nomic_dir)

    @classmethod
    def _log_resource_availability(cls, nomic_dir) -> None:
        """Log which optional resources are available at startup."""
        from aragora.config import DB_PERSONAS_PATH

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
                ("personas_db", DB_PERSONAS_PATH),
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
]
