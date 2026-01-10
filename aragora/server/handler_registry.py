"""
Handler registry for modular HTTP endpoint routing.

This module provides centralized initialization and routing for all modular
HTTP handlers. The HandlerRegistryMixin can be mixed into request handler
classes to add modular routing capabilities.

Features:
- O(1) exact path lookup via route index
- LRU cached prefix matching for dynamic routes
- Lazy handler initialization

Usage:
    class MyHandler(HandlerRegistryMixin, BaseHTTPRequestHandler):
        pass
"""

import logging
from functools import lru_cache
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.server.handlers.base import BaseHandler, HandlerResult
    from aragora.server.storage import DebateStorage
    from aragora.ranking.elo import EloSystem
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    from aragora.memory.store import CritiqueStore
    from aragora.agents.personas import PersonaManager
    from aragora.agents.positions import PositionLedger
    from pathlib import Path

logger = logging.getLogger(__name__)

# Import handlers with graceful fallback
try:
    from aragora.server.handlers import (
        SystemHandler,
        DebatesHandler,
        AgentsHandler,
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
        DocumentHandler,
        VerificationHandler,
        AuditingHandler,
        RelationshipHandler,
        MomentsHandler,
        PersonaHandler,
        DashboardHandler,
        IntrospectionHandler,
        CalibrationHandler,
        RoutingHandler,
        EvolutionHandler,
        PluginsHandler,
        BroadcastHandler,
        AudioHandler,
        SocialMediaHandler,
        LaboratoryHandler,
        ProbesHandler,
        InsightsHandler,
        BreakpointsHandler,
        LearningHandler,
        GalleryHandler,
        AuthHandler,
        BillingHandler,
        GraphDebatesHandler,
        MatrixDebatesHandler,
        FeaturesHandler,
        HandlerResult,
    )
    HANDLERS_AVAILABLE = True
except ImportError:
    HANDLERS_AVAILABLE = False
    # Set all handler classes to None for graceful degradation
    SystemHandler = None  # type: ignore[misc,assignment]
    DebatesHandler = None  # type: ignore[misc, assignment]
    AgentsHandler = None  # type: ignore[misc, assignment]
    PulseHandler = None  # type: ignore[misc, assignment]
    AnalyticsHandler = None  # type: ignore[misc, assignment]
    MetricsHandler = None  # type: ignore[misc, assignment]
    ConsensusHandler = None  # type: ignore[misc, assignment]
    BeliefHandler = None  # type: ignore[misc, assignment]
    CritiqueHandler = None  # type: ignore[misc, assignment]
    GenesisHandler = None  # type: ignore[misc, assignment]
    ReplaysHandler = None  # type: ignore[misc, assignment]
    TournamentHandler = None  # type: ignore[misc, assignment]
    MemoryHandler = None  # type: ignore[misc, assignment]
    LeaderboardViewHandler = None  # type: ignore[misc, assignment]
    DocumentHandler = None  # type: ignore[misc, assignment]
    VerificationHandler = None  # type: ignore[misc, assignment]
    AuditingHandler = None  # type: ignore[misc, assignment]
    RelationshipHandler = None  # type: ignore[misc, assignment]
    MomentsHandler = None  # type: ignore[misc, assignment]
    PersonaHandler = None  # type: ignore[misc, assignment]
    DashboardHandler = None  # type: ignore[misc, assignment]
    IntrospectionHandler = None  # type: ignore[misc, assignment]
    CalibrationHandler = None  # type: ignore[misc, assignment]
    RoutingHandler = None  # type: ignore[misc, assignment]
    EvolutionHandler = None  # type: ignore[misc, assignment]
    PluginsHandler = None  # type: ignore[misc, assignment]
    BroadcastHandler = None  # type: ignore[misc, assignment]
    AudioHandler = None  # type: ignore[misc, assignment]
    SocialMediaHandler = None  # type: ignore[misc, assignment]
    LaboratoryHandler = None  # type: ignore[misc, assignment]
    ProbesHandler = None  # type: ignore[misc, assignment]
    InsightsHandler = None  # type: ignore[misc, assignment]
    BreakpointsHandler = None  # type: ignore[misc, assignment]
    LearningHandler = None  # type: ignore[misc, assignment]
    GalleryHandler = None  # type: ignore[misc, assignment]
    AuthHandler = None  # type: ignore[misc, assignment]
    BillingHandler = None  # type: ignore[misc, assignment]
    GraphDebatesHandler = None  # type: ignore[misc, assignment]
    MatrixDebatesHandler = None  # type: ignore[misc, assignment]
    FeaturesHandler = None  # type: ignore[misc, assignment]
    HandlerResult = None  # type: ignore[misc, assignment]


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
]


class RouteIndex:
    """O(1) route lookup index for handler dispatch.

    Builds an index of exact paths and prefix patterns at initialization,
    enabling fast route resolution without iterating through all handlers.

    Performance:
    - Exact paths: O(1) dict lookup
    - Dynamic paths: O(1) LRU cache hit, O(n) cache miss with prefix scan
    """

    def __init__(self):
        # Exact path → (attr_name, handler) mapping
        self._exact_routes: Dict[str, Tuple[str, Any]] = {}
        # Prefix patterns for dynamic routes: [(prefix, attr_name, handler)]
        self._prefix_routes: List[Tuple[str, str, Any]] = []
        # Cache for resolved dynamic routes
        self._cache_size = 500

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
            "_agents_handler": ["/api/agent/", "/api/agents", "/api/leaderboard",
                               "/api/rankings", "/api/calibration/leaderboard",
                               "/api/matches/recent"],
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
        }

        for attr_name, _ in HANDLER_REGISTRY:
            handler = getattr(registry_mixin, attr_name, None)
            if handler is None:
                continue

            # Extract exact routes from ROUTES attribute
            routes = getattr(handler, 'ROUTES', [])
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

        Args:
            path: URL path to match

        Returns:
            Tuple of (attr_name, handler) or None if no match
        """
        # Fast path: exact match
        if path in self._exact_routes:
            return self._exact_routes[path]

        # Cached prefix lookup for dynamic routes
        return self._get_handler_cached(path)

    @lru_cache(maxsize=500)
    def _get_handler_cached(self, path: str) -> Optional[Tuple[str, Any]]:
        """Cached prefix matching for dynamic routes."""
        for prefix, attr_name, handler in self._prefix_routes:
            if path.startswith(prefix):
                # Verify with handler's can_handle for complex patterns
                if handler.can_handle(path):
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
        if hasattr(cls, 'nomic_state_file') and cls.nomic_state_file:
            nomic_dir = cls.nomic_state_file.parent

        ctx = {
            "storage": getattr(cls, 'storage', None),
            "elo_system": getattr(cls, 'elo_system', None),
            "nomic_dir": nomic_dir,
            "debate_embeddings": getattr(cls, 'debate_embeddings', None),
            "critique_store": getattr(cls, 'critique_store', None),
            "document_store": getattr(cls, 'document_store', None),
            "persona_manager": getattr(cls, 'persona_manager', None),
            "position_ledger": getattr(cls, 'position_ledger', None),
            "user_store": getattr(cls, 'user_store', None),
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
            "storage": getattr(cls, 'storage', None) is not None,
            "elo_system": getattr(cls, 'elo_system', None) is not None,
            "debate_embeddings": getattr(cls, 'debate_embeddings', None) is not None,
            "document_store": getattr(cls, 'document_store', None) is not None,
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
        through all handlers.

        Returns True if handled, False if should fall through to legacy routes.
        """
        if not HANDLERS_AVAILABLE:
            return False

        # Ensure handlers are initialized
        self._init_handlers()

        # Convert query params from {key: [val]} to {key: val}
        query_dict = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in query.items()}

        # Determine HTTP method for routing
        method = getattr(self, 'command', 'GET')

        # O(1) route lookup via index
        route_index = get_route_index()
        route_match = route_index.get_handler(path)

        if route_match is None:
            # Fallback: iterate through handlers for edge cases not in index
            for attr_name, _ in HANDLER_REGISTRY:
                handler = getattr(self, attr_name, None)
                if handler and handler.can_handle(path):
                    route_match = (attr_name, handler)
                    break

        if route_match is None:
            return False

        attr_name, handler = route_match

        try:
            # Dispatch to appropriate handler method based on HTTP method
            if method == 'POST' and hasattr(handler, 'handle_post'):
                result = handler.handle_post(path, query_dict, self)
            elif method == 'DELETE' and hasattr(handler, 'handle_delete'):
                result = handler.handle_delete(path, query_dict, self)
            elif method == 'PATCH' and hasattr(handler, 'handle_patch'):
                result = handler.handle_patch(path, query_dict, self)
            elif method == 'PUT' and hasattr(handler, 'handle_put'):
                result = handler.handle_put(path, query_dict, self)
            else:
                result = handler.handle(path, query_dict, self)

            if result:
                self.send_response(result.status_code)
                self.send_header('Content-Type', result.content_type)
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
