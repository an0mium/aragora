"""
Handler registry for modular HTTP endpoint routing.

This module provides centralized initialization and routing for all modular
HTTP handlers. The HandlerRegistryMixin can be mixed into request handler
classes to add modular routing capabilities.

Usage:
    class MyHandler(HandlerRegistryMixin, BaseHTTPRequestHandler):
        pass
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.server.handlers.base import BaseHandler, HandlerResult

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
        HandlerResult,
    )
    HANDLERS_AVAILABLE = True
except ImportError:
    HANDLERS_AVAILABLE = False
    # Set all handler classes to None for graceful degradation
    SystemHandler = None
    DebatesHandler = None
    AgentsHandler = None
    PulseHandler = None
    AnalyticsHandler = None
    MetricsHandler = None
    ConsensusHandler = None
    BeliefHandler = None
    CritiqueHandler = None
    GenesisHandler = None
    ReplaysHandler = None
    TournamentHandler = None
    MemoryHandler = None
    LeaderboardViewHandler = None
    DocumentHandler = None
    VerificationHandler = None
    AuditingHandler = None
    RelationshipHandler = None
    MomentsHandler = None
    PersonaHandler = None
    DashboardHandler = None
    IntrospectionHandler = None
    CalibrationHandler = None
    RoutingHandler = None
    EvolutionHandler = None
    PluginsHandler = None
    BroadcastHandler = None
    AudioHandler = None
    SocialMediaHandler = None
    LaboratoryHandler = None
    ProbesHandler = None
    InsightsHandler = None
    HandlerResult = None


# Handler class registry - ordered list of (attr_name, handler_class) pairs
# Handlers are tried in this order during routing
HANDLER_REGISTRY: List[tuple] = [
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
]


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
        }

        # Initialize all handlers from registry
        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is not None:
                setattr(cls, attr_name, handler_class(ctx))

        cls._handlers_initialized = True
        logger.info(f"[handlers] Modular handlers initialized ({len(HANDLER_REGISTRY)} handlers)")

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

        # Try each handler in order
        for attr_name, _ in HANDLER_REGISTRY:
            handler = getattr(self, attr_name, None)
            if handler and handler.can_handle(path):
                try:
                    # Call handle() for GET, handle_post() for POST if available
                    if method == 'POST' and hasattr(handler, 'handle_post'):
                        result = handler.handle_post(path, query_dict, self)
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
]
