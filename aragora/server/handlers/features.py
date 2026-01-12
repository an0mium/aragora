"""
Feature availability handler.

Provides endpoints for discovering which optional features are available
in the current Aragora installation. This enables the frontend to:
1. Show appropriate UI for available features
2. Hide/disable unavailable features gracefully
3. Display helpful messages about how to enable features
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Registry
# =============================================================================

@dataclass
class FeatureInfo:
    """Information about an optional feature."""
    name: str
    description: str
    requires: list[str]  # List of module/class names required
    endpoints: list[str]  # API endpoints this feature provides
    install_hint: str = ""  # How to enable the feature
    status: str = "optional"  # "optional", "coming_soon", "deprecated"
    category: str = "general"  # "analysis", "evolution", "memory", etc.


# Registry of all optional features
FEATURE_REGISTRY: dict[str, FeatureInfo] = {
    "pulse": FeatureInfo(
        name="Trending Topics",
        description="Real-time trending topic discovery from social platforms",
        requires=["pulse_manager"],
        endpoints=["/api/pulse/trending", "/api/pulse/categories", "/api/pulse/stats"],
        install_hint="Requires Twitter/Reddit API keys. Set TWITTER_API_KEY or REDDIT_CLIENT_ID.",
        category="discovery",
    ),
    "genesis": FeatureInfo(
        name="Agent Evolution",
        description="Genetic algorithm for evolving agent prompts and traits",
        requires=["genesis_ledger", "population_manager"],
        endpoints=["/api/genesis/stats", "/api/genesis/lineage", "/api/genesis/tree"],
        install_hint="Genesis is enabled by default. Check database connectivity.",
        category="evolution",
    ),
    "verification": FeatureInfo(
        name="Formal Verification",
        description="Z3/Lean proof generation for mathematical claims",
        requires=["z3_connector"],
        endpoints=["/api/verification/proofs", "/api/verification/validate"],
        install_hint="Install z3-solver package: pip install z3-solver",
        status="optional",
        category="analysis",
    ),
    "laboratory": FeatureInfo(
        name="Persona Laboratory",
        description="Agent personality trait detection and analysis",
        requires=["persona_laboratory"],
        endpoints=["/api/laboratory/emergent-traits", "/api/laboratory/analyze"],
        install_hint="Laboratory is enabled by default. Run debates to accumulate data.",
        category="analysis",
    ),
    "calibration": FeatureInfo(
        name="Calibration Tracking",
        description="Agent prediction accuracy measurement over time",
        requires=["calibration_tracker"],
        endpoints=["/api/calibration/curve", "/api/calibration/history"],
        install_hint="Enable in DebateProtocol: enable_calibration=True",
        category="analysis",
    ),
    "evolution": FeatureInfo(
        name="Prompt Evolution",
        description="Learn from debates to improve agent prompts",
        requires=["prompt_evolver"],
        endpoints=["/api/evolution/patterns", "/api/evolution/prompts"],
        install_hint="Enable in DebateProtocol: enable_evolution=True",
        category="evolution",
    ),
    "red_team": FeatureInfo(
        name="Red Team Analysis",
        description="Adversarial testing and vulnerability detection",
        requires=["red_team_mode"],
        endpoints=["/api/auditing/red-team", "/api/auditing/probes"],
        install_hint="Available after debates complete. Check auditing handler.",
        category="security",
    ),
    "capability_probes": FeatureInfo(
        name="Capability Probes",
        description="Test agent capabilities across different domains",
        requires=["probe_runner"],
        endpoints=["/api/auditing/capability-probes", "/api/auditing/probe-results"],
        install_hint="Run capability probes from the UI or API.",
        category="security",
    ),
    "continuum_memory": FeatureInfo(
        name="Continuum Memory",
        description="Multi-tier learning memory with surprise-based consolidation",
        requires=["continuum_memory"],
        endpoints=["/api/memory/tiers", "/api/memory/stats", "/api/memory/search"],
        install_hint="Memory is enabled by default. Data accumulates over debates.",
        category="memory",
    ),
    "consensus_memory": FeatureInfo(
        name="Consensus Memory",
        description="Historical debate outcomes and dissenting views",
        requires=["consensus_memory"],
        endpoints=["/api/consensus/history", "/api/consensus/dissent"],
        install_hint="Consensus memory is enabled by default.",
        category="memory",
    ),
    "insights": FeatureInfo(
        name="Debate Insights",
        description="Extract key learnings and patterns from debates",
        requires=["insight_store"],
        endpoints=["/api/insights/recent", "/api/insights/search"],
        install_hint="Insights are enabled by default.",
        category="analysis",
    ),
    "moments": FeatureInfo(
        name="Moment Detection",
        description="Detect significant narrative moments in debates",
        requires=["moment_detector"],
        endpoints=["/api/agents/network", "/api/moments/recent"],
        install_hint="Moments are enabled by default.",
        category="analysis",
    ),
    "tournaments": FeatureInfo(
        name="Tournaments",
        description="Run structured tournaments between agents",
        requires=["tournament_runner"],
        endpoints=["/api/tournaments/create", "/api/tournaments/results"],
        install_hint="Tournaments are enabled by default.",
        category="competition",
    ),
    "elo": FeatureInfo(
        name="ELO Rankings",
        description="Agent skill ratings and leaderboards",
        requires=["elo_system"],
        endpoints=["/api/rankings/leaderboard", "/api/rankings/history"],
        install_hint="ELO is enabled by default.",
        category="competition",
    ),
    "crux": FeatureInfo(
        name="Crux Analysis",
        description="Identify key points of disagreement in debates",
        requires=["crux_analyzer"],
        endpoints=["/api/crux/beliefs", "/api/crux/analyze"],
        install_hint="Crux analysis requires belief network data.",
        category="analysis",
    ),
    "rhetorical": FeatureInfo(
        name="Rhetorical Observer",
        description="Detect rhetorical patterns like concession and rebuttal",
        requires=["rhetorical_observer"],
        endpoints=[],  # Event-based only
        install_hint="Enable in DebateProtocol: enable_rhetorical_observer=True",
        category="analysis",
    ),
    "trickster": FeatureInfo(
        name="Hollow Consensus Detection",
        description="Detect and challenge artificial agreement",
        requires=["trickster"],
        endpoints=[],  # Event-based only
        install_hint="Enable in DebateProtocol: enable_trickster=True",
        category="analysis",
    ),
    "plugins": FeatureInfo(
        name="Plugin System",
        description="Install and manage plugins for extended functionality",
        requires=["plugin_runner"],
        endpoints=["/api/plugins/available", "/api/plugins/installed", "/api/plugins/install"],
        install_hint="Plugin system is enabled by default.",
        category="system",
    ),
    "memory": FeatureInfo(
        name="Memory System",
        description="Multi-tier memory with inspection and exploration",
        requires=["memory_manager"],
        endpoints=["/api/memory/stats", "/api/memory/search", "/api/memory/tiers"],
        install_hint="Memory system is enabled by default.",
        category="memory",
    ),
}


# =============================================================================
# Feature Detection
# =============================================================================

def _check_feature_available(feature_id: str) -> tuple[bool, Optional[str]]:
    """
    Check if a feature's required components are available.

    Returns:
        Tuple of (is_available, reason_if_not)
    """
    if feature_id not in FEATURE_REGISTRY:
        return False, f"Unknown feature: {feature_id}"

    feature = FEATURE_REGISTRY[feature_id]

    # Check status first
    if feature.status == "coming_soon":
        return False, "Feature is coming soon"
    if feature.status == "deprecated":
        return False, "Feature is deprecated"

    # Check each required component
    for requirement in feature.requires:
        available, reason = _check_requirement(requirement)
        if not available:
            return False, reason

    return True, None


def _check_requirement(requirement: str) -> tuple[bool, Optional[str]]:
    """Check if a specific requirement is available."""
    # Map requirement names to import checks
    checks = {
        "pulse_manager": _check_pulse,
        "genesis_ledger": _check_genesis,
        "population_manager": _check_genesis,
        "z3_connector": _check_z3,
        "lean_connector": _check_lean,
        "persona_laboratory": _check_laboratory,
        "calibration_tracker": _check_calibration,
        "prompt_evolver": _check_evolution,
        "red_team_mode": _check_red_team,
        "probe_runner": _check_probes,
        "continuum_memory": _check_continuum,
        "consensus_memory": _check_consensus,
        "insight_store": _check_insights,
        "moment_detector": _check_moments,
        "tournament_runner": _check_tournaments,
        "elo_system": _check_elo,
        "crux_analyzer": _check_crux,
        "rhetorical_observer": _check_rhetorical,
        "trickster": _check_trickster,
    }

    check_func = checks.get(requirement)
    if check_func is None:
        # Unknown requirement - assume available
        return True, None

    try:
        return check_func()
    except Exception as e:
        return False, str(e)


# Individual requirement checks
def _check_pulse() -> tuple[bool, Optional[str]]:
    try:
        from aragora.pulse.manager import PulseManager
        return True, None
    except ImportError:
        return False, "Pulse module not installed"


def _check_genesis() -> tuple[bool, Optional[str]]:
    try:
        from aragora.genesis.ledger import GenesisLedger
        return True, None
    except ImportError:
        return False, "Genesis module not installed"


def _check_z3() -> tuple[bool, Optional[str]]:
    try:
        import z3
        return True, None
    except ImportError:
        return False, "z3-solver not installed"


def _check_lean() -> tuple[bool, Optional[str]]:
    # Lean connector is still in development
    return False, "Lean connector not yet implemented"


def _check_laboratory() -> tuple[bool, Optional[str]]:
    try:
        from aragora.genesis.laboratory import PersonaLaboratory
        return True, None
    except ImportError:
        return False, "PersonaLaboratory not available"


def _check_calibration() -> tuple[bool, Optional[str]]:
    try:
        from aragora.ranking.calibration import CalibrationTracker
        return True, None
    except ImportError:
        return False, "CalibrationTracker not available"


def _check_evolution() -> tuple[bool, Optional[str]]:
    try:
        from aragora.genesis.prompt_evolver import PromptEvolver
        return True, None
    except ImportError:
        return False, "PromptEvolver not available"


def _check_red_team() -> tuple[bool, Optional[str]]:
    try:
        from aragora.debate.red_team import RedTeamMode
        return True, None
    except ImportError:
        return False, "RedTeamMode not available"


def _check_probes() -> tuple[bool, Optional[str]]:
    try:
        from aragora.debate.capability_probes import ProbeRunner
        return True, None
    except ImportError:
        return False, "ProbeRunner not available"


def _check_continuum() -> tuple[bool, Optional[str]]:
    try:
        from aragora.memory.continuum import ContinuumMemory
        return True, None
    except ImportError:
        return False, "ContinuumMemory not available"


def _check_consensus() -> tuple[bool, Optional[str]]:
    try:
        from aragora.memory.consensus import ConsensusMemory
        return True, None
    except ImportError:
        return False, "ConsensusMemory not available"


def _check_insights() -> tuple[bool, Optional[str]]:
    try:
        from aragora.memory.insights import InsightStore
        return True, None
    except ImportError:
        return False, "InsightStore not available"


def _check_moments() -> tuple[bool, Optional[str]]:
    try:
        from aragora.debate.moments import MomentDetector
        return True, None
    except ImportError:
        return False, "MomentDetector not available"


def _check_tournaments() -> tuple[bool, Optional[str]]:
    try:
        from aragora.ranking.tournaments import TournamentRunner
        return True, None
    except ImportError:
        return False, "TournamentRunner not available"


def _check_elo() -> tuple[bool, Optional[str]]:
    try:
        from aragora.ranking.elo import ELOSystem
        return True, None
    except ImportError:
        return False, "ELOSystem not available"


def _check_crux() -> tuple[bool, Optional[str]]:
    try:
        from aragora.reasoning.crux import CruxAnalyzer
        return True, None
    except ImportError:
        return False, "CruxAnalyzer not available"


def _check_rhetorical() -> tuple[bool, Optional[str]]:
    try:
        from aragora.debate.rhetorical_observer import RhetoricalObserver
        return True, None
    except ImportError:
        return False, "RhetoricalObserver not available"


def _check_trickster() -> tuple[bool, Optional[str]]:
    try:
        from aragora.debate.trickster import EvidencePoweredTrickster
        return True, None
    except ImportError:
        return False, "Trickster not available"


def get_all_features() -> dict[str, dict[str, Any]]:
    """Get full feature matrix with availability status."""
    result = {}
    for feature_id, feature in FEATURE_REGISTRY.items():
        available, reason = _check_feature_available(feature_id)
        result[feature_id] = {
            "id": feature_id,
            "name": feature.name,
            "description": feature.description,
            "category": feature.category,
            "status": feature.status,
            "available": available,
            "reason": reason,
            "install_hint": feature.install_hint if not available else None,
            "endpoints": feature.endpoints,
        }
    return result


def get_available_features() -> list[str]:
    """Get list of feature IDs that are currently available."""
    return [
        feature_id
        for feature_id in FEATURE_REGISTRY
        if _check_feature_available(feature_id)[0]
    ]


def get_unavailable_features() -> dict[str, str]:
    """Get dict of unavailable feature IDs with reasons."""
    result = {}
    for feature_id in FEATURE_REGISTRY:
        available, reason = _check_feature_available(feature_id)
        if not available:
            result[feature_id] = reason or "Unknown"
    return result


# =============================================================================
# Standard Error Helper
# =============================================================================

def feature_unavailable_response(
    feature_id: str,
    message: Optional[str] = None,
) -> HandlerResult:
    """
    Create a standardized response for unavailable features.

    This should be used by all handlers when an optional feature is missing.

    Args:
        feature_id: The feature identifier (e.g., "pulse", "genesis")
        message: Optional custom message (defaults to feature description)

    Returns:
        HandlerResult with 503 status and helpful information
    """
    feature = FEATURE_REGISTRY.get(feature_id)

    if feature:
        default_message = f"{feature.name} is not available"
        return error_response(
            message or default_message,
            status=503,
            code="FEATURE_UNAVAILABLE",
            suggestion=feature.install_hint,
            details={
                "feature": feature_id,
                "name": feature.name,
                "status": feature.status,
            },
        )
    else:
        return error_response(
            message or f"Feature '{feature_id}' is not available",
            status=503,
            code="FEATURE_UNAVAILABLE",
            details={"feature": feature_id},
        )


# =============================================================================
# Handler
# =============================================================================

class FeaturesHandler(BaseHandler):
    """Handler for feature availability endpoints."""

    ROUTES = {
        "/api/features": "_get_features_summary",
        "/api/features/available": "_get_available",
        "/api/features/all": "_get_all_features",
        "/api/features/handlers": "_get_handler_stability",
        "/api/features/config": "_handle_config",  # User preferences
        "/api/features/{feature_id}": "_get_feature_status",
    }

    # Default feature preferences (all toggleable features)
    DEFAULT_PREFERENCES = {
        "calibration": True,  # Agent calibration tracking
        "trickster": False,  # Hollow consensus detection (can be noisy)
        "rhetorical": True,  # Rhetorical pattern analysis
        "insights": True,  # Debate insights extraction
        "moments": True,  # Moment detection
        "crux": True,  # Crux analysis
        "evolution": True,  # Prompt evolution
        "continuum_memory": True,  # Multi-tier memory
        "consensus_memory": True,  # Consensus history
        "laboratory": True,  # Persona laboratory
        # Display preferences
        "show_advanced_metrics": False,
        "compact_mode": False,
        "theme": "system",  # light, dark, system
        # Debate preferences
        "default_mode": "standard",  # standard, graph, matrix
        "default_rounds": 3,
        "default_agents": "claude,gemini,gpt4",
        # Notification preferences
        "telegram_enabled": False,
        "email_digest": "none",  # none, daily, weekly
        "consensus_alert_threshold": 0.7,
    }

    def __init__(self, server_context: dict):
        """Initialize with server context."""
        super().__init__(server_context)

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        # Direct route match
        if path in self.ROUTES:
            return True
        # Handle parameterized routes: /api/features/{feature_id}
        if path.startswith("/api/features/") and path not in self.ROUTES:
            # Check it's not a nested path we don't handle
            parts = path.split("/")
            if len(parts) == 4:  # /api/features/{feature_id}
                return True
        return False

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Route GET/POST requests to appropriate methods."""
        # Direct route match
        if path in self.ROUTES:
            method_name = self.ROUTES[path]
            method = getattr(self, method_name, None)
            if method:
                # Config endpoint needs handler for auth and POST body
                if method_name == "_handle_config":
                    return method(handler)
                return method()

        # Parameterized route: /api/features/{feature_id}
        if path.startswith("/api/features/"):
            parts = path.split("/")
            if len(parts) == 4:
                feature_id = parts[3]
                return self._get_feature_status(feature_id)

        return None

    def _get_features_summary(self) -> HandlerResult:
        """Get summary of feature availability."""
        available = get_available_features()
        unavailable = get_unavailable_features()

        return json_response({
            "available_count": len(available),
            "unavailable_count": len(unavailable),
            "available": available,
            "unavailable": list(unavailable.keys()),
            "categories": self._get_categories(),
        })

    def _get_available(self) -> HandlerResult:
        """Get list of available features."""
        available = get_available_features()
        return json_response({
            "features": available,
            "count": len(available),
        })

    def _get_all_features(self) -> HandlerResult:
        """Get full feature matrix."""
        features = get_all_features()

        # Group by category
        by_category: dict[str, list] = {}
        for feature_id, info in features.items():
            category = info["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(info)

        return json_response({
            "features": features,
            "by_category": by_category,
            "total": len(features),
        })

    def _get_feature_status(self, feature_id: str) -> HandlerResult:
        """Get status of a specific feature."""
        if feature_id not in FEATURE_REGISTRY:
            return error_response(
                f"Unknown feature: {feature_id}",
                status=404,
                code="NOT_FOUND",
            )

        feature = FEATURE_REGISTRY[feature_id]
        available, reason = _check_feature_available(feature_id)

        return json_response({
            "id": feature_id,
            "name": feature.name,
            "description": feature.description,
            "category": feature.category,
            "status": feature.status,
            "available": available,
            "reason": reason,
            "install_hint": feature.install_hint if not available else None,
            "endpoints": feature.endpoints,
            "requires": feature.requires,
        })

    def _get_categories(self) -> dict[str, int]:
        """Get feature count by category."""
        categories: dict[str, int] = {}
        for feature in FEATURE_REGISTRY.values():
            category = feature.category
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _get_handler_stability(self) -> HandlerResult:
        """Get stability classification of all API handlers.

        Returns handler names with their stability levels:
        - stable: Production-ready, API unlikely to change
        - experimental: Works but may change, use with awareness
        - preview: Early access, expect changes
        - deprecated: Being phased out
        """
        from aragora.server.handlers import (
            HANDLER_STABILITY,
            ALL_HANDLERS,
            get_all_handler_stability,
        )

        stability_map = get_all_handler_stability()

        # Group by stability level
        by_stability: dict[str, list[str]] = {
            "stable": [],
            "experimental": [],
            "preview": [],
            "deprecated": [],
        }

        for handler_class in ALL_HANDLERS:
            handler_name = handler_class.__name__
            stability = stability_map.get(handler_name, "experimental")
            by_stability[stability].append(handler_name)

        return json_response({
            "handlers": stability_map,
            "by_stability": by_stability,
            "counts": {level: len(handlers) for level, handlers in by_stability.items()},
            "total": len(ALL_HANDLERS),
        })

    def _handle_config(self, handler=None) -> HandlerResult:
        """Handle feature configuration requests (GET/POST).

        GET: Returns user's feature preferences
        POST: Updates user's feature preferences

        For authenticated users, preferences are stored in users.db.
        For guests, returns defaults (frontend should use localStorage).
        """
        from aragora.billing.jwt_auth import extract_user_from_request

        method = getattr(handler, 'command', 'GET') if handler else 'GET'

        # Get user context if authenticated
        user_store = self.ctx.get('user_store')
        user_ctx = None
        if user_store and handler:
            user_ctx = extract_user_from_request(handler, user_store)

        if method == 'GET':
            return self._get_config(user_ctx, user_store)
        elif method == 'POST':
            return self._update_config(handler, user_ctx, user_store)
        else:
            return error_response(f"Method {method} not allowed", status=405)

    def _get_config(self, user_ctx, user_store) -> HandlerResult:
        """Get user's feature configuration."""
        preferences = dict(self.DEFAULT_PREFERENCES)

        # If authenticated, try to load user preferences
        if user_ctx and user_ctx.is_authenticated and user_store:
            try:
                user_prefs = user_store.get_user_preferences(user_ctx.user_id)
                if user_prefs:
                    # Merge with defaults (user prefs override)
                    preferences.update(user_prefs)
            except Exception as e:
                logger.warning(f"Failed to load user preferences: {e}")

        # Get feature availability for context
        features = get_all_features()
        feature_toggles = []

        for feature_id, info in features.items():
            if feature_id in preferences:
                feature_toggles.append({
                    "id": feature_id,
                    "name": info["name"],
                    "description": info["description"],
                    "category": info["category"],
                    "available": info["available"],
                    "enabled": preferences.get(feature_id, False) if info["available"] else False,
                    "install_hint": info.get("install_hint"),
                })

        return json_response({
            "preferences": preferences,
            "feature_toggles": feature_toggles,
            "is_authenticated": user_ctx.is_authenticated if user_ctx else False,
            "defaults": self.DEFAULT_PREFERENCES,
        })

    def _update_config(self, handler, user_ctx, user_store) -> HandlerResult:
        """Update user's feature configuration."""
        import json as json_module

        # Parse request body
        try:
            content_length = int(handler.headers.get('Content-Length', 0))
            body = handler.rfile.read(content_length) if content_length > 0 else b'{}'
            updates = json_module.loads(body.decode('utf-8'))
        except (json_module.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON body: {e}", status=400)

        # Validate updates
        valid_keys = set(self.DEFAULT_PREFERENCES.keys())
        invalid_keys = set(updates.keys()) - valid_keys
        if invalid_keys:
            return error_response(
                f"Invalid preference keys: {', '.join(invalid_keys)}",
                status=400,
                code="INVALID_KEYS",
                details={"valid_keys": list(valid_keys)},
            )

        # Validate value types
        for key, value in updates.items():
            expected_type = type(self.DEFAULT_PREFERENCES[key])
            if not isinstance(value, expected_type):
                return error_response(
                    f"Invalid type for '{key}': expected {expected_type.__name__}",
                    status=400,
                )

        # If authenticated, persist to user store
        if user_ctx and user_ctx.is_authenticated and user_store:
            try:
                # Get existing preferences and merge
                existing = user_store.get_user_preferences(user_ctx.user_id) or {}
                existing.update(updates)
                user_store.set_user_preferences(user_ctx.user_id, existing)
                logger.info(f"Updated preferences for user {user_ctx.user_id}")
            except Exception as e:
                logger.error(f"Failed to save user preferences: {e}")
                return error_response(
                    "Failed to save preferences",
                    status=500,
                    code="SAVE_FAILED",
                )

            return json_response({
                "success": True,
                "updated": list(updates.keys()),
                "message": "Preferences saved to account",
            })
        else:
            # For guests, just acknowledge (frontend handles localStorage)
            return json_response({
                "success": True,
                "updated": list(updates.keys()),
                "message": "Preferences acknowledged (save locally for guests)",
                "is_authenticated": False,
            })


# Export for use by other handlers
__all__ = [
    "FEATURE_REGISTRY",
    "FeatureInfo",
    "FeaturesHandler",
    "feature_unavailable_response",
    "get_all_features",
    "get_available_features",
    "get_unavailable_features",
]
