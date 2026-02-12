"""
Core infrastructure for handler registry.

This module provides:
- Safe import utility for graceful handler degradation
- Async coroutine runner for HTTP threads
- RouteIndex for O(1) handler dispatch
- Handler validation functions
- HandlerRegistryMixin base class

All other handler registry modules depend on this core.
"""

from __future__ import annotations

import asyncio
import glob as glob_mod
import importlib
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type alias for handler classes that may be None when handlers are unavailable
# This allows proper type hints without requiring type: ignore comments
HandlerType = Optional[type[Any]]


def _safe_import(module_path: str, class_name: str) -> HandlerType:
    """Safely import a handler class with graceful fallback.

    Returns the handler class if import succeeds, None otherwise.
    Individual failures are logged as warnings and don't cascade.
    """
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except Exception as e:
        logger.warning(f"Failed to import {class_name} from {module_path}: {e}")
        return None


def _run_handler_coroutine(coro: Any) -> Any:
    """Run an async handler coroutine from the sync HTTP thread.

    When a PostgreSQL pool is initialized, schedules the coroutine on the
    main event loop (where asyncpg pool lives) using run_coroutine_threadsafe.
    This ensures:
    - asyncpg pool.acquire() works (same event loop)
    - nest_asyncio allows nested run_until_complete() from sync store wrappers

    Falls back to creating a local event loop when no pool is configured
    (SQLite-only mode).
    """
    # Try to use the main event loop (where asyncpg pool was created)
    try:
        from aragora.storage.pool_manager import get_pool_event_loop

        main_loop = get_pool_event_loop()
        if main_loop is not None and main_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, main_loop)
            return future.result(timeout=60)
    except ImportError:
        # pool_manager not available - use fallback below
        pass
    except TimeoutError:
        # Timeout waiting for coroutine - close it and re-raise
        coro.close()
        raise
    except (RuntimeError, OSError):
        # Coroutine may have started execution - cannot reuse, must re-raise
        # (e.g., if sync store methods called run_async() from async context)
        coro.close()
        raise

    # Fallback: create a local event loop (works for SQLite-only mode)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class RouteIndex:
    """O(1) route lookup index for handler dispatch.

    Builds an index of exact paths and prefix patterns at initialization,
    enabling fast route resolution without iterating through all handlers.

    Performance:
    - Exact paths: O(1) dict lookup
    - Dynamic paths: O(1) LRU cache hit, O(n) cache miss with prefix scan
    """

    def __init__(self) -> None:
        # Exact path -> (attr_name, handler) mapping
        self._exact_routes: dict[str, tuple[str, Any]] = {}
        # Prefix patterns for dynamic routes: [(prefix, attr_name, handler)]
        self._prefix_routes: list[tuple[str, str, Any]] = []
        # Cache for resolved dynamic routes
        self._cache_size: int = 500

    def build(self, registry_mixin: Any, handler_registry: list[tuple[str, Any]]) -> None:
        """Build route index from initialized handlers.

        Extracts ROUTES from each handler for exact matching,
        and identifies prefix patterns from can_handle logic.

        Args:
            registry_mixin: The HandlerRegistryMixin instance with initialized handlers
            handler_registry: List of (attr_name, handler_class) pairs
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
            "_analytics_dashboard_handler": ["/api/analytics/"],
            "_endpoint_analytics_handler": ["/api/analytics/endpoints"],
            "_analytics_metrics_handler": ["/api/v1/analytics/"],
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
            "_plugins_handler": ["/api/plugins/", "/api/v1/plugins/"],
            "_audio_handler": ["/audio/", "/api/podcast/"],
            "_devices_handler": ["/api/devices/", "/api/v1/devices/"],
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
            "_feature_integrations_handler": ["/api/v1/integrations", "/api/integrations"],
            "_external_integrations_handler": [
                "/api/v1/integrations/zapier",
                "/api/v1/integrations/make",
                "/api/v1/integrations/n8n",
                "/api/integrations/zapier",
                "/api/integrations/make",
                "/api/integrations/n8n",
            ],
            "_integration_management_handler": ["/api/v2/integrations"],
            "_oauth_wizard_handler": ["/api/v2/integrations/wizard"],
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
            "_moderation_handler": ["/api/moderation/"],
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
            "_backup_handler": [
                "/api/v2/backups",
                "/api/v2/backups/",
            ],
            "_dr_handler": [
                "/api/v2/dr",
                "/api/v2/dr/",
            ],
            "_compliance_handler": [
                "/api/v2/compliance",
                "/api/v2/compliance/",
            ],
            "_routing_handler": [
                "/api/routing/",
                "/api/v1/routing/",
            ],
            "_workflow_handler": [
                "/api/workflows",
                "/api/workflow-templates",
                "/api/workflow-executions",
                "/api/v1/workflows",
            ],
            "_slo_handler": [
                "/api/slos",
                "/api/slos/",
                "/api/v1/slos",
            ],
            "_connectors_handler": [
                "/api/connectors",
                "/api/connectors/",
                "/api/v1/connectors",
            ],
            "_marketplace_handler": [
                "/api/marketplace",
                "/api/marketplace/",
                "/api/v1/marketplace",
            ],
            "_onboarding_handler": [
                "/api/onboarding/",
                "/api/v1/onboarding/",
            ],
            "_sme_usage_dashboard_handler": [
                "/api/v1/usage/",
            ],
            "_canvas_handler": [
                "/api/v1/canvas",
                "/api/v1/canvas/",
            ],
            "_gateway_handler": [
                "/api/v1/gateway/",
            ],
            "_scim_handler": [
                "/scim/",
                "/scim/v2/",
            ],
            "_computer_use_handler": [
                "/api/v1/computer-use/",
            ],
            "_unified_approvals_handler": [
                "/api/v1/approvals",
            ],
            "_rbac_handler": [
                "/api/v1/rbac/",
            ],
            "_cost_dashboard_handler": [
                "/api/v1/billing/dashboard",
            ],
            "_gastown_dashboard_handler": [
                "/api/v1/dashboard/gastown/",
            ],
            "_connector_management_handler": [
                "/api/v1/connectors/",
            ],
            "_task_execution_handler": [
                "/api/v2/tasks",
                "/api/v2/tasks/",
            ],
            "_security_debate_handler": [
                "/api/v1/audit/security/debate",
            ],
            "_autonomous_learning_handler": [
                "/api/v2/learning/",
            ],
        }

        for attr_name, _ in handler_registry:
            handler = getattr(registry_mixin, attr_name, None)
            if handler is None:
                continue

            # Extract exact routes from ROUTES attribute
            routes = getattr(handler, "ROUTES", [])
            for path in routes:
                if path not in self._exact_routes:
                    self._exact_routes[path] = (attr_name, handler)

            # Add prefix patterns (static mapping + handler-provided prefixes)
            prefixes = list(PREFIX_PATTERNS.get(attr_name, []))
            handler_prefixes = getattr(handler, "ROUTE_PREFIXES", None)
            if handler_prefixes:
                for prefix in handler_prefixes:
                    if prefix not in prefixes:
                        prefixes.append(prefix)
            for prefix in prefixes:
                self._prefix_routes.append((prefix, attr_name, handler))

        # Clear the LRU cache when index is rebuilt
        self._get_handler_cached.cache_clear()

        logger.debug(
            f"[route-index] Built index: {len(self._exact_routes)} exact, "
            f"{len(self._prefix_routes)} prefix patterns"
        )

    def get_handler(self, path: str) -> Optional[tuple[str, Any]]:
        """Get handler for path with O(1) lookup for known routes.

        Supports both versioned (/api/v1/debates) and legacy (/api/debates) paths.
        Versioned paths are normalized by stripping the version prefix before matching.

        Args:
            path: URL path to match

        Returns:
            Tuple of (attr_name, handler) or None if no match
        """
        from aragora.server.versioning import strip_version_prefix

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
    def _get_handler_cached(self, path: str, normalized_path: str) -> Optional[tuple[str, Any]]:
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
_route_index: RouteIndex | None = None


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


def validate_handler_class(handler_class: Any, handler_name: str) -> list[str]:
    """
    Validate that a handler class has required methods and attributes.

    Args:
        handler_class: The handler class to validate
        handler_name: Name for error messages

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[str] = []

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


def validate_handler_instance(handler: Any, handler_name: str) -> list[str]:
    """
    Validate an instantiated handler works correctly.

    Args:
        handler: The handler instance
        handler_name: Name for error messages

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[str] = []

    if handler is None:
        errors.append(f"{handler_name}: Handler instance is None")
        return errors

    # Verify can_handle doesn't crash with a test path
    try:
        result = handler.can_handle("/api/test-path-validation")
        if not isinstance(result, bool):
            errors.append(f"{handler_name}: can_handle() returned non-bool: {type(result)}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        errors.append(f"{handler_name}: can_handle() raised exception: {e}")

    return errors


def validate_all_handlers(
    handler_registry: list[tuple[str, Any]],
    handlers_available: bool,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """
    Validate all registered handler classes.

    This should be called at startup to catch configuration issues early.

    Args:
        handler_registry: List of (attr_name, handler_class) pairs
        handlers_available: Whether handler imports succeeded
        raise_on_error: If True, raise exception on validation failures

    Returns:
        Dict with validation results:
        - valid: List of valid handler names
        - invalid: Dict of handler name -> error messages
        - missing: List of handlers that couldn't be imported
    """
    if not handlers_available:
        logger.warning("[handler-validation] Handler imports failed, skipping validation")
        return {
            "valid": [],
            "invalid": {},
            "missing": [name for name, _ in handler_registry],
            "status": "imports_failed",
        }

    results: dict[str, Any] = {
        "valid": [],
        "invalid": {},
        "missing": [],
        "status": "ok",
    }

    for attr_name, handler_class in handler_registry:
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


def check_handler_coverage(handler_registry: list[tuple[str, Any]]) -> None:
    """Log warnings for handler classes that exist in the codebase but aren't registered.

    Scans aragora/server/handlers/ for classes ending in 'Handler' and compares
    against handler_registry. Unregistered handlers are logged as warnings.
    Called during _init_handlers to surface gaps early.
    """
    import ast

    registered_names = {
        handler_class.__name__ for _, handler_class in handler_registry if handler_class is not None
    }

    # Also include base/abstract classes that shouldn't be registered
    skip_names = {
        "BaseHandler",
        "BaseHTTPRequestHandler",
        "SecureHandler",
        "AuthenticatedHandler",
        "AsyncTypedHandler",
        "TypedHandler",
        "ResourceHandler",
        "VersionedAPIHandler",
        "CompositeHandler",
        "PermissionHandler",
        "ExampleAsyncHandler",
        "ExampleAuthenticatedHandler",
        "ExamplePermissionHandler",
        "ExampleResourceHandler",
        "ExampleTypedHandler",
        "HandlerResult",
        "MockHandler",
        "MyHandler",
        "MyResourceHandler",
        "MyBotHandler",
        # ABCs and aliased handlers (registered under different names)
        "GauntletSecureHandler",
        "IntelligenceHandler",
    }

    handler_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "handlers")
    if not os.path.isdir(handler_dir):
        return

    unregistered = []
    for py_file in glob_mod.glob(os.path.join(handler_dir, "**", "*.py"), recursive=True):
        try:
            with open(py_file) as f:
                tree = ast.parse(f.read(), filename=py_file)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Handler"):
                name = node.name
                if name not in registered_names and name not in skip_names:
                    rel_path = os.path.relpath(py_file, handler_dir)
                    unregistered.append((name, rel_path))

    if unregistered:
        logger.warning(
            f"[handlers] {len(unregistered)} handler class(es) found but not registered:"
        )
        for name, path in sorted(unregistered):
            logger.warning(f"[handlers]   - {name} ({path})")


def validate_handlers_on_init(
    registry_mixin: Any,
    handler_registry: list[tuple[str, Any]],
) -> dict[str, Any]:
    """
    Validate instantiated handlers after initialization.

    Called from _init_handlers to verify all handlers work correctly.

    Args:
        registry_mixin: The HandlerRegistryMixin instance with initialized handlers
        handler_registry: List of (attr_name, handler_class) pairs

    Returns:
        Dict with validation results
    """
    results: dict[str, Any] = {
        "valid": [],
        "invalid": {},
        "not_initialized": [],
    }

    for attr_name, handler_class in handler_registry:
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


__all__ = [
    # Types
    "HandlerType",
    # Utilities
    "_safe_import",
    "_run_handler_coroutine",
    # Route index
    "RouteIndex",
    "get_route_index",
    # Validation
    "HandlerValidationError",
    "validate_handler_class",
    "validate_handler_instance",
    "validate_all_handlers",
    "validate_handlers_on_init",
    "check_handler_coverage",
]
