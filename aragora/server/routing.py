"""
Type-safe route registry and dispatcher.

Provides declarative route definitions with typed parameter extraction,
pre-compiled pattern matching, and centralized handler registration.

Usage:
    from aragora.server.routing import RouteRegistry, Route, RouteMatch

    registry = RouteRegistry()

    # Define routes with typed parameters
    registry.register(
        Route("/api/agent/{name}/profile", handler=AgentsHandler, method="GET"),
        Route("/api/debates/{id}", handler=DebatesHandler, method="GET"),
    )

    # Match a request
    match = registry.match("/api/agent/claude/profile", "GET")
    if match:
        result = match.handler.handle(match.path, query_params, match.params, context)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)


# Parameter type converters

# Maximum values for integer/float parameters to prevent overflow attacks
_MAX_INT_VALUE = 2**31 - 1  # 32-bit signed int max
_MIN_INT_VALUE = -(2**31)  # 32-bit signed int min
_MAX_FLOAT_VALUE = 1e308  # Avoid float overflow


class ParameterConversionError(ValueError):
    """Raised when a route parameter cannot be converted to the expected type."""

    def __init__(self, param_name: str, value: str, expected_type: str, reason: str = ""):
        self.param_name = param_name
        self.value = value
        self.expected_type = expected_type
        self.reason = reason
        msg = f"Cannot convert parameter '{param_name}' value '{value}' to {expected_type}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


def _convert_str(value: str) -> str:
    """Identity converter for string parameters.

    Args:
        value: String value to return unchanged.

    Returns:
        The input string value.
    """
    return value


def _convert_int(value: str) -> int:
    """Convert string to integer with bounds checking.

    Args:
        value: String representation of an integer.

    Returns:
        Parsed integer value.

    Raises:
        ParameterConversionError: If value cannot be parsed or is out of bounds.
    """
    try:
        result = int(value)
        if result > _MAX_INT_VALUE or result < _MIN_INT_VALUE:
            raise ParameterConversionError(
                "unknown",
                value,
                "int",
                f"value out of bounds ({_MIN_INT_VALUE} to {_MAX_INT_VALUE})",
            )
        return result
    except ValueError as e:
        raise ParameterConversionError("unknown", value, "int", str(e))


def _convert_float(value: str) -> float:
    """Convert string to float with bounds checking.

    Args:
        value: String representation of a float.

    Returns:
        Parsed float value.

    Raises:
        ParameterConversionError: If value cannot be parsed or is out of bounds.
    """
    try:
        result = float(value)
        if abs(result) > _MAX_FLOAT_VALUE:
            raise ParameterConversionError(
                "unknown", value, "float", f"value out of bounds (magnitude <= {_MAX_FLOAT_VALUE})"
            )
        return result
    except ValueError as e:
        raise ParameterConversionError("unknown", value, "float", str(e))


# Parameter validation patterns
PARAM_PATTERNS = {
    "id": re.compile(r"^[a-zA-Z0-9_-]{1,64}$"),
    "name": re.compile(r"^[a-zA-Z0-9_-]{1,32}$"),
    "slug": re.compile(r"^[a-zA-Z0-9_-]{1,128}$"),
    "domain": re.compile(r"^[a-zA-Z0-9_-]{1,50}$"),
    "any": re.compile(r"^[^/]+$"),  # Any non-slash chars
}


@dataclass
class ParamSpec:
    """Specification for a route parameter."""

    name: str
    param_type: type = str
    pattern: Optional[str] = None  # Key in PARAM_PATTERNS or custom regex
    required: bool = True
    default: Any = None

    def validate(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate a parameter value."""
        if not value:
            if self.required:
                return False, f"Missing required parameter: {self.name}"
            return True, None

        # Check pattern if specified
        if self.pattern:
            if self.pattern in PARAM_PATTERNS:
                pattern = PARAM_PATTERNS[self.pattern]
            else:
                pattern = re.compile(self.pattern)

            if not pattern.match(value):
                return False, f"Invalid {self.name} format"

        return True, None

    def convert(self, value: str) -> Any:
        """Convert string value to typed value with proper error handling.

        Args:
            value: String value to convert.

        Returns:
            Converted value in the appropriate type.

        Raises:
            ParameterConversionError: If conversion fails or value is out of bounds.
        """
        if not value and not self.required:
            return self.default

        try:
            if self.param_type is int:
                return _convert_int(value)
            elif self.param_type is float:
                return _convert_float(value)
            return value
        except ParameterConversionError as e:
            # Re-raise with proper parameter name
            raise ParameterConversionError(self.name, value, e.expected_type, e.reason)


@dataclass
class Route:
    """
    A route definition with pattern matching and parameter extraction.

    Pattern syntax:
        /api/debates - exact match
        /api/agent/{name} - captures 'name' parameter
        /api/agent/{name}/profile - captures 'name', matches literal 'profile'
        /api/debates/* - wildcard, matches any suffix
    """

    pattern: str
    handler_name: str  # Handler class name for lookup
    method: str = "GET"
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    priority: int = 0  # Higher = matched first (for conflicts)

    # Compiled pattern (set during registration)
    _regex: Optional[Pattern] = field(default=None, repr=False)
    _param_names: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Compile the route pattern to regex."""
        self._compile()

    def _compile(self) -> None:
        """Compile pattern to regex with named groups."""
        pattern = self.pattern
        param_names = []

        # Extract parameter names and build regex
        # {name} -> (?P<name>[^/]+)
        # {name:pattern} -> (?P<name>pattern)
        # * -> .*

        def replace_param(match):
            full = match.group(0)
            if full == "*":
                return ".*"

            # Parse {name} or {name:pattern}
            inner = full[1:-1]
            if ":" in inner:
                name, pat = inner.split(":", 1)
            else:
                name = inner
                # Use param spec pattern if available
                if name in self.params:
                    pat_key = self.params[name].pattern or "any"
                    if pat_key in PARAM_PATTERNS:
                        pat = PARAM_PATTERNS[pat_key].pattern
                        # Strip anchors - they'll be applied to the full regex
                        if pat.startswith("^"):
                            pat = pat[1:]
                        if pat.endswith("$"):
                            pat = pat[:-1]
                    else:
                        pat = pat_key
                else:
                    pat = "[^/]+"

            param_names.append(name)
            return f"(?P<{name}>{pat})"

        regex_pattern = re.sub(r"\{[^}]+\}|\*", replace_param, pattern)
        self._regex = re.compile(f"^{regex_pattern}$")
        self._param_names = param_names

    def match(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Match path against this route pattern.

        Returns extracted parameters if match, None otherwise.
        """
        if self._regex is None:
            return None

        m = self._regex.match(path)
        if not m:
            return None

        # Extract and convert parameters
        params = {}
        for name in self._param_names:
            value = m.group(name)
            if name in self.params:
                spec = self.params[name]
                is_valid, err = spec.validate(value)
                if not is_valid:
                    return None  # Validation failed
                params[name] = spec.convert(value)
            else:
                params[name] = value

        return params


@dataclass
class RouteMatch:
    """Result of matching a request to a route."""

    route: Route
    path: str
    method: str
    params: Dict[str, Any]
    handler_name: str

    @property
    def matched(self) -> bool:
        """Whether a route was matched."""
        return self.route is not None


class RouteRegistry:
    """
    Central registry for routes with efficient matching.

    Routes are organized by method and sorted by priority for matching.
    Exact paths are indexed for O(1) lookup; patterns use linear search.
    """

    def __init__(self) -> None:
        # Exact path index: method -> path -> Route
        self._exact: Dict[str, Dict[str, Route]] = {}
        # Pattern routes: method -> List[Route] (sorted by priority)
        self._patterns: Dict[str, List[Route]] = {}
        # Handler instances: name -> handler
        self._handlers: Dict[str, Any] = {}
        # All registered routes
        self._routes: List[Route] = []

    def register(self, *routes: Route) -> "RouteRegistry":
        """
        Register one or more routes.

        Returns self for chaining.
        """
        for route in routes:
            self._add_route(route)
        return self

    def _add_route(self, route: Route) -> None:
        """Add a single route to the registry."""
        method = route.method.upper()

        # Check if it's an exact pattern (no params or wildcards)
        if "{" not in route.pattern and "*" not in route.pattern:
            if method not in self._exact:
                self._exact[method] = {}
            self._exact[method][route.pattern] = route
        else:
            if method not in self._patterns:
                self._patterns[method] = []
            self._patterns[method].append(route)
            # Sort by priority (higher first)
            self._patterns[method].sort(key=lambda r: -r.priority)

        self._routes.append(route)

    def register_handler(self, name: str, handler: Any) -> None:
        """Register a handler instance by name."""
        self._handlers[name] = handler

    def get_handler(self, name: str) -> Optional[Any]:
        """Get a registered handler by name."""
        return self._handlers.get(name)

    def match(self, path: str, method: str = "GET") -> Optional[RouteMatch]:
        """
        Match a path and method to a registered route.

        Returns RouteMatch if found, None otherwise.
        """
        method = method.upper()

        # Try exact match first (O(1))
        if method in self._exact and path in self._exact[method]:
            route = self._exact[method][path]
            return RouteMatch(
                route=route,
                path=path,
                method=method,
                params={},
                handler_name=route.handler_name,
            )

        # Try pattern matching (O(n) but sorted by priority)
        if method in self._patterns:
            for route in self._patterns[method]:
                params = route.match(path)
                if params is not None:
                    return RouteMatch(
                        route=route,
                        path=path,
                        method=method,
                        params=params,
                        handler_name=route.handler_name,
                    )

        return None

    def get_routes(self, method: str | None = None) -> List[Route]:
        """Get all registered routes, optionally filtered by method."""
        if method:
            method = method.upper()
            return [r for r in self._routes if r.method == method]
        return list(self._routes)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_routes": len(self._routes),
            "exact_routes": sum(len(v) for v in self._exact.values()),
            "pattern_routes": sum(len(v) for v in self._patterns.values()),
            "handlers": list(self._handlers.keys()),
            "methods": list(set(r.method for r in self._routes)),
        }


# Convenience function for creating routes with common patterns


def api_route(
    pattern: str,
    handler: str,
    method: str = "GET",
    **param_specs: Dict[str, Any],
) -> Route:
    """
    Create an API route with common defaults.

    Args:
        pattern: Route pattern (e.g., "/api/debates/{id}")
        handler: Handler class name
        method: HTTP method
        **param_specs: Parameter specifications as dicts

    Example:
        api_route(
            "/api/agent/{name}/profile",
            "AgentsHandler",
            name={"pattern": "name", "type": str},
        )
    """
    params = {}
    for name, spec in param_specs.items():
        if isinstance(spec, dict):
            params[name] = ParamSpec(
                name=name,
                param_type=spec.get("type", str),
                pattern=spec.get("pattern"),
                required=spec.get("required", True),
                default=spec.get("default"),
            )
        elif isinstance(spec, ParamSpec):
            params[name] = spec
        else:
            # Assume it's just a pattern string
            params[name] = ParamSpec(name=name, pattern=str(spec))

    return Route(pattern=pattern, handler_name=handler, method=method, params=params)


# Default routes for the aragora API
# These can be used to initialize a registry with standard routes


def create_default_routes() -> List[Route]:
    """Create the default routes for the aragora API."""
    return [
        # System routes
        api_route("/api/health", "SystemHandler"),
        api_route("/api/nomic/state", "SystemHandler"),
        api_route("/api/nomic/log", "SystemHandler"),
        api_route("/api/modes", "SystemHandler"),
        api_route("/api/history/cycles", "SystemHandler"),
        api_route("/api/history/events", "SystemHandler"),
        api_route("/api/history/debates", "SystemHandler"),
        api_route("/api/history/summary", "SystemHandler"),
        # Debates routes
        api_route("/api/debates", "DebatesHandler"),
        api_route("/api/debates/{id}", "DebatesHandler", id={"pattern": "slug"}),
        api_route("/api/debates/{id}/export", "DebatesHandler", id={"pattern": "slug"}),
        api_route("/api/debates/{id}/messages", "DebatesHandler", id={"pattern": "slug"}),
        # Agent routes
        api_route("/api/leaderboard", "AgentsHandler"),
        api_route("/api/rankings", "AgentsHandler"),
        api_route("/api/calibration/leaderboard", "AgentsHandler"),
        api_route("/api/matches/recent", "AgentsHandler"),
        api_route("/api/agent/compare", "AgentsHandler"),
        api_route("/api/agent/{name}/profile", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/history", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/calibration", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/consistency", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/flips", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/network", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/rivals", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/allies", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/moments", "AgentsHandler", name={"pattern": "name"}),
        api_route("/api/agent/{name}/positions", "AgentsHandler", name={"pattern": "name"}),
        api_route(
            "/api/agent/{name}/head-to-head/{opponent}",
            "AgentsHandler",
            name={"pattern": "name"},
            opponent={"pattern": "name"},
        ),
        api_route("/api/flips/recent", "AgentsHandler"),
        api_route("/api/flips/summary", "AgentsHandler"),
        # Analytics routes
        api_route("/api/analytics/disagreements", "AnalyticsHandler"),
        api_route("/api/analytics/role-rotation", "AnalyticsHandler"),
        api_route("/api/analytics/early-stops", "AnalyticsHandler"),
        api_route("/api/ranking/stats", "AnalyticsHandler"),
        api_route("/api/memory/stats", "AnalyticsHandler"),
        api_route("/api/memory/tier-stats", "AnalyticsHandler"),
        # Pulse routes
        api_route("/api/pulse/trending", "PulseHandler"),
        api_route("/api/pulse/suggest", "PulseHandler"),
        # Consensus routes
        api_route("/api/consensus/similar", "ConsensusHandler"),
        api_route("/api/consensus/settled", "ConsensusHandler"),
        api_route("/api/consensus/stats", "ConsensusHandler"),
        api_route("/api/consensus/dissents", "ConsensusHandler"),
        api_route("/api/consensus/contrarian-views", "ConsensusHandler"),
        api_route("/api/consensus/risk-warnings", "ConsensusHandler"),
        api_route(
            "/api/consensus/domain/{domain}", "ConsensusHandler", domain={"pattern": "domain"}
        ),
        # Belief network routes
        api_route(
            "/api/belief-network/{debate_id}/cruxes", "BeliefHandler", debate_id={"pattern": "id"}
        ),
        api_route(
            "/api/belief-network/{debate_id}/claims/{claim_id}/support",
            "BeliefHandler",
            debate_id={"pattern": "id"},
            claim_id={"pattern": "id"},
        ),
        api_route(
            "/api/belief-network/{debate_id}/trace", "BeliefHandler", debate_id={"pattern": "id"}
        ),
        # Decision explainability routes
        api_route(
            "/api/decisions/{request_id}/explain",
            "DecisionExplainHandler",
            request_id={"pattern": "id"},
        ),
        api_route(
            "/api/v1/decisions/{request_id}/explain",
            "DecisionExplainHandler",
            request_id={"pattern": "id"},
        ),
        # Critique routes
        api_route("/api/critique/patterns", "CritiqueHandler"),
        api_route(
            "/api/critique/patterns/domain/{domain}",
            "CritiqueHandler",
            domain={"pattern": "domain"},
        ),
        api_route("/api/critique/reputation/{agent}", "CritiqueHandler", agent={"pattern": "name"}),
        # Genesis routes
        api_route("/api/genesis/ledger", "GenesisHandler"),
        api_route("/api/genesis/genome/{genome_id}", "GenesisHandler", genome_id={"pattern": "id"}),
        api_route(
            "/api/genesis/debate/{debate_id}/events", "GenesisHandler", debate_id={"pattern": "id"}
        ),
        # Replay routes
        api_route("/api/replays", "ReplaysHandler"),
        api_route("/api/replays/{id}", "ReplaysHandler", id={"pattern": "id"}),
        api_route("/api/replays/{id}/evolution", "ReplaysHandler", id={"pattern": "id"}),
        api_route("/api/replays/{id}/events", "ReplaysHandler", id={"pattern": "id"}),
        # Tournament routes
        api_route("/api/tournaments/active", "TournamentHandler"),
        api_route("/api/tournaments/{id}", "TournamentHandler", id={"pattern": "id"}),
        api_route("/api/tournaments/{id}/brackets", "TournamentHandler", id={"pattern": "id"}),
        api_route("/api/tournaments/{id}/calibration", "TournamentHandler", id={"pattern": "id"}),
        # Memory routes
        api_route("/api/memory/continuum/retrieve", "MemoryHandler"),
        api_route("/api/memory/continuum/consolidate", "MemoryHandler", method="POST"),
        # Leaderboard view
        api_route("/api/leaderboard-view", "LeaderboardViewHandler"),
        # Metrics
        api_route("/api/metrics", "MetricsHandler"),
        api_route("/api/metrics/cache", "MetricsHandler"),
        # Dashboard
        api_route("/api/dashboard/debates", "DashboardHandler"),
        # Analytics Metrics
        api_route("/api/v1/analytics/debates/overview", "AnalyticsMetricsHandler"),
        api_route("/api/v1/analytics/debates/trends", "AnalyticsMetricsHandler"),
        api_route("/api/v1/analytics/debates/topics", "AnalyticsMetricsHandler"),
        api_route("/api/v1/analytics/debates/outcomes", "AnalyticsMetricsHandler"),
        api_route("/api/v1/analytics/agents/leaderboard", "AnalyticsMetricsHandler"),
        api_route(
            "/api/v1/analytics/agents/{agent_id}/performance",
            "AnalyticsMetricsHandler",
            agent_id={"pattern": "name"},
        ),
        api_route("/api/v1/analytics/agents/comparison", "AnalyticsMetricsHandler"),
        api_route("/api/v1/analytics/agents/trends", "AnalyticsMetricsHandler"),
        api_route("/api/v1/analytics/usage/tokens", "AnalyticsMetricsHandler"),
        api_route("/api/v1/analytics/usage/costs", "AnalyticsMetricsHandler"),
        api_route("/api/v1/analytics/usage/active_users", "AnalyticsMetricsHandler"),
    ]


# Global registry instance
_registry: Optional[RouteRegistry] = None


def get_registry() -> RouteRegistry:
    """Get or create the global route registry."""
    global _registry
    if _registry is None:
        _registry = RouteRegistry()
        for route in create_default_routes():
            _registry.register(route)
    return _registry


def set_registry(registry: RouteRegistry) -> None:
    """Set the global route registry (for testing)."""
    global _registry
    _registry = registry
