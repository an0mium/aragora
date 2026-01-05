"""Tests for type-safe route registry (server/routing.py)."""

import pytest
from aragora.server.routing import (
    Route,
    RouteRegistry,
    RouteMatch,
    ParamSpec,
    api_route,
    create_default_routes,
    get_registry,
    set_registry,
    PARAM_PATTERNS,
)


class TestParamSpec:
    """Test ParamSpec validation and conversion."""

    def test_string_param_default(self):
        """Test default string parameter."""
        spec = ParamSpec(name="id")
        is_valid, err = spec.validate("abc123")
        assert is_valid is True
        assert err is None

    def test_string_param_with_pattern(self):
        """Test string parameter with pattern validation."""
        spec = ParamSpec(name="id", pattern="id")
        is_valid, err = spec.validate("valid_id-123")
        assert is_valid is True

        is_valid, err = spec.validate("invalid/id")
        assert is_valid is False

    def test_int_conversion(self):
        """Test integer conversion."""
        spec = ParamSpec(name="limit", param_type=int)
        result = spec.convert("42")
        assert result == 42
        assert isinstance(result, int)

    def test_float_conversion(self):
        """Test float conversion."""
        spec = ParamSpec(name="score", param_type=float)
        result = spec.convert("0.75")
        assert result == 0.75
        assert isinstance(result, float)

    def test_optional_param_default(self):
        """Test optional parameter returns default."""
        spec = ParamSpec(name="page", param_type=int, required=False, default=1)
        result = spec.convert("")
        assert result == 1

    def test_required_param_validation(self):
        """Test required parameter fails on empty."""
        spec = ParamSpec(name="id", required=True)
        is_valid, err = spec.validate("")
        assert is_valid is False
        assert "Missing required" in err


class TestRoute:
    """Test Route pattern matching."""

    def test_exact_match(self):
        """Test exact path matching."""
        route = Route("/api/health", "SystemHandler")
        params = route.match("/api/health")
        assert params == {}

    def test_exact_match_fails(self):
        """Test exact path doesn't match different path."""
        route = Route("/api/health", "SystemHandler")
        params = route.match("/api/other")
        assert params is None

    def test_single_param(self):
        """Test single parameter extraction."""
        route = Route("/api/debates/{id}", "DebatesHandler")
        params = route.match("/api/debates/debate_123")
        assert params == {"id": "debate_123"}

    def test_multiple_params(self):
        """Test multiple parameter extraction."""
        route = Route(
            "/api/agent/{name}/head-to-head/{opponent}",
            "AgentsHandler",
        )
        params = route.match("/api/agent/claude/head-to-head/gpt4")
        assert params == {"name": "claude", "opponent": "gpt4"}

    def test_param_with_spec(self):
        """Test parameter with type spec."""
        route = Route(
            "/api/debates/{id}",
            "DebatesHandler",
            params={"id": ParamSpec(name="id", pattern="slug")},
        )
        params = route.match("/api/debates/valid-slug_123")
        assert params == {"id": "valid-slug_123"}

    def test_param_validation_fails(self):
        """Test parameter validation failure."""
        route = Route(
            "/api/debates/{id}",
            "DebatesHandler",
            params={"id": ParamSpec(name="id", pattern="id")},
        )
        # Pattern allows max 64 chars, this should match
        params = route.match("/api/debates/valid")
        assert params is not None

        # Invalid pattern (contains /)
        params = route.match("/api/debates/invalid/path")
        assert params is None

    def test_wildcard_match(self):
        """Test wildcard pattern."""
        route = Route("/api/static/*", "StaticHandler")
        params = route.match("/api/static/css/style.css")
        assert params is not None

    def test_int_param_conversion(self):
        """Test integer parameter conversion."""
        route = Route(
            "/api/page/{num}",
            "PageHandler",
            params={"num": ParamSpec(name="num", param_type=int)},
        )
        params = route.match("/api/page/42")
        assert params == {"num": 42}
        assert isinstance(params["num"], int)

    def test_nested_path_segments(self):
        """Test deeply nested path."""
        route = Route(
            "/api/belief-network/{debate_id}/claims/{claim_id}/support",
            "BeliefHandler",
        )
        params = route.match("/api/belief-network/d123/claims/c456/support")
        assert params == {"debate_id": "d123", "claim_id": "c456"}


class TestRouteRegistry:
    """Test RouteRegistry matching."""

    def test_register_and_match_exact(self):
        """Test registering and matching exact routes."""
        registry = RouteRegistry()
        registry.register(Route("/api/health", "SystemHandler"))

        match = registry.match("/api/health")
        assert match is not None
        assert match.handler_name == "SystemHandler"
        assert match.params == {}

    def test_register_and_match_pattern(self):
        """Test registering and matching pattern routes."""
        registry = RouteRegistry()
        registry.register(Route("/api/debates/{id}", "DebatesHandler"))

        match = registry.match("/api/debates/abc123")
        assert match is not None
        assert match.handler_name == "DebatesHandler"
        assert match.params == {"id": "abc123"}

    def test_no_match_returns_none(self):
        """Test unregistered path returns None."""
        registry = RouteRegistry()
        match = registry.match("/api/unknown")
        assert match is None

    def test_method_filtering(self):
        """Test route matching respects HTTP method."""
        registry = RouteRegistry()
        registry.register(Route("/api/data", "DataHandler", method="GET"))
        registry.register(Route("/api/data", "DataHandler", method="POST"))

        get_match = registry.match("/api/data", "GET")
        assert get_match is not None

        post_match = registry.match("/api/data", "POST")
        assert post_match is not None

        delete_match = registry.match("/api/data", "DELETE")
        assert delete_match is None

    def test_exact_match_before_pattern(self):
        """Test exact routes are matched before patterns."""
        registry = RouteRegistry()
        registry.register(Route("/api/debates/{id}", "GenericHandler"))
        registry.register(Route("/api/debates/special", "SpecialHandler"))

        # Exact match should win
        match = registry.match("/api/debates/special")
        assert match.handler_name == "SpecialHandler"

        # Pattern matches other paths
        match = registry.match("/api/debates/other")
        assert match.handler_name == "GenericHandler"

    def test_priority_ordering(self):
        """Test higher priority routes matched first."""
        registry = RouteRegistry()
        registry.register(Route("/api/{path}", "CatchAll", priority=0))
        registry.register(Route("/api/{id}", "Specific", priority=10))

        match = registry.match("/api/test")
        assert match.handler_name == "Specific"

    def test_register_handler(self):
        """Test handler instance registration."""
        registry = RouteRegistry()

        class MockHandler:
            pass

        handler = MockHandler()
        registry.register_handler("MockHandler", handler)

        assert registry.get_handler("MockHandler") is handler
        assert registry.get_handler("Unknown") is None

    def test_get_routes(self):
        """Test getting all routes."""
        registry = RouteRegistry()
        registry.register(
            Route("/api/a", "A", method="GET"),
            Route("/api/b", "B", method="POST"),
        )

        all_routes = registry.get_routes()
        assert len(all_routes) == 2

        get_routes = registry.get_routes("GET")
        assert len(get_routes) == 1

    def test_get_stats(self):
        """Test registry statistics."""
        registry = RouteRegistry()
        registry.register(
            Route("/api/health", "System"),
            Route("/api/debates/{id}", "Debates"),
        )
        registry.register_handler("System", object())

        stats = registry.get_stats()
        assert stats["total_routes"] == 2
        assert stats["exact_routes"] == 1
        assert stats["pattern_routes"] == 1
        assert "System" in stats["handlers"]


class TestApiRoute:
    """Test api_route convenience function."""

    def test_simple_route(self):
        """Test creating simple route."""
        route = api_route("/api/health", "SystemHandler")
        assert route.pattern == "/api/health"
        assert route.handler_name == "SystemHandler"
        assert route.method == "GET"

    def test_route_with_params(self):
        """Test creating route with parameter specs."""
        route = api_route(
            "/api/agent/{name}/profile",
            "AgentsHandler",
            name={"pattern": "name", "type": str},
        )
        assert "name" in route.params
        assert route.params["name"].pattern == "name"

    def test_route_with_method(self):
        """Test creating route with custom method."""
        route = api_route("/api/data", "DataHandler", method="POST")
        assert route.method == "POST"


class TestDefaultRoutes:
    """Test default route creation."""

    def test_creates_routes(self):
        """Test default routes are created."""
        routes = create_default_routes()
        assert len(routes) > 0

    def test_system_routes_included(self):
        """Test system routes are included."""
        routes = create_default_routes()
        patterns = [r.pattern for r in routes]
        assert "/api/health" in patterns
        assert "/api/modes" in patterns

    def test_agent_routes_included(self):
        """Test agent routes are included."""
        routes = create_default_routes()
        patterns = [r.pattern for r in routes]
        assert "/api/leaderboard" in patterns
        assert "/api/agent/{name}/profile" in patterns

    def test_debate_routes_included(self):
        """Test debate routes are included."""
        routes = create_default_routes()
        patterns = [r.pattern for r in routes]
        assert "/api/debates" in patterns
        assert "/api/debates/{id}" in patterns


class TestGlobalRegistry:
    """Test global registry management."""

    def test_get_registry_singleton(self):
        """Test get_registry returns singleton."""
        set_registry(None)  # Reset
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_set_registry(self):
        """Test setting custom registry."""
        custom = RouteRegistry()
        set_registry(custom)
        assert get_registry() is custom

    def test_default_registry_has_routes(self):
        """Test default registry is populated."""
        set_registry(None)  # Reset
        registry = get_registry()
        assert len(registry.get_routes()) > 0


class TestParamPatterns:
    """Test predefined parameter patterns."""

    def test_id_pattern(self):
        """Test id pattern."""
        pattern = PARAM_PATTERNS["id"]
        assert pattern.match("valid_id-123")
        assert pattern.match("a" * 64)  # Max length
        assert not pattern.match("a" * 65)  # Too long
        assert not pattern.match("invalid/id")

    def test_name_pattern(self):
        """Test name pattern."""
        pattern = PARAM_PATTERNS["name"]
        assert pattern.match("claude")
        assert pattern.match("gpt-4")
        assert pattern.match("a" * 32)  # Max length
        assert not pattern.match("a" * 33)  # Too long

    def test_slug_pattern(self):
        """Test slug pattern."""
        pattern = PARAM_PATTERNS["slug"]
        assert pattern.match("my-debate-slug_123")
        assert pattern.match("a" * 128)  # Max length
        assert not pattern.match("a" * 129)  # Too long

    def test_domain_pattern(self):
        """Test domain pattern."""
        pattern = PARAM_PATTERNS["domain"]
        assert pattern.match("tech")
        assert pattern.match("science-ai")
        assert pattern.match("a" * 50)  # Max length
        assert not pattern.match("a" * 51)  # Too long


class TestRouteMatch:
    """Test RouteMatch dataclass."""

    def test_match_properties(self):
        """Test match has expected properties."""
        route = Route("/api/test", "TestHandler")
        match = RouteMatch(
            route=route,
            path="/api/test",
            method="GET",
            params={},
            handler_name="TestHandler",
        )
        assert match.matched is True
        assert match.path == "/api/test"
        assert match.handler_name == "TestHandler"
