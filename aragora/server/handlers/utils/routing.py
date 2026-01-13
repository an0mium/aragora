"""
URL routing utilities for handler dispatch.

Provides pattern matching and dispatch for mapping URL paths to handler methods.
"""

from typing import Any, Callable


class PathMatcher:
    """Utility for matching URL paths against patterns.

    Simplifies the common pattern of parsing path segments and dispatching
    to handler methods.

    Example:
        matcher = PathMatcher("/api/agent/{name}/{action}")
        result = matcher.match("/api/agent/claude/profile")
        # result = {"name": "claude", "action": "profile"}

        matcher = PathMatcher("/api/debates")
        result = matcher.match("/api/debates")
        # result = {}  (empty dict = matched)

        result = matcher.match("/api/other")
        # result = None  (None = no match)
    """

    def __init__(self, pattern: str):
        """Initialize with a URL pattern.

        Args:
            pattern: URL pattern with {param} placeholders for path segments
        """
        self.pattern = pattern
        self.parts = pattern.strip("/").split("/")
        self.param_indices: dict[str, int] = {}

        for i, part in enumerate(self.parts):
            if part.startswith("{") and part.endswith("}"):
                param_name = part[1:-1]
                self.param_indices[param_name] = i

    def match(self, path: str) -> dict | None:
        """Match a path against this pattern.

        Returns:
            Dict of extracted parameters if matched, None otherwise
        """
        path_parts = path.strip("/").split("/")

        if len(path_parts) != len(self.parts):
            return None

        params = {}
        for i, (pattern_part, path_part) in enumerate(zip(self.parts, path_parts)):
            if pattern_part.startswith("{") and pattern_part.endswith("}"):
                param_name = pattern_part[1:-1]
                params[param_name] = path_part
            elif pattern_part != path_part:
                return None

        return params

    def matches(self, path: str) -> bool:
        """Check if a path matches this pattern."""
        return self.match(path) is not None


class RouteDispatcher:
    """Dispatcher for routing paths to handler methods.

    Simplifies the common pattern of if/elif chains in handle() methods.
    Uses segment-count indexing for O(n/k) lookup instead of O(n).

    Example:
        dispatcher = RouteDispatcher()
        dispatcher.add_route("/api/agents", self._list_agents)
        dispatcher.add_route("/api/agent/{name}/profile", self._get_profile)
        dispatcher.add_route("/api/agent/{name}/history", self._get_history)

        # In handle() method:
        result = dispatcher.dispatch(path, query_params)
        if result is not None:
            return result
    """

    def __init__(self):
        self.routes: list[tuple[PathMatcher, Callable]] = []
        # Index routes by segment count for faster lookup
        self._segment_index: dict[int, list[int]] = {}

    def add_route(self, pattern: str, handler: Callable) -> "RouteDispatcher":
        """Add a route pattern with its handler.

        Args:
            pattern: URL pattern with {param} placeholders
            handler: Callable that receives (params_dict, query_params)
                     or just () if no path params

        Returns:
            Self for chaining
        """
        matcher = PathMatcher(pattern)
        route_idx = len(self.routes)
        self.routes.append((matcher, handler))

        # Index by segment count
        segment_count = len(matcher.parts)
        if segment_count not in self._segment_index:
            self._segment_index[segment_count] = []
        self._segment_index[segment_count].append(route_idx)

        return self

    def dispatch(self, path: str, query_params: dict | None = None) -> Any:
        """Dispatch a path to its handler.

        Args:
            path: URL path to dispatch
            query_params: Query parameters dict

        Returns:
            Handler result if matched, None otherwise
        """
        query_params = query_params or {}

        # Count path segments once
        path_segments = len(path.strip("/").split("/"))

        # Only check routes with matching segment count
        route_indices = self._segment_index.get(path_segments, [])
        for idx in route_indices:
            matcher, handler = self.routes[idx]
            params = matcher.match(path)
            if params is not None:
                # Call handler with path params and query params
                if params:
                    return handler(params, query_params)
                else:
                    return handler(query_params)

        return None

    def can_handle(self, path: str) -> bool:
        """Check if any route can handle this path."""
        path_segments = len(path.strip("/").split("/"))
        route_indices = self._segment_index.get(path_segments, [])
        return any(self.routes[idx][0].matches(path) for idx in route_indices)
