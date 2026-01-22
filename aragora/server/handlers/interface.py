"""
Handler Interface Definitions.

Defines the contracts that handlers must implement. Using Protocol classes
allows handlers to be imported and used without requiring the full server
infrastructure, reducing coupling.

This module can be imported independently of the server:
    from aragora.server.handlers.interface import HandlerInterface
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    runtime_checkable,
)

if TYPE_CHECKING:
    from aragora.billing.auth.context import UserAuthContext


# =============================================================================
# Response Types
# =============================================================================


class HandlerResult(TypedDict, total=False):
    """Standard result type returned by handlers.

    All handlers should return this type or None.
    This is a subset of the full HandlerResult in utils.responses,
    kept here for interface documentation.
    """

    body: bytes
    content_type: str
    status: int
    headers: Dict[str, str]


# =============================================================================
# Handler Protocol
# =============================================================================


@runtime_checkable
class HandlerInterface(Protocol):
    """Protocol defining the handler contract.

    All endpoint handlers must implement this interface.
    This allows handlers to be type-checked without importing
    the full BaseHandler implementation.

    Example:
        def register_handler(handler: HandlerInterface) -> None:
            # Works with any handler implementing the interface
            result = handler.handle("/api/v1/test", {}, mock_handler)
    """

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle a GET request.

        Args:
            path: The request path (e.g., "/api/v1/debates/123")
            query_params: Parsed query parameters as dict
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        ...

    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle a POST request.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        ...

    def handle_delete(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle a DELETE request."""
        ...

    def handle_patch(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle a PATCH request."""
        ...

    def handle_put(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle a PUT request."""
        ...


@runtime_checkable
class AuthenticatedHandlerInterface(Protocol):
    """Protocol for handlers that require authentication.

    Handlers implementing this interface provide authentication utilities.
    """

    def get_current_user(self, handler: Any) -> Optional["UserAuthContext"]:
        """Get authenticated user from request, if any.

        Args:
            handler: HTTP request handler with headers

        Returns:
            UserAuthContext if authenticated, None otherwise
        """
        ...

    def require_auth_or_error(
        self, handler: Any
    ) -> Tuple[Optional["UserAuthContext"], Optional[HandlerResult]]:
        """Require authentication and return user or error response.

        Args:
            handler: HTTP request handler with headers

        Returns:
            Tuple of (UserAuthContext, None) if authenticated,
            or (None, HandlerResult) with 401 error if not
        """
        ...


@runtime_checkable
class PaginatedHandlerInterface(Protocol):
    """Protocol for handlers that support pagination."""

    def get_pagination(
        self,
        query_params: Dict[str, Any],
        default_limit: Optional[int] = None,
        max_limit: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Extract and validate pagination parameters.

        Args:
            query_params: Query parameters dict
            default_limit: Override default limit
            max_limit: Override max limit

        Returns:
            Tuple of (limit, offset) with validated bounds
        """
        ...

    def paginated_response(
        self,
        items: list,
        total: int,
        limit: int,
        offset: int,
        items_key: str = "items",
    ) -> HandlerResult:
        """Create a standardized paginated response.

        Args:
            items: List of items for this page
            total: Total count of all items
            limit: Page size used
            offset: Starting offset
            items_key: Key name for items in response

        Returns:
            JSON response with pagination metadata
        """
        ...


@runtime_checkable
class CachedHandlerInterface(Protocol):
    """Protocol for handlers that support caching."""

    def cached_response(
        self,
        cache_key: str,
        ttl_seconds: float,
        generator: Any,
    ) -> Any:
        """Get or generate a cached response.

        Args:
            cache_key: Unique key for this cached item
            ttl_seconds: How long to cache the result
            generator: Callable that generates the value if not cached

        Returns:
            Cached or freshly generated value
        """
        ...


# =============================================================================
# Server Context Interface
# =============================================================================


class MinimalServerContext(TypedDict, total=False):
    """Minimal server context for handler initialization.

    This is a subset of the full ServerContext that defines only
    the most commonly needed resources. Handlers should use
    ctx.get("key") to safely access optional fields.
    """

    # Core resources most handlers need
    storage: Any  # DebateStorage
    user_store: Any  # UserStore
    elo_system: Any  # EloSystem


class StorageAccessInterface(Protocol):
    """Protocol for accessing storage resources.

    Handlers that need storage access should implement this.
    """

    def get_storage(self) -> Optional[Any]:
        """Get debate storage instance."""
        ...

    def get_elo_system(self) -> Optional[Any]:
        """Get ELO system instance."""
        ...


# =============================================================================
# Handler Registration Types
# =============================================================================


class RouteConfig(TypedDict, total=False):
    """Configuration for a registered route.

    Used by handler registration systems.
    """

    path_pattern: str
    methods: list  # List of HTTP methods
    handler_class: type
    requires_auth: bool
    rate_limit: Optional[int]


class HandlerRegistration(TypedDict):
    """Handler registration entry.

    Used by lazy handler registration.
    """

    handler_class: type
    routes: list  # List of RouteConfig
    lazy: bool  # If True, handler is instantiated on first use


# =============================================================================
# Factory Functions
# =============================================================================


def is_handler(obj: Any) -> bool:
    """Check if an object implements the HandlerInterface.

    Args:
        obj: Object to check

    Returns:
        True if obj implements HandlerInterface
    """
    return isinstance(obj, HandlerInterface)


def is_authenticated_handler(obj: Any) -> bool:
    """Check if an object implements AuthenticatedHandlerInterface.

    Args:
        obj: Object to check

    Returns:
        True if obj implements AuthenticatedHandlerInterface
    """
    return isinstance(obj, AuthenticatedHandlerInterface)


__all__ = [
    # Result types
    "HandlerResult",
    # Protocols
    "HandlerInterface",
    "AuthenticatedHandlerInterface",
    "PaginatedHandlerInterface",
    "CachedHandlerInterface",
    "StorageAccessInterface",
    # Context types
    "MinimalServerContext",
    # Registration types
    "RouteConfig",
    "HandlerRegistration",
    # Factory functions
    "is_handler",
    "is_authenticated_handler",
]
