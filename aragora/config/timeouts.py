"""
Centralized Timeout Configuration.

Provides a single source of truth for all timeout values used throughout
the codebase. Using this module instead of hardcoded values ensures
consistent behavior and makes timeout tuning easier.

Usage:
    from aragora.config.timeouts import Timeouts

    # Use predefined timeouts
    response = await client.get(url, timeout=Timeouts.HTTP_DEFAULT)

    # Or use the category-specific values
    async with httpx.AsyncClient(timeout=Timeouts.CONNECTOR_API) as client:
        ...
"""

from __future__ import annotations

from aragora.config.env_helpers import env_float as _env_float


class Timeouts:
    """
    Centralized timeout configuration.

    All timeouts are in seconds unless otherwise noted.
    Environment variables can override defaults (prefix: ARAGORA_TIMEOUT_).

    Categories:
        - HTTP: General HTTP request timeouts
        - Database: Database connection/query timeouts
        - Debate: Debate orchestration timeouts
        - Connector: External API connector timeouts
        - WebSocket: WebSocket connection timeouts

    Example:
        # Override connector timeout via environment
        export ARAGORA_TIMEOUT_CONNECTOR_API=60

        # Then in code:
        timeout = Timeouts.CONNECTOR_API  # Returns 60.0
    """

    # ==========================================================================
    # HTTP Timeouts
    # ==========================================================================

    HTTP_SHORT: float = _env_float("ARAGORA_TIMEOUT_HTTP_SHORT", 5.0)
    """Short HTTP timeout for quick health checks and pings."""

    HTTP_DEFAULT: float = _env_float("ARAGORA_TIMEOUT_HTTP_DEFAULT", 30.0)
    """Default HTTP timeout for standard API requests."""

    HTTP_LONG: float = _env_float("ARAGORA_TIMEOUT_HTTP_LONG", 120.0)
    """Long HTTP timeout for file uploads and slow operations."""

    # ==========================================================================
    # Database Timeouts
    # ==========================================================================

    DATABASE_CONNECTION: float = _env_float("ARAGORA_TIMEOUT_DB_CONNECTION", 10.0)
    """Database connection establishment timeout."""

    DATABASE_QUERY: float = _env_float("ARAGORA_TIMEOUT_DB_QUERY", 30.0)
    """Default database query timeout."""

    DATABASE_MIGRATION: float = _env_float("ARAGORA_TIMEOUT_DB_MIGRATION", 180.0)
    """Database migration/DDL operation timeout."""

    # ==========================================================================
    # Debate Timeouts
    # ==========================================================================

    DEBATE_TOTAL: float = _env_float("ARAGORA_TIMEOUT_DEBATE_TOTAL", 900.0)
    """Maximum total debate duration (15 minutes)."""

    DEBATE_AGENT_CALL: float = _env_float("ARAGORA_TIMEOUT_DEBATE_AGENT", 240.0)
    """Maximum time for a single agent call (4 minutes)."""

    DEBATE_ROUND: float = _env_float("ARAGORA_TIMEOUT_DEBATE_ROUND", 300.0)
    """Maximum time for a single debate round (5 minutes)."""

    DEBATE_CONSENSUS: float = _env_float("ARAGORA_TIMEOUT_DEBATE_CONSENSUS", 60.0)
    """Timeout for consensus checking operations."""

    # ==========================================================================
    # Connector Timeouts
    # ==========================================================================

    CONNECTOR_API: float = _env_float("ARAGORA_TIMEOUT_CONNECTOR_API", 30.0)
    """Standard timeout for external API connectors."""

    CONNECTOR_AUTH: float = _env_float("ARAGORA_TIMEOUT_CONNECTOR_AUTH", 10.0)
    """Timeout for authentication requests."""

    CONNECTOR_SEARCH: float = _env_float("ARAGORA_TIMEOUT_CONNECTOR_SEARCH", 60.0)
    """Timeout for search/query operations (may involve pagination)."""

    CONNECTOR_UPLOAD: float = _env_float("ARAGORA_TIMEOUT_CONNECTOR_UPLOAD", 600.0)
    """Timeout for file uploads (10 minutes for large files)."""

    CONNECTOR_DOWNLOAD: float = _env_float("ARAGORA_TIMEOUT_CONNECTOR_DOWNLOAD", 120.0)
    """Timeout for file downloads."""

    # ==========================================================================
    # WebSocket Timeouts
    # ==========================================================================

    WEBSOCKET_CONNECT: float = _env_float("ARAGORA_TIMEOUT_WS_CONNECT", 30.0)
    """WebSocket connection establishment timeout."""

    WEBSOCKET_MESSAGE: float = _env_float("ARAGORA_TIMEOUT_WS_MESSAGE", 60.0)
    """Timeout waiting for WebSocket message."""

    WEBSOCKET_HEARTBEAT: float = _env_float("ARAGORA_TIMEOUT_WS_HEARTBEAT", 30.0)
    """WebSocket heartbeat interval."""

    # ==========================================================================
    # Process Timeouts
    # ==========================================================================

    PROCESS_SHORT: float = _env_float("ARAGORA_TIMEOUT_PROCESS_SHORT", 60.0)
    """Timeout for short subprocess operations."""

    PROCESS_LONG: float = _env_float("ARAGORA_TIMEOUT_PROCESS_LONG", 300.0)
    """Timeout for long subprocess operations (builds, tests)."""

    # ==========================================================================
    # Aliases for common use cases
    # ==========================================================================

    DEFAULT = HTTP_DEFAULT
    """Alias for HTTP_DEFAULT - common default timeout."""

    SHORT = HTTP_SHORT
    """Alias for HTTP_SHORT - quick operations."""

    LONG = HTTP_LONG
    """Alias for HTTP_LONG - slow operations."""

    @classmethod
    def get(cls, name: str, default: float = 30.0) -> float:
        """
        Get a timeout value by name.

        Args:
            name: Timeout name (e.g., "HTTP_DEFAULT", "CONNECTOR_API")
            default: Default value if name not found

        Returns:
            Timeout value in seconds
        """
        return getattr(cls, name.upper(), default)

    @classmethod
    def all(cls) -> dict[str, float]:
        """
        Get all timeout values as a dictionary.

        Returns:
            Dictionary mapping timeout names to values
        """
        return {
            name: value
            for name, value in vars(cls).items()
            if isinstance(value, float) and not name.startswith("_")
        }
