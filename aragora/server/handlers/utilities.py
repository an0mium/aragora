"""
Shared Handler Utilities.

Standalone utility functions that can be imported without requiring
the full server infrastructure. These utilities are commonly used
across multiple handlers.

This module consolidates functions that were previously scattered or
duplicated, providing a single import point.

Example:
    from aragora.server.handlers.utilities import (
        get_host_header,
        get_agent_name,
        agent_to_dict,
    )
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from aragora.protocols import AgentRating

# =============================================================================
# Environment Defaults
# =============================================================================

# Default host from environment (used when Host header is missing)
_DEFAULT_HOST = os.environ.get("ARAGORA_DEFAULT_HOST", "localhost:8080")


# =============================================================================
# Request Utilities
# =============================================================================


def get_host_header(
    handler: Optional[Any],
    default: Optional[str] = None,
) -> str:
    """Extract Host header from request handler.

    Consolidates the common pattern of extracting the Host header
    with proper fallback handling.

    Args:
        handler: HTTP request handler with headers attribute
        default: Default value if handler is None or Host header missing.
                 If None, uses ARAGORA_DEFAULT_HOST env var or 'localhost:8080'.

    Returns:
        Host header value or default

    Example:
        # Before (repeated 5+ times):
        host = handler.headers.get('Host', 'localhost:8080') if handler else 'localhost:8080'

        # After:
        host = get_host_header(handler)
    """
    if default is None:
        default = _DEFAULT_HOST
    if handler is None:
        return default
    return handler.headers.get("Host", default) if hasattr(handler, "headers") else default


def get_request_id(handler: Optional[Any]) -> Optional[str]:
    """Extract request/trace ID from request handler.

    Checks common header names for request tracing.

    Args:
        handler: HTTP request handler with headers

    Returns:
        Request ID if found, None otherwise
    """
    if handler is None or not hasattr(handler, "headers"):
        return None

    headers = handler.headers
    return (
        headers.get("X-Request-ID") or headers.get("X-Trace-ID") or headers.get("X-Correlation-ID")
    )


def get_content_length(handler: Any) -> int:
    """Extract Content-Length from request handler.

    Args:
        handler: HTTP request handler with headers

    Returns:
        Content length as int, 0 if not present or invalid
    """
    if not hasattr(handler, "headers"):
        return 0
    try:
        return int(handler.headers.get("Content-Length", 0))
    except (ValueError, TypeError):
        return 0


# =============================================================================
# Agent Utilities
# =============================================================================


def get_agent_name(agent: Union[dict, AgentRating, Any, None]) -> Optional[str]:
    """Extract agent name from dict or object.

    Handles the common pattern where agent data might be either
    a dict with 'name'/'agent_name' key or an object with name attribute.

    Args:
        agent: Dict or AgentRating-like object containing agent name

    Returns:
        Agent name string or None if not found

    Example:
        # Before (repeated 4+ times):
        name = agent.get("name") if isinstance(agent, dict) else getattr(agent, "name", None)

        # After:
        name = get_agent_name(agent)
    """
    if agent is None:
        return None
    if isinstance(agent, dict):
        return agent.get("agent_name") or agent.get("name")
    return getattr(agent, "agent_name", None) or getattr(agent, "name", None)


def agent_to_dict(
    agent: Union[dict, AgentRating, Any, None],
    include_name: bool = True,
) -> dict:
    """Convert agent object or dict to standardized dict with ELO fields.

    Handles the common pattern where agent data might be either a dict
    or an AgentRating object, extracting standard fields with safe defaults.

    Args:
        agent: Dict or object containing agent data
        include_name: Whether to include name/agent_name fields (default: True)

    Returns:
        Dict with standardized ELO-related fields

    Example:
        # Before (repeated 40+ times across handlers):
        agent_dict = {
            "name": getattr(agent, "name", "unknown"),
            "elo": getattr(agent, "elo", 1500),
            "wins": getattr(agent, "wins", 0),
            ...
        }

        # After:
        agent_dict = agent_to_dict(agent)
    """
    if agent is None:
        return {}

    if isinstance(agent, dict):
        return agent.copy()

    # Extract standard ELO fields from object
    name = get_agent_name(agent) or "unknown"
    result = {
        "elo": getattr(agent, "elo", 1500),
        "wins": getattr(agent, "wins", 0),
        "losses": getattr(agent, "losses", 0),
        "draws": getattr(agent, "draws", 0),
        "win_rate": getattr(agent, "win_rate", 0.0),
        "games": getattr(agent, "games_played", getattr(agent, "games", 0)),
        "matches": getattr(agent, "matches", 0),
    }

    if include_name:
        result["name"] = name
        result["agent_name"] = name

    return result


def normalize_agent_names(agents: list) -> list:
    """Normalize a list of agent names or objects to lowercase strings.

    Useful for consistent agent comparison and lookups.

    Args:
        agents: List of agent names (str) or agent objects

    Returns:
        List of lowercase agent name strings
    """
    result = []
    for agent in agents:
        name = get_agent_name(agent) if not isinstance(agent, str) else agent
        if name:
            result.append(name.lower())
    return result


# =============================================================================
# Path Utilities
# =============================================================================


def extract_path_segment(
    path: str,
    index: int,
    default: Optional[str] = None,
) -> Optional[str]:
    """Extract a segment from a URL path.

    Args:
        path: URL path (e.g., "/api/v1/debates/123/rounds")
        index: Zero-based index of segment to extract
        default: Default value if segment doesn't exist

    Returns:
        Path segment at index, or default if not found

    Example:
        path = "/api/v1/debates/123/rounds"
        extract_path_segment(path, 2)  # Returns "123"
        extract_path_segment(path, 5)  # Returns None
        extract_path_segment(path, 5, "default")  # Returns "default"
    """
    parts = path.strip("/").split("/")
    if index >= len(parts):
        return default
    segment = parts[index]
    return segment if segment else default


def build_api_url(
    *segments: str,
    query_params: Optional[dict] = None,
) -> str:
    """Build an API URL from path segments.

    Args:
        *segments: Path segments to join
        query_params: Optional query parameters to append

    Returns:
        URL string

    Example:
        build_api_url("api", "debates", "123")  # "/api/v1/debates/123"
        build_api_url("api", "agents", query_params={"limit": 10})  # "/api/v1/agents?limit=10"
    """
    path = "/" + "/".join(str(s).strip("/") for s in segments if s)

    if query_params:
        params = "&".join(f"{k}={v}" for k, v in query_params.items() if v is not None)
        if params:
            path = f"{path}?{params}"

    return path


# =============================================================================
# Content Type Utilities
# =============================================================================


def is_json_content_type(content_type: Optional[str]) -> bool:
    """Check if a content type is JSON.

    Args:
        content_type: Content-Type header value

    Returns:
        True if content type is JSON (application/json or text/json)
    """
    if not content_type:
        return False
    media_type = content_type.split(";")[0].strip().lower()
    return media_type in ("application/json", "text/json")


def get_media_type(content_type: Optional[str]) -> str:
    """Extract media type from Content-Type header.

    Strips charset and other parameters.

    Args:
        content_type: Content-Type header value

    Returns:
        Media type (e.g., "application/json"), or empty string if None
    """
    if not content_type:
        return ""
    return content_type.split(";")[0].strip().lower()


__all__ = [
    # Request utilities
    "get_host_header",
    "get_request_id",
    "get_content_length",
    # Agent utilities
    "get_agent_name",
    "agent_to_dict",
    "normalize_agent_names",
    # Path utilities
    "extract_path_segment",
    "build_api_url",
    # Content type utilities
    "is_json_content_type",
    "get_media_type",
]
