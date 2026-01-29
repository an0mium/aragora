"""
OAuth data models and utility functions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OAuthUserInfo:
    """User info from OAuth provider."""

    provider: str
    provider_user_id: str
    email: str
    name: str
    picture: str | None = None
    email_verified: bool = False


def _get_param(query_params: dict, name: str, default: str = None) -> str:
    """
    Safely extract a query parameter value.

    Handler registry converts single-element lists to scalars, so we need to
    handle both list and string formats.

    Args:
        query_params: Dict of query parameters
        name: Parameter name to extract
        default: Default value if not found

    Returns:
        Parameter value as string, or default if not found
    """
    value = query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value
