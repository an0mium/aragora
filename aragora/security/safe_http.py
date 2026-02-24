"""Safe HTTP wrappers with built-in SSRF protection.

All outbound HTTP requests that touch user-influenced URLs should use these
wrappers instead of raw httpx calls. Each function validates the target URL
against the SSRF protection rules before making the request.

Usage::

    from aragora.security.safe_http import safe_get, safe_post, async_safe_get

    # Synchronous
    response = safe_get("https://api.example.com/data", timeout=10.0)

    # Asynchronous
    response = await async_safe_get("https://api.example.com/data")

    # With SSRF options
    response = await async_safe_request(
        "GET", url, resolve_dns=True, allowed_domains={"api.example.com"},
    )
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from aragora.security.ssrf_protection import (
    SSRFValidationError,
    validate_url,
)

logger = logging.getLogger(__name__)

__all__ = [
    "safe_get",
    "safe_post",
    "safe_request",
    "async_safe_get",
    "async_safe_post",
    "async_safe_request",
    "SSRFBlockedError",
]


class SSRFBlockedError(SSRFValidationError):
    """Raised when a request is blocked by SSRF validation."""


def _validate(
    url: str,
    *,
    resolve_dns: bool = False,
    allowed_domains: set[str] | frozenset[str] | None = None,
) -> None:
    """Validate URL and raise SSRFBlockedError if unsafe."""
    result = validate_url(url, resolve_dns=resolve_dns, allowed_domains=allowed_domains)
    if not result.is_safe:
        logger.warning("SSRF blocked: url=%s error=%s", url, result.error)
        raise SSRFBlockedError(result.error, url=url)


# ---------------------------------------------------------------------------
# Synchronous wrappers
# ---------------------------------------------------------------------------


def safe_get(
    url: str,
    *,
    resolve_dns: bool = False,
    allowed_domains: set[str] | frozenset[str] | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """SSRF-safe synchronous GET request."""
    _validate(url, resolve_dns=resolve_dns, allowed_domains=allowed_domains)
    return httpx.get(url, **kwargs)


def safe_post(
    url: str,
    *,
    resolve_dns: bool = False,
    allowed_domains: set[str] | frozenset[str] | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """SSRF-safe synchronous POST request."""
    _validate(url, resolve_dns=resolve_dns, allowed_domains=allowed_domains)
    return httpx.post(url, **kwargs)


def safe_request(
    method: str,
    url: str,
    *,
    resolve_dns: bool = False,
    allowed_domains: set[str] | frozenset[str] | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """SSRF-safe synchronous request with arbitrary method."""
    _validate(url, resolve_dns=resolve_dns, allowed_domains=allowed_domains)
    return httpx.request(method, url, **kwargs)


# ---------------------------------------------------------------------------
# Asynchronous wrappers
# ---------------------------------------------------------------------------


async def async_safe_get(
    url: str,
    *,
    resolve_dns: bool = False,
    allowed_domains: set[str] | frozenset[str] | None = None,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """SSRF-safe asynchronous GET request.

    If *client* is provided, uses that client. Otherwise creates a
    one-shot ``AsyncClient``.
    """
    _validate(url, resolve_dns=resolve_dns, allowed_domains=allowed_domains)
    if client is not None:
        return await client.get(url, **kwargs)
    async with httpx.AsyncClient() as _client:
        return await _client.get(url, **kwargs)


async def async_safe_post(
    url: str,
    *,
    resolve_dns: bool = False,
    allowed_domains: set[str] | frozenset[str] | None = None,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """SSRF-safe asynchronous POST request."""
    _validate(url, resolve_dns=resolve_dns, allowed_domains=allowed_domains)
    if client is not None:
        return await client.post(url, **kwargs)
    async with httpx.AsyncClient() as _client:
        return await _client.post(url, **kwargs)


async def async_safe_request(
    method: str,
    url: str,
    *,
    resolve_dns: bool = False,
    allowed_domains: set[str] | frozenset[str] | None = None,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """SSRF-safe asynchronous request with arbitrary method."""
    _validate(url, resolve_dns=resolve_dns, allowed_domains=allowed_domains)
    if client is not None:
        return await client.request(method, url, **kwargs)
    async with httpx.AsyncClient() as _client:
        return await _client.request(method, url, **kwargs)
