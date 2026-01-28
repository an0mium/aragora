"""
AccountingConnectorBase - Base class for accounting and financial connectors.

Provides common patterns for:
- OAuth authentication flows
- Token refresh and credential management
- HTTP request handling with retry and rate limiting
- Standard CRUD operations
- Pagination support
- Error handling

Usage:
    class QuickBooksConnector(AccountingConnectorBase):
        PROVIDER_NAME = "QuickBooks"
        BASE_URL = "https://quickbooks.api.intuit.com"

        async def list_invoices(self, start_date=None, end_date=None):
            return await self._list_entities("Invoice", filters={
                "TxnDate": {"$gt": start_date, "$lt": end_date}
            })
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorAuthError,
    ConnectorRateLimitError,
    ConnectorTimeoutError,
)

logger = logging.getLogger(__name__)

# Type variable for credentials
C = TypeVar("C")


@dataclass
class RetryConfig:
    """Configuration for request retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    retryable_statuses: tuple = (429, 500, 502, 503, 504)
    retry_on_connection_error: bool = True


@dataclass
class PaginationConfig:
    """Configuration for pagination."""

    page_size: int = 100
    max_pages: Optional[int] = None
    style: str = "offset"  # "offset", "cursor", "page"


class AccountingConnectorBase(ABC, Generic[C]):
    """
    Base class for accounting and financial connectors.

    Provides common infrastructure for:
    - OAuth authentication and token management
    - HTTP request handling with retry logic
    - Rate limit handling with backoff
    - Standard CRUD operation patterns
    - Pagination support

    Subclasses must implement:
    - is_configured: property to check configuration
    - _get_auth_headers(): returns headers for authenticated requests
    - _parse_response(): parses API response to standard format
    """

    # Provider identification - override in subclasses
    PROVIDER_NAME: str = "Unknown"
    BASE_URL: str = ""

    # Default retry configuration
    DEFAULT_RETRY_CONFIG = RetryConfig()

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        environment: str = "sandbox",
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the accounting connector.

        Args:
            client_id: OAuth client ID (or API key)
            client_secret: OAuth client secret
            environment: "sandbox" or "production"
            timeout: Request timeout in seconds
            retry_config: Custom retry configuration
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.environment = environment
        self.timeout = timeout
        self.retry_config = retry_config or self.DEFAULT_RETRY_CONFIG

        self._credentials: Optional[C] = None
        self._http_client: Optional[Any] = None

    # =========================================================================
    # Configuration & Authentication (abstract/template methods)
    # =========================================================================

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the connector has required configuration."""
        ...

    @property
    def is_authenticated(self) -> bool:
        """Check if the connector has valid credentials."""
        return self._credentials is not None

    def set_credentials(self, credentials: C) -> None:
        """Set the credentials for authenticated requests."""
        self._credentials = credentials
        logger.info(f"[{self.PROVIDER_NAME}] Credentials set")

    def get_credentials(self) -> Optional[C]:
        """Get the current credentials."""
        return self._credentials

    @abstractmethod
    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Get the OAuth authorization URL.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            The authorization URL to redirect users to
        """
        ...

    @abstractmethod
    async def exchange_code(self, code: str, **kwargs) -> C:
        """
        Exchange an authorization code for credentials.

        Args:
            code: The authorization code from OAuth callback
            **kwargs: Additional parameters (redirect_uri, etc.)

        Returns:
            Credentials object
        """
        ...

    async def refresh_tokens(self) -> C:
        """
        Refresh the authentication tokens.

        Returns:
            Updated credentials

        Raises:
            ConnectorAuthError: If refresh fails
        """
        raise NotImplementedError(f"{self.PROVIDER_NAME} does not support token refresh")

    # =========================================================================
    # HTTP Request Handling
    # =========================================================================

    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        ...

    async def _get_http_client(self) -> Any:
        """Get or create the HTTP client."""
        if self._http_client is None:
            try:
                import httpx

                self._http_client = httpx.AsyncClient(timeout=self.timeout)
            except ImportError:
                import aiohttp

                self._http_client = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
        return self._http_client

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> Dict[str, Any]:
        """
        Make an authenticated HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            endpoint: API endpoint (relative to BASE_URL)
            data: Request body data
            params: Query parameters
            headers: Additional headers
            retry_config: Custom retry configuration

        Returns:
            Parsed JSON response

        Raises:
            ConnectorAPIError: On API errors
            ConnectorAuthError: On authentication errors
            ConnectorRateLimitError: On rate limit exceeded (after retries)
            ConnectorTimeoutError: On timeout
        """
        config = retry_config or self.retry_config
        url = f"{self.BASE_URL}{endpoint}"

        # Build headers
        request_headers = self._get_auth_headers()
        if headers:
            request_headers.update(headers)

        last_error: Optional[Exception] = None

        for attempt in range(config.max_retries + 1):
            try:
                response = await self._make_request(method, url, data, params, request_headers)

                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < config.max_retries:
                        delay = self._get_retry_delay(response, attempt, config)
                        logger.warning(f"[{self.PROVIDER_NAME}] Rate limited, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise ConnectorRateLimitError(
                            f"Rate limit exceeded after {config.max_retries} retries"
                        )

                # Handle retryable server errors
                if response.status_code in config.retryable_statuses:
                    if attempt < config.max_retries:
                        delay = self._get_retry_delay(response, attempt, config)
                        logger.warning(
                            f"[{self.PROVIDER_NAME}] Server error {response.status_code}, "
                            f"retrying in {delay}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                # Handle auth errors
                if response.status_code in (401, 403):
                    # Try to refresh tokens
                    if self._credentials and hasattr(self._credentials, "refresh_token"):
                        try:
                            await self.refresh_tokens()
                            request_headers = self._get_auth_headers()
                            continue
                        except Exception:
                            pass
                    raise ConnectorAuthError(f"Authentication failed: {response.status_code}")

                # Handle other client errors
                if 400 <= response.status_code < 500:
                    error_detail = await self._parse_error(response)
                    raise ConnectorAPIError(f"API error {response.status_code}: {error_detail}")

                # Handle server errors (non-retryable after exhausting retries)
                if response.status_code >= 500:
                    error_detail = await self._parse_error(response)
                    raise ConnectorAPIError(f"Server error {response.status_code}: {error_detail}")

                # Success - parse and return response
                return await self._parse_response(response)

            except (asyncio.TimeoutError, TimeoutError) as e:
                last_error = e
                if attempt < config.max_retries:
                    delay = self._get_retry_delay(None, attempt, config)
                    logger.warning(f"[{self.PROVIDER_NAME}] Timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                raise ConnectorTimeoutError(f"Request timed out: {endpoint}") from e

            except (ConnectionError, OSError) as e:
                last_error = e
                if config.retry_on_connection_error and attempt < config.max_retries:
                    delay = self._get_retry_delay(None, attempt, config)
                    logger.warning(f"[{self.PROVIDER_NAME}] Connection error, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                raise ConnectorAPIError(f"Connection error: {e}") from e

        # Should not reach here, but just in case
        if last_error:
            raise ConnectorAPIError(f"Request failed: {last_error}") from last_error
        raise ConnectorAPIError("Request failed for unknown reason")

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> Any:
        """Make the actual HTTP request (override for custom clients)."""
        client = await self._get_http_client()

        # httpx-style client
        if hasattr(client, "request"):
            return await client.request(
                method,
                url,
                json=data,
                params=params,
                headers=headers,
            )

        # aiohttp-style client
        async with client.request(
            method,
            url,
            json=data,
            params=params,
            headers=headers,
        ) as response:
            return response

    def _get_retry_delay(
        self,
        response: Optional[Any],
        attempt: int,
        config: RetryConfig,
    ) -> float:
        """Calculate retry delay with exponential backoff."""
        # Check for Retry-After header
        if response is not None:
            retry_after = None
            if hasattr(response, "headers"):
                retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), config.max_delay)
                except ValueError:
                    pass

        # Exponential backoff
        delay = config.base_delay * (2**attempt)
        return min(delay, config.max_delay)

    async def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse the response body as JSON."""
        if hasattr(response, "json"):
            # httpx
            if asyncio.iscoroutinefunction(response.json):
                return await response.json()
            return response.json()
        # aiohttp
        return await response.json()

    async def _parse_error(self, response: Any) -> str:
        """Parse error details from response."""
        try:
            data = await self._parse_response(response)
            if isinstance(data, dict):
                return data.get("error") or data.get("message") or str(data)
            return str(data)
        except Exception:
            if hasattr(response, "text"):
                if asyncio.iscoroutinefunction(response.text):
                    return await response.text()
                return response.text
            return f"HTTP {response.status_code}"

    # =========================================================================
    # Standard CRUD Operations (template methods)
    # =========================================================================

    async def _list_entities(
        self,
        entity_type: str,
        endpoint: str,
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[PaginationConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        List entities with pagination support.

        Args:
            entity_type: Type of entity for logging
            endpoint: API endpoint
            filters: Query filters
            pagination: Pagination configuration

        Returns:
            List of entity dictionaries
        """
        pagination = pagination or PaginationConfig()
        results: List[Dict[str, Any]] = []
        page = 0
        offset = 0

        while True:
            # Build pagination params
            params = dict(filters or {})
            if pagination.style == "offset":
                params["limit"] = pagination.page_size
                params["offset"] = offset
            elif pagination.style == "page":
                params["page"] = page + 1
                params["per_page"] = pagination.page_size

            response = await self._request("GET", endpoint, params=params)
            page_results = self._extract_entities(response, entity_type)

            if not page_results:
                break

            results.extend(page_results)
            page += 1
            offset += len(page_results)

            # Check termination conditions
            if len(page_results) < pagination.page_size:
                break
            if pagination.max_pages and page >= pagination.max_pages:
                break

        logger.debug(f"[{self.PROVIDER_NAME}] Listed {len(results)} {entity_type}(s)")
        return results

    def _extract_entities(self, response: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
        """
        Extract entity list from API response.

        Override in subclasses for provider-specific response formats.
        """
        # Common patterns
        if entity_type.lower() in response:
            return response[entity_type.lower()]
        if f"{entity_type}s" in response:
            return response[f"{entity_type}s"]
        if "data" in response:
            return response["data"]
        if "items" in response:
            return response["items"]
        if "results" in response:
            return response["results"]
        if isinstance(response, list):
            return response
        return []

    async def _get_entity(
        self,
        entity_type: str,
        endpoint: str,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single entity by ID.

        Args:
            entity_type: Type of entity for logging
            endpoint: API endpoint (should include ID placeholder or be complete)
            entity_id: Entity ID

        Returns:
            Entity dictionary or None if not found
        """
        try:
            response = await self._request("GET", endpoint)
            return self._extract_single_entity(response, entity_type)
        except ConnectorAPIError as e:
            if "404" in str(e):
                return None
            raise

    def _extract_single_entity(
        self, response: Dict[str, Any], entity_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract single entity from response (override for custom formats)."""
        if entity_type in response:
            return response[entity_type]
        if "data" in response:
            return response["data"]
        return response

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._http_client:
            if hasattr(self._http_client, "aclose"):
                await self._http_client.aclose()
            elif hasattr(self._http_client, "close"):
                await self._http_client.close()
            self._http_client = None

    async def __aenter__(self) -> "AccountingConnectorBase":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Stats & Debugging
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics (override for additional stats)."""
        return {
            "provider": self.PROVIDER_NAME,
            "environment": self.environment,
            "is_configured": self.is_configured,
            "is_authenticated": self.is_authenticated,
            "base_url": self.BASE_URL,
        }


__all__ = [
    "AccountingConnectorBase",
    "RetryConfig",
    "PaginationConfig",
]
