"""
Aragora API Client

Main HTTP client for interacting with the Aragora platform.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar
from urllib.parse import urljoin

import httpx

from .exceptions import (
    AragoraError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)

T = TypeVar("T")


class AragoraClient:
    """
    Synchronous Aragora API client.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="your-key")
        >>> debate = client.debates.create(task="Should we adopt microservices?")
        >>> print(debate.debate_id)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Aragora client.

        Args:
            base_url: Base URL of the Aragora API (e.g., "https://api.aragora.ai")
            api_key: API key for authentication (optional for some endpoints)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retries for failed requests (default: 3)
            retry_delay: Base delay between retries in seconds (default: 1.0)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = httpx.Client(
            timeout=timeout,
            headers=self._build_headers(),
        )

        # Initialize namespace APIs
        self._init_namespaces()

    def _build_headers(self) -> dict[str, str]:
        """Build default headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "aragora-python/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _init_namespaces(self) -> None:
        """Initialize namespace API objects."""
        from .namespaces.admin import AdminAPI
        from .namespaces.agents import AgentsAPI
        from .namespaces.analytics import AnalyticsAPI
        from .namespaces.debates import DebatesAPI
        from .namespaces.onboarding import OnboardingAPI
        from .namespaces.usage import UsageAPI
        from .namespaces.workflows import WorkflowsAPI

        self.admin = AdminAPI(self)
        self.agents = AgentsAPI(self)
        self.analytics = AnalyticsAPI(self)
        self.debates = DebatesAPI(self)
        self.onboarding = OnboardingAPI(self)
        self.usage = UsageAPI(self)
        self.workflows = WorkflowsAPI(self)

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """
        Make an HTTP request to the Aragora API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: API path (e.g., "/api/v1/debates")
            params: Query parameters
            json: JSON body for POST/PUT requests
            headers: Additional headers

        Returns:
            Parsed JSON response

        Raises:
            AragoraError: For API errors
        """
        url = urljoin(self.base_url, path)
        request_headers = {**self._build_headers(), **(headers or {})}

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=request_headers,
                )

                if response.is_success:
                    if response.content:
                        return response.json()
                    return None

                # Handle error responses
                self._handle_error_response(response)

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise AragoraError("Request timed out") from e

            except httpx.ConnectError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise AragoraError("Connection failed") from e

        if last_error:
            raise AragoraError(f"Request failed after {self.max_retries} retries") from last_error

        raise AragoraError("Unexpected error")

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses and raise appropriate exceptions."""
        try:
            body = response.json()
            message = body.get("error", body.get("message", response.text))
        except Exception:
            body = None
            message = response.text or "Unknown error"

        status = response.status_code

        if status == 401:
            raise AuthenticationError(message, response_body=body)
        elif status == 403:
            raise AuthorizationError(message, response_body=body)
        elif status == 404:
            raise NotFoundError(message, response_body=body)
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message,
                retry_after=int(retry_after) if retry_after else None,
                response_body=body,
            )
        elif status == 400:
            errors = body.get("errors") if body else None
            raise ValidationError(message, errors=errors, response_body=body)
        elif status >= 500:
            raise ServerError(message, status_code=status, response_body=body)
        else:
            raise AragoraError(message, status_code=status, response_body=body)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> AragoraClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class AragoraAsyncClient:
    """
    Asynchronous Aragora API client.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     debate = await client.debates.create(task="Should we adopt microservices?")
        ...     print(debate.debate_id)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the async Aragora client.

        Args:
            base_url: Base URL of the Aragora API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=self._build_headers(),
        )

        # Initialize namespace APIs
        self._init_namespaces()

    def _build_headers(self) -> dict[str, str]:
        """Build default headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "aragora-python/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _init_namespaces(self) -> None:
        """Initialize namespace API objects."""
        from .namespaces.admin import AsyncAdminAPI
        from .namespaces.agents import AsyncAgentsAPI
        from .namespaces.analytics import AsyncAnalyticsAPI
        from .namespaces.debates import AsyncDebatesAPI
        from .namespaces.onboarding import AsyncOnboardingAPI
        from .namespaces.usage import AsyncUsageAPI
        from .namespaces.workflows import AsyncWorkflowsAPI

        self.admin = AsyncAdminAPI(self)
        self.agents = AsyncAgentsAPI(self)
        self.analytics = AsyncAnalyticsAPI(self)
        self.debates = AsyncDebatesAPI(self)
        self.onboarding = AsyncOnboardingAPI(self)
        self.usage = AsyncUsageAPI(self)
        self.workflows = AsyncWorkflowsAPI(self)

    async def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """
        Make an async HTTP request to the Aragora API.

        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            json: JSON body
            headers: Additional headers

        Returns:
            Parsed JSON response
        """
        import asyncio

        url = urljoin(self.base_url, path)
        request_headers = {**self._build_headers(), **(headers or {})}

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=request_headers,
                )

                if response.is_success:
                    if response.content:
                        return response.json()
                    return None

                self._handle_error_response(response)

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise AragoraError("Request timed out") from e

            except httpx.ConnectError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise AragoraError("Connection failed") from e

        if last_error:
            raise AragoraError(f"Request failed after {self.max_retries} retries") from last_error

        raise AragoraError("Unexpected error")

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses."""
        try:
            body = response.json()
            message = body.get("error", body.get("message", response.text))
        except Exception:
            body = None
            message = response.text or "Unknown error"

        status = response.status_code

        if status == 401:
            raise AuthenticationError(message, response_body=body)
        elif status == 403:
            raise AuthorizationError(message, response_body=body)
        elif status == 404:
            raise NotFoundError(message, response_body=body)
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message,
                retry_after=int(retry_after) if retry_after else None,
                response_body=body,
            )
        elif status == 400:
            errors = body.get("errors") if body else None
            raise ValidationError(message, errors=errors, response_body=body)
        elif status >= 500:
            raise ServerError(message, status_code=status, response_body=body)
        else:
            raise AragoraError(message, status_code=status, response_body=body)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> AragoraAsyncClient:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
