"""
Aragora Python SDK Client.

Provides a type-safe interface for interacting with the Aragora API.

Usage:
    from aragora.client import AragoraClient

    # Synchronous usage
    client = AragoraClient(base_url="http://localhost:8080")
    debate = client.debates.create(task="Should we use microservices?")
    result = client.debates.get(debate.debate_id)

    # Async usage
    async with AragoraClient(base_url="http://localhost:8080") as client:
        debate = await client.debates.create_async(task="...")
        result = await client.debates.get_async(debate.debate_id)

    # With retry and rate limiting
    from aragora.client import AragoraClient, RetryConfig
    client = AragoraClient(
        base_url="http://localhost:8080",
        retry_config=RetryConfig(max_retries=3, backoff_factor=0.5),
        rate_limit_rps=10,
    )

    # Batch fetching
    debates = await client.debates.batch_get_async(["id1", "id2", "id3"])

    # Pagination
    async for debate in client.debates.iterate_async(status="completed"):
        print(debate.task)
"""

from __future__ import annotations

import json
import logging
import time as time_module
from typing import TYPE_CHECKING, Any, NoReturn, Optional
from urllib.parse import urljoin

# Import from refactored modules
from .errors import (
    AragoraAPIError,
    AuthenticationError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
)
from .transport import RateLimiter, RetryConfig
from .resources import (
    AgentsAPI,
    AnalyticsAPI,
    AuditAPI,
    AuthAPI,
    BillingAPI,
    CodebaseAPI,
    ConsensusAPI,
    CostManagementAPI,
    DebatesAPI,
    DecisionsAPI,
    DocumentsAPI,
    ExplainabilityAPI,
    GauntletAPI,
    GmailAPI,
    GraphDebatesAPI,
    KnowledgeAPI,
    LeaderboardAPI,
    MatrixDebatesAPI,
    MemoryAPI,
    NotificationsAPI,
    OnboardingAPI,
    OrganizationsAPI,
    PoliciesAPI,
    PulseAPI,
    RBACAPI,
    ReplayAPI,
    SystemAPI,
    TenantsAPI,
    TournamentsAPI,
    VerificationAPI,
    WorkflowsAPI,
)
from .models import (
    HealthCheck,
)

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)


# GauntletAPI, GraphDebatesAPI, MatrixDebatesAPI, ReplayAPI are imported from resources


class AragoraClient:
    """
    Aragora API client.

    Provides synchronous and asynchronous access to the Aragora API.

    Available API interfaces:
        - debates: Standard debates (create, get, list, run)
        - graph_debates: Graph-structured debates with branching
        - matrix_debates: Parallel scenario debates
        - documents: Document management, batch processing, and auditing
        - audit: Enterprise audit features (presets, workflow, quick audit)
        - verification: Formal claim verification
        - memory: Memory tier analytics
        - agents: Agent discovery and profiles
        - leaderboard: ELO rankings
        - gauntlet: Adversarial validation
        - replays: Debate replay viewing and export
        - consensus: Consensus memory, settled topics, and dissent tracking
        - pulse: Trending topics and debate suggestions
        - system: System health, stats, and circuit breaker management
        - tournaments: Agent tournament creation and management
        - billing: Subscription and usage management
        - rbac: Role-based access control
        - auth: Authentication and MFA management
        - knowledge: Knowledge Mound search and management
        - workflows: Workflow creation and execution
        - organizations: Multi-tenant organization management
        - tenants: Tenant isolation and settings
        - codebase: Codebase analysis integration
        - policies: Compliance policy management
        - explainability: Decision explanations and counterfactuals
        - cost_management: Usage tracking and budget management
        - notifications: Notification preferences and history
        - decisions: Unified decision-making API
        - onboarding: User onboarding flows
        - gmail: Gmail connector integration

    Usage:
        # Synchronous
        client = AragoraClient(base_url="http://localhost:8080")
        debate = client.debates.run(task="Should we use microservices?")

        # Graph debate with branching
        result = client.graph_debates.create(task="Design a distributed system")

        # Matrix debate with scenarios
        result = client.matrix_debates.create(
            task="Should we adopt microservices?",
            scenarios=[
                {"name": "small team", "parameters": {"team_size": 5}},
                {"name": "large team", "parameters": {"team_size": 50}},
            ]
        )

        # Verify a claim
        result = client.verification.verify(claim="All primes > 2 are odd")

        # Memory analytics
        analytics = client.memory.analytics(days=30)

        # Asynchronous
        async with AragoraClient(base_url="http://localhost:8080") as client:
            debate = await client.debates.run_async(task="...")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: int = 60,
        retry_config: RetryConfig | None = None,
        rate_limit_rps: float = 0,
    ):
        """
        Initialize the Aragora client.

        Args:
            base_url: Base URL of the Aragora API server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            retry_config: Optional retry configuration for resilient requests.
            rate_limit_rps: Client-side rate limit in requests per second (0 = disabled).
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_config = retry_config
        self._rate_limiter = RateLimiter(rate_limit_rps) if rate_limit_rps > 0 else None

        # Lazy-loaded HTTP clients
        self._session: Optional["aiohttp.ClientSession"] = None
        self._sync_client: Any = None

        # API interfaces
        self.debates = DebatesAPI(self)
        self.agents = AgentsAPI(self)
        self.leaderboard = LeaderboardAPI(self)
        self.gauntlet = GauntletAPI(self)

        # Extended API interfaces
        self.graph_debates = GraphDebatesAPI(self)
        self.matrix_debates = MatrixDebatesAPI(self)
        self.verification = VerificationAPI(self)
        self.memory = MemoryAPI(self)
        self.replays = ReplayAPI(self)

        # Document management and auditing
        self.documents = DocumentsAPI(self)

        # Enterprise audit features (presets, workflow, quick audit)
        self.audit = AuditAPI(self)

        # Consensus memory and dissent tracking
        self.consensus = ConsensusAPI(self)

        # Trending topics and debate suggestions
        self.pulse = PulseAPI(self)

        # System health and status
        self.system = SystemAPI(self)

        # Agent tournaments
        self.tournaments = TournamentsAPI(self)

        # Analytics (disagreements, role rotation, early stops, consensus quality)
        self.analytics = AnalyticsAPI(self)

        # Billing and subscription management
        self.billing = BillingAPI(self)

        # Role-Based Access Control
        self.rbac = RBACAPI(self)

        # Authentication and MFA
        self.auth = AuthAPI(self)

        # Knowledge Mound
        self.knowledge = KnowledgeAPI(self)

        # Workflows
        self.workflows = WorkflowsAPI(self)

        # Organizations (multi-tenant management)
        self.organizations = OrganizationsAPI(self)

        # Tenants (tenant isolation and settings)
        self.tenants = TenantsAPI(self)

        # Codebase analysis
        self.codebase = CodebaseAPI(self)

        # Compliance policies
        self.policies = PoliciesAPI(self)

        # Decision explainability
        self.explainability = ExplainabilityAPI(self)

        # Cost management
        self.cost_management = CostManagementAPI(self)

        # Notifications
        self.notifications = NotificationsAPI(self)

        # Unified decisions API
        self.decisions = DecisionsAPI(self)

        # User onboarding
        self.onboarding = OnboardingAPI(self)

        # Gmail connector
        self.gmail = GmailAPI(self)

    def _get_headers(self) -> dict[str, str]:
        """Get common request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Make a synchronous GET request with retry and rate limiting."""
        import urllib.error
        import urllib.parse
        import urllib.request

        url = urljoin(self.base_url, path)
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    def _post(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make a synchronous POST request with retry and rate limiting."""
        import urllib.error
        import urllib.request

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        req = urllib.request.Request(
            url,
            headers=request_headers,
            method="POST",
            data=json.dumps(data).encode(),
        )
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    def _delete(self, path: str, params: dict | None = None) -> dict:
        """Make a synchronous DELETE request with retry and rate limiting."""
        import urllib.error
        import urllib.parse
        import urllib.request

        url = urljoin(self.base_url, path)
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers=self._get_headers(), method="DELETE")
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    async def _delete_async(self, path: str, params: dict | None = None) -> dict:
        """Make an asynchronous DELETE request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.delete(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    def _put(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make a synchronous PUT request with retry and rate limiting."""
        import urllib.error
        import urllib.request

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        req = urllib.request.Request(
            url,
            headers=request_headers,
            method="PUT",
            data=json.dumps(data).encode(),
        )
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    async def _put_async(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make an asynchronous PUT request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.put(
                    url,
                    headers=request_headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    def _patch(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make a synchronous PATCH request with retry and rate limiting."""
        import urllib.error
        import urllib.request

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        req = urllib.request.Request(
            url,
            headers=request_headers,
            method="PATCH",
            data=json.dumps(data).encode(),
        )
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    async def _patch_async(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make an asynchronous PATCH request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.patch(
                    url,
                    headers=request_headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    def _handle_http_error(self, e: Any) -> NoReturn:
        """Handle HTTP errors with specific error classes."""
        try:
            body = json.loads(e.read().decode())
            error_msg = body.get("error", str(e))
            error_code = body.get("code", "HTTP_ERROR")
        except Exception as parse_err:
            # Failed to parse error response body - use raw error
            logger.debug(f"Could not parse HTTP error body: {parse_err}")
            error_msg = str(e)
            error_code = "HTTP_ERROR"

        # Map HTTP status codes to specific error classes
        status_code = getattr(e, "code", 500)
        if status_code == 401:
            raise AuthenticationError(error_msg)
        elif status_code == 402:
            raise QuotaExceededError(error_msg)
        elif status_code == 404:
            raise NotFoundError(error_msg)
        elif status_code == 429:
            raise RateLimitError(error_msg)
        elif status_code == 400:
            raise ValidationError(error_msg)
        else:
            raise AragoraAPIError(error_msg, error_code, status_code)

    async def _get_async(self, path: str, params: dict | None = None) -> dict:
        """Make an asynchronous GET request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    async def _post_async(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make an asynchronous POST request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.post(
                    url,
                    headers=request_headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    def health(self) -> HealthCheck:
        """Check API health."""
        response = self._get("/api/health")
        return HealthCheck(**response)

    async def health_async(self) -> HealthCheck:
        """Async version of health()."""
        response = await self._get_async("/api/health")
        return HealthCheck(**response)

    async def stream_debate(self, debate_id: str, options: Optional[Any] = None):
        """
        Stream events for a specific debate.

        This is a convenience method that creates a WebSocket connection
        and yields events for the specified debate.

        Args:
            debate_id: The ID of the debate to stream
            options: Optional WebSocketOptions for connection configuration

        Yields:
            DebateEvent objects for each event from the debate

        Example:
            async for event in client.stream_debate("debate-123"):
                print(f"{event.type}: {event.data}")
                if event.type == DebateEventType.DEBATE_END:
                    break
        """
        from aragora.client.websocket import stream_debate

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        async for event in stream_debate(
            base_url=ws_url,
            debate_id=debate_id,
            options=options,
        ):
            yield event

    async def stream_all_debates(self, options: Optional[Any] = None):
        """
        Stream events for all debates (no filter).

        This is a convenience method for monitoring all debate activity
        on the server.

        Args:
            options: Optional WebSocketOptions for connection configuration

        Yields:
            DebateEvent objects for events from all debates

        Example:
            async for event in client.stream_all_debates():
                print(f"[{event.debate_id}] {event.type}: {event.data}")
        """
        from aragora.client.websocket import stream_debate

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        async for event in stream_debate(
            base_url=ws_url,
            debate_id="",  # Empty string means no filter
            options=options,
        ):
            yield event

    async def create_debate_and_stream(
        self,
        task: str,
        agents: list,
        rounds: int = 3,
        stream_options: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        Create a debate and immediately start streaming its events.

        This is a convenience method that combines debate creation with
        event streaming for a streamlined workflow.

        Args:
            task: The debate topic/question
            agents: List of agent IDs to participate
            rounds: Number of debate rounds (default: 3)
            stream_options: Optional WebSocketOptions for connection configuration
            **kwargs: Additional arguments passed to debates.create_async()

        Returns:
            Tuple of (debate_response, event_stream)

        Example:
            debate, stream = await client.create_debate_and_stream(
                task="Should we use microservices?",
                agents=["claude", "gpt-4"],
            )
            print(f"Created debate: {debate['debate_id']}")
            async for event in stream:
                print(f"{event.type}: {event.data}")
        """
        # Create the debate first
        debate = await self.debates.create_async(
            task=task,
            agents=agents,
            rounds=rounds,
            **kwargs,
        )

        # Extract debate_id from response
        debate_id = debate.debate_id if hasattr(debate, "debate_id") else debate.get("debate_id")
        if not debate_id:
            raise ValueError("No debate_id in create_debate response")

        # Return debate and stream
        stream = self.stream_debate(debate_id, stream_options)
        return debate, stream

    def __enter__(self) -> "AragoraClient":
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit."""
        self.close()

    async def __aenter__(self) -> "AragoraClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close_async()

    async def close_async(self) -> None:
        """Close async HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def close(self) -> None:
        """Close any open resources."""
        # Sync client cleanup if needed
        self._sync_client = None
