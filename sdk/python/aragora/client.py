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
        from .namespaces.audit import AuditAPI
        from .namespaces.auth import AuthAPI
        from .namespaces.backups import BackupsAPI
        from .namespaces.belief import BeliefAPI
        from .namespaces.billing import BillingAPI
        from .namespaces.budgets import BudgetsAPI
        from .namespaces.consensus import ConsensusAPI
        from .namespaces.control_plane import ControlPlaneAPI
        from .namespaces.critiques import CritiquesAPI
        from .namespaces.debates import DebatesAPI
        from .namespaces.decisions import DecisionsAPI
        from .namespaces.documents import DocumentsAPI
        from .namespaces.expenses import ExpensesAPI
        from .namespaces.explainability import ExplainabilityAPI
        from .namespaces.gauntlet import GauntletAPI
        from .namespaces.genesis import GenesisAPI
        from .namespaces.health import HealthAPI
        from .namespaces.integrations import IntegrationsAPI
        from .namespaces.invoice_processing import InvoiceProcessingAPI
        from .namespaces.knowledge import KnowledgeAPI
        from .namespaces.marketplace import MarketplaceAPI
        from .namespaces.memory import MemoryAPI
        from .namespaces.monitoring import MonitoringAPI
        from .namespaces.nomic import NomicAPI
        from .namespaces.notifications import NotificationsAPI
        from .namespaces.onboarding import OnboardingAPI
        from .namespaces.organizations import OrganizationsAPI
        from .namespaces.payments import PaymentsAPI
        from .namespaces.policies import PoliciesAPI
        from .namespaces.pulse import PulseAPI
        from .namespaces.ranking import RankingAPI
        from .namespaces.rbac import RBACAPI
        from .namespaces.receipts import ReceiptsAPI
        from .namespaces.relationships import RelationshipsAPI
        from .namespaces.replays import ReplaysAPI
        from .namespaces.rlm import RLMAPI
        from .namespaces.sme import SMEAPI
        from .namespaces.teams import TeamsAPI
        from .namespaces.tenants import TenantsAPI
        from .namespaces.tournaments import TournamentsAPI
        from .namespaces.training import TrainingAPI
        from .namespaces.usage import UsageAPI
        from .namespaces.verification import VerificationAPI
        from .namespaces.webhooks import WebhooksAPI
        from .namespaces.workflows import WorkflowsAPI
        from .namespaces.workspaces import WorkspacesAPI

        self.admin = AdminAPI(self)
        self.agents = AgentsAPI(self)
        self.analytics = AnalyticsAPI(self)
        self.audit = AuditAPI(self)
        self.auth = AuthAPI(self)
        self.backups = BackupsAPI(self)
        self.belief = BeliefAPI(self)
        self.billing = BillingAPI(self)
        self.budgets = BudgetsAPI(self)
        self.consensus = ConsensusAPI(self)
        self.control_plane = ControlPlaneAPI(self)
        self.critiques = CritiquesAPI(self)
        self.debates = DebatesAPI(self)
        self.decisions = DecisionsAPI(self)
        self.documents = DocumentsAPI(self)
        self.expenses = ExpensesAPI(self)
        self.explainability = ExplainabilityAPI(self)
        self.gauntlet = GauntletAPI(self)
        self.genesis = GenesisAPI(self)
        self.health = HealthAPI(self)
        self.integrations = IntegrationsAPI(self)
        self.invoice_processing = InvoiceProcessingAPI(self)
        self.knowledge = KnowledgeAPI(self)
        self.marketplace = MarketplaceAPI(self)
        self.memory = MemoryAPI(self)
        self.monitoring = MonitoringAPI(self)
        self.nomic = NomicAPI(self)
        self.notifications = NotificationsAPI(self)
        self.onboarding = OnboardingAPI(self)
        self.organizations = OrganizationsAPI(self)
        self.payments = PaymentsAPI(self)
        self.policies = PoliciesAPI(self)
        self.pulse = PulseAPI(self)
        self.ranking = RankingAPI(self)
        self.rbac = RBACAPI(self)
        self.receipts = ReceiptsAPI(self)
        self.relationships = RelationshipsAPI(self)
        self.replays = ReplaysAPI(self)
        self.rlm = RLMAPI(self)
        self.sme = SMEAPI(self)
        self.teams = TeamsAPI(self)
        self.tenants = TenantsAPI(self)
        self.training = TrainingAPI(self)
        self.tournaments = TournamentsAPI(self)
        self.usage = UsageAPI(self)
        self.verification = VerificationAPI(self)
        self.webhooks = WebhooksAPI(self)
        self.workflows = WorkflowsAPI(self)
        self.workspaces = WorkspacesAPI(self)

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

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Backward-compatible alias for request()."""
        return self.request(method, path, params=params, json=json, headers=headers)

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
        from .namespaces.audit import AsyncAuditAPI
        from .namespaces.auth import AsyncAuthAPI
        from .namespaces.backups import AsyncBackupsAPI
        from .namespaces.belief import AsyncBeliefAPI
        from .namespaces.billing import AsyncBillingAPI
        from .namespaces.budgets import AsyncBudgetsAPI
        from .namespaces.consensus import AsyncConsensusAPI
        from .namespaces.control_plane import AsyncControlPlaneAPI
        from .namespaces.critiques import AsyncCritiquesAPI
        from .namespaces.debates import AsyncDebatesAPI
        from .namespaces.decisions import AsyncDecisionsAPI
        from .namespaces.documents import AsyncDocumentsAPI
        from .namespaces.expenses import AsyncExpensesAPI
        from .namespaces.explainability import AsyncExplainabilityAPI
        from .namespaces.gauntlet import AsyncGauntletAPI
        from .namespaces.genesis import AsyncGenesisAPI
        from .namespaces.health import AsyncHealthAPI
        from .namespaces.integrations import AsyncIntegrationsAPI
        from .namespaces.invoice_processing import AsyncInvoiceProcessingAPI
        from .namespaces.knowledge import AsyncKnowledgeAPI
        from .namespaces.marketplace import AsyncMarketplaceAPI
        from .namespaces.memory import AsyncMemoryAPI
        from .namespaces.monitoring import AsyncMonitoringAPI
        from .namespaces.nomic import AsyncNomicAPI
        from .namespaces.notifications import AsyncNotificationsAPI
        from .namespaces.onboarding import AsyncOnboardingAPI
        from .namespaces.organizations import AsyncOrganizationsAPI
        from .namespaces.payments import AsyncPaymentsAPI
        from .namespaces.policies import AsyncPoliciesAPI
        from .namespaces.pulse import AsyncPulseAPI
        from .namespaces.ranking import AsyncRankingAPI
        from .namespaces.rbac import AsyncRBACAPI
        from .namespaces.receipts import AsyncReceiptsAPI
        from .namespaces.relationships import AsyncRelationshipsAPI
        from .namespaces.replays import AsyncReplaysAPI
        from .namespaces.rlm import AsyncRLMAPI
        from .namespaces.sme import AsyncSMEAPI
        from .namespaces.teams import AsyncTeamsAPI
        from .namespaces.tenants import AsyncTenantsAPI
        from .namespaces.tournaments import AsyncTournamentsAPI
        from .namespaces.training import AsyncTrainingAPI
        from .namespaces.usage import AsyncUsageAPI
        from .namespaces.verification import AsyncVerificationAPI
        from .namespaces.webhooks import AsyncWebhooksAPI
        from .namespaces.workflows import AsyncWorkflowsAPI
        from .namespaces.workspaces import AsyncWorkspacesAPI

        self.admin = AsyncAdminAPI(self)
        self.agents = AsyncAgentsAPI(self)
        self.analytics = AsyncAnalyticsAPI(self)
        self.audit = AsyncAuditAPI(self)
        self.auth = AsyncAuthAPI(self)
        self.backups = AsyncBackupsAPI(self)
        self.belief = AsyncBeliefAPI(self)
        self.billing = AsyncBillingAPI(self)
        self.budgets = AsyncBudgetsAPI(self)
        self.consensus = AsyncConsensusAPI(self)
        self.control_plane = AsyncControlPlaneAPI(self)
        self.critiques = AsyncCritiquesAPI(self)
        self.debates = AsyncDebatesAPI(self)
        self.decisions = AsyncDecisionsAPI(self)
        self.documents = AsyncDocumentsAPI(self)
        self.expenses = AsyncExpensesAPI(self)
        self.explainability = AsyncExplainabilityAPI(self)
        self.gauntlet = AsyncGauntletAPI(self)
        self.genesis = AsyncGenesisAPI(self)
        self.health = AsyncHealthAPI(self)
        self.integrations = AsyncIntegrationsAPI(self)
        self.invoice_processing = AsyncInvoiceProcessingAPI(self)
        self.knowledge = AsyncKnowledgeAPI(self)
        self.marketplace = AsyncMarketplaceAPI(self)
        self.memory = AsyncMemoryAPI(self)
        self.monitoring = AsyncMonitoringAPI(self)
        self.nomic = AsyncNomicAPI(self)
        self.notifications = AsyncNotificationsAPI(self)
        self.onboarding = AsyncOnboardingAPI(self)
        self.organizations = AsyncOrganizationsAPI(self)
        self.payments = AsyncPaymentsAPI(self)
        self.policies = AsyncPoliciesAPI(self)
        self.pulse = AsyncPulseAPI(self)
        self.ranking = AsyncRankingAPI(self)
        self.rbac = AsyncRBACAPI(self)
        self.receipts = AsyncReceiptsAPI(self)
        self.relationships = AsyncRelationshipsAPI(self)
        self.replays = AsyncReplaysAPI(self)
        self.rlm = AsyncRLMAPI(self)
        self.sme = AsyncSMEAPI(self)
        self.teams = AsyncTeamsAPI(self)
        self.tenants = AsyncTenantsAPI(self)
        self.training = AsyncTrainingAPI(self)
        self.tournaments = AsyncTournamentsAPI(self)
        self.usage = AsyncUsageAPI(self)
        self.verification = AsyncVerificationAPI(self)
        self.webhooks = AsyncWebhooksAPI(self)
        self.workflows = AsyncWorkflowsAPI(self)
        self.workspaces = AsyncWorkspacesAPI(self)

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

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Backward-compatible alias for request()."""
        return await self.request(method, path, params=params, json=json, headers=headers)

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
