"""
Aragora API Client

Main HTTP client for interacting with the Aragora platform.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urljoin

if TYPE_CHECKING:
    from .websocket import AragoraWebSocket

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
            "User-Agent": "aragora-python-sdk/2.6.3",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _init_namespaces(self) -> None:
        """Initialize namespace API objects."""
        from .namespaces.a2a import A2AAPI
        from .namespaces.accounting import AccountingAPI
        from .namespaces.admin import AdminAPI
        from .namespaces.advertising import AdvertisingAPI
        from .namespaces.agent_selection import AgentSelectionAPI
        from .namespaces.agents import AgentsAPI
        from .namespaces.analytics import AnalyticsAPI
        from .namespaces.ap_automation import APAutomationAPI
        from .namespaces.approvals import ApprovalsAPI
        from .namespaces.ar_automation import ARAutomationAPI
        from .namespaces.audio import AudioAPI
        from .namespaces.audit_trail import AuditTrailAPI
        from .namespaces.audit import AuditAPI
        from .namespaces.auditing import AuditingAPI
        from .namespaces.auth import AuthAPI
        from .namespaces.backups import BackupsAPI
        from .namespaces.batch import BatchAPI
        from .namespaces.belief import BeliefAPI
        from .namespaces.billing import BillingAPI
        from .namespaces.bots import BotsAPI
        from .namespaces.budgets import BudgetsAPI
        from .namespaces.blockchain import BlockchainAPI
        from .namespaces.calibration import CalibrationAPI
        from .namespaces.canvas import CanvasAPI
        from .namespaces.chat import ChatAPI
        from .namespaces.checkpoints import CheckpointsAPI
        from .namespaces.classify import ClassifyAPI
        from .namespaces.code_review import CodeReviewAPI
        from .namespaces.codebase import CodebaseAPI
        from .namespaces.compliance import ComplianceAPI
        from .namespaces.computer_use import ComputerUseAPI
        from .namespaces.connectors import ConnectorsAPI
        from .namespaces.consensus import ConsensusAPI
        from .namespaces.control_plane import ControlPlaneAPI
        from .namespaces.cost_management import CostManagementAPI
        from .namespaces.critiques import CritiquesAPI
        from .namespaces.cross_pollination import CrossPollinationAPI
        from .namespaces.dashboard import DashboardAPI
        from .namespaces.debates import DebatesAPI
        from .namespaces.decisions import DecisionsAPI
        from .namespaces.deliberations import DeliberationsAPI
        from .namespaces.dependency_analysis import DependencyAnalysisAPI
        from .namespaces.devices import DevicesAPI
        from .namespaces.disaster_recovery import DisasterRecoveryAPI
        from .namespaces.documents import DocumentsAPI
        from .namespaces.email_debate import EmailDebateAPI
        from .namespaces.email_services import EmailServicesAPI
        from .namespaces.evaluation import EvaluationAPI
        from .namespaces.evolution import EvolutionAPI
        from .namespaces.expenses import ExpensesAPI
        from .namespaces.explainability import ExplainabilityAPI
        from .namespaces.facts import FactsAPI
        from .namespaces.feedback import FeedbackAPI
        from .namespaces.flips import FlipsAPI
        from .namespaces.gateway import GatewayAPI
        from .namespaces.gauntlet import GauntletAPI
        from .namespaces.genesis import GenesisAPI
        from .namespaces.gmail import GmailAPI
        from .namespaces.health import HealthAPI
        from .namespaces.hybrid_debates import HybridDebatesAPI
        from .namespaces.inbox_command import InboxCommandAPI
        from .namespaces.index import IndexAPI
        from .namespaces.insights import InsightsAPI
        from .namespaces.integrations import IntegrationsAPI
        from .namespaces.invoice_processing import InvoiceProcessingAPI
        from .namespaces.knowledge import KnowledgeAPI
        from .namespaces.knowledge_chat import KnowledgeChatAPI
        from .namespaces.marketplace import MarketplaceAPI
        from .namespaces.matches import MatchesAPI
        from .namespaces.media import MediaAPI
        from .namespaces.memory import MemoryAPI
        from .namespaces.metrics import MetricsAPI
        from .namespaces.ml import MLAPI
        from .namespaces.monitoring import MonitoringAPI
        from .namespaces.nomic import NomicAPI
        from .namespaces.notifications import NotificationsAPI
        from .namespaces.oauth_wizard import OAuthWizardAPI
        from .namespaces.onboarding import OnboardingAPI
        from .namespaces.openclaw import OpenclawAPI
        from .namespaces.openapi import OpenApiAPI
        from .namespaces.orchestration import OrchestrationAPI
        from .namespaces.organizations import OrganizationsAPI
        from .namespaces.outlook import OutlookAPI
        from .namespaces.partner import PartnerAPI
        from .namespaces.payments import PaymentsAPI
        from .namespaces.plugins import PluginsAPI
        from .namespaces.podcast import PodcastAPI
        from .namespaces.policies import PoliciesAPI
        from .namespaces.privacy import PrivacyAPI
        from .namespaces.probes import ProbesAPI
        from .namespaces.pulse import PulseAPI
        from .namespaces.queue import QueueAPI
        from .namespaces.ranking import RankingAPI
        from .namespaces.rbac import RBACAPI
        from .namespaces.receipts import ReceiptsAPI
        from .namespaces.reconciliation import ReconciliationAPI
        from .namespaces.relationships import RelationshipsAPI
        from .namespaces.replays import ReplaysAPI
        from .namespaces.repository import RepositoryAPI
        from .namespaces.reputation import ReputationAPI
        from .namespaces.reviews import ReviewsAPI
        from .namespaces.rlm import RLMAPI
        from .namespaces.routing import RoutingAPI
        from .namespaces.scim import SCIMAPI
        from .namespaces.security import SecurityAPI
        from .namespaces.slo import SLOAPI
        from .namespaces.sme import SMEAPI
        from .namespaces.social import SocialAPI
        from .namespaces.sso import SSOAPI
        from .namespaces.support import SupportAPI
        from .namespaces.system import SystemAPI
        from .namespaces.teams import TeamsAPI
        from .namespaces.tenants import TenantsAPI
        from .namespaces.threat_intel import ThreatIntelAPI
        from .namespaces.tournaments import TournamentsAPI
        from .namespaces.training import TrainingAPI
        from .namespaces.uncertainty import UncertaintyAPI
        from .namespaces.unified_inbox import UnifiedInboxAPI
        from .namespaces.usage import UsageAPI
        from .namespaces.verification import VerificationAPI
        from .namespaces.verticals import VerticalsAPI
        from .namespaces.voice import VoiceAPI
        from .namespaces.webhooks import WebhooksAPI
        from .namespaces.workflow_templates import WorkflowTemplatesAPI
        from .namespaces.workflows import WorkflowsAPI
        from .namespaces.workspaces import WorkspacesAPI
        from .namespaces.youtube import YouTubeAPI

        self.a2a = A2AAPI(self)
        self.accounting = AccountingAPI(self)
        self.admin = AdminAPI(self)
        self.advertising = AdvertisingAPI(self)
        self.agent_selection = AgentSelectionAPI(self)
        self.agents = AgentsAPI(self)
        self.analytics = AnalyticsAPI(self)
        self.ap_automation = APAutomationAPI(self)
        self.approvals = ApprovalsAPI(self)
        self.ar_automation = ARAutomationAPI(self)
        self.audio = AudioAPI(self)
        self.audit = AuditAPI(self)
        self.audit_trail = AuditTrailAPI(self)
        self.auditing = AuditingAPI(self)
        self.auth = AuthAPI(self)
        self.backups = BackupsAPI(self)
        self.batch = BatchAPI(self)
        self.belief = BeliefAPI(self)
        self.bots = BotsAPI(self)
        self.billing = BillingAPI(self)
        self.budgets = BudgetsAPI(self)
        self.blockchain = BlockchainAPI(self)
        self.calibration = CalibrationAPI(self)
        self.canvas = CanvasAPI(self)
        self.chat = ChatAPI(self)
        self.checkpoints = CheckpointsAPI(self)
        self.classify = ClassifyAPI(self)
        self.code_review = CodeReviewAPI(self)
        self.codebase = CodebaseAPI(self)
        self.compliance = ComplianceAPI(self)
        self.computer_use = ComputerUseAPI(self)
        self.connectors = ConnectorsAPI(self)
        self.consensus = ConsensusAPI(self)
        self.control_plane = ControlPlaneAPI(self)
        self.cost_management = CostManagementAPI(self)
        self.critiques = CritiquesAPI(self)
        self.cross_pollination = CrossPollinationAPI(self)
        self.dashboard = DashboardAPI(self)
        self.debates = DebatesAPI(self)
        self.decisions = DecisionsAPI(self)
        self.deliberations = DeliberationsAPI(self)
        self.dependency_analysis = DependencyAnalysisAPI(self)
        self.devices = DevicesAPI(self)
        self.disaster_recovery = DisasterRecoveryAPI(self)
        self.documents = DocumentsAPI(self)
        self.email_debate = EmailDebateAPI(self)
        self.email_services = EmailServicesAPI(self)
        self.evaluation = EvaluationAPI(self)
        self.evolution = EvolutionAPI(self)
        self.expenses = ExpensesAPI(self)
        self.explainability = ExplainabilityAPI(self)
        self.facts = FactsAPI(self)
        self.feedback = FeedbackAPI(self)
        self.flips = FlipsAPI(self)
        self.gateway = GatewayAPI(self)
        self.gauntlet = GauntletAPI(self)
        self.genesis = GenesisAPI(self)
        self.gmail = GmailAPI(self)
        self.health = HealthAPI(self)
        self.hybrid_debates = HybridDebatesAPI(self)
        self.inbox_command = InboxCommandAPI(self)
        self.index = IndexAPI(self)
        self.integrations = IntegrationsAPI(self)
        self.insights = InsightsAPI(self)
        self.invoice_processing = InvoiceProcessingAPI(self)
        self.knowledge = KnowledgeAPI(self)
        self.knowledge_chat = KnowledgeChatAPI(self)
        self.marketplace = MarketplaceAPI(self)
        self.memory = MemoryAPI(self)
        self.matches = MatchesAPI(self)
        self.media = MediaAPI(self)
        self.metrics = MetricsAPI(self)
        self.ml = MLAPI(self)
        self.monitoring = MonitoringAPI(self)
        self.nomic = NomicAPI(self)
        self.notifications = NotificationsAPI(self)
        self.oauth_wizard = OAuthWizardAPI(self)
        self.openclaw = OpenclawAPI(self)
        self.openapi = OpenApiAPI(self)
        self.orchestration = OrchestrationAPI(self)
        self.onboarding = OnboardingAPI(self)
        self.organizations = OrganizationsAPI(self)
        self.outlook = OutlookAPI(self)
        self.partner = PartnerAPI(self)
        self.payments = PaymentsAPI(self)
        self.plugins = PluginsAPI(self)
        self.podcast = PodcastAPI(self)
        self.policies = PoliciesAPI(self)
        self.privacy = PrivacyAPI(self)
        self.probes = ProbesAPI(self)
        self.pulse = PulseAPI(self)
        self.queue = QueueAPI(self)
        self.reconciliation = ReconciliationAPI(self)
        self.ranking = RankingAPI(self)
        self.rbac = RBACAPI(self)
        self.receipts = ReceiptsAPI(self)
        self.reputation = ReputationAPI(self)
        self.reviews = ReviewsAPI(self)
        self.relationships = RelationshipsAPI(self)
        self.replays = ReplaysAPI(self)
        self.repository = RepositoryAPI(self)
        self.rlm = RLMAPI(self)
        self.routing = RoutingAPI(self)
        self.scim = SCIMAPI(self)
        self.security = SecurityAPI(self)
        self.slo = SLOAPI(self)
        self.sme = SMEAPI(self)
        self.social = SocialAPI(self)
        self.sso = SSOAPI(self)
        self.support = SupportAPI(self)
        self.system = SystemAPI(self)
        self.teams = TeamsAPI(self)
        self.tenants = TenantsAPI(self)
        self.training = TrainingAPI(self)
        self.threat_intel = ThreatIntelAPI(self)
        self.tournaments = TournamentsAPI(self)
        self.uncertainty = UncertaintyAPI(self)
        self.unified_inbox = UnifiedInboxAPI(self)
        self.usage = UsageAPI(self)
        self.verification = VerificationAPI(self)
        self.verticals = VerticalsAPI(self)
        self.voice = VoiceAPI(self)
        self.webhooks = WebhooksAPI(self)
        self.workflow_templates = WorkflowTemplatesAPI(self)
        self.workflows = WorkflowsAPI(self)
        self.workspaces = WorkspacesAPI(self)
        self.youtube = YouTubeAPI(self)

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make an HTTP request to the Aragora API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: API path (e.g., "/api/v1/debates")
            params: Query parameters
            json: JSON body for POST/PUT requests
            headers: Additional headers

        Returns:
            Parsed JSON response as a dictionary.

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
                        return cast(dict[str, Any], response.json())
                    return {}

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

            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Use server-specified retry delay if available
                    delay = e.retry_after if e.retry_after else self.retry_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise

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
    ) -> dict[str, Any]:
        """Backward-compatible alias for request()."""
        return self.request(method, path, params=params, json=json, headers=headers)

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses and raise appropriate exceptions."""
        try:
            body = response.json()
            message = body.get("error", body.get("message", response.text))
            error_code = body.get("code")
            trace_id = body.get("trace_id")
        except (ValueError, AttributeError):
            body = None
            message = response.text or "Unknown error"
            error_code = None
            trace_id = None

        status = response.status_code

        if status == 401:
            raise AuthenticationError(
                message, error_code=error_code, trace_id=trace_id, response_body=body
            )
        elif status == 403:
            raise AuthorizationError(
                message, error_code=error_code, trace_id=trace_id, response_body=body
            )
        elif status == 404:
            raise NotFoundError(
                message, error_code=error_code, trace_id=trace_id, response_body=body
            )
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message,
                retry_after=int(retry_after) if retry_after else None,
                error_code=error_code,
                trace_id=trace_id,
                response_body=body,
            )
        elif status == 400:
            errors = body.get("errors") if body else None
            raise ValidationError(
                message,
                errors=errors,
                error_code=error_code,
                trace_id=trace_id,
                response_body=body,
            )
        elif status >= 500:
            raise ServerError(
                message,
                status_code=status,
                error_code=error_code,
                trace_id=trace_id,
                response_body=body,
            )
        else:
            raise AragoraError(
                message,
                status_code=status,
                error_code=error_code,
                trace_id=trace_id,
                response_body=body,
            )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> AragoraClient:
        return self

    def __exit__(self, *args: Any) -> None:
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
        ws_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the async Aragora client.

        Args:
            base_url: Base URL of the Aragora API
            api_key: API key for authentication
            ws_url: Explicit WebSocket URL (derived from base_url if None)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.ws_url = ws_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=self._build_headers(),
        )

        # WebSocket client (created lazily)
        self._stream: AragoraWebSocket | None = None

        # Initialize namespace APIs
        self._init_namespaces()

    def _build_headers(self) -> dict[str, str]:
        """Build default headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "aragora-python-sdk/2.6.3",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _init_namespaces(self) -> None:
        """Initialize namespace API objects."""
        from .namespaces.a2a import AsyncA2AAPI
        from .namespaces.accounting import AsyncAccountingAPI
        from .namespaces.admin import AsyncAdminAPI
        from .namespaces.advertising import AsyncAdvertisingAPI
        from .namespaces.agent_selection import AsyncAgentSelectionAPI
        from .namespaces.agents import AsyncAgentsAPI
        from .namespaces.analytics import AsyncAnalyticsAPI
        from .namespaces.ap_automation import AsyncAPAutomationAPI
        from .namespaces.approvals import AsyncApprovalsAPI
        from .namespaces.ar_automation import AsyncARAutomationAPI
        from .namespaces.audio import AsyncAudioAPI
        from .namespaces.audit import AsyncAuditAPI
        from .namespaces.audit_trail import AsyncAuditTrailAPI
        from .namespaces.auditing import AsyncAuditingAPI
        from .namespaces.auth import AsyncAuthAPI
        from .namespaces.backups import AsyncBackupsAPI
        from .namespaces.batch import AsyncBatchAPI
        from .namespaces.belief import AsyncBeliefAPI
        from .namespaces.billing import AsyncBillingAPI
        from .namespaces.bots import AsyncBotsAPI
        from .namespaces.budgets import AsyncBudgetsAPI
        from .namespaces.blockchain import AsyncBlockchainAPI
        from .namespaces.calibration import AsyncCalibrationAPI
        from .namespaces.canvas import AsyncCanvasAPI
        from .namespaces.chat import AsyncChatAPI
        from .namespaces.checkpoints import AsyncCheckpointsAPI
        from .namespaces.classify import AsyncClassifyAPI
        from .namespaces.code_review import AsyncCodeReviewAPI
        from .namespaces.codebase import AsyncCodebaseAPI
        from .namespaces.compliance import AsyncComplianceAPI
        from .namespaces.computer_use import AsyncComputerUseAPI
        from .namespaces.connectors import AsyncConnectorsAPI
        from .namespaces.consensus import AsyncConsensusAPI
        from .namespaces.control_plane import AsyncControlPlaneAPI
        from .namespaces.cost_management import AsyncCostManagementAPI
        from .namespaces.critiques import AsyncCritiquesAPI
        from .namespaces.cross_pollination import AsyncCrossPollinationAPI
        from .namespaces.dashboard import AsyncDashboardAPI
        from .namespaces.debates import AsyncDebatesAPI
        from .namespaces.decisions import AsyncDecisionsAPI
        from .namespaces.deliberations import AsyncDeliberationsAPI
        from .namespaces.dependency_analysis import AsyncDependencyAnalysisAPI
        from .namespaces.devices import AsyncDevicesAPI
        from .namespaces.disaster_recovery import AsyncDisasterRecoveryAPI
        from .namespaces.documents import AsyncDocumentsAPI
        from .namespaces.email_debate import AsyncEmailDebateAPI
        from .namespaces.email_services import AsyncEmailServicesAPI
        from .namespaces.evaluation import AsyncEvaluationAPI
        from .namespaces.evolution import AsyncEvolutionAPI
        from .namespaces.expenses import AsyncExpensesAPI
        from .namespaces.explainability import AsyncExplainabilityAPI
        from .namespaces.facts import AsyncFactsAPI
        from .namespaces.feedback import AsyncFeedbackAPI
        from .namespaces.flips import AsyncFlipsAPI
        from .namespaces.gateway import AsyncGatewayAPI
        from .namespaces.gauntlet import AsyncGauntletAPI
        from .namespaces.genesis import AsyncGenesisAPI
        from .namespaces.gmail import AsyncGmailAPI
        from .namespaces.health import AsyncHealthAPI
        from .namespaces.hybrid_debates import AsyncHybridDebatesAPI
        from .namespaces.inbox_command import AsyncInboxCommandAPI
        from .namespaces.index import AsyncIndexAPI
        from .namespaces.insights import AsyncInsightsAPI
        from .namespaces.integrations import AsyncIntegrationsAPI
        from .namespaces.invoice_processing import AsyncInvoiceProcessingAPI
        from .namespaces.knowledge import AsyncKnowledgeAPI
        from .namespaces.knowledge_chat import AsyncKnowledgeChatAPI
        from .namespaces.marketplace import AsyncMarketplaceAPI
        from .namespaces.matches import AsyncMatchesAPI
        from .namespaces.media import AsyncMediaAPI
        from .namespaces.memory import AsyncMemoryAPI
        from .namespaces.metrics import AsyncMetricsAPI
        from .namespaces.ml import AsyncMLAPI
        from .namespaces.monitoring import AsyncMonitoringAPI
        from .namespaces.nomic import AsyncNomicAPI
        from .namespaces.notifications import AsyncNotificationsAPI
        from .namespaces.oauth_wizard import AsyncOAuthWizardAPI
        from .namespaces.onboarding import AsyncOnboardingAPI
        from .namespaces.openclaw import AsyncOpenclawAPI
        from .namespaces.openapi import AsyncOpenApiAPI
        from .namespaces.orchestration import AsyncOrchestrationAPI
        from .namespaces.organizations import AsyncOrganizationsAPI
        from .namespaces.outlook import AsyncOutlookAPI
        from .namespaces.partner import AsyncPartnerAPI
        from .namespaces.payments import AsyncPaymentsAPI
        from .namespaces.plugins import AsyncPluginsAPI
        from .namespaces.podcast import AsyncPodcastAPI
        from .namespaces.policies import AsyncPoliciesAPI
        from .namespaces.privacy import AsyncPrivacyAPI
        from .namespaces.probes import AsyncProbesAPI
        from .namespaces.pulse import AsyncPulseAPI
        from .namespaces.queue import AsyncQueueAPI
        from .namespaces.ranking import AsyncRankingAPI
        from .namespaces.rbac import AsyncRBACAPI
        from .namespaces.receipts import AsyncReceiptsAPI
        from .namespaces.reconciliation import AsyncReconciliationAPI
        from .namespaces.relationships import AsyncRelationshipsAPI
        from .namespaces.replays import AsyncReplaysAPI
        from .namespaces.repository import AsyncRepositoryAPI
        from .namespaces.reputation import AsyncReputationAPI
        from .namespaces.reviews import AsyncReviewsAPI
        from .namespaces.rlm import AsyncRLMAPI
        from .namespaces.routing import AsyncRoutingAPI
        from .namespaces.scim import AsyncSCIMAPI
        from .namespaces.security import AsyncSecurityAPI
        from .namespaces.slo import AsyncSLOAPI
        from .namespaces.sme import AsyncSMEAPI
        from .namespaces.social import AsyncSocialAPI
        from .namespaces.sso import AsyncSSOAPI
        from .namespaces.support import AsyncSupportAPI
        from .namespaces.system import AsyncSystemAPI
        from .namespaces.teams import AsyncTeamsAPI
        from .namespaces.tenants import AsyncTenantsAPI
        from .namespaces.threat_intel import AsyncThreatIntelAPI
        from .namespaces.tournaments import AsyncTournamentsAPI
        from .namespaces.training import AsyncTrainingAPI
        from .namespaces.uncertainty import AsyncUncertaintyAPI
        from .namespaces.unified_inbox import AsyncUnifiedInboxAPI
        from .namespaces.usage import AsyncUsageAPI
        from .namespaces.verification import AsyncVerificationAPI
        from .namespaces.verticals import AsyncVerticalsAPI
        from .namespaces.voice import AsyncVoiceAPI
        from .namespaces.webhooks import AsyncWebhooksAPI
        from .namespaces.workflow_templates import AsyncWorkflowTemplatesAPI
        from .namespaces.workflows import AsyncWorkflowsAPI
        from .namespaces.workspaces import AsyncWorkspacesAPI
        from .namespaces.youtube import AsyncYouTubeAPI

        self.a2a = AsyncA2AAPI(self)
        self.accounting = AsyncAccountingAPI(self)
        self.admin = AsyncAdminAPI(self)
        self.advertising = AsyncAdvertisingAPI(self)
        self.agent_selection = AsyncAgentSelectionAPI(self)
        self.agents = AsyncAgentsAPI(self)
        self.analytics = AsyncAnalyticsAPI(self)
        self.ap_automation = AsyncAPAutomationAPI(self)
        self.approvals = AsyncApprovalsAPI(self)
        self.ar_automation = AsyncARAutomationAPI(self)
        self.audio = AsyncAudioAPI(self)
        self.audit = AsyncAuditAPI(self)
        self.audit_trail = AsyncAuditTrailAPI(self)
        self.auditing = AsyncAuditingAPI(self)
        self.auth = AsyncAuthAPI(self)
        self.backups = AsyncBackupsAPI(self)
        self.batch = AsyncBatchAPI(self)
        self.belief = AsyncBeliefAPI(self)
        self.bots = AsyncBotsAPI(self)
        self.billing = AsyncBillingAPI(self)
        self.budgets = AsyncBudgetsAPI(self)
        self.blockchain = AsyncBlockchainAPI(self)
        self.calibration = AsyncCalibrationAPI(self)
        self.canvas = AsyncCanvasAPI(self)
        self.chat = AsyncChatAPI(self)
        self.checkpoints = AsyncCheckpointsAPI(self)
        self.classify = AsyncClassifyAPI(self)
        self.code_review = AsyncCodeReviewAPI(self)
        self.codebase = AsyncCodebaseAPI(self)
        self.compliance = AsyncComplianceAPI(self)
        self.computer_use = AsyncComputerUseAPI(self)
        self.connectors = AsyncConnectorsAPI(self)
        self.consensus = AsyncConsensusAPI(self)
        self.control_plane = AsyncControlPlaneAPI(self)
        self.cost_management = AsyncCostManagementAPI(self)
        self.critiques = AsyncCritiquesAPI(self)
        self.cross_pollination = AsyncCrossPollinationAPI(self)
        self.dashboard = AsyncDashboardAPI(self)
        self.debates = AsyncDebatesAPI(self)
        self.decisions = AsyncDecisionsAPI(self)
        self.deliberations = AsyncDeliberationsAPI(self)
        self.dependency_analysis = AsyncDependencyAnalysisAPI(self)
        self.devices = AsyncDevicesAPI(self)
        self.disaster_recovery = AsyncDisasterRecoveryAPI(self)
        self.documents = AsyncDocumentsAPI(self)
        self.email_debate = AsyncEmailDebateAPI(self)
        self.email_services = AsyncEmailServicesAPI(self)
        self.evaluation = AsyncEvaluationAPI(self)
        self.evolution = AsyncEvolutionAPI(self)
        self.expenses = AsyncExpensesAPI(self)
        self.explainability = AsyncExplainabilityAPI(self)
        self.facts = AsyncFactsAPI(self)
        self.feedback = AsyncFeedbackAPI(self)
        self.flips = AsyncFlipsAPI(self)
        self.gateway = AsyncGatewayAPI(self)
        self.gauntlet = AsyncGauntletAPI(self)
        self.genesis = AsyncGenesisAPI(self)
        self.gmail = AsyncGmailAPI(self)
        self.health = AsyncHealthAPI(self)
        self.hybrid_debates = AsyncHybridDebatesAPI(self)
        self.inbox_command = AsyncInboxCommandAPI(self)
        self.index = AsyncIndexAPI(self)
        self.integrations = AsyncIntegrationsAPI(self)
        self.insights = AsyncInsightsAPI(self)
        self.invoice_processing = AsyncInvoiceProcessingAPI(self)
        self.knowledge = AsyncKnowledgeAPI(self)
        self.knowledge_chat = AsyncKnowledgeChatAPI(self)
        self.marketplace = AsyncMarketplaceAPI(self)
        self.memory = AsyncMemoryAPI(self)
        self.matches = AsyncMatchesAPI(self)
        self.media = AsyncMediaAPI(self)
        self.metrics = AsyncMetricsAPI(self)
        self.ml = AsyncMLAPI(self)
        self.monitoring = AsyncMonitoringAPI(self)
        self.nomic = AsyncNomicAPI(self)
        self.notifications = AsyncNotificationsAPI(self)
        self.oauth_wizard = AsyncOAuthWizardAPI(self)
        self.openclaw = AsyncOpenclawAPI(self)
        self.openapi = AsyncOpenApiAPI(self)
        self.orchestration = AsyncOrchestrationAPI(self)
        self.onboarding = AsyncOnboardingAPI(self)
        self.organizations = AsyncOrganizationsAPI(self)
        self.outlook = AsyncOutlookAPI(self)
        self.partner = AsyncPartnerAPI(self)
        self.payments = AsyncPaymentsAPI(self)
        self.plugins = AsyncPluginsAPI(self)
        self.podcast = AsyncPodcastAPI(self)
        self.policies = AsyncPoliciesAPI(self)
        self.privacy = AsyncPrivacyAPI(self)
        self.probes = AsyncProbesAPI(self)
        self.pulse = AsyncPulseAPI(self)
        self.queue = AsyncQueueAPI(self)
        self.reconciliation = AsyncReconciliationAPI(self)
        self.ranking = AsyncRankingAPI(self)
        self.rbac = AsyncRBACAPI(self)
        self.receipts = AsyncReceiptsAPI(self)
        self.reputation = AsyncReputationAPI(self)
        self.reviews = AsyncReviewsAPI(self)
        self.relationships = AsyncRelationshipsAPI(self)
        self.replays = AsyncReplaysAPI(self)
        self.repository = AsyncRepositoryAPI(self)
        self.rlm = AsyncRLMAPI(self)
        self.routing = AsyncRoutingAPI(self)
        self.scim = AsyncSCIMAPI(self)
        self.security = AsyncSecurityAPI(self)
        self.slo = AsyncSLOAPI(self)
        self.sme = AsyncSMEAPI(self)
        self.social = AsyncSocialAPI(self)
        self.sso = AsyncSSOAPI(self)
        self.support = AsyncSupportAPI(self)
        self.system = AsyncSystemAPI(self)
        self.teams = AsyncTeamsAPI(self)
        self.tenants = AsyncTenantsAPI(self)
        self.training = AsyncTrainingAPI(self)
        self.threat_intel = AsyncThreatIntelAPI(self)
        self.tournaments = AsyncTournamentsAPI(self)
        self.uncertainty = AsyncUncertaintyAPI(self)
        self.unified_inbox = AsyncUnifiedInboxAPI(self)
        self.usage = AsyncUsageAPI(self)
        self.verification = AsyncVerificationAPI(self)
        self.verticals = AsyncVerticalsAPI(self)
        self.voice = AsyncVoiceAPI(self)
        self.webhooks = AsyncWebhooksAPI(self)
        self.workflow_templates = AsyncWorkflowTemplatesAPI(self)
        self.workflows = AsyncWorkflowsAPI(self)
        self.workspaces = AsyncWorkspacesAPI(self)
        self.youtube = AsyncYouTubeAPI(self)

    async def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make an async HTTP request to the Aragora API.

        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            json: JSON body
            headers: Additional headers

        Returns:
            Parsed JSON response as a dictionary.
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
                        return cast(dict[str, Any], response.json())
                    return {}

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

            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Use server-specified retry delay if available
                    delay = e.retry_after if e.retry_after else self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                raise

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
    ) -> dict[str, Any]:
        """Backward-compatible alias for request()."""
        return await self.request(method, path, params=params, json=json, headers=headers)

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses."""
        try:
            body = response.json()
            message = body.get("error", body.get("message", response.text))
            error_code = body.get("code")
            trace_id = body.get("trace_id")
        except (ValueError, AttributeError):
            body = None
            message = response.text or "Unknown error"
            error_code = None
            trace_id = None

        status = response.status_code

        if status == 401:
            raise AuthenticationError(
                message, error_code=error_code, trace_id=trace_id, response_body=body
            )
        elif status == 403:
            raise AuthorizationError(
                message, error_code=error_code, trace_id=trace_id, response_body=body
            )
        elif status == 404:
            raise NotFoundError(
                message, error_code=error_code, trace_id=trace_id, response_body=body
            )
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message,
                retry_after=int(retry_after) if retry_after else None,
                error_code=error_code,
                trace_id=trace_id,
                response_body=body,
            )
        elif status == 400:
            errors = body.get("errors") if body else None
            raise ValidationError(
                message,
                errors=errors,
                error_code=error_code,
                trace_id=trace_id,
                response_body=body,
            )
        elif status >= 500:
            raise ServerError(
                message,
                status_code=status,
                error_code=error_code,
                trace_id=trace_id,
                response_body=body,
            )
        else:
            raise AragoraError(
                message,
                status_code=status,
                error_code=error_code,
                trace_id=trace_id,
                response_body=body,
            )

    @property
    def stream(self) -> AragoraWebSocket:
        """Lazy-initialized WebSocket client for real-time event streaming."""
        if self._stream is None:
            from .websocket import AragoraWebSocket

            self._stream = AragoraWebSocket(
                base_url=self.base_url,
                api_key=self.api_key,
                ws_url=self.ws_url,
            )
        return self._stream

    async def close(self) -> None:
        """Close the HTTP client and any open WebSocket connection."""
        if self._stream is not None:
            await self._stream.close()
            self._stream = None
        await self._client.aclose()

    async def __aenter__(self) -> AragoraAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
