"""
Aragora SDK Namespace APIs

Provides namespaced access to Aragora API endpoints.
"""

from .admin import AdminAPI, AsyncAdminAPI
from .agents import AgentsAPI, AsyncAgentsAPI
from .analytics import AnalyticsAPI, AsyncAnalyticsAPI
from .audit import AsyncAuditAPI, AuditAPI
from .auth import AsyncAuthAPI, AuthAPI
from .backups import AsyncBackupsAPI, BackupsAPI
from .belief import AsyncBeliefAPI, BeliefAPI
from .billing import AsyncBillingAPI, BillingAPI
from .budgets import AsyncBudgetsAPI, BudgetsAPI
from .consensus import AsyncConsensusAPI, ConsensusAPI
from .control_plane import AsyncControlPlaneAPI, ControlPlaneAPI
from .debates import AsyncDebatesAPI, DebatesAPI
from .decisions import AsyncDecisionsAPI, DecisionsAPI
from .documents import AsyncDocumentsAPI, DocumentsAPI
from .explainability import AsyncExplainabilityAPI, ExplainabilityAPI
from .gauntlet import AsyncGauntletAPI, GauntletAPI
from .genesis import AsyncGenesisAPI, GenesisAPI
from .health import AsyncHealthAPI, HealthAPI
from .integrations import AsyncIntegrationsAPI, IntegrationsAPI
from .knowledge import AsyncKnowledgeAPI, KnowledgeAPI
from .marketplace import AsyncMarketplaceAPI, MarketplaceAPI
from .memory import AsyncMemoryAPI, MemoryAPI
from .monitoring import AsyncMonitoringAPI, MonitoringAPI
from .notifications import AsyncNotificationsAPI, NotificationsAPI
from .onboarding import AsyncOnboardingAPI, OnboardingAPI
from .organizations import AsyncOrganizationsAPI, OrganizationsAPI
from .policies import AsyncPoliciesAPI, PoliciesAPI
from .pulse import AsyncPulseAPI, PulseAPI
from .ranking import AsyncRankingAPI, RankingAPI
from .rbac import RBACAPI, AsyncRBACAPI
from .receipts import AsyncReceiptsAPI, ReceiptsAPI
from .relationships import AsyncRelationshipsAPI, RelationshipsAPI
from .replays import AsyncReplaysAPI, ReplaysAPI
from .sme import SMEAPI, AsyncSMEAPI
from .teams import AsyncTeamsAPI, TeamsAPI
from .tenants import AsyncTenantsAPI, TenantsAPI
from .tournaments import AsyncTournamentsAPI, TournamentsAPI
from .usage import AsyncUsageAPI, UsageAPI
from .verification import AsyncVerificationAPI, VerificationAPI
from .webhooks import AsyncWebhooksAPI, WebhooksAPI
from .workflows import AsyncWorkflowsAPI, WorkflowsAPI
from .workspaces import AsyncWorkspacesAPI, WorkspacesAPI

__all__ = [
    "AdminAPI",
    "AsyncAdminAPI",
    "AgentsAPI",
    "AsyncAgentsAPI",
    "AnalyticsAPI",
    "AsyncAnalyticsAPI",
    "AuditAPI",
    "AsyncAuditAPI",
    "AuthAPI",
    "AsyncAuthAPI",
    "BackupsAPI",
    "AsyncBackupsAPI",
    "BeliefAPI",
    "AsyncBeliefAPI",
    "BillingAPI",
    "AsyncBillingAPI",
    "BudgetsAPI",
    "AsyncBudgetsAPI",
    "ConsensusAPI",
    "AsyncConsensusAPI",
    "ControlPlaneAPI",
    "AsyncControlPlaneAPI",
    "DebatesAPI",
    "AsyncDebatesAPI",
    "DecisionsAPI",
    "AsyncDecisionsAPI",
    "DocumentsAPI",
    "AsyncDocumentsAPI",
    "ExplainabilityAPI",
    "AsyncExplainabilityAPI",
    "GauntletAPI",
    "AsyncGauntletAPI",
    "GenesisAPI",
    "AsyncGenesisAPI",
    "HealthAPI",
    "AsyncHealthAPI",
    "IntegrationsAPI",
    "AsyncIntegrationsAPI",
    "KnowledgeAPI",
    "AsyncKnowledgeAPI",
    "MarketplaceAPI",
    "AsyncMarketplaceAPI",
    "MemoryAPI",
    "AsyncMemoryAPI",
    "MonitoringAPI",
    "AsyncMonitoringAPI",
    "NotificationsAPI",
    "AsyncNotificationsAPI",
    "OnboardingAPI",
    "AsyncOnboardingAPI",
    "OrganizationsAPI",
    "AsyncOrganizationsAPI",
    "PoliciesAPI",
    "AsyncPoliciesAPI",
    "PulseAPI",
    "AsyncPulseAPI",
    "RankingAPI",
    "AsyncRankingAPI",
    "RBACAPI",
    "AsyncRBACAPI",
    "ReceiptsAPI",
    "AsyncReceiptsAPI",
    "RelationshipsAPI",
    "AsyncRelationshipsAPI",
    "ReplaysAPI",
    "AsyncReplaysAPI",
    "SMEAPI",
    "AsyncSMEAPI",
    "TeamsAPI",
    "AsyncTeamsAPI",
    "TournamentsAPI",
    "AsyncTournamentsAPI",
    "TenantsAPI",
    "AsyncTenantsAPI",
    "UsageAPI",
    "AsyncUsageAPI",
    "VerificationAPI",
    "AsyncVerificationAPI",
    "WebhooksAPI",
    "AsyncWebhooksAPI",
    "WorkflowsAPI",
    "AsyncWorkflowsAPI",
    "WorkspacesAPI",
    "AsyncWorkspacesAPI",
]
