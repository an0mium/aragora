"""
Aragora SDK Namespace APIs

Provides namespaced access to Aragora API endpoints.
"""

from .accounting import AccountingAPI, AsyncAccountingAPI
from .admin import AdminAPI, AsyncAdminAPI
from .agents import AgentsAPI, AsyncAgentsAPI
from .analytics import AnalyticsAPI, AsyncAnalyticsAPI
from .ap_automation import APAutomationAPI, AsyncAPAutomationAPI
from .ar_automation import ARAutomationAPI, AsyncARAutomationAPI
from .audit import AsyncAuditAPI, AuditAPI
from .auth import AsyncAuthAPI, AuthAPI
from .backups import AsyncBackupsAPI, BackupsAPI
from .batch import AsyncBatchAPI, BatchAPI
from .belief import AsyncBeliefAPI, BeliefAPI
from .billing import AsyncBillingAPI, BillingAPI
from .budgets import AsyncBudgetsAPI, BudgetsAPI
from .code_review import AsyncCodeReviewAPI, CodeReviewAPI
from .codebase import AsyncCodebaseAPI, CodebaseAPI
from .consensus import AsyncConsensusAPI, ConsensusAPI
from .control_plane import AsyncControlPlaneAPI, ControlPlaneAPI
from .cost_management import AsyncCostManagementAPI, CostManagementAPI
from .critiques import AsyncCritiquesAPI, CritiquesAPI
from .debates import AsyncDebatesAPI, DebatesAPI
from .decisions import AsyncDecisionsAPI, DecisionsAPI
from .documents import AsyncDocumentsAPI, DocumentsAPI
from .expenses import AsyncExpensesAPI, ExpensesAPI
from .explainability import AsyncExplainabilityAPI, ExplainabilityAPI
from .gauntlet import AsyncGauntletAPI, GauntletAPI
from .genesis import AsyncGenesisAPI, GenesisAPI
from .health import AsyncHealthAPI, HealthAPI
from .integrations import AsyncIntegrationsAPI, IntegrationsAPI
from .invoice_processing import AsyncInvoiceProcessingAPI, InvoiceProcessingAPI
from .knowledge import AsyncKnowledgeAPI, KnowledgeAPI
from .marketplace import AsyncMarketplaceAPI, MarketplaceAPI
from .memory import AsyncMemoryAPI, MemoryAPI
from .monitoring import AsyncMonitoringAPI, MonitoringAPI
from .nomic import AsyncNomicAPI, NomicAPI
from .notifications import AsyncNotificationsAPI, NotificationsAPI
from .onboarding import AsyncOnboardingAPI, OnboardingAPI
from .organizations import AsyncOrganizationsAPI, OrganizationsAPI
from .payments import AsyncPaymentsAPI, PaymentsAPI
from .policies import AsyncPoliciesAPI, PoliciesAPI
from .pulse import AsyncPulseAPI, PulseAPI
from .ranking import AsyncRankingAPI, RankingAPI
from .rbac import RBACAPI, AsyncRBACAPI
from .receipts import AsyncReceiptsAPI, ReceiptsAPI
from .relationships import AsyncRelationshipsAPI, RelationshipsAPI
from .replays import AsyncReplaysAPI, ReplaysAPI
from .rlm import RLMAPI, AsyncRLMAPI
from .routing import AsyncRoutingAPI, RoutingAPI
from .sme import SMEAPI, AsyncSMEAPI
from .teams import AsyncTeamsAPI, TeamsAPI
from .tenants import AsyncTenantsAPI, TenantsAPI
from .tournaments import AsyncTournamentsAPI, TournamentsAPI
from .training import AsyncTrainingAPI, TrainingAPI
from .usage import AsyncUsageAPI, UsageAPI
from .verification import AsyncVerificationAPI, VerificationAPI
from .webhooks import AsyncWebhooksAPI, WebhooksAPI
from .workflows import AsyncWorkflowsAPI, WorkflowsAPI
from .workspaces import AsyncWorkspacesAPI, WorkspacesAPI

__all__ = [
    "AccountingAPI",
    "AsyncAccountingAPI",
    "AdminAPI",
    "AsyncAdminAPI",
    "AgentsAPI",
    "AsyncAgentsAPI",
    "AnalyticsAPI",
    "AsyncAnalyticsAPI",
    "APAutomationAPI",
    "AsyncAPAutomationAPI",
    "ARAutomationAPI",
    "AsyncARAutomationAPI",
    "AuditAPI",
    "AsyncAuditAPI",
    "AuthAPI",
    "AsyncAuthAPI",
    "BackupsAPI",
    "AsyncBackupsAPI",
    "BatchAPI",
    "AsyncBatchAPI",
    "BeliefAPI",
    "AsyncBeliefAPI",
    "BillingAPI",
    "AsyncBillingAPI",
    "BudgetsAPI",
    "AsyncBudgetsAPI",
    "CodeReviewAPI",
    "AsyncCodeReviewAPI",
    "CodebaseAPI",
    "AsyncCodebaseAPI",
    "ConsensusAPI",
    "AsyncConsensusAPI",
    "ControlPlaneAPI",
    "AsyncControlPlaneAPI",
    "CostManagementAPI",
    "AsyncCostManagementAPI",
    "CritiquesAPI",
    "AsyncCritiquesAPI",
    "DebatesAPI",
    "AsyncDebatesAPI",
    "DecisionsAPI",
    "AsyncDecisionsAPI",
    "DocumentsAPI",
    "AsyncDocumentsAPI",
    "ExpensesAPI",
    "AsyncExpensesAPI",
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
    "InvoiceProcessingAPI",
    "AsyncInvoiceProcessingAPI",
    "KnowledgeAPI",
    "AsyncKnowledgeAPI",
    "MarketplaceAPI",
    "AsyncMarketplaceAPI",
    "MemoryAPI",
    "AsyncMemoryAPI",
    "MonitoringAPI",
    "AsyncMonitoringAPI",
    "NomicAPI",
    "AsyncNomicAPI",
    "NotificationsAPI",
    "AsyncNotificationsAPI",
    "OnboardingAPI",
    "AsyncOnboardingAPI",
    "OrganizationsAPI",
    "AsyncOrganizationsAPI",
    "PaymentsAPI",
    "AsyncPaymentsAPI",
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
    "RLMAPI",
    "AsyncRLMAPI",
    "RoutingAPI",
    "AsyncRoutingAPI",
    "SMEAPI",
    "AsyncSMEAPI",
    "TeamsAPI",
    "AsyncTeamsAPI",
    "TournamentsAPI",
    "AsyncTournamentsAPI",
    "TenantsAPI",
    "AsyncTenantsAPI",
    "TrainingAPI",
    "AsyncTrainingAPI",
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
