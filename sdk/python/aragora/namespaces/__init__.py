"""
Aragora SDK Namespace APIs

Provides namespaced access to Aragora API endpoints.
"""

from .a2a import A2AAPI, AsyncA2AAPI
from .accounting import AccountingAPI, AsyncAccountingAPI
from .admin import AdminAPI, AsyncAdminAPI
from .advertising import AdvertisingAPI, AsyncAdvertisingAPI
from .agents import AgentsAPI, AsyncAgentsAPI
from .analytics import AnalyticsAPI, AsyncAnalyticsAPI
from .ap_automation import APAutomationAPI, AsyncAPAutomationAPI
from .ar_automation import ARAutomationAPI, AsyncARAutomationAPI
from .audit import AsyncAuditAPI, AuditAPI
from .auth import AsyncAuthAPI, AuthAPI
from .backups import AsyncBackupsAPI, BackupsAPI
from .batch import AsyncBatchAPI, BatchAPI
from .belief import AsyncBeliefAPI, BeliefAPI
from .belief_network import AsyncBeliefNetworkAPI, BeliefNetworkAPI
from .billing import AsyncBillingAPI, BillingAPI
from .budgets import AsyncBudgetsAPI, BudgetsAPI
from .code_review import AsyncCodeReviewAPI, CodeReviewAPI
from .codebase import AsyncCodebaseAPI, CodebaseAPI
from .compliance import AsyncComplianceAPI, ComplianceAPI
from .connectors import AsyncConnectorsAPI, ConnectorsAPI
from .consensus import AsyncConsensusAPI, ConsensusAPI
from .control_plane import AsyncControlPlaneAPI, ControlPlaneAPI
from .cost_management import AsyncCostManagementAPI, CostManagementAPI
from .critiques import AsyncCritiquesAPI, CritiquesAPI
from .cross_pollination import AsyncCrossPollinationAPI, CrossPollinationAPI
from .dashboard import AsyncDashboardAPI, DashboardAPI
from .debates import AsyncDebatesAPI, DebatesAPI
from .decisions import AsyncDecisionsAPI, DecisionsAPI
from .deliberations import AsyncDeliberationsAPI, DeliberationsAPI
from .devices import AsyncDevicesAPI, DevicesAPI
from .documents import AsyncDocumentsAPI, DocumentsAPI
from .email_services import AsyncEmailServicesAPI, EmailServicesAPI
from .expenses import AsyncExpensesAPI, ExpensesAPI
from .explainability import AsyncExplainabilityAPI, ExplainabilityAPI
from .feedback import AsyncFeedbackAPI, FeedbackAPI
from .gauntlet import AsyncGauntletAPI, GauntletAPI
from .genesis import AsyncGenesisAPI, GenesisAPI
from .gmail import AsyncGmailAPI, GmailAPI
from .health import AsyncHealthAPI, HealthAPI
from .integrations import AsyncIntegrationsAPI, IntegrationsAPI
from .invoice_processing import AsyncInvoiceProcessingAPI, InvoiceProcessingAPI
from .knowledge import AsyncKnowledgeAPI, KnowledgeAPI
from .leaderboard import AsyncLeaderboardAPI, LeaderboardAPI
from .learning import AsyncLearningAPI, LearningAPI
from .marketplace import AsyncMarketplaceAPI, MarketplaceAPI
from .memory import AsyncMemoryAPI, MemoryAPI
from .metrics import AsyncMetricsAPI, MetricsAPI
from .monitoring import AsyncMonitoringAPI, MonitoringAPI
from .nomic import AsyncNomicAPI, NomicAPI
from .notifications import AsyncNotificationsAPI, NotificationsAPI
from .onboarding import AsyncOnboardingAPI, OnboardingAPI
from .organizations import AsyncOrganizationsAPI, OrganizationsAPI
from .outlook import AsyncOutlookAPI, OutlookAPI
from .payments import AsyncPaymentsAPI, PaymentsAPI
from .plugins import AsyncPluginsAPI, PluginsAPI
from .policies import AsyncPoliciesAPI, PoliciesAPI
from .privacy import AsyncPrivacyAPI, PrivacyAPI
from .pulse import AsyncPulseAPI, PulseAPI
from .ranking import AsyncRankingAPI, RankingAPI
from .rbac import RBACAPI, AsyncRBACAPI
from .receipts import AsyncReceiptsAPI, ReceiptsAPI
from .relationships import AsyncRelationshipsAPI, RelationshipsAPI
from .replays import AsyncReplaysAPI, ReplaysAPI
from .rlm import RLMAPI, AsyncRLMAPI
from .routing import AsyncRoutingAPI, RoutingAPI
from .sme import SMEAPI, AsyncSMEAPI
from .system import AsyncSystemAPI, SystemAPI
from .teams import AsyncTeamsAPI, TeamsAPI
from .tenants import AsyncTenantsAPI, TenantsAPI
from .threat_intel import AsyncThreatIntelAPI, ThreatIntelAPI
from .tournaments import AsyncTournamentsAPI, TournamentsAPI
from .training import AsyncTrainingAPI, TrainingAPI
from .transcription import AsyncTranscriptionAPI, TranscriptionAPI
from .unified_inbox import AsyncUnifiedInboxAPI, UnifiedInboxAPI
from .usage import AsyncUsageAPI, UsageAPI
from .verification import AsyncVerificationAPI, VerificationAPI
from .verticals import AsyncVerticalsAPI, VerticalsAPI
from .webhooks import AsyncWebhooksAPI, WebhooksAPI
from .workflows import AsyncWorkflowsAPI, WorkflowsAPI
from .workspaces import AsyncWorkspacesAPI, WorkspacesAPI

__all__ = [
    "A2AAPI",
    "AsyncA2AAPI",
    "AccountingAPI",
    "AsyncAccountingAPI",
    "AdvertisingAPI",
    "AsyncAdvertisingAPI",
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
    "BeliefNetworkAPI",
    "AsyncBeliefNetworkAPI",
    "BillingAPI",
    "AsyncBillingAPI",
    "BudgetsAPI",
    "AsyncBudgetsAPI",
    "CodeReviewAPI",
    "AsyncCodeReviewAPI",
    "CodebaseAPI",
    "AsyncCodebaseAPI",
    "ComplianceAPI",
    "AsyncComplianceAPI",
    "ConnectorsAPI",
    "AsyncConnectorsAPI",
    "ConsensusAPI",
    "AsyncConsensusAPI",
    "ControlPlaneAPI",
    "AsyncControlPlaneAPI",
    "CostManagementAPI",
    "AsyncCostManagementAPI",
    "CrossPollinationAPI",
    "AsyncCrossPollinationAPI",
    "CritiquesAPI",
    "AsyncCritiquesAPI",
    "DashboardAPI",
    "AsyncDashboardAPI",
    "DebatesAPI",
    "AsyncDebatesAPI",
    "DecisionsAPI",
    "AsyncDecisionsAPI",
    "DeliberationsAPI",
    "AsyncDeliberationsAPI",
    "DevicesAPI",
    "AsyncDevicesAPI",
    "DocumentsAPI",
    "AsyncDocumentsAPI",
    "EmailServicesAPI",
    "AsyncEmailServicesAPI",
    "ExpensesAPI",
    "AsyncExpensesAPI",
    "ExplainabilityAPI",
    "AsyncExplainabilityAPI",
    "FeedbackAPI",
    "AsyncFeedbackAPI",
    "GauntletAPI",
    "AsyncGauntletAPI",
    "GenesisAPI",
    "AsyncGenesisAPI",
    "GmailAPI",
    "AsyncGmailAPI",
    "HealthAPI",
    "AsyncHealthAPI",
    "IntegrationsAPI",
    "AsyncIntegrationsAPI",
    "InvoiceProcessingAPI",
    "AsyncInvoiceProcessingAPI",
    "KnowledgeAPI",
    "AsyncKnowledgeAPI",
    "LeaderboardAPI",
    "AsyncLeaderboardAPI",
    "LearningAPI",
    "AsyncLearningAPI",
    "MarketplaceAPI",
    "AsyncMarketplaceAPI",
    "MemoryAPI",
    "AsyncMemoryAPI",
    "MetricsAPI",
    "AsyncMetricsAPI",
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
    "OutlookAPI",
    "AsyncOutlookAPI",
    "PaymentsAPI",
    "AsyncPaymentsAPI",
    "PluginsAPI",
    "AsyncPluginsAPI",
    "PoliciesAPI",
    "AsyncPoliciesAPI",
    "PrivacyAPI",
    "AsyncPrivacyAPI",
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
    "SystemAPI",
    "AsyncSystemAPI",
    "TeamsAPI",
    "AsyncTeamsAPI",
    "TournamentsAPI",
    "AsyncTournamentsAPI",
    "TenantsAPI",
    "AsyncTenantsAPI",
    "ThreatIntelAPI",
    "AsyncThreatIntelAPI",
    "TrainingAPI",
    "AsyncTrainingAPI",
    "TranscriptionAPI",
    "AsyncTranscriptionAPI",
    "UnifiedInboxAPI",
    "AsyncUnifiedInboxAPI",
    "UsageAPI",
    "AsyncUsageAPI",
    "VerificationAPI",
    "AsyncVerificationAPI",
    "VerticalsAPI",
    "AsyncVerticalsAPI",
    "WebhooksAPI",
    "AsyncWebhooksAPI",
    "WorkflowsAPI",
    "AsyncWorkflowsAPI",
    "WorkspacesAPI",
    "AsyncWorkspacesAPI",
]
