"""
Aragora SDK Namespace APIs

Provides namespaced access to Aragora API endpoints.
"""

from .admin import AdminAPI, AsyncAdminAPI
from .agents import AgentsAPI, AsyncAgentsAPI
from .analytics import AnalyticsAPI, AsyncAnalyticsAPI
from .audit import AsyncAuditAPI, AuditAPI
from .auth import AsyncAuthAPI, AuthAPI
from .consensus import AsyncConsensusAPI, ConsensusAPI
from .debates import AsyncDebatesAPI, DebatesAPI
from .explainability import AsyncExplainabilityAPI, ExplainabilityAPI
from .gauntlet import AsyncGauntletAPI, GauntletAPI
from .health import AsyncHealthAPI, HealthAPI
from .knowledge import AsyncKnowledgeAPI, KnowledgeAPI
from .memory import AsyncMemoryAPI, MemoryAPI
from .onboarding import AsyncOnboardingAPI, OnboardingAPI
from .rbac import RBACAPI, AsyncRBACAPI
from .receipts import AsyncReceiptsAPI, ReceiptsAPI
from .usage import AsyncUsageAPI, UsageAPI
from .verification import AsyncVerificationAPI, VerificationAPI
from .webhooks import AsyncWebhooksAPI, WebhooksAPI
from .workflows import AsyncWorkflowsAPI, WorkflowsAPI

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
    "HealthAPI",
    "AsyncHealthAPI",
    "ConsensusAPI",
    "AsyncConsensusAPI",
    "DebatesAPI",
    "AsyncDebatesAPI",
    "ExplainabilityAPI",
    "AsyncExplainabilityAPI",
    "GauntletAPI",
    "AsyncGauntletAPI",
    "KnowledgeAPI",
    "AsyncKnowledgeAPI",
    "MemoryAPI",
    "AsyncMemoryAPI",
    "OnboardingAPI",
    "AsyncOnboardingAPI",
    "RBACAPI",
    "AsyncRBACAPI",
    "ReceiptsAPI",
    "AsyncReceiptsAPI",
    "UsageAPI",
    "AsyncUsageAPI",
    "VerificationAPI",
    "AsyncVerificationAPI",
    "WebhooksAPI",
    "AsyncWebhooksAPI",
    "WorkflowsAPI",
    "AsyncWorkflowsAPI",
]
