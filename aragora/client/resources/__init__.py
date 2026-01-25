"""
Resource API modules for the Aragora client.

These modules contain the API interface classes that are used by AragoraClient.
"""

from .agents import AgentsAPI
from .analytics import AnalyticsAPI
from .audit import AuditAPI
from .auth import AuthAPI
from .billing import BillingAPI
from .codebase import CodebaseAPI
from .consensus import ConsensusAPI
from .cost_management import CostManagementAPI
from .debates import DebatesAPI
from .decisions import DecisionsAPI
from .documents import DocumentsAPI
from .explainability import ExplainabilityAPI
from .gauntlet import GauntletAPI
from .gmail import GmailAPI
from .graph_debates import GraphDebatesAPI
from .knowledge import KnowledgeAPI
from .leaderboard import LeaderboardAPI
from .matrix_debates import MatrixDebatesAPI
from .memory import MemoryAPI
from .notifications import NotificationsAPI
from .onboarding import OnboardingAPI
from .organizations import OrganizationsAPI
from .policies import PoliciesAPI
from .pulse import PulseAPI
from .rbac import RBACAPI
from .replay import ReplayAPI
from .system import SystemAPI
from .tenants import TenantsAPI
from .tournaments import TournamentsAPI
from .verification import VerificationAPI
from .workflows import WorkflowsAPI

__all__ = [
    "AgentsAPI",
    "AnalyticsAPI",
    "AuditAPI",
    "AuthAPI",
    "BillingAPI",
    "CodebaseAPI",
    "ConsensusAPI",
    "CostManagementAPI",
    "DebatesAPI",
    "DecisionsAPI",
    "DocumentsAPI",
    "ExplainabilityAPI",
    "GauntletAPI",
    "GmailAPI",
    "GraphDebatesAPI",
    "KnowledgeAPI",
    "LeaderboardAPI",
    "MatrixDebatesAPI",
    "MemoryAPI",
    "NotificationsAPI",
    "OnboardingAPI",
    "OrganizationsAPI",
    "PoliciesAPI",
    "PulseAPI",
    "RBACAPI",
    "ReplayAPI",
    "SystemAPI",
    "TenantsAPI",
    "TournamentsAPI",
    "VerificationAPI",
    "WorkflowsAPI",
]
