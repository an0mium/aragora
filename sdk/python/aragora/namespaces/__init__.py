"""
Aragora SDK Namespace APIs

Provides namespaced access to Aragora API endpoints.
"""

from .admin import AdminAPI, AsyncAdminAPI
from .agents import AgentsAPI, AsyncAgentsAPI
from .analytics import AnalyticsAPI, AsyncAnalyticsAPI
from .debates import AsyncDebatesAPI, DebatesAPI
from .onboarding import AsyncOnboardingAPI, OnboardingAPI
from .usage import AsyncUsageAPI, UsageAPI
from .workflows import AsyncWorkflowsAPI, WorkflowsAPI

__all__ = [
    "AdminAPI",
    "AsyncAdminAPI",
    "AgentsAPI",
    "AsyncAgentsAPI",
    "AnalyticsAPI",
    "AsyncAnalyticsAPI",
    "DebatesAPI",
    "AsyncDebatesAPI",
    "OnboardingAPI",
    "AsyncOnboardingAPI",
    "UsageAPI",
    "AsyncUsageAPI",
    "WorkflowsAPI",
    "AsyncWorkflowsAPI",
]
