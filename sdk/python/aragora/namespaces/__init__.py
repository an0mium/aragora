"""
Aragora SDK Namespace APIs

Provides namespaced access to Aragora API endpoints.
"""

from .agents import AgentsAPI, AsyncAgentsAPI
from .debates import AsyncDebatesAPI, DebatesAPI
from .workflows import AsyncWorkflowsAPI, WorkflowsAPI

__all__ = [
    "DebatesAPI",
    "AsyncDebatesAPI",
    "AgentsAPI",
    "AsyncAgentsAPI",
    "WorkflowsAPI",
    "AsyncWorkflowsAPI",
]
