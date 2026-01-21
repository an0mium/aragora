"""
Aragora Agent Template Marketplace.

A marketplace for sharing and discovering agent templates,
debate protocols, and workflow configurations.
"""

from .models import (
    AgentTemplate,
    DebateTemplate,
    WorkflowTemplate,
    TemplateMetadata,
    TemplateRating,
    TemplateCategory,
)
from .registry import TemplateRegistry
from .client import MarketplaceClient

__all__ = [
    "AgentTemplate",
    "DebateTemplate",
    "WorkflowTemplate",
    "TemplateMetadata",
    "TemplateRating",
    "TemplateCategory",
    "TemplateRegistry",
    "MarketplaceClient",
]
