"""Marketplace data models - enums and dataclasses.

Contains TemplateCategory, DeploymentStatus enums and
TemplateMetadata, TemplateDeployment, TemplateRating dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TemplateCategory(Enum):
    """Template categories for industry verticals."""

    ACCOUNTING = "accounting"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    SOFTWARE = "software"
    REGULATORY = "regulatory"
    ACADEMIC = "academic"
    FINANCE = "finance"
    GENERAL = "general"
    DEVOPS = "devops"
    MARKETING = "marketing"


class DeploymentStatus(Enum):
    """Deployment status."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class TemplateMetadata:
    """Template metadata for marketplace listing."""

    id: str
    name: str
    description: str
    version: str
    category: TemplateCategory
    tags: list[str] = field(default_factory=list)
    icon: str = "document"
    author: str = "Aragora"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    steps_count: int = 0
    has_debate: bool = False
    has_human_checkpoint: bool = False
    estimated_duration: str = "varies"
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value,
            "tags": self.tags,
            "icon": self.icon,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "downloads": self.downloads,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "steps_count": self.steps_count,
            "has_debate": self.has_debate,
            "has_human_checkpoint": self.has_human_checkpoint,
            "estimated_duration": self.estimated_duration,
        }


@dataclass
class TemplateDeployment:
    """Record of a deployed template."""

    id: str
    template_id: str
    tenant_id: str
    name: str
    status: DeploymentStatus
    config: dict[str, Any] = field(default_factory=dict)
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: datetime | None = None
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status.value,
            "config": self.config,
            "deployed_at": self.deployed_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
        }


@dataclass
class TemplateRating:
    """Template rating from a user."""

    id: str
    template_id: str
    tenant_id: str
    user_id: str
    rating: int  # 1-5
    review: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "rating": self.rating,
            "review": self.review,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Category Information
# =============================================================================

CATEGORY_INFO = {
    TemplateCategory.ACCOUNTING: {
        "name": "Accounting & Finance",
        "description": "Invoice processing, expense management, financial audits, and QBO integration",
        "icon": "calculator",
        "color": "#4299e1",
    },
    TemplateCategory.LEGAL: {
        "name": "Legal",
        "description": "Contract review, due diligence, compliance checking, and document analysis",
        "icon": "scale",
        "color": "#9f7aea",
    },
    TemplateCategory.HEALTHCARE: {
        "name": "Healthcare",
        "description": "HIPAA compliance, clinical reviews, patient data processing",
        "icon": "heart",
        "color": "#f56565",
    },
    TemplateCategory.SOFTWARE: {
        "name": "Software Development",
        "description": "Code review, security audits, bug detection, and CI/CD automation",
        "icon": "code",
        "color": "#48bb78",
    },
    TemplateCategory.REGULATORY: {
        "name": "Regulatory Compliance",
        "description": "SOC 2, GDPR, SOX compliance assessments and audit preparation",
        "icon": "shield",
        "color": "#ed8936",
    },
    TemplateCategory.ACADEMIC: {
        "name": "Academic & Research",
        "description": "Citation verification, research validation, peer review workflows",
        "icon": "book",
        "color": "#38b2ac",
    },
    TemplateCategory.FINANCE: {
        "name": "Investment & Finance",
        "description": "Investment analysis, risk assessment, portfolio management",
        "icon": "trending-up",
        "color": "#667eea",
    },
    TemplateCategory.GENERAL: {
        "name": "General",
        "description": "Multi-purpose research and analysis workflows",
        "icon": "folder",
        "color": "#718096",
    },
    TemplateCategory.DEVOPS: {
        "name": "DevOps & IT",
        "description": "Infrastructure automation, incident response, monitoring",
        "icon": "server",
        "color": "#2d3748",
    },
    TemplateCategory.MARKETING: {
        "name": "Marketing",
        "description": "Content strategy, campaign analysis, market research",
        "icon": "megaphone",
        "color": "#d53f8c",
    },
}
