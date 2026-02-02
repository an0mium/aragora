"""
CRM Models - Unified Data Models for CRM Entities.

This module provides unified data models for CRM entities across platforms:
- UnifiedContact: Contact representation
- UnifiedCompany: Company/Organization representation
- UnifiedDeal: Deal/Opportunity representation
- SUPPORTED_PLATFORMS: Platform metadata

Stability: STABLE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# =============================================================================
# Platform Configuration
# =============================================================================

SUPPORTED_PLATFORMS: dict[str, dict[str, Any]] = {
    "hubspot": {
        "name": "HubSpot",
        "description": "All-in-one CRM with marketing, sales, and service hubs",
        "features": ["contacts", "companies", "deals", "tickets", "marketing"],
    },
    "salesforce": {
        "name": "Salesforce",
        "description": "Enterprise CRM platform",
        "features": ["contacts", "accounts", "opportunities", "leads", "campaigns"],
        "coming_soon": True,
    },
    "pipedrive": {
        "name": "Pipedrive",
        "description": "Sales-focused CRM",
        "features": ["contacts", "organizations", "deals", "pipelines"],
        "coming_soon": True,
    },
}


# =============================================================================
# Unified Data Models
# =============================================================================


@dataclass
class UnifiedContact:
    """Unified contact representation across CRM platforms."""

    id: str
    platform: str
    email: str | None
    first_name: str | None
    last_name: str | None
    phone: str | None
    company: str | None
    job_title: str | None
    lifecycle_stage: str | None
    lead_status: str | None
    owner_id: str | None
    created_at: datetime | None
    updated_at: datetime | None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": f"{self.first_name or ''} {self.last_name or ''}".strip() or None,
            "phone": self.phone,
            "company": self.company,
            "job_title": self.job_title,
            "lifecycle_stage": self.lifecycle_stage,
            "lead_status": self.lead_status,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "properties": self.properties,
        }


@dataclass
class UnifiedCompany:
    """Unified company representation."""

    id: str
    platform: str
    name: str
    domain: str | None
    industry: str | None
    employee_count: int | None
    annual_revenue: float | None
    owner_id: str | None
    created_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "name": self.name,
            "domain": self.domain,
            "industry": self.industry,
            "employee_count": self.employee_count,
            "annual_revenue": self.annual_revenue,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class UnifiedDeal:
    """Unified deal/opportunity representation."""

    id: str
    platform: str
    name: str
    amount: float | None
    stage: str
    pipeline: str | None
    close_date: datetime | None
    probability: float | None
    contact_ids: list[str] = field(default_factory=list)
    company_id: str | None = None
    owner_id: str | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "name": self.name,
            "amount": self.amount,
            "stage": self.stage,
            "pipeline": self.pipeline,
            "close_date": self.close_date.isoformat() if self.close_date else None,
            "probability": self.probability,
            "contact_ids": self.contact_ids,
            "company_id": self.company_id,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


__all__ = [
    "SUPPORTED_PLATFORMS",
    "UnifiedContact",
    "UnifiedCompany",
    "UnifiedDeal",
]
