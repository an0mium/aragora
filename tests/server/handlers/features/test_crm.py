"""
Tests for CRM Integration Handler.

Tests cover basic dataclass creation and platform configuration.
"""

import pytest
from datetime import datetime

from aragora.server.handlers.features.crm import (
    CRMHandler,
    SUPPORTED_PLATFORMS,
    UnifiedContact,
    UnifiedCompany,
    UnifiedDeal,
)


class TestSupportedPlatforms:
    """Tests for CRM platform configuration."""

    def test_platforms_defined(self):
        """Test that CRM platforms are configured."""
        assert len(SUPPORTED_PLATFORMS) > 0

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config
            assert "features" in config


class TestUnifiedContact:
    """Tests for UnifiedContact dataclass."""

    def test_contact_creation(self):
        """Test creating a unified contact."""
        contact = UnifiedContact(
            id="contact_123",
            platform="salesforce",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            phone="+1234567890",
            company="Acme Inc",
            job_title="CEO",
            lifecycle_stage="customer",
            lead_status=None,
            owner_id="user_1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert contact.id == "contact_123"
        assert contact.email == "test@example.com"
        assert contact.company == "Acme Inc"

    def test_contact_to_dict(self):
        """Test contact serialization."""
        contact = UnifiedContact(
            id="contact_456",
            platform="hubspot",
            email="jane@example.com",
            first_name="Jane",
            last_name="Smith",
            phone=None,
            company=None,
            job_title=None,
            lifecycle_stage=None,
            lead_status=None,
            owner_id=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            properties={"source": "website"},
        )

        data = contact.to_dict()
        assert data["id"] == "contact_456"
        assert data["email"] == "jane@example.com"
        assert data["properties"]["source"] == "website"


class TestUnifiedCompany:
    """Tests for UnifiedCompany dataclass."""

    def test_company_creation(self):
        """Test creating a unified company."""
        company = UnifiedCompany(
            id="company_123",
            platform="hubspot",
            name="Acme Corporation",
            domain="acme.com",
            industry="Technology",
            employee_count=500,
            annual_revenue=50000000.0,
            owner_id="user_1",
            created_at=datetime.now(),
        )

        assert company.id == "company_123"
        assert company.name == "Acme Corporation"
        assert company.employee_count == 500

    def test_company_to_dict(self):
        """Test company serialization."""
        company = UnifiedCompany(
            id="company_456",
            platform="salesforce",
            name="Test Corp",
            domain="test.com",
            industry=None,
            employee_count=None,
            annual_revenue=None,
            owner_id=None,
            created_at=datetime.now(),
        )

        data = company.to_dict()
        assert data["id"] == "company_456"
        assert data["name"] == "Test Corp"


class TestUnifiedDeal:
    """Tests for UnifiedDeal dataclass."""

    def test_deal_creation(self):
        """Test creating a unified deal."""
        deal = UnifiedDeal(
            id="deal_123",
            platform="salesforce",
            name="Big Enterprise Deal",
            amount=100000.0,
            stage="negotiation",
            pipeline="Enterprise",
            close_date=datetime.now(),
            probability=75.0,
            contact_ids=["contact_123"],
            company_id="company_123",
            owner_id="user_1",
            created_at=datetime.now(),
        )

        assert deal.id == "deal_123"
        assert deal.amount == 100000.0
        assert deal.probability == 75.0


class TestCRMHandler:
    """Tests for CRMHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = CRMHandler(server_context={})
        assert handler is not None

    def test_handler_has_routes(self):
        """Test that handler has route definitions."""
        handler = CRMHandler(server_context={})
        # Handler should have methods for handling requests
        assert hasattr(handler, "handle_get") or hasattr(handler, "handle_request")
