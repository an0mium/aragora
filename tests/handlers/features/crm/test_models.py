"""Tests for CRM unified data models.

Covers all public classes, methods, and constants in
aragora.server.handlers.features.crm.models:

- SUPPORTED_PLATFORMS constant: keys, structure, platform metadata
- UnifiedContact: construction, to_dict(), full_name logic, datetime serialization
- UnifiedCompany: construction, to_dict(), datetime serialization
- UnifiedDeal: construction, to_dict(), default fields, datetime serialization
- Edge cases: None fields, empty strings, special characters, large values
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.server.handlers.features.crm.models import (
    SUPPORTED_PLATFORMS,
    UnifiedCompany,
    UnifiedContact,
    UnifiedDeal,
)


# =============================================================================
# SUPPORTED_PLATFORMS Tests
# =============================================================================


class TestSupportedPlatforms:
    """Tests for the SUPPORTED_PLATFORMS constant."""

    def test_contains_hubspot(self):
        assert "hubspot" in SUPPORTED_PLATFORMS

    def test_contains_salesforce(self):
        assert "salesforce" in SUPPORTED_PLATFORMS

    def test_contains_pipedrive(self):
        assert "pipedrive" in SUPPORTED_PLATFORMS

    def test_platform_count(self):
        assert len(SUPPORTED_PLATFORMS) == 3

    def test_hubspot_name(self):
        assert SUPPORTED_PLATFORMS["hubspot"]["name"] == "HubSpot"

    def test_salesforce_name(self):
        assert SUPPORTED_PLATFORMS["salesforce"]["name"] == "Salesforce"

    def test_pipedrive_name(self):
        assert SUPPORTED_PLATFORMS["pipedrive"]["name"] == "Pipedrive"

    def test_hubspot_has_description(self):
        assert "description" in SUPPORTED_PLATFORMS["hubspot"]
        assert isinstance(SUPPORTED_PLATFORMS["hubspot"]["description"], str)
        assert len(SUPPORTED_PLATFORMS["hubspot"]["description"]) > 0

    def test_hubspot_features(self):
        features = SUPPORTED_PLATFORMS["hubspot"]["features"]
        assert "contacts" in features
        assert "companies" in features
        assert "deals" in features

    def test_salesforce_features(self):
        features = SUPPORTED_PLATFORMS["salesforce"]["features"]
        assert "contacts" in features
        assert "accounts" in features
        assert "opportunities" in features

    def test_pipedrive_features(self):
        features = SUPPORTED_PLATFORMS["pipedrive"]["features"]
        assert "contacts" in features
        assert "organizations" in features
        assert "deals" in features

    def test_salesforce_coming_soon(self):
        assert SUPPORTED_PLATFORMS["salesforce"].get("coming_soon") is True

    def test_pipedrive_coming_soon(self):
        assert SUPPORTED_PLATFORMS["pipedrive"].get("coming_soon") is True

    def test_hubspot_not_coming_soon(self):
        assert SUPPORTED_PLATFORMS["hubspot"].get("coming_soon") is None

    def test_all_platforms_have_name(self):
        for key, platform in SUPPORTED_PLATFORMS.items():
            assert "name" in platform, f"Platform {key} missing 'name'"

    def test_all_platforms_have_description(self):
        for key, platform in SUPPORTED_PLATFORMS.items():
            assert "description" in platform, f"Platform {key} missing 'description'"

    def test_all_platforms_have_features(self):
        for key, platform in SUPPORTED_PLATFORMS.items():
            assert "features" in platform, f"Platform {key} missing 'features'"
            assert isinstance(platform["features"], list)
            assert len(platform["features"]) > 0


# =============================================================================
# UnifiedContact Tests
# =============================================================================


class TestUnifiedContact:
    """Tests for the UnifiedContact dataclass."""

    @pytest.fixture()
    def now(self):
        return datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture()
    def full_contact(self, now):
        return UnifiedContact(
            id="c-001",
            platform="hubspot",
            email="alice@example.com",
            first_name="Alice",
            last_name="Smith",
            phone="+1-555-1234",
            company="Acme Corp",
            job_title="CTO",
            lifecycle_stage="customer",
            lead_status="qualified",
            owner_id="owner-1",
            created_at=now,
            updated_at=now,
            properties={"source": "web"},
        )

    @pytest.fixture()
    def minimal_contact(self):
        return UnifiedContact(
            id="c-min",
            platform="salesforce",
            email=None,
            first_name=None,
            last_name=None,
            phone=None,
            company=None,
            job_title=None,
            lifecycle_stage=None,
            lead_status=None,
            owner_id=None,
            created_at=None,
            updated_at=None,
        )

    def test_construction(self, full_contact):
        assert full_contact.id == "c-001"
        assert full_contact.platform == "hubspot"
        assert full_contact.email == "alice@example.com"

    def test_properties_default_factory(self):
        c = UnifiedContact(
            id="x", platform="hubspot", email=None, first_name=None,
            last_name=None, phone=None, company=None, job_title=None,
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=None, updated_at=None,
        )
        assert c.properties == {}
        assert isinstance(c.properties, dict)

    def test_properties_default_not_shared(self):
        """Each instance gets its own default dict."""
        c1 = UnifiedContact(
            id="a", platform="p", email=None, first_name=None,
            last_name=None, phone=None, company=None, job_title=None,
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=None, updated_at=None,
        )
        c2 = UnifiedContact(
            id="b", platform="p", email=None, first_name=None,
            last_name=None, phone=None, company=None, job_title=None,
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=None, updated_at=None,
        )
        c1.properties["key"] = "val"
        assert "key" not in c2.properties

    def test_to_dict_all_fields(self, full_contact, now):
        d = full_contact.to_dict()
        assert d["id"] == "c-001"
        assert d["platform"] == "hubspot"
        assert d["email"] == "alice@example.com"
        assert d["first_name"] == "Alice"
        assert d["last_name"] == "Smith"
        assert d["phone"] == "+1-555-1234"
        assert d["company"] == "Acme Corp"
        assert d["job_title"] == "CTO"
        assert d["lifecycle_stage"] == "customer"
        assert d["lead_status"] == "qualified"
        assert d["owner_id"] == "owner-1"
        assert d["created_at"] == now.isoformat()
        assert d["updated_at"] == now.isoformat()
        assert d["properties"] == {"source": "web"}

    def test_full_name_both_parts(self, full_contact):
        d = full_contact.to_dict()
        assert d["full_name"] == "Alice Smith"

    def test_full_name_first_only(self, now):
        c = UnifiedContact(
            id="x", platform="p", email=None, first_name="Alice",
            last_name=None, phone=None, company=None, job_title=None,
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=None, updated_at=None,
        )
        assert c.to_dict()["full_name"] == "Alice"

    def test_full_name_last_only(self):
        c = UnifiedContact(
            id="x", platform="p", email=None, first_name=None,
            last_name="Smith", phone=None, company=None, job_title=None,
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=None, updated_at=None,
        )
        assert c.to_dict()["full_name"] == "Smith"

    def test_full_name_none_when_both_absent(self, minimal_contact):
        d = minimal_contact.to_dict()
        assert d["full_name"] is None

    def test_full_name_empty_strings(self):
        c = UnifiedContact(
            id="x", platform="p", email=None, first_name="",
            last_name="", phone=None, company=None, job_title=None,
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=None, updated_at=None,
        )
        # Empty strings become '' via `or ''`, strip() yields '', which is falsy => None
        assert c.to_dict()["full_name"] is None

    def test_full_name_whitespace_first_name(self):
        """first_name with whitespace still gets stripped."""
        c = UnifiedContact(
            id="x", platform="p", email=None, first_name="  ",
            last_name=None, phone=None, company=None, job_title=None,
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=None, updated_at=None,
        )
        # "  " is truthy so it uses "  " directly, then strip() produces ""
        # The expression: f"{self.first_name or ''} {self.last_name or ''}".strip()
        # = f"   {''}" = "   " -> strip() -> "" -> or None => None
        assert c.to_dict()["full_name"] is None

    def test_to_dict_none_datetimes(self, minimal_contact):
        d = minimal_contact.to_dict()
        assert d["created_at"] is None
        assert d["updated_at"] is None

    def test_to_dict_returns_dict(self, full_contact):
        assert isinstance(full_contact.to_dict(), dict)

    def test_to_dict_keys(self, full_contact):
        expected_keys = {
            "id", "platform", "email", "first_name", "last_name",
            "full_name", "phone", "company", "job_title",
            "lifecycle_stage", "lead_status", "owner_id",
            "created_at", "updated_at", "properties",
        }
        assert set(full_contact.to_dict().keys()) == expected_keys

    def test_datetime_isoformat_output(self):
        dt = datetime(2025, 6, 15, 8, 30, 45, tzinfo=timezone.utc)
        c = UnifiedContact(
            id="x", platform="p", email=None, first_name=None,
            last_name=None, phone=None, company=None, job_title=None,
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=dt, updated_at=dt,
        )
        d = c.to_dict()
        assert d["created_at"] == "2025-06-15T08:30:45+00:00"
        assert d["updated_at"] == "2025-06-15T08:30:45+00:00"

    def test_special_characters_in_fields(self):
        c = UnifiedContact(
            id="c-special", platform="hubspot",
            email="o'malley@example.com",
            first_name="Jean-Pierre",
            last_name="O'Brien",
            phone=None, company='Acme "Corp"', job_title="VP & GM",
            lifecycle_stage=None, lead_status=None, owner_id=None,
            created_at=None, updated_at=None,
        )
        d = c.to_dict()
        assert d["email"] == "o'malley@example.com"
        assert d["full_name"] == "Jean-Pierre O'Brien"
        assert d["company"] == 'Acme "Corp"'
        assert d["job_title"] == "VP & GM"


# =============================================================================
# UnifiedCompany Tests
# =============================================================================


class TestUnifiedCompany:
    """Tests for the UnifiedCompany dataclass."""

    @pytest.fixture()
    def now(self):
        return datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture()
    def full_company(self, now):
        return UnifiedCompany(
            id="comp-001",
            platform="hubspot",
            name="Acme Corp",
            domain="acme.com",
            industry="Technology",
            employee_count=500,
            annual_revenue=10_000_000.0,
            owner_id="owner-1",
            created_at=now,
        )

    @pytest.fixture()
    def minimal_company(self):
        return UnifiedCompany(
            id="comp-min",
            platform="pipedrive",
            name="Minimal LLC",
            domain=None,
            industry=None,
            employee_count=None,
            annual_revenue=None,
            owner_id=None,
            created_at=None,
        )

    def test_construction(self, full_company):
        assert full_company.id == "comp-001"
        assert full_company.platform == "hubspot"
        assert full_company.name == "Acme Corp"
        assert full_company.employee_count == 500

    def test_to_dict_all_fields(self, full_company, now):
        d = full_company.to_dict()
        assert d["id"] == "comp-001"
        assert d["platform"] == "hubspot"
        assert d["name"] == "Acme Corp"
        assert d["domain"] == "acme.com"
        assert d["industry"] == "Technology"
        assert d["employee_count"] == 500
        assert d["annual_revenue"] == 10_000_000.0
        assert d["owner_id"] == "owner-1"
        assert d["created_at"] == now.isoformat()

    def test_to_dict_none_fields(self, minimal_company):
        d = minimal_company.to_dict()
        assert d["domain"] is None
        assert d["industry"] is None
        assert d["employee_count"] is None
        assert d["annual_revenue"] is None
        assert d["owner_id"] is None
        assert d["created_at"] is None

    def test_to_dict_keys(self, full_company):
        expected_keys = {
            "id", "platform", "name", "domain", "industry",
            "employee_count", "annual_revenue", "owner_id", "created_at",
        }
        assert set(full_company.to_dict().keys()) == expected_keys

    def test_to_dict_returns_dict(self, full_company):
        assert isinstance(full_company.to_dict(), dict)

    def test_zero_employees(self):
        c = UnifiedCompany(
            id="c", platform="p", name="Startup",
            domain=None, industry=None, employee_count=0,
            annual_revenue=0.0, owner_id=None, created_at=None,
        )
        d = c.to_dict()
        assert d["employee_count"] == 0
        assert d["annual_revenue"] == 0.0

    def test_large_revenue(self):
        c = UnifiedCompany(
            id="c", platform="p", name="MegaCorp",
            domain=None, industry=None, employee_count=100_000,
            annual_revenue=99_999_999_999.99, owner_id=None, created_at=None,
        )
        d = c.to_dict()
        assert d["annual_revenue"] == 99_999_999_999.99

    def test_datetime_isoformat(self):
        dt = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        c = UnifiedCompany(
            id="c", platform="p", name="N",
            domain=None, industry=None, employee_count=None,
            annual_revenue=None, owner_id=None, created_at=dt,
        )
        assert c.to_dict()["created_at"] == "2024-12-31T23:59:59+00:00"

    def test_naive_datetime(self):
        """Naive (no tz) datetimes still serialise without offset."""
        dt = datetime(2025, 3, 1, 10, 0, 0)
        c = UnifiedCompany(
            id="c", platform="p", name="N",
            domain=None, industry=None, employee_count=None,
            annual_revenue=None, owner_id=None, created_at=dt,
        )
        assert c.to_dict()["created_at"] == "2025-03-01T10:00:00"


# =============================================================================
# UnifiedDeal Tests
# =============================================================================


class TestUnifiedDeal:
    """Tests for the UnifiedDeal dataclass."""

    @pytest.fixture()
    def now(self):
        return datetime(2026, 6, 30, 18, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture()
    def full_deal(self, now):
        return UnifiedDeal(
            id="deal-001",
            platform="hubspot",
            name="Enterprise License",
            amount=50_000.0,
            stage="negotiation",
            pipeline="enterprise",
            close_date=now,
            probability=0.75,
            contact_ids=["c-001", "c-002"],
            company_id="comp-001",
            owner_id="owner-1",
            created_at=now,
        )

    @pytest.fixture()
    def minimal_deal(self):
        return UnifiedDeal(
            id="deal-min",
            platform="pipedrive",
            name="Small Deal",
            amount=None,
            stage="prospect",
            pipeline=None,
            close_date=None,
            probability=None,
        )

    def test_construction(self, full_deal):
        assert full_deal.id == "deal-001"
        assert full_deal.platform == "hubspot"
        assert full_deal.name == "Enterprise License"
        assert full_deal.amount == 50_000.0
        assert full_deal.stage == "negotiation"

    def test_default_contact_ids(self, minimal_deal):
        assert minimal_deal.contact_ids == []

    def test_default_contact_ids_not_shared(self):
        d1 = UnifiedDeal(
            id="a", platform="p", name="A", amount=None,
            stage="s", pipeline=None, close_date=None, probability=None,
        )
        d2 = UnifiedDeal(
            id="b", platform="p", name="B", amount=None,
            stage="s", pipeline=None, close_date=None, probability=None,
        )
        d1.contact_ids.append("c-1")
        assert "c-1" not in d2.contact_ids

    def test_default_company_id(self, minimal_deal):
        assert minimal_deal.company_id is None

    def test_default_owner_id(self, minimal_deal):
        assert minimal_deal.owner_id is None

    def test_default_created_at(self, minimal_deal):
        assert minimal_deal.created_at is None

    def test_to_dict_all_fields(self, full_deal, now):
        d = full_deal.to_dict()
        assert d["id"] == "deal-001"
        assert d["platform"] == "hubspot"
        assert d["name"] == "Enterprise License"
        assert d["amount"] == 50_000.0
        assert d["stage"] == "negotiation"
        assert d["pipeline"] == "enterprise"
        assert d["close_date"] == now.isoformat()
        assert d["probability"] == 0.75
        assert d["contact_ids"] == ["c-001", "c-002"]
        assert d["company_id"] == "comp-001"
        assert d["owner_id"] == "owner-1"
        assert d["created_at"] == now.isoformat()

    def test_to_dict_none_fields(self, minimal_deal):
        d = minimal_deal.to_dict()
        assert d["amount"] is None
        assert d["pipeline"] is None
        assert d["close_date"] is None
        assert d["probability"] is None
        assert d["company_id"] is None
        assert d["owner_id"] is None
        assert d["created_at"] is None

    def test_to_dict_keys(self, full_deal):
        expected_keys = {
            "id", "platform", "name", "amount", "stage", "pipeline",
            "close_date", "probability", "contact_ids", "company_id",
            "owner_id", "created_at",
        }
        assert set(full_deal.to_dict().keys()) == expected_keys

    def test_to_dict_returns_dict(self, full_deal):
        assert isinstance(full_deal.to_dict(), dict)

    def test_empty_contact_ids_in_dict(self, minimal_deal):
        d = minimal_deal.to_dict()
        assert d["contact_ids"] == []

    def test_zero_amount(self):
        deal = UnifiedDeal(
            id="d", platform="p", name="Free",
            amount=0.0, stage="closed", pipeline=None,
            close_date=None, probability=1.0,
        )
        assert deal.to_dict()["amount"] == 0.0

    def test_zero_probability(self):
        deal = UnifiedDeal(
            id="d", platform="p", name="Unlikely",
            amount=100.0, stage="lost", pipeline=None,
            close_date=None, probability=0.0,
        )
        assert deal.to_dict()["probability"] == 0.0

    def test_probability_one(self):
        deal = UnifiedDeal(
            id="d", platform="p", name="Sure Thing",
            amount=100.0, stage="closed-won", pipeline=None,
            close_date=None, probability=1.0,
        )
        assert deal.to_dict()["probability"] == 1.0

    def test_close_date_isoformat(self):
        dt = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        deal = UnifiedDeal(
            id="d", platform="p", name="EOY Deal",
            amount=None, stage="s", pipeline=None,
            close_date=dt, probability=None,
        )
        assert deal.to_dict()["close_date"] == "2026-12-31T23:59:59+00:00"


# =============================================================================
# Module-Level / __all__ Tests
# =============================================================================


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_contains_supported_platforms(self):
        from aragora.server.handlers.features.crm import models
        assert "SUPPORTED_PLATFORMS" in models.__all__

    def test_all_contains_unified_contact(self):
        from aragora.server.handlers.features.crm import models
        assert "UnifiedContact" in models.__all__

    def test_all_contains_unified_company(self):
        from aragora.server.handlers.features.crm import models
        assert "UnifiedCompany" in models.__all__

    def test_all_contains_unified_deal(self):
        from aragora.server.handlers.features.crm import models
        assert "UnifiedDeal" in models.__all__

    def test_all_length(self):
        from aragora.server.handlers.features.crm import models
        assert len(models.__all__) == 4
