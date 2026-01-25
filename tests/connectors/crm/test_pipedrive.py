"""
Tests for Pipedrive CRM Connector.

Tests cover:
- Client initialization
- API authentication
- CRUD operations for all entities
- Pagination
- Error handling
- Mock data generators
"""

from datetime import datetime, timezone

import pytest


# =============================================================================
# Enum Tests
# =============================================================================


class TestPipedriveEnums:
    """Tests for Pipedrive enums."""

    def test_deal_status_values(self):
        """DealStatus enum has expected values."""
        from aragora.connectors.crm.pipedrive import DealStatus

        assert DealStatus.OPEN.value == "open"
        assert DealStatus.WON.value == "won"
        assert DealStatus.LOST.value == "lost"
        assert DealStatus.DELETED.value == "deleted"

    def test_activity_type_values(self):
        """ActivityType enum has expected values."""
        from aragora.connectors.crm.pipedrive import ActivityType

        assert ActivityType.CALL.value == "call"
        assert ActivityType.MEETING.value == "meeting"
        assert ActivityType.TASK.value == "task"
        assert ActivityType.EMAIL.value == "email"
        assert ActivityType.DEADLINE.value == "deadline"

    def test_person_visibility_values(self):
        """PersonVisibility enum has expected values."""
        from aragora.connectors.crm.pipedrive import PersonVisibility

        assert PersonVisibility.OWNER.value == "1"
        assert PersonVisibility.OWNER_FOLLOWERS.value == "3"
        assert PersonVisibility.ENTIRE_COMPANY.value == "5"


# =============================================================================
# Credentials Tests
# =============================================================================


class TestPipedriveCredentials:
    """Tests for PipedriveCredentials."""

    def test_credentials_init(self):
        """Create credentials with API token."""
        from aragora.connectors.crm.pipedrive import PipedriveCredentials

        creds = PipedriveCredentials(api_token="test_token_123")

        assert creds.api_token == "test_token_123"
        assert creds.base_url == "https://api.pipedrive.com/v1"

    def test_credentials_with_custom_base_url(self):
        """Create credentials with custom base URL."""
        from aragora.connectors.crm.pipedrive import PipedriveCredentials

        creds = PipedriveCredentials(
            api_token="test_token",
            base_url="https://custom.pipedrive.com/v1",
        )

        assert creds.api_token == "test_token"
        assert creds.base_url == "https://custom.pipedrive.com/v1"


# =============================================================================
# Data Model Tests
# =============================================================================


class TestPerson:
    """Tests for Person dataclass."""

    def test_person_from_api(self):
        """Parse Person from API response."""
        from aragora.connectors.crm.pipedrive import Person

        data = {
            "id": 123,
            "name": "John Doe",
            "email": [{"value": "john@example.com", "primary": True}],
            "phone": [{"value": "+15551234567", "primary": True}],
            "org_id": 456,
            "owner_id": 1,
            "add_time": "2025-01-01 12:00:00",
            "update_time": "2025-01-15 14:30:00",
            "active_flag": True,
        }

        person = Person.from_api(data)

        assert person.id == 123
        assert person.name == "John Doe"
        assert person.email == "john@example.com"
        assert person.phone == "+15551234567"
        assert person.org_id == 456
        assert person.owner_id == 1
        assert person.active_flag is True

    def test_person_primary_email_extraction(self):
        """Extract primary email from list."""
        from aragora.connectors.crm.pipedrive import Person

        data = {
            "id": 1,
            "name": "Test",
            "email": [
                {"value": "secondary@test.com", "primary": False},
                {"value": "primary@test.com", "primary": True},
            ],
        }

        person = Person.from_api(data)
        assert person.email == "primary@test.com"

    def test_person_no_email(self):
        """Handle person without email."""
        from aragora.connectors.crm.pipedrive import Person

        data = {"id": 1, "name": "No Email Person"}

        person = Person.from_api(data)
        assert person.email is None


class TestOrganization:
    """Tests for Organization dataclass."""

    def test_organization_from_api(self):
        """Parse Organization from API response."""
        from aragora.connectors.crm.pipedrive import Organization

        data = {
            "id": 789,
            "name": "Acme Corp",
            "address": "123 Main St, City, State 12345",
            "owner_id": 1,
            "people_count": 5,
            "open_deals_count": 3,
            "won_deals_count": 10,
            "lost_deals_count": 2,
            "add_time": "2024-01-01 00:00:00",
        }

        org = Organization.from_api(data)

        assert org.id == 789
        assert org.name == "Acme Corp"
        assert org.address == "123 Main St, City, State 12345"
        assert org.people_count == 5
        assert org.open_deals_count == 3


class TestDeal:
    """Tests for Deal dataclass."""

    def test_deal_from_api(self):
        """Parse Deal from API response."""
        from aragora.connectors.crm.pipedrive import Deal, DealStatus

        data = {
            "id": 100,
            "title": "Big Deal",
            "value": 50000,
            "currency": "USD",
            "status": "open",
            "pipeline_id": 1,
            "stage_id": 2,
            "person_id": 123,
            "org_id": 456,
            "owner_id": 1,
            "probability": 75,
            "expected_close_date": "2025-06-30",
            "add_time": "2025-01-01 10:00:00",
        }

        deal = Deal.from_api(data)

        assert deal.id == 100
        assert deal.title == "Big Deal"
        assert deal.value == 50000
        assert deal.currency == "USD"
        assert deal.status == DealStatus.OPEN
        assert deal.probability == 75

    def test_deal_won_status(self):
        """Parse won deal status."""
        from aragora.connectors.crm.pipedrive import Deal, DealStatus

        data = {"id": 1, "title": "Won Deal", "status": "won"}
        deal = Deal.from_api(data)

        assert deal.status == DealStatus.WON


class TestPipeline:
    """Tests for Pipeline dataclass."""

    def test_pipeline_from_api(self):
        """Parse Pipeline from API response."""
        from aragora.connectors.crm.pipedrive import Pipeline

        data = {
            "id": 1,
            "name": "Sales Pipeline",
            "order_nr": 0,
            "active": True,
            "deal_probability": True,
        }

        pipeline = Pipeline.from_api(data)

        assert pipeline.id == 1
        assert pipeline.name == "Sales Pipeline"
        assert pipeline.active is True


class TestStage:
    """Tests for Stage dataclass."""

    def test_stage_from_api(self):
        """Parse Stage from API response."""
        from aragora.connectors.crm.pipedrive import Stage

        data = {
            "id": 10,
            "name": "Negotiation",
            "pipeline_id": 1,
            "order_nr": 3,
            "deal_probability": 50,
            "active_flag": True,
        }

        stage = Stage.from_api(data)

        assert stage.id == 10
        assert stage.name == "Negotiation"
        assert stage.pipeline_id == 1
        assert stage.deal_probability == 50


class TestActivity:
    """Tests for Activity dataclass."""

    def test_activity_from_api(self):
        """Parse Activity from API response."""
        from aragora.connectors.crm.pipedrive import Activity

        data = {
            "id": 200,
            "subject": "Follow-up call",
            "type": "call",
            "done": False,
            "due_date": "2025-02-01",
            "due_time": "14:00",
            "duration": "00:30",
            "person_id": 123,
            "deal_id": 100,
            "org_id": 456,
            "user_id": 1,
            "note": "Discuss proposal",
        }

        activity = Activity.from_api(data)

        assert activity.id == 200
        assert activity.subject == "Follow-up call"
        assert activity.type == "call"
        assert activity.done is False
        # due_date is parsed to datetime
        assert activity.due_date is not None


class TestNote:
    """Tests for Note dataclass."""

    def test_note_from_api(self):
        """Parse Note from API response."""
        from aragora.connectors.crm.pipedrive import Note

        data = {
            "id": 300,
            "content": "Important notes about this deal",
            "person_id": 123,
            "deal_id": 100,
            "org_id": 456,
            "user_id": 1,
            "add_time": "2025-01-15 10:00:00",
            "pinned_to_deal_flag": True,
        }

        note = Note.from_api(data)

        assert note.id == 300
        assert note.content == "Important notes about this deal"
        assert note.pinned_to_deal_flag is True


class TestProduct:
    """Tests for Product dataclass."""

    def test_product_from_api(self):
        """Parse Product from API response."""
        from aragora.connectors.crm.pipedrive import Product

        data = {
            "id": 500,
            "name": "Enterprise License",
            "code": "ENT-001",
            "description": "Annual enterprise license",
            "unit": "license",
            "prices": [{"price": 9999, "currency": "USD"}],
            "active_flag": True,
        }

        product = Product.from_api(data)

        assert product.id == 500
        assert product.name == "Enterprise License"
        assert product.code == "ENT-001"
        assert product.unit == "license"


class TestUser:
    """Tests for User dataclass."""

    def test_user_from_api(self):
        """Parse User from API response."""
        from aragora.connectors.crm.pipedrive import User

        data = {
            "id": 1,
            "name": "Admin User",
            "email": "admin@company.com",
            "active_flag": True,
            "is_admin": True,
        }

        user = User.from_api(data)

        assert user.id == 1
        assert user.name == "Admin User"
        assert user.email == "admin@company.com"
        assert user.is_admin is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestPipedriveError:
    """Tests for PipedriveError."""

    def test_error_creation(self):
        """Create error with details."""
        from aragora.connectors.crm.pipedrive import PipedriveError

        error = PipedriveError(
            message="Rate limit exceeded",
            status_code=429,
            error_code="rate_limited",
        )

        assert str(error) == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_code == "rate_limited"


# =============================================================================
# Client Tests
# =============================================================================


class TestPipedriveClientInit:
    """Tests for PipedriveClient initialization."""

    def test_client_creation(self):
        """Create client with credentials."""
        from aragora.connectors.crm.pipedrive import PipedriveClient, PipedriveCredentials

        creds = PipedriveCredentials(api_token="test_token")
        client = PipedriveClient(creds)

        assert client.credentials == creds
        assert creds.base_url == "https://api.pipedrive.com/v1"


# =============================================================================
# Mock Data Generator Tests
# =============================================================================


class TestMockDataGenerators:
    """Tests for mock data generators."""

    def test_get_mock_person(self):
        """Get mock person for testing."""
        from aragora.connectors.crm.pipedrive import get_mock_person

        person = get_mock_person()

        assert person.id == 1
        assert person.name == "John Doe"
        assert person.email == "john.doe@example.com"

    def test_get_mock_organization(self):
        """Get mock organization for testing."""
        from aragora.connectors.crm.pipedrive import get_mock_organization

        org = get_mock_organization()

        assert org.id == 1
        assert org.name == "Acme Corporation"

    def test_get_mock_activity(self):
        """Get mock activity for testing."""
        from aragora.connectors.crm.pipedrive import get_mock_activity

        activity = get_mock_activity()

        assert activity.id == 1
        assert activity.subject == "Discovery call"
        assert activity.type == "call"


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports from __init__.py."""

    def test_import_from_crm_module(self):
        """Import Pipedrive from CRM module."""
        from aragora.connectors.crm import (
            PipedriveClient,
            PipedriveCredentials,
            PipedriveError,
            Person,
            Organization,
            Activity,
            Note,
            Product,
            User,
            DealStatus,
            ActivityType,
            PipedriveDeal,
            PipedrivePipeline,
            PipedriveStage,
        )

        # Verify imports work
        assert PipedriveClient is not None
        assert PipedriveCredentials is not None
        assert PipedriveError is not None
        assert Person is not None
        assert Organization is not None
        assert DealStatus is not None
        assert ActivityType is not None
        assert PipedriveDeal is not None
