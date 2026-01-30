"""
Comprehensive tests for the Pipedrive CRM Connector.

Tests cover:
- Enum values (DealStatus, ActivityType, PersonVisibility)
- PipedriveCredentials dataclass and env loading
- Data models: Person, Organization, Pipeline, Stage, Deal, Activity, Note, Product, User
- Model serialization (from_api, to_api)
- PipedriveClient initialization and async context manager
- API request handling (_request with auth, error responses, success=false)
- Persons CRUD operations (get, create, update, delete, search)
- Organizations CRUD operations
- Deals CRUD operations (including move_deal_to_stage, mark_deal_won, mark_deal_lost)
- Pipelines and Stages read operations
- Activities CRUD operations (including mark_activity_done)
- Notes CRUD operations
- Products CRUD operations (including search)
- Users read operations
- Deal-Person/Organization association operations
- Helper functions (_parse_datetime, _parse_date)
- Error handling (HTTP errors, success=false responses, uninitialized client)
- Edge cases (empty responses, missing fields, malformed data)
- Mock data generation helpers
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.crm.pipedrive import (
    Activity,
    ActivityType,
    Deal,
    DealStatus,
    Note,
    Organization,
    Person,
    PersonVisibility,
    Pipeline,
    PipedriveClient,
    PipedriveCredentials,
    PipedriveError,
    Product,
    Stage,
    User,
    _parse_date,
    _parse_datetime,
    get_mock_activity,
    get_mock_deal,
    get_mock_organization,
    get_mock_person,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credentials():
    """Standard test credentials."""
    return PipedriveCredentials(api_token="test-api-token-123")


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def pipedrive_client(credentials, mock_httpx_client):
    """Create a PipedriveClient with a mock HTTP client."""
    client = PipedriveClient(credentials)
    client._client = mock_httpx_client
    return client


def _make_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = b'{"data": "test"}'
    return resp


# =============================================================================
# Enum Tests
# =============================================================================


class TestDealStatus:
    """Tests for DealStatus enum."""

    def test_deal_status_values(self):
        assert DealStatus.OPEN.value == "open"
        assert DealStatus.WON.value == "won"
        assert DealStatus.LOST.value == "lost"
        assert DealStatus.DELETED.value == "deleted"

    def test_deal_status_is_str(self):
        assert isinstance(DealStatus.OPEN, str)
        assert DealStatus.OPEN == "open"

    def test_deal_status_from_string(self):
        assert DealStatus("won") == DealStatus.WON


class TestActivityType:
    """Tests for ActivityType enum."""

    def test_activity_type_values(self):
        assert ActivityType.CALL.value == "call"
        assert ActivityType.MEETING.value == "meeting"
        assert ActivityType.TASK.value == "task"
        assert ActivityType.DEADLINE.value == "deadline"
        assert ActivityType.EMAIL.value == "email"
        assert ActivityType.LUNCH.value == "lunch"

    def test_activity_type_is_str(self):
        assert isinstance(ActivityType.CALL, str)


class TestPersonVisibility:
    """Tests for PersonVisibility enum."""

    def test_visibility_values(self):
        assert PersonVisibility.OWNER.value == "1"
        assert PersonVisibility.OWNER_FOLLOWERS.value == "3"
        assert PersonVisibility.ENTIRE_COMPANY.value == "5"

    def test_visibility_is_str(self):
        assert isinstance(PersonVisibility.ENTIRE_COMPANY, str)


# =============================================================================
# Credentials Tests
# =============================================================================


class TestPipedriveCredentials:
    """Tests for PipedriveCredentials dataclass."""

    def test_basic_construction(self):
        creds = PipedriveCredentials(api_token="tok_abc")
        assert creds.api_token == "tok_abc"
        assert creds.base_url == "https://api.pipedrive.com/v1"

    def test_custom_base_url(self):
        creds = PipedriveCredentials(
            api_token="tok",
            base_url="https://custom.pipedrive.com/v1",
        )
        assert creds.base_url == "https://custom.pipedrive.com/v1"

    @patch.dict(
        "os.environ",
        {
            "PIPEDRIVE_API_TOKEN": "env_token_123",
            "PIPEDRIVE_BASE_URL": "https://env.pipedrive.com/v1",
        },
    )
    def test_from_env(self):
        creds = PipedriveCredentials.from_env()
        assert creds.api_token == "env_token_123"
        assert creds.base_url == "https://env.pipedrive.com/v1"

    @patch.dict(
        "os.environ",
        {"PIPEDRIVE_API_TOKEN": "env_token_123"},
    )
    def test_from_env_default_base_url(self):
        creds = PipedriveCredentials.from_env()
        assert creds.base_url == "https://api.pipedrive.com/v1"

    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_missing_token_raises(self):
        with pytest.raises(ValueError, match="Missing PIPEDRIVE_API_TOKEN"):
            PipedriveCredentials.from_env()

    @patch.dict(
        "os.environ",
        {"CUSTOM_API_TOKEN": "custom_tok"},
    )
    def test_from_env_custom_prefix(self):
        creds = PipedriveCredentials.from_env(prefix="CUSTOM_")
        assert creds.api_token == "custom_tok"


# =============================================================================
# Error Tests
# =============================================================================


class TestPipedriveError:
    """Tests for PipedriveError exception."""

    def test_error_basic(self):
        err = PipedriveError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.status_code is None
        assert err.error_code is None

    def test_error_with_status(self):
        err = PipedriveError("Not found", status_code=404, error_code="not_found")
        assert err.status_code == 404
        assert err.error_code == "not_found"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for _parse_datetime and _parse_date."""

    def test_parse_datetime_none(self):
        assert _parse_datetime(None) is None

    def test_parse_datetime_empty(self):
        assert _parse_datetime("") is None

    def test_parse_datetime_standard_format(self):
        result = _parse_datetime("2023-06-15 10:30:00")
        assert result is not None
        assert result.year == 2023
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_parse_datetime_iso_format(self):
        result = _parse_datetime("2023-06-15T10:30:00Z")
        assert result is not None
        assert result.year == 2023

    def test_parse_datetime_iso_with_offset(self):
        result = _parse_datetime("2023-06-15T10:30:00+00:00")
        assert result is not None

    def test_parse_datetime_invalid(self):
        assert _parse_datetime("not-a-date") is None

    def test_parse_date_none(self):
        assert _parse_date(None) is None

    def test_parse_date_empty(self):
        assert _parse_date("") is None

    def test_parse_date_valid(self):
        result = _parse_date("2023-06-15")
        assert result is not None
        assert result.year == 2023
        assert result.month == 6
        assert result.day == 15

    def test_parse_date_invalid(self):
        assert _parse_date("not-a-date") is None

    def test_parse_date_wrong_format(self):
        assert _parse_date("15/06/2023") is None


# =============================================================================
# Person Model Tests
# =============================================================================


class TestPersonModel:
    """Tests for Person dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 1,
            "name": "John Doe",
            "email": [
                {"value": "john@example.com", "primary": True},
                {"value": "john2@example.com", "primary": False},
            ],
            "phone": [
                {"value": "+1-555-0100", "primary": True},
            ],
            "org_id": 10,
            "org_name": "Acme Corp",
            "owner_id": 5,
            "visible_to": 3,
            "add_time": "2023-01-15 10:00:00",
            "update_time": "2023-06-20 14:30:00",
            "active_flag": True,
        }
        person = Person.from_api(data)
        assert person.id == 1
        assert person.name == "John Doe"
        assert person.email == "john@example.com"
        assert person.phone == "+1-555-0100"
        assert person.org_id == 10
        assert person.org_name == "Acme Corp"
        assert person.owner_id == 5
        assert person.visible_to == "3"
        assert person.add_time is not None
        assert person.active_flag is True

    def test_from_api_no_primary_email(self):
        data = {
            "id": 2,
            "name": "Jane",
            "email": [{"value": "fallback@example.com", "primary": False}],
            "phone": [],
        }
        person = Person.from_api(data)
        assert person.email == "fallback@example.com"

    def test_from_api_empty_email_list(self):
        data = {"id": 3, "name": "No Email", "email": [], "phone": []}
        person = Person.from_api(data)
        assert person.email is None

    def test_from_api_no_phone(self):
        data = {"id": 4, "name": "No Phone", "email": [], "phone": []}
        person = Person.from_api(data)
        assert person.phone is None

    def test_from_api_phone_fallback_no_primary(self):
        data = {
            "id": 5,
            "name": "Fallback Phone",
            "email": [],
            "phone": [{"value": "+1-555-0200", "primary": False}],
        }
        person = Person.from_api(data)
        assert person.phone == "+1-555-0200"

    def test_from_api_minimal(self):
        data = {"id": 6}
        person = Person.from_api(data)
        assert person.id == 6
        assert person.name == ""
        assert person.email is None
        assert person.phone is None

    def test_to_api_full(self):
        person = Person(
            id=1,
            name="John Doe",
            email="john@example.com",
            phone="+1-555-0100",
            org_id=10,
            owner_id=5,
            visible_to="3",
        )
        result = person.to_api()
        assert result["name"] == "John Doe"
        assert result["email"] == [{"value": "john@example.com", "primary": True}]
        assert result["phone"] == [{"value": "+1-555-0100", "primary": True}]
        assert result["org_id"] == 10
        assert result["owner_id"] == 5
        assert result["visible_to"] == "3"

    def test_to_api_minimal(self):
        person = Person(id=1, name="Min")
        result = person.to_api()
        assert result == {"name": "Min", "visible_to": "5"}
        assert "email" not in result
        assert "phone" not in result

    def test_from_api_email_list_with_non_dict(self):
        """Email list with string entries rather than dicts."""
        data = {
            "id": 7,
            "name": "Test",
            "email": ["simple@example.com"],
            "phone": ["555-0100"],
        }
        person = Person.from_api(data)
        assert person.email == "simple@example.com"
        assert person.phone == "555-0100"


# =============================================================================
# Organization Model Tests
# =============================================================================


class TestOrganizationModel:
    """Tests for Organization dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 10,
            "name": "Acme Corp",
            "address": "123 Main St",
            "owner_id": 5,
            "visible_to": 5,
            "add_time": "2023-01-01 00:00:00",
            "update_time": "2023-06-01 12:00:00",
            "active_flag": True,
            "people_count": 10,
            "open_deals_count": 3,
            "won_deals_count": 8,
            "lost_deals_count": 2,
        }
        org = Organization.from_api(data)
        assert org.id == 10
        assert org.name == "Acme Corp"
        assert org.address == "123 Main St"
        assert org.people_count == 10
        assert org.open_deals_count == 3
        assert org.won_deals_count == 8
        assert org.lost_deals_count == 2

    def test_from_api_minimal(self):
        data = {"id": 11}
        org = Organization.from_api(data)
        assert org.id == 11
        assert org.name == ""
        assert org.people_count == 0

    def test_to_api_full(self):
        org = Organization(
            id=10, name="Acme", address="123 Main", owner_id=5, visible_to="3"
        )
        result = org.to_api()
        assert result["name"] == "Acme"
        assert result["address"] == "123 Main"
        assert result["owner_id"] == 5
        assert result["visible_to"] == "3"

    def test_to_api_minimal(self):
        org = Organization(id=10, name="Min")
        result = org.to_api()
        assert result["name"] == "Min"
        assert result["visible_to"] == "5"

    def test_from_api_with_all_zero_counts(self):
        data = {
            "id": 12,
            "name": "New Org",
            "people_count": 0,
            "open_deals_count": 0,
            "won_deals_count": 0,
            "lost_deals_count": 0,
        }
        org = Organization.from_api(data)
        assert org.people_count == 0
        assert org.open_deals_count == 0


# =============================================================================
# Pipeline & Stage Model Tests
# =============================================================================


class TestPipelineModel:
    """Tests for Pipeline dataclass."""

    def test_from_api(self):
        data = {
            "id": 1,
            "name": "Sales Pipeline",
            "url_title": "sales-pipeline",
            "order_nr": 1,
            "active": True,
            "deal_probability": True,
            "add_time": "2023-01-01 00:00:00",
        }
        pipeline = Pipeline.from_api(data)
        assert pipeline.id == 1
        assert pipeline.name == "Sales Pipeline"
        assert pipeline.url_title == "sales-pipeline"
        assert pipeline.active is True
        assert pipeline.deal_probability is True

    def test_from_api_minimal(self):
        data = {"id": 2}
        pipeline = Pipeline.from_api(data)
        assert pipeline.id == 2
        assert pipeline.name == ""
        assert pipeline.active is True


class TestStageModel:
    """Tests for Stage dataclass."""

    def test_from_api(self):
        data = {
            "id": 1,
            "name": "Qualification",
            "pipeline_id": 1,
            "order_nr": 1,
            "active_flag": True,
            "deal_probability": 25,
            "rotten_flag": True,
            "rotten_days": 30,
        }
        stage = Stage.from_api(data)
        assert stage.id == 1
        assert stage.name == "Qualification"
        assert stage.pipeline_id == 1
        assert stage.deal_probability == 25
        assert stage.rotten_flag is True
        assert stage.rotten_days == 30

    def test_from_api_minimal(self):
        data = {"id": 2}
        stage = Stage.from_api(data)
        assert stage.id == 2
        assert stage.pipeline_id == 0
        assert stage.deal_probability == 100
        assert stage.rotten_flag is False
        assert stage.rotten_days is None


# =============================================================================
# Deal Model Tests
# =============================================================================


class TestDealModel:
    """Tests for Deal dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 100,
            "title": "Enterprise License",
            "value": 50000,
            "currency": "USD",
            "status": "open",
            "stage_id": 3,
            "pipeline_id": 1,
            "person_id": 1,
            "person_name": "John Doe",
            "org_id": 10,
            "org_name": "Acme Corp",
            "owner_id": 5,
            "expected_close_date": "2023-12-31",
            "won_time": "2023-11-15 10:00:00",
            "add_time": "2023-06-01 09:00:00",
            "probability": 75.0,
            "lost_reason": None,
            "visible_to": 5,
        }
        deal = Deal.from_api(data)
        assert deal.id == 100
        assert deal.title == "Enterprise License"
        assert deal.value == 50000.0
        assert deal.currency == "USD"
        assert deal.status == DealStatus.OPEN
        assert deal.stage_id == 3
        assert deal.pipeline_id == 1
        assert deal.person_name == "John Doe"
        assert deal.probability == 75.0
        assert deal.expected_close_date is not None
        assert deal.won_time is not None

    def test_from_api_won_status(self):
        data = {"id": 101, "title": "Won Deal", "status": "won"}
        deal = Deal.from_api(data)
        assert deal.status == DealStatus.WON

    def test_from_api_lost_status(self):
        data = {
            "id": 102,
            "title": "Lost Deal",
            "status": "lost",
            "lost_reason": "Price too high",
        }
        deal = Deal.from_api(data)
        assert deal.status == DealStatus.LOST
        assert deal.lost_reason == "Price too high"

    def test_from_api_minimal(self):
        data = {"id": 103}
        deal = Deal.from_api(data)
        assert deal.id == 103
        assert deal.title == ""
        assert deal.value == 0.0

    def test_to_api_full(self):
        deal = Deal(
            id=100,
            title="Big Deal",
            value=100000,
            currency="EUR",
            stage_id=5,
            pipeline_id=2,
            person_id=1,
            org_id=10,
            owner_id=5,
            expected_close_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
            probability=80.0,
            visible_to="3",
        )
        result = deal.to_api()
        assert result["title"] == "Big Deal"
        assert result["value"] == 100000
        assert result["currency"] == "EUR"
        assert result["stage_id"] == 5
        assert result["expected_close_date"] == "2024-06-30"
        assert result["probability"] == 80.0

    def test_to_api_minimal(self):
        deal = Deal(id=100, title="Min Deal")
        result = deal.to_api()
        assert result["title"] == "Min Deal"
        assert "value" not in result

    def test_from_api_zero_value(self):
        data = {"id": 104, "title": "Free deal", "value": 0, "status": "open"}
        deal = Deal.from_api(data)
        assert deal.value == 0.0

    def test_from_api_string_value(self):
        data = {"id": 105, "title": "Deal", "value": "5000.50", "status": "open"}
        deal = Deal.from_api(data)
        assert deal.value == 5000.50

    def test_to_api_with_zero_probability(self):
        deal = Deal(id=1, title="Test", probability=0.0)
        result = deal.to_api()
        assert result["probability"] == 0.0


# =============================================================================
# Activity Model Tests
# =============================================================================


class TestActivityModel:
    """Tests for Activity dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 200,
            "type": "call",
            "subject": "Follow-up call",
            "done": True,
            "due_date": "2023-07-15",
            "due_time": "14:00",
            "duration": "00:30",
            "deal_id": 100,
            "person_id": 1,
            "org_id": 10,
            "user_id": 5,
            "note": "Discussed pricing",
            "location": "Zoom",
            "public_description": "Call about pricing",
            "add_time": "2023-07-10 09:00:00",
            "update_time": "2023-07-15 15:00:00",
            "marked_as_done_time": "2023-07-15 14:30:00",
        }
        activity = Activity.from_api(data)
        assert activity.id == 200
        assert activity.type == "call"
        assert activity.subject == "Follow-up call"
        assert activity.done is True
        assert activity.due_time == "14:00"
        assert activity.duration == "00:30"
        assert activity.deal_id == 100
        assert activity.owner_id == 5
        assert activity.note == "Discussed pricing"
        assert activity.location == "Zoom"
        assert activity.marked_as_done_time is not None

    def test_from_api_owner_id_fallback(self):
        """owner_id uses user_id field from API, or falls back to owner_id."""
        data = {"id": 201, "owner_id": 7}
        activity = Activity.from_api(data)
        assert activity.owner_id == 7

    def test_from_api_minimal(self):
        data = {"id": 202}
        activity = Activity.from_api(data)
        assert activity.id == 202
        assert activity.type == ""
        assert activity.subject == ""
        assert activity.done is False

    def test_to_api_full(self):
        activity = Activity(
            id=200,
            type="meeting",
            subject="Team sync",
            done=True,
            due_date=datetime(2023, 7, 20, tzinfo=timezone.utc),
            due_time="10:00",
            duration="01:00",
            deal_id=100,
            person_id=1,
            org_id=10,
            owner_id=5,
            note="Weekly standup",
            location="Office",
        )
        result = activity.to_api()
        assert result["type"] == "meeting"
        assert result["subject"] == "Team sync"
        assert result["done"] == 1
        assert result["due_date"] == "2023-07-20"
        assert result["due_time"] == "10:00"
        assert result["duration"] == "01:00"
        assert result["deal_id"] == 100
        assert result["user_id"] == 5
        assert result["note"] == "Weekly standup"
        assert result["location"] == "Office"

    def test_to_api_not_done(self):
        activity = Activity(id=200, type="task", subject="Do stuff", done=False)
        result = activity.to_api()
        assert result["done"] == 0

    def test_to_api_minimal(self):
        activity = Activity(id=200, type="call", subject="Quick call")
        result = activity.to_api()
        assert result["type"] == "call"
        assert result["subject"] == "Quick call"
        assert result["done"] == 0
        assert "deal_id" not in result


# =============================================================================
# Note Model Tests
# =============================================================================


class TestNoteModel:
    """Tests for Note dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 300,
            "content": "Important note about the deal",
            "deal_id": 100,
            "person_id": 1,
            "org_id": 10,
            "user_id": 5,
            "add_time": "2023-07-10 09:00:00",
            "update_time": "2023-07-10 10:00:00",
            "pinned_to_deal_flag": True,
            "pinned_to_person_flag": False,
            "pinned_to_organization_flag": False,
        }
        note = Note.from_api(data)
        assert note.id == 300
        assert note.content == "Important note about the deal"
        assert note.deal_id == 100
        assert note.pinned_to_deal_flag is True
        assert note.pinned_to_person_flag is False

    def test_from_api_minimal(self):
        data = {"id": 301}
        note = Note.from_api(data)
        assert note.id == 301
        assert note.content == ""

    def test_to_api_full(self):
        note = Note(
            id=300,
            content="Deal note",
            deal_id=100,
            person_id=1,
            org_id=10,
            pinned_to_deal_flag=True,
            pinned_to_person_flag=True,
            pinned_to_organization_flag=True,
        )
        result = note.to_api()
        assert result["content"] == "Deal note"
        assert result["deal_id"] == 100
        assert result["person_id"] == 1
        assert result["org_id"] == 10
        assert result["pinned_to_deal_flag"] == 1
        assert result["pinned_to_person_flag"] == 1
        assert result["pinned_to_organization_flag"] == 1

    def test_to_api_minimal(self):
        note = Note(id=300, content="Just text")
        result = note.to_api()
        assert result == {"content": "Just text"}

    def test_to_api_no_pins(self):
        note = Note(id=1, content="Test")
        result = note.to_api()
        assert "pinned_to_deal_flag" not in result
        assert "pinned_to_person_flag" not in result
        assert "pinned_to_organization_flag" not in result


# =============================================================================
# Product Model Tests
# =============================================================================


class TestProductModel:
    """Tests for Product dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 400,
            "name": "Premium Plan",
            "code": "PREM-001",
            "description": "Our premium offering",
            "unit": "license",
            "tax": 8.5,
            "active_flag": True,
            "selectable": True,
            "first_char": "P",
            "visible_to": 5,
            "owner_id": 5,
            "prices": [{"price": 99.99, "currency": "USD"}],
            "add_time": "2023-01-01 00:00:00",
        }
        product = Product.from_api(data)
        assert product.id == 400
        assert product.name == "Premium Plan"
        assert product.code == "PREM-001"
        assert product.description == "Our premium offering"
        assert product.unit == "license"
        assert product.tax == 8.5
        assert len(product.prices) == 1
        assert product.first_char == "P"

    def test_from_api_minimal(self):
        data = {"id": 401}
        product = Product.from_api(data)
        assert product.id == 401
        assert product.name == ""
        assert product.tax == 0.0
        assert product.prices == []

    def test_to_api_full(self):
        product = Product(
            id=400,
            name="Plan A",
            code="A-001",
            description="Description",
            unit="seat",
            tax=10.0,
            visible_to="3",
            owner_id=5,
            prices=[{"price": 50, "currency": "USD"}],
        )
        result = product.to_api()
        assert result["name"] == "Plan A"
        assert result["code"] == "A-001"
        assert result["description"] == "Description"
        assert result["unit"] == "seat"
        assert result["tax"] == 10.0
        assert result["visible_to"] == "3"
        assert result["owner_id"] == 5
        assert len(result["prices"]) == 1

    def test_to_api_minimal(self):
        product = Product(id=400, name="Basic")
        result = product.to_api()
        assert result["name"] == "Basic"
        assert "code" not in result
        assert result["visible_to"] == "5"

    def test_from_api_with_empty_prices(self):
        data = {"id": 402, "name": "No Prices", "prices": []}
        product = Product.from_api(data)
        assert product.prices == []


# =============================================================================
# User Model Tests
# =============================================================================


class TestUserModel:
    """Tests for User dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 500,
            "name": "Admin User",
            "email": "admin@example.com",
            "active_flag": True,
            "is_admin": True,
            "is_you": True,
            "role_id": 1,
            "timezone_name": "US/Pacific",
            "icon_url": "https://example.com/icon.png",
            "created": "2023-01-01 00:00:00",
            "modified": "2023-06-01 12:00:00",
        }
        user = User.from_api(data)
        assert user.id == 500
        assert user.name == "Admin User"
        assert user.email == "admin@example.com"
        assert user.is_admin is True
        assert user.is_you is True
        assert user.role_id == 1
        assert user.timezone_name == "US/Pacific"
        assert user.created is not None

    def test_from_api_minimal(self):
        data = {"id": 501}
        user = User.from_api(data)
        assert user.id == 501
        assert user.name == ""
        assert user.email == ""
        assert user.is_admin is False


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestPipedriveClientInit:
    """Tests for PipedriveClient initialization and context manager."""

    def test_init(self, credentials):
        client = PipedriveClient(credentials)
        assert client.credentials is credentials
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, credentials):
        with patch("aragora.connectors.crm.pipedrive.httpx") as mock_httpx:
            mock_async_client = AsyncMock()
            mock_httpx.AsyncClient.return_value = mock_async_client
            mock_httpx.Timeout.return_value = MagicMock()

            async with PipedriveClient(credentials) as client:
                assert client._client is mock_async_client

            mock_async_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exit_clears_client(self, credentials):
        with patch("aragora.connectors.crm.pipedrive.httpx") as mock_httpx:
            mock_async_client = AsyncMock()
            mock_httpx.AsyncClient.return_value = mock_async_client
            mock_httpx.Timeout.return_value = MagicMock()

            client = PipedriveClient(credentials)
            await client.__aenter__()
            assert client._client is not None
            await client.__aexit__(None, None, None)
            assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager_exit_when_no_client(self, credentials):
        client = PipedriveClient(credentials)
        assert client._client is None
        await client.__aexit__(None, None, None)
        assert client._client is None


# =============================================================================
# API Request Tests
# =============================================================================


class TestApiRequest:
    """Tests for _request method."""

    @pytest.mark.asyncio
    async def test_request_not_initialized(self, credentials):
        client = PipedriveClient(credentials)
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await client._request("GET", "/persons")

    @pytest.mark.asyncio
    async def test_request_adds_api_token(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client._request("GET", "/persons")
        call_kwargs = mock_httpx_client.request.call_args
        assert call_kwargs.kwargs["params"]["api_token"] == "test-api-token-123"

    @pytest.mark.asyncio
    async def test_request_success(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 1}}
        )
        result = await pipedrive_client._request("GET", "/persons/1")
        assert result["data"]["id"] == 1

    @pytest.mark.asyncio
    async def test_request_http_error(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"error": "Not found", "error_code": "not_found"},
            status_code=404,
        )
        with pytest.raises(PipedriveError) as exc_info:
            await pipedrive_client._request("GET", "/persons/999")
        assert exc_info.value.status_code == 404
        assert exc_info.value.error_code == "not_found"

    @pytest.mark.asyncio
    async def test_request_http_error_empty_body(self, pipedrive_client, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 500
        resp.content = b""
        resp.json.return_value = {}
        mock_httpx_client.request.return_value = resp
        with pytest.raises(PipedriveError) as exc_info:
            await pipedrive_client._request("GET", "/test")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_success_false(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": False, "error": "Invalid data", "error_code": "validation_error"},
            status_code=200,
        )
        with pytest.raises(PipedriveError, match="Invalid data"):
            await pipedrive_client._request("POST", "/persons", json={"name": ""})

    @pytest.mark.asyncio
    async def test_request_with_params(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client._request(
            "GET", "/persons", params={"start": 0, "limit": 50}
        )
        call_kwargs = mock_httpx_client.request.call_args
        assert call_kwargs.kwargs["params"]["start"] == 0
        assert call_kwargs.kwargs["params"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_request_with_json_body(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 1}}
        )
        body = {"name": "Test Person"}
        await pipedrive_client._request("POST", "/persons", json=body)
        call_kwargs = mock_httpx_client.request.call_args
        assert call_kwargs.kwargs["json"] == body

    @pytest.mark.asyncio
    async def test_request_rate_limit_429(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"error": "Rate limit exceeded", "error_code": "rate_limit"},
            status_code=429,
        )
        with pytest.raises(PipedriveError) as exc_info:
            await pipedrive_client._request("GET", "/persons")
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_request_server_error_500(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"error": "Internal server error"},
            status_code=500,
        )
        with pytest.raises(PipedriveError) as exc_info:
            await pipedrive_client._request("GET", "/test")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_merges_params_with_api_token(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client._request(
            "GET", "/persons", params={"filter_id": 5}
        )
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["api_token"] == "test-api-token-123"
        assert params["filter_id"] == 5


# =============================================================================
# Person Operations Tests
# =============================================================================


class TestPersonOperations:
    """Tests for person CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_persons(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [
                    {"id": 1, "name": "John"},
                    {"id": 2, "name": "Jane"},
                ],
            }
        )
        persons = await pipedrive_client.get_persons()
        assert len(persons) == 2
        assert persons[0].name == "John"
        assert persons[1].name == "Jane"

    @pytest.mark.asyncio
    async def test_get_persons_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        persons = await pipedrive_client.get_persons()
        assert persons == []

    @pytest.mark.asyncio
    async def test_get_persons_with_filters(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client.get_persons(start=10, limit=50, filter_id=5, sort="name ASC")
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["start"] == 10
        assert params["limit"] == 50
        assert params["filter_id"] == 5
        assert params["sort"] == "name ASC"

    @pytest.mark.asyncio
    async def test_get_person(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 1, "name": "John Doe"}}
        )
        person = await pipedrive_client.get_person(1)
        assert person.id == 1
        assert person.name == "John Doe"

    @pytest.mark.asyncio
    async def test_create_person(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {
                    "id": 99,
                    "name": "New Person",
                    "email": [{"value": "new@example.com", "primary": True}],
                    "phone": [{"value": "+1-555-0300", "primary": True}],
                },
            }
        )
        person = await pipedrive_client.create_person(
            name="New Person",
            email="new@example.com",
            phone="+1-555-0300",
            org_id=10,
            owner_id=5,
            visible_to="3",
        )
        assert person.id == 99
        assert person.name == "New Person"
        assert person.email == "new@example.com"

    @pytest.mark.asyncio
    async def test_create_person_minimal(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 100, "name": "Just Name"}}
        )
        person = await pipedrive_client.create_person(name="Just Name")
        assert person.id == 100

    @pytest.mark.asyncio
    async def test_update_person(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 1, "name": "Updated Name"}}
        )
        person = await pipedrive_client.update_person(1, name="Updated Name")
        assert person.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_person(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"success": True})
        result = await pipedrive_client.delete_person(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_search_persons(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {
                    "items": [
                        {"item": {"id": 1, "name": "John Doe"}},
                        {"item": {"id": 2, "name": "John Smith"}},
                    ]
                },
            }
        )
        results = await pipedrive_client.search_persons("John")
        assert len(results) == 2
        assert results[0].name == "John Doe"

    @pytest.mark.asyncio
    async def test_search_persons_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"items": []}}
        )
        results = await pipedrive_client.search_persons("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_persons_with_fields(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"items": []}}
        )
        await pipedrive_client.search_persons("test", fields="name,email", limit=10)
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["fields"] == "name,email"
        assert params["limit"] == 10


# =============================================================================
# Organization Operations Tests
# =============================================================================


class TestOrganizationOperations:
    """Tests for organization CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_organizations(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [{"id": 10, "name": "Acme Corp"}],
            }
        )
        orgs = await pipedrive_client.get_organizations()
        assert len(orgs) == 1
        assert orgs[0].name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_organizations_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        orgs = await pipedrive_client.get_organizations()
        assert orgs == []

    @pytest.mark.asyncio
    async def test_get_organizations_with_filters(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client.get_organizations(
            start=5, limit=25, filter_id=3, sort="name DESC"
        )
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["start"] == 5
        assert params["limit"] == 25
        assert params["filter_id"] == 3
        assert params["sort"] == "name DESC"

    @pytest.mark.asyncio
    async def test_get_organization(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 10, "name": "Acme Corp"}}
        )
        org = await pipedrive_client.get_organization(10)
        assert org.id == 10

    @pytest.mark.asyncio
    async def test_create_organization(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 20, "name": "New Corp", "address": "456 Oak St"}}
        )
        org = await pipedrive_client.create_organization(
            name="New Corp", address="456 Oak St", owner_id=5, visible_to="3"
        )
        assert org.id == 20
        assert org.name == "New Corp"

    @pytest.mark.asyncio
    async def test_update_organization(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 10, "name": "Updated Corp"}}
        )
        org = await pipedrive_client.update_organization(10, name="Updated Corp")
        assert org.name == "Updated Corp"

    @pytest.mark.asyncio
    async def test_delete_organization(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"success": True})
        result = await pipedrive_client.delete_organization(10)
        assert result is True

    @pytest.mark.asyncio
    async def test_search_organizations(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {"items": [{"item": {"id": 10, "name": "Acme"}}]},
            }
        )
        results = await pipedrive_client.search_organizations("Acme")
        assert len(results) == 1
        assert results[0].name == "Acme"

    @pytest.mark.asyncio
    async def test_search_organizations_with_fields(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"items": []}}
        )
        await pipedrive_client.search_organizations("test", fields="name", limit=5)
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["fields"] == "name"
        assert params["limit"] == 5


# =============================================================================
# Deal Operations Tests
# =============================================================================


class TestDealOperations:
    """Tests for deal CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_deals(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [
                    {"id": 100, "title": "Deal A", "status": "open"},
                    {"id": 101, "title": "Deal B", "status": "won"},
                ],
            }
        )
        deals = await pipedrive_client.get_deals()
        assert len(deals) == 2
        assert deals[0].title == "Deal A"
        assert deals[1].status == DealStatus.WON

    @pytest.mark.asyncio
    async def test_get_deals_with_filters(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client.get_deals(
            filter_id=3, stage_id=5, status=DealStatus.OPEN, sort="value DESC"
        )
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["filter_id"] == 3
        assert params["stage_id"] == 5
        assert params["status"] == "open"
        assert params["sort"] == "value DESC"

    @pytest.mark.asyncio
    async def test_get_deals_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        deals = await pipedrive_client.get_deals()
        assert deals == []

    @pytest.mark.asyncio
    async def test_get_deal(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 100, "title": "Enterprise License", "status": "open"}}
        )
        deal = await pipedrive_client.get_deal(100)
        assert deal.id == 100
        assert deal.title == "Enterprise License"

    @pytest.mark.asyncio
    async def test_create_deal(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {
                    "id": 200,
                    "title": "New Deal",
                    "value": 50000,
                    "currency": "USD",
                    "status": "open",
                },
            }
        )
        deal = await pipedrive_client.create_deal(
            title="New Deal",
            value=50000,
            currency="USD",
            person_id=1,
            org_id=10,
            stage_id=3,
            pipeline_id=1,
            probability=75.0,
        )
        assert deal.id == 200
        assert deal.title == "New Deal"

    @pytest.mark.asyncio
    async def test_create_deal_minimal(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 201, "title": "Simple", "status": "open"}}
        )
        deal = await pipedrive_client.create_deal(title="Simple")
        assert deal.id == 201

    @pytest.mark.asyncio
    async def test_update_deal(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 100, "title": "Updated Deal", "status": "open"}}
        )
        deal = await pipedrive_client.update_deal(100, title="Updated Deal")
        assert deal.title == "Updated Deal"

    @pytest.mark.asyncio
    async def test_delete_deal(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"success": True})
        result = await pipedrive_client.delete_deal(100)
        assert result is True

    @pytest.mark.asyncio
    async def test_move_deal_to_stage(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 100, "title": "Deal", "stage_id": 5, "status": "open"}}
        )
        deal = await pipedrive_client.move_deal_to_stage(100, 5)
        assert deal.stage_id == 5

    @pytest.mark.asyncio
    async def test_mark_deal_won(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 100, "title": "Deal", "status": "won"}}
        )
        deal = await pipedrive_client.mark_deal_won(100)
        assert deal.status == DealStatus.WON

    @pytest.mark.asyncio
    async def test_mark_deal_lost(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {
                    "id": 100,
                    "title": "Deal",
                    "status": "lost",
                    "lost_reason": "Budget cuts",
                },
            }
        )
        deal = await pipedrive_client.mark_deal_lost(100, lost_reason="Budget cuts")
        assert deal.status == DealStatus.LOST
        assert deal.lost_reason == "Budget cuts"

    @pytest.mark.asyncio
    async def test_mark_deal_lost_no_reason(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 100, "title": "Deal", "status": "lost"}}
        )
        deal = await pipedrive_client.mark_deal_lost(100)
        assert deal.status == DealStatus.LOST

    @pytest.mark.asyncio
    async def test_search_deals(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {
                    "items": [
                        {"item": {"id": 100, "title": "Enterprise", "status": "open"}},
                    ]
                },
            }
        )
        results = await pipedrive_client.search_deals("Enterprise")
        assert len(results) == 1
        assert results[0].title == "Enterprise"

    @pytest.mark.asyncio
    async def test_search_deals_with_fields(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"items": []}}
        )
        await pipedrive_client.search_deals("test", fields="title", limit=10)
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["fields"] == "title"
        assert params["limit"] == 10


# =============================================================================
# Pipeline & Stage Operations Tests
# =============================================================================


class TestPipelineStageOperations:
    """Tests for pipeline and stage read operations."""

    @pytest.mark.asyncio
    async def test_get_pipelines(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [
                    {"id": 1, "name": "Sales"},
                    {"id": 2, "name": "Support"},
                ],
            }
        )
        pipelines = await pipedrive_client.get_pipelines()
        assert len(pipelines) == 2
        assert pipelines[0].name == "Sales"

    @pytest.mark.asyncio
    async def test_get_pipelines_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        pipelines = await pipedrive_client.get_pipelines()
        assert pipelines == []

    @pytest.mark.asyncio
    async def test_get_pipeline(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 1, "name": "Sales"}}
        )
        pipeline = await pipedrive_client.get_pipeline(1)
        assert pipeline.id == 1

    @pytest.mark.asyncio
    async def test_get_stages(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [
                    {"id": 1, "name": "Lead", "pipeline_id": 1},
                    {"id": 2, "name": "Qualified", "pipeline_id": 1},
                ],
            }
        )
        stages = await pipedrive_client.get_stages(pipeline_id=1)
        assert len(stages) == 2
        assert stages[0].name == "Lead"

    @pytest.mark.asyncio
    async def test_get_stages_no_pipeline_filter(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client.get_stages()
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert "pipeline_id" not in params

    @pytest.mark.asyncio
    async def test_get_stages_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        stages = await pipedrive_client.get_stages()
        assert stages == []

    @pytest.mark.asyncio
    async def test_get_stage(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 1, "name": "Lead", "pipeline_id": 1}}
        )
        stage = await pipedrive_client.get_stage(1)
        assert stage.id == 1
        assert stage.pipeline_id == 1


# =============================================================================
# Activity Operations Tests
# =============================================================================


class TestActivityOperations:
    """Tests for activity CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_activities(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [
                    {"id": 200, "type": "call", "subject": "Follow up"},
                ],
            }
        )
        activities = await pipedrive_client.get_activities()
        assert len(activities) == 1
        assert activities[0].type == "call"

    @pytest.mark.asyncio
    async def test_get_activities_with_filters(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client.get_activities(
            type="meeting", user_id=5, done=True
        )
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["type"] == "meeting"
        assert params["user_id"] == 5
        assert params["done"] == 1

    @pytest.mark.asyncio
    async def test_get_activities_done_false(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client.get_activities(done=False)
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["done"] == 0

    @pytest.mark.asyncio
    async def test_get_activities_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        activities = await pipedrive_client.get_activities()
        assert activities == []

    @pytest.mark.asyncio
    async def test_get_activity(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 200, "type": "call", "subject": "Call"}}
        )
        activity = await pipedrive_client.get_activity(200)
        assert activity.id == 200

    @pytest.mark.asyncio
    async def test_create_activity(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {"id": 300, "type": "meeting", "subject": "Team sync"},
            }
        )
        activity = await pipedrive_client.create_activity(
            type="meeting",
            subject="Team sync",
            deal_id=100,
            person_id=1,
            org_id=10,
            user_id=5,
            note="Weekly sync",
            location="Office",
            duration="01:00",
            due_time="10:00",
            due_date=datetime(2024, 7, 20, tzinfo=timezone.utc),
        )
        assert activity.id == 300
        assert activity.type == "meeting"

    @pytest.mark.asyncio
    async def test_create_activity_minimal(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 301, "type": "task", "subject": "Todo"}}
        )
        activity = await pipedrive_client.create_activity(type="task", subject="Todo")
        assert activity.id == 301

    @pytest.mark.asyncio
    async def test_update_activity(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 200, "type": "call", "subject": "Updated call"}}
        )
        activity = await pipedrive_client.update_activity(200, subject="Updated call")
        assert activity.subject == "Updated call"

    @pytest.mark.asyncio
    async def test_mark_activity_done(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 200, "type": "call", "subject": "Call", "done": True}}
        )
        activity = await pipedrive_client.mark_activity_done(200)
        assert activity.done is True

    @pytest.mark.asyncio
    async def test_delete_activity(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"success": True})
        result = await pipedrive_client.delete_activity(200)
        assert result is True


# =============================================================================
# Note Operations Tests
# =============================================================================


class TestNoteOperations:
    """Tests for note CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_notes(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [
                    {"id": 300, "content": "First note"},
                    {"id": 301, "content": "Second note"},
                ],
            }
        )
        notes = await pipedrive_client.get_notes()
        assert len(notes) == 2

    @pytest.mark.asyncio
    async def test_get_notes_filtered(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": []}
        )
        await pipedrive_client.get_notes(deal_id=100, person_id=1, org_id=10)
        call_kwargs = mock_httpx_client.request.call_args
        params = call_kwargs.kwargs["params"]
        assert params["deal_id"] == 100
        assert params["person_id"] == 1
        assert params["org_id"] == 10

    @pytest.mark.asyncio
    async def test_get_notes_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        notes = await pipedrive_client.get_notes()
        assert notes == []

    @pytest.mark.asyncio
    async def test_get_note(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 300, "content": "A note"}}
        )
        note = await pipedrive_client.get_note(300)
        assert note.id == 300
        assert note.content == "A note"

    @pytest.mark.asyncio
    async def test_create_note(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 400, "content": "New note", "deal_id": 100}}
        )
        note = await pipedrive_client.create_note(
            content="New note",
            deal_id=100,
            pinned_to_deal=True,
        )
        assert note.id == 400
        assert note.content == "New note"

    @pytest.mark.asyncio
    async def test_create_note_all_pins(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 401, "content": "Pinned note"}}
        )
        await pipedrive_client.create_note(
            content="Pinned note",
            deal_id=100,
            person_id=1,
            org_id=10,
            pinned_to_deal=True,
            pinned_to_person=True,
            pinned_to_organization=True,
        )
        call_kwargs = mock_httpx_client.request.call_args
        body = call_kwargs.kwargs["json"]
        assert body["pinned_to_deal_flag"] == 1
        assert body["pinned_to_person_flag"] == 1
        assert body["pinned_to_organization_flag"] == 1

    @pytest.mark.asyncio
    async def test_update_note(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 300, "content": "Updated content"}}
        )
        note = await pipedrive_client.update_note(300, content="Updated content")
        assert note.content == "Updated content"

    @pytest.mark.asyncio
    async def test_delete_note(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"success": True})
        result = await pipedrive_client.delete_note(300)
        assert result is True


# =============================================================================
# Product Operations Tests
# =============================================================================


class TestProductOperations:
    """Tests for product CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_products(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [
                    {"id": 400, "name": "Product A"},
                    {"id": 401, "name": "Product B"},
                ],
            }
        )
        products = await pipedrive_client.get_products()
        assert len(products) == 2
        assert products[0].name == "Product A"

    @pytest.mark.asyncio
    async def test_get_products_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        products = await pipedrive_client.get_products()
        assert products == []

    @pytest.mark.asyncio
    async def test_get_product(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 400, "name": "Product A"}}
        )
        product = await pipedrive_client.get_product(400)
        assert product.id == 400

    @pytest.mark.asyncio
    async def test_create_product(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 500, "name": "New Product", "code": "NP-001"}}
        )
        product = await pipedrive_client.create_product(
            name="New Product",
            code="NP-001",
            description="A new product",
            unit="piece",
            tax=10.0,
            prices=[{"price": 99.99, "currency": "USD"}],
            owner_id=5,
            visible_to="3",
        )
        assert product.id == 500
        assert product.name == "New Product"

    @pytest.mark.asyncio
    async def test_update_product(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 400, "name": "Updated Product"}}
        )
        product = await pipedrive_client.update_product(400, name="Updated Product")
        assert product.name == "Updated Product"

    @pytest.mark.asyncio
    async def test_delete_product(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"success": True})
        result = await pipedrive_client.delete_product(400)
        assert result is True

    @pytest.mark.asyncio
    async def test_search_products(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {
                    "items": [{"item": {"id": 400, "name": "Premium"}}],
                },
            }
        )
        results = await pipedrive_client.search_products("Premium")
        assert len(results) == 1
        assert results[0].name == "Premium"

    @pytest.mark.asyncio
    async def test_search_products_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"items": []}}
        )
        results = await pipedrive_client.search_products("nonexistent")
        assert results == []


# =============================================================================
# User Operations Tests
# =============================================================================


class TestUserOperations:
    """Tests for user read operations."""

    @pytest.mark.asyncio
    async def test_get_users(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [
                    {"id": 500, "name": "Admin", "email": "admin@test.com"},
                    {"id": 501, "name": "User", "email": "user@test.com"},
                ],
            }
        )
        users = await pipedrive_client.get_users()
        assert len(users) == 2
        assert users[0].name == "Admin"

    @pytest.mark.asyncio
    async def test_get_users_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        users = await pipedrive_client.get_users()
        assert users == []

    @pytest.mark.asyncio
    async def test_get_user(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": {"id": 500, "name": "Admin", "email": "admin@test.com"}}
        )
        user = await pipedrive_client.get_user(500)
        assert user.id == 500

    @pytest.mark.asyncio
    async def test_get_current_user(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": {
                    "id": 500,
                    "name": "Me",
                    "email": "me@test.com",
                    "is_you": True,
                },
            }
        )
        user = await pipedrive_client.get_current_user()
        assert user.id == 500
        assert user.is_you is True


# =============================================================================
# Association Operations Tests
# =============================================================================


class TestAssociationOperations:
    """Tests for deal-person/organization associations."""

    @pytest.mark.asyncio
    async def test_get_deal_persons(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}],
            }
        )
        persons = await pipedrive_client.get_deal_persons(100)
        assert len(persons) == 2

    @pytest.mark.asyncio
    async def test_get_deal_persons_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        persons = await pipedrive_client.get_deal_persons(100)
        assert persons == []

    @pytest.mark.asyncio
    async def test_get_person_deals(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [{"id": 100, "title": "Deal A", "status": "open"}],
            }
        )
        deals = await pipedrive_client.get_person_deals(1)
        assert len(deals) == 1
        assert deals[0].title == "Deal A"

    @pytest.mark.asyncio
    async def test_get_person_deals_empty(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"success": True, "data": None}
        )
        deals = await pipedrive_client.get_person_deals(1)
        assert deals == []

    @pytest.mark.asyncio
    async def test_get_organization_deals(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [{"id": 100, "title": "Org Deal", "status": "open"}],
            }
        )
        deals = await pipedrive_client.get_organization_deals(10)
        assert len(deals) == 1

    @pytest.mark.asyncio
    async def test_get_organization_persons(self, pipedrive_client, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "success": True,
                "data": [{"id": 1, "name": "John"}],
            }
        )
        persons = await pipedrive_client.get_organization_persons(10)
        assert len(persons) == 1


# =============================================================================
# Mock Data Generation Tests
# =============================================================================


class TestMockDataGenerators:
    """Tests for mock data helpers."""

    def test_get_mock_person(self):
        person = get_mock_person()
        assert person.id == 1
        assert person.name == "John Doe"
        assert person.email == "john.doe@example.com"
        assert person.phone == "+1-555-123-4567"
        assert person.org_id == 1
        assert person.org_name == "Acme Corp"
        assert person.active_flag is True
        assert person.add_time is not None

    def test_get_mock_organization(self):
        org = get_mock_organization()
        assert org.id == 1
        assert org.name == "Acme Corporation"
        assert org.address == "123 Main St, San Francisco, CA 94102"
        assert org.people_count == 5
        assert org.open_deals_count == 3
        assert org.won_deals_count == 10
        assert org.active_flag is True

    def test_get_mock_deal(self):
        deal = get_mock_deal()
        assert deal.id == 1
        assert deal.title == "Enterprise License"
        assert deal.value == 50000.0
        assert deal.currency == "USD"
        assert deal.status == DealStatus.OPEN
        assert deal.person_name == "John Doe"
        assert deal.probability == 75.0

    def test_get_mock_activity(self):
        activity = get_mock_activity()
        assert activity.id == 1
        assert activity.type == ActivityType.CALL.value
        assert activity.subject == "Discovery call"
        assert activity.done is False
        assert activity.due_time == "14:00"
        assert activity.duration == "00:30"
        assert activity.deal_id == 1


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports from __init__.py."""

    def test_import_from_crm_module(self):
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

        assert PipedriveClient is not None
        assert PipedriveCredentials is not None
        assert PipedriveError is not None
        assert Person is not None
        assert Organization is not None
        assert DealStatus is not None
        assert ActivityType is not None
        assert PipedriveDeal is not None
        assert PipedrivePipeline is not None
        assert PipedriveStage is not None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_person_from_api_email_none(self):
        data = {"id": 1, "name": "Test", "email": None, "phone": None}
        person = Person.from_api(data)
        assert person.email is None

    def test_parse_datetime_type_error(self):
        assert _parse_datetime(12345) is None

    def test_parse_date_type_error(self):
        assert _parse_date(12345) is None

    def test_deal_from_api_deleted_status(self):
        data = {"id": 1, "title": "Deleted", "status": "deleted"}
        deal = Deal.from_api(data)
        assert deal.status == DealStatus.DELETED

    def test_person_defaults(self):
        person = Person(id=1, name="Test")
        assert person.visible_to == PersonVisibility.ENTIRE_COMPANY.value
        assert person.active_flag is True
        assert person.custom_fields == {}

    def test_organization_defaults(self):
        org = Organization(id=1, name="Test")
        assert org.visible_to == PersonVisibility.ENTIRE_COMPANY.value
        assert org.active_flag is True
        assert org.custom_fields == {}

    def test_deal_defaults(self):
        deal = Deal(id=1, title="Test")
        assert deal.status == DealStatus.OPEN
        assert deal.currency == "USD"
        assert deal.value == 0.0
        assert deal.custom_fields == {}

    def test_product_defaults(self):
        product = Product(id=1, name="Test")
        assert product.active_flag is True
        assert product.selectable is True
        assert product.tax == 0.0
        assert product.prices == []

    def test_note_defaults(self):
        note = Note(id=1, content="Test")
        assert note.deal_id is None
        assert note.person_id is None
        assert note.org_id is None
        assert note.pinned_to_deal_flag is False
        assert note.pinned_to_person_flag is False
        assert note.pinned_to_organization_flag is False
