"""
Tests for Mixpanel Analytics Connector.

Tests for Mixpanel tracking and analytics API integration.
"""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, patch
import base64


class TestMixpanelCredentials:
    """Tests for Mixpanel credentials."""

    def test_credentials_creation(self):
        """Test MixpanelCredentials dataclass."""
        from aragora.connectors.analytics.mixpanel import MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="token_abc123",
            api_secret="secret_xyz789",
            project_id="12345",
        )

        assert creds.project_token == "token_abc123"
        assert creds.api_secret == "secret_xyz789"
        assert creds.project_id == "12345"
        assert creds.data_residency == "US"

    def test_credentials_us_base_url(self):
        """Test US data residency base URL."""
        from aragora.connectors.analytics.mixpanel import MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="token",
            api_secret="secret",
            data_residency="US",
        )

        assert creds.base_url == "https://mixpanel.com"
        assert creds.data_url == "https://data.mixpanel.com"

    def test_credentials_eu_base_url(self):
        """Test EU data residency base URL."""
        from aragora.connectors.analytics.mixpanel import MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="token",
            api_secret="secret",
            data_residency="EU",
        )

        assert creds.base_url == "https://eu.mixpanel.com"
        assert creds.data_url == "https://data-eu.mixpanel.com"

    def test_credentials_basic_auth(self):
        """Test basic auth encoding."""
        from aragora.connectors.analytics.mixpanel import MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="token",
            api_secret="my_secret",
        )

        expected = base64.b64encode(b"my_secret:").decode()
        assert creds.basic_auth == expected


class TestMixpanelDataclasses:
    """Tests for Mixpanel dataclass models."""

    def test_event_creation(self):
        """Test Event dataclass."""
        from aragora.connectors.analytics.mixpanel import Event

        event = Event(
            event="Button Clicked",
            properties={"button_name": "signup", "page": "/home"},
        )

        assert event.event == "Button Clicked"
        assert event.properties["button_name"] == "signup"
        assert event.time is None

    def test_event_with_time(self):
        """Test Event with timestamp."""
        from aragora.connectors.analytics.mixpanel import Event

        timestamp = datetime(2024, 1, 15, 12, 0, 0)
        event = Event(
            event="Purchase",
            properties={"amount": 99.99},
            time=timestamp,
        )

        assert event.time == timestamp

    def test_event_to_api(self):
        """Test Event.to_api conversion."""
        from aragora.connectors.analytics.mixpanel import Event

        timestamp = datetime(2024, 1, 15, 12, 0, 0)
        event = Event(
            event="Page View",
            properties={"url": "/products"},
            time=timestamp,
        )

        api_data = event.to_api()

        assert api_data["event"] == "Page View"
        assert api_data["properties"]["url"] == "/products"
        assert "time" in api_data["properties"]

    def test_user_profile_from_api(self):
        """Test UserProfile.from_api parsing."""
        from aragora.connectors.analytics.mixpanel import UserProfile

        data = {
            "$distinct_id": "user_123",
            "$properties": {
                "email": "test@example.com",
                "name": "Test User",
                "$created": "2024-01-01T00:00:00Z",
                "$last_seen": "2024-01-15T12:00:00Z",
            },
        }

        profile = UserProfile.from_api(data)

        assert profile.distinct_id == "user_123"
        assert profile.properties["email"] == "test@example.com"
        assert profile.first_seen is not None

    def test_insight_result_from_api(self):
        """Test InsightResult.from_api parsing."""
        from aragora.connectors.analytics.mixpanel import InsightResult

        data = {
            "series": {
                "Button Clicked": [100, 150, 200],
                "Page View": [500, 600, 700],
            },
            "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }

        result = InsightResult.from_api(data)

        assert "Button Clicked" in result.series
        assert "Page View" in result.series
        assert len(result.dates) == 3
        assert result.labels == ["Button Clicked", "Page View"]

    def test_funnel_result_from_api(self):
        """Test FunnelResult.from_api parsing."""
        from aragora.connectors.analytics.mixpanel import FunnelResult

        data = {
            "data": {
                "steps": [
                    {"event": "Signup Started", "count": 1000},
                    {"event": "Email Verified", "count": 800},
                    {"event": "Profile Completed", "count": 500},
                ]
            },
            "meta": {"overall_conversion_rate": 0.5},
        }

        result = FunnelResult.from_api(data)

        assert len(result.steps) == 3
        assert result.overall_conversion_rate == 0.5

    def test_retention_result_from_api(self):
        """Test RetentionResult.from_api parsing."""
        from aragora.connectors.analytics.mixpanel import RetentionResult

        data = {
            "data": [
                {"count": 1000, "first": True},
                {"count": 500, "first": False},
            ],
            "dates": ["2024-01-01", "2024-01-02"],
        }

        result = RetentionResult.from_api(data)

        assert len(result.data) == 2
        assert len(result.dates) == 2

    def test_cohort_from_api(self):
        """Test Cohort.from_api parsing."""
        from aragora.connectors.analytics.mixpanel import Cohort

        data = {
            "id": 123,
            "name": "Power Users",
            "description": "Users with 10+ sessions",
            "count": 5000,
            "created": "2024-01-01T00:00:00Z",
            "project_id": 456,
            "is_visible": True,
        }

        cohort = Cohort.from_api(data)

        assert cohort.id == 123
        assert cohort.name == "Power Users"
        assert cohort.count == 5000
        assert cohort.is_visible is True


class TestMixpanelEnums:
    """Tests for Mixpanel enum values."""

    def test_unit_enum(self):
        """Test Unit enum values."""
        from aragora.connectors.analytics.mixpanel import Unit

        assert Unit.MINUTE.value == "minute"
        assert Unit.HOUR.value == "hour"
        assert Unit.DAY.value == "day"
        assert Unit.WEEK.value == "week"
        assert Unit.MONTH.value == "month"

    def test_event_type_enum(self):
        """Test EventType enum values."""
        from aragora.connectors.analytics.mixpanel import EventType

        assert EventType.GENERAL.value == "general"
        assert EventType.UNIQUE.value == "unique"
        assert EventType.AVERAGE.value == "average"


class TestMixpanelError:
    """Tests for Mixpanel error handling."""

    def test_error_creation(self):
        """Test MixpanelError creation."""
        from aragora.connectors.analytics.mixpanel import MixpanelError

        error = MixpanelError(
            message="Invalid token",
            status_code=401,
            error_details={"error": "Unauthorized"},
        )

        assert str(error) == "Invalid token"
        assert error.status_code == 401
        assert error.error_details["error"] == "Unauthorized"

    def test_error_default_details(self):
        """Test error with default details."""
        from aragora.connectors.analytics.mixpanel import MixpanelError

        error = MixpanelError("Something went wrong")
        assert error.error_details == {}


class TestMixpanelMocks:
    """Tests for mock data generation."""

    def test_mock_event(self):
        """Test mock event generation."""
        from aragora.connectors.analytics.mixpanel import get_mock_event

        event = get_mock_event()

        assert event.event == "Button Clicked"
        assert event.properties["button_name"] == "Sign Up"
        assert event.time is not None

    def test_mock_insight_result(self):
        """Test mock insight result generation."""
        from aragora.connectors.analytics.mixpanel import get_mock_insight_result

        result = get_mock_insight_result()

        assert "Button Clicked" in result.series
        assert len(result.dates) == 5


class TestMixpanelConnector:
    """Tests for MixpanelConnector."""

    def test_connector_initialization(self):
        """Test connector initialization."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )
        connector = MixpanelConnector(creds)

        assert connector.credentials == creds
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_connector_context_manager(self):
        """Test connector as context manager."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )

        async with MixpanelConnector(creds) as connector:
            assert connector is not None

    @pytest.mark.asyncio
    async def test_track_event(self):
        """Test track method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )
        connector = MixpanelConnector(creds)

        mock_response = {"status": 1}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.track(
                event="Button Clicked",
                distinct_id="user_123",
                properties={"button_name": "signup"},
            )

            assert result is True
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "track" in call_args[0][1]

        await connector.close()

    @pytest.mark.asyncio
    async def test_track_batch(self):
        """Test track_batch method."""
        from aragora.connectors.analytics.mixpanel import (
            MixpanelConnector,
            MixpanelCredentials,
            Event,
        )

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )
        connector = MixpanelConnector(creds)

        events = [
            Event(event="Event 1", properties={}),
            Event(event="Event 2", properties={}),
        ]

        mock_response = {"status": 1}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.track_batch(events, distinct_id="user_123")
            assert result is True

        await connector.close()

    @pytest.mark.asyncio
    async def test_set_user_profile(self):
        """Test set_user_profile method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )
        connector = MixpanelConnector(creds)

        mock_response = {"status": 1}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.set_user_profile(
                distinct_id="user_123",
                properties={"email": "test@example.com", "plan": "premium"},
            )

            assert result is True
            call_args = mock_request.call_args
            assert "engage" in call_args[0][1]

        await connector.close()

    @pytest.mark.asyncio
    async def test_set_user_profile_once(self):
        """Test set_user_profile_once method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )
        connector = MixpanelConnector(creds)

        mock_response = {"status": 1}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.set_user_profile_once(
                distinct_id="user_123",
                properties={"first_seen": "2024-01-01"},
            )

            assert result is True

        await connector.close()

    @pytest.mark.asyncio
    async def test_increment_user_property(self):
        """Test increment_user_property method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )
        connector = MixpanelConnector(creds)

        mock_response = {"status": 1}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.increment_user_property(
                distinct_id="user_123",
                property_name="login_count",
                amount=1,
            )

            assert result is True

        await connector.close()

    @pytest.mark.asyncio
    async def test_delete_user_profile(self):
        """Test delete_user_profile method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )
        connector = MixpanelConnector(creds)

        mock_response = {"status": 1}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.delete_user_profile(distinct_id="user_123")
            assert result is True

        await connector.close()

    @pytest.mark.asyncio
    async def test_query_insights(self):
        """Test query_insights method."""
        from aragora.connectors.analytics.mixpanel import (
            MixpanelConnector,
            MixpanelCredentials,
            Unit,
        )

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
            project_id="12345",
        )
        connector = MixpanelConnector(creds)

        mock_response = {
            "series": {"Button Clicked": [100, 150, 200]},
            "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.query_insights(
                event="Button Clicked",
                from_date=date(2024, 1, 1),
                to_date=date(2024, 1, 31),
                unit=Unit.DAY,
            )

            assert "Button Clicked" in result.series

        await connector.close()

    @pytest.mark.asyncio
    async def test_query_funnel(self):
        """Test query_funnel method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
            project_id="12345",
        )
        connector = MixpanelConnector(creds)

        mock_response = {
            "data": {"steps": [{"event": "Step 1", "count": 100}]},
            "meta": {"overall_conversion_rate": 0.75},
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.query_funnel(
                funnel_id=123,
                from_date=date(2024, 1, 1),
                to_date=date(2024, 1, 31),
            )

            assert result.overall_conversion_rate == 0.75

        await connector.close()

    @pytest.mark.asyncio
    async def test_query_retention(self):
        """Test query_retention method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
            project_id="12345",
        )
        connector = MixpanelConnector(creds)

        mock_response = {
            "data": [{"count": 100}, {"count": 50}],
            "dates": ["2024-01-01", "2024-01-02"],
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.query_retention(
                from_date=date(2024, 1, 1),
                to_date=date(2024, 1, 31),
                born_event="Signup",
            )

            assert len(result.data) == 2

        await connector.close()

    @pytest.mark.asyncio
    async def test_list_cohorts(self):
        """Test list_cohorts method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
            project_id="12345",
        )
        connector = MixpanelConnector(creds)

        mock_response = [
            {"id": 1, "name": "Cohort 1", "count": 100},
            {"id": 2, "name": "Cohort 2", "count": 200},
        ]

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.list_cohorts()

            assert len(result) == 2
            assert result[0].name == "Cohort 1"

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_event_names(self):
        """Test get_event_names method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
            project_id="12345",
        )
        connector = MixpanelConnector(creds)

        mock_response = ["Button Clicked", "Page View", "Purchase"]

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.get_event_names()

            assert len(result) == 3
            assert "Button Clicked" in result

        await connector.close()

    @pytest.mark.asyncio
    async def test_append_to_user_list(self):
        """Test append_to_user_list method."""
        from aragora.connectors.analytics.mixpanel import MixpanelConnector, MixpanelCredentials

        creds = MixpanelCredentials(
            project_token="test_token",
            api_secret="test_secret",
        )
        connector = MixpanelConnector(creds)

        mock_response = {"status": 1}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.append_to_user_list(
                distinct_id="user_123",
                property_name="tags",
                values=["premium", "early_adopter"],
            )

            assert result is True

        await connector.close()


class TestMixpanelPackageImports:
    """Test that Mixpanel imports work correctly."""

    def test_mixpanel_imports(self):
        """Test Mixpanel can be imported."""
        from aragora.connectors.analytics.mixpanel import (
            MixpanelConnector,
            MixpanelCredentials,
            Event,
            UserProfile,
            InsightResult,
            FunnelResult,
            RetentionResult,
            Cohort,
            Unit,
            EventType,
            MixpanelError,
        )

        assert MixpanelConnector is not None
        assert MixpanelCredentials is not None
        assert Event is not None
        assert UserProfile is not None
        assert InsightResult is not None
        assert FunnelResult is not None
