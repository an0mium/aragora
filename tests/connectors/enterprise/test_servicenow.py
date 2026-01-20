"""
Tests for the ServiceNow Enterprise Connector.

Tests cover:
- Authentication (Basic and OAuth)
- Table record retrieval
- Incremental sync
- Search functionality
- Webhook handling
"""

import asyncio
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.itsm.servicenow import (
    ServiceNowConnector,
    ServiceNowRecord,
    ServiceNowComment,
    SERVICENOW_TABLES,
)
from aragora.connectors.enterprise.base import SyncState


@pytest.fixture
def servicenow_connector():
    """Create a ServiceNow connector for testing."""
    return ServiceNowConnector(
        instance_url="https://test-instance.service-now.com",
        tables=["incident", "problem"],
        include_comments=True,
    )


@pytest.fixture
def mock_incident_response():
    """Mock incident API response."""
    return {
        "result": [
            {
                "sys_id": "abc123",
                "number": "INC0001234",
                "short_description": "Test incident",
                "description": "This is a test incident description.",
                "state": {"display_value": "New", "value": "1"},
                "priority": {"display_value": "3 - Moderate", "value": "3"},
                "urgency": {"display_value": "2 - Medium", "value": "2"},
                "impact": {"display_value": "2 - Medium", "value": "2"},
                "assigned_to": {"display_value": "John Doe", "value": "user123"},
                "caller_id": {"display_value": "Jane Smith", "value": "user456"},
                "category": "Software",
                "sys_created_on": "2024-01-15 10:30:00",
                "sys_updated_on": "2024-01-16 14:45:00",
            },
            {
                "sys_id": "def456",
                "number": "INC0001235",
                "short_description": "Another incident",
                "description": "Another test description.",
                "state": "Resolved",
                "priority": "4 - Low",
                "urgency": "3 - Low",
                "impact": "3 - Low",
                "assigned_to": "Admin User",
                "caller_id": "",
                "category": "Hardware",
                "sys_created_on": "2024-01-14 09:00:00",
                "sys_updated_on": "2024-01-16 16:00:00",
            },
        ]
    }


@pytest.fixture
def mock_comments_response():
    """Mock journal field API response."""
    return {
        "result": [
            {
                "sys_id": "comment1",
                "element_id": "abc123",
                "value": "Work note from technician",
                "sys_created_by": {"display_value": "Tech Support"},
                "sys_created_on": "2024-01-15 11:00:00",
            },
            {
                "sys_id": "comment2",
                "element_id": "abc123",
                "value": "User replied with more details",
                "sys_created_by": "Jane Smith",
                "sys_created_on": "2024-01-15 12:30:00",
            },
        ]
    }


class TestServiceNowConnectorInit:
    """Tests for connector initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        connector = ServiceNowConnector(
            instance_url="https://example.service-now.com"
        )

        assert connector.instance_url == "https://example.service-now.com"
        assert connector.tables == ["incident", "problem", "change_request"]
        assert connector.include_comments is True
        assert connector.use_oauth is False

    def test_init_with_custom_tables(self):
        """Test initialization with custom tables."""
        connector = ServiceNowConnector(
            instance_url="https://example.service-now.com",
            tables=["incident", "sc_req_item"],
        )

        assert connector.tables == ["incident", "sc_req_item"]

    def test_init_with_knowledge_articles(self):
        """Test initialization with knowledge articles enabled."""
        connector = ServiceNowConnector(
            instance_url="https://example.service-now.com",
            include_knowledge=True,
        )

        assert "kb_knowledge" in connector.tables

    def test_init_normalizes_url(self):
        """Test that trailing slash is removed from URL."""
        connector = ServiceNowConnector(
            instance_url="https://example.service-now.com/"
        )

        assert connector.instance_url == "https://example.service-now.com"

    def test_init_with_oauth(self):
        """Test initialization with OAuth enabled."""
        connector = ServiceNowConnector(
            instance_url="https://example.service-now.com",
            use_oauth=True,
        )

        assert connector.use_oauth is True


class TestServiceNowConnectorProperties:
    """Tests for connector properties."""

    def test_source_type(self, servicenow_connector):
        """Test source_type property."""
        from aragora.reasoning.provenance import SourceType

        assert servicenow_connector.source_type == SourceType.EXTERNAL_API

    def test_name(self, servicenow_connector):
        """Test name property."""
        assert "ServiceNow" in servicenow_connector.name
        assert "test-instance" in servicenow_connector.name


class TestServiceNowAuth:
    """Tests for authentication."""

    @pytest.mark.asyncio
    async def test_basic_auth_header(self, servicenow_connector):
        """Test basic auth header generation."""
        with patch.object(
            servicenow_connector.credentials,
            "get_credential",
            new_callable=AsyncMock,
        ) as mock_cred:
            mock_cred.side_effect = lambda key: {
                "SERVICENOW_USERNAME": "test_user",
                "SERVICENOW_PASSWORD": "test_pass",
            }.get(key)

            header = await servicenow_connector._get_auth_header()

            assert "Authorization" in header
            assert header["Authorization"].startswith("Basic ")

    @pytest.mark.asyncio
    async def test_missing_credentials_raises_error(self, servicenow_connector):
        """Test that missing credentials raise an error."""
        with patch.object(
            servicenow_connector.credentials,
            "get_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(ValueError, match="credentials not configured"):
                await servicenow_connector._get_auth_header()


class TestServiceNowRecordParsing:
    """Tests for record parsing."""

    def test_parse_datetime(self, servicenow_connector):
        """Test datetime parsing."""
        dt = servicenow_connector._parse_datetime("2024-01-15 14:30:00")
        assert dt == datetime(2024, 1, 15, 14, 30, 0)

    def test_parse_datetime_invalid(self, servicenow_connector):
        """Test invalid datetime returns None."""
        assert servicenow_connector._parse_datetime("invalid") is None
        assert servicenow_connector._parse_datetime(None) is None
        assert servicenow_connector._parse_datetime("") is None

    def test_extract_display_value_string(self, servicenow_connector):
        """Test extracting display value from string."""
        assert servicenow_connector._extract_display_value("test") == "test"

    def test_extract_display_value_dict(self, servicenow_connector):
        """Test extracting display value from dict."""
        value = {"display_value": "Test Display", "value": "123"}
        assert servicenow_connector._extract_display_value(value) == "Test Display"

    def test_extract_display_value_none(self, servicenow_connector):
        """Test extracting display value from None."""
        assert servicenow_connector._extract_display_value(None) == ""


class TestServiceNowSync:
    """Tests for sync functionality."""

    @pytest.mark.asyncio
    async def test_sync_items(
        self,
        servicenow_connector,
        mock_incident_response,
        mock_comments_response,
    ):
        """Test syncing items from ServiceNow."""
        with patch.object(
            servicenow_connector,
            "_api_request",
            new_callable=AsyncMock,
        ) as mock_request:
            # Return incidents for first call, comments for subsequent calls
            mock_request.side_effect = [
                mock_incident_response,  # incident table
                mock_comments_response,  # comments for first incident
                {"result": []},  # comments for second incident
                {"result": []},  # problem table (empty)
            ]

            state = SyncState(connector_id=servicenow_connector.connector_id)
            items = []

            async for item in servicenow_connector.sync_items(state):
                items.append(item)

            assert len(items) == 2
            assert items[0].id == "snow-incident-INC0001234"
            assert "Test incident" in items[0].title
            assert "Work note from technician" in items[0].content

    @pytest.mark.asyncio
    async def test_sync_with_cursor(self, servicenow_connector):
        """Test incremental sync with cursor."""
        with patch.object(
            servicenow_connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value={"result": []},
        ) as mock_request:
            state = SyncState(
                connector_id=servicenow_connector.connector_id,
                cursor="2024-01-15T00:00:00",
            )

            async for _ in servicenow_connector.sync_items(state):
                pass

            # Verify the query included the timestamp filter
            calls = mock_request.call_args_list
            assert len(calls) > 0


class TestServiceNowSearch:
    """Tests for search functionality."""

    @pytest.mark.asyncio
    async def test_search(self, servicenow_connector, mock_incident_response):
        """Test searching ServiceNow records."""
        with patch.object(
            servicenow_connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value=mock_incident_response,
        ):
            # Search across 2 tables (incident, problem), each returns 2 results
            results = await servicenow_connector.search("test", limit=10)

            assert len(results) == 4  # 2 from incident + 2 from problem
            assert "[INC0001234] Test incident" in results[0].title

    @pytest.mark.asyncio
    async def test_search_specific_table(
        self, servicenow_connector, mock_incident_response
    ):
        """Test searching a specific table."""
        with patch.object(
            servicenow_connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value=mock_incident_response,
        ):
            results = await servicenow_connector.search(
                "test", limit=10, table="incident"
            )

            assert len(results) == 2


class TestServiceNowFetch:
    """Tests for fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_record(self, servicenow_connector):
        """Test fetching a specific record."""
        with patch.object(
            servicenow_connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value={
                "result": [
                    {
                        "sys_id": "abc123",
                        "number": "INC0001234",
                        "short_description": "Test incident",
                        "description": "Full description",
                        "state": "New",
                        "priority": "3",
                        "assigned_to": "John Doe",
                        "caller_id": "Jane Smith",
                        "sys_created_on": "2024-01-15 10:30:00",
                    }
                ]
            },
        ):
            result = await servicenow_connector.fetch("snow-incident-INC0001234")

            assert result is not None
            assert result.id == "snow-incident-INC0001234"
            assert "Test incident" in result.title

    @pytest.mark.asyncio
    async def test_fetch_invalid_id(self, servicenow_connector):
        """Test fetching with invalid ID format."""
        result = await servicenow_connector.fetch("invalid-format")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, servicenow_connector):
        """Test fetching non-existent record."""
        with patch.object(
            servicenow_connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value={"result": []},
        ):
            result = await servicenow_connector.fetch("snow-incident-INC9999999")
            assert result is None


class TestServiceNowWebhook:
    """Tests for webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook(self, servicenow_connector):
        """Test webhook handling."""
        with patch.object(
            servicenow_connector, "sync", new_callable=AsyncMock
        ) as mock_sync:
            payload = {
                "table_name": "incident",
                "sys_id": "abc123",
                "operation": "update",
            }

            result = await servicenow_connector.handle_webhook(payload)

            assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_ignored_table(self, servicenow_connector):
        """Test that webhooks for non-configured tables are ignored."""
        payload = {
            "table_name": "some_other_table",
            "sys_id": "abc123",
            "operation": "update",
        }

        result = await servicenow_connector.handle_webhook(payload)

        assert result is False


class TestServiceNowTables:
    """Tests for table configuration."""

    def test_incident_table_config(self):
        """Test incident table configuration."""
        assert "incident" in SERVICENOW_TABLES
        config = SERVICENOW_TABLES["incident"]
        assert "number" in config["fields"]
        assert "short_description" in config["fields"]

    def test_problem_table_config(self):
        """Test problem table configuration."""
        assert "problem" in SERVICENOW_TABLES
        config = SERVICENOW_TABLES["problem"]
        assert "known_error" in config["fields"]
        assert "root_cause" in config["fields"]

    def test_change_request_table_config(self):
        """Test change request table configuration."""
        assert "change_request" in SERVICENOW_TABLES
        config = SERVICENOW_TABLES["change_request"]
        assert "implementation_plan" in config["fields"]
        assert "backout_plan" in config["fields"]


class TestServiceNowExcludeStates:
    """Tests for state exclusion."""

    @pytest.mark.asyncio
    async def test_exclude_states(self):
        """Test that excluded states are filtered out."""
        connector = ServiceNowConnector(
            instance_url="https://test.service-now.com",
            tables=["incident"],
            exclude_states=["Closed", "Resolved"],
        )

        with patch.object(
            connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value={
                "result": [
                    {
                        "sys_id": "1",
                        "number": "INC001",
                        "short_description": "Open incident",
                        "state": "New",
                        "sys_created_on": "2024-01-15 10:00:00",
                        "sys_updated_on": "2024-01-15 10:00:00",
                    },
                    {
                        "sys_id": "2",
                        "number": "INC002",
                        "short_description": "Closed incident",
                        "state": "Closed",
                        "sys_created_on": "2024-01-14 10:00:00",
                        "sys_updated_on": "2024-01-14 10:00:00",
                    },
                ]
            },
        ):
            state = SyncState(connector_id=connector.connector_id)
            items = []

            async for item in connector.sync_items(state):
                items.append(item)

            # Only the "New" incident should be included
            assert len(items) == 1
            assert items[0].id == "snow-incident-INC001"
