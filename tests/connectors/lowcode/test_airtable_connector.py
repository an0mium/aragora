"""
Tests for AirtableConnector - Airtable API integration.

Tests cover:
- Base listing and schema retrieval
- Record CRUD operations
- Filtering and sorting
- Pagination
- Error handling
"""

import pytest
import httpx
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestAirtableCredentials:
    """Tests for AirtableCredentials dataclass."""

    def test_default_base_url(self):
        """Should use default Airtable API URL."""
        from aragora.connectors.lowcode.airtable import AirtableCredentials

        creds = AirtableCredentials(personal_access_token="pat_xxx")

        assert creds.personal_access_token == "pat_xxx"
        assert creds.base_url == "https://api.airtable.com/v0"

    def test_custom_base_url(self):
        """Should allow custom base URL."""
        from aragora.connectors.lowcode.airtable import AirtableCredentials

        creds = AirtableCredentials(
            personal_access_token="pat_xxx",
            base_url="https://custom.api.com/v0",
        )

        assert creds.base_url == "https://custom.api.com/v0"


class TestAirtableDataModels:
    """Tests for Airtable data models."""

    def test_airtable_base_from_api(self):
        """Should parse base from API response."""
        from aragora.connectors.lowcode.airtable import AirtableBase

        data = {
            "id": "appXXXXXXXXXXXXXX",
            "name": "Project Tracker",
            "permissionLevel": "create",
        }

        base = AirtableBase.from_api(data)

        assert base.id == "appXXXXXXXXXXXXXX"
        assert base.name == "Project Tracker"
        assert base.permission_level == "create"

    def test_airtable_field_from_api(self):
        """Should parse field from API response."""
        from aragora.connectors.lowcode.airtable import AirtableField, FieldType

        data = {
            "id": "fldXXXXXXXXXXXXXX",
            "name": "Status",
            "type": "singleSelect",
            "description": "Current status",
            "options": {"choices": [{"name": "Open"}, {"name": "Closed"}]},
        }

        field = AirtableField.from_api(data)

        assert field.id == "fldXXXXXXXXXXXXXX"
        assert field.name == "Status"
        assert field.type == FieldType.SINGLE_SELECT
        assert field.description == "Current status"
        assert "choices" in field.options

    def test_airtable_record_from_api(self):
        """Should parse record from API response."""
        from aragora.connectors.lowcode.airtable import AirtableRecord

        data = {
            "id": "recXXXXXXXXXXXXXX",
            "fields": {
                "Name": "Test Record",
                "Count": 42,
                "Active": True,
            },
            "createdTime": "2024-01-15T10:30:00.000Z",
        }

        record = AirtableRecord.from_api(data)

        assert record.id == "recXXXXXXXXXXXXXX"
        assert record.fields["Name"] == "Test Record"
        assert record.fields["Count"] == 42
        assert record.created_time is not None

    def test_record_get_field(self):
        """Should get field value with default."""
        from aragora.connectors.lowcode.airtable import AirtableRecord

        record = AirtableRecord(
            id="rec123",
            fields={"Name": "Test", "Count": 5},
        )

        assert record.get_field("Name") == "Test"
        assert record.get_field("Missing", "default") == "default"

    def test_record_get_linked_records(self):
        """Should extract linked record IDs."""
        from aragora.connectors.lowcode.airtable import AirtableRecord

        record = AirtableRecord(
            id="rec123",
            fields={"Links": ["recA", "recB", "recC"]},
        )

        links = record.get_linked_records("Links")

        assert links == ["recA", "recB", "recC"]

    def test_record_get_attachments(self):
        """Should parse attachments from field."""
        from aragora.connectors.lowcode.airtable import AirtableRecord

        record = AirtableRecord(
            id="rec123",
            fields={
                "Files": [
                    {
                        "id": "att123",
                        "url": "https://example.com/file.pdf",
                        "filename": "document.pdf",
                        "size": 1024,
                        "type": "application/pdf",
                    }
                ]
            },
        )

        attachments = record.get_attachments("Files")

        assert len(attachments) == 1
        assert attachments[0].filename == "document.pdf"
        assert attachments[0].size == 1024


class TestAirtableConnectorInit:
    """Tests for AirtableConnector initialization."""

    def test_connector_init(self):
        """Should initialize with credentials."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        creds = AirtableCredentials(personal_access_token="pat_xxx")
        connector = AirtableConnector(creds)

        assert connector.credentials == creds
        assert connector._client is None


class TestAirtableBasesAPI:
    """Tests for Airtable bases API."""

    @pytest.mark.asyncio
    async def test_list_bases(self):
        """Should list accessible bases."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bases": [
                {"id": "app1", "name": "Base 1", "permissionLevel": "create"},
                {"id": "app2", "name": "Base 2", "permissionLevel": "edit"},
            ],
            "offset": "next_page_token",
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            bases, offset = await connector.list_bases()

        assert len(bases) == 2
        assert bases[0].name == "Base 1"
        assert offset == "next_page_token"

    @pytest.mark.asyncio
    async def test_get_base_schema(self):
        """Should get base schema with tables."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tables": [
                {
                    "id": "tbl1",
                    "name": "Tasks",
                    "primaryFieldId": "fld1",
                    "fields": [
                        {"id": "fld1", "name": "Name", "type": "singleLineText"},
                    ],
                    "views": [
                        {"id": "viw1", "name": "Grid view", "type": "grid"},
                    ],
                }
            ]
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            tables = await connector.get_base_schema("app123")

        assert len(tables) == 1
        assert tables[0].name == "Tasks"
        assert len(tables[0].fields) == 1
        assert len(tables[0].views) == 1


class TestAirtableRecordsAPI:
    """Tests for Airtable records API."""

    @pytest.mark.asyncio
    async def test_list_records(self):
        """Should list records with pagination."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [
                {"id": "rec1", "fields": {"Name": "Record 1"}},
                {"id": "rec2", "fields": {"Name": "Record 2"}},
            ],
            "offset": "next_page",
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            records, offset = await connector.list_records("app123", "Tasks")

        assert len(records) == 2
        assert records[0].id == "rec1"
        assert offset == "next_page"

    @pytest.mark.asyncio
    async def test_list_records_with_filter(self):
        """Should pass filter formula to API."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": []}

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.list_records(
                "app123",
                "Tasks",
                filter_by_formula="{Status} = 'Active'",
            )

            # Verify filter was passed
            call_args = mock_client.request.call_args
            assert "filterByFormula" in call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_get_record(self):
        """Should get single record by ID."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "rec123",
            "fields": {"Name": "Test Record", "Count": 42},
            "createdTime": "2024-01-15T10:30:00.000Z",
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            record = await connector.get_record("app123", "Tasks", "rec123")

        assert record.id == "rec123"
        assert record.fields["Name"] == "Test Record"

    @pytest.mark.asyncio
    async def test_create_record(self):
        """Should create a single record."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [
                {"id": "recNew", "fields": {"Name": "New Record", "Count": 0}},
            ]
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            record = await connector.create_record(
                "app123",
                "Tasks",
                {"Name": "New Record", "Count": 0},
            )

        assert record.id == "recNew"
        assert record.fields["Name"] == "New Record"

    @pytest.mark.asyncio
    async def test_update_record(self):
        """Should update a record."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [
                {"id": "rec123", "fields": {"Name": "Updated Name", "Count": 5}},
            ]
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            record = await connector.update_record(
                "app123",
                "Tasks",
                "rec123",
                {"Name": "Updated Name"},
            )

        assert record.fields["Name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_record(self):
        """Should delete a record."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True, "id": "rec123"}

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.delete_record("app123", "Tasks", "rec123")

        assert result is True


class TestAirtableConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.mark.asyncio
    async def test_find_records(self):
        """Should find records by field value."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [
                {"id": "rec1", "fields": {"Name": "Test", "Status": "Active"}},
            ]
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            records = await connector.find_records("app123", "Tasks", "Status", "Active")

        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_get_all_records(self):
        """Should get all records with auto-pagination."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        # First page
        response_page1 = MagicMock()
        response_page1.status_code = 200
        response_page1.json.return_value = {
            "records": [{"id": "rec1", "fields": {}}],
            "offset": "page2",
        }

        # Second page (last)
        response_page2 = MagicMock()
        response_page2.status_code = 200
        response_page2.json.return_value = {
            "records": [{"id": "rec2", "fields": {}}],
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=[response_page1, response_page2])
            mock_get_client.return_value = mock_client

            records = await connector.get_all_records("app123", "Tasks")

        assert len(records) == 2
        assert records[0].id == "rec1"
        assert records[1].id == "rec2"


class TestAirtableErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Should raise AirtableError on API error."""
        from aragora.connectors.lowcode.airtable import (
            AirtableConnector,
            AirtableCredentials,
            AirtableError,
        )

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.json.return_value = {
            "error": {"type": "NOT_FOUND", "message": "Table not found"}
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(AirtableError) as exc_info:
                await connector.list_bases()

        assert exc_info.value.status_code == 404
        assert "Table not found" in str(exc_info.value)


class TestAirtableContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Should properly manage client lifecycle."""
        from aragora.connectors.lowcode.airtable import AirtableConnector, AirtableCredentials

        connector = AirtableConnector(AirtableCredentials(personal_access_token="pat_xxx"))

        # Create a mock client and set it on the connector
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        connector._client = mock_client

        async with connector:
            # Client should exist during context
            assert connector._client is not None

        # After context exit, close() should have been called
        mock_client.aclose.assert_called_once()
        assert connector._client is None


class TestMockHelpers:
    """Tests for mock helper functions."""

    def test_get_mock_record(self):
        """Should return valid mock record."""
        from aragora.connectors.lowcode.airtable import get_mock_record

        record = get_mock_record()

        assert record.id.startswith("rec")
        assert "Name" in record.fields
        assert record.created_time is not None

    def test_get_mock_base(self):
        """Should return valid mock base."""
        from aragora.connectors.lowcode.airtable import get_mock_base

        base = get_mock_base()

        assert base.id.startswith("app")
        assert base.name == "Test Base"
