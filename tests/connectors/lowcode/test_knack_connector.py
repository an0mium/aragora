"""
Tests for KnackConnector - Knack API integration.

Tests cover:
- Schema and object retrieval
- Record CRUD operations (object and view-based)
- Filtering and pagination
- Convenience methods (find, upsert)
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestKnackCredentials:
    """Tests for KnackCredentials dataclass."""

    def test_default_base_url(self):
        """Should use default Knack API URL."""
        from aragora.connectors.lowcode.knack import KnackCredentials

        creds = KnackCredentials(
            application_id="app_123",
            api_key="key_456",
        )

        assert creds.application_id == "app_123"
        assert creds.api_key == "key_456"
        assert creds.base_url == "https://api.knack.com/v1"

    def test_custom_base_url(self):
        """Should allow custom base URL."""
        from aragora.connectors.lowcode.knack import KnackCredentials

        creds = KnackCredentials(
            application_id="app_123",
            api_key="key_456",
            base_url="https://custom.knack.com/v1",
        )

        assert creds.base_url == "https://custom.knack.com/v1"


class TestKnackDataModels:
    """Tests for Knack data models."""

    def test_knack_field_from_api(self):
        """Should parse field from API response."""
        from aragora.connectors.lowcode.knack import KnackField, FieldType

        data = {
            "key": "field_1",
            "name": "Customer Name",
            "type": "short_text",
            "required": True,
            "unique": False,
        }

        field = KnackField.from_api(data)

        assert field.key == "field_1"
        assert field.name == "Customer Name"
        assert field.type == FieldType.SHORT_TEXT
        assert field.required is True
        assert field.unique is False

    def test_knack_object_from_api(self):
        """Should parse object from API response."""
        from aragora.connectors.lowcode.knack import KnackObject

        data = {
            "key": "object_1",
            "name": "Customers",
            "identifier": "field_1",
            "fields": [
                {"key": "field_1", "name": "Name", "type": "short_text"},
                {"key": "field_2", "name": "Email", "type": "email"},
            ],
            "connections": {"object_2": "Orders"},
        }

        obj = KnackObject.from_api(data)

        assert obj.key == "object_1"
        assert obj.name == "Customers"
        assert obj.identifier == "field_1"
        assert len(obj.fields) == 2
        assert obj.connections == {"object_2": "Orders"}

    def test_knack_record_from_api(self):
        """Should parse record from API response."""
        from aragora.connectors.lowcode.knack import KnackRecord

        data = {
            "id": "5f1234567890abcdef123456",
            "field_1": "Test Value",
            "field_2": 42,
            "field_3": True,
        }

        record = KnackRecord.from_api(data)

        assert record.id == "5f1234567890abcdef123456"
        assert record.fields["field_1"] == "Test Value"
        assert record.fields["field_2"] == 42
        assert "id" not in record.fields  # Should be excluded from fields

    def test_record_get_field(self):
        """Should get field value with default."""
        from aragora.connectors.lowcode.knack import KnackRecord

        record = KnackRecord(
            id="rec123",
            fields={"field_1": "Value", "field_2": 42},
        )

        assert record.get_field("field_1") == "Value"
        assert record.get_field("missing", "default") == "default"

    def test_record_get_raw_field(self):
        """Should get raw field value if available."""
        from aragora.connectors.lowcode.knack import KnackRecord

        record = KnackRecord(
            id="rec123",
            fields={
                "field_1": "Formatted Value",
                "field_1_raw": "raw_value",
            },
        )

        assert record.get_raw_field("field_1") == "raw_value"
        assert record.get_raw_field("field_2") is None

    def test_knack_view_from_api(self):
        """Should parse view from API response."""
        from aragora.connectors.lowcode.knack import KnackView

        data = {
            "key": "view_1",
            "name": "Customer List",
            "type": "table",
            "source": {"object": "object_1"},
        }

        view = KnackView.from_api(data)

        assert view.key == "view_1"
        assert view.name == "Customer List"
        assert view.type == "table"
        assert view.source_object == "object_1"

    def test_knack_scene_from_api(self):
        """Should parse scene from API response."""
        from aragora.connectors.lowcode.knack import KnackScene

        data = {
            "key": "scene_1",
            "name": "Dashboard",
            "slug": "dashboard",
            "views": [
                {"key": "view_1", "name": "Stats", "type": "chart"},
            ],
        }

        scene = KnackScene.from_api(data)

        assert scene.key == "scene_1"
        assert scene.name == "Dashboard"
        assert scene.slug == "dashboard"
        assert len(scene.views) == 1


class TestKnackConnectorInit:
    """Tests for KnackConnector initialization."""

    def test_connector_init(self):
        """Should initialize with credentials."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        creds = KnackCredentials(application_id="app_123", api_key="key_456")
        connector = KnackConnector(creds)

        assert connector.credentials == creds
        assert connector._client is None
        assert connector._schema is None


class TestKnackSchemaAPI:
    """Tests for Knack schema API."""

    @pytest.mark.asyncio
    async def test_get_application_schema(self):
        """Should get full application schema."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "application": {
                "id": "app_123",
                "name": "Test App",
                "objects": [],
                "scenes": [],
            }
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            schema = await connector.get_application_schema()

        assert schema["id"] == "app_123"
        assert connector._schema is not None

    @pytest.mark.asyncio
    async def test_get_objects(self):
        """Should get all objects in application."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        # Pre-populate schema
        connector._schema = {
            "objects": [
                {"key": "object_1", "name": "Customers", "fields": []},
                {"key": "object_2", "name": "Orders", "fields": []},
            ]
        }

        objects = await connector.get_objects()

        assert len(objects) == 2
        assert objects[0].name == "Customers"
        assert objects[1].name == "Orders"

    @pytest.mark.asyncio
    async def test_get_object(self):
        """Should get specific object by key."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        connector._schema = {
            "objects": [
                {"key": "object_1", "name": "Customers", "fields": []},
            ]
        }

        obj = await connector.get_object("object_1")

        assert obj.key == "object_1"
        assert obj.name == "Customers"

    @pytest.mark.asyncio
    async def test_get_object_not_found(self):
        """Should raise error for unknown object."""
        from aragora.connectors.lowcode.knack import (
            KnackConnector,
            KnackCredentials,
            KnackError,
        )

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        connector._schema = {"objects": []}

        with pytest.raises(KnackError) as exc_info:
            await connector.get_object("nonexistent")

        assert exc_info.value.status_code == 404


class TestKnackRecordsAPI:
    """Tests for Knack records API (object-based)."""

    @pytest.mark.asyncio
    async def test_get_records(self):
        """Should get records from object with pagination."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [
                {"id": "rec1", "field_1": "Value 1"},
                {"id": "rec2", "field_1": "Value 2"},
            ],
            "total_pages": 3,
            "total_records": 75,
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            records, total_pages, total_records = await connector.get_records("object_1")

        assert len(records) == 2
        assert total_pages == 3
        assert total_records == 75

    @pytest.mark.asyncio
    async def test_get_records_with_filters(self):
        """Should pass filters to API."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [],
            "total_pages": 1,
            "total_records": 0,
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_records(
                "object_1",
                filters=[{"field": "field_1", "operator": "is", "value": "test"}],
            )

            # Verify filters were passed
            call_args = mock_client.request.call_args
            assert "filters" in call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_get_record(self):
        """Should get single record by ID."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "rec123",
            "field_1": "Test Value",
            "field_2": 42,
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            record = await connector.get_record("object_1", "rec123")

        assert record.id == "rec123"
        assert record.fields["field_1"] == "Test Value"

    @pytest.mark.asyncio
    async def test_create_record(self):
        """Should create a record."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "recNew",
            "field_1": "New Value",
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            record = await connector.create_record(
                "object_1",
                {"field_1": "New Value"},
            )

        assert record.id == "recNew"

    @pytest.mark.asyncio
    async def test_update_record(self):
        """Should update a record."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "rec123",
            "field_1": "Updated Value",
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            record = await connector.update_record(
                "object_1",
                "rec123",
                {"field_1": "Updated Value"},
            )

        assert record.fields["field_1"] == "Updated Value"

    @pytest.mark.asyncio
    async def test_delete_record(self):
        """Should delete a record."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"delete": True}

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.delete_record("object_1", "rec123")

        assert result is True


class TestKnackViewRecordsAPI:
    """Tests for Knack view-based records API."""

    @pytest.mark.asyncio
    async def test_get_view_records(self):
        """Should get records from a view."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [{"id": "rec1", "field_1": "Value"}],
            "total_pages": 1,
            "total_records": 1,
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            records, total_pages, total = await connector.get_view_records("scene_1", "view_1")

        assert len(records) == 1
        assert records[0].id == "rec1"

    @pytest.mark.asyncio
    async def test_create_view_record(self):
        """Should create record through view."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "recNew",
            "field_1": "View Created",
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            record = await connector.create_view_record(
                "scene_1",
                "view_1",
                {"field_1": "View Created"},
            )

        assert record.id == "recNew"


class TestKnackConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.mark.asyncio
    async def test_find_records(self):
        """Should find records by field value."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [{"id": "rec1", "field_1": "match"}],
            "total_pages": 1,
            "total_records": 1,
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            records = await connector.find_records("object_1", "field_1", "match")

        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_get_all_records(self):
        """Should get all records with auto-pagination."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        # Page 1
        response_page1 = MagicMock()
        response_page1.status_code = 200
        response_page1.json.return_value = {
            "records": [{"id": "rec1", "field_1": "v1"}],
            "total_pages": 2,
            "total_records": 2,
        }

        # Page 2
        response_page2 = MagicMock()
        response_page2.status_code = 200
        response_page2.json.return_value = {
            "records": [{"id": "rec2", "field_1": "v2"}],
            "total_pages": 2,
            "total_records": 2,
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=[response_page1, response_page2])
            mock_get_client.return_value = mock_client

            records = await connector.get_all_records("object_1")

        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_upsert_record_update(self):
        """Should update existing record in upsert."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        # Find existing record
        find_response = MagicMock()
        find_response.status_code = 200
        find_response.json.return_value = {
            "records": [{"id": "existing_rec", "field_1": "email@test.com"}],
            "total_pages": 1,
            "total_records": 1,
        }

        # Update
        update_response = MagicMock()
        update_response.status_code = 200
        update_response.json.return_value = {
            "id": "existing_rec",
            "field_1": "email@test.com",
            "field_2": "Updated",
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=[find_response, update_response])
            mock_get_client.return_value = mock_client

            record = await connector.upsert_record(
                "object_1",
                "field_1",
                "email@test.com",
                {"field_2": "Updated"},
            )

        assert record.id == "existing_rec"

    @pytest.mark.asyncio
    async def test_upsert_record_create(self):
        """Should create new record in upsert if not found."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        # No existing record found
        find_response = MagicMock()
        find_response.status_code = 200
        find_response.json.return_value = {
            "records": [],
            "total_pages": 1,
            "total_records": 0,
        }

        # Create
        create_response = MagicMock()
        create_response.status_code = 200
        create_response.json.return_value = {
            "id": "new_rec",
            "field_1": "new@test.com",
            "field_2": "New Value",
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=[find_response, create_response])
            mock_get_client.return_value = mock_client

            record = await connector.upsert_record(
                "object_1",
                "field_1",
                "new@test.com",
                {"field_2": "New Value"},
            )

        assert record.id == "new_rec"


class TestKnackErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Should raise KnackError on API error."""
        from aragora.connectors.lowcode.knack import (
            KnackConnector,
            KnackCredentials,
            KnackError,
        )

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.json.return_value = {
            "message": "Invalid field",
            "errors": [{"field": "field_1", "message": "Required"}],
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(KnackError) as exc_info:
                await connector.get_records("object_1")

        assert exc_info.value.status_code == 400
        assert len(exc_info.value.errors) == 1


class TestKnackContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Should properly manage client lifecycle."""
        from aragora.connectors.lowcode.knack import KnackConnector, KnackCredentials

        connector = KnackConnector(KnackCredentials(application_id="app_123", api_key="key_456"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"objects": []}

        async with connector:
            with patch.object(connector, "_client", new=AsyncMock()) as mock_client:
                mock_client.request = AsyncMock(return_value=mock_response)
                # Make an API call to ensure client is used
                await connector.get_objects()
                assert mock_client.request.called

        # After context exit, close() should have been called
        # (client is set to None after close)
        assert connector._client is None


class TestMockHelpers:
    """Tests for mock helper functions."""

    def test_get_mock_record(self):
        """Should return valid mock record."""
        from aragora.connectors.lowcode.knack import get_mock_record

        record = get_mock_record()

        assert record.id.startswith("5f")
        assert "field_1" in record.fields

    def test_get_mock_object(self):
        """Should return valid mock object."""
        from aragora.connectors.lowcode.knack import get_mock_object

        obj = get_mock_object()

        assert obj.key == "object_1"
        assert obj.name == "Customers"
        assert len(obj.fields) == 3
