"""
Tests for Metabase BI Connector.

Tests for Metabase API integration.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch


class TestMetabaseCredentials:
    """Tests for Metabase credentials."""

    def test_credentials_creation(self):
        """Test MetabaseCredentials dataclass."""
        from aragora.connectors.analytics.metabase import MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="token_abc123",
        )

        assert creds.base_url == "https://metabase.example.com"
        assert creds.session_token == "token_abc123"
        assert creds.api_key is None
        assert creds.username is None
        assert creds.password is None

    def test_credentials_with_api_key(self):
        """Test credentials with API key."""
        from aragora.connectors.analytics.metabase import MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            api_key="mb_api_key_123",
        )

        assert creds.api_key == "mb_api_key_123"

    def test_credentials_with_username_password(self):
        """Test credentials with username/password."""
        from aragora.connectors.analytics.metabase import MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            username="admin@example.com",
            password="secret123",
        )

        assert creds.username == "admin@example.com"
        assert creds.password == "secret123"


class TestMetabaseDataclasses:
    """Tests for Metabase dataclass parsing."""

    def test_database_from_api(self):
        """Test Database.from_api parsing."""
        from aragora.connectors.analytics.metabase import Database

        data = {
            "id": 1,
            "name": "Production DB",
            "engine": "postgres",
            "description": "Main production database",
            "is_sample": False,
            "is_full_sync": True,
            "tables": [],
            "created_at": "2024-01-01T00:00:00Z",
        }

        db = Database.from_api(data)

        assert db.id == 1
        assert db.name == "Production DB"
        assert db.engine == "postgres"
        assert db.description == "Main production database"
        assert db.is_sample is False

    def test_table_from_api(self):
        """Test Table.from_api parsing."""
        from aragora.connectors.analytics.metabase import Table

        data = {
            "id": 10,
            "name": "users",
            "display_name": "Users",
            "schema": "public",
            "db_id": 1,
            "description": "User accounts table",
        }

        table = Table.from_api(data)

        assert table.id == 10
        assert table.name == "users"
        assert table.display_name == "Users"
        assert table.schema == "public"
        assert table.db_id == 1

    def test_collection_from_api(self):
        """Test Collection.from_api parsing."""
        from aragora.connectors.analytics.metabase import Collection

        data = {
            "id": 5,
            "name": "Sales Reports",
            "description": "Monthly sales analytics",
            "slug": "sales-reports",
            "color": "#509EE3",
            "archived": False,
        }

        collection = Collection.from_api(data)

        assert collection.id == 5
        assert collection.name == "Sales Reports"
        assert collection.slug == "sales-reports"
        assert collection.color == "#509EE3"
        assert collection.archived is False

    def test_card_from_api(self):
        """Test Card.from_api parsing."""
        from aragora.connectors.analytics.metabase import Card, DisplayType

        data = {
            "id": 123,
            "name": "Revenue by Month",
            "description": "Monthly revenue breakdown",
            "display": "bar",
            "database_id": 1,
            "table_id": 10,
            "collection_id": 5,
            "query_type": "query",
            "dataset_query": {"type": "query", "query": {}},
            "archived": False,
            "enable_embedding": True,
        }

        card = Card.from_api(data)

        assert card.id == 123
        assert card.name == "Revenue by Month"
        assert card.display == DisplayType.BAR
        assert card.database_id == 1
        assert card.enable_embedding is True

    def test_card_from_api_unknown_display(self):
        """Test Card.from_api with unknown display type."""
        from aragora.connectors.analytics.metabase import Card, DisplayType

        data = {
            "id": 123,
            "name": "Test Card",
            "display": "unknown_type",
        }

        card = Card.from_api(data)
        assert card.display == DisplayType.TABLE  # Default fallback

    def test_dashboard_from_api(self):
        """Test Dashboard.from_api parsing."""
        from aragora.connectors.analytics.metabase import Dashboard

        data = {
            "id": 456,
            "name": "Executive Dashboard",
            "description": "Key metrics overview",
            "collection_id": 5,
            "parameters": [{"name": "date_filter", "type": "date/range"}],
            "ordered_cards": [
                {"id": 1, "card_id": 123, "row": 0, "col": 0, "size_x": 4, "size_y": 4}
            ],
            "archived": False,
            "enable_embedding": True,
        }

        dashboard = Dashboard.from_api(data)

        assert dashboard.id == 456
        assert dashboard.name == "Executive Dashboard"
        assert len(dashboard.parameters) == 1
        assert len(dashboard.dashcards) == 1
        assert dashboard.enable_embedding is True

    def test_dashcard_from_api(self):
        """Test DashCard.from_api parsing."""
        from aragora.connectors.analytics.metabase import DashCard

        data = {
            "id": 1,
            "card_id": 123,
            "row": 0,
            "col": 4,
            "size_x": 8,
            "size_y": 6,
            "parameter_mappings": [{"parameter_id": "date", "card_id": 123}],
        }

        dashcard = DashCard.from_api(data)

        assert dashcard.id == 1
        assert dashcard.card_id == 123
        assert dashcard.row == 0
        assert dashcard.col == 4
        assert dashcard.size_x == 8
        assert dashcard.size_y == 6

    def test_query_result_from_api(self):
        """Test QueryResult.from_api parsing."""
        from aragora.connectors.analytics.metabase import QueryResult

        data = {
            "data": {
                "cols": [{"name": "name"}, {"name": "count"}],
                "rows": [["Product A", 100], ["Product B", 200]],
            },
            "database_id": 1,
            "row_count": 2,
            "running_time": 150,
            "status": "completed",
        }

        result = QueryResult.from_api(data)

        assert result.row_count == 2
        assert result.running_time == 150
        assert result.status == "completed"
        assert result.columns == ["name", "count"]
        assert len(result.rows) == 2


class TestMetabaseEnums:
    """Tests for Metabase enum values."""

    def test_display_type_enum(self):
        """Test DisplayType enum values."""
        from aragora.connectors.analytics.metabase import DisplayType

        assert DisplayType.TABLE.value == "table"
        assert DisplayType.BAR.value == "bar"
        assert DisplayType.LINE.value == "line"
        assert DisplayType.PIE.value == "pie"
        assert DisplayType.SCATTER.value == "scatter"
        assert DisplayType.MAP.value == "map"

    def test_collection_type_enum(self):
        """Test CollectionType enum values."""
        from aragora.connectors.analytics.metabase import CollectionType

        assert CollectionType.ROOT.value == "root"
        assert CollectionType.PERSONAL.value == "personal"
        assert CollectionType.REGULAR.value == "regular"


class TestMetabaseError:
    """Tests for Metabase error handling."""

    def test_error_creation(self):
        """Test MetabaseError creation."""
        from aragora.connectors.analytics.metabase import MetabaseError

        error = MetabaseError("Authentication failed", status_code=401)

        assert str(error) == "Authentication failed"
        assert error.status_code == 401

    def test_error_without_status_code(self):
        """Test error without status code."""
        from aragora.connectors.analytics.metabase import MetabaseError

        error = MetabaseError("Something went wrong")
        assert error.status_code is None


class TestMetabaseMocks:
    """Tests for mock data generation."""

    def test_mock_card(self):
        """Test mock card generation."""
        from aragora.connectors.analytics.metabase import get_mock_card

        card = get_mock_card()

        assert card.id == 123
        assert card.name == "Sales by Region"
        assert card.database_id == 1

    def test_mock_dashboard(self):
        """Test mock dashboard generation."""
        from aragora.connectors.analytics.metabase import get_mock_dashboard

        dashboard = get_mock_dashboard()

        assert dashboard.id == 456
        assert dashboard.name == "Executive Dashboard"


class TestMetabaseConnector:
    """Tests for MetabaseConnector."""

    def test_connector_initialization(self):
        """Test connector initialization."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        assert connector.credentials == creds
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_connector_context_manager(self):
        """Test connector as context manager."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )

        async with MetabaseConnector(creds) as connector:
            assert connector is not None

    @pytest.mark.asyncio
    async def test_authenticate(self):
        """Test authenticate method."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            username="admin@example.com",
            password="secret",
        )
        connector = MetabaseConnector(creds)

        # Create a mock client - json() is a regular method, not async
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"id": "session_token_123"}
        mock_client.post.return_value = mock_response

        connector._client = mock_client

        token = await connector.authenticate()

        assert token == "session_token_123"
        mock_client.post.assert_called_once()

        await connector.close()

    @pytest.mark.asyncio
    async def test_authenticate_requires_credentials(self):
        """Test authenticate raises error without credentials."""
        from aragora.connectors.analytics.metabase import (
            MetabaseConnector,
            MetabaseCredentials,
            MetabaseError,
        )

        creds = MetabaseCredentials(base_url="https://metabase.example.com")
        connector = MetabaseConnector(creds)

        with pytest.raises(MetabaseError, match="Username and password required"):
            await connector.authenticate()

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_databases(self):
        """Test get_databases method."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        mock_response = [
            {"id": 1, "name": "Production", "engine": "postgres"},
            {"id": 2, "name": "Analytics", "engine": "bigquery"},
        ]

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.get_databases()

            assert len(result) == 2
            assert result[0].name == "Production"
            assert result[1].engine == "bigquery"

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_cards(self):
        """Test get_cards method."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        mock_response = [
            {"id": 1, "name": "Card 1", "display": "bar"},
            {"id": 2, "name": "Card 2", "display": "line"},
        ]

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.get_cards()

            assert len(result) == 2
            mock_request.assert_called_once()

        await connector.close()

    @pytest.mark.asyncio
    async def test_create_card(self):
        """Test create_card method."""
        from aragora.connectors.analytics.metabase import (
            MetabaseConnector,
            MetabaseCredentials,
            DisplayType,
        )

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        mock_response = {
            "id": 999,
            "name": "New Card",
            "display": "bar",
            "database_id": 1,
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.create_card(
                name="New Card",
                database_id=1,
                query={"source-table": 10},
                display=DisplayType.BAR,
            )

            assert result.id == 999
            assert result.name == "New Card"
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert "/card" in call_args[0][1]

        await connector.close()

    @pytest.mark.asyncio
    async def test_execute_card(self):
        """Test execute_card method."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        mock_response = {
            "data": {
                "cols": [{"name": "product"}, {"name": "sales"}],
                "rows": [["Product A", 1000], ["Product B", 2000]],
            },
            "row_count": 2,
            "running_time": 100,
            "status": "completed",
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.execute_card(card_id=123)

            assert result.row_count == 2
            assert result.status == "completed"
            assert len(result.rows) == 2

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_dashboards(self):
        """Test get_dashboards method."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        mock_response = [
            {"id": 1, "name": "Dashboard 1"},
            {"id": 2, "name": "Dashboard 2"},
        ]

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.get_dashboards()
            assert len(result) == 2

        await connector.close()

    @pytest.mark.asyncio
    async def test_create_dashboard(self):
        """Test create_dashboard method."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        mock_response = {
            "id": 999,
            "name": "New Dashboard",
            "collection_id": 5,
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.create_dashboard(
                name="New Dashboard",
                collection_id=5,
                description="Test dashboard",
            )

            assert result.id == 999
            assert result.name == "New Dashboard"

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_collections(self):
        """Test get_collections method."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        mock_response = [
            {"id": 1, "name": "Collection 1"},
            {"id": 2, "name": "Collection 2"},
        ]

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.get_collections()
            assert len(result) == 2

        await connector.close()

    @pytest.mark.asyncio
    async def test_execute_native_query(self):
        """Test execute_native_query method."""
        from aragora.connectors.analytics.metabase import MetabaseConnector, MetabaseCredentials

        creds = MetabaseCredentials(
            base_url="https://metabase.example.com",
            session_token="test_token",
        )
        connector = MetabaseConnector(creds)

        mock_response = {
            "data": {
                "cols": [{"name": "count"}],
                "rows": [[100]],
            },
            "row_count": 1,
            "running_time": 50,
            "status": "completed",
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.execute_native_query(
                database_id=1,
                query="SELECT COUNT(*) as count FROM users",
            )

            assert result.row_count == 1
            assert result.rows[0][0] == 100

        await connector.close()


class TestMetabasePackageImports:
    """Test that Metabase imports work correctly."""

    def test_metabase_imports(self):
        """Test Metabase can be imported."""
        from aragora.connectors.analytics.metabase import (
            MetabaseConnector,
            MetabaseCredentials,
            Database,
            Table,
            Collection,
            Card,
            Dashboard,
            DashCard,
            QueryResult,
            DisplayType,
            MetabaseError,
        )

        assert MetabaseConnector is not None
        assert MetabaseCredentials is not None
        assert Database is not None
        assert Card is not None
        assert Dashboard is not None
