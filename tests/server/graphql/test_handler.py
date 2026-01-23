"""Tests for GraphQL HTTP handler."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.server.graphql.handler import (
    GraphQLHandler,
    GraphQLSchemaHandler,
    GRAPHIQL_HTML,
)
from aragora.server.graphql.schema import SCHEMA_SDL


class TestGraphQLHandler:
    """Tests for the GraphQL HTTP handler."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock server context."""
        context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "control_plane_coordinator": None,
        }

        # Setup mock storage
        context["storage"].get_debate.return_value = {
            "id": "test-debate-1",
            "task": "Test question",
            "status": "completed",
            "messages": [],
            "agents": ["claude", "gpt4"],
            "rounds": 3,
            "consensus_reached": True,
        }
        context["storage"].list_recent.return_value = [
            {
                "id": "test-debate-1",
                "task": "Test question",
                "status": "completed",
                "agents": ["claude", "gpt4"],
            }
        ]

        # Setup mock ELO system
        mock_rating = MagicMock()
        mock_rating.elo = 1600
        mock_rating.wins = 10
        mock_rating.losses = 5
        mock_rating.draws = 2
        mock_rating.name = "claude"
        context["elo_system"].get_rating.return_value = mock_rating
        context["elo_system"].get_leaderboard.return_value = [mock_rating]
        context["elo_system"].get_cached_leaderboard.return_value = [mock_rating]

        return context

    @pytest.fixture
    def handler(self, mock_context):
        """Create a GraphQL handler instance."""
        return GraphQLHandler(mock_context)

    @pytest.fixture
    def mock_http_handler(self):
        """Create a mock HTTP request handler."""
        handler = MagicMock()
        handler.headers = {"Content-Type": "application/json"}
        handler.rfile = MagicMock()
        return handler

    def test_can_handle_graphql_paths(self, handler):
        """Test that handler recognizes GraphQL paths."""
        assert handler.can_handle("/graphql")
        assert handler.can_handle("/api/graphql")
        assert handler.can_handle("/api/v1/graphql")
        assert handler.can_handle("/graphql/")
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/graphiql")

    def test_get_returns_graphiql_in_dev(self, handler, mock_http_handler):
        """Test that GET returns GraphiQL playground in development."""
        handler._enable_graphiql = True
        result = handler.handle("/graphql", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        assert "text/html" in result.content_type
        assert b"GraphiQL" in result.body

    def test_get_blocked_in_production(self, handler, mock_http_handler):
        """Test that GET is blocked in production mode."""
        handler._enable_graphiql = False
        result = handler.handle("/graphql", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 405

    def test_post_requires_query(self, handler, mock_http_handler):
        """Test that POST requires a query field."""
        handler.read_json_body = MagicMock(return_value={})
        result = handler.handle_post("/graphql", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        response = json.loads(result.body)
        assert "error" in response

    def test_post_invalid_json(self, handler, mock_http_handler):
        """Test handling of invalid JSON body."""
        handler.read_json_body = MagicMock(return_value=None)
        result = handler.handle_post("/graphql", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400

    def test_post_simple_query(self, handler, mock_http_handler):
        """Test executing a simple query."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query {
                systemHealth {
                    status
                    version
                }
            }
            """
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        result = handler.handle_post("/graphql", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        response = json.loads(result.body)
        assert "data" in response
        assert "systemHealth" in response["data"]

    def test_post_query_with_variables(self, handler, mock_http_handler):
        """Test executing a query with variables."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query GetDebate($id: ID!) {
                debate(id: $id) {
                    id
                    topic
                }
            }
            """,
                "variables": {"id": "test-debate-1"},
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        result = handler.handle_post("/graphql", {}, mock_http_handler)

        assert result is not None
        response = json.loads(result.body)
        assert "data" in response

    def test_post_invalid_query(self, handler, mock_http_handler):
        """Test handling of invalid GraphQL query."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query {
                nonExistentField {
                    id
                }
            }
            """
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        result = handler.handle_post("/graphql", {}, mock_http_handler)

        assert result is not None
        response = json.loads(result.body)
        assert "errors" in response

    def test_operation_name_selection(self, handler, mock_http_handler):
        """Test selecting operation by name."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query GetHealth {
                systemHealth {
                    status
                }
            }
            query GetStats {
                stats {
                    activeJobs
                }
            }
            """,
                "operationName": "GetHealth",
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        result = handler.handle_post("/graphql", {}, mock_http_handler)

        assert result is not None
        response = json.loads(result.body)
        assert "data" in response
        # Should only execute GetHealth
        if response["data"]:
            assert "systemHealth" in response["data"] or "errors" in response

    def test_multiple_operations_requires_name(self, handler, mock_http_handler):
        """Test that multiple operations require operationName."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query GetHealth {
                systemHealth {
                    status
                }
            }
            query GetStats {
                stats {
                    activeJobs
                }
            }
            """
                # No operationName specified
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        result = handler.handle_post("/graphql", {}, mock_http_handler)

        assert result is not None
        response = json.loads(result.body)
        assert "errors" in response


class TestGraphQLSchemaHandler:
    """Tests for the GraphQL schema handler."""

    @pytest.fixture
    def handler(self):
        """Create a schema handler instance."""
        return GraphQLSchemaHandler({})

    def test_can_handle_schema_paths(self, handler):
        """Test that handler recognizes schema paths."""
        assert handler.can_handle("/graphql/schema")
        assert handler.can_handle("/api/graphql/schema")
        assert handler.can_handle("/api/v1/graphql/schema")
        assert not handler.can_handle("/graphql")
        assert not handler.can_handle("/api/schema")

    def test_returns_sdl(self, handler):
        """Test that handler returns SDL by default."""
        result = handler.handle("/graphql/schema", {}, None)

        assert result is not None
        assert result.status_code == 200
        assert "text/plain" in result.content_type
        assert b"type Query" in result.body
        assert b"type Mutation" in result.body

    def test_returns_json_format(self, handler):
        """Test that handler can return JSON format."""
        result = handler.handle("/graphql/schema", {"format": "json"}, None)

        assert result is not None
        assert result.status_code == 200
        assert "application/json" in result.content_type

        response = json.loads(result.body)
        assert "data" in response
        assert "__schema" in response["data"]


class TestGraphiQLHTML:
    """Tests for the GraphiQL HTML template."""

    def test_graphiql_html_contains_required_elements(self):
        """Test that GraphiQL HTML has required elements."""
        assert "graphiql" in GRAPHIQL_HTML.lower()
        assert "react" in GRAPHIQL_HTML.lower()
        assert "/graphql" in GRAPHIQL_HTML  # Endpoint URL
        assert "fetcher" in GRAPHIQL_HTML  # GraphQL fetcher

    def test_graphiql_html_valid(self):
        """Test that GraphiQL HTML is valid."""
        assert GRAPHIQL_HTML.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in GRAPHIQL_HTML
        assert "<head>" in GRAPHIQL_HTML
        assert "<body" in GRAPHIQL_HTML


class TestHandlerIntegration:
    """Integration tests for GraphQL handler with resolvers."""

    @pytest.fixture
    def full_context(self):
        """Create a more complete mock context."""
        context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "control_plane_coordinator": None,
            "_start_time": 1000000,
        }

        # Setup comprehensive storage mock
        context["storage"].get_debate.return_value = {
            "id": "debate-123",
            "task": "Should AI be regulated?",
            "status": "completed",
            "messages": [
                {"role": "agent", "agent": "claude", "content": "Yes", "round": 1},
                {"role": "agent", "agent": "gpt4", "content": "No", "round": 1},
            ],
            "critiques": [],
            "agents": ["claude", "gpt4"],
            "rounds": 3,
            "consensus_reached": True,
            "confidence": 0.85,
            "winner": "claude",
            "created_at": "2024-01-01T00:00:00Z",
        }

        context["storage"].list_recent.return_value = [
            {
                "id": "debate-123",
                "task": "Should AI be regulated?",
                "status": "completed",
                "agents": ["claude", "gpt4"],
            },
            {
                "id": "debate-124",
                "task": "Is consciousness computable?",
                "status": "running",
                "agents": ["claude", "gemini"],
            },
        ]

        # Setup ELO system
        mock_ratings = []
        for name, elo in [("claude", 1650), ("gpt4", 1580), ("gemini", 1520)]:
            rating = MagicMock()
            rating.name = name
            rating.elo = elo
            rating.wins = 10
            rating.losses = 5
            rating.draws = 2
            mock_ratings.append(rating)

        context["elo_system"].get_leaderboard.return_value = mock_ratings
        context["elo_system"].get_cached_leaderboard.return_value = mock_ratings
        context["elo_system"].get_rating.return_value = mock_ratings[0]

        return context

    @pytest.fixture
    def handler(self, full_context):
        """Create handler with full context."""
        return GraphQLHandler(full_context)

    def test_debates_query(self, handler):
        """Test full debates query execution."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query {
                debates(limit: 10) {
                    debates {
                        id
                        topic
                        status
                    }
                    total
                    hasMore
                }
            }
            """
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        mock_http = MagicMock()
        result = handler.handle_post("/graphql", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        response = json.loads(result.body)
        assert "data" in response
        assert "debates" in response["data"]

    def test_leaderboard_query(self, handler):
        """Test leaderboard query execution."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query {
                leaderboard(limit: 10) {
                    name
                    elo
                    stats {
                        wins
                        losses
                    }
                }
            }
            """
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        mock_http = MagicMock()
        result = handler.handle_post("/graphql", {}, mock_http)

        assert result is not None
        response = json.loads(result.body)
        assert "data" in response

    def test_system_health_query(self, handler):
        """Test system health query execution."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query {
                systemHealth {
                    status
                    version
                    uptimeSeconds
                    components {
                        name
                        status
                    }
                }
            }
            """
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        mock_http = MagicMock()
        result = handler.handle_post("/graphql", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        response = json.loads(result.body)
        assert "data" in response
        assert "systemHealth" in response["data"]

    def test_combined_query(self, handler):
        """Test query combining multiple root fields."""
        handler.read_json_body = MagicMock(
            return_value={
                "query": """
            query Dashboard {
                debates(limit: 5) {
                    debates {
                        id
                        topic
                    }
                    total
                }
                leaderboard(limit: 5) {
                    name
                    elo
                }
                systemHealth {
                    status
                }
                stats {
                    activeJobs
                    totalAgents
                }
            }
            """
            }
        )
        handler.get_current_user = MagicMock(return_value=None)

        mock_http = MagicMock()
        result = handler.handle_post("/graphql", {}, mock_http)

        assert result is not None
        response = json.loads(result.body)
        assert "data" in response

        # Should have all requested fields
        data = response["data"]
        assert "debates" in data or "errors" in response
        assert "leaderboard" in data or "errors" in response
        assert "systemHealth" in data or "errors" in response
        assert "stats" in data or "errors" in response
