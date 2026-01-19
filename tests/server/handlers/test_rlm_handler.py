"""Tests for RLM handler endpoints.

Validates the REST API endpoints for RLM operations including:
- Query debates using RLM with iterative refinement
- Compress debate context
- Get debate at specific abstraction level
- Check refinement status
- Query knowledge mound using RLM
"""

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.rlm import RLMHandler
from aragora.billing.auth.context import UserAuthContext


def mock_authenticated_user():
    """Create a mock authenticated user context."""
    user_ctx = MagicMock(spec=UserAuthContext)
    user_ctx.is_authenticated = True
    user_ctx.user_id = "test-user-123"
    user_ctx.email = "test@example.com"
    return user_ctx


@pytest.fixture
def rlm_handler():
    """Create an RLM handler with mocked dependencies."""
    ctx = {"storage": None}
    handler = RLMHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


def create_request_body(data: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.rfile = BytesIO(body)
    handler.command = "POST"

    def get_json_body():
        return data
    handler.get_json_body = get_json_body

    return handler


def create_auth_request_body(data: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON body and auth token."""
    handler = create_request_body(data)
    handler.headers["Authorization"] = "Bearer test-token"
    return handler


class TestRLMHandlerCanHandle:
    """Test RLMHandler.can_handle method."""

    def test_can_handle_query_rlm(self, rlm_handler):
        """Test can_handle returns True for query-rlm endpoint."""
        assert rlm_handler.can_handle("/api/debates/debate-123/query-rlm")

    def test_can_handle_compress(self, rlm_handler):
        """Test can_handle returns True for compress endpoint."""
        assert rlm_handler.can_handle("/api/debates/debate-123/compress")

    def test_can_handle_context_level(self, rlm_handler):
        """Test can_handle returns True for context level endpoint."""
        assert rlm_handler.can_handle("/api/debates/debate-123/context/SUMMARY")

    def test_can_handle_refinement_status(self, rlm_handler):
        """Test can_handle returns True for refinement-status endpoint."""
        assert rlm_handler.can_handle("/api/debates/debate-123/refinement-status")

    def test_can_handle_knowledge_query(self, rlm_handler):
        """Test can_handle returns True for knowledge query endpoint."""
        assert rlm_handler.can_handle("/api/knowledge/query-rlm")

    def test_cannot_handle_unknown(self, rlm_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not rlm_handler.can_handle("/api/unknown")
        assert not rlm_handler.can_handle("/api/debates")
        assert not rlm_handler.can_handle("/api/debates/123/other")


class TestRLMHandlerExtractDebateId:
    """Test debate ID extraction from path."""

    def test_extract_debate_id_valid(self, rlm_handler):
        """Test extracting debate ID from valid path."""
        debate_id = rlm_handler._extract_debate_id("/api/debates/debate-123/query-rlm")
        assert debate_id == "debate-123"

    def test_extract_debate_id_uuid(self, rlm_handler):
        """Test extracting UUID debate ID."""
        debate_id = rlm_handler._extract_debate_id("/api/debates/550e8400-e29b-41d4-a716-446655440000/compress")
        assert debate_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_extract_debate_id_invalid(self, rlm_handler):
        """Test extracting from invalid path returns None."""
        assert rlm_handler._extract_debate_id("/api/debates") is None
        assert rlm_handler._extract_debate_id("/api/knowledge/query-rlm") is None


class TestRLMHandlerExtractLevel:
    """Test abstraction level extraction from path."""

    def test_extract_level_summary(self, rlm_handler):
        """Test extracting SUMMARY level."""
        level = rlm_handler._extract_level("/api/debates/123/context/summary")
        assert level == "SUMMARY"

    def test_extract_level_abstract(self, rlm_handler):
        """Test extracting ABSTRACT level."""
        level = rlm_handler._extract_level("/api/debates/123/context/abstract")
        assert level == "ABSTRACT"

    def test_extract_level_detailed(self, rlm_handler):
        """Test extracting DETAILED level."""
        level = rlm_handler._extract_level("/api/debates/123/context/DETAILED")
        assert level == "DETAILED"

    def test_extract_level_invalid(self, rlm_handler):
        """Test extracting from invalid path returns None."""
        assert rlm_handler._extract_level("/api/debates/123/compress") is None


class TestRLMHandlerGetMethods:
    """Test GET endpoint handlers."""

    def test_handle_context_level_invalid_debate(self, rlm_handler, mock_http_handler):
        """Test context level with invalid debate ID returns 400."""
        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle("/api/debates//context/SUMMARY", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400

    def test_handle_refinement_status_invalid_debate(self, rlm_handler, mock_http_handler):
        """Test refinement status with invalid debate ID returns 400."""
        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle("/api/debates//refinement-status", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400

    def test_handle_query_rlm_get_returns_405(self, rlm_handler, mock_http_handler):
        """Test that GET on query-rlm returns 405."""
        result = rlm_handler.handle("/api/debates/123/query-rlm", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 405


class TestRLMHandlerQueryDebate:
    """Test POST /api/debates/{id}/query-rlm endpoint."""

    def test_query_debate_requires_body(self, rlm_handler):
        """Test query requires request body."""
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer test-token"}
        handler.get_json_body = MagicMock(return_value=None)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle_post(
                "/api/debates/debate-123/query-rlm", {}, handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "body required" in body["error"].lower()

    def test_query_debate_requires_query(self, rlm_handler):
        """Test query requires query parameter."""
        handler = create_auth_request_body({"strategy": "auto"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle_post(
                "/api/debates/debate-123/query-rlm", {}, handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "query" in body["error"].lower()

    def test_query_debate_invalid_id(self, rlm_handler):
        """Test query with invalid debate ID returns 400."""
        handler = create_auth_request_body({"query": "What was decided?"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle_post(
                "/api/debates//query-rlm", {}, handler
            )

        assert result is not None
        assert result.status_code == 400

    def test_query_debate_success(self, rlm_handler):
        """Test successful RLM query."""
        handler = create_auth_request_body({
            "query": "What was the consensus on pricing?",
            "strategy": "grep",
            "max_iterations": 2,
        })

        mock_result = MagicMock()
        mock_result.answer = "The consensus was to price at $99"
        mock_result.ready = True
        mock_result.iteration = 1
        mock_result.refinement_history = ["First attempt"]
        mock_result.confidence = 0.85
        mock_result.nodes_examined = ["node1", "node2"]
        mock_result.tokens_processed = 5000
        mock_result.sub_calls_made = 2

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            with patch.object(rlm_handler, '_execute_rlm_query', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = mock_result

                result = rlm_handler.handle_post(
                    "/api/debates/debate-123/query-rlm", {}, handler
                )

        assert result is not None
        body = json.loads(result.body)
        assert body["answer"] == "The consensus was to price at $99"
        assert body["ready"] is True
        assert body["confidence"] == 0.85


class TestRLMHandlerCompressDebate:
    """Test POST /api/debates/{id}/compress endpoint."""

    def test_compress_debate_invalid_id(self, rlm_handler):
        """Test compress with invalid debate ID returns 400."""
        handler = create_auth_request_body({})

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle_post(
                "/api/debates//compress", {}, handler
            )

        assert result is not None
        assert result.status_code == 400

    def test_compress_debate_success(self, rlm_handler):
        """Test successful debate compression."""
        handler = create_auth_request_body({
            "target_levels": ["ABSTRACT", "SUMMARY"],
            "compression_ratio": 0.2,
        })

        mock_compression_result = {
            "original_tokens": 50000,
            "compressed_tokens": {"ABSTRACT": 500, "SUMMARY": 2500},
            "compression_ratios": {"ABSTRACT": 0.01, "SUMMARY": 0.05},
            "time_seconds": 1.5,
            "levels_created": 2,
        }

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            with patch.object(rlm_handler, '_execute_compression', new_callable=AsyncMock) as mock_compress:
                mock_compress.return_value = mock_compression_result

                result = rlm_handler.handle_post(
                    "/api/debates/debate-123/compress", {}, handler
                )

        assert result is not None
        body = json.loads(result.body)
        assert body["original_tokens"] == 50000
        assert body["levels_created"] == 2

    def test_compress_debate_default_levels(self, rlm_handler):
        """Test compress uses default levels when not specified."""
        handler = create_auth_request_body({})

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            with patch.object(rlm_handler, '_execute_compression', new_callable=AsyncMock) as mock_compress:
                mock_compress.return_value = {
                    "original_tokens": 10000,
                    "compressed_tokens": {},
                    "compression_ratios": {},
                    "time_seconds": 0.5,
                    "levels_created": 3,
                }

                result = rlm_handler.handle_post(
                    "/api/debates/debate-123/compress", {}, handler
                )

                # Check that default levels were passed
                call_kwargs = mock_compress.call_args[1]
                assert "ABSTRACT" in call_kwargs["target_levels"]
                assert "SUMMARY" in call_kwargs["target_levels"]
                assert "DETAILED" in call_kwargs["target_levels"]


class TestRLMHandlerRefinementStatus:
    """Test GET /api/debates/{id}/refinement-status endpoint."""

    def test_refinement_status_success(self, rlm_handler, mock_http_handler):
        """Test getting refinement status."""
        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle(
                "/api/debates/debate-123/refinement-status", {}, mock_http_handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body["debate_id"] == "debate-123"
        assert "active_queries" in body
        assert "cached_contexts" in body
        assert body["status"] == "idle"


class TestRLMHandlerKnowledgeQuery:
    """Test POST /api/knowledge/query-rlm endpoint."""

    def test_knowledge_query_requires_body(self, rlm_handler):
        """Test knowledge query requires request body."""
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer test-token"}
        handler.get_json_body = MagicMock(return_value=None)

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle_post(
                "/api/knowledge/query-rlm", {}, handler
            )

        assert result is not None
        assert result.status_code == 400

    def test_knowledge_query_requires_workspace(self, rlm_handler):
        """Test knowledge query requires workspace_id."""
        handler = create_auth_request_body({"query": "What are the requirements?"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle_post(
                "/api/knowledge/query-rlm", {}, handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "workspace_id" in body["error"]

    def test_knowledge_query_requires_query(self, rlm_handler):
        """Test knowledge query requires query parameter."""
        handler = create_auth_request_body({"workspace_id": "ws_123"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            result = rlm_handler.handle_post(
                "/api/knowledge/query-rlm", {}, handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "query" in body["error"]

    def test_knowledge_query_success(self, rlm_handler):
        """Test successful knowledge query."""
        handler = create_auth_request_body({
            "workspace_id": "ws_123",
            "query": "What are the security requirements?",
            "max_nodes": 50,
        })

        mock_result = {
            "answer": "The security requirements include...",
            "sources": ["doc1", "doc2"],
            "confidence": 0.9,
            "ready": True,
            "iteration": 0,
        }

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            with patch.object(rlm_handler, '_execute_knowledge_query', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = mock_result

                result = rlm_handler.handle_post(
                    "/api/knowledge/query-rlm", {}, handler
                )

        assert result is not None
        body = json.loads(result.body)
        assert body["answer"] == "The security requirements include..."
        assert body["confidence"] == 0.9


class TestRLMHandlerIntegration:
    """Integration tests for RLM handler."""

    def test_routes_constant_matches_can_handle(self, rlm_handler):
        """Test that ROUTES constant matches can_handle logic."""
        for route in RLMHandler.ROUTES:
            # Replace placeholders with test values
            test_path = route.replace("{debate_id}", "test-123").replace("{level}", "SUMMARY")
            assert rlm_handler.can_handle(test_path), f"can_handle should return True for {test_path}"

    def test_handler_exports(self):
        """Test that RLMHandler is properly exported."""
        from aragora.server.handlers.features import RLMHandler as ExportedHandler
        assert ExportedHandler is RLMHandler

    def test_handler_inherits_base(self, rlm_handler):
        """Test that RLMHandler inherits from BaseHandler."""
        from aragora.server.handlers.base import BaseHandler
        assert isinstance(rlm_handler, BaseHandler)


class TestRLMHandlerErrorHandling:
    """Test error handling in RLM handler."""

    def test_query_debate_handles_exception(self, rlm_handler):
        """Test that query handles exceptions gracefully."""
        handler = create_auth_request_body({
            "query": "What was decided?",
        })

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            with patch.object(rlm_handler, '_execute_rlm_query', new_callable=AsyncMock) as mock_execute:
                mock_execute.side_effect = Exception("Database connection failed")

                result = rlm_handler.handle_post(
                    "/api/debates/debate-123/query-rlm", {}, handler
                )

        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert "failed" in body["error"].lower()

    def test_compress_debate_handles_exception(self, rlm_handler):
        """Test that compress handles exceptions gracefully."""
        handler = create_auth_request_body({})

        with patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_authenticated_user()):
            with patch.object(rlm_handler, '_execute_compression', new_callable=AsyncMock) as mock_compress:
                mock_compress.side_effect = ValueError("Debate not found")

                result = rlm_handler.handle_post(
                    "/api/debates/debate-123/compress", {}, handler
                )

        assert result is not None
        assert result.status_code == 500


class TestRLMHandlerAuthentication:
    """Test authentication requirements."""

    def test_query_debate_requires_auth(self, rlm_handler):
        """Test query endpoint requires authentication."""
        handler = create_request_body({"query": "test"})

        result = rlm_handler.handle_post(
            "/api/debates/debate-123/query-rlm", {}, handler
        )

        assert result is not None
        assert result.status_code == 401

    def test_compress_requires_auth(self, rlm_handler):
        """Test compress endpoint requires authentication."""
        handler = create_request_body({})

        result = rlm_handler.handle_post(
            "/api/debates/debate-123/compress", {}, handler
        )

        assert result is not None
        assert result.status_code == 401

    def test_knowledge_query_requires_auth(self, rlm_handler):
        """Test knowledge query requires authentication."""
        handler = create_request_body({
            "workspace_id": "ws_123",
            "query": "test",
        })

        result = rlm_handler.handle_post(
            "/api/knowledge/query-rlm", {}, handler
        )

        assert result is not None
        assert result.status_code == 401
