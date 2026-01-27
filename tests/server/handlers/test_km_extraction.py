"""
Tests for Knowledge Mound Extraction handler endpoints.

Tests extraction operations:
- POST /api/knowledge/mound/extraction/debate - Extract from debate
- POST /api/knowledge/mound/extraction/promote - Promote extracted claims
- GET /api/knowledge/mound/extraction/stats - Get extraction statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.extraction import (
    ExtractionOperationsMixin,
)


@dataclass
class MockExtractionResult:
    """Mock extraction result for testing."""

    claims: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    debate_id: str = "debate-123"
    extraction_id: str = "extraction-456"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": self.claims,
            "relationships": self.relationships,
            "debate_id": self.debate_id,
            "extraction_id": self.extraction_id,
        }


class MockMound:
    """Mock KnowledgeMound for testing."""

    def __init__(self):
        self.extract_from_debate = AsyncMock(return_value=MockExtractionResult())
        self.promote_extracted_knowledge = AsyncMock(return_value=5)
        self.get_extraction_stats = MagicMock(
            return_value={
                "total_extractions": 100,
                "total_claims": 500,
                "pending_promotion": 50,
            }
        )


class MockExtractionHandler(ExtractionOperationsMixin):
    """Test handler that implements ExtractionOperationsMixin."""

    def __init__(self):
        self.mound = MockMound()
        self.ctx = {"user_id": "test-user", "org_id": "test-org"}

    def _get_mound(self):
        return self.mound


class MockExtractionHandlerNoMound(ExtractionOperationsMixin):
    """Test handler with no mound available."""

    def __init__(self):
        self.ctx = {}

    def _get_mound(self):
        return None


def parse_json_response(result):
    """Parse JSON response from HandlerResult dataclass."""
    import json

    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


@pytest.fixture
def handler():
    """Create test handler with mocked mound."""
    return MockExtractionHandler()


@pytest.fixture
def handler_no_mound():
    """Create test handler without mound."""
    return MockExtractionHandlerNoMound()


# Mock the decorators to bypass RBAC and rate limiting for tests
@pytest.fixture(autouse=True)
def mock_decorators():
    """Mock RBAC and rate limit decorators."""
    with (
        patch(
            "aragora.server.handlers.knowledge_base.mound.extraction.require_permission",
            lambda perm: lambda fn: fn,  # No-op decorator
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.extraction.rate_limit",
            lambda **kwargs: lambda fn: fn,  # No-op decorator
        ),
    ):
        yield


class TestExtractFromDebate:
    """Tests for extract_from_debate endpoint."""

    @pytest.mark.asyncio
    async def test_extract_success(self, handler):
        """Test successful extraction from debate."""
        messages = [
            {"agent_id": "agent-1", "content": "This is a claim", "round": 1},
            {"agent_id": "agent-2", "content": "I agree with that", "round": 1},
        ]

        # Create fresh handler to get decorator-free method
        test_handler = MockExtractionHandler()
        result = await test_handler.extract_from_debate(
            debate_id="debate-123",
            messages=messages,
            consensus_text="We reached agreement",
            topic="Test topic",
        )

        data = parse_json_response(result)
        assert data["debate_id"] == "debate-123"
        assert "claims" in data
        assert "relationships" in data

    @pytest.mark.asyncio
    async def test_extract_missing_debate_id(self, handler):
        """Test extraction fails without debate_id."""
        test_handler = MockExtractionHandler()
        result = await test_handler.extract_from_debate(
            debate_id="",
            messages=[{"content": "test"}],
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "debate_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_extract_missing_messages(self, handler):
        """Test extraction fails without messages."""
        test_handler = MockExtractionHandler()
        result = await test_handler.extract_from_debate(
            debate_id="debate-123",
            messages=[],
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "messages" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_extract_mound_unavailable(self, handler_no_mound):
        """Test extraction when mound is unavailable."""
        result = await handler_no_mound.extract_from_debate(
            debate_id="debate-123",
            messages=[{"content": "test"}],
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_extract_mound_error(self, handler):
        """Test extraction handles mound errors."""
        test_handler = MockExtractionHandler()
        test_handler.mound.extract_from_debate = AsyncMock(side_effect=Exception("Mound error"))

        result = await test_handler.extract_from_debate(
            debate_id="debate-123",
            messages=[{"content": "test"}],
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data

    @pytest.mark.asyncio
    async def test_extract_with_optional_params(self, handler):
        """Test extraction with optional parameters."""
        test_handler = MockExtractionHandler()
        result = await test_handler.extract_from_debate(
            debate_id="debate-456",
            messages=[{"agent_id": "agent-1", "content": "Test claim"}],
            consensus_text="Final consensus",
            topic="Important topic",
        )

        data = parse_json_response(result)
        assert data["debate_id"] == "debate-123"  # From mock

        # Verify mound was called with correct params
        test_handler.mound.extract_from_debate.assert_called_once_with(
            debate_id="debate-456",
            messages=[{"agent_id": "agent-1", "content": "Test claim"}],
            consensus_text="Final consensus",
            topic="Important topic",
        )


class TestPromoteExtractedKnowledge:
    """Tests for promote_extracted_knowledge endpoint."""

    @pytest.mark.asyncio
    async def test_promote_success(self, handler):
        """Test successful promotion of extracted knowledge."""
        test_handler = MockExtractionHandler()
        result = await test_handler.promote_extracted_knowledge(
            workspace_id="workspace-123",
            min_confidence=0.7,
        )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["workspace_id"] == "workspace-123"
        assert data["promoted_count"] == 5
        assert data["min_confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_promote_missing_workspace_id(self, handler):
        """Test promotion fails without workspace_id."""
        test_handler = MockExtractionHandler()
        result = await test_handler.promote_extracted_knowledge(
            workspace_id="",
            min_confidence=0.6,
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_promote_mound_unavailable(self, handler_no_mound):
        """Test promotion when mound is unavailable."""
        result = await handler_no_mound.promote_extracted_knowledge(
            workspace_id="workspace-123",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_promote_mound_error(self, handler):
        """Test promotion handles mound errors."""
        test_handler = MockExtractionHandler()
        test_handler.mound.promote_extracted_knowledge = AsyncMock(
            side_effect=Exception("Promotion error")
        )

        result = await test_handler.promote_extracted_knowledge(
            workspace_id="workspace-123",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data

    @pytest.mark.asyncio
    async def test_promote_with_claim_ids(self, handler):
        """Test promotion with specific claim IDs."""
        test_handler = MockExtractionHandler()
        result = await test_handler.promote_extracted_knowledge(
            workspace_id="workspace-123",
            claim_ids=["claim-1", "claim-2"],
            min_confidence=0.8,
        )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["min_confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_promote_default_confidence(self, handler):
        """Test promotion uses default min_confidence."""
        test_handler = MockExtractionHandler()
        result = await test_handler.promote_extracted_knowledge(
            workspace_id="workspace-123",
        )

        data = parse_json_response(result)
        assert data["min_confidence"] == 0.6  # Default value


class TestGetExtractionStats:
    """Tests for get_extraction_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler):
        """Test successful stats retrieval."""
        test_handler = MockExtractionHandler()
        result = await test_handler.get_extraction_stats()

        data = parse_json_response(result)
        assert data["total_extractions"] == 100
        assert data["total_claims"] == 500
        assert data["pending_promotion"] == 50

    @pytest.mark.asyncio
    async def test_get_stats_mound_unavailable(self, handler_no_mound):
        """Test stats when mound is unavailable."""
        result = await handler_no_mound.get_extraction_stats()

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_stats_mound_error(self, handler):
        """Test stats handles mound errors."""
        test_handler = MockExtractionHandler()
        test_handler.mound.get_extraction_stats = MagicMock(side_effect=Exception("Stats error"))

        result = await test_handler.get_extraction_stats()

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, handler):
        """Test stats with empty results."""
        test_handler = MockExtractionHandler()
        test_handler.mound.get_extraction_stats = MagicMock(
            return_value={
                "total_extractions": 0,
                "total_claims": 0,
                "pending_promotion": 0,
            }
        )

        result = await test_handler.get_extraction_stats()

        data = parse_json_response(result)
        assert data["total_extractions"] == 0
        assert data["total_claims"] == 0
