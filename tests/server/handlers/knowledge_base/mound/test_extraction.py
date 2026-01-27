"""
Tests for ExtractionOperationsMixin.

Tests knowledge extraction API endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.server.handlers.knowledge_base.mound.extraction import (
    ExtractionOperationsMixin,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockClaim:
    """Mock extracted claim."""

    id: str
    content: str
    confidence: float
    source_agent: str
    round: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "source_agent": self.source_agent,
            "round": self.round,
        }


@dataclass
class MockRelationship:
    """Mock extracted relationship."""

    source_id: str
    target_id: str
    relationship_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
        }


@dataclass
class MockExtractionResult:
    """Mock extraction result."""

    debate_id: str
    claims: List[MockClaim]
    relationships: List[MockRelationship]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "claims": [c.to_dict() for c in self.claims],
            "relationships": [r.to_dict() for r in self.relationships],
            "claim_count": len(self.claims),
            "relationship_count": len(self.relationships),
        }


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    extract_from_debate: AsyncMock = field(default_factory=AsyncMock)
    promote_extracted_knowledge: AsyncMock = field(default_factory=AsyncMock)
    get_extraction_stats: MagicMock = field(default_factory=MagicMock)


class TestExtractionHandler(ExtractionOperationsMixin):
    """Test implementation of ExtractionOperationsMixin."""

    def __init__(self, mound: Optional[MockKnowledgeMound] = None):
        self._mound = mound
        self.ctx = {}

    def _get_mound(self):
        return self._mound


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    return MockKnowledgeMound()


@pytest.fixture
def handler(mock_mound):
    """Create a test handler with mock mound."""
    return TestExtractionHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a test handler without mound."""
    return TestExtractionHandler(mound=None)


@pytest.fixture
def sample_messages():
    """Create sample debate messages."""
    return [
        {"agent_id": "claude", "content": "We should use microservices", "round": 1},
        {
            "agent_id": "gpt",
            "content": "I agree, microservices provide better scalability",
            "round": 1,
        },
        {"agent_id": "gemini", "content": "Consider the operational complexity", "round": 2},
    ]


# =============================================================================
# Test extract_from_debate
# =============================================================================


class TestExtractFromDebate:
    """Tests for extract_from_debate endpoint."""

    @pytest.mark.asyncio
    async def test_extract_success(self, handler, mock_mound, sample_messages):
        """Test successful extraction from debate."""
        mock_result = MockExtractionResult(
            debate_id="debate-123",
            claims=[
                MockClaim(
                    id="claim-1",
                    content="Microservices provide better scalability",
                    confidence=0.85,
                    source_agent="claude",
                    round=1,
                ),
                MockClaim(
                    id="claim-2",
                    content="Operational complexity is a consideration",
                    confidence=0.75,
                    source_agent="gemini",
                    round=2,
                ),
            ],
            relationships=[
                MockRelationship(
                    source_id="claim-1",
                    target_id="claim-2",
                    relationship_type="supports",
                ),
            ],
        )
        mock_mound.extract_from_debate.return_value = mock_result

        result = await handler.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
            consensus_text="Use microservices for scalability",
            topic="Architecture decision",
        )

        assert result.status_code == 200
        assert parse_response(result)["debate_id"] == "debate-123"
        assert parse_response(result)["claim_count"] == 2
        assert parse_response(result)["relationship_count"] == 1

    @pytest.mark.asyncio
    async def test_extract_without_consensus(self, handler, mock_mound, sample_messages):
        """Test extraction without consensus text."""
        mock_result = MockExtractionResult(
            debate_id="debate-123",
            claims=[],
            relationships=[],
        )
        mock_mound.extract_from_debate.return_value = mock_result

        result = await handler.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        assert result.status_code == 200
        mock_mound.extract_from_debate.assert_called_once_with(
            debate_id="debate-123",
            messages=sample_messages,
            consensus_text=None,
            topic=None,
        )

    @pytest.mark.asyncio
    async def test_extract_missing_debate_id(self, handler, sample_messages):
        """Test extraction with missing debate_id."""
        result = await handler.extract_from_debate(
            debate_id="",
            messages=sample_messages,
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_extract_missing_messages(self, handler):
        """Test extraction with missing messages."""
        result = await handler.extract_from_debate(
            debate_id="debate-123",
            messages=[],
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_extract_no_mound(self, handler_no_mound, sample_messages):
        """Test extraction when mound not available."""
        result = await handler_no_mound.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        assert result.status_code == 503
        assert "not available" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_extract_error(self, handler, mock_mound, sample_messages):
        """Test extraction error handling."""
        mock_mound.extract_from_debate.side_effect = Exception("Extraction failed")

        result = await handler.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        assert result.status_code == 500


# =============================================================================
# Test promote_extracted_knowledge
# =============================================================================


class TestPromoteExtractedKnowledge:
    """Tests for promote_extracted_knowledge endpoint."""

    @pytest.mark.asyncio
    async def test_promote_success(self, handler, mock_mound):
        """Test successful knowledge promotion."""
        mock_mound.promote_extracted_knowledge.return_value = 5

        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-123",
            min_confidence=0.7,
        )

        assert result.status_code == 200
        assert parse_response(result)["success"] is True
        assert parse_response(result)["promoted_count"] == 5
        assert parse_response(result)["workspace_id"] == "ws-123"
        assert parse_response(result)["min_confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_promote_with_default_confidence(self, handler, mock_mound):
        """Test promotion with default min_confidence."""
        mock_mound.promote_extracted_knowledge.return_value = 3

        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-123",
        )

        assert result.status_code == 200
        assert parse_response(result)["min_confidence"] == 0.6  # Default

    @pytest.mark.asyncio
    async def test_promote_zero_claims(self, handler, mock_mound):
        """Test promotion when no claims meet threshold."""
        mock_mound.promote_extracted_knowledge.return_value = 0

        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-123",
            min_confidence=0.99,
        )

        assert result.status_code == 200
        assert parse_response(result)["success"] is True
        assert parse_response(result)["promoted_count"] == 0

    @pytest.mark.asyncio
    async def test_promote_missing_workspace(self, handler):
        """Test promotion with missing workspace_id."""
        result = await handler.promote_extracted_knowledge(workspace_id="")

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_promote_no_mound(self, handler_no_mound):
        """Test promotion when mound not available."""
        result = await handler_no_mound.promote_extracted_knowledge(workspace_id="ws-123")

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_promote_error(self, handler, mock_mound):
        """Test promotion error handling."""
        mock_mound.promote_extracted_knowledge.side_effect = Exception("Database error")

        result = await handler.promote_extracted_knowledge(workspace_id="ws-123")

        assert result.status_code == 500


# =============================================================================
# Test get_extraction_stats
# =============================================================================


class TestGetExtractionStats:
    """Tests for get_extraction_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler, mock_mound):
        """Test successful stats retrieval."""
        mock_stats = {
            "total_debates_processed": 100,
            "total_claims_extracted": 500,
            "total_relationships_found": 200,
            "average_claims_per_debate": 5.0,
            "claims_promoted": 350,
            "by_confidence_level": {
                "high": 150,
                "medium": 200,
                "low": 150,
            },
        }
        mock_mound.get_extraction_stats.return_value = mock_stats

        result = await handler.get_extraction_stats()

        assert result.status_code == 200
        assert parse_response(result)["total_debates_processed"] == 100
        assert parse_response(result)["total_claims_extracted"] == 500
        assert parse_response(result)["average_claims_per_debate"] == 5.0

    @pytest.mark.asyncio
    async def test_get_stats_no_mound(self, handler_no_mound):
        """Test stats when mound not available."""
        result = await handler_no_mound.get_extraction_stats()

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_stats_error(self, handler, mock_mound):
        """Test stats error handling."""
        mock_mound.get_extraction_stats.side_effect = Exception("Stats error")

        result = await handler.get_extraction_stats()

        assert result.status_code == 500


# =============================================================================
# Integration Tests
# =============================================================================


class TestExtractionIntegration:
    """Integration tests for extraction workflow."""

    @pytest.mark.asyncio
    async def test_extract_then_promote_workflow(self, handler, mock_mound, sample_messages):
        """Test full extraction to promotion workflow."""
        # First, extract from debate
        mock_result = MockExtractionResult(
            debate_id="debate-123",
            claims=[
                MockClaim(
                    id="claim-1",
                    content="Important insight",
                    confidence=0.9,
                    source_agent="claude",
                    round=1,
                ),
            ],
            relationships=[],
        )
        mock_mound.extract_from_debate.return_value = mock_result

        extract_result = await handler.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        assert extract_result.status_code == 200
        assert parse_response(extract_result)["claim_count"] == 1

        # Then, promote extracted knowledge
        mock_mound.promote_extracted_knowledge.return_value = 1

        promote_result = await handler.promote_extracted_knowledge(
            workspace_id="ws-456",
            min_confidence=0.8,
        )

        assert promote_result.status_code == 200
        assert parse_response(promote_result)["promoted_count"] == 1
