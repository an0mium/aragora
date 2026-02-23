"""Tests for ExtractionOperationsMixin (aragora/server/handlers/knowledge_base/mound/extraction.py).

Covers all routes and behavior of the extraction mixin:
- POST /api/knowledge/mound/extraction/debate  - Extract knowledge from a debate
- POST /api/knowledge/mound/extraction/promote - Promote extracted claims
- GET  /api/knowledge/mound/extraction/stats   - Get extraction statistics
- Error cases: missing mound, empty inputs, exceptions of each handled type
- Edge cases: optional parameters, boundary confidence values
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.server.handlers.knowledge_base.mound.extraction import (
    ExtractionOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock ExtractionResult (returned by mound.extract_from_debate)
# ---------------------------------------------------------------------------


@dataclass
class MockExtractionResult:
    """Mock extraction result with to_dict method."""

    debate_id: str = "debate-001"
    claims: list[dict[str, Any]] = field(default_factory=lambda: [
        {"claim_id": "c1", "text": "Claim one", "confidence": 0.9},
        {"claim_id": "c2", "text": "Claim two", "confidence": 0.7},
    ])
    relationships: list[dict[str, Any]] = field(default_factory=lambda: [
        {"source": "c1", "target": "c2", "type": "supports"},
    ])
    topic: str | None = "test topic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "claims": self.claims,
            "relationships": self.relationships,
            "topic": self.topic,
        }


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class ExtractionTestHandler(ExtractionOperationsMixin):
    """Concrete handler for testing the extraction mixin."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with extraction methods."""
    mound = MagicMock()
    mound.extract_from_debate = AsyncMock(return_value=MockExtractionResult())
    mound.promote_extracted_knowledge = AsyncMock(return_value=5)
    mound.get_extraction_stats = MagicMock(return_value={
        "total_extractions": 42,
        "total_claims": 128,
        "avg_confidence": 0.75,
        "debates_processed": 20,
    })
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create an ExtractionTestHandler with a mocked mound."""
    return ExtractionTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create an ExtractionTestHandler with no mound (returns None)."""
    return ExtractionTestHandler(mound=None)


@pytest.fixture
def sample_messages():
    """Return sample debate messages for extraction tests."""
    return [
        {"agent_id": "agent-1", "content": "I propose we use microservices.", "round": 1},
        {"agent_id": "agent-2", "content": "I disagree, monolith is better.", "round": 1},
        {"agent_id": "agent-1", "content": "Fair point, hybrid approach works.", "round": 2},
    ]


# ===========================================================================
# Tests for extract_from_debate
# ===========================================================================


class TestExtractFromDebate:
    """Tests for POST /api/knowledge/mound/extraction/debate."""

    @pytest.mark.asyncio
    async def test_successful_extraction(self, handler, mock_mound, sample_messages):
        """Happy path: extract from debate returns extraction result."""
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=sample_messages,
            consensus_text="Hybrid approach is best.",
            topic="Architecture decision",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-001"
        assert len(body["claims"]) == 2
        assert len(body["relationships"]) == 1
        mock_mound.extract_from_debate.assert_awaited_once_with(
            debate_id="debate-001",
            messages=sample_messages,
            consensus_text="Hybrid approach is best.",
            topic="Architecture decision",
        )

    @pytest.mark.asyncio
    async def test_extraction_without_optional_params(self, handler, mock_mound, sample_messages):
        """Extract succeeds when consensus_text and topic are None."""
        result = await handler.extract_from_debate(
            debate_id="debate-002",
            messages=sample_messages,
        )
        assert _status(result) == 200
        mock_mound.extract_from_debate.assert_awaited_once_with(
            debate_id="debate-002",
            messages=sample_messages,
            consensus_text=None,
            topic=None,
        )

    @pytest.mark.asyncio
    async def test_extraction_with_consensus_only(self, handler, mock_mound, sample_messages):
        """Extract with consensus_text but no topic."""
        result = await handler.extract_from_debate(
            debate_id="debate-003",
            messages=sample_messages,
            consensus_text="All agreed.",
        )
        assert _status(result) == 200
        mock_mound.extract_from_debate.assert_awaited_once_with(
            debate_id="debate-003",
            messages=sample_messages,
            consensus_text="All agreed.",
            topic=None,
        )

    @pytest.mark.asyncio
    async def test_extraction_with_topic_only(self, handler, mock_mound, sample_messages):
        """Extract with topic but no consensus_text."""
        result = await handler.extract_from_debate(
            debate_id="debate-004",
            messages=sample_messages,
            topic="Rate limiting",
        )
        assert _status(result) == 200
        mock_mound.extract_from_debate.assert_awaited_once_with(
            debate_id="debate-004",
            messages=sample_messages,
            consensus_text=None,
            topic="Rate limiting",
        )

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound, sample_messages):
        """No mound available returns 503."""
        result = await handler_no_mound.extract_from_debate(
            debate_id="debate-001",
            messages=sample_messages,
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_empty_debate_id_returns_400(self, handler, sample_messages):
        """Empty debate_id returns 400."""
        result = await handler.extract_from_debate(
            debate_id="",
            messages=sample_messages,
        )
        assert _status(result) == 400
        body = _body(result)
        assert "debate_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_none_debate_id_returns_400(self, handler, sample_messages):
        """None debate_id treated as falsy returns 400."""
        result = await handler.extract_from_debate(
            debate_id=None,
            messages=sample_messages,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_messages_returns_400(self, handler):
        """Empty messages list returns 400."""
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=[],
        )
        assert _status(result) == 400
        body = _body(result)
        assert "messages" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_none_messages_returns_400(self, handler):
        """None messages returns 400."""
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_key_error_returns_500(self, handler, mock_mound, sample_messages):
        """KeyError during extraction returns 500."""
        mock_mound.extract_from_debate = AsyncMock(side_effect=KeyError("missing_key"))
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=sample_messages,
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, handler, mock_mound, sample_messages):
        """ValueError during extraction returns 500."""
        mock_mound.extract_from_debate = AsyncMock(side_effect=ValueError("bad value"))
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=sample_messages,
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler, mock_mound, sample_messages):
        """OSError during extraction returns 500."""
        mock_mound.extract_from_debate = AsyncMock(side_effect=OSError("disk error"))
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=sample_messages,
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_type_error_returns_500(self, handler, mock_mound, sample_messages):
        """TypeError during extraction returns 500."""
        mock_mound.extract_from_debate = AsyncMock(side_effect=TypeError("wrong type"))
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=sample_messages,
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler, mock_mound, sample_messages):
        """RuntimeError during extraction returns 500."""
        mock_mound.extract_from_debate = AsyncMock(side_effect=RuntimeError("runtime issue"))
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=sample_messages,
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_attribute_error_returns_500(self, handler, mock_mound, sample_messages):
        """AttributeError during extraction returns 500."""
        mock_mound.extract_from_debate = AsyncMock(side_effect=AttributeError("no attr"))
        result = await handler.extract_from_debate(
            debate_id="debate-001",
            messages=sample_messages,
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_extraction_result_structure(self, handler, mock_mound, sample_messages):
        """Verify the structure of the returned extraction result."""
        custom_result = MockExtractionResult(
            debate_id="debate-xyz",
            claims=[{"claim_id": "c99", "text": "Only claim", "confidence": 0.95}],
            relationships=[],
            topic="Single claim test",
        )
        mock_mound.extract_from_debate = AsyncMock(return_value=custom_result)
        result = await handler.extract_from_debate(
            debate_id="debate-xyz",
            messages=sample_messages,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-xyz"
        assert len(body["claims"]) == 1
        assert body["claims"][0]["claim_id"] == "c99"
        assert body["claims"][0]["confidence"] == 0.95
        assert body["relationships"] == []
        assert body["topic"] == "Single claim test"

    @pytest.mark.asyncio
    async def test_extraction_single_message(self, handler, mock_mound):
        """Extract with a single message works."""
        single_msg = [{"agent_id": "a1", "content": "Only message", "round": 1}]
        result = await handler.extract_from_debate(
            debate_id="debate-single",
            messages=single_msg,
        )
        assert _status(result) == 200
        mock_mound.extract_from_debate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extraction_many_messages(self, handler, mock_mound):
        """Extract with many messages works."""
        many_msgs = [
            {"agent_id": f"agent-{i}", "content": f"Message {i}", "round": i % 5 + 1}
            for i in range(50)
        ]
        result = await handler.extract_from_debate(
            debate_id="debate-large",
            messages=many_msgs,
        )
        assert _status(result) == 200


# ===========================================================================
# Tests for promote_extracted_knowledge
# ===========================================================================


class TestPromoteExtractedKnowledge:
    """Tests for POST /api/knowledge/mound/extraction/promote."""

    @pytest.mark.asyncio
    async def test_successful_promotion(self, handler, mock_mound):
        """Happy path: promote returns count and metadata."""
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-001",
            claim_ids=["c1", "c2"],
            min_confidence=0.7,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["workspace_id"] == "ws-001"
        assert body["min_confidence"] == 0.7
        assert body["promoted_count"] == 5

    @pytest.mark.asyncio
    async def test_promotion_default_confidence(self, handler, mock_mound):
        """Promote uses default min_confidence=0.6 when not specified."""
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-002",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["min_confidence"] == 0.6
        mock_mound.promote_extracted_knowledge.assert_awaited_once_with(
            workspace_id="ws-002",
            claims=None,
            min_confidence=0.6,
        )

    @pytest.mark.asyncio
    async def test_promotion_no_claim_ids(self, handler, mock_mound):
        """Promote without claim_ids promotes all eligible claims."""
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-003",
            min_confidence=0.5,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["promoted_count"] == 5

    @pytest.mark.asyncio
    async def test_promotion_with_claim_ids(self, handler, mock_mound):
        """Promote with specific claim_ids still passes claims=None (current impl)."""
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-004",
            claim_ids=["c1", "c2", "c3"],
            min_confidence=0.8,
        )
        assert _status(result) == 200
        # Current implementation passes claims=None regardless of claim_ids
        mock_mound.promote_extracted_knowledge.assert_awaited_once_with(
            workspace_id="ws-004",
            claims=None,
            min_confidence=0.8,
        )

    @pytest.mark.asyncio
    async def test_promotion_zero_promoted(self, handler, mock_mound):
        """Promote returns 0 when nothing meets threshold."""
        mock_mound.promote_extracted_knowledge = AsyncMock(return_value=0)
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-005",
            min_confidence=0.99,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["promoted_count"] == 0
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_promotion_high_count(self, handler, mock_mound):
        """Promote returns large count successfully."""
        mock_mound.promote_extracted_knowledge = AsyncMock(return_value=1000)
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-006",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["promoted_count"] == 1000

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """No mound available returns 503."""
        result = await handler_no_mound.promote_extracted_knowledge(
            workspace_id="ws-001",
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_id_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.promote_extracted_knowledge(
            workspace_id="",
        )
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_none_workspace_id_returns_400(self, handler):
        """None workspace_id returns 400."""
        result = await handler.promote_extracted_knowledge(
            workspace_id=None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_key_error_returns_500(self, handler, mock_mound):
        """KeyError during promotion returns 500."""
        mock_mound.promote_extracted_knowledge = AsyncMock(side_effect=KeyError("ws"))
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-001",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, handler, mock_mound):
        """ValueError during promotion returns 500."""
        mock_mound.promote_extracted_knowledge = AsyncMock(
            side_effect=ValueError("invalid confidence")
        )
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-001",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler, mock_mound):
        """OSError during promotion returns 500."""
        mock_mound.promote_extracted_knowledge = AsyncMock(side_effect=OSError("storage fail"))
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-001",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_type_error_returns_500(self, handler, mock_mound):
        """TypeError during promotion returns 500."""
        mock_mound.promote_extracted_knowledge = AsyncMock(side_effect=TypeError("bad type"))
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-001",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError during promotion returns 500."""
        mock_mound.promote_extracted_knowledge = AsyncMock(
            side_effect=RuntimeError("failed")
        )
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-001",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError during promotion returns 500."""
        mock_mound.promote_extracted_knowledge = AsyncMock(
            side_effect=AttributeError("no attr")
        )
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-001",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_promotion_min_confidence_zero(self, handler, mock_mound):
        """Promote with min_confidence=0.0 passes through."""
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-007",
            min_confidence=0.0,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["min_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_promotion_min_confidence_one(self, handler, mock_mound):
        """Promote with min_confidence=1.0 passes through."""
        mock_mound.promote_extracted_knowledge = AsyncMock(return_value=0)
        result = await handler.promote_extracted_knowledge(
            workspace_id="ws-008",
            min_confidence=1.0,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["min_confidence"] == 1.0
        assert body["promoted_count"] == 0


# ===========================================================================
# Tests for get_extraction_stats
# ===========================================================================


class TestGetExtractionStats:
    """Tests for GET /api/knowledge/mound/extraction/stats."""

    @pytest.mark.asyncio
    async def test_successful_stats(self, handler, mock_mound):
        """Happy path: returns extraction statistics."""
        result = await handler.get_extraction_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_extractions"] == 42
        assert body["total_claims"] == 128
        assert body["avg_confidence"] == 0.75
        assert body["debates_processed"] == 20

    @pytest.mark.asyncio
    async def test_stats_empty(self, handler, mock_mound):
        """Stats with zero values."""
        mock_mound.get_extraction_stats = MagicMock(return_value={
            "total_extractions": 0,
            "total_claims": 0,
            "avg_confidence": 0.0,
            "debates_processed": 0,
        })
        result = await handler.get_extraction_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_extractions"] == 0
        assert body["total_claims"] == 0

    @pytest.mark.asyncio
    async def test_stats_large_numbers(self, handler, mock_mound):
        """Stats with large values serialize correctly."""
        mock_mound.get_extraction_stats = MagicMock(return_value={
            "total_extractions": 999999,
            "total_claims": 5000000,
            "avg_confidence": 0.823456,
            "debates_processed": 100000,
        })
        result = await handler.get_extraction_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_extractions"] == 999999
        assert body["total_claims"] == 5000000

    @pytest.mark.asyncio
    async def test_stats_additional_fields(self, handler, mock_mound):
        """Stats with extra fields are passed through."""
        mock_mound.get_extraction_stats = MagicMock(return_value={
            "total_extractions": 10,
            "custom_field": "custom_value",
            "nested": {"key": "val"},
        })
        result = await handler.get_extraction_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["custom_field"] == "custom_value"
        assert body["nested"]["key"] == "val"

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """No mound available returns 503."""
        result = await handler_no_mound.get_extraction_stats()
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_key_error_returns_500(self, handler, mock_mound):
        """KeyError in stats returns 500."""
        mock_mound.get_extraction_stats = MagicMock(side_effect=KeyError("stats_key"))
        result = await handler.get_extraction_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, handler, mock_mound):
        """ValueError in stats returns 500."""
        mock_mound.get_extraction_stats = MagicMock(side_effect=ValueError("bad"))
        result = await handler.get_extraction_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler, mock_mound):
        """OSError in stats returns 500."""
        mock_mound.get_extraction_stats = MagicMock(side_effect=OSError("io"))
        result = await handler.get_extraction_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_type_error_returns_500(self, handler, mock_mound):
        """TypeError in stats returns 500."""
        mock_mound.get_extraction_stats = MagicMock(side_effect=TypeError("type"))
        result = await handler.get_extraction_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError in stats returns 500."""
        mock_mound.get_extraction_stats = MagicMock(side_effect=RuntimeError("fail"))
        result = await handler.get_extraction_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError in stats returns 500."""
        mock_mound.get_extraction_stats = MagicMock(side_effect=AttributeError("attr"))
        result = await handler.get_extraction_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_stats_calls_mound_correctly(self, handler, mock_mound):
        """Verify get_extraction_stats calls mound.get_extraction_stats()."""
        await handler.get_extraction_stats()
        mock_mound.get_extraction_stats.assert_called_once_with()


# ===========================================================================
# Tests for mixin protocol / structure
# ===========================================================================


class TestExtractionMixinStructure:
    """Tests for the mixin class structure and protocol."""

    def test_mixin_has_extract_method(self):
        """Mixin exposes extract_from_debate method."""
        assert hasattr(ExtractionOperationsMixin, "extract_from_debate")

    def test_mixin_has_promote_method(self):
        """Mixin exposes promote_extracted_knowledge method."""
        assert hasattr(ExtractionOperationsMixin, "promote_extracted_knowledge")

    def test_mixin_has_stats_method(self):
        """Mixin exposes get_extraction_stats method."""
        assert hasattr(ExtractionOperationsMixin, "get_extraction_stats")

    def test_mixin_has_get_mound_stub(self):
        """Mixin includes a _get_mound stub method."""
        assert hasattr(ExtractionOperationsMixin, "_get_mound")

    def test_handler_inherits_mixin(self):
        """Test handler correctly inherits from mixin."""
        handler = ExtractionTestHandler(mound=None)
        assert isinstance(handler, ExtractionOperationsMixin)

    def test_handler_get_mound_returns_none(self):
        """Handler with no mound returns None from _get_mound."""
        handler = ExtractionTestHandler(mound=None)
        assert handler._get_mound() is None

    def test_handler_get_mound_returns_mound(self):
        """Handler with mound returns it from _get_mound."""
        mound = MagicMock()
        handler = ExtractionTestHandler(mound=mound)
        assert handler._get_mound() is mound
