"""
Tests for BusinessKnowledgeIngester extended methods.

Covers: customer tracking, decision recording, lesson recording,
customer history queries, and decision context queries.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora.memory.business import BusinessKnowledgeIngester
from aragora.memory.fabric import FabricResult, MemoryFabric, RememberResult


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


def _make_remember_result(stored: bool = True) -> RememberResult:
    return RememberResult(
        stored=stored,
        systems_written=["continuum"] if stored else [],
        surprise_combined=0.6 if stored else 0.1,
        reason="test",
    )


def _make_fabric_result(content: str = "result") -> FabricResult:
    return FabricResult(
        content=content,
        source_system="test",
        relevance=0.8,
        recency=0.9,
    )


@pytest.fixture()
def mock_fabric() -> AsyncMock:
    fabric = AsyncMock(spec=MemoryFabric)
    fabric.remember = AsyncMock(return_value=_make_remember_result(stored=True))
    fabric.query = AsyncMock(return_value=[])
    return fabric


# =====================================================================
# TestCustomerTracking
# =====================================================================


class TestCustomerTracking:
    """Tests for track_customer."""

    @pytest.mark.asyncio
    async def test_tracks_customer_interaction(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        result = await ingester.track_customer("C001", "Called about invoice")
        assert result.stored is True
        mock_fabric.remember.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_content_includes_customer_id(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.track_customer("C042", "Renewal request")
        content = mock_fabric.remember.call_args.kwargs["content"]
        assert "[customer:C042]" in content
        assert "Renewal request" in content

    @pytest.mark.asyncio
    async def test_sentiment_in_content(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.track_customer("C001", "Great feedback", sentiment=0.95)
        content = mock_fabric.remember.call_args.kwargs["content"]
        assert "0.95" in content

    @pytest.mark.asyncio
    async def test_metadata_includes_customer_id(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.track_customer("C099", "Complaint", sentiment=0.1)
        meta = mock_fabric.remember.call_args.kwargs["metadata"]
        assert meta["customer_id"] == "C099"
        assert meta["sentiment"] == 0.1
        assert meta["doc_type"] == "customer_interaction"

    @pytest.mark.asyncio
    async def test_fetches_existing_context(self, mock_fabric: AsyncMock) -> None:
        mock_fabric.query = AsyncMock(return_value=[_make_fabric_result("prior call")])
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.track_customer("C001", "Follow-up")
        mock_fabric.query.assert_awaited_once_with("customer C001", limit=5)
        # existing_context should be passed to remember
        existing = mock_fabric.remember.call_args.kwargs["existing_context"]
        assert "prior call" in existing

    @pytest.mark.asyncio
    async def test_default_sentiment(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.track_customer("C001", "Neutral call")
        content = mock_fabric.remember.call_args.kwargs["content"]
        assert "0.50" in content


# =====================================================================
# TestDecisionRecording
# =====================================================================


class TestDecisionRecording:
    """Tests for record_decision."""

    @pytest.mark.asyncio
    async def test_records_decision(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        result = await ingester.record_decision(
            decision="Switch to vendor B",
            context="Vendor A raised prices 30%",
        )
        assert result.stored is True

    @pytest.mark.asyncio
    async def test_content_includes_decision_and_context(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.record_decision(
            decision="Hire two engineers",
            context="Team is understaffed",
        )
        content = mock_fabric.remember.call_args.kwargs["content"]
        assert "[decision]" in content
        assert "Hire two engineers" in content
        assert "Team is understaffed" in content

    @pytest.mark.asyncio
    async def test_outcome_included_when_provided(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.record_decision(
            decision="Launch feature X",
            context="Market demand",
            outcome="Revenue up 15%",
        )
        content = mock_fabric.remember.call_args.kwargs["content"]
        assert "Revenue up 15%" in content

    @pytest.mark.asyncio
    async def test_outcome_omitted_when_none(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.record_decision(
            decision="Delay launch",
            context="Not ready",
        )
        content = mock_fabric.remember.call_args.kwargs["content"]
        assert "Outcome" not in content

    @pytest.mark.asyncio
    async def test_metadata_has_outcome_flag(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.record_decision("A", "B", outcome="C")
        meta1 = mock_fabric.remember.call_args.kwargs["metadata"]
        assert meta1["has_outcome"] is True

        mock_fabric.remember.reset_mock()
        await ingester.record_decision("D", "E")
        meta2 = mock_fabric.remember.call_args.kwargs["metadata"]
        assert meta2["has_outcome"] is False

    @pytest.mark.asyncio
    async def test_fetches_existing_decisions(self, mock_fabric: AsyncMock) -> None:
        mock_fabric.query = AsyncMock(return_value=[_make_fabric_result("prior decision")])
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.record_decision("New plan", "Cost reduction")
        mock_fabric.query.assert_awaited()
        existing = mock_fabric.remember.call_args.kwargs["existing_context"]
        assert "prior decision" in existing


# =====================================================================
# TestLessonRecording
# =====================================================================


class TestLessonRecording:
    """Tests for record_lesson."""

    @pytest.mark.asyncio
    async def test_records_lesson(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        result = await ingester.record_lesson(
            lesson="Always test in staging first",
            category="process",
            source="retrospective",
        )
        assert result.stored is True

    @pytest.mark.asyncio
    async def test_content_format(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.record_lesson(
            lesson="Cache invalidation is hard",
            category="technical",
            source="incident",
        )
        content = mock_fabric.remember.call_args.kwargs["content"]
        assert "[lesson:technical]" in content
        assert "Cache invalidation is hard" in content
        assert "Source: incident" in content

    @pytest.mark.asyncio
    async def test_metadata(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.record_lesson("L", "strategy", "planning")
        meta = mock_fabric.remember.call_args.kwargs["metadata"]
        assert meta["doc_type"] == "lesson_learned"
        assert meta["category"] == "strategy"
        assert meta["lesson_source"] == "planning"

    @pytest.mark.asyncio
    async def test_source_field_in_remember(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.record_lesson("L", "customer", "survey")
        assert mock_fabric.remember.call_args.kwargs["source"] == "business_lesson"


# =====================================================================
# TestCustomerHistory
# =====================================================================


class TestCustomerHistory:
    """Tests for get_customer_history."""

    @pytest.mark.asyncio
    async def test_delegates_to_fabric(self, mock_fabric: AsyncMock) -> None:
        expected = [_make_fabric_result("interaction 1")]
        mock_fabric.query = AsyncMock(return_value=expected)
        ingester = BusinessKnowledgeIngester(mock_fabric)
        results = await ingester.get_customer_history("C001")
        assert results == expected
        mock_fabric.query.assert_awaited_once_with(
            "customer C001 interaction", limit=20,
        )

    @pytest.mark.asyncio
    async def test_custom_limit(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.get_customer_history("C001", limit=5)
        mock_fabric.query.assert_awaited_once_with(
            "customer C001 interaction", limit=5,
        )

    @pytest.mark.asyncio
    async def test_empty_history(self, mock_fabric: AsyncMock) -> None:
        mock_fabric.query = AsyncMock(return_value=[])
        ingester = BusinessKnowledgeIngester(mock_fabric)
        results = await ingester.get_customer_history("C999")
        assert results == []


# =====================================================================
# TestDecisionContext
# =====================================================================


class TestDecisionContext:
    """Tests for get_decision_context."""

    @pytest.mark.asyncio
    async def test_delegates_to_fabric(self, mock_fabric: AsyncMock) -> None:
        expected = [_make_fabric_result("past decision")]
        mock_fabric.query = AsyncMock(return_value=expected)
        ingester = BusinessKnowledgeIngester(mock_fabric)
        results = await ingester.get_decision_context("pricing")
        assert results == expected
        mock_fabric.query.assert_awaited_once_with("decision pricing", limit=10)

    @pytest.mark.asyncio
    async def test_custom_limit(self, mock_fabric: AsyncMock) -> None:
        ingester = BusinessKnowledgeIngester(mock_fabric)
        await ingester.get_decision_context("hiring", limit=3)
        mock_fabric.query.assert_awaited_once_with("decision hiring", limit=3)
