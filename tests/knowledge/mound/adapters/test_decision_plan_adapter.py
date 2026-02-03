"""
Tests for DecisionPlanAdapter - bridges Decision Plans to Knowledge Mound.

Tests cover:
- ingest_plan_outcome with mock KM
- query_similar_plans returns correctly shaped data
- get_lessons_for_domain filters correctly
- PlanIngestionResult dataclass
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.decision_plan_adapter import (
    DecisionPlanAdapter,
    PlanIngestionResult,
    get_decision_plan_adapter,
)


class TestPlanIngestionResult:
    """Tests for PlanIngestionResult dataclass."""

    def test_result_success_when_items_ingested(self):
        """success is True when items were ingested with no errors."""
        result = PlanIngestionResult(
            plan_id="plan_123",
            items_ingested=2,
            lessons_ingested=1,
            knowledge_item_ids=["item_1", "item_2"],
        )

        assert result.success is True

    def test_result_failure_when_errors(self):
        """success is False when errors occurred."""
        result = PlanIngestionResult(
            plan_id="plan_456",
            items_ingested=1,
            errors=["KM unavailable"],
        )

        assert result.success is False

    def test_result_failure_when_no_items(self):
        """success is False when no items were ingested."""
        result = PlanIngestionResult(
            plan_id="plan_789",
            items_ingested=0,
        )

        assert result.success is False

    def test_result_to_dict(self):
        """to_dict serializes all fields."""
        result = PlanIngestionResult(
            plan_id="plan_test",
            items_ingested=3,
            lessons_ingested=2,
            relationships_created=1,
            knowledge_item_ids=["a", "b", "c"],
            errors=[],
        )

        data = result.to_dict()

        assert data["plan_id"] == "plan_test"
        assert data["items_ingested"] == 3
        assert data["lessons_ingested"] == 2
        assert data["success"] is True


class TestDecisionPlanAdapterInit:
    """Tests for DecisionPlanAdapter initialization."""

    def test_init_with_km(self):
        """Adapter initializes with provided KM."""
        mock_km = MagicMock()
        adapter = DecisionPlanAdapter(knowledge_mound=mock_km)

        assert adapter.knowledge_mound is mock_km

    def test_init_without_km(self):
        """Adapter handles missing KM gracefully."""
        adapter = DecisionPlanAdapter(knowledge_mound=None)

        # Should not raise, may return None
        _ = adapter.knowledge_mound


class TestIngestPlanOutcome:
    """Tests for ingest_plan_outcome method."""

    @pytest.fixture
    def mock_km(self):
        """Create a mock Knowledge Mound."""
        km = MagicMock()
        km.add_item = AsyncMock(return_value=MagicMock(id="item_123"))
        km.add_relationship = AsyncMock(return_value=True)
        km.search = AsyncMock(return_value=[])
        return km

    @pytest.fixture
    def mock_plan(self):
        """Create a mock DecisionPlan."""
        plan = MagicMock()
        plan.id = "plan_test_123"
        plan.task = "Implement new feature"
        plan.debate_id = "debate_456"
        return plan

    @pytest.fixture
    def mock_outcome(self):
        """Create a mock PlanOutcome."""
        outcome = MagicMock()
        outcome.success = True
        outcome.tasks_completed = 5
        outcome.tasks_total = 5
        outcome.verification_passed = 10
        outcome.verification_total = 10
        outcome.duration_seconds = 120.0
        outcome.total_cost_usd = 0.50
        outcome.receipt_id = "receipt_789"
        outcome.error = None
        outcome.lessons = ["Use caching for performance", "Add retry logic"]
        return outcome

    @pytest.mark.asyncio
    async def test_ingest_plan_creates_items(self, mock_km, mock_plan, mock_outcome):
        """Ingestion creates plan outcome and lesson items."""
        adapter = DecisionPlanAdapter(knowledge_mound=mock_km)

        result = await adapter.ingest_plan_outcome(mock_plan, mock_outcome)

        assert result.success
        assert result.items_ingested >= 1  # At least main outcome
        assert mock_km.add_item.called

    @pytest.mark.asyncio
    async def test_ingest_plan_creates_lessons(self, mock_km, mock_plan, mock_outcome):
        """Lessons are ingested as separate items."""
        adapter = DecisionPlanAdapter(knowledge_mound=mock_km)

        result = await adapter.ingest_plan_outcome(mock_plan, mock_outcome)

        # Should have main item + 2 lessons
        assert result.lessons_ingested == 2

    @pytest.mark.asyncio
    async def test_ingest_plan_handles_no_km(self, mock_plan, mock_outcome):
        """Gracefully handles missing Knowledge Mound."""
        # Create adapter with explicit None and mock the property to stay None
        adapter = DecisionPlanAdapter(knowledge_mound=None)
        adapter._km = None  # Ensure it's None

        # Mock the property to return None
        with pytest.MonkeyPatch.context() as m:
            m.setattr(type(adapter), "knowledge_mound", property(lambda self: None))
            result = await adapter.ingest_plan_outcome(mock_plan, mock_outcome)

        assert result.success is False
        assert "Knowledge Mound not available" in result.errors

    @pytest.mark.asyncio
    async def test_ingest_plan_emits_event(self, mock_km, mock_plan, mock_outcome):
        """Event is emitted after successful ingestion."""
        events = []

        def capture_event(event_type: str, data: dict):
            events.append((event_type, data))

        adapter = DecisionPlanAdapter(
            knowledge_mound=mock_km,
            event_callback=capture_event,
        )

        await adapter.ingest_plan_outcome(mock_plan, mock_outcome)

        assert len(events) == 1
        assert events[0][0] == "plan_ingested"
        assert events[0][1]["plan_id"] == "plan_test_123"


class TestQuerySimilarPlans:
    """Tests for query_similar_plans method."""

    @pytest.fixture
    def mock_km_with_results(self):
        """Create mock KM with search results."""
        km = MagicMock()

        mock_result = MagicMock()
        mock_result.metadata = {
            "plan_id": "plan_old_123",
            "task": "Similar task description",
            "success": True,
        }
        mock_result.content = "Plan outcome content..."
        mock_result.score = 0.85

        km.search = AsyncMock(return_value=[mock_result])
        return km

    @pytest.mark.asyncio
    async def test_query_returns_similar_plans(self, mock_km_with_results):
        """query_similar_plans returns formatted results."""
        adapter = DecisionPlanAdapter(knowledge_mound=mock_km_with_results)

        results = await adapter.query_similar_plans(
            task="Implement similar feature",
            limit=5,
        )

        assert len(results) == 1
        assert results[0]["plan_id"] == "plan_old_123"
        assert results[0]["success"] is True
        assert results[0]["similarity"] == 0.85

    @pytest.mark.asyncio
    async def test_query_filters_by_decision_plan_tag(self, mock_km_with_results):
        """Search filters by decision_plan and outcome tags."""
        adapter = DecisionPlanAdapter(knowledge_mound=mock_km_with_results)

        await adapter.query_similar_plans(task="Test task", limit=3)

        mock_km_with_results.search.assert_called_once()
        call_kwargs = mock_km_with_results.search.call_args[1]
        assert "decision_plan" in call_kwargs["filters"]["tags"]
        assert "outcome" in call_kwargs["filters"]["tags"]

    @pytest.mark.asyncio
    async def test_query_handles_no_km(self):
        """Returns empty list when KM unavailable."""
        adapter = DecisionPlanAdapter(knowledge_mound=None)

        results = await adapter.query_similar_plans(task="Any task", limit=5)

        assert results == []


class TestGetLessonsForDomain:
    """Tests for get_lessons_for_domain method."""

    @pytest.fixture
    def mock_km_with_lessons(self):
        """Create mock KM with lesson search results."""
        km = MagicMock()

        mock_result = MagicMock()
        mock_result.content = """[Lesson Learned]
From Plan: plan_123
Task: Implement caching

Always add cache invalidation when implementing new cache layers."""

        km.search = AsyncMock(return_value=[mock_result])
        return km

    @pytest.mark.asyncio
    async def test_get_lessons_extracts_content(self, mock_km_with_lessons):
        """Lessons are extracted from KM items."""
        adapter = DecisionPlanAdapter(knowledge_mound=mock_km_with_lessons)

        lessons = await adapter.get_lessons_for_domain(domain="caching", limit=10)

        assert len(lessons) == 1
        assert "cache invalidation" in lessons[0]

    @pytest.mark.asyncio
    async def test_get_lessons_filters_by_lesson_tag(self, mock_km_with_lessons):
        """Search filters by lesson_learned tag."""
        adapter = DecisionPlanAdapter(knowledge_mound=mock_km_with_lessons)

        await adapter.get_lessons_for_domain(domain="api", limit=5)

        mock_km_with_lessons.search.assert_called_once()
        call_kwargs = mock_km_with_lessons.search.call_args[1]
        assert "lesson_learned" in call_kwargs["filters"]["tags"]

    @pytest.mark.asyncio
    async def test_get_lessons_handles_no_km(self):
        """Returns empty list when KM unavailable."""
        adapter = DecisionPlanAdapter(knowledge_mound=None)

        lessons = await adapter.get_lessons_for_domain(domain="any", limit=5)

        assert lessons == []


class TestGetDecisionPlanAdapter:
    """Tests for the singleton accessor."""

    def test_get_adapter_creates_instance(self):
        """get_decision_plan_adapter creates an adapter."""
        adapter = get_decision_plan_adapter()

        assert adapter is not None
        assert isinstance(adapter, DecisionPlanAdapter)

    def test_get_adapter_with_km(self):
        """get_decision_plan_adapter accepts KM parameter."""
        mock_km = MagicMock()
        adapter = get_decision_plan_adapter(knowledge_mound=mock_km)

        assert adapter.knowledge_mound is mock_km
