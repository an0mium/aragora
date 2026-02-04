"""
Integration tests for Decision Plan ↔ Knowledge Mound bidirectional sync.

Verifies:
1. Plan outcomes flow INTO Knowledge Mound
2. Historical data flows OUT to enrich new plans
3. Lessons learned persist and are retrieved
4. Relationships between plans and debates are created
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.core_types import Critique, DebateResult
from aragora.pipeline.decision_plan.core import DecisionPlan, PlanStatus
from aragora.pipeline.decision_plan.factory import DecisionPlanFactory
from aragora.pipeline.risk_register import RiskLevel


@pytest.fixture
def sample_debate_result() -> DebateResult:
    """Create a sample debate result for testing."""
    return DebateResult(
        debate_id="debate-test-001",
        task="Implement user authentication with OAuth2",
        final_answer="""
1. Create OAuth2 provider configuration in `auth/providers.py`
2. Implement token validation in `auth/tokens.py`
3. Add middleware for protected routes in `server/middleware.py`
4. Create user session management in `auth/sessions.py`
""",
        confidence=0.85,
        consensus_reached=True,
        critiques=[
            Critique(
                agent="critic-1",
                target_agent="proposer-1",
                target_content="OAuth2 implementation",
                issues=["Consider token refresh strategy", "Handle rate limiting"],
                suggestions=["Add refresh token rotation", "Use exponential backoff"],
                severity=6.5,
                reasoning="Token refresh strategy is critical for security and UX",
            )
        ],
        dissenting_views=["Some concerns about token storage security"],
        debate_cruxes=[{"claim": "OAuth2 vs SAML choice"}],
        total_cost_usd=0.15,
    )


@pytest.fixture
def sample_plan_outcome():
    """Create a sample plan outcome for testing."""
    from aragora.pipeline.decision_plan import PlanOutcome

    return PlanOutcome(
        plan_id="plan-test-001",
        debate_id="debate-test-001",
        task="Implement user authentication with OAuth2",
        success=True,
        tasks_completed=4,
        tasks_total=4,
        verification_passed=3,
        verification_total=3,
        duration_seconds=120.5,
        total_cost_usd=0.25,
        lessons=[
            "Token rotation should happen before expiry, not after",
            "Rate limiting middleware should be added early in the pipeline",
        ],
        receipt_id="receipt-001",
        review_passed=True,
        review={"model": "gpt-4", "duration_seconds": 5.2},
    )


class TestPlanToKMFlow:
    """Tests for DecisionPlan → Knowledge Mound ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_plan_outcome_creates_knowledge_item(
        self, sample_debate_result, sample_plan_outcome
    ):
        """Test that plan outcomes are ingested as knowledge items."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
        )

        # Create plan
        plan = DecisionPlanFactory.from_debate_result(sample_debate_result)

        # Mock Knowledge Mound
        mock_km = MagicMock()
        mock_km.add_item = AsyncMock(return_value=MagicMock(id="km-item-001"))
        mock_km.add_relationship = AsyncMock()
        mock_km.search = AsyncMock(return_value=[])

        adapter = DecisionPlanAdapter(knowledge_mound=mock_km)

        # Ingest the outcome
        result = await adapter.ingest_plan_outcome(plan, sample_plan_outcome)

        # Verify ingestion succeeded
        assert result.success
        assert result.items_ingested >= 1
        assert "km-item-001" in result.knowledge_item_ids

        # Verify add_item was called
        mock_km.add_item.assert_called()

    @pytest.mark.asyncio
    async def test_ingest_plan_outcome_stores_lessons(
        self, sample_debate_result, sample_plan_outcome
    ):
        """Test that lessons learned are stored separately."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
        )

        plan = DecisionPlanFactory.from_debate_result(sample_debate_result)

        mock_km = MagicMock()
        mock_km.add_item = AsyncMock(return_value=MagicMock(id="km-item-001"))
        mock_km.add_relationship = AsyncMock()
        mock_km.search = AsyncMock(return_value=[])

        adapter = DecisionPlanAdapter(knowledge_mound=mock_km)
        result = await adapter.ingest_plan_outcome(plan, sample_plan_outcome)

        # 2 lessons in the outcome
        assert result.lessons_ingested == 2
        # Main item + 2 lessons = 3 items
        assert result.items_ingested == 1
        assert len(result.knowledge_item_ids) == 3

    @pytest.mark.asyncio
    async def test_ingest_creates_relationships(self, sample_debate_result, sample_plan_outcome):
        """Test that relationships are created between items."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
        )

        plan = DecisionPlanFactory.from_debate_result(sample_debate_result)

        mock_km = MagicMock()
        mock_km.add_item = AsyncMock(return_value=MagicMock(id="km-item-001"))
        mock_km.add_relationship = AsyncMock()
        mock_km.search = AsyncMock(return_value=[])

        adapter = DecisionPlanAdapter(knowledge_mound=mock_km)
        result = await adapter.ingest_plan_outcome(plan, sample_plan_outcome)

        # Should create relationships between main item and lessons
        assert result.relationships_created >= 2


class TestKMToPlanFlow:
    """Tests for Knowledge Mound → DecisionPlan enrichment."""

    @pytest.mark.asyncio
    async def test_factory_enriches_risks_from_history(self, sample_debate_result):
        """Test that factory enriches risks with historical data."""
        # Mock the decision plan adapter
        mock_similar_plans = [
            {
                "plan_id": "plan-old-001",
                "task": "Implement OAuth authentication",
                "success": False,
                "content": "Token handling failed due to rate limiting",
                "similarity": 0.85,
            },
            {
                "plan_id": "plan-old-002",
                "task": "Add OAuth2 login flow",
                "success": True,
                "content": "Successfully implemented with token refresh",
                "similarity": 0.82,
            },
        ]

        with patch(
            "aragora.knowledge.mound.adapters.decision_plan_adapter.get_decision_plan_adapter"
        ) as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.query_similar_plans = AsyncMock(return_value=mock_similar_plans)
            mock_adapter.get_lessons_for_domain = AsyncMock(return_value=[])
            mock_get_adapter.return_value = mock_adapter

            plan = await DecisionPlanFactory.from_debate_result_async(
                sample_debate_result,
                enrich_from_history=True,
            )

            # Verify historical enrichment was attempted
            mock_adapter.query_similar_plans.assert_called_once()

            # Check that risk register has historical context
            assert plan.risk_register is not None
            # With 50% success rate (1/2), should add a historical risk
            history_risks = [
                r
                for r in plan.risk_register.risks
                if "history" in r.id.lower() or "historical" in r.title.lower()
            ]
            assert len(history_risks) >= 1

    @pytest.mark.asyncio
    async def test_factory_retrieves_historical_lessons(self, sample_debate_result):
        """Test that factory retrieves and attaches historical lessons."""
        mock_lessons = [
            "Always implement token refresh before expiry",
            "Rate limiting should be handled at middleware level",
        ]

        with patch(
            "aragora.knowledge.mound.adapters.decision_plan_adapter.get_decision_plan_adapter"
        ) as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.query_similar_plans = AsyncMock(return_value=[])
            mock_adapter.get_lessons_for_domain = AsyncMock(return_value=mock_lessons)
            mock_get_adapter.return_value = mock_adapter

            plan = await DecisionPlanFactory.from_debate_result_async(
                sample_debate_result,
                enrich_from_history=True,
            )

            # Verify lessons were attached to metadata
            assert "historical_lessons" in plan.metadata
            assert len(plan.metadata["historical_lessons"]) == 2
            assert plan.metadata["historical_lessons_count"] == 2

    @pytest.mark.asyncio
    async def test_factory_handles_km_unavailable(self, sample_debate_result):
        """Test that factory handles KM being unavailable gracefully."""
        with patch(
            "aragora.knowledge.mound.adapters.decision_plan_adapter.get_decision_plan_adapter"
        ) as mock_get_adapter:
            mock_get_adapter.side_effect = ImportError("KM not available")

            # Should not raise, just skip enrichment
            plan = await DecisionPlanFactory.from_debate_result_async(
                sample_debate_result,
                enrich_from_history=True,
            )

            assert plan is not None
            assert plan.debate_id == "debate-test-001"


class TestBidirectionalRoundTrip:
    """Tests for complete round-trip: Plan → KM → New Plan."""

    @pytest.mark.asyncio
    async def test_ingested_outcome_enriches_future_plan(self, sample_debate_result):
        """Test that an ingested outcome can enrich a future similar plan."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
            PlanIngestionResult,
        )

        # Create initial plan and outcome
        plan1 = DecisionPlanFactory.from_debate_result(sample_debate_result)

        # Simulate successful ingestion
        ingestion_result = PlanIngestionResult(
            plan_id=plan1.id,
            items_ingested=3,
            lessons_ingested=2,
            relationships_created=2,
            knowledge_item_ids=["km-001", "km-002", "km-003"],
        )
        assert ingestion_result.success

        # Now create a new similar plan and verify it can be enriched
        # This simulates the historical data being available
        similar_result = DebateResult(
            debate_id="debate-test-002",
            task="Add OAuth2 authentication to the API",  # Similar task
            final_answer="Implement OAuth2 with refresh tokens",
            confidence=0.80,
            consensus_reached=True,
            total_cost_usd=0.10,
        )

        mock_historical = [
            {
                "plan_id": plan1.id,
                "task": plan1.task,
                "success": True,
                "content": "OAuth2 implementation succeeded",
                "similarity": 0.90,
            }
        ]

        with patch(
            "aragora.knowledge.mound.adapters.decision_plan_adapter.get_decision_plan_adapter"
        ) as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.query_similar_plans = AsyncMock(return_value=mock_historical)
            mock_adapter.get_lessons_for_domain = AsyncMock(
                return_value=["Token refresh should happen proactively"]
            )
            mock_get_adapter.return_value = mock_adapter

            plan2 = await DecisionPlanFactory.from_debate_result_async(
                similar_result,
                enrich_from_history=True,
            )

            # Verify the new plan was enriched with data from the old one
            assert plan2 is not None
            assert "historical_lessons" in plan2.metadata
            assert len(plan2.metadata["historical_lessons"]) >= 1


class TestAdapterQueryMethods:
    """Tests for Knowledge Mound query methods in the adapter."""

    @pytest.mark.asyncio
    async def test_query_similar_plans(self):
        """Test querying for similar historical plans."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
        )

        mock_km = MagicMock()
        mock_results = [
            MagicMock(
                metadata={"plan_id": "plan-001", "task": "OAuth impl", "success": True},
                content="OAuth implementation with tokens",
                score=0.85,
            )
        ]
        mock_km.search = AsyncMock(return_value=mock_results)

        adapter = DecisionPlanAdapter(knowledge_mound=mock_km)
        results = await adapter.query_similar_plans("Implement OAuth", limit=5)

        assert len(results) == 1
        assert results[0]["plan_id"] == "plan-001"
        assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_get_lessons_for_domain(self):
        """Test retrieving lessons for a specific domain."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
        )

        mock_km = MagicMock()
        mock_results = [
            MagicMock(
                content="[Lesson Learned]\nFrom Plan: xyz\nTask: auth\n\nAlways validate tokens server-side"
            )
        ]
        mock_km.search = AsyncMock(return_value=mock_results)

        adapter = DecisionPlanAdapter(knowledge_mound=mock_km)
        lessons = await adapter.get_lessons_for_domain("authentication", limit=5)

        assert len(lessons) == 1
        assert "validate tokens" in lessons[0]


class TestEventEmission:
    """Tests for event emission during plan ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_emits_event(self, sample_debate_result, sample_plan_outcome):
        """Test that plan ingestion emits an event."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
        )

        plan = DecisionPlanFactory.from_debate_result(sample_debate_result)

        mock_km = MagicMock()
        mock_km.add_item = AsyncMock(return_value=MagicMock(id="km-item-001"))
        mock_km.add_relationship = AsyncMock()
        mock_km.search = AsyncMock(return_value=[])

        events_received: list[tuple[str, dict]] = []

        def event_callback(event_type: str, data: dict):
            events_received.append((event_type, data))

        adapter = DecisionPlanAdapter(knowledge_mound=mock_km, event_callback=event_callback)
        await adapter.ingest_plan_outcome(plan, sample_plan_outcome)

        # Verify event was emitted
        assert len(events_received) == 1
        event_type, event_data = events_received[0]
        assert event_type == "plan_ingested"
        assert event_data["plan_id"] == plan.id
        assert event_data["success"] is True


class TestConfidenceLevelMapping:
    """Tests for confidence level calculation in knowledge items."""

    @pytest.mark.asyncio
    async def test_high_confidence_outcome(self, sample_debate_result):
        """Test that high success rate maps to HIGH confidence."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
        )
        from aragora.knowledge.unified.types import ConfidenceLevel
        from aragora.pipeline.decision_plan import PlanOutcome

        plan = DecisionPlanFactory.from_debate_result(sample_debate_result)

        # 100% success
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=4,
            tasks_total=4,
            verification_passed=4,
            verification_total=4,
            duration_seconds=60.0,
            total_cost_usd=0.1,
            lessons=[],
        )

        # Test the confidence calculation via the adapter's internal method
        adapter = DecisionPlanAdapter()
        item = adapter._create_plan_outcome_item(plan, outcome)

        assert item.confidence == ConfidenceLevel.HIGH

    @pytest.mark.asyncio
    async def test_low_confidence_outcome(self, sample_debate_result):
        """Test that low success rate maps to LOW confidence."""
        from aragora.knowledge.mound.adapters.decision_plan_adapter import (
            DecisionPlanAdapter,
        )
        from aragora.knowledge.unified.types import ConfidenceLevel
        from aragora.pipeline.decision_plan import PlanOutcome

        plan = DecisionPlanFactory.from_debate_result(sample_debate_result)

        # Low success
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=False,
            tasks_completed=1,
            tasks_total=4,
            verification_passed=0,
            verification_total=4,
            duration_seconds=60.0,
            total_cost_usd=0.1,
            lessons=[],
            error="Multiple failures",
        )

        adapter = DecisionPlanAdapter()
        item = adapter._create_plan_outcome_item(plan, outcome)

        assert item.confidence == ConfidenceLevel.LOW
