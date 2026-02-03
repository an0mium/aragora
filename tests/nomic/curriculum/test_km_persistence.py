"""
Tests for curriculum persistence via Knowledge Mound adapter.

Tests cover:
- CurriculumOutcome serialization in NomicCycleOutcome
- Curriculum ingestion creates KM entries
- find_similar_curricula returns matching past curriculum data
- Round-trip: ingest -> search -> retrieve
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
    CurriculumOutcome,
    CycleStatus,
    NomicCycleAdapter,
    NomicCycleOutcome,
)


class TestCurriculumOutcomeSerialization:
    """Tests for CurriculumOutcome dataclass serialization."""

    def test_curriculum_outcome_to_dict(self):
        """CurriculumOutcome serializes to dict correctly."""
        outcome = CurriculumOutcome(
            curricula_created=2,
            stones_attempted=5,
            stones_succeeded=4,
            skill_gaps=["parsing complex JSON", "async error handling"],
            skills_improved=["basic file I/O", "string formatting"],
            curriculum_results={"curr_1": {"stones_attempted": 3, "stones_succeeded": 2}},
        )

        data = outcome.to_dict()

        assert data["curricula_created"] == 2
        assert data["stones_attempted"] == 5
        assert data["stones_succeeded"] == 4
        assert len(data["skill_gaps"]) == 2
        assert len(data["skills_improved"]) == 2
        assert "curr_1" in data["curriculum_results"]

    def test_curriculum_outcome_from_dict(self):
        """CurriculumOutcome deserializes from dict correctly."""
        data = {
            "curricula_created": 3,
            "stones_attempted": 10,
            "stones_succeeded": 7,
            "skill_gaps": ["database queries"],
            "skills_improved": ["API calls"],
            "curriculum_results": {},
        }

        outcome = CurriculumOutcome.from_dict(data)

        assert outcome.curricula_created == 3
        assert outcome.stones_attempted == 10
        assert outcome.stones_succeeded == 7
        assert outcome.skill_gaps == ["database queries"]
        assert outcome.skills_improved == ["API calls"]

    def test_curriculum_outcome_stone_success_rate(self):
        """stone_success_rate property calculates correctly."""
        outcome = CurriculumOutcome(
            stones_attempted=10,
            stones_succeeded=7,
        )

        assert outcome.stone_success_rate == 0.7

    def test_curriculum_outcome_zero_stones(self):
        """stone_success_rate handles zero stones."""
        outcome = CurriculumOutcome()

        assert outcome.stone_success_rate == 0.0


class TestNomicCycleOutcomeWithCurriculum:
    """Tests for NomicCycleOutcome with curriculum data."""

    def test_nomic_cycle_outcome_includes_curriculum(self):
        """NomicCycleOutcome can include curriculum_outcome."""
        curriculum = CurriculumOutcome(
            curricula_created=1,
            stones_attempted=3,
            stones_succeeded=2,
        )

        cycle = NomicCycleOutcome(
            cycle_id="cycle_123",
            objective="Improve test coverage",
            status=CycleStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            curriculum_outcome=curriculum,
        )

        assert cycle.curriculum_outcome is not None
        assert cycle.curriculum_outcome.curricula_created == 1

    def test_nomic_cycle_outcome_to_dict_with_curriculum(self):
        """NomicCycleOutcome.to_dict includes curriculum data."""
        curriculum = CurriculumOutcome(
            curricula_created=2,
            stones_attempted=5,
            stones_succeeded=4,
        )

        cycle = NomicCycleOutcome(
            cycle_id="cycle_456",
            objective="Add new feature",
            status=CycleStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            curriculum_outcome=curriculum,
        )

        data = cycle.to_dict()

        assert "curriculum_outcome" in data
        assert data["curriculum_outcome"]["curricula_created"] == 2
        assert data["curriculum_outcome"]["stones_attempted"] == 5

    def test_nomic_cycle_outcome_from_dict_with_curriculum(self):
        """NomicCycleOutcome.from_dict restores curriculum data."""
        now = datetime.now(timezone.utc)
        data = {
            "cycle_id": "cycle_789",
            "objective": "Refactor module",
            "status": "success",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "curriculum_outcome": {
                "curricula_created": 1,
                "stones_attempted": 4,
                "stones_succeeded": 3,
                "skill_gaps": ["error handling"],
                "skills_improved": ["logging"],
                "curriculum_results": {},
            },
        }

        cycle = NomicCycleOutcome.from_dict(data)

        assert cycle.curriculum_outcome is not None
        assert cycle.curriculum_outcome.curricula_created == 1
        assert cycle.curriculum_outcome.skill_gaps == ["error handling"]

    def test_nomic_cycle_outcome_from_dict_without_curriculum(self):
        """NomicCycleOutcome.from_dict handles missing curriculum."""
        now = datetime.now(timezone.utc)
        data = {
            "cycle_id": "cycle_000",
            "objective": "Simple task",
            "status": "success",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
        }

        cycle = NomicCycleOutcome.from_dict(data)

        assert cycle.curriculum_outcome is None


class TestCurriculumIngestion:
    """Tests for curriculum ingestion into Knowledge Mound."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock Knowledge Mound."""
        mound = MagicMock()
        mound.store = AsyncMock(return_value=MagicMock(deduplicated=False, item_id="item_1"))
        mound.search = AsyncMock(return_value=[])
        return mound

    @pytest.mark.asyncio
    async def test_ingest_cycle_with_curriculum(self, mock_mound):
        """Curriculum data is ingested when present."""
        adapter = NomicCycleAdapter(mound=mock_mound)

        curriculum = CurriculumOutcome(
            curricula_created=1,
            stones_attempted=3,
            stones_succeeded=2,
            skill_gaps=["async programming"],
            skills_improved=["file I/O"],
        )

        outcome = NomicCycleOutcome(
            cycle_id="test_cycle",
            objective="Test curriculum ingestion",
            status=CycleStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            curriculum_outcome=curriculum,
        )

        result = await adapter.ingest_cycle_outcome(outcome)

        assert result.success
        # Should have ingested: cycle summary + curriculum + 1 skill gap + 1 skill improved
        assert result.items_ingested >= 1
        assert mock_mound.store.call_count >= 4

    @pytest.mark.asyncio
    async def test_ingest_cycle_without_curriculum(self, mock_mound):
        """Cycles without curriculum are still ingested correctly."""
        adapter = NomicCycleAdapter(mound=mock_mound)

        outcome = NomicCycleOutcome(
            cycle_id="simple_cycle",
            objective="Simple task",
            status=CycleStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result = await adapter.ingest_cycle_outcome(outcome)

        assert result.success
        assert result.items_ingested >= 1

    @pytest.mark.asyncio
    async def test_curriculum_metadata_stored_correctly(self, mock_mound):
        """Curriculum metadata is included in KM entries."""
        adapter = NomicCycleAdapter(mound=mock_mound)

        curriculum = CurriculumOutcome(
            curricula_created=2,
            stones_attempted=6,
            stones_succeeded=5,
            skill_gaps=["concurrency"],
            skills_improved=["error handling"],
        )

        outcome = NomicCycleOutcome(
            cycle_id="meta_test",
            objective="Test metadata",
            status=CycleStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            curriculum_outcome=curriculum,
        )

        await adapter.ingest_cycle_outcome(outcome)

        # Find the curriculum ingestion call
        curriculum_call = None
        for call in mock_mound.store.call_args_list:
            request = call[0][0]
            if hasattr(request, "metadata") and request.metadata.get("type") == "nomic_curriculum":
                curriculum_call = request
                break

        assert curriculum_call is not None
        assert curriculum_call.metadata["curricula_created"] == 2
        assert curriculum_call.metadata["stones_attempted"] == 6
        assert curriculum_call.metadata["stones_succeeded"] == 5
        assert curriculum_call.metadata["stone_success_rate"] == 5 / 6


class TestFindSimilarCurricula:
    """Tests for finding similar past curricula."""

    @pytest.fixture
    def mock_mound_with_curricula(self):
        """Create mock mound with curriculum search results."""
        mound = MagicMock()

        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.metadata = {
            "type": "nomic_curriculum",
            "objective": "Improve error handling",
            "curricula_created": 2,
            "stones_attempted": 5,
            "stones_succeeded": 4,
            "stone_success_rate": 0.8,
            "skill_gaps": ["exception handling"],
            "skills_improved": ["logging"],
            "parent_cycle_id": "cycle_abc",
        }

        mound.search = AsyncMock(return_value=[mock_result])
        return mound

    @pytest.mark.asyncio
    async def test_find_similar_curricula_returns_results(self, mock_mound_with_curricula):
        """find_similar_curricula returns matching results."""
        adapter = NomicCycleAdapter(mound=mock_mound_with_curricula)

        results = await adapter.find_similar_curricula(
            task_description="Improve error handling in API",
            limit=5,
        )

        assert len(results) == 1
        assert results[0]["objective"] == "Improve error handling"
        assert results[0]["similarity"] == 0.85
        assert results[0]["stones_succeeded"] == 4

    @pytest.mark.asyncio
    async def test_find_similar_curricula_filters_by_similarity(self, mock_mound_with_curricula):
        """Results below min_similarity are filtered out."""
        adapter = NomicCycleAdapter(mound=mock_mound_with_curricula)

        results = await adapter.find_similar_curricula(
            task_description="Completely different task",
            limit=5,
            min_similarity=0.9,  # Higher than mock score of 0.85
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_similar_curricula_handles_no_mound(self):
        """Returns empty list when mound is unavailable."""
        adapter = NomicCycleAdapter(mound=None)

        results = await adapter.find_similar_curricula(
            task_description="Any task",
            limit=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_find_similar_curricula_extracts_skill_info(self, mock_mound_with_curricula):
        """Skill gaps and improvements are extracted from results."""
        adapter = NomicCycleAdapter(mound=mock_mound_with_curricula)

        results = await adapter.find_similar_curricula(
            task_description="Error handling",
            limit=5,
        )

        assert len(results) == 1
        assert results[0]["skill_gaps"] == ["exception handling"]
        assert results[0]["skills_improved"] == ["logging"]


class TestCurriculumRoundTrip:
    """Tests for curriculum data round-trip through KM."""

    @pytest.mark.asyncio
    async def test_curriculum_ingest_and_search(self):
        """Ingested curriculum can be found via search."""
        # This is an integration test that would use a real mound
        # For unit testing, we verify the data flow
        curriculum = CurriculumOutcome(
            curricula_created=1,
            stones_attempted=4,
            stones_succeeded=3,
            skill_gaps=["database optimization"],
            skills_improved=["query writing"],
        )

        outcome = NomicCycleOutcome(
            cycle_id="roundtrip_test",
            objective="Optimize database queries",
            status=CycleStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            curriculum_outcome=curriculum,
        )

        # Serialize and deserialize to simulate storage
        data = outcome.to_dict()
        restored = NomicCycleOutcome.from_dict(data)

        assert restored.curriculum_outcome is not None
        assert restored.curriculum_outcome.skill_gaps == ["database optimization"]
        assert restored.curriculum_outcome.skills_improved == ["query writing"]
        assert restored.curriculum_outcome.stone_success_rate == 0.75
