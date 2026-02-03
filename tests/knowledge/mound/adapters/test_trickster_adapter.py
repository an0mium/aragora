"""
Tests for TricksterAdapter - Knowledge Mound integration for Trickster interventions.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from aragora.knowledge.mound.adapters.trickster_adapter import (
    TricksterAdapter,
    TricksterSearchResult,
    InterventionRecord,
)


class TestTricksterAdapter:
    """Test suite for TricksterAdapter."""

    def test_init_without_trickster(self) -> None:
        """Test adapter can be created without a trickster instance."""
        adapter = TricksterAdapter()
        assert adapter.trickster is None
        assert len(adapter._records) == 0

    def test_init_with_trickster(self) -> None:
        """Test adapter can be created with a trickster instance."""
        mock_trickster = MagicMock()
        adapter = TricksterAdapter(trickster=mock_trickster)
        assert adapter.trickster is mock_trickster

    def test_set_trickster(self) -> None:
        """Test setting trickster after construction."""
        adapter = TricksterAdapter()
        mock_trickster = MagicMock()
        adapter.set_trickster(mock_trickster)
        assert adapter.trickster is mock_trickster

    @pytest.mark.asyncio
    async def test_persist_debate_interventions(self) -> None:
        """Test persisting interventions from a trickster instance."""
        from aragora.debate.trickster import (
            EvidencePoweredTrickster,
            TricksterIntervention,
            InterventionType,
        )

        trickster = EvidencePoweredTrickster()
        # Add a mock intervention to state
        trickster._state.interventions.append(
            TricksterIntervention(
                intervention_type=InterventionType.CHALLENGE_PROMPT,
                round_num=2,
                target_agents=["agent1", "agent2"],
                challenge_text="Please provide more evidence for your claims.",
                evidence_gaps={"agent1": ["missing citation"]},
                priority=0.8,
                metadata={"test": True},
            )
        )

        adapter = TricksterAdapter(trickster=trickster)
        count = await adapter.persist_debate_interventions(
            debate_id="debate_123",
            topic="API rate limiting",
            domain="engineering",
        )

        assert count == 1
        assert len(adapter._records) == 1
        assert "debate_123" in adapter._by_debate
        assert "engineering" in adapter._by_domain

        # Check record content
        record = list(adapter._records.values())[0]
        assert record.debate_id == "debate_123"
        assert record.topic == "API rate limiting"
        assert record.domain == "engineering"
        assert record.intervention_type == "challenge_prompt"
        assert record.priority == 0.8

    @pytest.mark.asyncio
    async def test_search_by_topic(self) -> None:
        """Test searching for similar interventions by topic."""
        adapter = TricksterAdapter()

        # Add some test records
        adapter._records["tr_1"] = InterventionRecord(
            id="tr_1",
            debate_id="d1",
            domain="engineering",
            topic="rate limiting implementation",
            intervention_type="challenge_prompt",
            round_num=2,
            target_agents=["agent1"],
            challenge_text="Need more evidence for rate limits",
            evidence_gaps={},
            priority=0.7,
            timestamp=datetime.now(),
        )
        adapter._records["tr_2"] = InterventionRecord(
            id="tr_2",
            debate_id="d2",
            domain="engineering",
            topic="database optimization",
            intervention_type="evidence_gap",
            round_num=3,
            target_agents=["agent2"],
            challenge_text="Please cite sources for optimization claims",
            evidence_gaps={},
            priority=0.5,
            timestamp=datetime.now(),
        )

        # Search for rate limiting
        results = await adapter.search_by_topic("rate limiting", limit=10)
        assert len(results) == 1
        assert results[0].topic == "rate limiting implementation"
        assert results[0].similarity > 0

    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self) -> None:
        """Test searching with domain filter."""
        adapter = TricksterAdapter()

        adapter._records["tr_1"] = InterventionRecord(
            id="tr_1",
            debate_id="d1",
            domain="engineering",
            topic="API design",
            intervention_type="challenge_prompt",
            round_num=1,
            target_agents=["a1"],
            challenge_text="test",
            evidence_gaps={},
            priority=0.8,
            timestamp=datetime.now(),
        )
        adapter._records["tr_2"] = InterventionRecord(
            id="tr_2",
            debate_id="d2",
            domain="product",
            topic="API design",
            intervention_type="challenge_prompt",
            round_num=1,
            target_agents=["a2"],
            challenge_text="test",
            evidence_gaps={},
            priority=0.8,
            timestamp=datetime.now(),
        )

        # Search with domain filter
        results = await adapter.search_by_topic("API design", domain="engineering")
        assert len(results) == 1
        assert results[0].domain == "engineering"

    def test_get_record(self) -> None:
        """Test getting a specific record by ID."""
        adapter = TricksterAdapter()
        adapter._records["tr_test123"] = InterventionRecord(
            id="tr_test123",
            debate_id="d1",
            domain="test",
            topic="test topic",
            intervention_type="challenge_prompt",
            round_num=1,
            target_agents=[],
            challenge_text="test",
            evidence_gaps={},
            priority=0.5,
            timestamp=datetime.now(),
        )

        # Get with full ID
        record = adapter.get("tr_test123")
        assert record is not None
        assert record.id == "tr_test123"

        # Non-existent record
        assert adapter.get("nonexistent") is None

    def test_to_knowledge_item(self) -> None:
        """Test converting intervention record to KnowledgeItem."""
        from aragora.knowledge.mound.types import ConfidenceLevel, KnowledgeSource

        adapter = TricksterAdapter()
        record = InterventionRecord(
            id="tr_test",
            debate_id="d1",
            domain="engineering",
            topic="rate limiting",
            intervention_type="challenge_prompt",
            round_num=2,
            target_agents=["agent1", "agent2"],
            challenge_text="Please provide evidence for rate limits",
            evidence_gaps={"agent1": ["missing citation"]},
            priority=0.85,
            timestamp=datetime.now(),
        )

        item = adapter.to_knowledge_item(record)

        assert item.id == "tr_test"
        assert "challenge_prompt" in item.content
        assert item.source == KnowledgeSource.INSIGHT
        assert item.confidence == ConfidenceLevel.HIGH  # priority >= 0.8
        assert item.importance == 0.85
        assert item.metadata["domain"] == "engineering"
        assert item.metadata["intervention_type"] == "challenge_prompt"

    def test_get_domain_patterns(self) -> None:
        """Test getting domain intervention patterns."""
        adapter = TricksterAdapter()

        # Add multiple records for same domain
        for i in range(5):
            record_id = f"tr_{i}"
            adapter._records[record_id] = InterventionRecord(
                id=record_id,
                debate_id=f"d{i}",
                domain="engineering",
                topic=f"topic {i}",
                intervention_type="challenge_prompt" if i < 3 else "evidence_gap",
                round_num=1,
                target_agents=[],
                challenge_text=f"challenge {i}",
                evidence_gaps={},
                priority=0.5,
                timestamp=datetime.now(),
            )
            if "engineering" not in adapter._by_domain:
                adapter._by_domain["engineering"] = []
            adapter._by_domain["engineering"].append(record_id)

        patterns = adapter.get_domain_patterns("engineering")
        assert len(patterns) == 2  # Two types: challenge_prompt and evidence_gap
        assert patterns[0]["intervention_type"] == "challenge_prompt"
        assert patterns[0]["frequency"] == 3
        assert patterns[1]["intervention_type"] == "evidence_gap"
        assert patterns[1]["frequency"] == 2

    def test_get_debate_interventions(self) -> None:
        """Test getting all interventions for a debate."""
        adapter = TricksterAdapter()

        # Add records for different debates
        adapter._records["tr_1"] = InterventionRecord(
            id="tr_1",
            debate_id="debate_A",
            domain=None,
            topic="topic",
            intervention_type="challenge_prompt",
            round_num=1,
            target_agents=[],
            challenge_text="test",
            evidence_gaps={},
            priority=0.5,
            timestamp=datetime.now(),
        )
        adapter._records["tr_2"] = InterventionRecord(
            id="tr_2",
            debate_id="debate_A",
            domain=None,
            topic="topic",
            intervention_type="evidence_gap",
            round_num=2,
            target_agents=[],
            challenge_text="test",
            evidence_gaps={},
            priority=0.6,
            timestamp=datetime.now(),
        )
        adapter._by_debate["debate_A"] = ["tr_1", "tr_2"]

        interventions = adapter.get_debate_interventions("debate_A")
        assert len(interventions) == 2

    def test_record_outcome(self) -> None:
        """Test recording intervention outcome."""
        adapter = TricksterAdapter()
        adapter._records["tr_test"] = InterventionRecord(
            id="tr_test",
            debate_id="d1",
            domain=None,
            topic="topic",
            intervention_type="challenge_prompt",
            round_num=1,
            target_agents=[],
            challenge_text="test",
            evidence_gaps={},
            priority=0.5,
            timestamp=datetime.now(),
        )

        result = adapter.record_outcome("tr_test", "effective", {"quality_improved": True})
        assert result is True

        record = adapter.get("tr_test")
        assert record.outcome == "effective"
        assert record.metadata["outcome_data"]["quality_improved"] is True

    def test_get_stats(self) -> None:
        """Test getting adapter statistics."""
        adapter = TricksterAdapter()

        adapter._records["tr_1"] = InterventionRecord(
            id="tr_1",
            debate_id="d1",
            domain="eng",
            topic="t1",
            intervention_type="challenge_prompt",
            round_num=1,
            target_agents=[],
            challenge_text="test",
            evidence_gaps={},
            priority=0.5,
            timestamp=datetime.now(),
            outcome="effective",
        )
        adapter._by_debate["d1"] = ["tr_1"]
        adapter._by_domain["eng"] = ["tr_1"]

        stats = adapter.get_stats()
        assert stats["total_interventions"] == 1
        assert stats["debates_with_interventions"] == 1
        assert stats["domains_with_interventions"] == 1
        assert stats["intervention_types"]["challenge_prompt"] == 1
        assert stats["outcomes"]["effective"] == 1

    @pytest.mark.asyncio
    async def test_sync_to_km(self) -> None:
        """Test syncing records to Knowledge Mound."""
        adapter = TricksterAdapter()

        adapter._records["tr_1"] = InterventionRecord(
            id="tr_1",
            debate_id="d1",
            domain="eng",
            topic="rate limiting",
            intervention_type="challenge_prompt",
            round_num=1,
            target_agents=["a1"],
            challenge_text="Need more evidence",
            evidence_gaps={},
            priority=0.7,
            timestamp=datetime.now(),
            metadata={"km_sync_pending": True},
        )

        mock_mound = MagicMock()
        mock_mound.store_item = AsyncMock()

        result = await adapter.sync_to_km(mock_mound, min_priority=0.5)

        assert result.records_synced == 1
        assert result.records_failed == 0
        mock_mound.store_item.assert_called_once()

        # Verify record was marked as synced
        record = adapter._records["tr_1"]
        assert record.metadata["km_sync_pending"] is False
        assert "km_synced_at" in record.metadata


class TestTricksterAdapterFactory:
    """Test TricksterAdapter registration in factory."""

    def test_factory_has_trickster_spec(self) -> None:
        """Test that trickster adapter is registered in factory."""
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        assert "trickster" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["trickster"]
        assert spec.required_deps == ["trickster"]
        assert spec.forward_method == "sync_to_km"
        assert spec.config_key == "km_trickster_adapter"
