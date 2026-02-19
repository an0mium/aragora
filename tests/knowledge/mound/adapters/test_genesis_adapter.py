"""Tests for the GenesisAdapter Knowledge Mound adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.genesis_adapter import (
    GenesisAdapter,
    GenesisEvolutionItem,
    GenesisSearchResult,
)


@pytest.fixture
def adapter() -> GenesisAdapter:
    return GenesisAdapter()


@pytest.fixture
def sample_genome() -> GenesisEvolutionItem:
    return GenesisEvolutionItem(
        genome_id="genome-001",
        name="claude-security-v1",
        traits={"analytical": 0.9, "cautious": 0.8, "creative": 0.6},
        expertise={"security": 0.85, "api_design": 0.7, "testing": 0.6},
        model_preference="claude",
        parent_genomes=[],
        generation=0,
        fitness_score=0.75,
        birth_debate_id="debate-100",
        consensus_contributions=5,
        critiques_accepted=3,
        predictions_correct=4,
        debates_participated=8,
    )


@pytest.fixture
def child_genome() -> GenesisEvolutionItem:
    return GenesisEvolutionItem(
        genome_id="genome-002",
        name="claude-security-v2",
        traits={"analytical": 0.95, "cautious": 0.85, "creative": 0.5},
        expertise={"security": 0.9, "api_design": 0.75},
        model_preference="claude",
        parent_genomes=["genome-001"],
        generation=1,
        fitness_score=0.82,
        birth_debate_id="debate-200",
        consensus_contributions=8,
        critiques_accepted=5,
        predictions_correct=6,
        debates_participated=12,
    )


class TestGenesisAdapterInit:
    def test_init_defaults(self, adapter: GenesisAdapter) -> None:
        assert adapter.adapter_name == "genesis"
        assert adapter.source_type == "genesis"
        assert adapter._pending_genomes == []
        assert adapter._synced_genomes == {}

    def test_init_with_callback(self) -> None:
        callback = MagicMock()
        adapter = GenesisAdapter(event_callback=callback)
        assert adapter._event_callback == callback


class TestGenesisEvolutionItem:
    def test_from_genome(self) -> None:
        mock_genome = MagicMock()
        mock_genome.genome_id = "g-123"
        mock_genome.name = "test-agent"
        mock_genome.traits = {"bold": 0.7}
        mock_genome.expertise = {"security": 0.9}
        mock_genome.model_preference = "gpt4"
        mock_genome.parent_genomes = ["g-parent"]
        mock_genome.generation = 2
        mock_genome.fitness_score = 0.8
        mock_genome.birth_debate_id = "d-1"
        mock_genome.consensus_contributions = 3
        mock_genome.critiques_accepted = 2
        mock_genome.predictions_correct = 4
        mock_genome.debates_participated = 5

        item = GenesisEvolutionItem.from_genome(mock_genome)

        assert item.genome_id == "g-123"
        assert item.name == "test-agent"
        assert item.traits == {"bold": 0.7}
        assert item.expertise == {"security": 0.9}
        assert item.model_preference == "gpt4"
        assert item.parent_genomes == ["g-parent"]
        assert item.generation == 2
        assert item.fitness_score == 0.8
        assert item.consensus_contributions == 3

    def test_from_genome_defaults(self) -> None:
        mock_genome = MagicMock(spec=[])
        item = GenesisEvolutionItem.from_genome(mock_genome)
        assert item.genome_id == ""
        assert item.fitness_score == 0.5
        assert item.generation == 0


class TestStoreGenome:
    def test_store_genome_adds_to_pending(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter.store_genome(sample_genome)

        assert len(adapter._pending_genomes) == 1
        assert adapter._pending_genomes[0].genome_id == "genome-001"
        assert adapter._pending_genomes[0].metadata["km_sync_pending"] is True

    def test_store_genome_emits_event(self, sample_genome: GenesisEvolutionItem) -> None:
        callback = MagicMock()
        adapter = GenesisAdapter(event_callback=callback)
        adapter.store_genome(sample_genome)

        callback.assert_called_once()
        event_type, event_data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync"
        assert event_data["genome_id"] == "genome-001"
        assert event_data["fitness_score"] == 0.75

    def test_store_from_agent_genome(self, adapter: GenesisAdapter) -> None:
        mock_genome = MagicMock()
        mock_genome.genome_id = "g-auto"
        mock_genome.name = "auto-agent"
        mock_genome.traits = {}
        mock_genome.expertise = {}
        mock_genome.model_preference = "claude"
        mock_genome.parent_genomes = []
        mock_genome.generation = 0
        mock_genome.fitness_score = 0.6
        mock_genome.birth_debate_id = None
        mock_genome.consensus_contributions = 0
        mock_genome.critiques_accepted = 0
        mock_genome.predictions_correct = 0
        mock_genome.debates_participated = 0

        adapter.store_genome(mock_genome)
        assert len(adapter._pending_genomes) == 1
        assert adapter._pending_genomes[0].genome_id == "g-auto"

    def test_store_multiple_genomes(
        self,
        adapter: GenesisAdapter,
        sample_genome: GenesisEvolutionItem,
        child_genome: GenesisEvolutionItem,
    ) -> None:
        adapter.store_genome(sample_genome)
        adapter.store_genome(child_genome)
        assert len(adapter._pending_genomes) == 2


class TestGetGenome:
    def test_get_returns_none_when_empty(self, adapter: GenesisAdapter) -> None:
        assert adapter.get("nonexistent") is None

    def test_get_returns_synced_genome(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        result = adapter.get("genome-001")
        assert result is not None
        assert result.genome_id == "genome-001"

    def test_get_strips_prefix(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        result = adapter.get("gen_genome-001")
        assert result is not None
        assert result.genome_id == "genome-001"

    @pytest.mark.asyncio
    async def test_get_async(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        result = await adapter.get_async("genome-001")
        assert result is not None
        assert result.genome_id == "genome-001"


class TestGetLineage:
    def test_lineage_single_genome(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        lineage = adapter.get_lineage("genome-001")
        assert len(lineage) == 1
        assert lineage[0].genome_id == "genome-001"

    def test_lineage_parent_child(
        self,
        adapter: GenesisAdapter,
        sample_genome: GenesisEvolutionItem,
        child_genome: GenesisEvolutionItem,
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        adapter._synced_genomes["genome-002"] = child_genome
        lineage = adapter.get_lineage("genome-002")
        assert len(lineage) == 2
        assert lineage[0].genome_id == "genome-002"
        assert lineage[1].genome_id == "genome-001"

    def test_lineage_nonexistent(self, adapter: GenesisAdapter) -> None:
        lineage = adapter.get_lineage("nonexistent")
        assert lineage == []

    def test_lineage_includes_pending(
        self,
        adapter: GenesisAdapter,
        sample_genome: GenesisEvolutionItem,
        child_genome: GenesisEvolutionItem,
    ) -> None:
        adapter._pending_genomes.append(sample_genome)
        adapter._pending_genomes.append(child_genome)
        lineage = adapter.get_lineage("genome-002")
        assert len(lineage) == 2

    def test_lineage_handles_cycles(self, adapter: GenesisAdapter) -> None:
        g1 = GenesisEvolutionItem(
            genome_id="cycle-a", name="a", parent_genomes=["cycle-b"]
        )
        g2 = GenesisEvolutionItem(
            genome_id="cycle-b", name="b", parent_genomes=["cycle-a"]
        )
        adapter._synced_genomes["cycle-a"] = g1
        adapter._synced_genomes["cycle-b"] = g2
        lineage = adapter.get_lineage("cycle-a")
        assert len(lineage) == 2


class TestFindHighFitnessGenomes:
    def test_returns_high_fitness(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        results = adapter.find_high_fitness_genomes(min_fitness=0.7)
        assert len(results) == 1
        assert results[0].genome_id == "genome-001"
        assert results[0].fitness_score == 0.75

    def test_filters_below_threshold(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        results = adapter.find_high_fitness_genomes(min_fitness=0.8)
        assert len(results) == 0

    def test_filters_by_domain(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        results = adapter.find_high_fitness_genomes(min_fitness=0.5, domain="security")
        assert len(results) == 1

        results = adapter.find_high_fitness_genomes(min_fitness=0.5, domain="quantum_physics")
        assert len(results) == 0

    def test_sorted_by_fitness_descending(
        self,
        adapter: GenesisAdapter,
        sample_genome: GenesisEvolutionItem,
        child_genome: GenesisEvolutionItem,
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        adapter._synced_genomes["genome-002"] = child_genome
        results = adapter.find_high_fitness_genomes(min_fitness=0.5)
        assert len(results) == 2
        assert results[0].fitness_score >= results[1].fitness_score

    def test_respects_limit(self, adapter: GenesisAdapter) -> None:
        for i in range(5):
            item = GenesisEvolutionItem(
                genome_id=f"g-{i}", name=f"agent-{i}", fitness_score=0.8 + i * 0.01
            )
            adapter._synced_genomes[f"g-{i}"] = item
        results = adapter.find_high_fitness_genomes(min_fitness=0.5, limit=3)
        assert len(results) == 3

    def test_includes_pending_genomes(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        adapter._pending_genomes.append(sample_genome)
        results = adapter.find_high_fitness_genomes(min_fitness=0.5)
        assert len(results) == 1


class TestToKnowledgeItem:
    def test_converts_to_km_item(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        km_item = adapter.to_knowledge_item(sample_genome)
        assert km_item.id == "gen_genome-001"
        assert km_item.source_id == "genome-001"
        assert "claude-security-v1" in km_item.content
        assert "Generation: 0" in km_item.content
        assert "Fitness: 0.75" in km_item.content
        assert km_item.metadata["adapter"] == "genesis"
        assert km_item.metadata["fitness_score"] == 0.75
        assert km_item.metadata["generation"] == 0


class TestSyncToKM:
    @pytest.mark.asyncio
    async def test_sync_stores_to_mound(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_genome(sample_genome)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 1
        assert result.records_skipped == 0
        assert result.records_failed == 0
        mound.store_item.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_skips_low_fitness(self, adapter: GenesisAdapter) -> None:
        low_fitness = GenesisEvolutionItem(
            genome_id="low-1", name="weak-agent", fitness_score=0.1
        )
        adapter.store_genome(low_fitness)

        mound = MagicMock()
        mound.store_item = AsyncMock()
        result = await adapter.sync_to_km(mound, min_confidence=0.3)

        assert result.records_synced == 0
        assert result.records_skipped == 1
        mound.store_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_handles_store_error(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock(side_effect=RuntimeError("Store failed"))

        adapter.store_genome(sample_genome)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 0
        assert result.records_failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_sync_removes_from_pending(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_genome(sample_genome)
        assert len(adapter._pending_genomes) == 1

        await adapter.sync_to_km(mound)
        assert len(adapter._pending_genomes) == 0
        assert "genome-001" in adapter._synced_genomes

    @pytest.mark.asyncio
    async def test_sync_emits_event(
        self, sample_genome: GenesisEvolutionItem
    ) -> None:
        callback = MagicMock()
        adapter = GenesisAdapter(event_callback=callback)
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_genome(sample_genome)
        callback.reset_mock()
        await adapter.sync_to_km(mound)

        callback.assert_called_once()
        event_type, event_data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync_complete"
        assert event_data["genome_id"] == "genome-001"

    @pytest.mark.asyncio
    async def test_sync_batch_size(self, adapter: GenesisAdapter) -> None:
        for i in range(10):
            item = GenesisEvolutionItem(
                genome_id=f"g-{i}", name=f"agent-{i}", fitness_score=0.8
            )
            adapter.store_genome(item)

        mound = MagicMock()
        mound.store_item = AsyncMock()
        result = await adapter.sync_to_km(mound, batch_size=3)

        assert result.records_synced == 3
        assert len(adapter._pending_genomes) == 7

    @pytest.mark.asyncio
    async def test_sync_uses_store_fallback(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        mound = MagicMock(spec=[])
        mound.store = AsyncMock()

        adapter.store_genome(sample_genome)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 1
        mound.store.assert_called_once()


class TestGetStats:
    def test_empty_stats(self, adapter: GenesisAdapter) -> None:
        stats = adapter.get_stats()
        assert stats["total_synced"] == 0
        assert stats["pending_sync"] == 0
        assert stats["avg_fitness"] == 0.0
        assert stats["max_generation"] == 0

    def test_stats_with_data(
        self,
        adapter: GenesisAdapter,
        sample_genome: GenesisEvolutionItem,
        child_genome: GenesisEvolutionItem,
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        adapter._synced_genomes["genome-002"] = child_genome
        adapter._pending_genomes.append(
            GenesisEvolutionItem(genome_id="pending-1", name="pending")
        )

        stats = adapter.get_stats()
        assert stats["total_synced"] == 2
        assert stats["pending_sync"] == 1
        assert stats["avg_fitness"] == pytest.approx((0.75 + 0.82) / 2)
        assert stats["max_generation"] == 1
        assert stats["total_debates"] == 20


class TestMixinMethods:
    def test_record_to_dict(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        result = adapter._record_to_dict(sample_genome, similarity=0.9)
        assert result["id"] == "genome-001"
        assert result["name"] == "claude-security-v1"
        assert result["fitness_score"] == 0.75
        assert result["similarity"] == 0.9

    def test_apply_km_validation(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        success = adapter._apply_km_validation(
            sample_genome, km_confidence=0.9, cross_refs=["ref-1"]
        )
        assert success is True
        assert sample_genome.metadata["km_validated"] is True
        assert sample_genome.metadata["km_validation_confidence"] == 0.9
        assert sample_genome.metadata["km_cross_references"] == ["ref-1"]

    def test_extract_source_id(self, adapter: GenesisAdapter) -> None:
        assert adapter._extract_source_id({"source_id": "gen_abc"}) == "abc"
        assert adapter._extract_source_id({"source_id": "raw-id"}) == "raw-id"
        assert adapter._extract_source_id({}) is None

    def test_get_fusion_sources(self, adapter: GenesisAdapter) -> None:
        sources = adapter._get_fusion_sources()
        assert "elo" in sources
        assert "debate" in sources

    def test_extract_fusible_data(self, adapter: GenesisAdapter) -> None:
        item = {"metadata": {"adapter": "genesis", "fitness_score": 0.8, "generation": 2}}
        data = adapter._extract_fusible_data(item)
        assert data is not None
        assert data["fitness_score"] == 0.8
        assert data["generation"] == 2

    def test_extract_fusible_data_wrong_adapter(self, adapter: GenesisAdapter) -> None:
        item = {"metadata": {"adapter": "debate"}}
        assert adapter._extract_fusible_data(item) is None

    def test_apply_fusion_result(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        fusion = MagicMock()
        fusion.fused_confidence = 0.88
        success = adapter._apply_fusion_result(sample_genome, fusion)
        assert success is True
        assert sample_genome.metadata["fusion_applied"] is True
        assert sample_genome.metadata["fused_confidence"] == 0.88
