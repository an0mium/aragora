"""Tests for the GenesisAdapter Knowledge Mound adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.adapters.genesis_adapter import (
    GenesisAdapter,
    GenesisEvolutionItem,
    GenesisSearchResult,
    get_genesis_adapter,
    _HAS_GENESIS,
)


@pytest.fixture
def adapter() -> GenesisAdapter:
    return GenesisAdapter()


@pytest.fixture
def sample_genome() -> GenesisEvolutionItem:
    return GenesisEvolutionItem(
        genome_id="genome-001",
        name="claude-security-v1",
        item_type="genome",
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
        item_type="genome",
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


@pytest.fixture
def fractal_item() -> GenesisEvolutionItem:
    return GenesisEvolutionItem(
        genome_id="frac_abc123",
        name="fractal_orchestrator",
        item_type="fractal_result",
        fitness_score=0.85,
        root_debate_id="debate-root-1",
        sub_debate_count=3,
        total_depth=2,
        tensions_resolved=2,
        tensions_unresolved=1,
        evolved_genome_ids=["genome-001", "genome-002"],
        metadata={"task": "Design a distributed cache"},
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
        assert item.item_type == "genome"
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
        assert item.item_type == "genome"

    def test_from_fractal_result(self) -> None:
        mock_result = MagicMock()
        mock_result.root_debate_id = "root-123"
        mock_result.sub_debates = [MagicMock(), MagicMock()]
        mock_result.total_depth = 2
        mock_result.tensions_resolved = 1
        mock_result.tensions_unresolved = 1
        mock_result.evolved_genomes = []
        mock_result.main_result = MagicMock()
        mock_result.main_result.task = "Test task"
        mock_result.main_result.confidence = 0.9

        item = GenesisEvolutionItem.from_fractal_result(mock_result)

        assert item.item_type == "fractal_result"
        assert item.name == "fractal_orchestrator"
        assert item.root_debate_id == "root-123"
        assert item.sub_debate_count == 2
        assert item.total_depth == 2
        assert item.tensions_resolved == 1
        assert item.fitness_score == 0.9
        assert item.metadata["task"] == "Test task"
        assert item.genome_id.startswith("frac_")

    def test_from_fractal_result_with_evolved_genomes(self) -> None:
        genome_mock = MagicMock()
        genome_mock.genome_id = "evolved-1"

        mock_result = MagicMock()
        mock_result.root_debate_id = "root-456"
        mock_result.sub_debates = []
        mock_result.total_depth = 1
        mock_result.tensions_resolved = 0
        mock_result.tensions_unresolved = 0
        mock_result.evolved_genomes = [genome_mock]
        mock_result.main_result = MagicMock()
        mock_result.main_result.task = "Evolve"
        mock_result.main_result.confidence = 0.7

        item = GenesisEvolutionItem.from_fractal_result(mock_result)
        assert item.evolved_genome_ids == ["evolved-1"]

    def test_from_dict(self) -> None:
        data = {
            "genome_id": "dict-g1",
            "name": "dict-agent",
            "item_type": "genome",
            "traits": {"bold": 0.5},
            "expertise": {"testing": 0.8},
            "fitness_score": 0.65,
            "generation": 3,
            "parent_genomes": ["parent-1"],
        }
        item = GenesisEvolutionItem.from_dict(data)

        assert item.genome_id == "dict-g1"
        assert item.name == "dict-agent"
        assert item.item_type == "genome"
        assert item.traits == {"bold": 0.5}
        assert item.fitness_score == 0.65
        assert item.generation == 3

    def test_from_dict_defaults(self) -> None:
        item = GenesisEvolutionItem.from_dict({})
        assert item.genome_id == ""
        assert item.name == ""
        assert item.item_type == "genome"
        assert item.fitness_score == 0.5


class TestIngest:
    def test_ingest_evolution_item(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        result = adapter.ingest(sample_genome)

        assert result is True
        assert len(adapter._pending_genomes) == 1
        assert adapter._pending_genomes[0].genome_id == "genome-001"
        assert adapter._pending_genomes[0].metadata["km_sync_pending"] is True

    def test_ingest_dict(self, adapter: GenesisAdapter) -> None:
        data = {
            "genome_id": "dict-001",
            "name": "dict-agent",
            "fitness_score": 0.7,
            "traits": {"creative": 0.9},
        }
        result = adapter.ingest(data)

        assert result is True
        assert len(adapter._pending_genomes) == 1
        assert adapter._pending_genomes[0].genome_id == "dict-001"

    def test_ingest_genome_like_object(self, adapter: GenesisAdapter) -> None:
        mock_genome = MagicMock()
        mock_genome.genome_id = "duck-g1"
        mock_genome.name = "duck-agent"
        mock_genome.traits = {"bold": 0.8}
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

        result = adapter.ingest(mock_genome)

        assert result is True
        assert adapter._pending_genomes[0].genome_id == "duck-g1"
        assert adapter._pending_genomes[0].item_type == "genome"

    def test_ingest_fractal_like_object(self, adapter: GenesisAdapter) -> None:
        # Use spec=[] to prevent MagicMock from auto-generating attributes,
        # then set only the fractal-specific attributes.
        mock_result = MagicMock(spec=[])
        mock_result.root_debate_id = "root-789"
        mock_result.sub_debates = [MagicMock()]
        mock_result.total_depth = 1
        mock_result.tensions_resolved = 1
        mock_result.tensions_unresolved = 0
        mock_result.evolved_genomes = []
        mock_result.main_result = MagicMock()
        mock_result.main_result.task = "Fractal task"
        mock_result.main_result.confidence = 0.8

        result = adapter.ingest(mock_result)

        assert result is True
        assert adapter._pending_genomes[0].item_type == "fractal_result"

    def test_ingest_emits_event(self, sample_genome: GenesisEvolutionItem) -> None:
        callback = MagicMock()
        adapter = GenesisAdapter(event_callback=callback)
        adapter.ingest(sample_genome)

        callback.assert_called_once()
        event_type, event_data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync"
        assert event_data["genome_id"] == "genome-001"
        assert event_data["item_type"] == "genome"

    def test_ingest_unknown_type_returns_false(self, adapter: GenesisAdapter) -> None:
        # Object with no genome_id/traits and no root_debate_id/sub_debates
        unknown_obj = MagicMock(spec=[])
        result = adapter.ingest(unknown_obj)
        assert result is False
        assert len(adapter._pending_genomes) == 0

    def test_ingest_multiple_items(
        self,
        adapter: GenesisAdapter,
        sample_genome: GenesisEvolutionItem,
        fractal_item: GenesisEvolutionItem,
    ) -> None:
        adapter.ingest(sample_genome)
        adapter.ingest(fractal_item)
        assert len(adapter._pending_genomes) == 2


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

    def test_lineage_three_generations(self, adapter: GenesisAdapter) -> None:
        g0 = GenesisEvolutionItem(genome_id="g0", name="ancestor", generation=0)
        g1 = GenesisEvolutionItem(
            genome_id="g1", name="parent", generation=1, parent_genomes=["g0"]
        )
        g2 = GenesisEvolutionItem(
            genome_id="g2", name="child", generation=2, parent_genomes=["g1"]
        )
        adapter._synced_genomes["g0"] = g0
        adapter._synced_genomes["g1"] = g1
        adapter._synced_genomes["g2"] = g2

        lineage = adapter.get_lineage("g2")
        assert len(lineage) == 3
        assert [g.genome_id for g in lineage] == ["g2", "g1", "g0"]


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

    def test_excludes_fractal_items(
        self, adapter: GenesisAdapter, fractal_item: GenesisEvolutionItem
    ) -> None:
        adapter._synced_genomes[fractal_item.genome_id] = fractal_item
        results = adapter.find_high_fitness_genomes(min_fitness=0.0)
        assert len(results) == 0


class TestToKnowledgeItem:
    def test_converts_genome_to_km_item(
        self, adapter: GenesisAdapter, sample_genome: GenesisEvolutionItem
    ) -> None:
        km_item = adapter.to_knowledge_item(sample_genome)
        assert km_item.id == "gen_genome-001"
        assert km_item.source_id == "genome-001"
        assert "claude-security-v1" in km_item.content
        assert "Generation: 0" in km_item.content
        assert "Fitness: 0.75" in km_item.content
        assert km_item.metadata["adapter"] == "genesis"
        assert km_item.metadata["item_type"] == "genome"
        assert km_item.metadata["fitness_score"] == 0.75
        assert km_item.metadata["generation"] == 0

        from aragora.knowledge.unified.types import KnowledgeSource

        assert km_item.source == KnowledgeSource.DEBATE

    def test_converts_fractal_to_km_item(
        self, adapter: GenesisAdapter, fractal_item: GenesisEvolutionItem
    ) -> None:
        km_item = adapter.to_knowledge_item(fractal_item)
        assert km_item.id.startswith("gen_frac_")
        assert km_item.source_id == "debate-root-1"
        assert "Fractal Debate Result" in km_item.content
        assert "Depth: 2" in km_item.content
        assert "Sub-debates: 3" in km_item.content
        assert "Tensions resolved: 2" in km_item.content
        assert km_item.metadata["item_type"] == "fractal_result"
        assert km_item.metadata["root_debate_id"] == "debate-root-1"
        assert km_item.metadata["evolved_genome_ids"] == ["genome-001", "genome-002"]


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

    @pytest.mark.asyncio
    async def test_sync_fractal_item(
        self, adapter: GenesisAdapter, fractal_item: GenesisEvolutionItem
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.ingest(fractal_item)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 1
        mound.store_item.assert_called_once()
        stored_item = mound.store_item.call_args[0][0]
        assert "Fractal Debate Result" in stored_item.content


class TestGetStats:
    def test_empty_stats(self, adapter: GenesisAdapter) -> None:
        stats = adapter.get_stats()
        assert stats["total_synced"] == 0
        assert stats["pending_sync"] == 0
        assert stats["avg_fitness"] == 0.0
        assert stats["max_generation"] == 0
        assert stats["genomes_stored"] == 0
        assert stats["fractals_stored"] == 0

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
        assert stats["genomes_stored"] == 2
        assert stats["fractals_stored"] == 0
        assert stats["avg_fitness"] == pytest.approx((0.75 + 0.82) / 2)
        assert stats["max_generation"] == 1
        assert stats["total_debates"] == 20

    def test_stats_with_fractals(
        self,
        adapter: GenesisAdapter,
        sample_genome: GenesisEvolutionItem,
        fractal_item: GenesisEvolutionItem,
    ) -> None:
        adapter._synced_genomes["genome-001"] = sample_genome
        adapter._synced_genomes[fractal_item.genome_id] = fractal_item

        stats = adapter.get_stats()
        assert stats["genomes_stored"] == 1
        assert stats["fractals_stored"] == 1
        # avg_fitness only counts genomes
        assert stats["avg_fitness"] == 0.75


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

    def test_extract_source_id_gen_prefix(self, adapter: GenesisAdapter) -> None:
        assert adapter._extract_source_id({"source_id": "gen_abc"}) == "abc"

    def test_extract_source_id_frac_prefix(self, adapter: GenesisAdapter) -> None:
        assert adapter._extract_source_id({"source_id": "frac_xyz"}) == "xyz"

    def test_extract_source_id_raw(self, adapter: GenesisAdapter) -> None:
        assert adapter._extract_source_id({"source_id": "raw-id"}) == "raw-id"

    def test_extract_source_id_empty(self, adapter: GenesisAdapter) -> None:
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


class TestGracefulDegradation:
    """Tests that the adapter works even when genesis module is unavailable."""

    def test_adapter_loads_without_genesis(self) -> None:
        """The adapter module can be imported regardless of genesis availability."""
        # If we got here, the import at module level already succeeded.
        # The _HAS_GENESIS flag indicates whether genesis was importable.
        from aragora.knowledge.mound.adapters import genesis_adapter

        assert hasattr(genesis_adapter, "GenesisAdapter")
        assert hasattr(genesis_adapter, "GenesisEvolutionItem")

    def test_ingest_works_with_mock_genome_no_genesis(
        self, adapter: GenesisAdapter
    ) -> None:
        """ingest() works via duck-typing even if genesis types are unavailable."""
        mock_genome = MagicMock(spec=[])
        # Add only the duck-type attributes
        mock_genome.genome_id = "no-genesis-1"
        mock_genome.traits = {"stub": 0.5}
        mock_genome.name = "stub-agent"
        mock_genome.expertise = {}
        mock_genome.model_preference = "claude"
        mock_genome.parent_genomes = []
        mock_genome.generation = 0
        mock_genome.fitness_score = 0.5
        mock_genome.birth_debate_id = None
        mock_genome.consensus_contributions = 0
        mock_genome.critiques_accepted = 0
        mock_genome.predictions_correct = 0
        mock_genome.debates_participated = 0

        result = adapter.ingest(mock_genome)
        assert result is True

    def test_ingest_fractal_via_duck_typing(self, adapter: GenesisAdapter) -> None:
        """Fractal results are detected via duck-typing, not isinstance."""
        # Use spec=[] so MagicMock does not auto-generate genome_id/traits
        mock_fractal = MagicMock(spec=[])
        mock_fractal.root_debate_id = "duck-root"
        mock_fractal.sub_debates = []
        mock_fractal.total_depth = 0
        mock_fractal.tensions_resolved = 0
        mock_fractal.tensions_unresolved = 0
        mock_fractal.evolved_genomes = []
        mock_fractal.main_result = MagicMock()
        mock_fractal.main_result.task = "Duck task"
        mock_fractal.main_result.confidence = 0.5

        result = adapter.ingest(mock_fractal)
        assert result is True
        assert adapter._pending_genomes[0].item_type == "fractal_result"


class TestSingleton:
    def test_get_genesis_adapter_returns_singleton(self) -> None:
        import aragora.knowledge.mound.adapters.genesis_adapter as mod

        mod._genesis_adapter_singleton = None
        a1 = get_genesis_adapter()
        a2 = get_genesis_adapter()
        assert a1 is a2
        # Clean up
        mod._genesis_adapter_singleton = None

    def test_singleton_is_genesis_adapter(self) -> None:
        import aragora.knowledge.mound.adapters.genesis_adapter as mod

        mod._genesis_adapter_singleton = None
        adapter = get_genesis_adapter()
        assert isinstance(adapter, GenesisAdapter)
        mod._genesis_adapter_singleton = None
