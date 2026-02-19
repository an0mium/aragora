"""
GenesisAdapter - Bridges agent evolution data to the Knowledge Mound.

Persists AgentGenome evolution history, fitness trajectories, and lineage
data so that the KM can inform future agent selection and evolution decisions.

The adapter provides:
- Genome evolution persistence (fitness, traits, expertise)
- Lineage tracking across generations
- High-fitness genome discovery for agent selection
- Domain expertise search across evolved agents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem

EventCallback = Callable[[str, dict[str, Any]], None]

logger = logging.getLogger(__name__)

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin
from aragora.knowledge.mound.adapters._reverse_flow_base import ReverseFlowMixin
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin
from aragora.knowledge.mound.adapters._types import SyncResult


@dataclass
class GenesisEvolutionItem:
    """Lightweight representation of a genome for adapter storage.

    Decoupled from the full AgentGenome dataclass to avoid importing
    genesis-specific dependencies.
    """

    genome_id: str
    name: str
    traits: dict[str, float] = field(default_factory=dict)
    expertise: dict[str, float] = field(default_factory=dict)
    model_preference: str = "claude"
    parent_genomes: list[str] = field(default_factory=list)
    generation: int = 0
    fitness_score: float = 0.5
    birth_debate_id: str | None = None
    consensus_contributions: int = 0
    critiques_accepted: int = 0
    predictions_correct: int = 0
    debates_participated: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_genome(cls, genome: Any) -> GenesisEvolutionItem:
        """Create a GenesisEvolutionItem from an AgentGenome object."""
        return cls(
            genome_id=getattr(genome, "genome_id", ""),
            name=getattr(genome, "name", ""),
            traits=getattr(genome, "traits", {}),
            expertise=getattr(genome, "expertise", {}),
            model_preference=getattr(genome, "model_preference", "claude"),
            parent_genomes=getattr(genome, "parent_genomes", []),
            generation=getattr(genome, "generation", 0),
            fitness_score=getattr(genome, "fitness_score", 0.5),
            birth_debate_id=getattr(genome, "birth_debate_id", None),
            consensus_contributions=getattr(genome, "consensus_contributions", 0),
            critiques_accepted=getattr(genome, "critiques_accepted", 0),
            predictions_correct=getattr(genome, "predictions_correct", 0),
            debates_participated=getattr(genome, "debates_participated", 0),
        )


@dataclass
class GenesisSearchResult:
    """Wrapper for genesis search results with similarity metadata."""

    genome_id: str
    name: str
    fitness_score: float
    generation: int
    expertise: dict[str, float]
    traits: dict[str, float]
    similarity: float = 0.0
    parent_genomes: list[str] = field(default_factory=list)


class GenesisAdapter(FusionMixin, ReverseFlowMixin, SemanticSearchMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges agent evolution data to the Knowledge Mound.

    Provides genome persistence, lineage tracking, and high-fitness genome
    discovery. Enables the KM to inform agent selection and evolution decisions
    based on historical performance data.

    Usage:
        adapter = GenesisAdapter()
        adapter.store_genome(agent_genome)     # Mark for sync
        await adapter.sync_to_km(mound)        # Persist to KM
        results = adapter.find_high_fitness_genomes(min_fitness=0.7)
    """

    adapter_name = "genesis"
    source_type = "genesis"

    def __init__(
        self,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
    ):
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )
        self._pending_genomes: list[GenesisEvolutionItem] = []
        self._synced_genomes: dict[str, GenesisEvolutionItem] = {}

    def store_genome(self, genome: Any) -> None:
        """Store a genome for KM sync.

        Args:
            genome: An AgentGenome or GenesisEvolutionItem object.
        """
        if isinstance(genome, GenesisEvolutionItem):
            item = genome
        else:
            item = GenesisEvolutionItem.from_genome(genome)

        item.metadata["km_sync_pending"] = True
        item.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_genomes.append(item)

        self._emit_event(
            "km_adapter_forward_sync",
            {
                "adapter": self.adapter_name,
                "genome_id": item.genome_id,
                "name": item.name,
                "fitness_score": item.fitness_score,
                "generation": item.generation,
            },
        )

    def get(self, record_id: str) -> GenesisEvolutionItem | None:
        """Get a genome by ID."""
        clean_id = record_id[4:] if record_id.startswith("gen_") else record_id
        return self._synced_genomes.get(clean_id)

    async def get_async(self, record_id: str) -> GenesisEvolutionItem | None:
        """Async version of get."""
        return self.get(record_id)

    def get_lineage(self, genome_id: str) -> list[GenesisEvolutionItem]:
        """Get the lineage chain for a genome.

        Traces parent_genomes references to build the ancestry chain.

        Args:
            genome_id: The genome to trace lineage for.

        Returns:
            List of GenesisEvolutionItem from the genome to its oldest ancestor.
        """
        lineage: list[GenesisEvolutionItem] = []
        all_genomes = {**self._synced_genomes}
        for item in self._pending_genomes:
            all_genomes.setdefault(item.genome_id, item)

        visited: set[str] = set()
        current_id: str | None = genome_id

        while current_id and current_id not in visited:
            genome = all_genomes.get(current_id)
            if genome is None:
                break
            lineage.append(genome)
            visited.add(current_id)
            current_id = genome.parent_genomes[0] if genome.parent_genomes else None

        return lineage

    def find_high_fitness_genomes(
        self,
        min_fitness: float = 0.7,
        limit: int = 10,
        domain: str | None = None,
    ) -> list[GenesisSearchResult]:
        """Find genomes with high fitness scores.

        Args:
            min_fitness: Minimum fitness threshold.
            limit: Maximum results to return.
            domain: Optional domain expertise filter.

        Returns:
            List of GenesisSearchResult sorted by fitness score descending.
        """
        results: list[GenesisSearchResult] = []

        all_genomes = list(self._synced_genomes.values()) + self._pending_genomes
        for item in all_genomes:
            if item.fitness_score < min_fitness:
                continue
            if domain and domain not in item.expertise:
                continue

            results.append(
                GenesisSearchResult(
                    genome_id=item.genome_id,
                    name=item.name,
                    fitness_score=item.fitness_score,
                    generation=item.generation,
                    expertise=item.expertise,
                    traits=item.traits,
                    parent_genomes=item.parent_genomes,
                )
            )

        results.sort(key=lambda r: r.fitness_score, reverse=True)
        return results[:limit]

    def to_knowledge_item(self, item: GenesisEvolutionItem) -> KnowledgeItem:
        """Convert a GenesisEvolutionItem to a KnowledgeItem for KM storage."""
        from aragora.knowledge.mound.types import KnowledgeItem, KnowledgeSource
        from aragora.knowledge.unified.types import ConfidenceLevel

        top_traits = sorted(item.traits.items(), key=lambda x: x[1], reverse=True)[:3]
        traits_str = ", ".join(f"{t}: {w:.2f}" for t, w in top_traits)

        top_expertise = sorted(item.expertise.items(), key=lambda x: x[1], reverse=True)[:3]
        expertise_str = ", ".join(f"{d}: {s:.0%}" for d, s in top_expertise)

        content = (
            f"Agent Genome: {item.name}\n"
            f"Generation: {item.generation} | Fitness: {item.fitness_score:.2f}\n"
            f"Top Traits: {traits_str}\n"
            f"Top Expertise: {expertise_str}\n"
            f"Debates: {item.debates_participated} | "
            f"Consensus: {item.consensus_contributions} | "
            f"Critiques: {item.critiques_accepted}"
        )

        return KnowledgeItem(
            id=f"gen_{item.genome_id}",
            content=content,
            source=KnowledgeSource.EXTERNAL,
            source_id=item.genome_id,
            confidence=ConfidenceLevel.from_float(item.fitness_score),
            created_at=item.created_at,
            updated_at=item.created_at,
            metadata={
                "adapter": "genesis",
                "name": item.name,
                "generation": item.generation,
                "fitness_score": item.fitness_score,
                "model_preference": item.model_preference,
                "parent_genomes": item.parent_genomes,
                "traits": item.traits,
                "expertise": item.expertise,
                "debates_participated": item.debates_participated,
                "consensus_contributions": item.consensus_contributions,
                "critiques_accepted": item.critiques_accepted,
                "predictions_correct": item.predictions_correct,
                "birth_debate_id": item.birth_debate_id,
            },
        )

    async def sync_to_km(
        self,
        mound: Any,
        min_confidence: float = 0.3,
        batch_size: int = 50,
    ) -> SyncResult:
        """Sync pending genomes to Knowledge Mound.

        Args:
            mound: The KnowledgeMound instance.
            min_confidence: Minimum fitness to sync (maps to confidence).
            batch_size: Max records per batch.

        Returns:
            SyncResult with sync statistics.
        """
        start = datetime.now(timezone.utc)
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        pending = self._pending_genomes[:batch_size]

        for item in pending:
            if item.fitness_score < min_confidence:
                skipped += 1
                continue

            try:
                km_item = self.to_knowledge_item(item)

                if hasattr(mound, "store_item"):
                    await mound.store_item(km_item)
                elif hasattr(mound, "store"):
                    await mound.store(km_item)
                elif hasattr(mound, "_semantic_store"):
                    await mound._semantic_store.store(km_item)

                item.metadata["km_sync_pending"] = False
                item.metadata["km_synced_at"] = datetime.now(timezone.utc).isoformat()
                item.metadata["km_item_id"] = km_item.id

                self._synced_genomes[item.genome_id] = item
                synced += 1

                self._emit_event(
                    "km_adapter_forward_sync_complete",
                    {
                        "adapter": self.adapter_name,
                        "genome_id": item.genome_id,
                        "km_item_id": km_item.id,
                    },
                )

            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                failed += 1
                error_msg = f"Failed to sync genome {item.genome_id}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                item.metadata["km_sync_error"] = f"Sync failed: {type(e).__name__}"

        synced_ids = {i.genome_id for i in pending if i.metadata.get("km_sync_pending") is False}
        self._pending_genomes = [
            i for i in self._pending_genomes if i.genome_id not in synced_ids
        ]

        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SyncResult(
            records_synced=synced,
            records_skipped=skipped,
            records_failed=failed,
            errors=errors,
            duration_ms=duration_ms,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored genomes."""
        all_genomes = list(self._synced_genomes.values())
        return {
            "total_synced": len(self._synced_genomes),
            "pending_sync": len(self._pending_genomes),
            "avg_fitness": (
                sum(g.fitness_score for g in all_genomes) / len(all_genomes) if all_genomes else 0.0
            ),
            "max_generation": max((g.generation for g in all_genomes), default=0),
            "total_debates": sum(g.debates_participated for g in all_genomes),
        }

    # --- SemanticSearchMixin required methods ---

    def _get_record_by_id(self, record_id: str) -> GenesisEvolutionItem | None:
        return self.get(record_id)

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        return {
            "id": record.genome_id,
            "name": record.name,
            "fitness_score": record.fitness_score,
            "generation": record.generation,
            "expertise": record.expertise,
            "traits": record.traits,
            "similarity": similarity,
        }

    # --- ReverseFlowMixin required methods ---

    def _get_record_for_validation(self, source_id: str) -> GenesisEvolutionItem | None:
        return self.get(source_id)

    def _apply_km_validation(
        self,
        record: Any,
        km_confidence: float,
        cross_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record.metadata["km_validated"] = True
        record.metadata["km_validation_confidence"] = km_confidence
        record.metadata["km_validation_timestamp"] = datetime.now(timezone.utc).isoformat()
        if cross_refs:
            record.metadata["km_cross_references"] = cross_refs
        return True

    def _extract_source_id(self, item: dict[str, Any]) -> str | None:
        source_id = item.get("source_id", "")
        if source_id.startswith("gen_"):
            return source_id[4:]
        return source_id or None

    # --- FusionMixin required methods ---

    def _get_fusion_sources(self) -> list[str]:
        return ["elo", "debate", "ranking"]

    def _extract_fusible_data(self, km_item: dict[str, Any]) -> dict[str, Any] | None:
        if km_item.get("metadata", {}).get("adapter") == "genesis":
            return {
                "fitness_score": km_item.get("metadata", {}).get("fitness_score", 0.0),
                "generation": km_item.get("metadata", {}).get("generation", 0),
            }
        return None

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record.metadata["fusion_applied"] = True
        record.metadata["fusion_timestamp"] = datetime.now(timezone.utc).isoformat()
        if hasattr(fusion_result, "fused_confidence"):
            record.metadata["fused_confidence"] = fusion_result.fused_confidence
        return True
