"""
GenesisAdapter - Bridges agent evolution and fractal debate data to the Knowledge Mound.

This adapter enables persistence of evolutionary agent data:

- Data flow IN: Agent genomes (traits, fitness, lineage) stored as knowledge items
- Data flow IN: Fractal debate results (recursive trees) stored with provenance
- Reverse flow: KM can retrieve lineage and validate genomes
- Lineage queries: Retrieve full agent evolution history
- Fitness search: Find high-performing genomes by domain or threshold

The adapter provides:
- Lightweight GenesisEvolutionItem dataclass to decouple from genesis module
- Automatic extraction from AgentGenome and FractalResult objects
- Batch sync of pending evolution items to Knowledge Mound
- Agent lineage retrieval for cross-debate learning
- Semantic search over high-fitness genomes

"Every agent carries the memory of its ancestors."
"""

from __future__ import annotations

import hashlib
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

# Graceful import of genesis types -- adapter works without them
_HAS_GENESIS = False
try:
    from aragora.genesis.genome import AgentGenome as _AgentGenome
    from aragora.genesis.fractal import FractalResult as _FractalResult

    _HAS_GENESIS = True
except ImportError:
    _AgentGenome = None  # type: ignore[assignment,misc]
    _FractalResult = None  # type: ignore[assignment,misc]


@dataclass
class GenesisEvolutionItem:
    """Lightweight representation of agent evolution data for adapter storage.

    Decoupled from the full AgentGenome / FractalResult dataclasses to avoid
    importing genesis-specific dependencies. Supports both genome and fractal
    result item types.
    """

    genome_id: str
    name: str
    item_type: str = "genome"  # "genome" or "fractal_result"
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
    # Fractal-specific fields (when item_type == "fractal_result")
    root_debate_id: str = ""
    sub_debate_count: int = 0
    total_depth: int = 0
    tensions_resolved: int = 0
    tensions_unresolved: int = 0
    evolved_genome_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_genome(cls, genome: Any) -> GenesisEvolutionItem:
        """Create a GenesisEvolutionItem from an AgentGenome object."""
        return cls(
            genome_id=getattr(genome, "genome_id", ""),
            name=getattr(genome, "name", ""),
            item_type="genome",
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

    @classmethod
    def from_fractal_result(cls, result: Any) -> GenesisEvolutionItem:
        """Create a GenesisEvolutionItem from a FractalResult object."""
        main_result = getattr(result, "main_result", None)
        task = getattr(main_result, "task", "") if main_result else ""
        confidence = getattr(main_result, "confidence", 0.5) if main_result else 0.5
        evolved = getattr(result, "evolved_genomes", [])

        root_id = getattr(result, "root_debate_id", "")
        item_id_hash = hashlib.sha256(root_id.encode()).hexdigest()[:12]

        return cls(
            genome_id=f"frac_{item_id_hash}",
            name="fractal_orchestrator",
            item_type="fractal_result",
            fitness_score=confidence,
            root_debate_id=root_id,
            sub_debate_count=len(getattr(result, "sub_debates", [])),
            total_depth=getattr(result, "total_depth", 0),
            tensions_resolved=getattr(result, "tensions_resolved", 0),
            tensions_unresolved=getattr(result, "tensions_unresolved", 0),
            evolved_genome_ids=[getattr(g, "genome_id", "") for g in evolved],
            metadata={"task": task},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenesisEvolutionItem:
        """Create a GenesisEvolutionItem from a plain dict."""
        return cls(
            genome_id=data.get("genome_id", ""),
            name=data.get("name", data.get("agent_name", "")),
            item_type=data.get("item_type", "genome"),
            traits=data.get("traits", {}),
            expertise=data.get("expertise", {}),
            model_preference=data.get("model_preference", "claude"),
            parent_genomes=data.get("parent_genomes", []),
            generation=data.get("generation", 0),
            fitness_score=data.get("fitness_score", 0.5),
            birth_debate_id=data.get("birth_debate_id"),
            consensus_contributions=data.get("consensus_contributions", 0),
            critiques_accepted=data.get("critiques_accepted", 0),
            predictions_correct=data.get("predictions_correct", 0),
            debates_participated=data.get("debates_participated", 0),
            root_debate_id=data.get("root_debate_id", ""),
            sub_debate_count=data.get("sub_debate_count", 0),
            total_depth=data.get("total_depth", 0),
            tensions_resolved=data.get("tensions_resolved", 0),
            tensions_unresolved=data.get("tensions_unresolved", 0),
            evolved_genome_ids=data.get("evolved_genome_ids", []),
            metadata=data.get("metadata", {}),
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

    def ingest(self, item: Any) -> bool:
        """Ingest a genesis data item for KM sync.

        Accepts AgentGenome, FractalResult, GenesisEvolutionItem, or a plain dict.

        Args:
            item: The genesis data to ingest.

        Returns:
            True if ingestion succeeded, False otherwise.
        """
        try:
            if isinstance(item, GenesisEvolutionItem):
                evolution_item = item
            elif isinstance(item, dict):
                evolution_item = GenesisEvolutionItem.from_dict(item)
            elif _HAS_GENESIS and _AgentGenome is not None and isinstance(item, _AgentGenome):
                evolution_item = GenesisEvolutionItem.from_genome(item)
            elif _HAS_GENESIS and _FractalResult is not None and isinstance(item, _FractalResult):
                evolution_item = GenesisEvolutionItem.from_fractal_result(item)
            elif hasattr(item, "genome_id") and hasattr(item, "traits"):
                # Duck-type check for AgentGenome-like objects (check before fractal)
                evolution_item = GenesisEvolutionItem.from_genome(item)
            elif hasattr(item, "root_debate_id") and hasattr(item, "sub_debates"):
                # Duck-type check for FractalResult-like objects
                evolution_item = GenesisEvolutionItem.from_fractal_result(item)
            else:
                logger.warning(
                    "[genesis_adapter] Unknown item type: %s",
                    type(item).__name__,
                )
                return False

            evolution_item.metadata["km_sync_pending"] = True
            evolution_item.metadata["km_sync_requested_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._pending_genomes.append(evolution_item)

            self._emit_event(
                "km_adapter_forward_sync",
                {
                    "adapter": self.adapter_name,
                    "genome_id": evolution_item.genome_id,
                    "name": evolution_item.name,
                    "item_type": evolution_item.item_type,
                    "fitness_score": evolution_item.fitness_score,
                    "generation": evolution_item.generation,
                },
            )

            logger.info(
                "[genesis_adapter] Ingested %s item %s",
                evolution_item.item_type,
                evolution_item.genome_id,
            )
            return True

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.warning("[genesis_adapter] Ingest failed: %s", e)
            return False

    def store_genome(self, genome: Any) -> None:
        """Store a genome for KM sync.

        Convenience wrapper around ingest() for backward compatibility.

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
            if item.item_type != "genome":
                continue
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

        if item.item_type == "fractal_result":
            content = (
                f"Fractal Debate Result: {item.metadata.get('task', 'unknown')}\n"
                f"Depth: {item.total_depth}, Sub-debates: {item.sub_debate_count}\n"
                f"Tensions resolved: {item.tensions_resolved}, "
                f"Unresolved: {item.tensions_unresolved}\n"
                f"Evolved genomes: {len(item.evolved_genome_ids)}"
            )
            source_id = item.root_debate_id or item.genome_id
        else:
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
            source_id = item.genome_id

        return KnowledgeItem(
            id=f"gen_{item.genome_id}",
            content=content,
            source=KnowledgeSource.DEBATE,
            source_id=source_id,
            confidence=ConfidenceLevel.from_float(item.fitness_score),
            created_at=item.created_at,
            updated_at=item.created_at,
            metadata={
                "adapter": "genesis",
                "item_type": item.item_type,
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
                "root_debate_id": item.root_debate_id,
                "sub_debate_count": item.sub_debate_count,
                "total_depth": item.total_depth,
                "tensions_resolved": item.tensions_resolved,
                "tensions_unresolved": item.tensions_unresolved,
                "evolved_genome_ids": item.evolved_genome_ids,
                "tags": [
                    "genesis",
                    f"type:{item.item_type}",
                    f"agent:{item.name}",
                ],
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
        """Get statistics about stored genesis items."""
        all_items = list(self._synced_genomes.values())
        genomes = [i for i in all_items if i.item_type == "genome"]
        fractals = [i for i in all_items if i.item_type == "fractal_result"]

        return {
            "total_synced": len(self._synced_genomes),
            "pending_sync": len(self._pending_genomes),
            "genomes_stored": len(genomes),
            "fractals_stored": len(fractals),
            "avg_fitness": (
                sum(g.fitness_score for g in genomes) / len(genomes) if genomes else 0.0
            ),
            "max_generation": max((g.generation for g in genomes), default=0),
            "total_debates": sum(g.debates_participated for g in genomes),
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
        if source_id.startswith("frac_"):
            return source_id[5:]
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


# Module-level singleton for cross-module access
_genesis_adapter_singleton: GenesisAdapter | None = None


def get_genesis_adapter() -> GenesisAdapter:
    """Get or create the module-level GenesisAdapter singleton.

    Returns:
        The singleton GenesisAdapter instance.
    """
    global _genesis_adapter_singleton
    if _genesis_adapter_singleton is None:
        _genesis_adapter_singleton = GenesisAdapter()
    return _genesis_adapter_singleton


__all__ = [
    "GenesisAdapter",
    "GenesisEvolutionItem",
    "GenesisSearchResult",
    "get_genesis_adapter",
]
