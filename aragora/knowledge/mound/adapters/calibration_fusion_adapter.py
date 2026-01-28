"""
CalibrationFusionAdapter - Multi-Party Calibration Fusion for Knowledge Mound Phase A3.

This adapter bridges the CalibrationFusionEngine to the Knowledge Mound,
enabling multi-party prediction fusion with proper uncertainty quantification.

Features:
- Fuse agent predictions into consensus with weighted averaging
- Track outliers and inter-rater agreement (Krippendorff's alpha)
- Store calibration consensus in Knowledge Mound
- Integrate with debate orchestrator for real-time consensus

Usage:
    from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
    from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

    adapter = CalibrationFusionAdapter()

    # Fuse predictions from multiple agents
    predictions = [
        AgentPrediction("claude", 0.8, "winner_a"),
        AgentPrediction("gpt-4", 0.75, "winner_a"),
        AgentPrediction("gemini", 0.6, "winner_b"),
    ]
    consensus = adapter.fuse_predictions(predictions, debate_id="debate_123")

    # Store consensus in Knowledge Mound
    item = adapter.to_knowledge_item(consensus)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter, EventCallback
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin
from aragora.knowledge.mound.ops.calibration_fusion import (
    AgentPrediction,
    CalibrationConsensus,
    CalibrationFusionConfig,
    CalibrationFusionEngine,
    CalibrationFusionStrategy,
    get_calibration_fusion_engine,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem

logger = logging.getLogger(__name__)


class CalibrationSyncResult(TypedDict):
    """Result type for calibration sync operations."""

    predictions_processed: int
    consensus_stored: int
    outliers_detected: int
    high_confidence_count: int
    needs_review_count: int
    errors: List[str]
    duration_ms: float


@dataclass
class CalibrationSearchResult:
    """Wrapper for calibration search results with metadata."""

    consensus: CalibrationConsensus
    similarity: float = 0.0
    stored_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "consensus": self.consensus.to_dict(),
            "similarity": self.similarity,
            "stored_at": self.stored_at.isoformat() if self.stored_at else None,
        }


class CalibrationFusionAdapter(FusionMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges CalibrationFusionEngine to the Knowledge Mound.

    Provides methods for multi-party calibration fusion:
    - fuse_predictions: Aggregate agent predictions into consensus
    - store_consensus: Persist consensus to Knowledge Mound
    - get_consensus: Retrieve stored consensus by debate ID
    - to_knowledge_item: Convert consensus to unified format

    The adapter tracks:
    - Consensus strength and agreement ratios
    - Outlier detection using modified Z-scores
    - Krippendorff's alpha for inter-rater agreement
    - Historical fusion performance by agent

    Example:
        >>> adapter = CalibrationFusionAdapter()
        >>> predictions = [
        ...     AgentPrediction("claude", 0.8, "winner_a"),
        ...     AgentPrediction("gpt-4", 0.75, "winner_a"),
        ... ]
        >>> consensus = adapter.fuse_predictions(predictions, "debate_123")
        >>> print(f"Fused confidence: {consensus.fused_confidence:.2f}")
    """

    adapter_name = "calibration_fusion"
    source_type = "calibration"

    def __init__(
        self,
        engine: Optional[CalibrationFusionEngine] = None,
        config: Optional[CalibrationFusionConfig] = None,
        enable_dual_write: bool = False,
        event_callback: Optional[EventCallback] = None,
        enable_tracing: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            engine: Optional CalibrationFusionEngine instance. Uses singleton if not provided.
            config: Optional configuration for the engine.
            enable_dual_write: If True, writes go to both systems during migration.
            event_callback: Optional callback for emitting events (event_type, data).
            enable_tracing: If True, OpenTelemetry tracing is enabled.
        """
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_tracing=enable_tracing,
        )

        # Use provided engine or get/create singleton
        if engine is not None:
            self._engine = engine
        else:
            self._engine = get_calibration_fusion_engine(config)

        # In-memory storage for consensus (KM integration would persist externally)
        self._stored_consensus: Dict[str, CalibrationConsensus] = {}
        self._consensus_by_topic: Dict[str, List[str]] = {}  # topic -> list of debate_ids

        # Initialize fusion state
        self._init_fusion_state()

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured.

        Override required because FusionMixin stub shadows KnowledgeMoundAdapter.
        """
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"[{self.adapter_name}] Failed to emit event {event_type}: {e}")

    @property
    def engine(self) -> CalibrationFusionEngine:
        """Access the underlying CalibrationFusionEngine."""
        return self._engine

    def fuse_predictions(
        self,
        predictions: List[AgentPrediction],
        debate_id: str = "",
        weights: Optional[Dict[str, float]] = None,
        strategy: CalibrationFusionStrategy = CalibrationFusionStrategy.WEIGHTED_AVERAGE,
        store: bool = True,
    ) -> CalibrationConsensus:
        """
        Fuse multiple agent predictions into a calibration consensus.

        This is the main entry point for multi-party calibration fusion.
        Predictions are aggregated using the specified strategy, with
        automatic outlier detection and agreement metrics.

        Args:
            predictions: List of agent predictions to fuse.
            debate_id: ID of the debate or event being predicted.
            weights: Optional explicit weights per agent.
            strategy: Fusion strategy to use.
            store: If True, store the consensus for later retrieval.

        Returns:
            CalibrationConsensus with fused result and metrics.
        """
        start_time = time.time()

        # Emit start event
        self._emit_event(
            "calibration_fusion_start",
            {
                "adapter": self.adapter_name,
                "debate_id": debate_id,
                "prediction_count": len(predictions),
                "strategy": strategy.value,
            },
        )

        try:
            # Perform fusion
            consensus = self._engine.fuse_predictions(
                predictions=predictions,
                debate_id=debate_id,
                weights=weights,
                strategy=strategy,
            )

            # Store if requested
            if store and debate_id:
                self._store_consensus(consensus)

            # Update fusion state
            self._fusion_state.fusions_performed += 1
            if consensus.consensus_strength >= 0.7:
                self._fusion_state.source_participation["high_confidence"] = (
                    self._fusion_state.source_participation.get("high_confidence", 0) + 1
                )

            latency = time.time() - start_time

            # Record metrics
            self._record_metric("calibration_fusion", success=True, latency=latency)

            # Emit completion event
            self._emit_event(
                "calibration_fusion_complete",
                {
                    "adapter": self.adapter_name,
                    "debate_id": debate_id,
                    "fused_confidence": consensus.fused_confidence,
                    "consensus_strength": consensus.consensus_strength,
                    "agreement_ratio": consensus.agreement_ratio,
                    "outliers_detected": consensus.outliers_detected,
                    "duration_ms": latency * 1000,
                },
            )

            logger.debug(
                f"Calibration fusion complete: debate={debate_id}, "
                f"confidence={consensus.fused_confidence:.2f}, "
                f"strength={consensus.consensus_strength:.2f}"
            )

            return consensus

        except Exception as e:
            latency = time.time() - start_time
            self._record_metric("calibration_fusion", success=False, latency=latency)
            logger.error(f"Calibration fusion failed: {e}")
            raise

    def _store_consensus(self, consensus: CalibrationConsensus) -> None:
        """Store consensus for later retrieval."""
        debate_id = consensus.debate_id
        self._stored_consensus[debate_id] = consensus

        # Index by topic if available
        if consensus.metadata.get("topic"):
            topic = consensus.metadata["topic"]
            if topic not in self._consensus_by_topic:
                self._consensus_by_topic[topic] = []
            if debate_id not in self._consensus_by_topic[topic]:
                self._consensus_by_topic[topic].append(debate_id)

    def get_consensus(self, debate_id: str) -> Optional[CalibrationConsensus]:
        """
        Get stored calibration consensus by debate ID.

        Args:
            debate_id: The debate ID to look up.

        Returns:
            CalibrationConsensus or None if not found.
        """
        return self._stored_consensus.get(debate_id)

    def search_by_topic(
        self,
        topic: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[CalibrationSearchResult]:
        """
        Search for calibration consensus by topic.

        Args:
            topic: Topic to search for (substring match).
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of CalibrationSearchResult sorted by confidence.
        """
        results: List[CalibrationSearchResult] = []

        # Search in indexed topics
        for stored_topic, debate_ids in self._consensus_by_topic.items():
            if topic.lower() in stored_topic.lower():
                for debate_id in debate_ids:
                    consensus = self._stored_consensus.get(debate_id)
                    if consensus and consensus.fused_confidence >= min_confidence:
                        # Compute similarity based on topic match
                        similarity = (
                            1.0
                            if topic.lower() == stored_topic.lower()
                            else 0.8
                            if stored_topic.lower().startswith(topic.lower())
                            else 0.5
                        )
                        results.append(
                            CalibrationSearchResult(
                                consensus=consensus,
                                similarity=similarity,
                                stored_at=consensus.fused_at,
                            )
                        )

        # Sort by confidence and limit
        results.sort(key=lambda r: (r.similarity, r.consensus.fused_confidence), reverse=True)
        return results[:limit]

    def to_knowledge_item(
        self,
        consensus: CalibrationConsensus,
    ) -> "KnowledgeItem":
        """
        Convert a CalibrationConsensus to a KnowledgeItem.

        Args:
            consensus: The calibration consensus to convert.

        Returns:
            KnowledgeItem for unified knowledge mound API.
        """
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Map consensus strength to confidence level
        if consensus.consensus_strength >= 0.8:
            confidence = ConfidenceLevel.VERIFIED
        elif consensus.consensus_strength >= 0.6:
            confidence = ConfidenceLevel.HIGH
        elif consensus.consensus_strength >= 0.4:
            confidence = ConfidenceLevel.MEDIUM
        elif consensus.consensus_strength >= 0.2:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        # Build content from predicted outcome and confidence
        content = (
            f"Calibration consensus: {consensus.predicted_outcome} "
            f"(confidence: {consensus.fused_confidence:.2f}, "
            f"agreement: {consensus.agreement_ratio:.1%})"
        )

        # Build comprehensive metadata
        metadata: Dict[str, Any] = {
            "debate_id": consensus.debate_id,
            "predicted_outcome": consensus.predicted_outcome,
            "fused_confidence": consensus.fused_confidence,
            "consensus_strength": consensus.consensus_strength,
            "agreement_ratio": consensus.agreement_ratio,
            "disagreement_score": consensus.disagreement_score,
            "krippendorff_alpha": consensus.krippendorff_alpha,
            "strategy_used": consensus.strategy_used.value,
            "outliers_detected": consensus.outliers_detected,
            "confidence_interval": list(consensus.confidence_interval),
            "participating_agents": consensus.participating_agents,
            "weights_used": consensus.weights_used,
            "is_high_confidence": consensus.is_high_confidence,
            "needs_review": consensus.needs_review,
            "source_adapter": self.adapter_name,
        }

        return KnowledgeItem(
            id=f"cf_{consensus.debate_id}",
            content=content,
            source=KnowledgeSource.CALIBRATION,
            source_id=consensus.debate_id,
            confidence=confidence,
            created_at=consensus.fused_at,
            updated_at=consensus.fused_at,
            metadata=metadata,
            importance=consensus.fused_confidence,
        )

    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for an agent across calibration fusions.

        Args:
            agent_name: Name of the agent.

        Returns:
            Dict with performance metrics.
        """
        return self._engine.get_agent_performance(agent_name)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall calibration fusion statistics.

        Returns:
            Dict with fusion metrics and performance data.
        """
        engine_stats = self._engine.get_stats()
        adapter_stats = {
            "adapter_name": self.adapter_name,
            "stored_consensus_count": len(self._stored_consensus),
            "indexed_topics_count": len(self._consensus_by_topic),
            "fusion_state": self._fusion_state.to_dict(),
        }
        return {**engine_stats, **adapter_stats}

    # =========================================================================
    # FusionMixin Implementation
    # =========================================================================

    def _get_fusion_sources(self) -> List[str]:
        """Return list of adapter names this adapter can fuse data from.

        CalibrationFusionAdapter can fuse validations from ELO (agent performance),
        Consensus (debate outcomes), and Belief (claim confidence) adapters.
        """
        return ["elo", "consensus", "belief", "evidence", "ranking"]

    def _extract_fusible_data(self, km_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract data from a KM item that can be used for fusion.

        Args:
            km_item: Knowledge Mound item dict.

        Returns:
            Dict with fusible fields, or None if not fusible.
        """
        metadata = km_item.get("metadata", {})

        # Extract calibration-relevant fields
        item_id = metadata.get("source_id") or metadata.get("debate_id") or km_item.get("id")

        if not item_id:
            return None

        confidence = km_item.get("confidence", 0.5)
        if isinstance(confidence, str):
            confidence = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(confidence.lower(), 0.5)

        return {
            "item_id": item_id,
            "confidence": confidence,
            "source_adapter": metadata.get("source_adapter", "unknown"),
            "fused_confidence": metadata.get("fused_confidence"),
            "consensus_strength": metadata.get("consensus_strength"),
            "agreement_ratio": metadata.get("agreement_ratio"),
            "is_valid": confidence >= 0.5,
            "sources": metadata.get("participating_agents", []),
            "reasoning": metadata.get("reasoning"),
        }

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: Any,  # FusedValidation from ops.fusion
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Apply a fusion result to a calibration record.

        Args:
            record: The CalibrationConsensus to update.
            fusion_result: FusedValidation with fused confidence/validity.
            metadata: Optional additional metadata.

        Returns:
            True if successfully applied, False otherwise.
        """
        try:
            # Update record metadata with fusion results
            record.metadata["multi_adapter_fusion_applied"] = True
            record.metadata["multi_adapter_fused_confidence"] = fusion_result.fused_confidence
            record.metadata["multi_adapter_is_valid"] = fusion_result.is_valid
            record.metadata["multi_adapter_strategy"] = fusion_result.strategy_used.value
            record.metadata["multi_adapter_source_count"] = len(fusion_result.source_validations)
            record.metadata["multi_adapter_timestamp"] = datetime.now(timezone.utc).isoformat()

            if metadata:
                record.metadata["multi_adapter_metadata"] = metadata

            # Emit event for fusion application
            self._emit_event(
                "km_adapter_fusion_applied",
                {
                    "adapter": self.adapter_name,
                    "record_id": getattr(record, "debate_id", "unknown"),
                    "fused_confidence": fusion_result.fused_confidence,
                    "is_valid": fusion_result.is_valid,
                    "source_count": len(fusion_result.source_validations),
                },
            )

            logger.debug(
                f"Applied multi-adapter fusion to calibration record: "
                f"confidence={fusion_result.fused_confidence:.2f}, "
                f"sources={len(fusion_result.source_validations)}"
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to apply fusion result: {e}")
            return False

    def _get_record_for_fusion(self, source_id: str) -> Optional[CalibrationConsensus]:
        """Get a calibration record by source ID for fusion.

        Args:
            source_id: The source ID (debate_id).

        Returns:
            CalibrationConsensus or None if not found.
        """
        # Strip prefix if present
        if source_id.startswith("cf_"):
            source_id = source_id[3:]

        return self.get_consensus(source_id)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def sync_to_km(
        self,
        consensus_list: Optional[List[CalibrationConsensus]] = None,
        limit: int = 100,
    ) -> CalibrationSyncResult:
        """
        Sync calibration consensus to Knowledge Mound.

        Args:
            consensus_list: Optional list of consensus to sync. Uses stored if not provided.
            limit: Maximum items to sync.

        Returns:
            CalibrationSyncResult with sync statistics.
        """
        start_time = time.time()
        result: CalibrationSyncResult = {
            "predictions_processed": 0,
            "consensus_stored": 0,
            "outliers_detected": 0,
            "high_confidence_count": 0,
            "needs_review_count": 0,
            "errors": [],
            "duration_ms": 0.0,
        }

        # Use stored consensus if not provided
        if consensus_list is None:
            consensus_list = list(self._stored_consensus.values())[:limit]

        for consensus in consensus_list:
            try:
                result["predictions_processed"] += len(consensus.predictions)
                result["consensus_stored"] += 1
                result["outliers_detected"] += len(consensus.outliers_detected)

                if consensus.is_high_confidence:
                    result["high_confidence_count"] += 1
                if consensus.needs_review:
                    result["needs_review_count"] += 1

            except Exception as e:
                error_msg = f"Error syncing consensus {consensus.debate_id}: {str(e)}"
                result["errors"].append(error_msg)
                logger.warning(error_msg)

        result["duration_ms"] = (time.time() - start_time) * 1000

        # Record metrics
        self._record_metric(
            "calibration_sync",
            success=len(result["errors"]) == 0,
            latency=result["duration_ms"] / 1000,
        )

        return result

    def health_check(self) -> Dict[str, Any]:
        """Return adapter health status for monitoring.

        Returns:
            Dict containing health status and calibration metrics.
        """
        base_health = super().health_check()
        calibration_health = {
            "stored_consensus_count": len(self._stored_consensus),
            "engine_stats": self._engine.get_stats(),
        }
        return {**base_health, **calibration_health}


__all__ = [
    "CalibrationFusionAdapter",
    "CalibrationSearchResult",
    "CalibrationSyncResult",
]
