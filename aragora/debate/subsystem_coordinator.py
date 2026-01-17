"""
Subsystem coordinator for Arena tracking and detection systems.

This module extracts subsystem management from the Arena god object,
following the Single Responsibility Principle. The coordinator handles:

1. **Position Tracking**: PositionTracker, PositionLedger
2. **Agent Ranking**: ELO system, calibration tracking
3. **Memory Systems**: ConsensusMemory, DissentRetriever, ContinuumMemory
4. **Detection Systems**: FlipDetector, MomentDetector
5. **Relationship Tracking**: RelationshipTracker, TierAnalyticsTracker

Usage:
    # Create coordinator with optional pre-configured subsystems
    coordinator = SubsystemCoordinator(
        protocol=protocol,
        loop_id="debate-123",
        elo_system=elo,  # Pre-configured
        enable_position_ledger=True,  # Auto-create
    )

    # Access subsystems (lazy initialization)
    ledger = coordinator.position_ledger
    if coordinator.has_calibration:
        tracker = coordinator.calibration_tracker

    # After debate, update tracking
    coordinator.on_debate_complete(ctx, result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.grounded import MomentDetector
    from aragora.agents.positions import PositionLedger
    from aragora.agents.truth_grounding import PositionTracker
    from aragora.core import DebateResult
    from aragora.insights.flip_detector import FlipDetector
    from aragora.debate.context import DebateContext
    from aragora.debate.protocol import DebateProtocol
    from aragora.memory.consensus import ConsensusMemory, DissentRetriever
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.tier_analytics import TierAnalyticsTracker
    from aragora.ranking.elo import EloSystem
    from aragora.relationships.tracker import RelationshipTracker


@runtime_checkable
class Resettable(Protocol):
    """Protocol for objects that can be reset."""

    def reset(self) -> None:
        """Reset internal state."""
        ...

logger = logging.getLogger(__name__)


@dataclass
class SubsystemCoordinator:
    """Coordinates tracking and detection subsystems for Arena.

    Provides a centralized place to manage optional subsystems that enhance
    debate capabilities. Handles lazy initialization and graceful fallbacks.

    Subsystems are grouped by function:

    **Position Systems** (track agent stances):
    - position_tracker: Real-time position tracking during debate
    - position_ledger: Persistent record of all positions across debates

    **Agent Ranking** (track agent skill):
    - elo_system: ELO ratings for agent skill ranking
    - calibration_tracker: Prediction accuracy tracking

    **Memory Systems** (cross-debate learning):
    - consensus_memory: Historical debate outcomes
    - dissent_retriever: Historical minority viewpoints
    - continuum_memory: Cross-debate learning memory

    **Detection Systems** (identify patterns):
    - flip_detector: Position reversal detection
    - moment_detector: Significant moment identification

    **Relationship Systems** (agent interactions):
    - relationship_tracker: Inter-agent relationship tracking
    - tier_analytics_tracker: Memory tier ROI analysis
    """

    # Protocol reference for breakpoint configuration
    protocol: Optional["DebateProtocol"] = None
    loop_id: str = ""

    # Position tracking subsystems
    position_tracker: Optional["PositionTracker"] = None
    position_ledger: Optional["PositionLedger"] = None
    enable_position_ledger: bool = False

    # Agent ranking subsystems
    elo_system: Optional["EloSystem"] = None
    calibration_tracker: Optional["CalibrationTracker"] = None
    enable_calibration: bool = False

    # Persona management
    persona_manager: Optional[Any] = None

    # Memory subsystems
    consensus_memory: Optional["ConsensusMemory"] = None
    dissent_retriever: Optional["DissentRetriever"] = None
    continuum_memory: Optional["ContinuumMemory"] = None

    # Detection subsystems
    flip_detector: Optional["FlipDetector"] = None
    moment_detector: Optional["MomentDetector"] = None
    enable_moment_detection: bool = False

    # Relationship subsystems
    relationship_tracker: Optional["RelationshipTracker"] = None
    tier_analytics_tracker: Optional["TierAnalyticsTracker"] = None

    # Internal state
    _initialized: bool = field(default=False, repr=False)
    _init_errors: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize subsystems after dataclass fields are set."""
        self._auto_init_subsystems()
        self._initialized = True

    # =========================================================================
    # Property accessors with capability checks
    # =========================================================================

    @property
    def has_position_tracking(self) -> bool:
        """Check if position tracking is available."""
        return self.position_tracker is not None or self.position_ledger is not None

    @property
    def has_elo(self) -> bool:
        """Check if ELO ranking is available."""
        return self.elo_system is not None

    @property
    def has_calibration(self) -> bool:
        """Check if calibration tracking is available."""
        return self.calibration_tracker is not None

    @property
    def has_consensus_memory(self) -> bool:
        """Check if consensus memory is available."""
        return self.consensus_memory is not None

    @property
    def has_dissent_retrieval(self) -> bool:
        """Check if dissent retrieval is available."""
        return self.dissent_retriever is not None

    @property
    def has_moment_detection(self) -> bool:
        """Check if moment detection is available."""
        return self.moment_detector is not None

    @property
    def has_relationship_tracking(self) -> bool:
        """Check if relationship tracking is available."""
        return self.relationship_tracker is not None

    @property
    def has_continuum_memory(self) -> bool:
        """Check if cross-debate memory is available."""
        return self.continuum_memory is not None

    # =========================================================================
    # Auto-initialization methods
    # =========================================================================

    def _auto_init_subsystems(self) -> None:
        """Auto-initialize subsystems based on flags and dependencies."""
        # Position ledger
        if self.enable_position_ledger and self.position_ledger is None:
            self._auto_init_position_ledger()

        # Calibration tracker
        if self.enable_calibration and self.calibration_tracker is None:
            self._auto_init_calibration_tracker()

        # Dissent retriever (requires consensus_memory)
        if self.consensus_memory is not None and self.dissent_retriever is None:
            self._auto_init_dissent_retriever()

        # Moment detector (benefits from elo_system)
        if self.enable_moment_detection and self.moment_detector is None:
            self._auto_init_moment_detector()

    def _auto_init_position_ledger(self) -> None:
        """Auto-initialize PositionLedger for tracking agent positions.

        PositionLedger tracks every position agents take across debates,
        including outcomes and reversals.
        """
        try:
            from aragora.agents.positions import PositionLedger

            self.position_ledger = PositionLedger()
            logger.debug("Auto-initialized PositionLedger for position tracking")
        except ImportError:
            logger.warning("PositionLedger not available - position tracking disabled")
            self._init_errors.append("PositionLedger import failed")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("PositionLedger auto-init failed: %s", e)
            self._init_errors.append(f"PositionLedger init failed: {e}")

    def _auto_init_calibration_tracker(self) -> None:
        """Auto-initialize CalibrationTracker for prediction accuracy."""
        try:
            from aragora.agents.calibration import CalibrationTracker

            self.calibration_tracker = CalibrationTracker()
            logger.debug("Auto-initialized CalibrationTracker for prediction calibration")
        except ImportError:
            logger.warning("CalibrationTracker not available - calibration disabled")
            self._init_errors.append("CalibrationTracker import failed")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("CalibrationTracker auto-init failed: %s", e)
            self._init_errors.append(f"CalibrationTracker init failed: {e}")

    def _auto_init_dissent_retriever(self) -> None:
        """Auto-initialize DissentRetriever for historical minority views.

        The DissentRetriever enables seeding new debates with historical minority
        views, helping agents avoid past groupthink.
        """
        try:
            from aragora.memory.consensus import DissentRetriever

            self.dissent_retriever = DissentRetriever(self.consensus_memory)
            logger.debug("Auto-initialized DissentRetriever for historical minority views")
        except ImportError:
            logger.debug("DissentRetriever not available - historical dissent disabled")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("DissentRetriever auto-init failed: %s", e)
            self._init_errors.append(f"DissentRetriever init failed: {e}")

    def _auto_init_moment_detector(self) -> None:
        """Auto-initialize MomentDetector for significant moment detection."""
        try:
            from aragora.agents.grounded import MomentDetector

            self.moment_detector = MomentDetector(
                elo_system=self.elo_system,
                position_ledger=self.position_ledger,
                relationship_tracker=self.relationship_tracker,
            )
            logger.debug("Auto-initialized MomentDetector for significant moment detection")
        except ImportError:
            logger.debug("MomentDetector not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("MomentDetector auto-init failed: %s", e)
            self._init_errors.append(f"MomentDetector init failed: {e}")

    # =========================================================================
    # Lifecycle hooks
    # =========================================================================

    def on_debate_start(self, ctx: "DebateContext") -> None:
        """Called when a debate starts.

        Args:
            ctx: The debate context being initialized
        """
        # Reset moment detector for new debate if it supports reset
        if self.moment_detector and isinstance(self.moment_detector, Resettable):
            try:
                self.moment_detector.reset()
            except Exception as e:
                logger.debug("MomentDetector reset failed: %s", e)

    def on_round_complete(
        self,
        ctx: "DebateContext",
        round_num: int,
        positions: dict[str, str],
    ) -> None:
        """Called when a debate round completes.

        Args:
            ctx: The debate context
            round_num: The round number that completed
            positions: Agent name -> position mapping
        """
        # Record positions in ledger
        if self.position_ledger:
            for agent_name, position in positions.items():
                try:
                    self.position_ledger.record_position(
                        agent_name=agent_name,
                        claim=position,
                        confidence=0.5,  # Default confidence when not specified
                        debate_id=ctx.debate_id,
                        round_num=round_num,
                    )
                except Exception as e:
                    logger.debug("Position recording failed: %s", e)

    def on_debate_complete(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
    ) -> None:
        """Called when a debate completes.

        Updates all tracking subsystems with debate outcome.

        Args:
            ctx: The debate context
            result: The final debate result
        """
        # Update consensus memory
        if self.consensus_memory and result:
            try:
                # Get task from environment
                task = ctx.env.task if ctx.env else ""
                consensus_text = getattr(result, "consensus", "") or ""
                confidence = getattr(result, "consensus_confidence", 0.0)
                participants = [a.name for a in ctx.agents] if ctx.agents else []

                # Import ConsensusStrength for the call
                from aragora.memory.consensus import ConsensusStrength

                # Determine strength based on confidence
                if confidence >= 0.9:
                    strength = ConsensusStrength.UNANIMOUS
                elif confidence >= 0.8:
                    strength = ConsensusStrength.STRONG
                elif confidence >= 0.6:
                    strength = ConsensusStrength.MODERATE
                elif confidence >= 0.5:
                    strength = ConsensusStrength.WEAK
                else:
                    strength = ConsensusStrength.SPLIT

                self.consensus_memory.store_consensus(
                    topic=task,
                    conclusion=consensus_text,
                    strength=strength,
                    confidence=confidence,
                    participating_agents=participants,
                    agreeing_agents=participants,  # Simplified: assume all agree at consensus
                    metadata={"debate_id": ctx.debate_id},
                )
            except Exception as e:
                logger.warning("Consensus memory update failed: %s", e)

        # Update calibration if agents made predictions
        if self.calibration_tracker and result:
            try:
                # Record prediction outcomes for calibration
                predictions: dict[str, Any] = getattr(result, "predictions", {})
                actual_outcome = getattr(result, "consensus", "")
                for agent_name, prediction in predictions.items():
                    # CalibrationTracker.record_prediction expects:
                    # (agent, confidence, correct, domain, debate_id, position_id)
                    predicted_value = prediction.get("prediction", "") if isinstance(prediction, dict) else str(prediction)
                    pred_confidence = prediction.get("confidence", 0.5) if isinstance(prediction, dict) else 0.5
                    is_correct = predicted_value == actual_outcome
                    self.calibration_tracker.record_prediction(
                        agent=agent_name,
                        confidence=pred_confidence,
                        correct=is_correct,
                        domain=ctx.domain,
                        debate_id=ctx.debate_id,
                    )
            except Exception as e:
                logger.debug("Calibration update failed: %s", e)

        # Update continuum memory with debate outcome
        if self.continuum_memory and result:
            try:
                # ContinuumMemory uses add() method
                # Store the debate outcome as a memory entry
                task = ctx.env.task if ctx.env else ""
                consensus_text = getattr(result, "consensus", "") or ""
                confidence = getattr(result, "consensus_confidence", 0.0)

                from aragora.memory.continuum import MemoryTier

                self.continuum_memory.add(
                    id=f"debate:{ctx.debate_id}",
                    content=f"Debate outcome: {consensus_text[:200]}",
                    tier=MemoryTier.MEDIUM,
                    importance=confidence,
                    metadata={
                        "debate_id": ctx.debate_id,
                        "task": task,
                        "consensus": consensus_text,
                        "confidence": confidence,
                    },
                )
            except Exception as e:
                logger.debug("Continuum memory update failed: %s", e)

    # =========================================================================
    # Query methods
    # =========================================================================

    def get_historical_dissent(
        self,
        task: str,
        limit: int = 3,
    ) -> list[dict]:
        """Get historical minority viewpoints related to a task.

        Args:
            task: The debate task/question
            limit: Maximum number of dissenting views to return

        Returns:
            List of dissenting view records with position, agent, outcome
        """
        if not self.dissent_retriever:
            return []

        try:
            # DissentRetriever uses retrieve_for_new_debate() method
            result = self.dissent_retriever.retrieve_for_new_debate(task)
            # Extract relevant dissents from the result dict
            dissents = result.get("relevant_dissents", [])
            return dissents[:limit]
        except Exception as e:
            logger.debug("Dissent retrieval failed: %s", e)
            return []

    def get_agent_calibration_weight(self, agent_name: str) -> float:
        """Get calibration weight for an agent.

        Higher weights indicate better prediction accuracy.

        Args:
            agent_name: Name of the agent

        Returns:
            Weight between 0.5 and 2.0, default 1.0
        """
        if not self.calibration_tracker:
            return 1.0

        try:
            # CalibrationTracker uses get_calibration_summary() method
            summary = self.calibration_tracker.get_calibration_summary(agent_name)
            if summary and summary.total_predictions > 0:
                # Convert calibration score to weight
                # CalibrationSummary has brier_score (lower is better)
                # Convert: perfect (0.0) -> weight 1.5, poor (0.25) -> weight 0.8
                # Using 1 - brier_score as calibration quality
                calibration_quality = 1.0 - min(summary.brier_score, 0.5)
                return 0.5 + calibration_quality  # Range: 0.5 to 1.5
            return 1.0
        except (KeyError, AttributeError, TypeError) as e:
            logger.debug(f"Could not get calibration weight for {agent_name}: {e}")
            return 1.0

    def get_continuum_context(self, task: str, limit: int = 5) -> str:
        """Get cross-debate context from continuum memory.

        Args:
            task: The debate task for context retrieval
            limit: Maximum number of relevant memories

        Returns:
            Formatted context string or empty string
        """
        if not self.continuum_memory:
            return ""

        try:
            memories = self.continuum_memory.retrieve(query=task, limit=limit)
            if not memories:
                return ""

            # Format memories for prompt injection
            # ContinuumMemory.retrieve() returns List[ContinuumMemoryEntry]
            lines = ["Relevant learnings from past debates:"]
            for mem in memories:
                # ContinuumMemoryEntry has content attribute and metadata dict
                summary = mem.metadata.get("summary", "") if mem.metadata else ""
                content = summary or mem.content
                lines.append(f"- {content}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Continuum context retrieval failed: %s", e)
            return ""

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_status(self) -> dict:
        """Get status of all subsystems.

        Returns:
            Dictionary with subsystem availability and any init errors
        """
        return {
            "subsystems": {
                "position_tracker": self.position_tracker is not None,
                "position_ledger": self.position_ledger is not None,
                "elo_system": self.elo_system is not None,
                "calibration_tracker": self.calibration_tracker is not None,
                "consensus_memory": self.consensus_memory is not None,
                "dissent_retriever": self.dissent_retriever is not None,
                "continuum_memory": self.continuum_memory is not None,
                "flip_detector": self.flip_detector is not None,
                "moment_detector": self.moment_detector is not None,
                "relationship_tracker": self.relationship_tracker is not None,
                "tier_analytics_tracker": self.tier_analytics_tracker is not None,
                "persona_manager": self.persona_manager is not None,
            },
            "capabilities": {
                "position_tracking": self.has_position_tracking,
                "elo_ranking": self.has_elo,
                "calibration": self.has_calibration,
                "consensus_memory": self.has_consensus_memory,
                "dissent_retrieval": self.has_dissent_retrieval,
                "moment_detection": self.has_moment_detection,
                "relationship_tracking": self.has_relationship_tracking,
                "continuum_memory": self.has_continuum_memory,
            },
            "init_errors": self._init_errors,
            "initialized": self._initialized,
        }


@dataclass
class SubsystemConfig:
    """Configuration for creating SubsystemCoordinator.

    This provides a clean way to configure subsystems before
    creating the coordinator.
    """

    # Enable flags
    enable_position_ledger: bool = False
    enable_calibration: bool = False
    enable_moment_detection: bool = False

    # Pre-configured subsystems (optional)
    position_tracker: Optional[Any] = None
    position_ledger: Optional[Any] = None
    elo_system: Optional[Any] = None
    calibration_tracker: Optional[Any] = None
    persona_manager: Optional[Any] = None
    consensus_memory: Optional[Any] = None
    dissent_retriever: Optional[Any] = None
    continuum_memory: Optional[Any] = None
    flip_detector: Optional[Any] = None
    moment_detector: Optional[Any] = None
    relationship_tracker: Optional[Any] = None
    tier_analytics_tracker: Optional[Any] = None

    def create_coordinator(
        self,
        protocol: Optional["DebateProtocol"] = None,
        loop_id: str = "",
    ) -> SubsystemCoordinator:
        """Create SubsystemCoordinator from this configuration.

        Args:
            protocol: The debate protocol (for breakpoint config)
            loop_id: Loop ID for multi-loop scoping

        Returns:
            Configured SubsystemCoordinator instance
        """
        return SubsystemCoordinator(
            protocol=protocol,
            loop_id=loop_id,
            position_tracker=self.position_tracker,
            position_ledger=self.position_ledger,
            enable_position_ledger=self.enable_position_ledger,
            elo_system=self.elo_system,
            calibration_tracker=self.calibration_tracker,
            enable_calibration=self.enable_calibration,
            persona_manager=self.persona_manager,
            consensus_memory=self.consensus_memory,
            dissent_retriever=self.dissent_retriever,
            continuum_memory=self.continuum_memory,
            flip_detector=self.flip_detector,
            moment_detector=self.moment_detector,
            enable_moment_detection=self.enable_moment_detection,
            relationship_tracker=self.relationship_tracker,
            tier_analytics_tracker=self.tier_analytics_tracker,
        )


__all__ = ["SubsystemCoordinator", "SubsystemConfig"]
