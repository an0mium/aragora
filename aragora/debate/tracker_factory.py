"""
Tracker factory for Arena initialization.

Centralizes the creation and auto-initialization of tracking subsystems
to reduce orchestrator complexity and improve testability.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.grounded import MomentDetector
    from aragora.agents.positions import PositionLedger
    from aragora.debate.breakpoints import BreakpointManager, BreakpointConfig
    from aragora.insights.flip_detector import FlipDetector
    from aragora.memory.consensus import ConsensusMemory, DissentRetriever
    from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)


@dataclass
class TrackerBundle:
    """Bundle of tracking subsystems created by TrackerFactory."""

    position_tracker: Any  # PositionTracker
    position_ledger: Optional["PositionLedger"]
    elo_system: Optional["EloSystem"]
    flip_detector: Optional["FlipDetector"]
    calibration_tracker: Optional["CalibrationTracker"]
    dissent_retriever: Optional["DissentRetriever"]
    moment_detector: Optional["MomentDetector"]
    relationship_tracker: Any  # RelationshipTracker
    breakpoint_manager: Optional["BreakpointManager"]


class TrackerFactory:
    """
    Factory for creating and auto-initializing tracking subsystems.

    Consolidates the tracker initialization logic from Arena to improve
    testability and reduce orchestrator complexity.
    """

    @staticmethod
    def create_position_ledger() -> Optional["PositionLedger"]:
        """Create PositionLedger for tracking agent positions.

        PositionLedger tracks every position agents take across debates,
        including outcomes and reversals.
        """
        try:
            from aragora.agents.positions import PositionLedger

            ledger = PositionLedger()
            logger.debug("Created PositionLedger for position tracking")
            return ledger
        except ImportError:
            logger.debug("PositionLedger not available")
            return None
        except Exception as e:
            logger.warning(f"PositionLedger creation failed: {e}")
            return None

    @staticmethod
    def create_calibration_tracker() -> Optional["CalibrationTracker"]:
        """Create CalibrationTracker for prediction calibration."""
        try:
            from aragora.agents.calibration import CalibrationTracker

            tracker = CalibrationTracker()
            logger.debug("Created CalibrationTracker for prediction calibration")
            return tracker
        except ImportError:
            logger.debug("CalibrationTracker not available")
            return None
        except Exception as e:
            logger.warning(f"CalibrationTracker creation failed: {e}")
            return None

    @staticmethod
    def create_dissent_retriever(
        consensus_memory: Optional["ConsensusMemory"],
    ) -> Optional["DissentRetriever"]:
        """Create DissentRetriever for historical minority views.

        The DissentRetriever enables seeding new debates with historical minority
        views, helping agents avoid past groupthink.
        """
        if consensus_memory is None:
            return None

        try:
            from aragora.memory.consensus import DissentRetriever

            retriever = DissentRetriever(consensus_memory)
            logger.debug("Created DissentRetriever for historical minority views")
            return retriever
        except ImportError:
            logger.debug("DissentRetriever not available")
            return None
        except Exception as e:
            logger.warning(f"DissentRetriever creation failed: {e}")
            return None

    @staticmethod
    def create_moment_detector(
        elo_system: Optional["EloSystem"],
        position_ledger: Optional["PositionLedger"] = None,
        relationship_tracker: Any = None,
    ) -> Optional["MomentDetector"]:
        """Create MomentDetector for significant moment detection."""
        if elo_system is None:
            return None

        try:
            from aragora.agents.grounded import MomentDetector

            detector = MomentDetector(
                elo_system=elo_system,
                position_ledger=position_ledger,
                relationship_tracker=relationship_tracker,
            )
            logger.debug("Created MomentDetector for significant moment detection")
            return detector
        except ImportError:
            logger.debug("MomentDetector not available")
            return None
        except Exception as e:
            logger.debug(f"MomentDetector creation failed: {e}")
            return None

    @staticmethod
    def create_breakpoint_manager(
        config: Optional["BreakpointConfig"] = None,
    ) -> Optional["BreakpointManager"]:
        """Create BreakpointManager for human-in-the-loop breakpoints."""
        try:
            from aragora.debate.breakpoints import BreakpointManager, BreakpointConfig

            config = config or BreakpointConfig()
            manager = BreakpointManager(config=config)
            logger.debug("Created BreakpointManager for human-in-the-loop breakpoints")
            return manager
        except ImportError:
            logger.debug("BreakpointManager not available")
            return None
        except Exception as e:
            logger.warning(f"BreakpointManager creation failed: {e}")
            return None

    @staticmethod
    def create_flip_detector() -> Optional["FlipDetector"]:
        """Create FlipDetector for position reversal detection."""
        try:
            from aragora.insights.flip_detector import FlipDetector

            detector = FlipDetector()
            logger.debug("Created FlipDetector for position reversal detection")
            return detector
        except ImportError:
            logger.debug("FlipDetector not available")
            return None
        except Exception as e:
            logger.warning(f"FlipDetector creation failed: {e}")
            return None

    @classmethod
    def auto_init_missing(
        cls,
        *,
        position_ledger: Optional["PositionLedger"],
        calibration_tracker: Optional["CalibrationTracker"],
        dissent_retriever: Optional["DissentRetriever"],
        moment_detector: Optional["MomentDetector"],
        breakpoint_manager: Optional["BreakpointManager"],
        # Dependencies for auto-init
        elo_system: Optional["EloSystem"] = None,
        consensus_memory: Optional["ConsensusMemory"] = None,
        relationship_tracker: Any = None,
        # Config flags
        enable_position_ledger: bool = False,
        enable_calibration: bool = False,
        enable_breakpoints: bool = False,
        breakpoint_config: Optional["BreakpointConfig"] = None,
    ) -> dict:
        """
        Auto-initialize missing trackers based on config flags and dependencies.

        Returns a dict with any newly created trackers.

        Args:
            position_ledger: Existing PositionLedger or None
            calibration_tracker: Existing CalibrationTracker or None
            dissent_retriever: Existing DissentRetriever or None
            moment_detector: Existing MomentDetector or None
            breakpoint_manager: Existing BreakpointManager or None
            elo_system: EloSystem for MomentDetector
            consensus_memory: ConsensusMemory for DissentRetriever
            relationship_tracker: RelationshipTracker for MomentDetector
            enable_position_ledger: Create PositionLedger if missing
            enable_calibration: Create CalibrationTracker if missing
            enable_breakpoints: Create BreakpointManager if missing
            breakpoint_config: Config for BreakpointManager

        Returns:
            Dict with keys for any trackers that were created
        """
        created: dict[str, Any] = {}

        # Auto-init PositionLedger
        if enable_position_ledger and position_ledger is None:
            new_ledger = cls.create_position_ledger()
            if new_ledger:
                created["position_ledger"] = new_ledger

        # Auto-init CalibrationTracker
        if enable_calibration and calibration_tracker is None:
            new_tracker = cls.create_calibration_tracker()
            if new_tracker:
                created["calibration_tracker"] = new_tracker

        # Auto-init DissentRetriever
        if consensus_memory and dissent_retriever is None:
            new_retriever = cls.create_dissent_retriever(consensus_memory)
            if new_retriever:
                created["dissent_retriever"] = new_retriever

        # Auto-init MomentDetector
        if elo_system and moment_detector is None:
            # Use newly created position_ledger if available
            ledger = created.get("position_ledger", position_ledger)
            new_detector = cls.create_moment_detector(
                elo_system=elo_system,
                position_ledger=ledger,
                relationship_tracker=relationship_tracker,
            )
            if new_detector:
                created["moment_detector"] = new_detector

        # Auto-init BreakpointManager
        if enable_breakpoints and breakpoint_manager is None:
            new_manager = cls.create_breakpoint_manager(breakpoint_config)
            if new_manager:
                created["breakpoint_manager"] = new_manager

        return created
