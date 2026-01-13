"""
Server initialization - subsystem setup and configuration.

This module centralizes all the initialization logic for optional subsystems
like InsightStore, EloSystem, PersonaManager, etc.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from aragora.config import (
    DB_INSIGHTS_PATH,
    DB_PERSONAS_PATH,
)
from aragora.persistence.db_config import (
    DatabaseType,
    get_db_path,
)
from aragora.utils.optional_imports import try_import

logger = logging.getLogger(__name__)

# =============================================================================
# Optional Imports - Centralized
# =============================================================================

# Persistence
_imp, PERSISTENCE_AVAILABLE = try_import("aragora.persistence", "SupabaseClient")
SupabaseClient = _imp.get("SupabaseClient")

# InsightStore for debate insights
_imp, INSIGHTS_AVAILABLE = try_import("aragora.insights.store", "InsightStore")
InsightStore = _imp.get("InsightStore")

# EloSystem for agent rankings
_imp, RANKING_AVAILABLE = try_import("aragora.ranking.elo", "EloSystem")
EloSystem = _imp.get("EloSystem")

# FlipDetector for position reversal detection
_imp, FLIP_DETECTOR_AVAILABLE = try_import(
    "aragora.insights.flip_detector",
    "FlipDetector",
    "format_flip_for_ui",
    "format_consistency_for_ui",
)
FlipDetector = _imp.get("FlipDetector")
format_flip_for_ui = _imp.get("format_flip_for_ui")
format_consistency_for_ui = _imp.get("format_consistency_for_ui")

# Debate orchestrator for ad-hoc debates
_imp1, _avail1 = try_import("aragora.debate.orchestrator", "Arena", "DebateProtocol")
_imp2, _avail2 = try_import("aragora.agents.base", "create_agent")
_imp3, _avail3 = try_import("aragora.core", "Environment")
DEBATE_AVAILABLE = _avail1 and _avail2 and _avail3
Arena = _imp1.get("Arena")
DebateProtocol = _imp1.get("DebateProtocol")
create_agent = _imp2.get("create_agent")
Environment = _imp3.get("Environment")

# PersonaManager for agent specialization
_imp, PERSONAS_AVAILABLE = try_import("aragora.agents.personas", "PersonaManager")
PersonaManager = _imp.get("PersonaManager")

# DebateEmbeddingsDatabase for historical memory
_imp, EMBEDDINGS_AVAILABLE = try_import("aragora.debate.embeddings", "DebateEmbeddingsDatabase")
DebateEmbeddingsDatabase = _imp.get("DebateEmbeddingsDatabase")

# ConsensusMemory for historical consensus data
_imp, CONSENSUS_MEMORY_AVAILABLE = try_import(
    "aragora.memory.consensus", "ConsensusMemory", "DissentRetriever"
)
ConsensusMemory = _imp.get("ConsensusMemory")
DissentRetriever = _imp.get("DissentRetriever")

# CalibrationTracker for agent calibration
_imp, CALIBRATION_AVAILABLE = try_import("aragora.agents.calibration", "CalibrationTracker")
CalibrationTracker = _imp.get("CalibrationTracker")

# PulseManager for trending topics
_imp, PULSE_AVAILABLE = try_import(
    "aragora.pulse.ingestor", "PulseManager", "TrendingTopic", "TwitterIngestor"
)
PulseManager = _imp.get("PulseManager")
TrendingTopic = _imp.get("TrendingTopic")

# FormalVerificationManager for theorem proving
_imp, VERIFICATION_AVAILABLE = try_import(
    "aragora.verification.formal", "FormalVerificationManager"
)
FormalVerificationManager = _imp.get("FormalVerificationManager")

# ContinuumMemory for multi-tier memory
_imp, CONTINUUM_AVAILABLE = try_import("aragora.memory.continuum", "ContinuumMemory")
ContinuumMemory = _imp.get("ContinuumMemory")

# PositionLedger for truth-grounded personas
_imp, POSITION_LEDGER_AVAILABLE = try_import("aragora.genesis.ledger", "PositionLedger")
PositionLedger = _imp.get("PositionLedger")

# MomentDetector for significant agent moments
_imp, MOMENT_DETECTOR_AVAILABLE = try_import("aragora.insights.moments", "MomentDetector")
MomentDetector = _imp.get("MomentDetector")

# PositionTracker for agent positions
_imp, POSITION_TRACKER_AVAILABLE = try_import("aragora.insights.positions", "PositionTracker")
PositionTracker = _imp.get("PositionTracker")

# Broadcast module for podcast generation
_imp1, _avail1 = try_import("aragora.broadcast", "broadcast_debate")
_imp2, _avail2 = try_import("aragora.debate.traces", "DebateTrace")
BROADCAST_AVAILABLE = _avail1 and _avail2
broadcast_debate = _imp1.get("broadcast_debate")
DebateTrace = _imp2.get("DebateTrace")

# RelationshipTracker for agent network analysis
_imp, RELATIONSHIP_TRACKER_AVAILABLE = try_import("aragora.agents.grounded", "RelationshipTracker")
RelationshipTracker = _imp.get("RelationshipTracker")

# CritiqueStore for pattern retrieval
_imp, CRITIQUE_STORE_AVAILABLE = try_import("aragora.memory.store", "CritiqueStore")
CritiqueStore = _imp.get("CritiqueStore")

# Export module for debate artifact export
_imp, EXPORT_AVAILABLE = try_import(
    "aragora.export", "DebateArtifact", "CSVExporter", "DOTExporter", "StaticHTMLExporter"
)
DebateArtifact = _imp.get("DebateArtifact")
CSVExporter = _imp.get("CSVExporter")
DOTExporter = _imp.get("DOTExporter")
StaticHTMLExporter = _imp.get("StaticHTMLExporter")

# CapabilityProber for vulnerability detection
_imp, PROBER_AVAILABLE = try_import("aragora.modes.prober", "CapabilityProber")
CapabilityProber = _imp.get("CapabilityProber")

# RedTeamMode for adversarial testing
_imp, REDTEAM_AVAILABLE = try_import("aragora.modes.redteam", "RedTeamMode")
RedTeamMode = _imp.get("RedTeamMode")

# PersonaLaboratory for emergent traits
_imp, LABORATORY_AVAILABLE = try_import("aragora.agents.laboratory", "PersonaLaboratory")
PersonaLaboratory = _imp.get("PersonaLaboratory")

# BeliefNetwork for debate cruxes
_imp, BELIEF_NETWORK_AVAILABLE = try_import(
    "aragora.reasoning.belief", "BeliefNetwork", "BeliefPropagationAnalyzer"
)
BeliefNetwork = _imp.get("BeliefNetwork")
BeliefPropagationAnalyzer = _imp.get("BeliefPropagationAnalyzer")

# ProvenanceTracker for claim support
_imp, PROVENANCE_AVAILABLE = try_import("aragora.reasoning.provenance", "ProvenanceTracker")
ProvenanceTracker = _imp.get("ProvenanceTracker")

# FormalVerificationManager singleton accessor
_imp, FORMAL_VERIFICATION_AVAILABLE = try_import(
    "aragora.verification.formal", "FormalVerificationManager", "get_formal_verification_manager"
)
FormalVerificationManager = _imp.get("FormalVerificationManager")
get_formal_verification_manager = _imp.get("get_formal_verification_manager")

# ImpasseDetector for debate deadlock detection
_imp, IMPASSE_DETECTOR_AVAILABLE = try_import("aragora.debate.counterfactual", "ImpasseDetector")
ImpasseDetector = _imp.get("ImpasseDetector")

# ConvergenceDetector for semantic position convergence
_imp, CONVERGENCE_DETECTOR_AVAILABLE = try_import(
    "aragora.debate.convergence", "ConvergenceDetector"
)
ConvergenceDetector = _imp.get("ConvergenceDetector")

# AgentSelector for routing recommendations and auto team selection
_imp, ROUTING_AVAILABLE = try_import(
    "aragora.routing.selection", "AgentSelector", "AgentProfile", "TaskRequirements"
)
AgentSelector = _imp.get("AgentSelector")
AgentProfile = _imp.get("AgentProfile")
TaskRequirements = _imp.get("TaskRequirements")

# TournamentManager for tournament standings
_imp, TOURNAMENT_AVAILABLE = try_import("aragora.tournaments.tournament", "TournamentManager")
TournamentManager = _imp.get("TournamentManager")

# PromptEvolver for evolution history
_imp, EVOLUTION_AVAILABLE = try_import("aragora.evolution.evolver", "PromptEvolver")
PromptEvolver = _imp.get("PromptEvolver")

# MemoryTier enum for ContinuumMemory
_imp, _mem_tier_avail = try_import("aragora.memory.continuum", "MemoryTier")
MemoryTier = _imp.get("MemoryTier")

# InsightExtractor for debate insights
_imp, INSIGHT_EXTRACTOR_AVAILABLE = try_import("aragora.insights.extractor", "InsightExtractor")
InsightExtractor = _imp.get("InsightExtractor")


# =============================================================================
# Subsystem Initialization Functions
# =============================================================================


def init_persistence(enable: bool = True) -> Optional[Any]:
    """Initialize Supabase persistence if available and enabled."""
    if not enable or not PERSISTENCE_AVAILABLE or not SupabaseClient:
        return None

    client = SupabaseClient()
    if client.is_configured:
        logger.info("[init] Supabase persistence enabled")
        return client

    return None


def init_insight_store(nomic_dir: Path) -> Optional[Any]:
    """Initialize InsightStore for debate insights."""
    if not INSIGHTS_AVAILABLE or not InsightStore:
        return None

    insights_path = nomic_dir / DB_INSIGHTS_PATH
    if insights_path.exists():
        store = InsightStore(str(insights_path))
        logger.info("[init] InsightStore loaded for API access")
        return store

    return None


def init_elo_system(nomic_dir: Path) -> Optional[Any]:
    """Initialize EloSystem for agent rankings."""
    if not RANKING_AVAILABLE or not EloSystem:
        return None

    elo_path = get_db_path(DatabaseType.ELO, nomic_dir)
    if elo_path.exists():
        system = EloSystem(str(elo_path))
        logger.info("[init] EloSystem loaded for leaderboard API")
        return system

    return None


def init_flip_detector(nomic_dir: Path) -> Optional[Any]:
    """Initialize FlipDetector for position reversal detection."""
    if not FLIP_DETECTOR_AVAILABLE or not FlipDetector:
        return None

    positions_path = get_db_path(DatabaseType.POSITIONS, nomic_dir)
    if positions_path.exists():
        detector = FlipDetector(str(positions_path))
        logger.info("[init] FlipDetector loaded for position reversal API")
        return detector

    return None


def init_persona_manager(nomic_dir: Path) -> Optional[Any]:
    """Initialize PersonaManager for agent specialization."""
    if not PERSONAS_AVAILABLE or not PersonaManager:
        return None

    personas_path = get_db_path(DatabaseType.PERSONAS, nomic_dir)
    manager = PersonaManager(str(personas_path))
    logger.info("[init] PersonaManager loaded for agent specialization")
    return manager


def init_position_ledger(nomic_dir: Path) -> Optional[Any]:
    """Initialize PositionLedger for truth-grounded personas."""
    if not POSITION_LEDGER_AVAILABLE or not PositionLedger:
        return None

    ledger_path = get_db_path(DatabaseType.TRUTH_GROUNDING, nomic_dir)
    try:
        ledger = PositionLedger(db_path=str(ledger_path))
        logger.info("[init] PositionLedger loaded for truth-grounded personas")
        return ledger
    except Exception as e:
        logger.warning(f"[init] PositionLedger initialization failed: {e}")
        return None


def init_debate_embeddings(nomic_dir: Path) -> Optional[Any]:
    """Initialize DebateEmbeddingsDatabase for historical memory."""
    if not EMBEDDINGS_AVAILABLE or not DebateEmbeddingsDatabase:
        return None

    embeddings_path = get_db_path(DatabaseType.EMBEDDINGS, nomic_dir)
    try:
        db = DebateEmbeddingsDatabase(str(embeddings_path))
        logger.info("[init] DebateEmbeddings loaded for historical memory")
        return db
    except Exception as e:
        logger.warning(f"[init] DebateEmbeddings initialization failed: {e}")
        return None


def init_consensus_memory() -> tuple[Optional[Any], Optional[Any]]:
    """Initialize ConsensusMemory and DissentRetriever."""
    if not CONSENSUS_MEMORY_AVAILABLE or not ConsensusMemory or not DissentRetriever:
        return None, None

    try:
        memory = ConsensusMemory()
        retriever = DissentRetriever(memory)
        logger.info("[init] DissentRetriever loaded for historical minority views")
        return memory, retriever
    except Exception as e:
        logger.warning(f"[init] DissentRetriever initialization failed: {e}")
        return None, None


def init_moment_detector(
    elo_system: Optional[Any] = None,
    position_ledger: Optional[Any] = None,
) -> Optional[Any]:
    """Initialize MomentDetector for significant agent moments."""
    if not MOMENT_DETECTOR_AVAILABLE or not MomentDetector:
        return None

    try:
        detector = MomentDetector(
            elo_system=elo_system,
            position_ledger=position_ledger,
        )
        logger.info("[init] MomentDetector loaded for agent moments API")
        return detector
    except Exception as e:
        logger.warning(f"[init] MomentDetector initialization failed: {e}")
        return None


def init_position_tracker(nomic_dir: Path) -> Optional[Any]:
    """Initialize PositionTracker for agent positions."""
    if not POSITION_TRACKER_AVAILABLE or not PositionTracker:
        return None

    positions_path = nomic_dir / "aragora_positions.db"
    try:
        tracker = PositionTracker(str(positions_path))
        logger.info("[init] PositionTracker loaded for agent positions")
        return tracker
    except Exception as e:
        logger.warning(f"[init] PositionTracker initialization failed: {e}")
        return None


def init_continuum_memory(nomic_dir: Path) -> Optional[Any]:
    """Initialize ContinuumMemory for multi-tier memory."""
    if not CONTINUUM_AVAILABLE or not ContinuumMemory:
        return None

    try:
        memory = ContinuumMemory(base_dir=str(nomic_dir))
        logger.info("[init] ContinuumMemory loaded for multi-tier memory")
        return memory
    except Exception as e:
        logger.warning(f"[init] ContinuumMemory initialization failed: {e}")
        return None


def init_verification_manager() -> Optional[Any]:
    """Initialize FormalVerificationManager for theorem proving."""
    if not VERIFICATION_AVAILABLE or not FormalVerificationManager:
        return None

    try:
        manager = FormalVerificationManager()
        logger.info("[init] FormalVerificationManager loaded")
        return manager
    except Exception as e:
        logger.warning(f"[init] FormalVerificationManager initialization failed: {e}")
        return None


# =============================================================================
# Batch Initialization
# =============================================================================


class SubsystemRegistry:
    """
    Registry of initialized subsystems.

    Provides a single point of access for all optional subsystems.
    """

    def __init__(self):
        self.persistence = None
        self.insight_store = None
        self.elo_system = None
        self.flip_detector = None
        self.persona_manager = None
        self.position_ledger = None
        self.debate_embeddings = None
        self.consensus_memory = None
        self.dissent_retriever = None
        self.moment_detector = None
        self.position_tracker = None
        self.continuum_memory = None
        self.verification_manager = None

    def initialize_all(
        self,
        nomic_dir: Optional[Path] = None,
        enable_persistence: bool = True,
    ) -> "SubsystemRegistry":
        """
        Initialize all available subsystems.

        Args:
            nomic_dir: Path to nomic state directory
            enable_persistence: Whether to enable Supabase persistence

        Returns:
            Self for chaining
        """
        # Persistence (no nomic_dir required)
        self.persistence = init_persistence(enable_persistence)

        if nomic_dir:
            # Core subsystems
            self.insight_store = init_insight_store(nomic_dir)
            self.elo_system = init_elo_system(nomic_dir)
            self.flip_detector = init_flip_detector(nomic_dir)
            self.persona_manager = init_persona_manager(nomic_dir)
            self.position_ledger = init_position_ledger(nomic_dir)
            self.debate_embeddings = init_debate_embeddings(nomic_dir)
            self.position_tracker = init_position_tracker(nomic_dir)
            self.continuum_memory = init_continuum_memory(nomic_dir)

            # Memory subsystems
            self.consensus_memory, self.dissent_retriever = init_consensus_memory()

            # Dependent subsystems (need other subsystems initialized first)
            self.moment_detector = init_moment_detector(
                elo_system=self.elo_system,
                position_ledger=self.position_ledger,
            )

        # Standalone subsystems
        self.verification_manager = init_verification_manager()

        return self

    async def initialize_all_async(
        self,
        nomic_dir: Optional[Path] = None,
        enable_persistence: bool = True,
    ) -> "SubsystemRegistry":
        """
        Initialize all available subsystems in parallel where possible.

        This is the async version that parallelizes independent initializations
        using a thread pool executor, significantly reducing startup time.

        Args:
            nomic_dir: Path to nomic state directory
            enable_persistence: Whether to enable Supabase persistence

        Returns:
            Self for chaining
        """
        loop = asyncio.get_running_loop()

        # Use a thread pool for I/O-bound initialization
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Persistence (no nomic_dir required, run first)
            self.persistence = await loop.run_in_executor(
                executor, init_persistence, enable_persistence
            )

            if nomic_dir:
                # Phase 1: Initialize independent subsystems in parallel
                # These don't depend on each other
                independent_futures = [
                    loop.run_in_executor(executor, init_insight_store, nomic_dir),
                    loop.run_in_executor(executor, init_elo_system, nomic_dir),
                    loop.run_in_executor(executor, init_flip_detector, nomic_dir),
                    loop.run_in_executor(executor, init_persona_manager, nomic_dir),
                    loop.run_in_executor(executor, init_position_ledger, nomic_dir),
                    loop.run_in_executor(executor, init_debate_embeddings, nomic_dir),
                    loop.run_in_executor(executor, init_position_tracker, nomic_dir),
                    loop.run_in_executor(executor, init_continuum_memory, nomic_dir),
                    loop.run_in_executor(executor, init_consensus_memory),
                ]

                results = await asyncio.gather(*independent_futures)

                # Assign results
                (
                    self.insight_store,
                    self.elo_system,
                    self.flip_detector,
                    self.persona_manager,
                    self.position_ledger,
                    self.debate_embeddings,
                    self.position_tracker,
                    self.continuum_memory,
                    consensus_result,
                ) = results

                # Unpack consensus memory result
                self.consensus_memory, self.dissent_retriever = consensus_result

                # Phase 2: Initialize dependent subsystems
                # MomentDetector depends on elo_system and position_ledger
                self.moment_detector = await loop.run_in_executor(
                    executor,
                    lambda: init_moment_detector(
                        elo_system=self.elo_system,
                        position_ledger=self.position_ledger,
                    ),
                )

            # Standalone subsystems (can run in parallel with nothing)
            self.verification_manager = await loop.run_in_executor(
                executor, init_verification_manager
            )

        logger.info("[init] Async initialization completed")
        return self

    def log_availability(self) -> None:
        """Log which subsystems are available."""
        available = []
        unavailable = []

        checks = [
            ("Persistence", self.persistence),
            ("InsightStore", self.insight_store),
            ("EloSystem", self.elo_system),
            ("FlipDetector", self.flip_detector),
            ("PersonaManager", self.persona_manager),
            ("PositionLedger", self.position_ledger),
            ("DebateEmbeddings", self.debate_embeddings),
            ("ConsensusMemory", self.consensus_memory),
            ("DissentRetriever", self.dissent_retriever),
            ("MomentDetector", self.moment_detector),
            ("PositionTracker", self.position_tracker),
            ("ContinuumMemory", self.continuum_memory),
            ("VerificationManager", self.verification_manager),
        ]

        for name, instance in checks:
            if instance is not None:
                available.append(name)
            else:
                unavailable.append(name)

        if available:
            logger.info(f"[init] Available: {', '.join(available)}")
        if unavailable:
            logger.debug(f"[init] Unavailable: {', '.join(unavailable)}")


# Global registry instance
_registry: Optional[SubsystemRegistry] = None


def get_registry() -> SubsystemRegistry:
    """Get or create the global subsystem registry."""
    global _registry
    if _registry is None:
        _registry = SubsystemRegistry()
    return _registry


def initialize_subsystems(
    nomic_dir: Optional[Path] = None,
    enable_persistence: bool = True,
) -> SubsystemRegistry:
    """
    Initialize all subsystems and return the registry.

    This is the main entry point for server initialization (synchronous version).
    """
    registry = get_registry()
    registry.initialize_all(nomic_dir, enable_persistence)
    registry.log_availability()
    return registry


async def initialize_subsystems_async(
    nomic_dir: Optional[Path] = None,
    enable_persistence: bool = True,
) -> SubsystemRegistry:
    """
    Initialize all subsystems asynchronously and return the registry.

    This is the async entry point for server initialization, parallelizing
    independent subsystem initializations for faster startup.
    """
    registry = get_registry()
    await registry.initialize_all_async(nomic_dir, enable_persistence)
    registry.log_availability()
    return registry


# =============================================================================
# Cache Pre-Warming
# =============================================================================


async def prewarm_caches(
    registry: Optional[SubsystemRegistry] = None,
    nomic_dir: Optional[Path] = None,
) -> dict:
    """
    Pre-warm caches with commonly accessed data.

    Call during server startup after subsystem initialization to reduce
    cold-start latency for the first requests.

    Pre-warms:
    - Top 20 leaderboard entries
    - Top 10 agent profiles
    - Consensus stats summary

    Args:
        registry: Optional SubsystemRegistry with initialized subsystems
        nomic_dir: Path to nomic directory for database access

    Returns:
        Dictionary with counts of pre-warmed entries
    """
    result = {
        "leaderboard_entries": 0,
        "agent_profiles": 0,
        "consensus_stats": False,
    }

    if registry is None:
        registry = get_registry()

    loop = asyncio.get_running_loop()

    # Pre-warm leaderboard cache
    if registry.elo_system is not None:
        try:
            # Import cache and populate
            from aragora.utils.cache import get_method_cache

            cache = get_method_cache()

            # Fetch top 20 leaderboard entries
            def _fetch_leaderboard():
                leaderboard = registry.elo_system.get_leaderboard(limit=20)
                # Cache each entry
                for entry in leaderboard:
                    agent_name = entry.get("agent", entry.get("name", ""))
                    if agent_name:
                        cache.set(f"leaderboard:agent:{agent_name}", entry)
                # Cache the full list
                cache.set("leaderboard:top20", leaderboard)
                return len(leaderboard)

            result["leaderboard_entries"] = await loop.run_in_executor(None, _fetch_leaderboard)
            logger.debug(f"[prewarm] Cached {result['leaderboard_entries']} leaderboard entries")

        except Exception as e:
            logger.debug(f"[prewarm] Leaderboard cache failed: {e}")

    # Pre-warm agent profiles
    if registry.persona_manager is not None:
        try:
            from aragora.utils.cache import get_method_cache

            cache = get_method_cache()

            def _fetch_profiles():
                # Get top agents by activity/score
                profiles = registry.persona_manager.get_all_profiles(limit=10)
                for profile in profiles:
                    agent_name = profile.get("name", "")
                    if agent_name:
                        cache.set(f"profile:{agent_name}", profile)
                return len(profiles)

            result["agent_profiles"] = await loop.run_in_executor(None, _fetch_profiles)
            logger.debug(f"[prewarm] Cached {result['agent_profiles']} agent profiles")

        except Exception as e:
            logger.debug(f"[prewarm] Agent profile cache failed: {e}")

    # Pre-warm consensus stats
    if registry.consensus_memory is not None:
        try:
            from aragora.utils.cache import get_method_cache

            cache = get_method_cache()

            def _fetch_consensus_stats():
                stats = registry.consensus_memory.get_summary_stats()
                cache.set("consensus:summary_stats", stats)
                return True

            result["consensus_stats"] = await loop.run_in_executor(None, _fetch_consensus_stats)
            logger.debug("[prewarm] Cached consensus stats")

        except Exception as e:
            logger.debug(f"[prewarm] Consensus stats cache failed: {e}")

    total = (
        result["leaderboard_entries"]
        + result["agent_profiles"]
        + (1 if result["consensus_stats"] else 0)
    )
    if total > 0:
        logger.info(f"[prewarm] Pre-warmed {total} cache entries")

    return result
