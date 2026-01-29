"""
AdapterFactory - Auto-create and register KM adapters from Arena subsystems.

This factory enables automatic adapter injection in Arena by creating
appropriate adapters based on available subsystems.

Usage:
    from aragora.knowledge.mound.adapters.factory import AdapterFactory

    # Create adapters from Arena subsystems
    factory = AdapterFactory()
    adapters = factory.create_from_config(
        elo_system=arena.elo_system,
        continuum_memory=arena.continuum_memory,
        evidence_store=arena.evidence_collector,
        insight_store=arena.insight_store,
    )

    # Register with coordinator
    factory.register_with_coordinator(coordinator, adapters)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

logger = logging.getLogger(__name__)

@dataclass
class AdapterSpec:
    """Specification for an adapter."""

    name: str
    adapter_class: type
    required_deps: list[str]
    forward_method: str = "sync_to_km"
    reverse_method: str | None = "sync_from_km"
    priority: int = 0
    enabled_by_default: bool = True
    config_key: str | None = None  # Key in ArenaConfig to check for explicit adapter

# Registry of available adapter specifications
ADAPTER_SPECS: dict[str, AdapterSpec] = {}

def register_adapter_spec(spec: AdapterSpec) -> None:
    """Register an adapter specification."""
    ADAPTER_SPECS[spec.name] = spec

# Import and register adapter specs
def _init_specs() -> None:
    """Initialize adapter specifications."""
    from .continuum_adapter import ContinuumAdapter
    from .consensus_adapter import ConsensusAdapter
    from .critique_adapter import CritiqueAdapter
    from .evidence_adapter import EvidenceAdapter
    from .belief_adapter import BeliefAdapter
    from .insights_adapter import InsightsAdapter
    from .performance_adapter import PerformanceAdapter, EloAdapter
    from .pulse_adapter import PulseAdapter
    from .cost_adapter import CostAdapter
    from .provenance_adapter import ProvenanceAdapter

    # Core memory adapters
    register_adapter_spec(
        AdapterSpec(
            name="continuum",
            adapter_class=ContinuumAdapter,
            required_deps=["continuum_memory"],
            forward_method="store",
            reverse_method="sync_validations_to_continuum",
            priority=100,  # High priority - core memory
            config_key="km_continuum_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="consensus",
            adapter_class=ConsensusAdapter,
            required_deps=["consensus_memory"],
            forward_method="sync_to_km",  # Sync pending consensus records to KM
            reverse_method="sync_validations_from_km",  # KM validations update consensus metadata
            priority=90,
            config_key="km_consensus_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="critique",
            adapter_class=CritiqueAdapter,
            required_deps=["memory"],  # CritiqueStore is called "memory" in Arena
            forward_method="store",
            reverse_method="sync_validations_from_km",
            priority=80,
            config_key="km_critique_adapter",
        )
    )

    # Bidirectional integration adapters
    register_adapter_spec(
        AdapterSpec(
            name="evidence",
            adapter_class=EvidenceAdapter,
            required_deps=["evidence_store"],
            forward_method="store",
            reverse_method="update_reliability_from_km",
            priority=70,
            config_key="km_evidence_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="belief",
            adapter_class=BeliefAdapter,
            required_deps=[],  # No external dep required, uses internal storage
            forward_method="store_converged_belief",
            reverse_method="sync_validations_from_km",
            priority=60,
            config_key="km_belief_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="insights",
            adapter_class=InsightsAdapter,
            required_deps=["insight_store"],
            forward_method="store_insight",
            reverse_method="sync_validations_from_km",
            priority=50,
            config_key="km_insights_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="elo",
            adapter_class=EloAdapter,
            required_deps=["elo_system"],
            forward_method="store_match",
            reverse_method="sync_km_to_elo",
            priority=40,
            config_key="km_elo_bridge",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="performance",
            adapter_class=PerformanceAdapter,
            required_deps=["elo_system"],
            forward_method="store_match",
            reverse_method="sync_km_to_elo",
            priority=45,
            enabled_by_default=False,  # Avoid double-writes with EloAdapter unless explicitly enabled
            config_key="km_performance_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="pulse",
            adapter_class=PulseAdapter,
            required_deps=["pulse_manager"],
            forward_method="store_trending_topic",
            reverse_method="sync_validations_from_km",
            priority=30,
            config_key="km_pulse_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="cost",
            adapter_class=CostAdapter,
            required_deps=["cost_tracker"],
            forward_method="store_anomaly",
            reverse_method="sync_validations_from_km",
            priority=10,  # Low priority - operational
            enabled_by_default=False,  # Opt-in
            config_key="km_cost_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="provenance",
            adapter_class=ProvenanceAdapter,
            required_deps=["provenance_store"],
            forward_method="ingest_provenance",
            reverse_method=None,  # No reverse sync (provenance is append-only)
            priority=75,  # High priority - core evidence tracking
            config_key="km_provenance_adapter",
        )
    )

    # Extension module adapters
    from .fabric_adapter import FabricAdapter
    from .workspace_adapter import WorkspaceAdapter

    register_adapter_spec(
        AdapterSpec(
            name="fabric",
            adapter_class=FabricAdapter,
            required_deps=["fabric"],  # AgentFabric instance
            forward_method="sync_from_fabric",
            reverse_method="get_pool_recommendations",
            priority=35,  # Medium priority - extension module
            config_key="km_fabric_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="workspace",
            adapter_class=WorkspaceAdapter,
            required_deps=["workspace_manager"],  # WorkspaceManager instance
            forward_method="sync_from_workspace",
            reverse_method="get_rig_recommendations",
            priority=34,  # Medium priority - extension module
            config_key="km_workspace_adapter",
        )
    )

    from .computer_use_adapter import ComputerUseAdapter
    from .gateway_adapter import GatewayAdapter
    from .calibration_fusion_adapter import CalibrationFusionAdapter
    from .control_plane_adapter import ControlPlaneAdapter
    from .culture_adapter import CultureAdapter
    from .receipt_adapter import ReceiptAdapter
    from .rlm_adapter import RlmAdapter

    register_adapter_spec(
        AdapterSpec(
            name="computer_use",
            adapter_class=ComputerUseAdapter,
            required_deps=["computer_use_orchestrator"],  # ComputerUseOrchestrator instance
            forward_method="sync_from_orchestrator",
            reverse_method="get_similar_tasks",
            priority=33,  # Medium priority - extension module
            config_key="km_computer_use_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="gateway",
            adapter_class=GatewayAdapter,
            required_deps=["gateway"],  # LocalGateway instance
            forward_method="sync_from_gateway",
            reverse_method="get_routing_recommendations",
            priority=32,  # Medium priority - extension module
            config_key="km_gateway_adapter",
        )
    )

    # Additional integration adapters
    register_adapter_spec(
        AdapterSpec(
            name="calibration_fusion",
            adapter_class=CalibrationFusionAdapter,
            required_deps=[],  # Uses KM singleton internally
            forward_method="sync_from_calibration",
            reverse_method="get_calibration_insights",
            priority=55,  # High priority - calibration data
            config_key="km_calibration_fusion_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="control_plane",
            adapter_class=ControlPlaneAdapter,
            required_deps=["control_plane"],  # ControlPlane coordinator instance
            forward_method="sync_from_control_plane",
            reverse_method="get_policy_recommendations",
            priority=85,  # High priority - control plane integration
            config_key="km_control_plane_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="culture",
            adapter_class=CultureAdapter,
            required_deps=[],  # Uses KM singleton internally
            forward_method="sync_to_mound",
            reverse_method="load_from_mound",
            priority=25,  # Lower priority - derived patterns
            config_key="km_culture_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="receipt",
            adapter_class=ReceiptAdapter,
            required_deps=[],  # KM mound set via set_mound()
            forward_method="ingest_receipt",
            reverse_method="find_related_decisions",
            priority=65,  # Medium-high priority - decision audit trail
            config_key="km_receipt_adapter",
        )
    )

    register_adapter_spec(
        AdapterSpec(
            name="rlm",
            adapter_class=RlmAdapter,
            required_deps=["compressor"],  # RLM Compressor instance
            forward_method="sync_to_mound",
            reverse_method="load_from_mound",
            priority=20,  # Lower priority - compression optimization
            config_key="km_rlm_adapter",
        )
    )

# Initialize specs on import
_init_specs()

@dataclass
class CreatedAdapter:
    """A created adapter with metadata."""

    name: str
    adapter: Any
    spec: AdapterSpec
    deps_used: dict[str, Any] = field(default_factory=dict)

class AdapterFactory:
    """
    Factory for creating and registering KM adapters.

    This factory automatically creates adapters based on available
    subsystem dependencies and can register them with a coordinator.

    Example:
        factory = AdapterFactory()

        # Create adapters from subsystems
        adapters = factory.create_from_subsystems(
            continuum_memory=my_continuum,
            elo_system=my_elo,
        )

        # Or from ArenaConfig
        adapters = factory.create_from_arena_config(config)

        # Register with coordinator
        factory.register_with_coordinator(coordinator, adapters)
    """

    def __init__(
        self,
        event_callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ):
        """
        Initialize the factory.

        Args:
            event_callback: Optional callback for WebSocket events
        """
        self._event_callback = event_callback

    def set_event_callback(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Set the event callback for all created adapters."""
        self._event_callback = callback

    def create_from_subsystems(
        self,
        continuum_memory: Any | None = None,
        consensus_memory: Any | None = None,
        memory: Any | None = None,  # CritiqueStore
        evidence_store: Any | None = None,
        insight_store: Any | None = None,
        elo_system: Any | None = None,
        pulse_manager: Any | None = None,
        cost_tracker: Any | None = None,
        flip_detector: Any | None = None,
        provenance_store: Any | None = None,
        **kwargs,
    ) -> dict[str, CreatedAdapter]:
        """
        Create adapters from provided subsystems.

        Args:
            continuum_memory: ContinuumMemory instance
            consensus_memory: ConsensusMemory instance
            memory: CritiqueStore instance
            evidence_store: EvidenceStore/Collector instance
            insight_store: InsightStore instance
            elo_system: EloSystem instance
            pulse_manager: PulseManager instance
            cost_tracker: CostTracker instance
            flip_detector: FlipDetector instance (for insights)
            provenance_store: ProvenanceStore instance
            **kwargs: Additional dependencies

        Returns:
            Dict of adapter name -> CreatedAdapter
        """
        # Collect all available dependencies
        deps = {
            "continuum_memory": continuum_memory,
            "consensus_memory": consensus_memory,
            "memory": memory,
            "evidence_store": evidence_store,
            "insight_store": insight_store,
            "elo_system": elo_system,
            "pulse_manager": pulse_manager,
            "cost_tracker": cost_tracker,
            "flip_detector": flip_detector,
            "provenance_store": provenance_store,
        }
        deps.update(kwargs)

        # Filter to non-None deps
        available_deps = {k: v for k, v in deps.items() if v is not None}

        return self._create_adapters(available_deps)

    def create_from_arena_config(
        self,
        config: Any,  # ArenaConfig
        subsystems: Optional[dict[str, Any]] = None,
    ) -> dict[str, CreatedAdapter]:
        """
        Create adapters from ArenaConfig.

        Checks for explicitly configured adapters first, then
        falls back to auto-creation from subsystems.

        Args:
            config: ArenaConfig instance
            subsystems: Optional dict of subsystem instances (e.g., from Arena)

        Returns:
            Dict of adapter name -> CreatedAdapter
        """
        adapters = {}
        subsystems = subsystems or {}

        # Collect dependencies from config and subsystems
        deps = {
            "continuum_memory": getattr(config, "continuum_memory", None)
            or subsystems.get("continuum_memory"),
            "consensus_memory": getattr(config, "consensus_memory", None)
            or subsystems.get("consensus_memory"),
            "memory": getattr(config, "memory", None) or subsystems.get("memory"),
            "evidence_store": subsystems.get("evidence_store")
            or subsystems.get("evidence_collector"),
            "insight_store": getattr(config, "insight_store", None)
            or subsystems.get("insight_store"),
            "elo_system": getattr(config, "elo_system", None) or subsystems.get("elo_system"),
            "pulse_manager": getattr(config, "pulse_manager", None)
            or subsystems.get("pulse_manager"),
            "cost_tracker": getattr(config, "usage_tracker", None)
            or subsystems.get("cost_tracker"),
            "flip_detector": getattr(config, "flip_detector", None)
            or subsystems.get("flip_detector"),
        }

        # Check for explicitly configured adapters
        for spec_name, spec in ADAPTER_SPECS.items():
            if spec.config_key:
                explicit_adapter = getattr(config, spec.config_key, None)
                if explicit_adapter is not None:
                    # Use the explicitly configured adapter
                    adapters[spec_name] = CreatedAdapter(
                        name=spec_name,
                        adapter=explicit_adapter,
                        spec=spec,
                        deps_used={},
                    )
                    logger.debug(f"Using explicit adapter: {spec_name}")

        # Auto-create remaining adapters
        available_deps = {k: v for k, v in deps.items() if v is not None}
        auto_adapters = self._create_adapters(available_deps, exclude=set(adapters.keys()))

        adapters.update(auto_adapters)
        return adapters

    def _create_adapters(
        self,
        deps: dict[str, Any],
        exclude: set | None = None,
    ) -> dict[str, CreatedAdapter]:
        """
        Create adapters based on available dependencies.

        Args:
            deps: Available dependencies
            exclude: Adapter names to skip

        Returns:
            Dict of adapter name -> CreatedAdapter
        """
        exclude = exclude or set()
        adapters = {}

        for spec_name, spec in ADAPTER_SPECS.items():
            if spec_name in exclude:
                continue

            # Check if all required deps are available
            # (Empty required_deps means the adapter can work standalone)
            missing_deps = [d for d in spec.required_deps if d not in deps]
            if missing_deps:
                logger.debug(f"Skipping adapter '{spec_name}': missing deps {missing_deps}")
                continue

            try:
                # Create the adapter
                adapter_deps = {d: deps[d] for d in spec.required_deps if d in deps}
                adapter = self._create_single_adapter(spec, adapter_deps)

                if adapter:
                    adapters[spec_name] = CreatedAdapter(
                        name=spec_name,
                        adapter=adapter,
                        spec=spec,
                        deps_used=adapter_deps,
                    )
                    logger.info(f"Created adapter: {spec_name}")

            except Exception as e:
                logger.warning(f"Failed to create adapter '{spec_name}': {e}")

        return adapters

    def _create_single_adapter(
        self,
        spec: AdapterSpec,
        deps: dict[str, Any],
    ) -> Any | None:
        """Create a single adapter from spec."""
        adapter_class = spec.adapter_class

        try:
            # Different adapters have different constructor signatures
            # We try to be smart about what to pass

            if spec.name == "continuum":
                adapter = adapter_class(
                    continuum=deps.get("continuum_memory"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "consensus":
                adapter = adapter_class(
                    consensus_memory=deps.get("consensus_memory"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "critique":
                adapter = adapter_class(
                    store=deps.get("memory"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "evidence":
                adapter = adapter_class(
                    store=deps.get("evidence_store"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "belief":
                adapter = adapter_class(
                    event_callback=self._event_callback,
                )
            elif spec.name == "insights":
                adapter = adapter_class(
                    store=deps.get("insight_store"),
                    flip_detector=deps.get("flip_detector"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "elo":
                adapter = adapter_class(
                    elo_system=deps.get("elo_system"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "pulse":
                adapter = adapter_class(
                    manager=deps.get("pulse_manager"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "cost":
                adapter = adapter_class(
                    cost_tracker=deps.get("cost_tracker"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "fabric":
                adapter = adapter_class(
                    fabric=deps.get("fabric"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "workspace":
                adapter = adapter_class(
                    workspace_manager=deps.get("workspace_manager"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "computer_use":
                adapter = adapter_class(
                    orchestrator=deps.get("computer_use_orchestrator"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "gateway":
                adapter = adapter_class(
                    gateway=deps.get("gateway"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "calibration_fusion":
                adapter = adapter_class(
                    mound=deps.get("mound"),
                )
            elif spec.name == "control_plane":
                adapter = adapter_class(
                    control_plane=deps.get("control_plane"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "culture":
                adapter = adapter_class(
                    mound=deps.get("mound"),
                )
            elif spec.name == "receipt":
                adapter = adapter_class(
                    mound=deps.get("mound"),
                    event_callback=self._event_callback,
                )
            elif spec.name == "rlm":
                adapter = adapter_class(
                    compressor=deps.get("compressor"),
                )
            else:
                # Generic construction attempt
                adapter = adapter_class(
                    event_callback=self._event_callback,
                    **deps,
                )

            return adapter

        except TypeError as e:
            logger.warning(f"Constructor mismatch for {spec.name}: {e}")
            # Try without event_callback
            try:
                if spec.name == "continuum":
                    return adapter_class(continuum=deps.get("continuum_memory"))
                elif spec.name == "consensus":
                    return adapter_class(consensus_memory=deps.get("consensus_memory"))
                elif spec.name == "critique":
                    return adapter_class(store=deps.get("memory"))
                elif spec.name == "evidence":
                    return adapter_class(store=deps.get("evidence_store"))
                elif spec.name == "belief":
                    return adapter_class()
                elif spec.name == "insights":
                    return adapter_class(
                        store=deps.get("insight_store"), flip_detector=deps.get("flip_detector")
                    )
                elif spec.name == "elo":
                    return adapter_class(elo_system=deps.get("elo_system"))
                elif spec.name == "pulse":
                    return adapter_class(manager=deps.get("pulse_manager"))
                elif spec.name == "cost":
                    return adapter_class(cost_tracker=deps.get("cost_tracker"))
                elif spec.name == "fabric":
                    return adapter_class(fabric=deps.get("fabric"))
                elif spec.name == "workspace":
                    return adapter_class(workspace_manager=deps.get("workspace_manager"))
                elif spec.name == "computer_use":
                    return adapter_class(orchestrator=deps.get("computer_use_orchestrator"))
                elif spec.name == "gateway":
                    return adapter_class(gateway=deps.get("gateway"))
                elif spec.name == "calibration_fusion":
                    return adapter_class(mound=deps.get("mound"))
                elif spec.name == "control_plane":
                    return adapter_class(control_plane=deps.get("control_plane"))
                elif spec.name == "culture":
                    return adapter_class(mound=deps.get("mound"))
                elif spec.name == "receipt":
                    return adapter_class(mound=deps.get("mound"))
                elif spec.name == "rlm":
                    return adapter_class(compressor=deps.get("compressor"))
                else:
                    return adapter_class(**deps)
            except Exception as e2:
                logger.error(f"Failed to create {spec.name} adapter: {e2}")
                return None

    def register_with_coordinator(
        self,
        coordinator: "BidirectionalCoordinator",
        adapters: dict[str, CreatedAdapter],
    ) -> int:
        """
        Register created adapters with a BidirectionalCoordinator.

        Args:
            coordinator: The coordinator to register with
            adapters: Dict of adapters from create_from_* methods

        Returns:
            Number of successfully registered adapters
        """
        registered = 0

        for name, created in adapters.items():
            spec = created.spec

            success = coordinator.register_adapter(
                name=name,
                adapter=created.adapter,
                forward_method=spec.forward_method,
                reverse_method=spec.reverse_method,
                priority=spec.priority,
                metadata={"deps": list(created.deps_used.keys())},
            )

            if success:
                # Enable/disable based on spec
                if not spec.enabled_by_default:
                    coordinator.disable_adapter(name)
                registered += 1

        logger.info(f"Registered {registered}/{len(adapters)} adapters with coordinator")
        return registered

    def get_available_adapter_specs(self) -> dict[str, AdapterSpec]:
        """Get all available adapter specifications."""
        return ADAPTER_SPECS.copy()

__all__ = [
    "AdapterFactory",
    "AdapterSpec",
    "CreatedAdapter",
    "ADAPTER_SPECS",
    "register_adapter_spec",
]
