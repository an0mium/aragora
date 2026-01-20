"""Feature dependency validation for debate configuration.

Ensures that enabled features have their required dependencies available
before Arena initialization to prevent runtime errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from aragora.debate.arena_config import ArenaConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureDependency:
    """Defines a feature and its requirements."""

    name: str
    description: str
    required_modules: list[str] = field(default_factory=list)
    required_config: list[str] = field(default_factory=list)
    check_fn: Optional[Callable[[], bool]] = None


# Feature dependency registry
FEATURE_DEPENDENCIES: dict[str, FeatureDependency] = {
    "formal_verification": FeatureDependency(
        name="formal_verification",
        description="Z3/Lean proof verification",
        required_modules=["z3"],
        check_fn=lambda: _check_module("z3"),
    ),
    "belief_guidance": FeatureDependency(
        name="belief_guidance",
        description="Historical crux injection",
        required_config=["dissent_retriever", "consensus_memory"],
    ),
    "knowledge_mound": FeatureDependency(
        name="knowledge_mound",
        description="Knowledge Mound integration",
        required_config=["knowledge_mound"],
    ),
    "rlm_compression": FeatureDependency(
        name="rlm_compression",
        description="RLM context compression",
        required_modules=["aragora.rlm.compressor"],
    ),
    "checkpointing": FeatureDependency(
        name="checkpointing",
        description="Debate checkpoint/resume",
        required_config=["checkpoint_manager"],
    ),
    "calibration": FeatureDependency(
        name="calibration",
        description="Prediction calibration tracking",
        required_config=["calibration_tracker"],
    ),
    "performance_monitoring": FeatureDependency(
        name="performance_monitoring",
        description="Agent performance tracking",
        required_config=["performance_monitor"],
    ),
    "population_evolution": FeatureDependency(
        name="population_evolution",
        description="Genesis persona evolution",
        required_config=["population_manager"],
    ),
}


def _check_module(module_name: str) -> bool:
    """Check if a module is available."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


@dataclass
class ValidationResult:
    """Result of feature validation."""

    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_feature_dependencies(config: "ArenaConfig") -> ValidationResult:
    """Validate that enabled features have their dependencies met.

    Args:
        config: ArenaConfig to validate

    Returns:
        ValidationResult with any warnings or errors
    """
    result = ValidationResult(valid=True)

    # Check belief guidance
    if getattr(config, "enable_belief_guidance", False):
        dep = FEATURE_DEPENDENCIES["belief_guidance"]
        if not config.dissent_retriever and not config.consensus_memory:
            result.warnings.append(
                f"[{dep.name}] Enabled but no dissent_retriever or consensus_memory configured. "
                "Historical crux injection will be skipped."
            )

    # Check knowledge mound
    if getattr(config, "enable_knowledge_retrieval", False) or getattr(
        config, "enable_knowledge_ingestion", False
    ):
        dep = FEATURE_DEPENDENCIES["knowledge_mound"]
        if not config.knowledge_mound:
            result.warnings.append(
                f"[{dep.name}] Knowledge retrieval/ingestion enabled but no knowledge_mound configured. "
                "Knowledge operations will be skipped."
            )

    # Check RLM compression
    if getattr(config, "use_rlm_limiter", False):
        dep = FEATURE_DEPENDENCIES["rlm_compression"]
        if not _check_module("aragora.rlm.compressor"):
            result.warnings.append(
                f"[{dep.name}] RLM limiter enabled but aragora.rlm.compressor not available. "
                "Context compression will be skipped."
            )

    # Check checkpointing
    if getattr(config, "enable_checkpointing", False):
        dep = FEATURE_DEPENDENCIES["checkpointing"]
        if not config.checkpoint_manager:
            result.warnings.append(
                f"[{dep.name}] Checkpointing enabled but no checkpoint_manager configured. "
                "A default CheckpointManager will be created."
            )

    # Check formal verification (requires Z3)
    protocol = getattr(config, "protocol", None)
    if protocol and getattr(protocol, "enable_formal_verification", False):
        dep = FEATURE_DEPENDENCIES["formal_verification"]
        if dep.check_fn and not dep.check_fn():
            result.errors.append(
                f"[{dep.name}] Formal verification enabled but Z3 is not installed. "
                "Install with: pip install z3-solver"
            )
            result.valid = False

    # Check calibration
    if protocol and getattr(protocol, "enable_calibration", False):
        dep = FEATURE_DEPENDENCIES["calibration"]
        if not config.calibration_tracker:
            result.warnings.append(
                f"[{dep.name}] Calibration enabled but no calibration_tracker configured. "
                "A default tracker will be created."
            )

    # Check performance monitoring
    if getattr(config, "enable_performance_monitor", False):
        dep = FEATURE_DEPENDENCIES["performance_monitoring"]
        if not config.performance_monitor:
            result.warnings.append(
                f"[{dep.name}] Performance monitoring enabled but no monitor configured. "
                "A default PerformanceMonitor will be created."
            )

    # Check population evolution
    if getattr(config, "auto_evolve", False):
        dep = FEATURE_DEPENDENCIES["population_evolution"]
        if not config.population_manager:
            result.errors.append(
                f"[{dep.name}] auto_evolve enabled but no population_manager configured. "
                "Evolution cannot proceed without a PopulationManager."
            )
            result.valid = False

    # Log results
    for warning in result.warnings:
        logger.warning(f"  [feature_validator] {warning}")
    for error in result.errors:
        logger.error(f"  [feature_validator] {error}")

    return result


def validate_and_warn(config: "ArenaConfig") -> None:
    """Validate config and log warnings. Does not raise errors.

    This is a convenience function for non-strict validation that
    logs warnings but allows the Arena to proceed.

    Args:
        config: ArenaConfig to validate
    """
    result = validate_feature_dependencies(config)
    if not result.valid:
        logger.warning(
            "  [feature_validator] Configuration has errors but proceeding. "
            "Some features may not work correctly."
        )
