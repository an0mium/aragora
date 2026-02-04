"""
Unified Research Integration Configuration.

Provides a master configuration for all research integrations with:
- Feature flags for each integration
- Presets for common configurations (MINIMAL, STANDARD, FULL, CUSTOM)
- Runtime configuration adjustment
- Environment variable overrides
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class IntegrationLevel(Enum):
    """Preset integration levels."""

    MINIMAL = "minimal"  # Only essential features, lowest overhead
    STANDARD = "standard"  # Balanced features and performance
    FULL = "full"  # All features enabled
    CUSTOM = "custom"  # User-defined configuration


@dataclass
class AdaptiveStoppingConfig:
    """Configuration for adaptive stopping (Issue #1)."""

    enabled: bool = True
    stability_threshold: float = 0.85
    ks_threshold: float = 0.1
    min_stable_rounds: int = 1
    muse_disagreement_gate: float = 0.4
    ascot_fragility_gate: float = 0.7
    min_rounds_before_check: int = 2


@dataclass
class MUSEConfig:
    """Configuration for MUSE ensemble uncertainty (Issue #2)."""

    enabled: bool = True
    min_subset_size: int = 2
    max_subset_size: int = 5
    muse_weight: float = 0.15


@dataclass
class LaRAConfig:
    """Configuration for LaRA retrieval routing (Issue #3)."""

    enabled: bool = True
    auto_routing: bool = True
    long_context_min_tokens: int = 50000
    long_context_max_tokens: int = 200000
    vector_top_k: int = 10
    max_hops: int = 2
    graph_weight: float = 0.3


@dataclass
class ASCoTConfig:
    """Configuration for ASCoT fragility detection (Issue #4)."""

    enabled: bool = True
    lambda_factor: float = 2.0
    critical_threshold: float = 0.8
    high_threshold: float = 0.6
    base_error_rate: float = 0.05


@dataclass
class ThinkPRMConfig:
    """Configuration for ThinkPRM verification (Issue #8)."""

    enabled: bool = True
    verifier_agent_id: str = "claude"
    parallel_verification: bool = True
    max_parallel: int = 3
    critical_round_threshold: float = 0.7
    cache_verifications: bool = True


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG hybrid retrieval (Issue #10)."""

    enabled: bool = True
    max_hops: int = 2
    max_neighbors_per_hop: int = 5
    enable_community_detection: bool = True
    min_community_size: int = 3


@dataclass
class ClaimCheckConfig:
    """Configuration for ClaimCheck verification (Issue #11)."""

    enabled: bool = True
    use_llm_decomposition: bool = True
    max_atomic_claims: int = 10
    semantic_threshold: float = 0.75
    inference_threshold: float = 0.7


@dataclass
class AHMADConfig:
    """Configuration for A-HMAD role specialization (Issue #12)."""

    enabled: bool = True
    min_diversity_score: float = 0.6
    elo_weight: float = 0.3
    calibration_weight: float = 0.25
    domain_weight: float = 0.25
    diversity_penalty: float = 0.3


@dataclass
class TelemetryConfig:
    """Configuration for research integration telemetry (Issue #6)."""

    enabled: bool = True
    buffer_size: int = 100
    flush_interval_seconds: int = 60
    record_all_events: bool = False  # If false, only aggregates


@dataclass
class ResearchIntegrationConfig:
    """
    Master configuration for all research integrations.

    Example:
        # Use a preset
        config = ResearchIntegrationConfig.from_preset(IntegrationLevel.STANDARD)

        # Or customize
        config = ResearchIntegrationConfig(
            level=IntegrationLevel.CUSTOM,
            adaptive_stopping=AdaptiveStoppingConfig(stability_threshold=0.9),
        )

        # Check if feature is enabled
        if config.adaptive_stopping.enabled:
            detector = create_stability_detector(config.adaptive_stopping)
    """

    level: IntegrationLevel = IntegrationLevel.STANDARD

    # Phase 1: Foundation
    adaptive_stopping: AdaptiveStoppingConfig = field(default_factory=AdaptiveStoppingConfig)
    muse: MUSEConfig = field(default_factory=MUSEConfig)
    lara: LaRAConfig = field(default_factory=LaRAConfig)
    ascot: ASCoTConfig = field(default_factory=ASCoTConfig)

    # Phase 2: Verification
    think_prm: ThinkPRMConfig = field(default_factory=ThinkPRMConfig)

    # Phase 3: Knowledge & Evidence
    graph_rag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    claim_check: ClaimCheckConfig = field(default_factory=ClaimCheckConfig)

    # Phase 4: Team Selection
    ahmad: AHMADConfig = field(default_factory=AHMADConfig)

    # Infrastructure
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def from_preset(cls, level: IntegrationLevel) -> "ResearchIntegrationConfig":
        """Create configuration from a preset level.

        Args:
            level: The integration level preset

        Returns:
            Configured ResearchIntegrationConfig
        """
        if level == IntegrationLevel.MINIMAL:
            return cls._minimal_preset()
        elif level == IntegrationLevel.STANDARD:
            return cls._standard_preset()
        elif level == IntegrationLevel.FULL:
            return cls._full_preset()
        else:
            return cls(level=level)

    @classmethod
    def _minimal_preset(cls) -> "ResearchIntegrationConfig":
        """Minimal preset: only essential features."""
        return cls(
            level=IntegrationLevel.MINIMAL,
            adaptive_stopping=AdaptiveStoppingConfig(enabled=True),
            muse=MUSEConfig(enabled=False),
            lara=LaRAConfig(enabled=False, auto_routing=False),
            ascot=ASCoTConfig(enabled=False),
            think_prm=ThinkPRMConfig(enabled=False),
            graph_rag=GraphRAGConfig(enabled=False),
            claim_check=ClaimCheckConfig(enabled=False, use_llm_decomposition=False),
            ahmad=AHMADConfig(enabled=False),
            telemetry=TelemetryConfig(enabled=True, record_all_events=False),
        )

    @classmethod
    def _standard_preset(cls) -> "ResearchIntegrationConfig":
        """Standard preset: balanced features."""
        return cls(
            level=IntegrationLevel.STANDARD,
            adaptive_stopping=AdaptiveStoppingConfig(enabled=True),
            muse=MUSEConfig(enabled=True),
            lara=LaRAConfig(enabled=True, auto_routing=True),
            ascot=ASCoTConfig(enabled=True),
            think_prm=ThinkPRMConfig(enabled=True, parallel_verification=True),
            graph_rag=GraphRAGConfig(enabled=True, max_hops=2),
            claim_check=ClaimCheckConfig(enabled=True, use_llm_decomposition=False),
            ahmad=AHMADConfig(enabled=True),
            telemetry=TelemetryConfig(enabled=True, record_all_events=False),
        )

    @classmethod
    def _full_preset(cls) -> "ResearchIntegrationConfig":
        """Full preset: all features enabled."""
        return cls(
            level=IntegrationLevel.FULL,
            adaptive_stopping=AdaptiveStoppingConfig(enabled=True),
            muse=MUSEConfig(enabled=True),
            lara=LaRAConfig(enabled=True, auto_routing=True, max_hops=3),
            ascot=ASCoTConfig(enabled=True),
            think_prm=ThinkPRMConfig(
                enabled=True,
                parallel_verification=True,
                max_parallel=5,
            ),
            graph_rag=GraphRAGConfig(
                enabled=True,
                max_hops=3,
                enable_community_detection=True,
            ),
            claim_check=ClaimCheckConfig(
                enabled=True,
                use_llm_decomposition=True,
            ),
            ahmad=AHMADConfig(enabled=True),
            telemetry=TelemetryConfig(
                enabled=True,
                record_all_events=True,
            ),
        )

    @classmethod
    def from_env(cls) -> "ResearchIntegrationConfig":
        """Create configuration from environment variables.

        Environment variables:
            ARAGORA_RESEARCH_LEVEL: Integration level (minimal/standard/full/custom)
            ARAGORA_ADAPTIVE_STOPPING: Enable adaptive stopping (true/false)
            ARAGORA_MUSE_ENABLED: Enable MUSE (true/false)
            ARAGORA_LARA_ENABLED: Enable LaRA routing (true/false)
            ARAGORA_LARA_AUTO_ROUTING: Enable auto routing (true/false)
            ARAGORA_THINK_PRM_ENABLED: Enable ThinkPRM (true/false)
            ARAGORA_GRAPH_RAG_ENABLED: Enable GraphRAG (true/false)
            ARAGORA_CLAIM_CHECK_ENABLED: Enable ClaimCheck (true/false)
            ARAGORA_AHMAD_ENABLED: Enable A-HMAD (true/false)

        Returns:
            Configured ResearchIntegrationConfig
        """
        level_str = os.getenv("ARAGORA_RESEARCH_LEVEL", "standard").lower()
        level = {
            "minimal": IntegrationLevel.MINIMAL,
            "standard": IntegrationLevel.STANDARD,
            "full": IntegrationLevel.FULL,
            "custom": IntegrationLevel.CUSTOM,
        }.get(level_str, IntegrationLevel.STANDARD)

        # Start with preset
        config = cls.from_preset(level)

        # Override from environment
        def env_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default

        config.adaptive_stopping.enabled = env_bool(
            "ARAGORA_ADAPTIVE_STOPPING",
            config.adaptive_stopping.enabled,
        )
        config.muse.enabled = env_bool("ARAGORA_MUSE_ENABLED", config.muse.enabled)
        config.lara.enabled = env_bool("ARAGORA_LARA_ENABLED", config.lara.enabled)
        config.lara.auto_routing = env_bool(
            "ARAGORA_LARA_AUTO_ROUTING",
            config.lara.auto_routing,
        )
        config.think_prm.enabled = env_bool(
            "ARAGORA_THINK_PRM_ENABLED",
            config.think_prm.enabled,
        )
        config.graph_rag.enabled = env_bool(
            "ARAGORA_GRAPH_RAG_ENABLED",
            config.graph_rag.enabled,
        )
        config.claim_check.enabled = env_bool(
            "ARAGORA_CLAIM_CHECK_ENABLED",
            config.claim_check.enabled,
        )
        config.ahmad.enabled = env_bool("ARAGORA_AHMAD_ENABLED", config.ahmad.enabled)

        logger.info(
            "research_config_loaded level=%s from_env=True",
            config.level.value,
        )

        return config

    def get_enabled_features(self) -> list[str]:
        """Get list of enabled feature names.

        Returns:
            List of enabled feature names
        """
        features = []

        if self.adaptive_stopping.enabled:
            features.append("adaptive_stopping")
        if self.muse.enabled:
            features.append("muse")
        if self.lara.enabled:
            features.append("lara")
        if self.ascot.enabled:
            features.append("ascot")
        if self.think_prm.enabled:
            features.append("think_prm")
        if self.graph_rag.enabled:
            features.append("graph_rag")
        if self.claim_check.enabled:
            features.append("claim_check")
        if self.ahmad.enabled:
            features.append("ahmad")
        if self.telemetry.enabled:
            features.append("telemetry")

        return features

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "level": self.level.value,
            "adaptive_stopping": vars(self.adaptive_stopping),
            "muse": vars(self.muse),
            "lara": vars(self.lara),
            "ascot": vars(self.ascot),
            "think_prm": vars(self.think_prm),
            "graph_rag": vars(self.graph_rag),
            "claim_check": vars(self.claim_check),
            "ahmad": vars(self.ahmad),
            "telemetry": vars(self.telemetry),
        }

    def validate(self) -> list[str]:
        """Validate configuration for consistency.

        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []

        # Check for conflicting settings
        if self.adaptive_stopping.enabled and not self.ascot.enabled:
            # ASCoT provides fragility gating for adaptive stopping
            warnings.append("adaptive_stopping enabled without ascot may miss late-stage fragility")

        if self.adaptive_stopping.enabled and not self.muse.enabled:
            # MUSE provides disagreement gating
            warnings.append(
                "adaptive_stopping enabled without muse may stop despite high disagreement"
            )

        if self.lara.auto_routing and not self.graph_rag.enabled:
            warnings.append(
                "lara auto_routing enabled but graph_rag disabled - GRAPH mode unavailable"
            )

        if self.claim_check.use_llm_decomposition and not self.think_prm.enabled:
            # Both use LLM verification, should be coordinated
            warnings.append(
                "claim_check LLM decomposition without think_prm - consider enabling for consistency"
            )

        return warnings


# Global configuration instance
_global_config: Optional[ResearchIntegrationConfig] = None


def get_research_config() -> ResearchIntegrationConfig:
    """Get the global research integration configuration.

    Returns:
        Global ResearchIntegrationConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ResearchIntegrationConfig.from_env()
    return _global_config


def set_research_config(config: ResearchIntegrationConfig) -> None:
    """Set the global research integration configuration.

    Args:
        config: Configuration to set as global
    """
    global _global_config
    _global_config = config
    logger.info(
        "research_config_set level=%s features=%s",
        config.level.value,
        config.get_enabled_features(),
    )


def reset_research_config() -> None:
    """Reset global configuration to None (will reload from env on next get)."""
    global _global_config
    _global_config = None


# Convenience function for CLI
def get_level_from_args(args: Any) -> IntegrationLevel:
    """Extract integration level from CLI args.

    Args:
        args: Parsed CLI arguments with optional 'research_level' attribute

    Returns:
        IntegrationLevel from args or default
    """
    level_str = getattr(args, "research_level", "standard")
    return {
        "minimal": IntegrationLevel.MINIMAL,
        "standard": IntegrationLevel.STANDARD,
        "full": IntegrationLevel.FULL,
        "custom": IntegrationLevel.CUSTOM,
    }.get(level_str, IntegrationLevel.STANDARD)
