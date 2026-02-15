"""
Research integration configuration.

Provides preset configurations for research-oriented debate features
such as adaptive stopping, MUSE scoring, LARA analysis, ThinkPRM
verification, GraphRAG, and ASCOT calibration.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum


class IntegrationLevel(Enum):
    """Level of research integration."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"
    CUSTOM = "custom"


@dataclass
class FeatureConfig:
    """Base feature toggle."""

    enabled: bool = False


@dataclass
class AdaptiveStoppingConfig(FeatureConfig):
    """Adaptive stopping configuration."""

    threshold: float = 0.85
    min_rounds: int = 2
    max_rounds: int = 10


@dataclass
class MuseConfig(FeatureConfig):
    """MUSE scoring configuration."""

    weight: float = 0.3


@dataclass
class LaraConfig(FeatureConfig):
    """LARA analysis configuration."""

    max_hops: int = 2
    depth: int = 3


@dataclass
class ThinkPrmConfig(FeatureConfig):
    """ThinkPRM verification configuration."""

    max_parallel: int = 3
    timeout: float = 30.0


@dataclass
class GraphRagConfig(FeatureConfig):
    """GraphRAG configuration."""

    max_hops: int = 2
    chunk_size: int = 512


@dataclass
class TelemetryConfig(FeatureConfig):
    """Telemetry configuration."""

    record_all_events: bool = False
    export_interval: float = 60.0


@dataclass
class AscotConfig(FeatureConfig):
    """ASCOT calibration configuration."""

    recalibrate_interval: int = 5


@dataclass
class DebateAnalyticsConfig(FeatureConfig):
    """Debate analytics configuration."""

    track_convergence: bool = True
    export_graphs: bool = False


@dataclass
class ResearchIntegrationConfig:
    """Configuration for research integration features."""

    level: IntegrationLevel = IntegrationLevel.STANDARD
    adaptive_stopping: AdaptiveStoppingConfig = field(default_factory=AdaptiveStoppingConfig)
    muse: MuseConfig = field(default_factory=MuseConfig)
    lara: LaraConfig = field(default_factory=LaraConfig)
    think_prm: ThinkPrmConfig = field(default_factory=ThinkPrmConfig)
    graph_rag: GraphRagConfig = field(default_factory=GraphRagConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    ascot: AscotConfig = field(default_factory=AscotConfig)
    debate_analytics: DebateAnalyticsConfig = field(default_factory=DebateAnalyticsConfig)

    @classmethod
    def from_preset(cls, level: IntegrationLevel) -> "ResearchIntegrationConfig":
        """Create configuration from a preset level."""
        if level == IntegrationLevel.MINIMAL:
            return cls(
                level=level,
                adaptive_stopping=AdaptiveStoppingConfig(enabled=True),
                muse=MuseConfig(enabled=False),
                lara=LaraConfig(enabled=False),
                think_prm=ThinkPrmConfig(enabled=False),
                graph_rag=GraphRagConfig(enabled=False),
                telemetry=TelemetryConfig(enabled=True),
                ascot=AscotConfig(enabled=False),
                debate_analytics=DebateAnalyticsConfig(enabled=False),
            )
        elif level == IntegrationLevel.FULL:
            return cls(
                level=level,
                adaptive_stopping=AdaptiveStoppingConfig(enabled=True),
                muse=MuseConfig(enabled=True),
                lara=LaraConfig(enabled=True, max_hops=3, depth=5),
                think_prm=ThinkPrmConfig(enabled=True, max_parallel=5),
                graph_rag=GraphRagConfig(enabled=True, max_hops=3),
                telemetry=TelemetryConfig(enabled=True, record_all_events=True),
                ascot=AscotConfig(enabled=True),
                debate_analytics=DebateAnalyticsConfig(enabled=True, export_graphs=True),
            )
        elif level == IntegrationLevel.STANDARD:
            return cls(
                level=level,
                adaptive_stopping=AdaptiveStoppingConfig(enabled=True),
                muse=MuseConfig(enabled=True),
                lara=LaraConfig(enabled=True),
                think_prm=ThinkPrmConfig(enabled=True),
                graph_rag=GraphRagConfig(enabled=False),
                telemetry=TelemetryConfig(enabled=True),
                ascot=AscotConfig(enabled=True),
                debate_analytics=DebateAnalyticsConfig(enabled=True),
            )
        else:
            # CUSTOM: all defaults (disabled)
            return cls(level=level)

    def get_enabled_features(self) -> list[str]:
        """Return names of all enabled features."""
        features = []
        for name in (
            "adaptive_stopping",
            "muse",
            "lara",
            "think_prm",
            "graph_rag",
            "telemetry",
            "ascot",
            "debate_analytics",
        ):
            cfg = getattr(self, name)
            if cfg.enabled:
                features.append(name)
        return features

    def validate(self) -> list[str]:
        """Validate configuration and return warnings."""
        warnings: list[str] = []
        if self.adaptive_stopping.enabled and not self.muse.enabled:
            warnings.append(
                "MUSE scoring is disabled; adaptive stopping may lack quality signal gating."
            )
        if self.adaptive_stopping.enabled and not self.ascot.enabled:
            warnings.append(
                "ASCOT calibration is disabled; adaptive stopping may lack calibration gating."
            )
        if self.lara.enabled and not self.graph_rag.enabled:
            warnings.append(
                "GraphRAG is disabled; LARA analysis may have limited knowledge retrieval."
            )
        return warnings

    def to_dict(self) -> dict:
        """Convert to dictionary with enum values as strings."""
        result = asdict(self)
        result["level"] = self.level.value
        return result
