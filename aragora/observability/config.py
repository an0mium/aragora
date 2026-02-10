"""
Observability configuration for Aragora.

Provides configuration for OpenTelemetry tracing and Prometheus metrics.
All settings can be configured via environment variables.

Standard OpenTelemetry Environment Variables (recommended, takes precedence):
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (e.g., http://localhost:4317)
    OTEL_SERVICE_NAME: Service name for tracing (default: aragora)
    OTEL_TRACES_SAMPLER: Sampler type (always_on, always_off, traceidratio,
                         parentbased_always_on, parentbased_always_off,
                         parentbased_traceidratio)
    OTEL_TRACES_SAMPLER_ARG: Argument for sampler (e.g., 0.1 for 10% sampling)
    OTEL_PROPAGATORS: Context propagators (default: tracecontext,baggage)
    OTEL_RESOURCE_ATTRIBUTES: Additional resource attributes (key=value,key=value)

Legacy/Compatibility Variables:
    OTEL_ENABLED: Enable/disable OpenTelemetry tracing (default: false)
    OTEL_SAMPLE_RATE: Trace sampling rate 0.0-1.0 (default: 1.0)
    METRICS_ENABLED: Enable/disable Prometheus metrics (default: true)
    METRICS_PORT: Port for metrics endpoint (default: 9090)

OTLP Export Variables (Aragora-specific, fallback when OTEL_* not set):
    ARAGORA_OTLP_EXPORTER: Exporter type (none, jaeger, zipkin, otlp_grpc, otlp_http, datadog)
    ARAGORA_OTLP_ENDPOINT: Collector endpoint URL
    ARAGORA_SERVICE_NAME: Service name for traces (default: aragora)
    ARAGORA_SERVICE_VERSION: Service version (default: 1.0.0)
    ARAGORA_ENVIRONMENT: Deployment environment (default: development)
    ARAGORA_TRACE_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)
    ARAGORA_OTLP_HEADERS: JSON-encoded headers for authenticated endpoints
    ARAGORA_OTLP_BATCH_SIZE: Batch processor queue size (default: 512)
    ARAGORA_OTLP_EXPORT_TIMEOUT_MS: Export timeout in milliseconds (default: 30000)
    ARAGORA_OTLP_INSECURE: Allow insecure connections (default: false)
    DATADOG_API_KEY: Datadog API key (for datadog exporter type)

See docs/ENVIRONMENT.md for full configuration reference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry distributed tracing.

    Supports both standard OTEL_* and ARAGORA_* environment variables.
    Standard OTEL_* variables take precedence when both are set.
    """

    enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "aragora"
    service_version: str = "1.0.0"
    environment: str = "development"
    sample_rate: float = 1.0
    sampler_type: str = "parentbased_traceidratio"
    propagators: list[str] = field(default_factory=lambda: ["tracecontext", "baggage"])
    batch_size: int = 512
    export_timeout_ms: int = 30000
    insecure: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be between 0.0 and 1.0, got {self.sample_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class MetricsConfig:
    """Configuration for Prometheus metrics."""

    enabled: bool = True
    port: int = 9090
    path: str = "/metrics"
    include_host_metrics: bool = False
    histogram_buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )


# Global configuration instances
_tracing_config: TracingConfig | None = None
_metrics_config: MetricsConfig | None = None


def get_tracing_config() -> TracingConfig:
    """Get tracing configuration from environment variables.

    Prioritizes standard OTEL_* variables over ARAGORA_* ones.
    """
    global _tracing_config
    if _tracing_config is None:
        # Check if any OTEL endpoint is configured (auto-enable if so)
        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        aragora_exporter = os.getenv("ARAGORA_OTLP_EXPORTER", "none").lower()
        auto_enabled = bool(otel_endpoint) or (aragora_exporter != "none")

        # Explicit enable flag takes precedence
        explicit_enabled = os.getenv("OTEL_ENABLED", "").lower() in ("true", "1", "yes")
        enabled = explicit_enabled or auto_enabled

        # Endpoint: standard OTEL takes precedence
        endpoint = otel_endpoint or os.getenv("ARAGORA_OTLP_ENDPOINT") or "http://localhost:4317"

        # Service name
        service_name = (
            os.getenv("OTEL_SERVICE_NAME") or os.getenv("ARAGORA_SERVICE_NAME") or "aragora"
        )

        # Service version
        service_version = os.getenv("ARAGORA_SERVICE_VERSION", "1.0.0")

        # Environment
        environment = os.getenv("ARAGORA_ENVIRONMENT", "development")

        # Sampler type and rate
        sampler_type = os.getenv("OTEL_TRACES_SAMPLER", "parentbased_traceidratio")
        sample_rate = float(
            os.getenv("OTEL_TRACES_SAMPLER_ARG")
            or os.getenv("OTEL_SAMPLE_RATE")
            or os.getenv("ARAGORA_TRACE_SAMPLE_RATE")
            or "1.0"
        )

        # Propagators
        propagators_str = os.getenv("OTEL_PROPAGATORS", "tracecontext,baggage")
        propagators = [p.strip() for p in propagators_str.split(",") if p.strip()]

        # Batch settings
        batch_size = int(os.getenv("ARAGORA_OTLP_BATCH_SIZE", "512"))
        export_timeout = int(os.getenv("ARAGORA_OTLP_EXPORT_TIMEOUT_MS", "30000"))

        # Insecure mode
        insecure = os.getenv("ARAGORA_OTLP_INSECURE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        _tracing_config = TracingConfig(
            enabled=enabled,
            otlp_endpoint=endpoint,
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            sample_rate=sample_rate,
            sampler_type=sampler_type,
            propagators=propagators,
            batch_size=batch_size,
            export_timeout_ms=export_timeout,
            insecure=insecure,
        )
    return _tracing_config


def get_metrics_config() -> MetricsConfig:
    """Get metrics configuration from environment variables."""
    global _metrics_config
    if _metrics_config is None:
        # Offline mode should be network-free and quiet. Prometheus metrics are
        # primarily a server concern, so default them off when ARAGORA_OFFLINE is set.
        from aragora.utils.env import is_offline_mode

        enabled = os.getenv("METRICS_ENABLED", "true").lower() in ("true", "1", "yes")
        if is_offline_mode():
            enabled = False
        _metrics_config = MetricsConfig(
            enabled=enabled,
            port=int(os.getenv("METRICS_PORT", "9090")),
        )
    return _metrics_config


def set_tracing_config(config: TracingConfig) -> None:
    """Set custom tracing configuration (for testing)."""
    global _tracing_config
    _tracing_config = config


def set_metrics_config(config: MetricsConfig) -> None:
    """Set custom metrics configuration (for testing)."""
    global _metrics_config
    _metrics_config = config


def reset_config() -> None:
    """Reset configuration to defaults (for testing)."""
    global _tracing_config, _metrics_config
    _tracing_config = None
    _metrics_config = None


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return get_tracing_config().enabled


def is_metrics_enabled() -> bool:
    """Check if metrics are enabled."""
    return get_metrics_config().enabled


def is_otlp_enabled() -> bool:
    """Check if OTLP export is enabled.

    Returns True if ARAGORA_OTLP_EXPORTER is set to a value other than 'none'.
    """
    from aragora.observability.otlp_export import OTLPExporterType, get_otlp_config

    config = get_otlp_config()
    return config.exporter_type != OTLPExporterType.NONE
