"""
Observability configuration for Aragora.

Provides configuration for OpenTelemetry tracing and Prometheus metrics.
All settings can be configured via environment variables.

Environment Variables:
    OTEL_ENABLED: Enable/disable OpenTelemetry tracing (default: false)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4317)
    OTEL_SERVICE_NAME: Service name for tracing (default: aragora)
    OTEL_SAMPLE_RATE: Trace sampling rate 0.0-1.0 (default: 1.0)
    METRICS_ENABLED: Enable/disable Prometheus metrics (default: true)
    METRICS_PORT: Port for metrics endpoint (default: 9090)

OTLP Export Variables (for advanced distributed tracing):
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
from typing import Optional


@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry distributed tracing."""

    enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "aragora"
    sample_rate: float = 1.0
    propagators: list[str] = field(default_factory=lambda: ["tracecontext", "baggage"])
    batch_size: int = 512
    export_timeout_ms: int = 30000

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
_tracing_config: Optional[TracingConfig] = None
_metrics_config: Optional[MetricsConfig] = None


def get_tracing_config() -> TracingConfig:
    """Get tracing configuration from environment variables."""
    global _tracing_config
    if _tracing_config is None:
        _tracing_config = TracingConfig(
            enabled=os.getenv("OTEL_ENABLED", "false").lower() in ("true", "1", "yes"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            service_name=os.getenv("OTEL_SERVICE_NAME", "aragora"),
            sample_rate=float(os.getenv("OTEL_SAMPLE_RATE", "1.0")),
        )
    return _tracing_config


def get_metrics_config() -> MetricsConfig:
    """Get metrics configuration from environment variables."""
    global _metrics_config
    if _metrics_config is None:
        _metrics_config = MetricsConfig(
            enabled=os.getenv("METRICS_ENABLED", "true").lower() in ("true", "1", "yes"),
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
