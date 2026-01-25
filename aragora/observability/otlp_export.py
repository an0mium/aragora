"""
OpenTelemetry OTLP Export Configuration.

Provides flexible OTLP exporter configuration supporting multiple backends:
- Jaeger (legacy Thrift protocol)
- Zipkin (JSON format)
- OTLP/gRPC (standard OpenTelemetry protocol)
- OTLP/HTTP (standard OpenTelemetry protocol over HTTP)
- Datadog (via OTLP with Datadog-specific configuration)

Environment Variables:
    ARAGORA_OTLP_EXPORTER: Exporter type (none, jaeger, zipkin, otlp_grpc, otlp_http, datadog)
    ARAGORA_OTLP_ENDPOINT: Exporter endpoint URL
    ARAGORA_SERVICE_NAME: Service name for traces (default: aragora)
    ARAGORA_SERVICE_VERSION: Service version (default: 1.0.0)
    ARAGORA_ENVIRONMENT: Deployment environment (default: development)
    ARAGORA_TRACE_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)
    ARAGORA_OTLP_HEADERS: JSON-encoded headers for authenticated endpoints
    ARAGORA_OTLP_BATCH_SIZE: Batch processor queue size (default: 512)
    ARAGORA_OTLP_EXPORT_TIMEOUT_MS: Export timeout in milliseconds (default: 30000)
    ARAGORA_OTLP_INSECURE: Allow insecure connections (default: false)
    DATADOG_API_KEY: Datadog API key (for datadog exporter type)

Usage:
    from aragora.observability.otlp_export import configure_otlp_exporter, OTLPConfig

    # Configure from environment
    provider = configure_otlp_exporter()

    # Or with custom config
    config = OTLPConfig(
        exporter_type=OTLPExporterType.OTLP_GRPC,
        endpoint="http://localhost:4317",
        service_name="aragora",
    )
    provider = configure_otlp_exporter(config)

See docs/OBSERVABILITY.md for deployment configuration examples.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OTLPExporterType(str, Enum):
    """Supported OTLP exporter types."""

    NONE = "none"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"
    DATADOG = "datadog"


# Default endpoints for each exporter type
DEFAULT_ENDPOINTS: Dict[OTLPExporterType, str] = {
    OTLPExporterType.JAEGER: "localhost",  # Jaeger uses host only
    OTLPExporterType.ZIPKIN: "http://localhost:9411/api/v2/spans",
    OTLPExporterType.OTLP_GRPC: "http://localhost:4317",
    OTLPExporterType.OTLP_HTTP: "http://localhost:4318/v1/traces",
    OTLPExporterType.DATADOG: "http://localhost:4317",  # Datadog Agent OTLP endpoint
}


@dataclass
class OTLPConfig:
    """Configuration for OTLP trace export.

    Attributes:
        exporter_type: The type of exporter to use (none, jaeger, zipkin, otlp_grpc, otlp_http, datadog)
        endpoint: The collector/agent endpoint URL
        service_name: Service name for distributed tracing
        service_version: Service version string
        environment: Deployment environment (development, staging, production)
        sample_rate: Trace sampling rate (0.0 to 1.0, where 1.0 = 100%)
        headers: HTTP headers for authenticated endpoints (e.g., API keys)
        batch_size: Maximum queue size for the batch processor
        export_timeout_ms: Export timeout in milliseconds
        insecure: Allow insecure (non-TLS) connections
        datadog_api_key: Datadog API key (only used with datadog exporter type)
    """

    exporter_type: OTLPExporterType = OTLPExporterType.NONE
    endpoint: Optional[str] = None
    service_name: str = "aragora"
    service_version: str = "1.0.0"
    environment: str = "development"
    sample_rate: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)
    batch_size: int = 512
    export_timeout_ms: int = 30000
    insecure: bool = False
    datadog_api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be between 0.0 and 1.0, got {self.sample_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.export_timeout_ms < 1:
            raise ValueError(f"export_timeout_ms must be positive, got {self.export_timeout_ms}")

    @classmethod
    def from_env(cls) -> "OTLPConfig":
        """Create configuration from environment variables.

        Returns:
            OTLPConfig instance configured from environment variables.

        Environment Variables:
            ARAGORA_OTLP_EXPORTER: Exporter type
            ARAGORA_OTLP_ENDPOINT: Collector endpoint
            ARAGORA_SERVICE_NAME: Service name
            ARAGORA_SERVICE_VERSION: Service version
            ARAGORA_ENVIRONMENT: Deployment environment
            ARAGORA_TRACE_SAMPLE_RATE: Sample rate (0.0-1.0)
            ARAGORA_OTLP_HEADERS: JSON-encoded headers dict
            ARAGORA_OTLP_BATCH_SIZE: Batch queue size
            ARAGORA_OTLP_EXPORT_TIMEOUT_MS: Export timeout
            ARAGORA_OTLP_INSECURE: Allow insecure connections
            DATADOG_API_KEY: Datadog API key
        """
        # Parse exporter type
        exporter_str = os.environ.get("ARAGORA_OTLP_EXPORTER", "none").lower()
        try:
            exporter_type = OTLPExporterType(exporter_str)
        except ValueError:
            logger.warning(
                f"Unknown OTLP exporter type: {exporter_str}, using 'none'. "
                f"Valid options: {[e.value for e in OTLPExporterType]}"
            )
            exporter_type = OTLPExporterType.NONE

        # Parse headers from JSON
        headers: Dict[str, str] = {}
        headers_json = os.environ.get("ARAGORA_OTLP_HEADERS", "")
        if headers_json:
            try:
                headers = json.loads(headers_json)
                if not isinstance(headers, dict):
                    raise ValueError("Headers must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse ARAGORA_OTLP_HEADERS: {e}")

        return cls(
            exporter_type=exporter_type,
            endpoint=os.environ.get("ARAGORA_OTLP_ENDPOINT"),
            service_name=os.environ.get("ARAGORA_SERVICE_NAME", "aragora"),
            service_version=os.environ.get("ARAGORA_SERVICE_VERSION", "1.0.0"),
            environment=os.environ.get("ARAGORA_ENVIRONMENT", "development"),
            sample_rate=float(os.environ.get("ARAGORA_TRACE_SAMPLE_RATE", "1.0")),
            headers=headers,
            batch_size=int(os.environ.get("ARAGORA_OTLP_BATCH_SIZE", "512")),
            export_timeout_ms=int(os.environ.get("ARAGORA_OTLP_EXPORT_TIMEOUT_MS", "30000")),
            insecure=os.environ.get("ARAGORA_OTLP_INSECURE", "false").lower()
            in ("true", "1", "yes"),
            datadog_api_key=os.environ.get("DATADOG_API_KEY"),
        )

    def get_effective_endpoint(self) -> Optional[str]:
        """Get the effective endpoint, using defaults if not specified.

        Returns:
            The endpoint URL or None if exporter type is NONE.
        """
        if self.exporter_type == OTLPExporterType.NONE:
            return None
        return self.endpoint or DEFAULT_ENDPOINTS.get(self.exporter_type)


# Global singleton for configured provider
_tracer_provider: Optional[Any] = None
_config: Optional[OTLPConfig] = None


def get_otlp_config() -> OTLPConfig:
    """Get the current OTLP configuration.

    Returns:
        The current OTLPConfig, loading from environment if not yet configured.
    """
    global _config
    if _config is None:
        _config = OTLPConfig.from_env()
    return _config


def set_otlp_config(config: OTLPConfig) -> None:
    """Set custom OTLP configuration (primarily for testing).

    Args:
        config: The OTLPConfig to use.
    """
    global _config
    _config = config


def reset_otlp_config() -> None:
    """Reset OTLP configuration to be re-read from environment."""
    global _config, _tracer_provider
    _config = None
    _tracer_provider = None


def _get_jaeger_exporter(config: OTLPConfig) -> Any:
    """Create Jaeger exporter.

    Args:
        config: OTLP configuration.

    Returns:
        JaegerExporter instance or None if unavailable.
    """
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter

        endpoint = config.get_effective_endpoint() or "localhost"
        return JaegerExporter(
            agent_host_name=endpoint,
            agent_port=6831,  # Default Jaeger agent UDP port
        )
    except ImportError:
        logger.warning(
            "Jaeger exporter not available. Install with: pip install opentelemetry-exporter-jaeger"
        )
        return None


def _get_zipkin_exporter(config: OTLPConfig) -> Any:
    """Create Zipkin exporter.

    Args:
        config: OTLP configuration.

    Returns:
        ZipkinExporter instance or None if unavailable.
    """
    try:
        from opentelemetry.exporter.zipkin.json import ZipkinExporter

        endpoint = config.get_effective_endpoint()
        return ZipkinExporter(endpoint=endpoint)
    except ImportError:
        logger.warning(
            "Zipkin exporter not available. Install with: pip install opentelemetry-exporter-zipkin"
        )
        return None


def _get_otlp_grpc_exporter(config: OTLPConfig) -> Any:
    """Create OTLP/gRPC exporter.

    Args:
        config: OTLP configuration.

    Returns:
        OTLPSpanExporter instance or None if unavailable.
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        endpoint = config.get_effective_endpoint()
        return OTLPSpanExporter(
            endpoint=endpoint,
            headers=config.headers or None,
            insecure=config.insecure,
            timeout=config.export_timeout_ms // 1000,  # Convert to seconds
        )
    except ImportError:
        logger.warning(
            "OTLP gRPC exporter not available. Install with: "
            "pip install opentelemetry-exporter-otlp-proto-grpc"
        )
        return None


def _get_otlp_http_exporter(config: OTLPConfig) -> Any:
    """Create OTLP/HTTP exporter.

    Args:
        config: OTLP configuration.

    Returns:
        OTLPSpanExporter instance or None if unavailable.
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = config.get_effective_endpoint()
        return OTLPSpanExporter(
            endpoint=endpoint,
            headers=config.headers or None,
            timeout=config.export_timeout_ms // 1000,  # Convert to seconds
        )
    except ImportError:
        logger.warning(
            "OTLP HTTP exporter not available. Install with: "
            "pip install opentelemetry-exporter-otlp-proto-http"
        )
        return None


def _get_datadog_exporter(config: OTLPConfig) -> Any:
    """Create Datadog exporter via OTLP.

    Datadog supports OTLP ingest via the Datadog Agent.

    Args:
        config: OTLP configuration.

    Returns:
        OTLPSpanExporter configured for Datadog or None if unavailable.
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        endpoint = config.get_effective_endpoint()

        # Datadog-specific headers
        headers = dict(config.headers) if config.headers else {}
        if config.datadog_api_key:
            headers["DD-API-KEY"] = config.datadog_api_key

        return OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers or None,
            insecure=config.insecure,
            timeout=config.export_timeout_ms // 1000,
        )
    except ImportError:
        logger.warning(
            "OTLP gRPC exporter not available for Datadog. Install with: "
            "pip install opentelemetry-exporter-otlp-proto-grpc"
        )
        return None


def _get_exporter(config: OTLPConfig) -> Any:
    """Get the appropriate exporter based on configuration.

    Args:
        config: OTLP configuration.

    Returns:
        Exporter instance or None if exporter type is NONE or unavailable.
    """
    if config.exporter_type == OTLPExporterType.NONE:
        return None
    elif config.exporter_type == OTLPExporterType.JAEGER:
        return _get_jaeger_exporter(config)
    elif config.exporter_type == OTLPExporterType.ZIPKIN:
        return _get_zipkin_exporter(config)
    elif config.exporter_type == OTLPExporterType.OTLP_GRPC:
        return _get_otlp_grpc_exporter(config)
    elif config.exporter_type == OTLPExporterType.OTLP_HTTP:
        return _get_otlp_http_exporter(config)
    elif config.exporter_type == OTLPExporterType.DATADOG:
        return _get_datadog_exporter(config)
    else:
        logger.warning(f"Unknown exporter type: {config.exporter_type}")
        return None


def configure_otlp_exporter(config: Optional[OTLPConfig] = None) -> Any:
    """Configure OpenTelemetry with OTLP exporter.

    This function sets up the OpenTelemetry tracing infrastructure with the
    specified exporter backend. It configures:
    - Resource attributes (service name, version, environment)
    - Trace sampling based on sample_rate
    - Batch span processor for efficient export
    - The global tracer provider

    Args:
        config: OTLP configuration. If None, loads from environment variables.

    Returns:
        The configured TracerProvider, or None if tracing is disabled or
        OpenTelemetry is not available.

    Example:
        # Configure from environment
        provider = configure_otlp_exporter()

        # Configure for Jaeger
        config = OTLPConfig(
            exporter_type=OTLPExporterType.JAEGER,
            endpoint="jaeger.example.com",
        )
        provider = configure_otlp_exporter(config)
    """
    global _tracer_provider, _config

    config = config or OTLPConfig.from_env()
    _config = config

    if config.exporter_type == OTLPExporterType.NONE:
        logger.debug("OTLP tracing disabled (exporter_type=none)")
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # Create resource with service attributes
        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.version": config.service_version,
                "deployment.environment": config.environment,
            }
        )

        # Create sampler based on sample rate
        sampler = TraceIdRatioBased(config.sample_rate)

        # Create provider with resource and sampler
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Get the appropriate exporter
        exporter = _get_exporter(config)
        if exporter:
            # Create batch processor for efficient export
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=config.batch_size * 2,
                max_export_batch_size=config.batch_size,
                export_timeout_millis=config.export_timeout_ms,
            )
            provider.add_span_processor(processor)

            # Set as global provider
            trace.set_tracer_provider(provider)
            _tracer_provider = provider

            logger.info(
                f"OTLP tracing configured: exporter={config.exporter_type.value}, "
                f"endpoint={config.get_effective_endpoint()}, "
                f"service={config.service_name}, "
                f"environment={config.environment}, "
                f"sample_rate={config.sample_rate}"
            )

            return provider
        else:
            logger.warning(
                f"Failed to create {config.exporter_type.value} exporter, tracing disabled"
            )
            return None

    except ImportError as e:
        logger.warning(
            f"OpenTelemetry SDK not available: {e}. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to configure OTLP exporter: {e}")
        return None


def get_tracer_provider() -> Any:
    """Get the configured tracer provider.

    Returns:
        The TracerProvider if configured, None otherwise.
    """
    return _tracer_provider


def shutdown_otlp() -> None:
    """Shutdown the OTLP tracer provider gracefully.

    This should be called during application shutdown to ensure
    all pending spans are exported.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("OTLP tracer provider shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down OTLP tracer: {e}")
        finally:
            _tracer_provider = None


__all__ = [
    "OTLPExporterType",
    "OTLPConfig",
    "DEFAULT_ENDPOINTS",
    "configure_otlp_exporter",
    "get_otlp_config",
    "set_otlp_config",
    "reset_otlp_config",
    "get_tracer_provider",
    "shutdown_otlp",
]
