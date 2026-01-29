"""
Server startup observability initialization.

This module handles error monitoring, OpenTelemetry tracing,
OTLP exporter, and Prometheus metrics initialization.
"""

import logging

logger = logging.getLogger(__name__)


async def init_error_monitoring() -> bool:
    """Initialize error monitoring (Sentry).

    Returns:
        True if monitoring was enabled, False otherwise
    """
    try:
        from aragora.server.error_monitoring import init_monitoring

        if init_monitoring():
            logger.info("Error monitoring enabled (Sentry)")
            return True
    except ImportError:
        pass
    return False


async def init_opentelemetry() -> bool:
    """Initialize OpenTelemetry tracing.

    Returns:
        True if tracing was enabled, False otherwise
    """
    try:
        from aragora.observability.config import is_tracing_enabled
        from aragora.observability.tracing import get_tracer

        if is_tracing_enabled():
            get_tracer()  # Initialize tracer singleton
            logger.info("OpenTelemetry tracing enabled")
            return True
        else:
            logger.debug("OpenTelemetry tracing disabled (set OTEL_ENABLED=true to enable)")
    except ImportError as e:
        logger.debug(f"OpenTelemetry not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")
    return False


async def init_otlp_exporter() -> bool:
    """Initialize OpenTelemetry OTLP exporter for distributed tracing.

    Configures trace export to external backends like Jaeger, Zipkin,
    or Datadog via the OTLP protocol. This is separate from the basic
    OpenTelemetry setup and provides more flexible backend options.

    Environment Variables:
        ARAGORA_OTLP_EXPORTER: Exporter type (none, jaeger, zipkin, otlp_grpc, otlp_http, datadog)
        ARAGORA_OTLP_ENDPOINT: Collector endpoint URL
        ARAGORA_SERVICE_NAME: Service name for traces (default: aragora)
        ARAGORA_ENVIRONMENT: Deployment environment (default: development)
        ARAGORA_TRACE_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)
        See docs/ENVIRONMENT.md for full configuration reference.

    Returns:
        True if OTLP exporter was configured, False otherwise
    """
    try:
        from aragora.observability.config import is_otlp_enabled
        from aragora.observability.otlp_export import configure_otlp_exporter, get_otlp_config

        if not is_otlp_enabled():
            logger.debug(
                "OTLP exporter disabled (set ARAGORA_OTLP_EXPORTER to jaeger/zipkin/otlp_grpc/otlp_http/datadog)"
            )
            return False

        config = get_otlp_config()
        provider = configure_otlp_exporter(config)

        if provider:
            logger.info(
                f"OTLP exporter initialized: type={config.exporter_type.value}, "
                f"endpoint={config.get_effective_endpoint()}"
            )
            return True
        else:
            logger.warning("OTLP exporter configuration failed")
            return False

    except ImportError as e:
        logger.debug(f"OTLP exporter not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize OTLP exporter: {e}")

    return False


async def init_prometheus_metrics() -> bool:
    """Initialize Prometheus metrics server.

    Returns:
        True if metrics were enabled, False otherwise
    """
    try:
        from aragora.observability.config import is_metrics_enabled
        from aragora.observability.metrics import start_metrics_server

        if is_metrics_enabled():
            start_metrics_server()
            logger.info("Prometheus metrics server started")
            return True
    except ImportError as e:
        logger.debug(f"Prometheus metrics not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")
    return False
