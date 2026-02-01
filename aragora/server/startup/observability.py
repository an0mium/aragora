"""
Server startup observability initialization.

This module handles error monitoring, OpenTelemetry tracing,
OTLP exporter, and Prometheus metrics initialization.
"""

import logging
import os

logger = logging.getLogger(__name__)


def init_structured_logging() -> bool:
    """Initialize structured JSON logging.

    Uses JSON format in production (ARAGORA_ENV=production or ARAGORA_LOG_FORMAT=json),
    text format otherwise for easier local development.

    Environment Variables:
        ARAGORA_ENV: Environment name (production, staging, development)
        ARAGORA_LOG_FORMAT: Log format (json or text)
        ARAGORA_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        True if JSON logging was enabled, False for text logging
    """
    try:
        from aragora.server.middleware.structured_logging import configure_structured_logging

        env = os.environ.get("ARAGORA_ENV", "development")
        log_format = os.environ.get("ARAGORA_LOG_FORMAT", "")
        log_level = os.environ.get("ARAGORA_LOG_LEVEL", "INFO")

        # Use JSON format in production or if explicitly set
        use_json = log_format == "json" or (not log_format and env == "production")

        configure_structured_logging(
            level=log_level,
            json_output=use_json,
            service_name="aragora",
        )

        if use_json:
            logger.info("Structured JSON logging enabled")
        else:
            logger.debug("Text logging enabled (set ARAGORA_LOG_FORMAT=json for JSON)")

        return use_json
    except ImportError as e:
        logger.debug(f"Structured logging not available: {e}")
    except (ValueError, TypeError, OSError, RuntimeError) as e:
        logger.warning(f"Failed to initialize structured logging: {e}")
    return False


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

    Tries the unified otel.py setup first for consistent configuration,
    then falls back to the legacy tracing.py initializer.

    Returns:
        True if tracing was enabled, False otherwise
    """
    # Try unified setup first
    try:
        from aragora.observability.otel import setup_otel, is_initialized

        if setup_otel():
            logger.info("OpenTelemetry tracing enabled via unified setup")
            return True
        elif is_initialized():
            logger.debug("OpenTelemetry already initialized")
            return True
    except ImportError:
        logger.debug("Unified OTel setup not available, trying legacy path")
    except (ValueError, TypeError, OSError, RuntimeError) as e:
        logger.warning("Unified OTel setup failed: %s", e)

    # Fallback to legacy tracing initializer
    try:
        from aragora.observability.config import is_tracing_enabled
        from aragora.observability.tracing import get_tracer

        if is_tracing_enabled():
            get_tracer()  # Initialize tracer singleton
            logger.info("OpenTelemetry tracing enabled (legacy path)")
            return True
        else:
            logger.debug("OpenTelemetry tracing disabled (set OTEL_ENABLED=true to enable)")
    except ImportError as e:
        logger.debug("OpenTelemetry not available: %s", e)
    except (ValueError, TypeError, OSError, RuntimeError) as e:
        logger.warning("Failed to initialize OpenTelemetry: %s", e)
    return False


async def init_otlp_exporter() -> bool:
    """Initialize OpenTelemetry OTLP exporter for distributed tracing.

    Configures trace export to external backends like Jaeger, Zipkin,
    or Datadog via the OTLP protocol. This function supports both standard
    OpenTelemetry environment variables and Aragora-specific ones.

    Standard OpenTelemetry Variables (recommended):
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (e.g., http://localhost:4317)
        OTEL_SERVICE_NAME: Service name for traces (default: aragora)
        OTEL_TRACES_SAMPLER: Sampler type (parentbased_traceidratio, etc.)
        OTEL_TRACES_SAMPLER_ARG: Sampler argument (e.g., 0.1 for 10% sampling)

    Aragora-specific Variables (fallback):
        ARAGORA_OTLP_EXPORTER: Exporter type (none, jaeger, zipkin, otlp_grpc, otlp_http, datadog)
        ARAGORA_OTLP_ENDPOINT: Collector endpoint URL
        ARAGORA_SERVICE_NAME: Service name for traces (default: aragora)
        ARAGORA_ENVIRONMENT: Deployment environment (default: development)
        ARAGORA_TRACE_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)
        See docs/ENVIRONMENT.md for full configuration reference.

    Returns:
        True if OTLP exporter was configured, False otherwise
    """
    # First try the new OTEL bridge (supports standard OTEL_* variables)
    try:
        from aragora.server.middleware.otel_bridge import (
            get_bridge_config,
            init_otel_bridge,
        )

        config = get_bridge_config()
        if config.enabled:
            if init_otel_bridge(config):
                logger.info(
                    f"OTLP bridge initialized: endpoint={config.endpoint}, "
                    f"service={config.service_name}, sampler={config.sampler_type.value}"
                )
                return True
    except ImportError as e:
        logger.debug(f"OTEL bridge not available: {e}")
    except (ValueError, TypeError, OSError, RuntimeError, ConnectionError) as e:
        logger.debug(f"OTEL bridge initialization failed: {e}")

    # Fall back to legacy OTLP exporter
    try:
        from aragora.observability.config import is_otlp_enabled
        from aragora.observability.otlp_export import configure_otlp_exporter, get_otlp_config

        if not is_otlp_enabled():
            logger.debug(
                "OTLP exporter disabled. Set OTEL_EXPORTER_OTLP_ENDPOINT or "
                "ARAGORA_OTLP_EXPORTER to enable distributed tracing."
            )
            return False

        otlp_config = get_otlp_config()
        provider = configure_otlp_exporter(otlp_config)

        if provider:
            logger.info(
                f"OTLP exporter initialized: type={otlp_config.exporter_type.value}, "
                f"endpoint={otlp_config.get_effective_endpoint()}"
            )
            return True
        else:
            logger.warning("OTLP exporter configuration failed")
            return False

    except ImportError as e:
        logger.debug(f"OTLP exporter not available: {e}")
    except (ValueError, TypeError, OSError, RuntimeError, ConnectionError) as e:
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
    except (ValueError, TypeError, OSError, RuntimeError) as e:
        logger.warning(f"Failed to start metrics server: {e}")
    return False
