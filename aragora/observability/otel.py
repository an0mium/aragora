"""
Unified OpenTelemetry setup for Aragora.

Provides a single entry point for configuring distributed tracing with
OpenTelemetry. This module coordinates the TracerProvider, span processors,
resource attributes, context propagation, and span enrichment.

All OpenTelemetry dependencies are optional. When not installed, the module
provides no-op implementations that are safe to call without side effects.

Configuration is driven by environment variables:
    OTEL_ENABLED: Set to "true" to enable tracing (auto-enabled if OTEL_EXPORTER_OTLP_ENDPOINT is set)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (e.g., http://localhost:4317)
    OTEL_SERVICE_NAME: Service name for traces (default: aragora)
    OTEL_TRACES_SAMPLER: Sampler type (default: parentbased_traceidratio)
    OTEL_TRACES_SAMPLER_ARG: Sampler argument, e.g. ratio (default: 1.0)
    OTEL_PROPAGATORS: Context propagators (default: tracecontext,baggage)
    ARAGORA_ENVIRONMENT: Deployment environment (default: development)
    ARAGORA_SERVICE_VERSION: Service version (default: 1.0.0)
    ARAGORA_OTEL_DEV_MODE: Use SimpleSpanProcessor for immediate export in development (default: false)
    ARAGORA_OTLP_BATCH_SIZE: Batch processor queue size (default: 512)
    ARAGORA_OTLP_EXPORT_TIMEOUT_MS: Export timeout in milliseconds (default: 30000)

Usage:
    from aragora.observability.otel import setup_otel, shutdown_otel, get_tracer

    # At application startup
    setup_otel()

    # Get a tracer for instrumentation
    tracer = get_tracer("aragora.debate")

    # At application shutdown
    shutdown_otel()

See docs/OBSERVABILITY.md for full configuration reference.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# State
# =============================================================================

_initialized: bool = False
_tracer_provider: Any = None
_propagator: Any = None
_tracers: dict[str, Any] = {}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OTelConfig:
    """Unified OpenTelemetry configuration.

    Consolidates all tracing configuration into a single dataclass.
    Supports both standard OTEL_* and ARAGORA_* environment variables.

    Attributes:
        enabled: Whether OTel tracing is enabled
        endpoint: OTLP collector endpoint
        service_name: Service name for resource attributes
        service_version: Service version for resource attributes
        environment: Deployment environment (development, staging, production)
        sampler_type: OTel sampler type string
        sample_rate: Sampling ratio (0.0 to 1.0)
        propagators: List of context propagator names
        dev_mode: Use SimpleSpanProcessor for immediate span export (useful in development)
        batch_size: Batch span processor max export batch size
        export_timeout_ms: Batch span processor export timeout in milliseconds
        insecure: Allow insecure gRPC connections
        additional_resource_attrs: Extra resource attributes to include
    """

    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    service_name: str = "aragora"
    service_version: str = "1.0.0"
    environment: str = "development"
    sampler_type: str = "parentbased_traceidratio"
    sample_rate: float = 1.0
    propagators: list[str] = field(default_factory=lambda: ["tracecontext", "baggage"])
    dev_mode: bool = False
    batch_size: int = 512
    export_timeout_ms: int = 30000
    insecure: bool = False
    additional_resource_attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be between 0.0 and 1.0, got {self.sample_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.export_timeout_ms < 1:
            raise ValueError(f"export_timeout_ms must be positive, got {self.export_timeout_ms}")

    @classmethod
    def from_env(cls) -> OTelConfig:
        """Create configuration from environment variables.

        Prioritizes standard OTEL_* variables over ARAGORA_* ones.

        Returns:
            OTelConfig instance
        """
        # Determine if enabled
        otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        aragora_exporter = os.environ.get("ARAGORA_OTLP_EXPORTER", "none").lower()
        explicit_enabled = os.environ.get("OTEL_ENABLED", "").lower() in ("true", "1", "yes")
        auto_enabled = bool(otel_endpoint) or (aragora_exporter != "none")
        enabled = explicit_enabled or auto_enabled

        # Endpoint
        endpoint = (
            otel_endpoint or os.environ.get("ARAGORA_OTLP_ENDPOINT") or "http://localhost:4317"
        )

        # Service identity
        service_name = (
            os.environ.get("OTEL_SERVICE_NAME")
            or os.environ.get("ARAGORA_SERVICE_NAME")
            or "aragora"
        )
        service_version = os.environ.get("ARAGORA_SERVICE_VERSION", "1.0.0")
        environment = os.environ.get("ARAGORA_ENVIRONMENT", "development")

        # Sampling
        sampler_type = os.environ.get("OTEL_TRACES_SAMPLER", "parentbased_traceidratio")
        sample_rate_str = (
            os.environ.get("OTEL_TRACES_SAMPLER_ARG")
            or os.environ.get("OTEL_SAMPLE_RATE")
            or os.environ.get("ARAGORA_TRACE_SAMPLE_RATE")
            or "1.0"
        )
        try:
            sample_rate = float(sample_rate_str)
            if not 0.0 <= sample_rate <= 1.0:
                logger.warning("Sample rate %s out of range, using 1.0", sample_rate)
                sample_rate = 1.0
        except ValueError:
            logger.warning("Invalid sample rate '%s', using 1.0", sample_rate_str)
            sample_rate = 1.0

        # Propagators
        propagators_str = os.environ.get("OTEL_PROPAGATORS", "tracecontext,baggage")
        propagators = [p.strip() for p in propagators_str.split(",") if p.strip()]

        # Dev mode (SimpleSpanProcessor for immediate export)
        dev_mode = os.environ.get("ARAGORA_OTEL_DEV_MODE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # Batch processor settings
        batch_size = int(os.environ.get("ARAGORA_OTLP_BATCH_SIZE", "512"))
        export_timeout_ms = int(os.environ.get("ARAGORA_OTLP_EXPORT_TIMEOUT_MS", "30000"))

        # Connection security
        insecure = os.environ.get("ARAGORA_OTLP_INSECURE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # Additional resource attributes from OTEL_RESOURCE_ATTRIBUTES
        additional_attrs: dict[str, str] = {}
        otel_resource_attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        if otel_resource_attrs:
            for pair in otel_resource_attrs.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    additional_attrs[k.strip()] = v.strip()

        return cls(
            enabled=enabled,
            endpoint=endpoint,
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            sampler_type=sampler_type,
            sample_rate=sample_rate,
            propagators=propagators,
            dev_mode=dev_mode,
            batch_size=batch_size,
            export_timeout_ms=export_timeout_ms,
            insecure=insecure,
            additional_resource_attrs=additional_attrs,
        )


# =============================================================================
# Sampler factory
# =============================================================================


def _create_sampler(config: OTelConfig) -> Any:
    """Create an OpenTelemetry sampler from configuration.

    Args:
        config: OTel configuration

    Returns:
        A sampler instance, or None if OTel SDK is not available
    """
    try:
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_OFF,
            ALWAYS_ON,
            ParentBased,
            TraceIdRatioBased,
        )
    except ImportError:
        return None

    sampler_map = {
        "always_on": lambda: ALWAYS_ON,
        "always_off": lambda: ALWAYS_OFF,
        "traceidratio": lambda: TraceIdRatioBased(config.sample_rate),
        "parentbased_always_on": lambda: ParentBased(root=ALWAYS_ON),
        "parentbased_always_off": lambda: ParentBased(root=ALWAYS_OFF),
        "parentbased_traceidratio": lambda: ParentBased(root=TraceIdRatioBased(config.sample_rate)),
    }

    factory = sampler_map.get(config.sampler_type.lower())
    if factory is None:
        logger.warning(
            "Unknown sampler type '%s', falling back to parentbased_traceidratio. "
            "Valid options: %s",
            config.sampler_type,
            list(sampler_map.keys()),
        )
        return ParentBased(root=TraceIdRatioBased(config.sample_rate))

    return factory()


# =============================================================================
# Setup and shutdown
# =============================================================================


def setup_otel(config: OTelConfig | None = None) -> bool:
    """Initialize OpenTelemetry tracing infrastructure.

    Sets up the TracerProvider, span processor, resource attributes,
    and context propagation. This function is idempotent -- calling it
    multiple times has no additional effect after the first successful call.

    Uses BatchSpanProcessor in production for efficient batched export,
    and SimpleSpanProcessor in dev mode (ARAGORA_OTEL_DEV_MODE=true) for
    immediate span visibility during development.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        True if OTel was initialized successfully, False otherwise.
    """
    global _initialized, _tracer_provider, _propagator

    if _initialized:
        logger.debug("OpenTelemetry already initialized, skipping setup")
        return True

    config = config or OTelConfig.from_env()

    if not config.enabled:
        logger.debug(
            "OpenTelemetry tracing disabled. Set OTEL_ENABLED=true or "
            "OTEL_EXPORTER_OTLP_ENDPOINT to enable."
        )
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
    except ImportError as e:
        logger.warning(
            "OpenTelemetry SDK not installed, tracing disabled: %s. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-grpc",
            e,
        )
        return False

    try:
        # Build resource attributes
        resource_attrs: dict[str, str] = {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
            "telemetry.sdk.language": "python",
        }
        resource_attrs.update(config.additional_resource_attrs)

        resource = Resource.create(resource_attrs)

        # Create sampler
        sampler = _create_sampler(config)

        # Create TracerProvider
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Create exporter
        exporter = OTLPSpanExporter(
            endpoint=config.endpoint,
            insecure=config.insecure,
        )

        # Choose processor based on mode
        processor: SimpleSpanProcessor | BatchSpanProcessor
        if config.dev_mode:
            processor = SimpleSpanProcessor(exporter)
            processor_type = "SimpleSpanProcessor (dev mode)"
        else:
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=config.batch_size * 2,
                max_export_batch_size=config.batch_size,
                export_timeout_millis=config.export_timeout_ms,
            )
            processor_type = "BatchSpanProcessor"

        provider.add_span_processor(processor)

        # Set as global provider
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        # Set up context propagation
        _propagator = _setup_propagators(config.propagators)

        _initialized = True

        logger.info(
            "OpenTelemetry initialized: endpoint=%s, service=%s/%s, "
            "environment=%s, sampler=%s(%.2f), processor=%s",
            config.endpoint,
            config.service_name,
            config.service_version,
            config.environment,
            config.sampler_type,
            config.sample_rate,
            processor_type,
        )
        return True

    except Exception as e:
        logger.error("Failed to initialize OpenTelemetry: %s", e)
        return False


def _setup_propagators(propagator_names: list[str]) -> Any:
    """Set up context propagators.

    Args:
        propagator_names: List of propagator names (e.g., ["tracecontext", "baggage"])

    Returns:
        CompositePropagator or None
    """
    try:
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.propagators.composite import CompositePropagator
    except ImportError:
        return None

    propagators: list[Any] = []

    if "tracecontext" in propagator_names:
        try:
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            propagators.append(TraceContextTextMapPropagator())
        except ImportError:
            logger.debug("W3C TraceContext propagator not available")

    if "baggage" in propagator_names:
        try:
            from opentelemetry.baggage.propagation import W3CBaggagePropagator

            propagators.append(W3CBaggagePropagator())
        except ImportError:
            logger.debug("W3C Baggage propagator not available")

    if propagators:
        composite = CompositePropagator(propagators)
        set_global_textmap(composite)
        return composite

    return None


def shutdown_otel() -> None:
    """Shutdown OpenTelemetry tracing and flush pending spans.

    Ensures all buffered spans are exported before the process exits.
    Safe to call even if OTel was never initialized.
    """
    global _initialized, _tracer_provider, _tracers

    if not _initialized:
        return

    try:
        if _tracer_provider is not None and hasattr(_tracer_provider, "shutdown"):
            _tracer_provider.shutdown()
            logger.info("OpenTelemetry shutdown complete")
    except Exception as e:
        logger.error("Error during OpenTelemetry shutdown: %s", e)
    finally:
        _initialized = False
        _tracer_provider = None
        _tracers.clear()


def is_initialized() -> bool:
    """Check whether OpenTelemetry has been initialized.

    Returns:
        True if setup_otel() completed successfully
    """
    return _initialized


# =============================================================================
# Tracer access
# =============================================================================


def get_tracer(
    instrumentation_name: str = "aragora",
    version: str = "1.0.0",
) -> Any:
    """Get an OpenTelemetry tracer instance.

    Returns a real tracer if OTel is initialized, otherwise returns a
    no-op tracer that silently discards all spans.

    Args:
        instrumentation_name: Name of the instrumentation scope
            (e.g., "aragora.debate", "aragora.agents")
        version: Version of the instrumentation scope

    Returns:
        A tracer instance (real or no-op)
    """
    cache_key = f"{instrumentation_name}:{version}"

    if cache_key in _tracers:
        return _tracers[cache_key]

    if _initialized and _tracer_provider is not None:
        try:
            from opentelemetry import trace

            otel_tracer = trace.get_tracer(instrumentation_name, version)
            _tracers[cache_key] = otel_tracer
            return otel_tracer
        except Exception as e:
            logger.debug("Failed to get OTel tracer: %s", e)

    noop_tracer: Any = _NoOpTracer()
    _tracers[cache_key] = noop_tracer
    return noop_tracer


# =============================================================================
# Span helpers
# =============================================================================


@contextmanager
def start_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "aragora",
) -> Iterator[Any]:
    """Create a traced span as a context manager.

    This is a convenience wrapper that gets a tracer and creates a span.
    If OTel is not initialized, yields a no-op span.

    Args:
        name: Span name (e.g., "debate.execute", "agent.respond")
        attributes: Optional initial span attributes
        tracer_name: Instrumentation scope name for the tracer

    Yields:
        The span object (real OTel span or NoOpSpan)

    Example:
        with start_span("debate.round", {"debate.id": "abc", "debate.round": 3}) as span:
            result = await run_round()
            span.set_attribute("debate.consensus", result.consensus)
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        if attributes and hasattr(span, "set_attribute"):
            for key, value in attributes.items():
                if value is not None:
                    try:
                        if isinstance(value, (list, dict)):
                            value = str(value)
                        span.set_attribute(key, value)
                    except (TypeError, ValueError) as e:
                        logger.debug("start span encountered an error: %s", e)
        yield span


def record_span_error(span: Any, exception: BaseException) -> None:
    """Record an exception on a span and set error status.

    Safe to call with no-op spans or None.

    Args:
        span: The span to record the error on
        exception: The exception to record
    """
    if span is None:
        return

    try:
        if hasattr(span, "record_exception"):
            span.record_exception(exception)
    except Exception as e:
        logger.debug("Failed to record exception on span: %s", e)

    try:
        from opentelemetry.trace import StatusCode

        if hasattr(span, "set_status"):
            span.set_status(StatusCode.ERROR, str(exception))
    except ImportError:
        # OTel not installed, try the duck-type approach
        try:
            if hasattr(span, "set_status"):
                span.set_status("ERROR")
        except Exception as e:
            logger.debug("Failed to set span error status: %s", e)


def set_span_ok(span: Any) -> None:
    """Set OK status on a span.

    Args:
        span: The span to mark as OK
    """
    if span is None:
        return

    try:
        from opentelemetry.trace import StatusCode

        if hasattr(span, "set_status"):
            span.set_status(StatusCode.OK)
    except ImportError:
        pass


# =============================================================================
# Context propagation helpers
# =============================================================================


def inject_context(carrier: dict[str, str]) -> dict[str, str]:
    """Inject the current trace context into a carrier (e.g., HTTP headers).

    Uses the globally configured propagator. Falls back to manual
    W3C traceparent injection if the propagator is not available.

    Args:
        carrier: A mutable dictionary to inject trace context into

    Returns:
        The carrier with trace context headers added

    Example:
        headers = {"Authorization": "Bearer token"}
        inject_context(headers)
        response = await client.post(url, headers=headers)
    """
    if _propagator is not None:
        try:
            from opentelemetry import context

            _propagator.inject(carrier, context.get_current())
            return carrier
        except Exception as e:
            logger.debug("Failed to inject trace context via propagator: %s", e)

    # Fall back to manual injection from internal tracing
    try:
        from aragora.server.middleware.tracing import get_trace_id, get_span_id

        trace_id = get_trace_id()
        span_id = get_span_id()

        if trace_id:
            carrier["X-Trace-ID"] = trace_id
            parent_id = span_id or "0000000000000000"
            trace_id_padded = trace_id.ljust(32, "0")[:32]
            parent_id_padded = parent_id.ljust(16, "0")[:16]
            carrier["traceparent"] = f"00-{trace_id_padded}-{parent_id_padded}-01"
    except ImportError:
        pass

    return carrier


def extract_context(carrier: dict[str, str]) -> Any:
    """Extract trace context from a carrier (e.g., incoming HTTP headers).

    Uses the globally configured propagator. Returns None if no context
    can be extracted.

    Args:
        carrier: A dictionary containing trace context headers

    Returns:
        An OpenTelemetry context, or None
    """
    if _propagator is not None:
        try:
            return _propagator.extract(carrier)
        except Exception as e:
            logger.debug("Failed to extract trace context: %s", e)

    return None


# =============================================================================
# Debate-specific span helpers
# =============================================================================


@contextmanager
def trace_debate_lifecycle(
    debate_id: str,
    task: str = "",
    agent_count: int = 0,
    protocol_rounds: int = 0,
) -> Iterator[Any]:
    """Create a root span for the entire debate lifecycle.

    This span encompasses the full debate from initiation to result.
    Child spans should be created for individual phases and agent calls.

    Args:
        debate_id: Unique debate identifier
        task: The debate task/question
        agent_count: Number of participating agents
        protocol_rounds: Configured number of rounds

    Yields:
        The debate root span

    Example:
        with trace_debate_lifecycle("debate-123", task="Design a rate limiter", agent_count=5) as span:
            result = await arena.run()
            span.set_attribute("debate.consensus_reached", result.consensus_reached)
    """
    attrs = {
        "debate.id": debate_id,
        "debate.agent_count": agent_count,
        "debate.protocol_rounds": protocol_rounds,
    }
    if task:
        # Truncate long tasks to avoid huge span attributes
        attrs["debate.task"] = task[:500]

    with start_span("debate.lifecycle", attrs, tracer_name="aragora.debate") as span:
        yield span


@contextmanager
def trace_debate_round(
    debate_id: str,
    round_number: int,
) -> Iterator[Any]:
    """Create a span for a single debate round.

    Args:
        debate_id: Debate identifier
        round_number: Current round number (1-indexed)

    Yields:
        The round span
    """
    with start_span(
        "debate.round",
        {
            "debate.id": debate_id,
            "debate.round_number": round_number,
        },
        tracer_name="aragora.debate",
    ) as span:
        yield span


@contextmanager
def trace_agent_operation(
    agent_name: str,
    operation: str,
    debate_id: str | None = None,
    round_number: int | None = None,
    model: str | None = None,
) -> Iterator[Any]:
    """Create a span for an agent operation (propose, critique, vote, etc.).

    Args:
        agent_name: Name of the agent
        operation: Operation type (propose, critique, vote, revise, etc.)
        debate_id: Optional debate identifier for correlation
        round_number: Optional round number
        model: Optional model name/identifier

    Yields:
        The agent operation span
    """
    attrs: dict[str, Any] = {
        "agent.name": agent_name,
        "agent.operation": operation,
    }
    if debate_id:
        attrs["debate.id"] = debate_id
    if round_number is not None:
        attrs["debate.round_number"] = round_number
    if model:
        attrs["agent.model"] = model

    with start_span(
        f"agent.{operation}",
        attrs,
        tracer_name="aragora.agents",
    ) as span:
        yield span


@contextmanager
def trace_consensus_evaluation(
    debate_id: str,
    round_number: int,
    method: str = "majority",
) -> Iterator[Any]:
    """Create a span for consensus evaluation.

    Args:
        debate_id: Debate identifier
        round_number: Round being evaluated
        method: Consensus method (majority, unanimous, convergence, etc.)

    Yields:
        The consensus span
    """
    with start_span(
        "debate.consensus",
        {
            "debate.id": debate_id,
            "debate.round_number": round_number,
            "debate.consensus_method": method,
        },
        tracer_name="aragora.debate",
    ) as span:
        yield span


# =============================================================================
# Bridge: export internal debate/tracing spans to OTel
# =============================================================================


def export_debate_span_to_otel(span: Any) -> None:
    """Export a debate.tracing.Span to OpenTelemetry.

    Bridges spans from aragora.debate.tracing.Span (the internal debate
    tracing system) into the OTel pipeline for unified export.

    Args:
        span: An aragora.debate.tracing.Span instance
    """
    if not _initialized or _tracer_provider is None:
        return

    try:
        tracer = get_tracer("aragora.debate.bridge")
        with tracer.start_as_current_span(
            getattr(span, "name", "unknown"),
            start_time=int(getattr(span, "start_time", 0) * 1e9),
        ) as otel_span:
            # Copy attributes
            for key, value in getattr(span, "attributes", {}).items():
                if value is not None:
                    try:
                        if isinstance(value, (list, dict)):
                            value = str(value)
                        otel_span.set_attribute(key, value)
                    except (TypeError, ValueError) as e:
                        logger.debug("export debate span to otel encountered an error: %s", e)

            # Copy trace context
            if hasattr(span, "trace_id"):
                otel_span.set_attribute("aragora.trace_id", span.trace_id)
            if hasattr(span, "span_id"):
                otel_span.set_attribute("aragora.span_id", span.span_id)
            if hasattr(span, "parent_span_id") and span.parent_span_id:
                otel_span.set_attribute("aragora.parent_span_id", span.parent_span_id)

            # Copy events
            for event in getattr(span, "events", []):
                otel_span.add_event(
                    event.get("name", "event"),
                    attributes=event.get("attributes", {}),
                )

            # Set status
            status = getattr(span, "status", "OK")
            if status == "ERROR" or status == "error":
                record_span_error(otel_span, RuntimeError(getattr(span, "error", "Unknown error")))
            else:
                set_span_ok(otel_span)

    except Exception as e:
        logger.debug("Failed to export debate span to OTel: %s", e)


# =============================================================================
# No-op implementations
# =============================================================================


class _NoOpSpan:
    """No-op span for when OpenTelemetry is not available."""

    def set_attribute(self, key: str, value: Any) -> _NoOpSpan:
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> _NoOpSpan:
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> _NoOpSpan:
        return self

    def record_exception(self, exception: BaseException, **kwargs: Any) -> _NoOpSpan:
        return self

    def set_status(self, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return self

    def update_name(self, name: str) -> _NoOpSpan:
        return self

    def end(self, end_time: Any = None) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def get_span_context(self) -> _NoOpSpanContext:
        return _NoOpSpanContext()

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpSpanContext:
    """No-op span context."""

    trace_id: int = 0
    span_id: int = 0
    is_valid: bool = False
    is_remote: bool = False
    trace_flags: int = 0
    trace_state: Any = None


class _NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


# =============================================================================
# Reset (for testing)
# =============================================================================


def reset_otel() -> None:
    """Reset OTel state for testing purposes.

    This is intended only for use in test suites, not production code.
    """
    global _initialized, _tracer_provider, _propagator, _tracers
    _initialized = False
    _tracer_provider = None
    _propagator = None
    _tracers.clear()


__all__ = [
    # Configuration
    "OTelConfig",
    # Setup/shutdown
    "setup_otel",
    "shutdown_otel",
    "is_initialized",
    "reset_otel",
    # Tracer access
    "get_tracer",
    # Span helpers
    "start_span",
    "record_span_error",
    "set_span_ok",
    # Context propagation
    "inject_context",
    "extract_context",
    # Debate-specific tracing
    "trace_debate_lifecycle",
    "trace_debate_round",
    "trace_agent_operation",
    "trace_consensus_evaluation",
    # Bridge
    "export_debate_span_to_otel",
    # No-op implementations (for testing)
    "_NoOpTracer",
    "_NoOpSpan",
]
