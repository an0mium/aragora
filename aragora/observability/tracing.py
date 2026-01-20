"""
OpenTelemetry distributed tracing for Aragora.

Provides decorators and utilities for instrumenting API handlers
and agent calls with distributed tracing spans.

Usage:
    from aragora.observability.tracing import tracer, trace_handler

    @trace_handler("debates.create")
    def handle_create_debate(self, handler):
        with tracer.start_as_current_span("validate_input"):
            ...

Requirements:
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

Environment Variables:
    OTEL_ENABLED: Set to "true" to enable tracing
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint

See docs/OBSERVABILITY.md for configuration guide.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar, cast

from aragora.observability.config import get_tracing_config

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# OpenTelemetry imports - only imported when tracing is enabled
_tracer = None
_tracer_provider = None


def _init_tracer() -> Any:
    """Initialize OpenTelemetry tracer lazily."""
    global _tracer, _tracer_provider

    if _tracer is not None:
        return _tracer

    config = get_tracing_config()

    if not config.enabled:
        # Return a no-op tracer
        return _NoOpTracer()

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: config.service_name})

        # Create sampler based on sample rate
        sampler = TraceIdRatioBased(config.sample_rate)

        # Create provider with sampler and resource
        _tracer_provider = TracerProvider(sampler=sampler, resource=resource)

        # Create exporter and processor
        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=config.batch_size * 2,
            max_export_batch_size=config.batch_size,
            export_timeout_millis=config.export_timeout_ms,
        )
        _tracer_provider.add_span_processor(processor)

        # Set as global provider
        trace.set_tracer_provider(_tracer_provider)

        # Get tracer
        _tracer = trace.get_tracer("aragora", "1.0.0")

        logger.info(
            f"OpenTelemetry tracing initialized: endpoint={config.otlp_endpoint}, "
            f"service={config.service_name}, sample_rate={config.sample_rate}"
        )

        return _tracer

    except ImportError as e:
        logger.warning(
            f"OpenTelemetry not installed, tracing disabled: {e}. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return _NoOpTracer()
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        return _NoOpTracer()


class _NoOpSpan:
    """No-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> "_NoOpSpan":
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "_NoOpSpan":
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "_NoOpSpan":
        return self

    def record_exception(self, exception: BaseException) -> "_NoOpSpan":
        return self

    def set_status(self, status: Any) -> "_NoOpSpan":
        return self

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer for when tracing is disabled."""

    def start_as_current_span(
        self,
        name: str,
        **kwargs: Any,
    ) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


def get_tracer() -> Any:
    """Get the OpenTelemetry tracer instance."""
    return _init_tracer()


@contextmanager
def create_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Create a span context manager.

    Args:
        name: Span name
        attributes: Optional span attributes

    Yields:
        The span object (or NoOpSpan if tracing disabled)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            add_span_attributes(span, attributes)
        yield span


def trace_handler(name: str) -> Callable[[F], F]:
    """Decorator to trace HTTP handler methods.

    Args:
        name: Span name for the handler (e.g., "debates.create")

    Returns:
        Decorated function with tracing

    Example:
        @trace_handler("debates.list")
        def handle_list_debates(self, handler):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, handler: Any, *args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                # Add HTTP attributes
                if hasattr(handler, "path"):
                    span.set_attribute("http.path", handler.path)
                if hasattr(handler, "command"):
                    span.set_attribute("http.method", handler.command)
                if hasattr(handler, "client_address"):
                    span.set_attribute("net.peer.ip", str(handler.client_address[0]))

                try:
                    result = func(self, handler, *args, **kwargs)

                    # Add response attributes
                    if hasattr(result, "status_code"):
                        span.set_attribute("http.status_code", result.status_code)

                    return result
                except Exception as e:
                    span.record_exception(e)
                    _set_error_status(span)
                    raise

        return cast(F, wrapper)

    return decorator


def trace_async_handler(name: str) -> Callable[[F], F]:
    """Decorator to trace async HTTP handler methods.

    Args:
        name: Span name for the handler

    Returns:
        Decorated async function with tracing
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(self: Any, handler: Any, *args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                if hasattr(handler, "path"):
                    span.set_attribute("http.path", handler.path)
                if hasattr(handler, "command"):
                    span.set_attribute("http.method", handler.command)

                try:
                    result = await func(self, handler, *args, **kwargs)
                    if hasattr(result, "status_code"):
                        span.set_attribute("http.status_code", result.status_code)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    _set_error_status(span)
                    raise

        return cast(F, wrapper)

    return decorator


def trace_agent_call(agent_name: str) -> Callable[[F], F]:
    """Decorator to trace agent API calls.

    Args:
        agent_name: Name of the agent being called

    Returns:
        Decorated async function with tracing

    Example:
        @trace_agent_call("anthropic")
        async def respond(self, prompt: str) -> str:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(f"agent.{agent_name}") as span:
                span.set_attribute("agent.name", agent_name)

                # Extract prompt if available
                if args and len(args) > 1:
                    prompt = args[1] if len(args) > 1 else kwargs.get("prompt", "")
                    if isinstance(prompt, str):
                        span.set_attribute("agent.prompt_length", len(prompt))

                try:
                    result = await func(*args, **kwargs)

                    # Record response length
                    if isinstance(result, str):
                        span.set_attribute("agent.response_length", len(result))

                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("agent.error", str(e))
                    _set_error_status(span)
                    raise

        return cast(F, wrapper)

    return decorator


def add_span_attributes(span: Any, attributes: Dict[str, Any]) -> None:
    """Add attributes to a span safely.

    Args:
        span: The span to add attributes to
        attributes: Dictionary of attribute key-value pairs
    """
    if span is None:
        return

    for key, value in attributes.items():
        if value is not None:
            # Convert complex types to strings
            if isinstance(value, (list, dict)):
                value = str(value)
            try:
                span.set_attribute(key, value)
            except (TypeError, ValueError) as e:
                # Ignore invalid attribute types (e.g., unsupported types)
                logger.debug("Failed to set span attribute %s: %s", key, e)


def record_exception(span: Any, exception: BaseException) -> None:
    """Record an exception on a span.

    Args:
        span: The span to record the exception on
        exception: The exception to record
    """
    if span is not None and hasattr(span, "record_exception"):
        span.record_exception(exception)
        _set_error_status(span)


def _set_error_status(span: Any) -> None:
    """Set error status on a span."""
    if span is None:
        return

    try:
        from opentelemetry.trace import StatusCode

        span.set_status(StatusCode.ERROR)
    except ImportError:
        pass


def shutdown() -> None:
    """Shutdown the tracer provider gracefully."""
    global _tracer_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("OpenTelemetry tracer shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down tracer: {e}")


# =============================================================================
# Debate-Specific Tracing
# =============================================================================


def trace_debate(debate_id: str) -> Callable[[F], F]:
    """Decorator to trace an entire debate lifecycle.

    Args:
        debate_id: The debate ID (can be a function arg name to extract)

    Returns:
        Decorated async function with debate tracing

    Example:
        @trace_debate("debate_id")
        async def run_debate(self, debate_id: str) -> DebateResult:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            # Try to get debate_id from kwargs or args
            did = kwargs.get(debate_id) or (
                args[1] if len(args) > 1 and isinstance(args[1], str) else debate_id
            )

            with tracer.start_as_current_span("debate") as span:
                span.set_attribute("debate.id", str(did))

                try:
                    result = await func(*args, **kwargs)

                    # Record result attributes
                    if hasattr(result, "consensus_reached"):
                        span.set_attribute("debate.consensus_reached", result.consensus_reached)
                    if hasattr(result, "rounds_used"):
                        span.set_attribute("debate.rounds", result.rounds_used)
                    if hasattr(result, "confidence"):
                        span.set_attribute("debate.confidence", result.confidence)

                    return result
                except Exception as e:
                    span.record_exception(e)
                    _set_error_status(span)
                    raise

        return cast(F, wrapper)

    return decorator


@contextmanager
def trace_debate_phase(
    phase_name: str, debate_id: str, round_num: Optional[int] = None
) -> Iterator[Any]:
    """Context manager for tracing a debate phase.

    Args:
        phase_name: Name of the phase (propose, critique, vote, etc.)
        debate_id: The debate ID
        round_num: Optional round number

    Yields:
        The span object

    Example:
        with trace_debate_phase("propose", debate_id, round_num=1) as span:
            proposal = await agent.generate(prompt)
            span.set_attribute("proposal.length", len(proposal))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(f"debate.phase.{phase_name}") as span:
        span.set_attribute("debate.id", debate_id)
        span.set_attribute("debate.phase", phase_name)
        if round_num is not None:
            span.set_attribute("debate.round", round_num)
        yield span


@contextmanager
def trace_consensus_check(debate_id: str, round_num: int) -> Iterator[Any]:
    """Context manager for tracing consensus checking.

    Args:
        debate_id: The debate ID
        round_num: Current round number

    Yields:
        The span object
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("debate.consensus_check") as span:
        span.set_attribute("debate.id", debate_id)
        span.set_attribute("debate.round", round_num)
        yield span


def trace_memory_operation(operation: str, tier: str) -> Callable[[F], F]:
    """Decorator to trace memory operations.

    Args:
        operation: Operation type (store, query, promote, demote)
        tier: Memory tier (fast, medium, slow, glacial)

    Returns:
        Decorated function with tracing
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(f"memory.{operation}") as span:
                span.set_attribute("memory.operation", operation)
                span.set_attribute("memory.tier", tier)
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    _set_error_status(span)
                    raise

        return cast(F, wrapper)

    return decorator


# =============================================================================
# Decision Pipeline Tracing
# =============================================================================


@contextmanager
def trace_decision_routing(
    request_id: str,
    decision_type: str,
    source: str,
    priority: str = "normal",
) -> Iterator[Any]:
    """Context manager for tracing decision routing.

    Args:
        request_id: The unique request ID
        decision_type: Type of decision (debate, workflow, gauntlet, quick)
        source: Input source (http_api, slack, voice, etc.)
        priority: Request priority level

    Yields:
        The span object

    Example:
        with trace_decision_routing(request.request_id, "debate", "slack") as span:
            result = await router.route(request)
            span.set_attribute("decision.confidence", result.confidence)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("decision.route") as span:
        span.set_attribute("decision.request_id", request_id)
        span.set_attribute("decision.type", decision_type)
        span.set_attribute("decision.source", source)
        span.set_attribute("decision.priority", priority)
        yield span


@contextmanager
def trace_decision_engine(
    engine_type: str,
    request_id: str,
) -> Iterator[Any]:
    """Context manager for tracing decision engine execution.

    Args:
        engine_type: Type of engine (debate, workflow, gauntlet, quick)
        request_id: The request ID being processed

    Yields:
        The span object
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(f"decision.engine.{engine_type}") as span:
        span.set_attribute("decision.engine", engine_type)
        span.set_attribute("decision.request_id", request_id)
        yield span


@contextmanager
def trace_response_delivery(
    platform: str,
    channel_id: Optional[str] = None,
    voice_enabled: bool = False,
) -> Iterator[Any]:
    """Context manager for tracing response delivery.

    Args:
        platform: Target platform (slack, discord, http, etc.)
        channel_id: Optional channel/destination ID
        voice_enabled: Whether voice response is enabled

    Yields:
        The span object
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("decision.deliver") as span:
        span.set_attribute("delivery.platform", platform)
        if channel_id:
            span.set_attribute("delivery.channel_id", channel_id)
        span.set_attribute("delivery.voice_enabled", voice_enabled)
        yield span


def trace_decision(func: F) -> F:
    """Decorator to trace the entire decision routing lifecycle.

    Automatically extracts request attributes and records result metrics.

    Example:
        @trace_decision
        async def route(self, request: DecisionRequest) -> DecisionResult:
            ...
    """

    @wraps(func)
    async def wrapper(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer()

        # Extract attributes from request
        request_id = getattr(request, "request_id", "unknown")
        decision_type = getattr(request, "decision_type", None)
        decision_type_str = decision_type.value if hasattr(decision_type, "value") else str(decision_type)
        source = getattr(request, "source", None)
        source_str = source.value if hasattr(source, "value") else str(source)
        priority = getattr(request, "priority", None)
        priority_str = priority.value if hasattr(priority, "value") else str(priority)

        with tracer.start_as_current_span("decision.route") as span:
            span.set_attribute("decision.request_id", request_id)
            span.set_attribute("decision.type", decision_type_str)
            span.set_attribute("decision.source", source_str)
            span.set_attribute("decision.priority", priority_str)

            # Extract content length
            content = getattr(request, "content", "")
            if isinstance(content, str):
                span.set_attribute("decision.content_length", len(content))

            # Extract agent config
            config = getattr(request, "config", None)
            if config:
                agents = getattr(config, "agents", [])
                span.set_attribute("decision.agent_count", len(agents))

            try:
                result = await func(self, request, *args, **kwargs)

                # Record result attributes
                if result:
                    if hasattr(result, "confidence"):
                        span.set_attribute("decision.confidence", result.confidence)
                    if hasattr(result, "consensus_reached"):
                        span.set_attribute("decision.consensus_reached", result.consensus_reached)
                    if hasattr(result, "success"):
                        span.set_attribute("decision.success", result.success)
                    if hasattr(result, "duration_seconds"):
                        span.set_attribute("decision.duration_seconds", result.duration_seconds)
                    if hasattr(result, "error") and result.error:
                        span.set_attribute("decision.error", str(result.error)[:200])

                return result

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("decision.error", str(e)[:200])
                _set_error_status(span)
                raise

    return cast(F, wrapper)
