"""
KMMetricsHealthBridge — wires ConnectionHealthMonitor into KMMetrics.

The async paths of ``ConnectionHealthMonitor`` (now well-tested via
PR #6784) produce ``HealthStatus`` snapshots: ``healthy``,
``consecutive_failures``, ``latency_ms``. ``KMMetrics`` separately
exposes ``get_health()``, but it has no awareness of underlying
connection health. This bridge module periodically polls the monitor
and feeds each snapshot into ``KMMetrics.record(...)`` as an
operation sample so that connection latency and connection failures
appear inside the existing health report alongside query / store /
cache metrics.

This is the first deliverable proposed by P4 in
``docs/plans/2026-04-28-p4-first-deliverable-km-observability.md``.

Design constraints
------------------

- **Additive only**: this module does not modify ``KMMetrics`` or
  ``ConnectionHealthMonitor``. It depends on the public surface of
  both and on no private attribute.
- **Cancel-safe**: the bridge owns one ``asyncio.Task``. ``stop()``
  cancels it and awaits cancellation, never raising.
- **Bounded**: each tick performs at most one ``check_health()`` call
  and one ``record()`` call. Exceptions are logged and absorbed —
  bridge failures must not break the underlying monitor or metrics.
- **No I/O outside what the monitor and metrics already do**: this
  module never touches the file system, network, gh, or git.
- **Operation type reuse**: connection health samples are recorded as
  ``OperationType.QUERY`` (a real DB round-trip) so that latency rolls
  into the existing query-latency thresholds without needing a new
  enum value or a metrics schema migration. This is documented and
  tested.

Public surface
--------------

- :class:`KMMetricsHealthBridge`
- :func:`build_bridge`
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from aragora.knowledge.mound.metrics import KMMetrics, OperationType
from aragora.knowledge.mound.resilience.health import (
    ConnectionHealthMonitor,
    HealthStatus,
)

logger = logging.getLogger(__name__)


__all__ = [
    "KMMetricsHealthBridge",
    "HealthSnapshotSink",
    "build_bridge",
]


class HealthSnapshotSink(Protocol):
    """Minimal interface a sink must implement.

    Used to keep ``KMMetricsHealthBridge`` testable without coupling
    to the full ``KMMetrics`` API surface. ``KMMetrics`` itself
    satisfies this protocol via its ``record`` method.
    """

    def record(
        self,
        operation: OperationType,
        latency_ms: float,
        success: bool = True,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class _BridgeConfig:
    interval_s: float
    operation_type: OperationType


class KMMetricsHealthBridge:
    """Periodically poll a ConnectionHealthMonitor and record each
    HealthStatus snapshot into a KMMetrics-shaped sink.

    Lifecycle mirrors ``ConnectionHealthMonitor``:

    >>> bridge = KMMetricsHealthBridge(monitor, sink)
    >>> await bridge.start()
    ... # ... bridge polls in the background ...
    >>> await bridge.stop()

    A single bridge instance owns at most one task. Calling
    ``start()`` while already running is a no-op.
    """

    def __init__(
        self,
        monitor: ConnectionHealthMonitor,
        sink: HealthSnapshotSink,
        *,
        interval_s: float = 10.0,
        operation_type: OperationType = OperationType.QUERY,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s!r}")
        self._monitor = monitor
        self._sink = sink
        self._config = _BridgeConfig(interval_s=interval_s, operation_type=operation_type)
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._tick_count = 0

    @property
    def tick_count(self) -> int:
        """Number of completed poll cycles. Exposed for testing."""
        return self._tick_count

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the background polling task. Idempotent."""
        if self.is_running:
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "KMMetricsHealthBridge started (interval=%.2fs, op=%s)",
            self._config.interval_s,
            self._config.operation_type.value,
        )

    async def stop(self) -> None:
        """Stop the background polling task. Idempotent and cancel-safe."""
        if self._task is None:
            return
        self._stopped.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            logger.info("KMMetricsHealthBridge stopped")

    async def tick_once(self) -> HealthStatus:
        """Perform a single poll cycle without scheduling.

        Useful in tests and in deterministic call sites that want to
        force a sample on demand. Returns the ``HealthStatus`` that
        was recorded.
        """
        status = await self._monitor.check_health()
        self._record_snapshot(status)
        self._tick_count += 1
        return status

    def _record_snapshot(self, status: HealthStatus) -> None:
        """Translate a HealthStatus into a metrics sample."""
        try:
            self._sink.record(
                operation=self._config.operation_type,
                latency_ms=status.latency_ms or 0.0,
                success=status.healthy,
                error=status.last_error,
                metadata={
                    "source": "connection_health_monitor",
                    "consecutive_failures": status.consecutive_failures,
                    "last_check": status.last_check.isoformat(),
                },
            )
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("KMMetricsHealthBridge: sink.record raised: %s", exc)

    async def _poll_loop(self) -> None:
        """Background poll loop. Cancelled via ``stop()``."""
        while not self._stopped.is_set():
            try:
                await asyncio.sleep(self._config.interval_s)
                if self._stopped.is_set():
                    break
                status = await self._monitor.check_health()
                self._record_snapshot(status)
                self._tick_count += 1
            except asyncio.CancelledError:
                break
            except (ConnectionError, TimeoutError, OSError) as exc:
                logger.debug("Bridge poll error (expected): %s", exc)
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                logger.warning("Bridge poll error (unexpected): %s", exc)


def build_bridge(
    monitor: ConnectionHealthMonitor,
    metrics: KMMetrics,
    *,
    interval_s: float = 10.0,
) -> KMMetricsHealthBridge:
    """Convenience factory for the common case where the sink is a
    full :class:`KMMetrics` instance.

    Returns an unstarted bridge; the caller is responsible for
    invoking ``await bridge.start()`` and ``await bridge.stop()``.
    """
    return KMMetricsHealthBridge(monitor=monitor, sink=metrics, interval_s=interval_s)
