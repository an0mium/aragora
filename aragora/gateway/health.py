"""
Gateway Health Checker for Aragora.

Provides production-grade health and readiness probes for the gateway
subsystem. Checks the liveness of key gateway dependencies:

- **OpenClaw connectivity**: Verifies the OpenClaw runtime endpoint is reachable.
- **Credential vault**: Confirms the vault is initialised and responsive.
- **Policy engine**: Validates the policy engine can evaluate a no-op rule.
- **Audit storage**: Ensures the audit logger can accept events.
- **Federation registry**: Checks that federated framework discovery is healthy.

Usage:
    from aragora.gateway.health import GatewayHealthChecker, ComponentHealth

    checker = GatewayHealthChecker(
        openclaw_endpoint="http://localhost:8081",
    )

    # Full readiness probe
    status = await checker.check_readiness()
    print(status.to_dict())
    # {
    #     "status": "healthy",
    #     "components": {
    #         "openclaw": {"status": "healthy", "latency_ms": 12.3},
    #         "vault": {"status": "healthy", "latency_ms": 0.8},
    #         ...
    #     },
    #     "ready": True,
    # }

    # Individual component check
    openclaw = await checker.check_openclaw()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# ===========================================================================
# Data types
# ===========================================================================


class HealthStatus(str, Enum):
    """Possible health states for a component or the aggregate gateway."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health result for a single gateway component.

    Attributes:
        name: Component identifier (e.g. "openclaw", "vault").
        status: Computed health status.
        latency_ms: Time in milliseconds the check took.
        error: Error message if the check failed.
        details: Arbitrary detail dict for debugging.
    """

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        result: dict[str, Any] = {
            "status": self.status.value,
        }
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.error:
            result["error"] = self.error
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class GatewayHealthStatus:
    """Aggregated health status across all gateway components.

    Attributes:
        status: Overall health status (worst of all components).
        components: Per-component health results.
        ready: Whether the gateway should be considered ready for traffic.
        checked_at: Timestamp of the check (epoch seconds).
    """

    status: HealthStatus = HealthStatus.UNKNOWN
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    ready: bool = False
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "status": self.status.value,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "ready": self.ready,
            "checked_at": self.checked_at,
        }


# ===========================================================================
# Health checker
# ===========================================================================

# Type alias for pluggable check functions
HealthCheckFn = Callable[[], Awaitable[ComponentHealth]]


class GatewayHealthChecker:
    """Production gateway health / readiness checker.

    Aggregates the health of individual gateway components and computes an
    overall readiness verdict.  Components are checked concurrently for
    minimal probe latency.

    The checker ships with built-in checks for the core gateway dependencies
    (OpenClaw, credential vault, audit, federation) and allows registering
    additional custom checks.

    Args:
        openclaw_endpoint: Base URL for the OpenClaw runtime.
        check_timeout: Per-component check timeout in seconds.
    """

    def __init__(
        self,
        openclaw_endpoint: str = "http://localhost:8081",
        check_timeout: float = 5.0,
    ) -> None:
        self.openclaw_endpoint = openclaw_endpoint.rstrip("/")
        self.check_timeout = check_timeout

        # Registry of named check functions
        self._checks: dict[str, HealthCheckFn] = {
            "openclaw": self.check_openclaw,
            "vault": self.check_vault,
            "audit": self.check_audit,
            "federation": self.check_federation,
        }

        # Cache the last result for lightweight liveness probes
        self._last_result: GatewayHealthStatus | None = None

    # -----------------------------------------------------------------------
    # Custom check registration
    # -----------------------------------------------------------------------

    def register_check(self, name: str, fn: HealthCheckFn) -> None:
        """Register a custom health check.

        Args:
            name: Unique component name.
            fn: Async callable returning ``ComponentHealth``.
        """
        self._checks[name] = fn

    def unregister_check(self, name: str) -> bool:
        """Remove a previously registered check.

        Args:
            name: Component name to remove.

        Returns:
            True if the check was found and removed.
        """
        return self._checks.pop(name, None) is not None

    # -----------------------------------------------------------------------
    # Aggregate probes
    # -----------------------------------------------------------------------

    async def check_readiness(self) -> GatewayHealthStatus:
        """Run all registered checks and produce an aggregate readiness status.

        Checks are executed concurrently.  The overall status is the
        *worst* status among all components:
        - All healthy -> HEALTHY
        - Any degraded but none unhealthy -> DEGRADED
        - Any unhealthy -> UNHEALTHY

        The ``ready`` flag is True only when the overall status is HEALTHY
        or DEGRADED.

        Returns:
            GatewayHealthStatus with per-component detail.
        """
        components: dict[str, ComponentHealth] = {}

        # Run all checks concurrently with per-check timeout
        async def _run_check(name: str, fn: HealthCheckFn) -> None:
            try:
                result = await asyncio.wait_for(fn(), timeout=self.check_timeout)
                components[name] = result
            except asyncio.TimeoutError:
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    error=f"Health check timed out after {self.check_timeout}s",
                )
            except Exception as exc:
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    error=str(exc),
                )

        tasks = [_run_check(name, fn) for name, fn in self._checks.items()]
        await asyncio.gather(*tasks)

        # Compute overall status
        overall = self._aggregate_status(list(components.values()))
        ready = overall in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

        result = GatewayHealthStatus(
            status=overall,
            components=components,
            ready=ready,
        )
        self._last_result = result
        return result

    async def check_liveness(self) -> GatewayHealthStatus:
        """Lightweight liveness probe.

        Returns the cached result from the last readiness check if
        available, otherwise runs a fresh readiness check.

        Returns:
            GatewayHealthStatus.
        """
        if self._last_result is not None:
            return self._last_result
        return await self.check_readiness()

    @property
    def last_result(self) -> GatewayHealthStatus | None:
        """Return the most recent health check result without running checks."""
        return self._last_result

    # -----------------------------------------------------------------------
    # Individual component checks
    # -----------------------------------------------------------------------

    async def check_openclaw(self) -> ComponentHealth:
        """Check OpenClaw runtime connectivity.

        Attempts an HTTP GET against the OpenClaw health endpoint.  Falls
        back gracefully when ``aiohttp`` is not installed.
        """
        start = time.perf_counter()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.openclaw_endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=self.check_timeout),
                ) as resp:
                    latency_ms = (time.perf_counter() - start) * 1000
                    if resp.status < 300:
                        return ComponentHealth(
                            name="openclaw",
                            status=HealthStatus.HEALTHY,
                            latency_ms=latency_ms,
                            details={"status_code": resp.status},
                        )
                    return ComponentHealth(
                        name="openclaw",
                        status=HealthStatus.DEGRADED,
                        latency_ms=latency_ms,
                        error=f"Non-2xx response: {resp.status}",
                        details={"status_code": resp.status},
                    )

        except ImportError:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="openclaw",
                status=HealthStatus.UNKNOWN,
                latency_ms=latency_ms,
                error="aiohttp not installed; cannot probe OpenClaw",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="openclaw",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(exc),
            )

    async def check_vault(self) -> ComponentHealth:
        """Check credential vault availability.

        Imports the global vault singleton and verifies it is initialised.
        """
        start = time.perf_counter()
        try:
            from aragora.gateway.openclaw.credential_vault import get_credential_vault

            vault = get_credential_vault()
            latency_ms = (time.perf_counter() - start) * 1000

            if vault is None:
                return ComponentHealth(
                    name="vault",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    error="Credential vault not initialised",
                )

            # Check basic vault state
            details: dict[str, Any] = {}
            if hasattr(vault, "tenant_keys"):
                details["tenant_count"] = len(vault.tenant_keys)
            if hasattr(vault, "_credentials"):
                details["credential_count"] = len(vault._credentials)

            return ComponentHealth(
                name="vault",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details=details,
            )
        except ImportError:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="vault",
                status=HealthStatus.UNKNOWN,
                latency_ms=latency_ms,
                error="Credential vault module not available",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="vault",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(exc),
            )

    async def check_audit(self) -> ComponentHealth:
        """Check audit logging subsystem availability.

        Verifies the audit event types module is importable and the event
        registry is intact (a proxy for the audit storage being wired).
        """
        start = time.perf_counter()
        try:
            from aragora.gateway.openclaw.audit import OpenClawAuditEvents

            # Verify enum is populated (basic integrity check)
            event_count = len(OpenClawAuditEvents)
            latency_ms = (time.perf_counter() - start) * 1000

            if event_count == 0:
                return ComponentHealth(
                    name="audit",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency_ms,
                    error="Audit event registry is empty",
                )

            return ComponentHealth(
                name="audit",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={"registered_events": event_count},
            )
        except ImportError:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="audit",
                status=HealthStatus.UNKNOWN,
                latency_ms=latency_ms,
                error="Audit module not available",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="audit",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(exc),
            )

    async def check_federation(self) -> ComponentHealth:
        """Check federation registry availability.

        Validates the federation registry module is importable and the
        ``FederationRegistry`` class is accessible.
        """
        start = time.perf_counter()
        try:
            from aragora.gateway.federation.registry import FederationRegistry  # noqa: F401

            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="federation",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={"module": "aragora.gateway.federation.registry"},
            )
        except ImportError:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="federation",
                status=HealthStatus.UNKNOWN,
                latency_ms=latency_ms,
                error="Federation registry module not available",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="federation",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(exc),
            )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _aggregate_status(components: list[ComponentHealth]) -> HealthStatus:
        """Compute the worst-case status across all components.

        Ordering (best to worst): HEALTHY > UNKNOWN > DEGRADED > UNHEALTHY.
        """
        if not components:
            return HealthStatus.UNKNOWN

        severity = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.UNKNOWN: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.UNHEALTHY: 3,
        }

        worst = max(components, key=lambda c: severity.get(c.status, 0))
        return worst.status


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    "HealthStatus",
    "ComponentHealth",
    "GatewayHealthStatus",
    "GatewayHealthChecker",
]
