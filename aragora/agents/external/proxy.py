"""Proxy wrapper for external agent adapters.

Provides Airlock-style protection with:
- Timeout handling with configurable limits
- Pre-execution policy enforcement
- Output sanitization
- Retry logic with exponential backoff
- Metrics collection
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from .base import ExternalAgentAdapter
from .models import (
    HealthStatus,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from .security import ExternalAgentSecurityPolicy, PolicyCheckResult

if TYPE_CHECKING:
    from aragora.rbac.models import AuthorizationContext

logger = logging.getLogger(__name__)


class PolicyDeniedError(Exception):
    """Raised when a policy check denies execution."""

    def __init__(self, reason: str, result: PolicyCheckResult | None = None):
        super().__init__(reason)
        self.reason = reason
        self.result = result


@dataclass
class ProxyConfig:
    """Configuration for the external agent proxy wrapper."""

    submit_timeout: float = 60.0  # Timeout for task submission
    status_timeout: float = 30.0  # Timeout for status checks
    result_timeout: float = 60.0  # Timeout for result retrieval
    cancel_timeout: float = 30.0  # Timeout for cancellation
    max_retries: int = 3
    retry_delay: float = 2.0
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    enable_policy_checks: bool = True
    redact_output_secrets: bool = True


class ExternalAgentProxy:
    """Airlock-style proxy wrapper for external agent adapters.

    Provides:
    - Timeout handling with configurable limits per operation
    - Pre-execution policy enforcement
    - Output sanitization
    - Retry logic with exponential backoff
    - Metrics collection

    Usage:
        adapter = OpenHandsAdapter(config)
        proxy = ExternalAgentProxy(adapter, auth_context)

        task_id = await proxy.submit_task(request)
        result = await proxy.get_task_result(task_id)

    Note:
        The proxy is stateless except for metrics. Create one per
        request or reuse across requests for the same user.
    """

    def __init__(
        self,
        adapter: ExternalAgentAdapter,
        auth_context: "AuthorizationContext",
        config: ProxyConfig | None = None,
        security_policy: ExternalAgentSecurityPolicy | None = None,
    ):
        """Initialize the proxy.

        Args:
            adapter: The underlying external agent adapter.
            auth_context: Aragora authorization context for permission checks.
            config: Proxy configuration.
            security_policy: Optional custom security policy.
        """
        self._adapter = adapter
        self._auth_context = auth_context
        self._config = config or ProxyConfig()
        self._security = security_policy or ExternalAgentSecurityPolicy()

        # Metrics
        self._total_calls: int = 0
        self._successful_calls: int = 0
        self._timeout_errors: int = 0
        self._policy_denials: int = 0
        self._retry_count: int = 0

    @property
    def wrapped_adapter(self) -> ExternalAgentAdapter:
        """Get the wrapped adapter."""
        return self._adapter

    @property
    def metrics(self) -> dict[str, Any]:
        """Get proxy metrics."""
        success_rate = (
            self._successful_calls / self._total_calls * 100 if self._total_calls > 0 else 100.0
        )
        return {
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "timeout_errors": self._timeout_errors,
            "policy_denials": self._policy_denials,
            "retry_count": self._retry_count,
            "success_rate": success_rate,
        }

    async def submit_task(self, request: TaskRequest) -> str:
        """Submit a task with policy enforcement and timeout handling.

        Args:
            request: Task request.

        Returns:
            Task ID.

        Raises:
            PolicyDeniedError: If policy check fails.
            TimeoutError: If submission times out.
        """
        self._total_calls += 1

        # Pre-execution policy check
        if self._config.enable_policy_checks:
            policy_result = self._security.check_pre_execution(
                request=request,
                context=self._auth_context,
                adapter_config=self._adapter.config,
            )

            if not policy_result.allowed:
                self._policy_denials += 1
                self._security.audit_task(
                    "denied",
                    request.id,
                    getattr(self._auth_context, "user_id", None),
                    {"reason": policy_result.reason},
                )
                raise PolicyDeniedError(
                    policy_result.reason or "Policy check failed",
                    policy_result,
                )

        # Submit with timeout and retry
        try:
            task_id = await self._with_retry(
                lambda: self._adapter.submit_task(request),
                timeout=self._config.submit_timeout,
                operation="submit_task",
            )

            self._successful_calls += 1
            self._security.audit_task(
                "submitted",
                task_id,
                getattr(self._auth_context, "user_id", None),
                {"prompt_preview": request.prompt[:200]},
            )

            return task_id

        except asyncio.TimeoutError:
            self._timeout_errors += 1
            logger.error(f"Task submission timeout for adapter {self._adapter.adapter_name}")
            raise

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status with timeout handling.

        Args:
            task_id: Task ID to check.

        Returns:
            Current task status.

        Raises:
            TimeoutError: If status check times out.
        """
        try:
            return await asyncio.wait_for(
                self._adapter.get_task_status(task_id),
                timeout=self._config.status_timeout,
            )
        except asyncio.TimeoutError:
            self._timeout_errors += 1
            logger.warning(f"Status check timeout for task {task_id}")
            raise

    async def get_task_result(self, task_id: str) -> TaskResult:
        """Get task result with sanitization.

        Args:
            task_id: Task ID.

        Returns:
            Sanitized TaskResult.

        Raises:
            TimeoutError: If result retrieval times out.
        """
        try:
            result = await asyncio.wait_for(
                self._adapter.get_task_result(task_id),
                timeout=self._config.result_timeout,
            )

            # Sanitize output
            if self._config.redact_output_secrets:
                result = self._security.sanitize_output(result)

            # Audit completed task
            event = "completed" if result.status == TaskStatus.COMPLETED else "failed"
            self._security.audit_task(
                event,
                task_id,
                getattr(self._auth_context, "user_id", None),
                {
                    "status": result.status.value,
                    "tokens_used": result.tokens_used,
                    "cost_usd": result.cost_usd,
                },
            )

            return result

        except asyncio.TimeoutError:
            self._timeout_errors += 1
            logger.warning(f"Result retrieval timeout for task {task_id}")
            raise

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task with audit logging.

        Args:
            task_id: Task ID to cancel.

        Returns:
            True if cancelled, False otherwise.
        """
        try:
            cancelled = await asyncio.wait_for(
                self._adapter.cancel_task(task_id),
                timeout=self._config.cancel_timeout,
            )

            if cancelled:
                self._security.audit_task(
                    "cancelled",
                    task_id,
                    getattr(self._auth_context, "user_id", None),
                    {},
                )

            return cancelled

        except asyncio.TimeoutError:
            self._timeout_errors += 1
            logger.warning(f"Cancel timeout for task {task_id}")
            return False

    async def stream_progress(self, task_id: str) -> AsyncIterator[TaskProgress]:
        """Stream progress updates.

        Args:
            task_id: Task ID to stream.

        Yields:
            TaskProgress updates.
        """
        async for progress in self._adapter.stream_progress(task_id):
            yield progress

    async def health_check(self) -> HealthStatus:
        """Check adapter health.

        Returns:
            Health status of the underlying adapter.
        """
        return await self._adapter.health_check()

    async def _with_retry(
        self,
        coro_factory: Callable[[], Any],
        timeout: float,
        operation: str,
    ) -> Any:
        """Execute coroutine with retry logic.

        Args:
            coro_factory: Callable that returns a coroutine to execute.
            timeout: Timeout in seconds.
            operation: Operation name for logging.

        Returns:
            Result of the coroutine.

        Raises:
            The last exception if all retries fail.
        """
        last_error: Exception | None = None
        delay = self._config.retry_delay

        for attempt in range(self._config.max_retries):
            try:
                coro = coro_factory()
                return await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(f"{operation} timed out after {timeout}s")
                logger.warning(
                    f"{operation} timeout, attempt {attempt + 1}/{self._config.max_retries}"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{operation} failed: {e}, attempt {attempt + 1}/{self._config.max_retries}"
                )

            if attempt < self._config.max_retries - 1:
                self._retry_count += 1
                await asyncio.sleep(delay)
                delay *= self._config.retry_backoff

        if last_error:
            raise last_error
        raise RuntimeError(f"{operation} failed after {self._config.max_retries} attempts")
