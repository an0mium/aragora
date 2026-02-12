"""
Startup validation runner module.

Provides functions to run deployment validation early during server startup,
before other components are initialized. This ensures the server doesn't start
with an invalid or insecure configuration.

Usage:
    from aragora.server.startup.validation_runner import (
        run_startup_validation,
        run_startup_validation_sync,
    )

    # Async version
    result = await run_startup_validation(strict=False)

    # Sync wrapper (for early startup before event loop)
    result = run_startup_validation_sync(strict=False)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.ops.deployment_validator import ValidationResult

logger = logging.getLogger(__name__)


class StartupValidationError(RuntimeError):
    """Raised when startup validation fails with critical issues.

    This is a wrapper around DeploymentNotReadyError that provides
    a cleaner interface for startup-specific validation failures.
    """

    def __init__(self, message: str, result: ValidationResult | None = None) -> None:
        super().__init__(message)
        self.result = result


def _should_skip_validation() -> bool:
    """Check if startup validation should be skipped.

    Returns True if ARAGORA_SKIP_STARTUP_VALIDATION is set to a truthy value.
    This is useful for tests or development environments where full validation
    is not needed.
    """
    return os.environ.get("ARAGORA_SKIP_STARTUP_VALIDATION", "").lower() in (
        "true",
        "1",
        "yes",
        "skip",
    )


def _is_production() -> bool:
    """Check if running in production environment."""
    env = os.environ.get("ARAGORA_ENV", "development").lower()
    return env in ("production", "prod", "live")


async def run_startup_validation(
    strict: bool | None = None,
) -> ValidationResult | None:
    """Run deployment validation during server startup.

    This function should be called early in the server startup sequence,
    before other components are initialized. It validates the deployment
    configuration and logs any issues found.

    Args:
        strict: If True, raise StartupValidationError for critical issues.
            If False, only log warnings and return the result.
            If None, automatically set based on ARAGORA_ENV:
            - Production: strict=True (unless ARAGORA_STRICT_DEPLOYMENT=false)
            - Development: strict=False

    Returns:
        ValidationResult from the deployment validator, or None if validation
        was skipped (ARAGORA_SKIP_STARTUP_VALIDATION=true).

    Raises:
        StartupValidationError: When strict=True and critical issues are found.

    Environment Variables:
        ARAGORA_SKIP_STARTUP_VALIDATION: Set to "true" to skip validation entirely.
            Useful for tests or development environments.
        ARAGORA_ENV: Environment name (production, development, etc.)
        ARAGORA_STRICT_DEPLOYMENT: If "false" in production, don't fail on critical issues.
    """
    # Check if validation should be skipped
    if _should_skip_validation():
        logger.info("[STARTUP VALIDATION] Skipped (ARAGORA_SKIP_STARTUP_VALIDATION=true)")
        return None

    # Determine strictness based on environment if not explicitly set
    if strict is None:
        is_prod = _is_production()
        # In production, default to strict unless explicitly disabled
        if is_prod:
            explicit_strict = os.environ.get("ARAGORA_STRICT_DEPLOYMENT", "").lower()
            strict = explicit_strict not in ("false", "0", "no")
        else:
            strict = False

    try:
        from aragora.ops.deployment_validator import (
            DeploymentNotReadyError,
            Severity,
            validate_deployment,
        )

        logger.info("[STARTUP VALIDATION] Running deployment validation...")

        try:
            result = await validate_deployment(strict=strict)
        except DeploymentNotReadyError as e:
            # When strict=True and validation fails, this is raised
            logger.error(f"[STARTUP VALIDATION] FAILED: {e}")
            raise StartupValidationError(str(e), result=e.result) from e

        # Log validation results
        critical_count = sum(1 for i in result.issues if i.severity == Severity.CRITICAL)
        warning_count = sum(1 for i in result.issues if i.severity == Severity.WARNING)
        info_count = sum(1 for i in result.issues if i.severity == Severity.INFO)

        # Log based on result
        if result.ready:
            if warning_count > 0:
                logger.info(
                    f"[STARTUP VALIDATION] Passed with {warning_count} warning(s), "
                    f"{info_count} info message(s). Duration: {result.validation_duration_ms:.1f}ms"
                )
            else:
                logger.info(
                    f"[STARTUP VALIDATION] All checks passed. "
                    f"Duration: {result.validation_duration_ms:.1f}ms"
                )
        else:
            logger.warning(
                f"[STARTUP VALIDATION] {critical_count} critical issue(s), "
                f"{warning_count} warning(s). Server may not function correctly."
            )

        # Log warnings (always)
        for issue in result.issues:
            if issue.severity == Severity.WARNING:
                logger.warning(f"[STARTUP VALIDATION] WARNING - {issue.component}: {issue.message}")
                if issue.suggestion:
                    logger.warning(f"  Suggestion: {issue.suggestion}")

        # Log critical issues (always)
        for issue in result.issues:
            if issue.severity == Severity.CRITICAL:
                logger.error(f"[STARTUP VALIDATION] CRITICAL - {issue.component}: {issue.message}")
                if issue.suggestion:
                    logger.error(f"  Suggestion: {issue.suggestion}")

        return result

    except ImportError as e:
        logger.warning(f"[STARTUP VALIDATION] Deployment validator not available: {e}")
        return None
    except StartupValidationError:
        # Re-raise our own error
        raise
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning(f"[STARTUP VALIDATION] Validation failed with error: {e}")
        return None


def run_startup_validation_sync(strict: bool | None = None) -> ValidationResult | None:
    """Synchronous wrapper for run_startup_validation.

    This function runs the async validation in a new event loop if no loop
    is currently running, or uses asyncio.run() for simplicity.

    Args:
        strict: If True, raise StartupValidationError for critical issues.
            See run_startup_validation() for details.

    Returns:
        ValidationResult from the deployment validator, or None if validation
        was skipped.

    Raises:
        StartupValidationError: When strict=True and critical issues are found.
    """
    # Check if validation should be skipped (fast path, no async needed)
    if _should_skip_validation():
        logger.info("[STARTUP VALIDATION] Skipped (ARAGORA_SKIP_STARTUP_VALIDATION=true)")
        return None

    try:
        # Check if we're already in an async context
        asyncio.get_running_loop()  # Raises RuntimeError if no loop
        # We're in an async context - can't use asyncio.run()
        # Create a future and run it in the existing loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, run_startup_validation(strict=strict))
            return future.result(timeout=60.0)
    except RuntimeError:
        # No event loop running - safe to use asyncio.run()
        return asyncio.run(run_startup_validation(strict=strict))


__all__ = [
    "StartupValidationError",
    "run_startup_validation",
    "run_startup_validation_sync",
]
