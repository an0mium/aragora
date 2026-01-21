"""Deployment Validator for Aragora.

Provides comprehensive runtime validation for production deployments,
including connectivity checks, API key validation, and component health.

Usage:
    from aragora.ops import validate_deployment, quick_health_check

    # Full validation (async)
    result = await validate_deployment()
    if not result.ready:
        for issue in result.issues:
            print(f"[{issue.severity}] {issue.component}: {issue.message}")

    # Quick sync check for startup
    if not quick_health_check():
        sys.exit(1)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"  # Blocks deployment
    WARNING = "warning"  # Should be addressed
    INFO = "info"  # Informational


class ComponentStatus(str, Enum):
    """Component health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ValidationIssue:
    """A validation issue found during deployment check."""

    component: str
    message: str
    severity: Severity
    suggestion: Optional[str] = None


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: ComponentStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Complete deployment validation result."""

    ready: bool  # True if deployment can proceed
    live: bool  # True if basic health checks pass
    issues: list[ValidationIssue] = field(default_factory=list)
    components: list[ComponentHealth] = field(default_factory=list)
    validated_at: float = field(default_factory=time.time)
    validation_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ready": self.ready,
            "live": self.live,
            "issues": [
                {
                    "component": i.component,
                    "message": i.message,
                    "severity": i.severity.value,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "message": c.message,
                    "metadata": c.metadata,
                }
                for c in self.components
            ],
            "validated_at": self.validated_at,
            "validation_duration_ms": self.validation_duration_ms,
        }


class DeploymentValidator:
    """Validates deployment readiness with runtime checks."""

    def __init__(self):
        self.issues: list[ValidationIssue] = []
        self.components: list[ComponentHealth] = []

    async def validate(self) -> ValidationResult:
        """Run all validation checks.

        Returns:
            ValidationResult with readiness status and any issues found
        """
        start = time.time()
        self.issues = []
        self.components = []

        # Run all checks
        await self._check_environment()
        await self._check_storage()
        await self._check_database()
        await self._check_redis()
        await self._check_api_keys()
        await self._check_encryption()

        duration_ms = (time.time() - start) * 1000

        # Determine readiness
        critical_issues = [i for i in self.issues if i.severity == Severity.CRITICAL]
        ready = len(critical_issues) == 0

        # Determine liveness (basic health)
        unhealthy = [c for c in self.components if c.status == ComponentStatus.UNHEALTHY]
        live = len(unhealthy) == 0 or all(
            c.name in ("redis", "postgres")
            for c in unhealthy  # Optional components
        )

        return ValidationResult(
            ready=ready,
            live=live,
            issues=self.issues,
            components=self.components,
            validation_duration_ms=duration_ms,
        )

    async def _check_environment(self) -> None:
        """Check environment configuration."""
        env = os.environ.get("ARAGORA_ENV", "development")
        is_production = env.lower() in ("production", "prod", "live")

        self.components.append(
            ComponentHealth(
                name="environment",
                status=ComponentStatus.HEALTHY,
                metadata={"env": env, "is_production": is_production},
            )
        )

        # Check for debug mode in production
        if is_production and os.environ.get("ARAGORA_DEBUG", "").lower() in ("true", "1"):
            self.issues.append(
                ValidationIssue(
                    component="environment",
                    message="Debug mode enabled in production",
                    severity=Severity.WARNING,
                    suggestion="Set ARAGORA_DEBUG=false",
                )
            )

    async def _check_storage(self) -> None:
        """Check storage paths are writable."""
        start = time.time()

        # Check data directory
        data_dir = os.environ.get("ARAGORA_DATA_DIR", "./data")
        data_path = Path(data_dir)

        try:
            data_path.mkdir(parents=True, exist_ok=True)
            test_file = data_path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()

            self.components.append(
                ComponentHealth(
                    name="storage",
                    status=ComponentStatus.HEALTHY,
                    latency_ms=(time.time() - start) * 1000,
                    metadata={"data_dir": str(data_path.absolute())},
                )
            )
        except (OSError, PermissionError) as e:
            self.components.append(
                ComponentHealth(
                    name="storage",
                    status=ComponentStatus.UNHEALTHY,
                    message=f"Cannot write to data directory: {e}",
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="storage",
                    message=f"Data directory not writable: {data_dir}",
                    severity=Severity.CRITICAL,
                    suggestion=f"Ensure {data_dir} exists and is writable",
                )
            )

    async def _check_database(self) -> None:
        """Check database connectivity."""
        start = time.time()
        db_backend = os.environ.get("ARAGORA_DB_BACKEND", "sqlite").lower()

        if db_backend in ("postgres", "postgresql"):
            await self._check_postgres()
        else:
            # SQLite - just verify we can create/access the file
            try:
                from aragora.storage.schema import get_database_manager  # type: ignore[attr-defined]

                db = get_database_manager()
                # Quick query to verify connectivity
                async with db.get_session() as session:
                    await session.execute("SELECT 1")

                self.components.append(
                    ComponentHealth(
                        name="database",
                        status=ComponentStatus.HEALTHY,
                        latency_ms=(time.time() - start) * 1000,
                        metadata={"backend": "sqlite"},
                    )
                )
            except Exception as e:
                self.components.append(
                    ComponentHealth(
                        name="database",
                        status=ComponentStatus.UNHEALTHY,
                        message=str(e),
                        metadata={"backend": "sqlite"},
                    )
                )
                self.issues.append(
                    ValidationIssue(
                        component="database",
                        message=f"SQLite database error: {e}",
                        severity=Severity.CRITICAL,
                    )
                )

    async def _check_postgres(self) -> None:
        """Check PostgreSQL connectivity."""
        start = time.time()
        dsn = os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")

        if not dsn:
            self.components.append(
                ComponentHealth(
                    name="database",
                    status=ComponentStatus.UNHEALTHY,
                    message="PostgreSQL DSN not configured",
                    metadata={"backend": "postgres"},
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="database",
                    message="PostgreSQL configured but no DSN provided",
                    severity=Severity.CRITICAL,
                    suggestion="Set ARAGORA_POSTGRES_DSN or DATABASE_URL",
                )
            )
            return

        try:
            import asyncpg

            conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=5.0)
            await conn.execute("SELECT 1")
            await conn.close()

            self.components.append(
                ComponentHealth(
                    name="database",
                    status=ComponentStatus.HEALTHY,
                    latency_ms=(time.time() - start) * 1000,
                    metadata={"backend": "postgres"},
                )
            )
        except asyncio.TimeoutError:
            self.components.append(
                ComponentHealth(
                    name="database",
                    status=ComponentStatus.UNHEALTHY,
                    message="Connection timeout",
                    metadata={"backend": "postgres"},
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="database",
                    message="PostgreSQL connection timeout",
                    severity=Severity.CRITICAL,
                    suggestion="Check database host and network connectivity",
                )
            )
        except ImportError:
            self.components.append(
                ComponentHealth(
                    name="database",
                    status=ComponentStatus.UNHEALTHY,
                    message="asyncpg not installed",
                    metadata={"backend": "postgres"},
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="database",
                    message="asyncpg package not installed",
                    severity=Severity.CRITICAL,
                    suggestion="pip install asyncpg",
                )
            )
        except Exception as e:
            self.components.append(
                ComponentHealth(
                    name="database",
                    status=ComponentStatus.UNHEALTHY,
                    message=str(e),
                    metadata={"backend": "postgres"},
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="database",
                    message=f"PostgreSQL connection failed: {e}",
                    severity=Severity.CRITICAL,
                )
            )

    async def _check_redis(self) -> None:
        """Check Redis connectivity (optional)."""
        start = time.time()
        redis_url = os.environ.get("ARAGORA_REDIS_URL") or os.environ.get("REDIS_URL")

        if not redis_url:
            self.components.append(
                ComponentHealth(
                    name="redis",
                    status=ComponentStatus.UNKNOWN,
                    message="Not configured (optional)",
                )
            )
            return

        try:
            import redis.asyncio as redis

            client = redis.from_url(redis_url)
            await asyncio.wait_for(client.ping(), timeout=5.0)
            await client.close()

            self.components.append(
                ComponentHealth(
                    name="redis",
                    status=ComponentStatus.HEALTHY,
                    latency_ms=(time.time() - start) * 1000,
                )
            )
        except asyncio.TimeoutError:
            self.components.append(
                ComponentHealth(
                    name="redis",
                    status=ComponentStatus.UNHEALTHY,
                    message="Connection timeout",
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="redis",
                    message="Redis connection timeout",
                    severity=Severity.WARNING,
                    suggestion="Check Redis host or remove REDIS_URL to use SQLite fallback",
                )
            )
        except ImportError:
            self.components.append(
                ComponentHealth(
                    name="redis",
                    status=ComponentStatus.UNKNOWN,
                    message="redis package not installed",
                )
            )
        except Exception as e:
            self.components.append(
                ComponentHealth(
                    name="redis",
                    status=ComponentStatus.UNHEALTHY,
                    message=str(e),
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="redis",
                    message=f"Redis connection failed: {e}",
                    severity=Severity.WARNING,
                )
            )

    async def _check_api_keys(self) -> None:
        """Check API key configuration."""
        providers = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        configured = []
        for name, env_var in providers.items():
            if os.environ.get(env_var):
                configured.append(name)

        if not configured:
            self.components.append(
                ComponentHealth(
                    name="api_keys",
                    status=ComponentStatus.UNHEALTHY,
                    message="No AI provider API keys configured",
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="api_keys",
                    message="No AI provider API keys configured",
                    severity=Severity.CRITICAL,
                    suggestion="Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY",
                )
            )
        else:
            self.components.append(
                ComponentHealth(
                    name="api_keys",
                    status=ComponentStatus.HEALTHY,
                    metadata={"providers": configured},
                )
            )

    async def _check_encryption(self) -> None:
        """Check encryption configuration."""
        env = os.environ.get("ARAGORA_ENV", "development")
        is_production = env.lower() in ("production", "prod", "live")
        encryption_key = os.environ.get("ARAGORA_ENCRYPTION_KEY")

        if is_production and not encryption_key:
            self.components.append(
                ComponentHealth(
                    name="encryption",
                    status=ComponentStatus.UNHEALTHY,
                    message="Encryption key not configured for production",
                )
            )
            self.issues.append(
                ValidationIssue(
                    component="encryption",
                    message="ARAGORA_ENCRYPTION_KEY required in production",
                    severity=Severity.CRITICAL,
                    suggestion="Generate a 32-byte hex key: openssl rand -hex 32",
                )
            )
        elif encryption_key:
            # Validate key format
            try:
                key_bytes = bytes.fromhex(encryption_key)
                if len(key_bytes) != 32:
                    raise ValueError("Key must be 32 bytes")

                self.components.append(
                    ComponentHealth(
                        name="encryption",
                        status=ComponentStatus.HEALTHY,
                        metadata={"configured": True},
                    )
                )
            except (ValueError, TypeError) as e:
                self.components.append(
                    ComponentHealth(
                        name="encryption",
                        status=ComponentStatus.UNHEALTHY,
                        message=f"Invalid encryption key format: {e}",
                    )
                )
                self.issues.append(
                    ValidationIssue(
                        component="encryption",
                        message="Invalid ARAGORA_ENCRYPTION_KEY format",
                        severity=Severity.CRITICAL,
                        suggestion="Key must be 64 hex characters (32 bytes)",
                    )
                )
        else:
            self.components.append(
                ComponentHealth(
                    name="encryption",
                    status=ComponentStatus.UNKNOWN,
                    message="Not configured (optional in development)",
                )
            )


async def validate_deployment() -> ValidationResult:
    """Run deployment validation.

    Returns:
        ValidationResult with readiness status
    """
    validator = DeploymentValidator()
    return await validator.validate()


def quick_health_check() -> bool:
    """Quick synchronous health check for startup.

    Returns:
        True if basic health checks pass
    """
    try:
        # Check minimum requirements
        from aragora.config.validator import validate_all

        result = validate_all(strict=False)

        if result.get("errors"):
            for error in result["errors"]:
                logger.error(f"Configuration error: {error}")
            return False

        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


__all__ = [
    "DeploymentValidator",
    "ValidationResult",
    "ValidationIssue",
    "ComponentHealth",
    "ComponentStatus",
    "Severity",
    "validate_deployment",
    "quick_health_check",
]
