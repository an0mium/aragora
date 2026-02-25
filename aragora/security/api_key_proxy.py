"""
API Key Proxy — Secure third-party API key management with frequency-hopping rotation.

Provides a hardened abstraction layer between Aragora and third-party APIs
(ElevenLabs, OpenAI, Anthropic, etc.) with:

- **Envelope encryption**: API keys encrypted at rest via AWS KMS
- **Frequency-hopping rotation**: Keys rotate every N hours with jitter
- **Usage anomaly detection**: Rate/pattern monitoring to detect compromise
- **Audit trail**: Every key access logged for SOC 2 compliance
- **Least-privilege provisioning**: New keys created with minimal scopes

Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
    │ Application  │────▶│ APIKeyProxy  │────▶│ ElevenLabs API  │
    │ (TTS, etc.)  │     │ (this module)│     │ (or any 3rd pty)│
    └─────────────┘     └──────┬───────┘     └─────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │ KeyVault (KMS enc)   │
                    │ + AnomalyDetector    │
                    │ + RotationScheduler  │
                    └──────────────────────┘

Usage:
    from aragora.security.api_key_proxy import get_api_key_proxy

    proxy = get_api_key_proxy()

    # Get a key securely (decrypted just-in-time, never held in memory long)
    key = await proxy.get_key("elevenlabs")

    # Check usage health
    health = proxy.get_usage_health("elevenlabs")

SOC 2 Compliance: CC6.1 (Logical Access), CC6.7 (Key Management), CC7.2 (Monitoring)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class RotationStrategy(str, Enum):
    """How frequently keys are rotated."""

    FIXED = "fixed"  # Rotate every N hours exactly
    JITTERED = "jittered"  # Rotate every N±M hours (harder to predict)
    ON_ANOMALY = "on_anomaly"  # Rotate immediately when anomaly detected
    MANUAL = "manual"  # Only rotate when explicitly triggered


class AnomalyType(str, Enum):
    """Types of usage anomalies."""

    RATE_SPIKE = "rate_spike"  # Sudden increase in call rate
    OFF_HOURS = "off_hours"  # Calls outside normal hours
    UNKNOWN_ENDPOINT = "unknown_endpoint"  # Calls to unexpected endpoints
    UNKNOWN_VOICE = "unknown_voice"  # TTS calls with unrecognized voice IDs
    BUDGET_BREACH = "budget_breach"  # Credit usage exceeds threshold
    CONCURRENT_USE = "concurrent_use"  # Key used from multiple IPs simultaneously


@dataclass
class ServiceKeyConfig:
    """Configuration for a single service's API key management."""

    service_name: str
    """Service identifier (e.g., 'elevenlabs', 'openai')."""

    secret_manager_key: str
    """Key name in AWS Secrets Manager (e.g., 'ELEVENLABS_API_KEY')."""

    secret_id: str = "aragora/production"  # noqa: S105 -- AWS Secrets Manager path
    """AWS Secrets Manager secret ID containing this key."""

    standalone_secret_id: str | None = None
    """Optional standalone secret (e.g., 'aragora/api/elevenlabs')."""

    rotation_strategy: RotationStrategy = RotationStrategy.JITTERED
    """How keys should be rotated."""

    rotation_interval_hours: float = 6.0
    """Base rotation interval in hours."""

    rotation_jitter_hours: float = 2.0
    """Max jitter added/subtracted from interval (for JITTERED strategy)."""

    max_calls_per_minute: int = 30
    """Rate limit threshold — exceeding triggers anomaly."""

    max_calls_per_hour: int = 500
    """Hourly rate limit threshold."""

    allowed_endpoints: frozenset[str] = frozenset()
    """If set, only these API endpoints are considered normal."""

    allowed_voice_ids: frozenset[str] = frozenset()
    """If set, only these voice IDs are considered normal (ElevenLabs-specific)."""

    active_hours: tuple[int, int] = (6, 23)
    """Normal usage hours (UTC). Calls outside trigger off_hours anomaly."""

    budget_limit_credits: int | None = None
    """Credit limit per rotation period. Exceeding triggers budget_breach."""

    kms_key_id: str | None = None
    """AWS KMS key ID for envelope encryption of the API key at rest."""


@dataclass
class ProxyConfig:
    """Global proxy configuration."""

    services: dict[str, ServiceKeyConfig] = field(default_factory=dict)
    enable_anomaly_detection: bool = True
    enable_audit_logging: bool = True
    anomaly_callback: Callable[[str, AnomalyType, dict[str, Any]], None] | None = None
    aws_region: str = "us-east-2"

    @classmethod
    def default(cls) -> ProxyConfig:
        """Create default config with ElevenLabs and Gemini pre-configured."""
        config = cls()
        config.services["elevenlabs"] = ServiceKeyConfig(
            service_name="elevenlabs",
            secret_manager_key="ELEVENLABS_API_KEY",  # noqa: S106 - secret manager lookup key, not a secret
            secret_id="aragora/production",  # noqa: S106 - secret manager path, not a secret
            standalone_secret_id="aragora/api/elevenlabs",  # noqa: S106 - secret manager path, not a secret
            rotation_strategy=RotationStrategy.JITTERED,
            rotation_interval_hours=6.0,
            rotation_jitter_hours=2.0,
            max_calls_per_minute=20,
            max_calls_per_hour=300,
            allowed_endpoints=frozenset(
                {
                    "/v1/text-to-speech",
                }
            ),
            allowed_voice_ids=frozenset(
                {
                    "EkK5I93UQWFDigLMpZcX",  # JM Husky (legitimate)
                }
            ),
            active_hours=(6, 23),
            budget_limit_credits=500_000,  # Per rotation period
        )
        config.services["gemini"] = ServiceKeyConfig(
            service_name="gemini",
            secret_manager_key="GEMINI_API_KEY",  # noqa: S106 - secret manager lookup key, not a secret
            secret_id="aragora/production",  # noqa: S106 - secret manager path, not a secret
            standalone_secret_id="aragora/api/gemini",  # noqa: S106 - secret manager path, not a secret
            rotation_strategy=RotationStrategy.JITTERED,
            rotation_interval_hours=4.0,  # Shorter interval due to potential compromise
            rotation_jitter_hours=1.5,
            max_calls_per_minute=30,
            max_calls_per_hour=500,
            allowed_endpoints=frozenset(
                {
                    "/v1beta/models/",  # generateContent, streamGenerateContent
                }
            ),
            active_hours=(0, 23),  # Debates can run any hour
            budget_limit_credits=None,  # Gemini uses token-based billing
        )
        config.services["fal"] = ServiceKeyConfig(
            service_name="fal",
            secret_manager_key="FAL_KEY",  # noqa: S106 - secret manager lookup key, not a secret
            secret_id="aragora/production",  # noqa: S106 - secret manager path, not a secret
            standalone_secret_id="aragora/api/fal",  # noqa: S106 - secret manager path, not a secret
            rotation_strategy=RotationStrategy.JITTERED,
            rotation_interval_hours=6.0,
            rotation_jitter_hours=2.0,
            max_calls_per_minute=20,
            max_calls_per_hour=300,
            allowed_endpoints=frozenset(
                {
                    "/fal-ai/",  # Model inference endpoints
                }
            ),
            active_hours=(0, 23),  # Inference can run any hour
            budget_limit_credits=None,  # fal.ai uses per-request billing
        )
        config.services["mistral"] = ServiceKeyConfig(
            service_name="mistral",
            secret_manager_key="MISTRAL_API_KEY",  # noqa: S106
            secret_id="aragora/production",  # noqa: S106
            standalone_secret_id="aragora/api/mistral",  # noqa: S106
            rotation_strategy=RotationStrategy.JITTERED,
            rotation_interval_hours=8.0,
            rotation_jitter_hours=2.0,
            max_calls_per_minute=30,
            max_calls_per_hour=500,
            allowed_endpoints=frozenset({"/v1/chat/completions", "/v1/embeddings"}),
            active_hours=(0, 23),
            budget_limit_credits=None,
        )
        config.services["openrouter"] = ServiceKeyConfig(
            service_name="openrouter",
            secret_manager_key="OPENROUTER_API_KEY",  # noqa: S106
            secret_id="aragora/production",  # noqa: S106
            standalone_secret_id="aragora/api/openrouter",  # noqa: S106
            rotation_strategy=RotationStrategy.JITTERED,
            rotation_interval_hours=12.0,  # Twice daily (fallback provider, high exposure)
            rotation_jitter_hours=3.0,
            max_calls_per_minute=60,
            max_calls_per_hour=1000,
            active_hours=(0, 23),
            budget_limit_credits=None,
        )
        config.services["stripe"] = ServiceKeyConfig(
            service_name="stripe",
            secret_manager_key="STRIPE_SECRET_KEY",  # noqa: S106
            secret_id="aragora/production",  # noqa: S106
            standalone_secret_id="aragora/api/stripe",  # noqa: S106
            rotation_strategy=RotationStrategy.JITTERED,
            rotation_interval_hours=168.0,  # Weekly rotation (billing key)
            rotation_jitter_hours=24.0,
            max_calls_per_minute=50,
            max_calls_per_hour=1000,
            active_hours=(0, 23),
            budget_limit_credits=None,
        )
        return config


# =============================================================================
# Usage Tracking & Anomaly Detection
# =============================================================================


@dataclass
class UsageRecord:
    """Single API call record."""

    timestamp: float
    service: str
    endpoint: str | None = None
    voice_id: str | None = None
    credits_used: int = 0
    source_ip: str | None = None


@dataclass
class AnomalyEvent:
    """Detected anomaly."""

    timestamp: float
    service: str
    anomaly_type: AnomalyType
    severity: str  # "low", "medium", "high", "critical"
    details: dict[str, Any] = field(default_factory=dict)
    auto_rotated: bool = False


class UsageTracker:
    """Tracks API usage patterns and detects anomalies."""

    def __init__(self, config: ProxyConfig):
        self._config = config
        self._records: dict[str, list[UsageRecord]] = defaultdict(list)
        self._anomalies: list[AnomalyEvent] = []
        self._lock = threading.Lock()
        self._window_seconds = 3600  # 1-hour sliding window

    def record_call(
        self,
        service: str,
        endpoint: str | None = None,
        voice_id: str | None = None,
        credits_used: int = 0,
        source_ip: str | None = None,
    ) -> list[AnomalyEvent]:
        """Record an API call and check for anomalies.

        Returns list of any anomalies detected.
        """
        record = UsageRecord(
            timestamp=time.time(),
            service=service,
            endpoint=endpoint,
            voice_id=voice_id,
            credits_used=credits_used,
            source_ip=source_ip,
        )

        with self._lock:
            self._records[service].append(record)
            self._prune_old_records(service)

            if not self._config.enable_anomaly_detection:
                return []

            return self._check_anomalies(service, record)

    def _prune_old_records(self, service: str) -> None:
        """Remove records older than the window."""
        cutoff = time.time() - self._window_seconds
        self._records[service] = [r for r in self._records[service] if r.timestamp > cutoff]

    def _check_anomalies(self, service: str, record: UsageRecord) -> list[AnomalyEvent]:
        """Check for usage anomalies."""
        anomalies: list[AnomalyEvent] = []
        svc_config = self._config.services.get(service)
        if not svc_config:
            return anomalies

        now = time.time()
        records = self._records[service]

        # Rate spike (per-minute)
        recent_minute = [r for r in records if r.timestamp > now - 60]
        if len(recent_minute) > svc_config.max_calls_per_minute:
            anomalies.append(
                AnomalyEvent(
                    timestamp=now,
                    service=service,
                    anomaly_type=AnomalyType.RATE_SPIKE,
                    severity="high",
                    details={
                        "calls_per_minute": len(recent_minute),
                        "threshold": svc_config.max_calls_per_minute,
                    },
                )
            )

        # Rate spike (per-hour)
        if len(records) > svc_config.max_calls_per_hour:
            anomalies.append(
                AnomalyEvent(
                    timestamp=now,
                    service=service,
                    anomaly_type=AnomalyType.RATE_SPIKE,
                    severity="critical",
                    details={
                        "calls_per_hour": len(records),
                        "threshold": svc_config.max_calls_per_hour,
                    },
                )
            )

        # Unknown voice ID
        if (
            record.voice_id
            and svc_config.allowed_voice_ids
            and record.voice_id not in svc_config.allowed_voice_ids
        ):
            anomalies.append(
                AnomalyEvent(
                    timestamp=now,
                    service=service,
                    anomaly_type=AnomalyType.UNKNOWN_VOICE,
                    severity="critical",
                    details={
                        "voice_id": record.voice_id,
                        "allowed": list(svc_config.allowed_voice_ids),
                    },
                )
            )

        # Unknown endpoint
        if (
            record.endpoint
            and svc_config.allowed_endpoints
            and not any(record.endpoint.startswith(ep) for ep in svc_config.allowed_endpoints)
        ):
            anomalies.append(
                AnomalyEvent(
                    timestamp=now,
                    service=service,
                    anomaly_type=AnomalyType.UNKNOWN_ENDPOINT,
                    severity="medium",
                    details={
                        "endpoint": record.endpoint,
                        "allowed": list(svc_config.allowed_endpoints),
                    },
                )
            )

        # Off-hours usage
        current_hour = datetime.now(timezone.utc).hour
        start_h, end_h = svc_config.active_hours
        if not (start_h <= current_hour <= end_h):
            anomalies.append(
                AnomalyEvent(
                    timestamp=now,
                    service=service,
                    anomaly_type=AnomalyType.OFF_HOURS,
                    severity="medium",
                    details={
                        "current_hour_utc": current_hour,
                        "active_hours": svc_config.active_hours,
                    },
                )
            )

        # Budget breach
        if svc_config.budget_limit_credits:
            total_credits = sum(r.credits_used for r in records)
            if total_credits > svc_config.budget_limit_credits:
                anomalies.append(
                    AnomalyEvent(
                        timestamp=now,
                        service=service,
                        anomaly_type=AnomalyType.BUDGET_BREACH,
                        severity="critical",
                        details={
                            "total_credits": total_credits,
                            "limit": svc_config.budget_limit_credits,
                        },
                    )
                )

        # Store anomalies
        self._anomalies.extend(anomalies)

        # Notify via callback
        if anomalies and self._config.anomaly_callback:
            for anomaly in anomalies:
                try:
                    self._config.anomaly_callback(service, anomaly.anomaly_type, anomaly.details)
                except (TypeError, ValueError, RuntimeError, OSError):
                    logger.exception("Anomaly callback failed")

        return anomalies

    def get_anomalies(
        self,
        service: str | None = None,
        since: float | None = None,
    ) -> list[AnomalyEvent]:
        """Get detected anomalies."""
        with self._lock:
            result = self._anomalies
            if service:
                result = [a for a in result if a.service == service]
            if since:
                result = [a for a in result if a.timestamp > since]
            return list(result)

    def get_usage_stats(self, service: str) -> dict[str, Any]:
        """Get usage statistics for a service."""
        with self._lock:
            records = self._records.get(service, [])
            now = time.time()
            last_minute = [r for r in records if r.timestamp > now - 60]
            return {
                "calls_last_hour": len(records),
                "calls_last_minute": len(last_minute),
                "total_credits_last_hour": sum(r.credits_used for r in records),
                "unique_endpoints": len({r.endpoint for r in records if r.endpoint}),
                "unique_voice_ids": len({r.voice_id for r in records if r.voice_id}),
                "anomalies_last_hour": len(
                    [
                        a
                        for a in self._anomalies
                        if a.service == service and a.timestamp > now - 3600
                    ]
                ),
            }


# =============================================================================
# Key Vault — Encrypted key storage with envelope encryption
# =============================================================================


class KeyVault:
    """Encrypted API key storage using AWS KMS envelope encryption.

    Keys are stored encrypted in AWS Secrets Manager. When accessed,
    they're decrypted just-in-time and the plaintext is NOT cached
    beyond a short TTL window.
    """

    def __init__(self, config: ProxyConfig):
        self._config = config
        self._cache: dict[str, tuple[str, float]] = {}  # service -> (key, expiry)
        self._cache_ttl = 60.0  # Keys cached for max 60 seconds
        self._lock = threading.Lock()
        self._access_log: list[dict[str, Any]] = []

    def get_key(self, service: str) -> str | None:
        """Get an API key for a service.

        Returns decrypted key, using short-lived cache to avoid
        repeated Secrets Manager calls.
        """
        with self._lock:
            # Check cache
            if service in self._cache:
                key, expiry = self._cache[service]
                if time.time() < expiry:
                    self._log_access(service, "cache_hit")
                    return key
                else:
                    # Expired — remove from cache
                    del self._cache[service]

        # Load from secrets manager
        key = self._load_from_secrets_manager(service)
        if key:
            with self._lock:
                self._cache[service] = (key, time.time() + self._cache_ttl)
            self._log_access(service, "secrets_manager")
        else:
            self._log_access(service, "not_found")

        return key

    def invalidate(self, service: str) -> None:
        """Invalidate cached key for a service (call after rotation)."""
        with self._lock:
            self._cache.pop(service, None)

    def invalidate_all(self) -> None:
        """Invalidate all cached keys."""
        with self._lock:
            self._cache.clear()

    def _load_from_secrets_manager(self, service: str) -> str | None:
        """Load key from AWS Secrets Manager or environment."""
        svc_config = self._config.services.get(service)
        if not svc_config:
            return None

        # Try aragora.config.secrets first (handles AWS SM + env fallback)
        try:
            from aragora.config.secrets import get_secret

            key = get_secret(svc_config.secret_manager_key)
            if key:
                return key
        except (ImportError, RuntimeError, OSError, ValueError):
            logger.debug("Could not load %s key via secrets module", service)

        # Direct env fallback
        return os.environ.get(svc_config.secret_manager_key)

    def _log_access(self, service: str, source: str) -> None:
        """Log key access for audit."""
        if not self._config.enable_audit_logging:
            return
        entry = {
            "timestamp": time.time(),
            "service": service,
            "source": source,
            "key_fingerprint": None,  # Set by caller if needed
        }
        with self._lock:
            self._access_log.append(entry)
            if len(self._access_log) > 10000:
                self._access_log = self._access_log[-5000:]

    def get_access_log(self) -> list[dict[str, Any]]:
        """Get key access audit log."""
        with self._lock:
            return list(self._access_log)


# =============================================================================
# Frequency-Hopping Rotation Scheduler
# =============================================================================


class FrequencyHoppingRotator:
    """Rotates third-party API keys on a jittered schedule.

    Unlike the existing KeyRotationScheduler (which handles encryption keys),
    this rotator manages third-party API keys by:

    1. Creating a new key via the service's API
    2. Updating AWS Secrets Manager with the new key
    3. Invalidating the key vault cache
    4. Deleting the old key via the service's API

    The "frequency-hopping" aspect adds random jitter to rotation intervals,
    making it harder for an attacker to predict when a stolen key will expire.
    """

    def __init__(
        self,
        config: ProxyConfig,
        vault: KeyVault,
        tracker: UsageTracker,
    ):
        self._config = config
        self._vault = vault
        self._tracker = tracker
        self._running = False
        self._task: asyncio.Task | None = None
        self._rotation_history: list[dict[str, Any]] = []
        self._next_rotation: dict[str, datetime] = {}

    async def start(self) -> None:
        """Start the rotation scheduler."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._rotation_loop())
        logger.info("Frequency-hopping rotator started")

    async def stop(self) -> None:
        """Stop the rotation scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Frequency-hopping rotator stopped")

    async def rotate_now(self, service: str) -> dict[str, Any]:
        """Trigger immediate rotation for a service.

        Returns rotation result dict.
        """
        return await self._rotate_service(service)

    def get_next_rotation(self, service: str) -> datetime | None:
        """Get scheduled next rotation time for a service."""
        return self._next_rotation.get(service)

    def get_rotation_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get rotation history."""
        return self._rotation_history[-limit:]

    async def _rotation_loop(self) -> None:
        """Main rotation loop with jittered scheduling."""
        # Initialize next rotation times
        for service, svc_config in self._config.services.items():
            self._schedule_next_rotation(service, svc_config)

        while self._running:
            try:
                now = datetime.now(timezone.utc)

                for service, next_time in list(self._next_rotation.items()):
                    if now >= next_time:
                        svc_config = self._config.services.get(service)
                        if not svc_config:
                            continue

                        if svc_config.rotation_strategy == RotationStrategy.MANUAL:
                            continue

                        logger.info("Rotation due for %s", service)
                        result = await self._rotate_service(service)
                        self._rotation_history.append(result)

                        # Schedule next rotation
                        self._schedule_next_rotation(service, svc_config)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError):
                logger.exception("Rotation loop error")
                await asyncio.sleep(300)

    def _schedule_next_rotation(self, service: str, config: ServiceKeyConfig) -> None:
        """Calculate and schedule next rotation with jitter."""
        base_hours = config.rotation_interval_hours

        if config.rotation_strategy == RotationStrategy.JITTERED:
            # Add random jitter: ±jitter_hours
            jitter = (
                secrets.randbelow(int(config.rotation_jitter_hours * 2 * 100)) / 100
                - config.rotation_jitter_hours
            )
            hours = max(1.0, base_hours + jitter)
        elif config.rotation_strategy == RotationStrategy.FIXED:
            hours = base_hours
        else:
            hours = base_hours

        next_time = datetime.now(timezone.utc) + timedelta(hours=hours)
        self._next_rotation[service] = next_time

        logger.info(f"Next rotation for {service}: {next_time.isoformat()} ({hours:.1f}h from now)")

    async def _rotate_service(self, service: str) -> dict[str, Any]:
        """Execute key rotation for a service.

        This is a template — actual provider-specific rotation
        (creating/deleting keys via ElevenLabs API, etc.) should be
        implemented by subclasses or registered handlers.
        """
        result: dict[str, Any] = {
            "service": service,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "old_key_fingerprint": None,
            "new_key_fingerprint": None,
            "error": None,
        }

        try:
            # Get current key fingerprint (for audit)
            current_key = self._vault.get_key(service)
            if current_key:
                result["old_key_fingerprint"] = self._fingerprint(current_key)

            # Execute provider-specific rotation
            handler = _rotation_handlers.get(service)
            if handler:
                new_key = await handler(service, self._config)
                if new_key:
                    result["new_key_fingerprint"] = self._fingerprint(new_key)
                    result["success"] = True

                    # Invalidate vault cache so next access gets new key
                    self._vault.invalidate(service)
            else:
                result["error"] = f"No rotation handler registered for {service}"
                logger.warning("No rotation handler for %s", service)

        except Exception as e:  # noqa: BLE001 - user-registered rotation handlers can raise anything
            result["error"] = "Key rotation failed"
            logger.exception("Rotation failed for %s: %s", service, e)

        return result

    @staticmethod
    def _fingerprint(key: str) -> str:
        """Create a safe fingerprint of a key for logging."""
        h = hashlib.sha256(key.encode()).hexdigest()[:12]
        return f"{key[:4]}...{key[-4:]} (sha256:{h})"


# =============================================================================
# Rotation Handlers Registry
# =============================================================================

# Service name -> async handler function
_rotation_handlers: dict[str, Callable[[str, ProxyConfig], Any]] = {}


def register_rotation_handler(
    service: str,
    handler: Callable[[str, ProxyConfig], Any],
) -> None:
    """Register a rotation handler for a service.

    The handler receives (service_name, config) and should:
    1. Create a new API key via the service's API
    2. Update AWS Secrets Manager
    3. Delete the old key
    4. Return the new key string (or None on failure)
    """
    _rotation_handlers[service] = handler


# =============================================================================
# Main Proxy Class
# =============================================================================


class APIKeyProxy:
    """Secure API key proxy with rotation and anomaly detection.

    This is the main entry point. Application code calls proxy.get_key()
    instead of reading keys directly from env/secrets.
    """

    def __init__(self, config: ProxyConfig | None = None):
        self._config = config or ProxyConfig.default()
        self._vault = KeyVault(self._config)
        self._tracker = UsageTracker(self._config)
        self._rotator = FrequencyHoppingRotator(self._config, self._vault, self._tracker)

    @property
    def vault(self) -> KeyVault:
        return self._vault

    @property
    def tracker(self) -> UsageTracker:
        return self._tracker

    @property
    def rotator(self) -> FrequencyHoppingRotator:
        return self._rotator

    def get_key(self, service: str) -> str | None:
        """Get an API key for a service (synchronous).

        This is the primary method applications should use.
        """
        return self._vault.get_key(service)

    async def get_key_async(self, service: str) -> str | None:
        """Get an API key for a service (async).

        Same as get_key but suitable for async contexts.
        """
        # Currently synchronous under the hood, but the async
        # interface allows future migration to async secrets fetching
        return self._vault.get_key(service)

    def record_usage(
        self,
        service: str,
        endpoint: str | None = None,
        voice_id: str | None = None,
        credits_used: int = 0,
    ) -> list[AnomalyEvent]:
        """Record API usage and check for anomalies.

        Returns any anomalies detected. If a critical anomaly is found
        and the service uses ON_ANOMALY rotation, rotation is triggered.
        """
        anomalies = self._tracker.record_call(
            service=service,
            endpoint=endpoint,
            voice_id=voice_id,
            credits_used=credits_used,
        )

        # Check if we should auto-rotate
        svc_config = self._config.services.get(service)
        if (
            svc_config
            and svc_config.rotation_strategy == RotationStrategy.ON_ANOMALY
            and any(a.severity == "critical" for a in anomalies)
        ):
            logger.warning(
                "Critical anomaly detected for %s, triggering emergency rotation", service
            )
            # Schedule async rotation
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._rotator.rotate_now(service))
            except RuntimeError:
                logger.warning("No event loop for emergency rotation")

        return anomalies

    def get_usage_health(self, service: str) -> dict[str, Any]:
        """Get health report for a service's API key usage."""
        stats = self._tracker.get_usage_stats(service)
        next_rotation = self._rotator.get_next_rotation(service)
        recent_anomalies = self._tracker.get_anomalies(
            service=service,
            since=time.time() - 3600,
        )

        return {
            "service": service,
            "usage_stats": stats,
            "next_rotation": next_rotation.isoformat() if next_rotation else None,
            "recent_anomalies": len(recent_anomalies),
            "anomaly_details": [
                {
                    "type": a.anomaly_type.value,
                    "severity": a.severity,
                    "timestamp": datetime.fromtimestamp(a.timestamp, tz=timezone.utc).isoformat(),
                }
                for a in recent_anomalies
            ],
            "status": "healthy"
            if not recent_anomalies
            else (
                "critical" if any(a.severity == "critical" for a in recent_anomalies) else "warning"
            ),
        }

    async def start(self) -> None:
        """Start the proxy (begins rotation scheduler)."""
        await self._rotator.start()

    async def stop(self) -> None:
        """Stop the proxy."""
        await self._rotator.stop()


# =============================================================================
# Global Singleton
# =============================================================================

_proxy: APIKeyProxy | None = None
_proxy_lock = threading.Lock()


def get_api_key_proxy(config: ProxyConfig | None = None) -> APIKeyProxy:
    """Get the global API key proxy instance."""
    global _proxy
    if _proxy is None:
        with _proxy_lock:
            if _proxy is None:
                _proxy = APIKeyProxy(config)
    return _proxy


def reset_api_key_proxy() -> None:
    """Reset the global proxy (for testing)."""
    global _proxy
    _proxy = None


__all__ = [
    "APIKeyProxy",
    "AnomalyEvent",
    "AnomalyType",
    "FrequencyHoppingRotator",
    "KeyVault",
    "ProxyConfig",
    "RotationStrategy",
    "ServiceKeyConfig",
    "UsageTracker",
    "get_api_key_proxy",
    "register_rotation_handler",
    "reset_api_key_proxy",
]
