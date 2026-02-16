"""
Device Security - Secure device pairing and presence monitoring.

Provides enhanced security for device registration:
- Cryptographic pairing ceremony with verification codes
- Rate limiting to prevent abuse
- Presence monitoring with automatic offline detection
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from aragora.gateway.device_registry import DeviceNode, DeviceRegistry, DeviceStatus
from aragora.stores.canonical import get_canonical_gateway_stores

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def _audit_device_action(
    actor_id: str,
    action: str,
    device_id: str,
    granted: bool,
    **details: Any,
) -> None:
    """Emit a structured audit event for device security actions."""
    try:
        from aragora.observability.security_audit import audit_rbac_decision

        await audit_rbac_decision(
            user_id=actor_id,
            permission=f"device:{action}",
            granted=granted,
            resource_type="device",
            resource_id=device_id,
            **details,
        )
    except (ImportError, TypeError, RuntimeError):
        pass


class PairingStatus(Enum):
    """Status of a pairing request."""

    REQUESTED = "requested"
    CODE_SENT = "code_sent"
    APPROVED = "approved"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class PairingRequest:
    """A pending device pairing request."""

    request_id: str
    device_id: str
    device_name: str
    device_type: str
    verification_code: str
    challenge: str
    requested_at: float
    status: PairingStatus = PairingStatus.REQUESTED
    approved_by: str | None = None
    approved_at: float | None = None
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


class SecureDeviceRegistry:
    """
    Enhanced device registry with secure pairing.

    Features:
    - Verification code based pairing ceremony
    - Rate limiting for pairing requests
    - Presence monitoring with heartbeats
    - Automatic offline detection
    """

    def __init__(
        self,
        registry: DeviceRegistry | None = None,
        pairing_timeout: float = 300.0,  # 5 minutes
        max_requests_per_minute: int = 5,
        heartbeat_interval: float = 30.0,
        offline_threshold: float = 90.0,  # 3 missed heartbeats
        on_device_offline: Callable[[str], None] | None = None,
    ) -> None:
        if registry is None:
            store = get_canonical_gateway_stores().gateway_store()
            registry = DeviceRegistry(store=store)
        self._registry = registry
        self._pairing_timeout = pairing_timeout
        self._max_requests_per_minute = max_requests_per_minute
        self._heartbeat_interval = heartbeat_interval
        self._offline_threshold = offline_threshold
        self._on_device_offline = on_device_offline

        self._pending_requests: dict[str, PairingRequest] = {}
        self._request_timestamps: list[float] = []
        self._monitor_task: asyncio.Task | None = None
        self._running = False

    @property
    def registry(self) -> DeviceRegistry:
        """Get the underlying device registry."""
        return self._registry

    async def request_pairing(
        self,
        device_name: str,
        device_type: str,
        capabilities: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> PairingRequest:
        """
        Request to pair a new device.

        Generates a verification code that must be confirmed
        through a trusted channel (e.g., displayed on user's device).

        Args:
            device_name: Human-readable device name
            device_type: Type of device (laptop, phone, etc.)
            capabilities: Device capabilities
            metadata: Additional device metadata

        Returns:
            PairingRequest with verification code

        Raises:
            ValueError: If rate limit exceeded
        """
        # Rate limiting
        now = time.time()
        self._request_timestamps = [ts for ts in self._request_timestamps if now - ts < 60.0]

        if len(self._request_timestamps) >= self._max_requests_per_minute:
            raise ValueError("Rate limit exceeded. Please wait before requesting another pairing.")

        self._request_timestamps.append(now)

        # Generate secure IDs
        request_id = f"pair-{secrets.token_hex(8)}"
        device_id = f"dev-{secrets.token_hex(8)}"
        verification_code = f"{secrets.randbelow(900000) + 100000}"  # 6-digit code
        challenge = secrets.token_hex(32)

        request = PairingRequest(
            request_id=request_id,
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            verification_code=verification_code,
            challenge=challenge,
            requested_at=now,
            capabilities=capabilities or [],
            metadata=metadata or {},
        )

        self._pending_requests[request_id] = request
        logger.info(f"Pairing requested for {device_name} ({device_type}): {request_id}")

        await _audit_device_action(
            actor_id="system",
            action="pairing_requested",
            device_id=device_id,
            granted=True,
            device_name=device_name,
            device_type=device_type,
        )

        return request

    async def approve_pairing(
        self,
        request_id: str,
        approved_by: str = "user",
    ) -> bool:
        """
        Approve a pairing request (user action).

        Called when user approves the pairing on their trusted device.

        Args:
            request_id: The pairing request ID
            approved_by: Who approved the pairing

        Returns:
            True if approved, False if request not found or expired
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        # Check expiration
        if time.time() - request.requested_at > self._pairing_timeout:
            request.status = PairingStatus.EXPIRED
            return False

        if request.status != PairingStatus.REQUESTED:
            return False

        request.status = PairingStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = time.time()

        logger.info(f"Pairing approved for {request.device_name}: {request_id}")

        await _audit_device_action(
            actor_id=approved_by,
            action="pairing_approved",
            device_id=request.device_id,
            granted=True,
            request_id=request_id,
        )

        return True

    async def confirm_pairing(
        self,
        request_id: str,
        verification_code: str,
    ) -> DeviceNode | None:
        """
        Confirm pairing with verification code.

        Called from the device being paired with the code displayed
        to the user on a trusted device.

        Args:
            request_id: The pairing request ID
            verification_code: The 6-digit code shown to user

        Returns:
            DeviceNode if confirmed successfully, None otherwise
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return None

        # Check expiration
        if time.time() - request.requested_at > self._pairing_timeout:
            request.status = PairingStatus.EXPIRED
            return None

        if request.status != PairingStatus.APPROVED:
            logger.warning(f"Pairing not approved yet: {request_id}")
            return None

        # Verify code (timing-safe comparison)
        if not secrets.compare_digest(request.verification_code, verification_code):
            logger.warning(f"Invalid verification code for {request_id}")
            return None

        # Create and register the device
        device = DeviceNode(
            device_id=request.device_id,
            name=request.device_name,
            device_type=request.device_type,
            capabilities=request.capabilities,
            metadata=request.metadata,
        )

        await self._registry.register(device)
        request.status = PairingStatus.CONFIRMED

        # Clean up
        del self._pending_requests[request_id]

        logger.info(f"Device paired successfully: {device.device_id}")

        await _audit_device_action(
            actor_id=request.approved_by or "system",
            action="pairing_confirmed",
            device_id=device.device_id,
            granted=True,
            request_id=request_id,
        )

        return device

    async def reject_pairing(self, request_id: str) -> bool:
        """Reject a pairing request."""
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        request.status = PairingStatus.REJECTED
        del self._pending_requests[request_id]
        logger.info(f"Pairing rejected: {request_id}")

        await _audit_device_action(
            actor_id="system",
            action="pairing_rejected",
            device_id=request.device_id,
            granted=False,
            request_id=request_id,
        )

        return True

    async def get_pending_requests(self) -> list[PairingRequest]:
        """Get all pending pairing requests."""
        now = time.time()
        pending = []

        for request in list(self._pending_requests.values()):
            if now - request.requested_at > self._pairing_timeout:
                request.status = PairingStatus.EXPIRED
            elif request.status in (PairingStatus.REQUESTED, PairingStatus.APPROVED):
                pending.append(request)

        return pending

    async def start(self) -> None:
        """Start presence monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._presence_monitor())
        logger.info("Presence monitoring started")

    async def stop(self) -> None:
        """Stop presence monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Presence monitoring stopped")

    async def _presence_monitor(self) -> None:
        """Background task that monitors device presence."""
        while self._running:
            try:
                await self._check_presence()
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.error(f"Error in presence monitor: {e}")
                await asyncio.sleep(self._heartbeat_interval)

    async def _check_presence(self) -> None:
        """Check all devices and mark offline if no recent heartbeat."""
        now = time.time()
        devices = await self._registry.list_devices()

        for device in devices:
            if device.status == DeviceStatus.BLOCKED:
                continue

            if device.last_seen and now - device.last_seen > self._offline_threshold:
                if device.status == DeviceStatus.ONLINE:
                    device.status = DeviceStatus.OFFLINE
                    logger.info(f"Device went offline: {device.device_id}")

                    if self._on_device_offline:
                        try:
                            self._on_device_offline(device.device_id)
                        except (RuntimeError, ValueError, TypeError) as e:  # noqa: BLE001 - user-provided offline callback
                            logger.error(f"Error in offline callback: {e}")

    async def heartbeat(self, device_id: str) -> bool:
        """
        Record a heartbeat from a device.

        Args:
            device_id: The device sending the heartbeat

        Returns:
            True if heartbeat recorded, False if device not found
        """
        return await self._registry.heartbeat(device_id)

    # Delegate common operations to underlying registry
    async def get(self, device_id: str) -> DeviceNode | None:
        """Get a device by ID."""
        return await self._registry.get(device_id)

    async def list_devices(
        self,
        status: DeviceStatus | None = None,
        device_type: str | None = None,
    ) -> list[DeviceNode]:
        """List devices with optional filters."""
        return await self._registry.list_devices(status, device_type)

    async def unregister(self, device_id: str) -> bool:
        """Unregister a device."""
        return await self._registry.unregister(device_id)

    async def block(self, device_id: str) -> bool:
        """Block a device."""
        return await self._registry.block(device_id)


__all__ = [
    "PairingStatus",
    "PairingRequest",
    "SecureDeviceRegistry",
]
