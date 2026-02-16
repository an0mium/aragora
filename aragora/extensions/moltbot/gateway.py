"""
Local Gateway - Edge Orchestration for Device Networks.

Provides local/edge orchestration for IoT and device networks,
managing device registration, heartbeats, and command routing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable

from aragora.gateway.device_registry import (
    DeviceNode as GatewayDeviceNode,
    DeviceRegistry,
    DeviceStatus,
)
from aragora.stores.canonical import get_canonical_gateway_stores

from .models import DeviceNode, DeviceNodeConfig


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _gateway_status_to_moltbot(status: DeviceStatus) -> str:
    mapping = {
        DeviceStatus.ONLINE: "online",
        DeviceStatus.OFFLINE: "offline",
        DeviceStatus.PAIRED: "offline",
        DeviceStatus.BLOCKED: "error",
    }
    return mapping.get(status, "offline")


def _moltbot_device_from_gateway(device: GatewayDeviceNode) -> DeviceNode:
    metadata = device.metadata or {}
    status = metadata.get("status")
    if status not in {"online", "offline", "error"}:
        status = _gateway_status_to_moltbot(device.status)

    created_at = _parse_dt(metadata.get("created_at")) or datetime.now(timezone.utc)
    updated_at = _parse_dt(metadata.get("updated_at")) or created_at
    last_seen = _parse_dt(metadata.get("last_seen"))
    if last_seen is None and device.last_seen:
        last_seen = datetime.utcfromtimestamp(device.last_seen)
    last_heartbeat = _parse_dt(metadata.get("last_heartbeat"))

    config = DeviceNodeConfig(
        name=device.name,
        device_type=device.device_type,
        capabilities=list(device.capabilities),
        connection_type=metadata.get("connection_type", "mqtt"),
        heartbeat_interval=int(metadata.get("heartbeat_interval", 60)),
        metadata=dict(metadata.get("moltbot_metadata", {})),
    )

    return DeviceNode(
        id=device.device_id,
        config=config,
        user_id=metadata.get("user_id", ""),
        gateway_id=metadata.get("gateway_id", ""),
        tenant_id=metadata.get("tenant_id"),
        created_at=created_at,
        updated_at=updated_at,
        status=status,
        last_seen=last_seen,
        last_heartbeat=last_heartbeat,
        state=dict(metadata.get("state", {})),
        firmware_version=str(metadata.get("firmware_version", "")),
        battery_level=metadata.get("battery_level"),
        signal_strength=metadata.get("signal_strength"),
        messages_sent=int(metadata.get("messages_sent", 0) or 0),
        messages_received=int(metadata.get("messages_received", 0) or 0),
        errors=int(metadata.get("errors", 0) or 0),
        uptime_seconds=int(metadata.get("uptime_seconds", 0) or 0),
        metadata=dict(metadata.get("metadata", {})),
    )


def _gateway_device_from_moltbot(
    device: DeviceNode,
    status: DeviceStatus | None = None,
) -> GatewayDeviceNode:
    gateway_status = status or DeviceStatus.PAIRED
    return GatewayDeviceNode(
        device_id=device.id,
        name=device.config.name,
        device_type=device.config.device_type,
        capabilities=list(device.config.capabilities),
        status=gateway_status,
        last_seen=device.last_seen.timestamp() if device.last_seen else None,
        metadata={
            "user_id": device.user_id,
            "tenant_id": device.tenant_id,
            "gateway_id": device.gateway_id,
            "status": device.status,
            "state": device.state,
            "firmware_version": device.firmware_version,
            "battery_level": device.battery_level,
            "signal_strength": device.signal_strength,
            "last_seen": device.last_seen.isoformat() if device.last_seen else None,
            "last_heartbeat": device.last_heartbeat.isoformat() if device.last_heartbeat else None,
            "connection_type": device.config.connection_type,
            "heartbeat_interval": device.config.heartbeat_interval,
            "created_at": device.created_at.isoformat(),
            "updated_at": device.updated_at.isoformat(),
            "messages_sent": device.messages_sent,
            "messages_received": device.messages_received,
            "errors": device.errors,
            "uptime_seconds": device.uptime_seconds,
            "moltbot_metadata": dict(device.config.metadata),
            "metadata": dict(device.metadata),
        },
    )


logger = logging.getLogger(__name__)


class LocalGateway:
    """
    Edge gateway for device network orchestration.

    Manages device registration, heartbeats, state synchronization,
    and command routing for IoT and edge device networks.
    """

    def __init__(
        self,
        gateway_id: str | None = None,
        storage_path: str | Path | None = None,
        heartbeat_timeout: float = 120.0,
        registry: DeviceRegistry | None = None,
        mirror_registry: bool | None = None,
    ) -> None:
        """
        Initialize the local gateway.

        Args:
            gateway_id: Gateway identifier (auto-generated if None)
            storage_path: Path for device state storage
            heartbeat_timeout: Seconds before device is marked offline
            registry: Optional shared device registry
            mirror_registry: Mirror device updates into the registry (env MOLTBOT_GATEWAY_REGISTRY)
        """
        self._gateway_id = gateway_id or str(uuid.uuid4())
        self._storage_path = Path(storage_path) if storage_path else None
        self._heartbeat_timeout = heartbeat_timeout
        if mirror_registry is None:
            mirror_registry = (
                True if registry is not None else os.getenv("MOLTBOT_GATEWAY_REGISTRY", "0") == "1"
            )
        if registry is None:
            if mirror_registry:
                store = get_canonical_gateway_stores().gateway_store()
                registry = DeviceRegistry(store=store)
            else:
                registry = DeviceRegistry()
        self._registry = registry
        self._mirror_registry = mirror_registry

        self._devices: dict[str, DeviceNode] = {}
        self._command_handlers: dict[str, Callable] = {}
        self._event_subscribers: list[Callable] = []
        self._lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task | None = None

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    @property
    def gateway_id(self) -> str:
        """Get the gateway identifier."""
        return self._gateway_id

    def _use_registry(self) -> bool:
        return self._mirror_registry

    async def _get_registry_device(self, device_id: str) -> DeviceNode | None:
        if not self._use_registry():
            return None
        reg_device = await self._registry.get(device_id)
        if not reg_device:
            return None
        device = _moltbot_device_from_gateway(reg_device)
        self._devices[device.id] = device
        return device

    async def _list_registry_devices(
        self,
        status: str | None = None,
        device_type: str | None = None,
    ) -> list[DeviceNode]:
        if not self._use_registry():
            return []
        status_filter = None
        if status:
            status_filter = {
                "online": DeviceStatus.ONLINE,
                "offline": DeviceStatus.OFFLINE,
                "error": DeviceStatus.BLOCKED,
            }.get(status)
        reg_devices = await self._registry.list_devices(
            status=status_filter,
            device_type=device_type,
        )
        devices = [_moltbot_device_from_gateway(dev) for dev in reg_devices]
        for device in devices:
            self._devices[device.id] = device
        return devices

    async def start(self) -> None:
        """Start the gateway (heartbeat monitoring, etc.)."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            logger.info(f"Gateway {self._gateway_id} started")

    async def stop(self) -> None:
        """Stop the gateway."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info(f"Gateway {self._gateway_id} stopped")

    # ========== Device Management ==========

    async def register_device(
        self,
        config: DeviceNodeConfig,
        user_id: str,
        tenant_id: str | None = None,
    ) -> DeviceNode:
        """
        Register a new device with the gateway.

        Args:
            config: Device configuration
            user_id: Owner user ID
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Registered device node
        """
        async with self._lock:
            device_id = str(uuid.uuid4())

            device = DeviceNode(
                id=device_id,
                config=config,
                user_id=user_id,
                gateway_id=self._gateway_id,
                tenant_id=tenant_id,
            )

            self._devices[device_id] = device
            logger.info(f"Registered device {config.name} ({device_id})")

            await self._mirror_registry_device(device, status=DeviceStatus.PAIRED)
            await self._emit_event("device_registered", device)
            return device

    async def get_device(self, device_id: str) -> DeviceNode | None:
        """Get a device by ID."""
        device = await self._get_registry_device(device_id)
        if device:
            return device
        return self._devices.get(device_id)

    async def list_devices(
        self,
        user_id: str | None = None,
        device_type: str | None = None,
        status: str | None = None,
        tenant_id: str | None = None,
    ) -> list[DeviceNode]:
        """List devices with optional filters."""
        if self._use_registry():
            devices = await self._list_registry_devices(status=status, device_type=device_type)
        else:
            devices = list(self._devices.values())

        if user_id:
            devices = [d for d in devices if d.user_id == user_id]
        if device_type:
            devices = [d for d in devices if d.config.device_type == device_type]
        if status:
            devices = [d for d in devices if d.status == status]
        if tenant_id:
            devices = [d for d in devices if d.tenant_id == tenant_id]

        return devices

    async def unregister_device(self, device_id: str) -> bool:
        """Unregister a device from the gateway."""
        async with self._lock:
            device = await self.get_device(device_id)
            if not device:
                return False

            if device_id in self._devices:
                del self._devices[device_id]
            logger.info(f"Unregistered device {device_id}")

            await self._mirror_registry_unreg(device_id)
            await self._emit_event("device_unregistered", device)
            return True

    # ========== Device State ==========

    async def heartbeat(
        self,
        device_id: str,
        state: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> DeviceNode | None:
        """
        Process a device heartbeat.

        Args:
            device_id: Device sending heartbeat
            state: Current device state
            metrics: Device metrics (battery, signal, etc.)

        Returns:
            Updated device node
        """
        async with self._lock:
            device = await self.get_device(device_id)
            if not device:
                return None

            now = datetime.now(timezone.utc)

            # Update status to online if was offline
            was_offline = device.status != "online"
            device.status = "online"
            device.last_heartbeat = now
            device.last_seen = now
            device.updated_at = now

            # Update state if provided
            if state:
                device.state.update(state)

            # Update metrics if provided
            if metrics:
                if "battery_level" in metrics:
                    device.battery_level = metrics["battery_level"]
                if "signal_strength" in metrics:
                    device.signal_strength = metrics["signal_strength"]
                if "firmware_version" in metrics:
                    device.firmware_version = metrics["firmware_version"]

            device.messages_received += 1
            self._devices[device.id] = device

            if was_offline:
                await self._emit_event("device_online", device)

            await self._mirror_registry_device(device, status=DeviceStatus.ONLINE)
            return device

    async def update_state(
        self,
        device_id: str,
        state: dict[str, Any],
        merge: bool = True,
    ) -> DeviceNode | None:
        """
        Update a device's state.

        Args:
            device_id: Device to update
            state: State update
            merge: Merge with existing state (vs replace)

        Returns:
            Updated device node
        """
        async with self._lock:
            device = await self.get_device(device_id)
            if not device:
                return None

            if merge:
                device.state.update(state)
            else:
                device.state = state

            device.updated_at = datetime.now(timezone.utc)
            self._devices[device.id] = device

            await self._mirror_registry_device(device)
            await self._emit_event("state_updated", device)
            return device

    async def get_state(self, device_id: str) -> dict[str, Any] | None:
        """Get a device's current state."""
        device = await self.get_device(device_id)
        if not device:
            return None
        return device.state.copy()

    # ========== Commands ==========

    async def send_command(
        self,
        device_id: str,
        command: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Send a command to a device.

        Args:
            device_id: Target device
            command: Command name
            payload: Command payload
            timeout: Response timeout in seconds

        Returns:
            Command result
        """
        device = await self.get_device(device_id)
        if not device:
            return {"success": False, "error": "Device not found"}

        if device.status != "online":
            return {"success": False, "error": "Device is offline"}

        # Check if device supports this command
        if command not in device.config.capabilities:
            return {"success": False, "error": f"Device does not support command: {command}"}

        # Route to command handler if registered
        handler = self._command_handlers.get(command)
        if handler:
            try:
                result = await asyncio.wait_for(
                    handler(device, payload or {}),
                    timeout=timeout,
                )
                device.messages_sent += 1
                self._devices[device.id] = device
                await self._mirror_registry_device(device)
                return {"success": True, "result": result}
            except asyncio.TimeoutError:
                return {"success": False, "error": "Command timed out"}
            except (RuntimeError, ValueError, AttributeError) as e:
                device.errors += 1
                self._devices[device.id] = device
                await self._mirror_registry_device(device)
                logger.error("Command handler failed for device %s: %s", device.id, e)
                return {"success": False, "error": "Command execution failed"}

        # Default: simulate successful command
        device.messages_sent += 1
        self._devices[device.id] = device
        await self._mirror_registry_device(device)
        return {
            "success": True,
            "result": {"command": command, "acknowledged": True},
        }

    async def broadcast_command(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
        device_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Broadcast a command to multiple devices.

        Args:
            command: Command name
            payload: Command payload
            device_filter: Filter criteria for target devices

        Returns:
            Broadcast results
        """
        devices = await self.list_devices()

        # Apply filters
        if device_filter:
            if "device_type" in device_filter:
                devices = [
                    d for d in devices if d.config.device_type == device_filter["device_type"]
                ]
            if "status" in device_filter:
                devices = [d for d in devices if d.status == device_filter["status"]]
            if "capability" in device_filter:
                devices = [
                    d for d in devices if device_filter["capability"] in d.config.capabilities
                ]

        results = {}
        for device in devices:
            result = await self.send_command(device.id, command, payload)
            results[device.id] = result

        return {
            "total": len(devices),
            "success": sum(1 for r in results.values() if r.get("success")),
            "failed": sum(1 for r in results.values() if not r.get("success")),
            "results": results,
        }

    def register_command_handler(
        self,
        command: str,
        handler: Callable,
    ) -> None:
        """Register a handler for a command type."""
        self._command_handlers[command] = handler
        logger.info(f"Registered command handler: {command}")

    # ========== Events ==========

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to gateway events."""
        self._event_subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from gateway events."""
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)

    async def _emit_event(self, event_type: str, device: DeviceNode) -> None:
        """Emit an event to all subscribers."""
        event = {
            "type": event_type,
            "device_id": device.id,
            "gateway_id": self._gateway_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": {
                "name": device.config.name,
                "type": device.config.device_type,
                "status": device.status,
            },
        }

        for callback in self._event_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except (RuntimeError, ValueError, AttributeError) as e:  # user-supplied subscriber
                logger.error(f"Event subscriber error: {e}")

    # ========== Heartbeat Monitoring ==========

    async def _heartbeat_monitor(self) -> None:
        """Monitor device heartbeats and mark offline devices."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._check_heartbeats()
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError, ValueError) as e:
                logger.error(f"Heartbeat monitor error: {e}")

    async def _check_heartbeats(self) -> None:
        """Check for stale heartbeats and mark devices offline."""
        now = datetime.now(timezone.utc)

        async with self._lock:
            devices = await self.list_devices()
            for device in devices:
                if device.status != "online":
                    continue

                if device.last_heartbeat:
                    elapsed = (now - device.last_heartbeat).total_seconds()
                    if elapsed > self._heartbeat_timeout:
                        device.status = "offline"
                        device.updated_at = now
                        self._devices[device.id] = device
                        logger.warning(f"Device {device.id} marked offline (heartbeat timeout)")
                        await self._emit_event("device_offline", device)
                        await self._mirror_registry_device(
                            device,
                            status=DeviceStatus.OFFLINE,
                        )

    # ========== Statistics ==========

    async def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        async with self._lock:
            devices = await self.list_devices()
            online = sum(1 for d in devices if d.status == "online")
            offline = sum(1 for d in devices if d.status == "offline")
            error = sum(1 for d in devices if d.status == "error")

            by_type: dict[str, int] = {}
            total_messages = 0
            total_errors = 0

            for device in devices:
                dt = device.config.device_type
                by_type[dt] = by_type.get(dt, 0) + 1
                total_messages += device.messages_sent + device.messages_received
                total_errors += device.errors

            return {
                "gateway_id": self._gateway_id,
                "devices_total": len(devices),
                "devices_online": online,
                "devices_offline": offline,
                "devices_error": error,
                "devices_by_type": by_type,
                "total_messages": total_messages,
                "total_errors": total_errors,
                "command_handlers": len(self._command_handlers),
                "event_subscribers": len(self._event_subscribers),
            }

    async def _mirror_registry_device(
        self,
        device: DeviceNode,
        status: DeviceStatus | None = None,
    ) -> None:
        if not self._mirror_registry:
            return

        reg_device = await self._registry.get(device.id)
        if reg_device is None:
            reg_device = _gateway_device_from_moltbot(device, status=status)
            await self._registry.register(reg_device)
        else:
            reg_device.name = device.config.name
            reg_device.device_type = device.config.device_type
            reg_device.capabilities = list(device.config.capabilities)
            reg_device.metadata = self._registry_metadata(device)
            reg_device.last_seen = (
                device.last_seen.timestamp() if device.last_seen else reg_device.last_seen
            )
            if status is not None:
                reg_device.status = status
            await self._registry.save(reg_device)

    async def _mirror_registry_unreg(self, device_id: str) -> None:
        if not self._mirror_registry:
            return
        await self._registry.unregister(device_id)

    def _registry_metadata(self, device: DeviceNode) -> dict[str, Any]:
        return {
            "user_id": device.user_id,
            "tenant_id": device.tenant_id,
            "gateway_id": device.gateway_id,
            "status": device.status,
            "state": device.state,
            "firmware_version": device.firmware_version,
            "battery_level": device.battery_level,
            "signal_strength": device.signal_strength,
            "last_seen": device.last_seen.isoformat() if device.last_seen else None,
            "last_heartbeat": (
                device.last_heartbeat.isoformat() if device.last_heartbeat else None
            ),
            "created_at": device.created_at.isoformat(),
            "messages_sent": device.messages_sent,
            "messages_received": device.messages_received,
            "errors": device.errors,
            "uptime_seconds": device.uptime_seconds,
            "connection_type": device.config.connection_type,
            "heartbeat_interval": device.config.heartbeat_interval,
            "moltbot_metadata": dict(device.config.metadata),
            "metadata": dict(device.metadata),
            "updated_at": device.updated_at.isoformat(),
            "registry_updated_at": time.time(),
        }
