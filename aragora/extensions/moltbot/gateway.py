"""
Local Gateway - Edge Orchestration for Device Networks.

Provides local/edge orchestration for IoT and device networks,
managing device registration, heartbeats, and command routing.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .models import (
    DeviceNode,
    DeviceNodeConfig,
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
    ) -> None:
        """
        Initialize the local gateway.

        Args:
            gateway_id: Gateway identifier (auto-generated if None)
            storage_path: Path for device state storage
            heartbeat_timeout: Seconds before device is marked offline
        """
        self._gateway_id = gateway_id or str(uuid.uuid4())
        self._storage_path = Path(storage_path) if storage_path else None
        self._heartbeat_timeout = heartbeat_timeout

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

            await self._emit_event("device_registered", device)
            return device

    async def get_device(self, device_id: str) -> DeviceNode | None:
        """Get a device by ID."""
        return self._devices.get(device_id)

    async def list_devices(
        self,
        user_id: str | None = None,
        device_type: str | None = None,
        status: str | None = None,
        tenant_id: str | None = None,
    ) -> list[DeviceNode]:
        """List devices with optional filters."""
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
            device = self._devices.get(device_id)
            if not device:
                return False

            del self._devices[device_id]
            logger.info(f"Unregistered device {device_id}")

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
            device = self._devices.get(device_id)
            if not device:
                return None

            now = datetime.utcnow()

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

            if was_offline:
                await self._emit_event("device_online", device)

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
            device = self._devices.get(device_id)
            if not device:
                return None

            if merge:
                device.state.update(state)
            else:
                device.state = state

            device.updated_at = datetime.utcnow()

            await self._emit_event("state_updated", device)
            return device

    async def get_state(self, device_id: str) -> dict[str, Any] | None:
        """Get a device's current state."""
        device = self._devices.get(device_id)
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
        device = self._devices.get(device_id)
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
                return {"success": True, "result": result}
            except asyncio.TimeoutError:
                return {"success": False, "error": "Command timed out"}
            except Exception as e:
                device.errors += 1
                return {"success": False, "error": str(e)}

        # Default: simulate successful command
        device.messages_sent += 1
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
        devices = list(self._devices.values())

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
            "timestamp": datetime.utcnow().isoformat(),
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
            except Exception as e:
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
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")

    async def _check_heartbeats(self) -> None:
        """Check for stale heartbeats and mark devices offline."""
        now = datetime.utcnow()

        async with self._lock:
            for device in self._devices.values():
                if device.status != "online":
                    continue

                if device.last_heartbeat:
                    elapsed = (now - device.last_heartbeat).total_seconds()
                    if elapsed > self._heartbeat_timeout:
                        device.status = "offline"
                        device.updated_at = now
                        logger.warning(f"Device {device.id} marked offline (heartbeat timeout)")
                        await self._emit_event("device_offline", device)

    # ========== Statistics ==========

    async def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        async with self._lock:
            online = sum(1 for d in self._devices.values() if d.status == "online")
            offline = sum(1 for d in self._devices.values() if d.status == "offline")
            error = sum(1 for d in self._devices.values() if d.status == "error")

            by_type: dict[str, int] = {}
            total_messages = 0
            total_errors = 0

            for device in self._devices.values():
                dt = device.config.device_type
                by_type[dt] = by_type.get(dt, 0) + 1
                total_messages += device.messages_sent + device.messages_received
                total_errors += device.errors

            return {
                "gateway_id": self._gateway_id,
                "devices_total": len(self._devices),
                "devices_online": online,
                "devices_offline": offline,
                "devices_error": error,
                "devices_by_type": by_type,
                "total_messages": total_messages,
                "total_errors": total_errors,
                "command_handlers": len(self._command_handlers),
                "event_subscribers": len(self._event_subscribers),
            }
