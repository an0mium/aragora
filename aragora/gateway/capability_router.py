"""
Capability-Aware Router - Device capability-based message routing.

Extends the base AgentRouter with capability checking to route messages
based on what the target device can actually do.

Example:
    router = CapabilityRouter(
        device_registry=registry,
        default_agent="default",
    )

    # Add a rule requiring camera for video calls
    await router.add_rule(
        CapabilityRule(
            rule_id="video-support",
            agent_id="video-agent",
            required_capabilities=["camera", "mic"],
            fallback_capabilities=["mic"],
            fallback_agent_id="audio-agent",
        )
    )

    # Route - will use fallback if device lacks camera
    result = await router.route_with_details("slack", message, device_id="dev-1")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aragora.gateway.inbox import InboxMessage
from aragora.gateway.router import AgentRouter, RoutingRule

if TYPE_CHECKING:
    from aragora.gateway.device_registry import DeviceRegistry

logger = logging.getLogger(__name__)


@dataclass
class CapabilityRule(RoutingRule):
    """A routing rule with capability requirements."""

    required_capabilities: list[str] = field(default_factory=list)
    fallback_capabilities: list[str] = field(default_factory=list)
    fallback_agent_id: str | None = None


@dataclass
class RoutingResult:
    """Result of routing a message with capability details."""

    agent_id: str
    rule_id: str | None = None
    used_fallback: bool = False
    missing_capabilities: list[str] = field(default_factory=list)


class CapabilityRouter(AgentRouter):
    """
    Routes messages with device capability awareness.

    Extends AgentRouter to check device capabilities before routing
    and supports fallback agents when capabilities are missing.
    """

    def __init__(
        self,
        default_agent: str = "default",
        device_registry: "DeviceRegistry | None" = None,
    ) -> None:
        super().__init__(default_agent)
        self._device_registry = device_registry
        self._capability_rules: dict[str, CapabilityRule] = {}

    def set_device_registry(self, registry: "DeviceRegistry") -> None:
        """Set the device registry for capability checking."""
        self._device_registry = registry

    async def add_capability_rule(self, rule: CapabilityRule) -> None:
        """Add a capability-aware routing rule."""
        self._capability_rules[rule.rule_id] = rule
        # Also add to base rules for pattern matching
        await self.add_rule(rule)

    async def remove_capability_rule(self, rule_id: str) -> bool:
        """Remove a capability-aware routing rule."""
        if rule_id in self._capability_rules:
            del self._capability_rules[rule_id]
        return await self.remove_rule(rule_id)

    async def route_with_capabilities(
        self,
        channel: str,
        message: InboxMessage,
        device_id: str | None = None,
    ) -> str:
        """
        Route a message with capability checking.

        Args:
            channel: Source channel.
            message: The incoming message.
            device_id: Optional device ID for capability checking.

        Returns:
            Agent ID to handle the message.
        """
        result = await self.route_with_details(channel, message, device_id)
        return result.agent_id

    async def route_with_details(
        self,
        channel: str,
        message: InboxMessage,
        device_id: str | None = None,
    ) -> RoutingResult:
        """
        Route a message with detailed result including capability info.

        Args:
            channel: Source channel.
            message: The incoming message.
            device_id: Optional device ID for capability checking.

        Returns:
            RoutingResult with agent ID and capability details.
        """
        # Get all rules sorted by priority
        all_rules = await self.list_rules()

        for rule in all_rules:
            if not rule.enabled:
                continue

            if not self._matches(rule, channel, message):
                continue

            # Check if this is a capability rule
            cap_rule = self._capability_rules.get(rule.rule_id)

            if cap_rule and cap_rule.required_capabilities and device_id:
                if self._device_registry:
                    has_all, missing = await self._check_capabilities(
                        device_id, cap_rule.required_capabilities
                    )

                    if not has_all:
                        # Try fallback
                        if cap_rule.fallback_capabilities and cap_rule.fallback_agent_id:
                            has_fallback, _ = await self._check_capabilities(
                                device_id, cap_rule.fallback_capabilities
                            )
                            if has_fallback:
                                logger.info(
                                    f"Using fallback agent {cap_rule.fallback_agent_id} "
                                    f"for {message.message_id} (missing: {missing})"
                                )
                                return RoutingResult(
                                    agent_id=cap_rule.fallback_agent_id,
                                    rule_id=rule.rule_id,
                                    used_fallback=True,
                                    missing_capabilities=missing,
                                )

                        # Skip this rule
                        logger.debug(f"Skipping rule {rule.rule_id} - missing: {missing}")
                        continue

            # Rule matches
            logger.debug(f"Routed {message.message_id} to {rule.agent_id} via {rule.rule_id}")
            return RoutingResult(
                agent_id=rule.agent_id,
                rule_id=rule.rule_id,
            )

        return RoutingResult(agent_id=self._default_agent)

    async def _check_capabilities(
        self,
        device_id: str,
        required: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Check if a device has all required capabilities.

        Returns:
            Tuple of (has_all, missing_list)
        """
        if not self._device_registry:
            return True, []

        missing = []
        for cap in required:
            has_cap = await self._device_registry.has_capability(device_id, cap)
            if not has_cap:
                missing.append(cap)

        return len(missing) == 0, missing

    async def find_capable_device(
        self,
        capabilities: list[str],
        prefer_online: bool = True,
    ) -> str | None:
        """
        Find a device that has all required capabilities.

        Args:
            capabilities: Required capabilities
            prefer_online: Prefer online devices

        Returns:
            Device ID or None if no capable device found
        """
        if not self._device_registry:
            return None

        from aragora.gateway.device_registry import DeviceStatus

        # Get devices based on preference
        if prefer_online:
            devices = await self._device_registry.list_devices(status=DeviceStatus.ONLINE)
        else:
            devices = await self._device_registry.list_devices()

        for device in devices:
            has_all = all(cap in device.capabilities for cap in capabilities)
            if has_all:
                return device.device_id

        # If no online device found, check offline
        if prefer_online:
            devices = await self._device_registry.list_devices()
            for device in devices:
                if device.status == DeviceStatus.ONLINE:
                    continue
                has_all = all(cap in device.capabilities for cap in capabilities)
                if has_all:
                    return device.device_id

        return None


__all__ = [
    "CapabilityRule",
    "CapabilityRouter",
    "RoutingResult",
]
