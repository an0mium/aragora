"""
Binding-Based Message Routing.

Implements the ClawdBot pattern for flexible message routing based on
bindings that map message sources to agent configurations.

A binding defines how messages from a specific provider/account/peer
combination should be handled, including which agents to use and
what configuration to apply.

Usage:
    router = BindingRouter()

    # Add bindings
    router.add_binding(MessageBinding(
        provider="slack",
        account_id="T12345",
        peer_pattern="channel:*",
        agent_binding="default",
        priority=10,
    ))

    router.add_binding(MessageBinding(
        provider="slack",
        account_id="T12345",
        peer_pattern="dm:U67890",
        agent_binding="claude-opus",
        priority=20,  # Higher priority for specific user
    ))

    # Resolve binding for a message
    resolution = router.resolve("slack", "T12345", "dm:U67890")
    # Returns the claude-opus binding due to higher priority
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class BindingType(str, Enum):
    """Types of agent bindings."""

    DEFAULT = "default"  # Use default agent configuration
    SPECIFIC_AGENT = "specific_agent"  # Use a specific agent
    AGENT_POOL = "agent_pool"  # Select from a pool of agents
    DEBATE_TEAM = "debate_team"  # Use a full debate team
    CUSTOM = "custom"  # Custom routing logic


@dataclass
class MessageBinding:
    """
    A binding that maps message sources to agent configurations.

    Bindings are matched in priority order (higher priority first).
    The first matching binding determines how the message is handled.
    """

    provider: str  # Platform: "slack", "telegram", "discord"
    account_id: str  # Workspace/team/server ID
    peer_pattern: str  # Pattern: "channel:*", "dm:user123", "*"

    # Agent configuration
    agent_binding: str  # Binding name or agent name
    binding_type: BindingType = BindingType.DEFAULT

    # Priority (higher = more specific, checked first)
    priority: int = 0

    # Optional constraints
    time_window_start: Optional[int] = None  # Hour (0-23)
    time_window_end: Optional[int] = None  # Hour (0-23)
    allowed_users: Optional[Set[str]] = None
    blocked_users: Optional[Set[str]] = None

    # Configuration overrides
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def matches_peer(self, peer_id: str) -> bool:
        """Check if peer ID matches this binding's pattern."""
        return fnmatch.fnmatch(peer_id, self.peer_pattern)

    def matches_time(self, hour: Optional[int] = None) -> bool:
        """Check if current time is within the binding's time window."""
        if self.time_window_start is None or self.time_window_end is None:
            return True

        if hour is None:
            hour = datetime.now(timezone.utc).hour

        if self.time_window_start <= self.time_window_end:
            # Normal range (e.g., 9-17)
            return self.time_window_start <= hour < self.time_window_end
        else:
            # Wrapping range (e.g., 22-6)
            return hour >= self.time_window_start or hour < self.time_window_end

    def matches_user(self, user_id: Optional[str]) -> bool:
        """Check if user is allowed by this binding."""
        if user_id is None:
            return True

        if self.blocked_users and user_id in self.blocked_users:
            return False

        if self.allowed_users and user_id not in self.allowed_users:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "account_id": self.account_id,
            "peer_pattern": self.peer_pattern,
            "agent_binding": self.agent_binding,
            "binding_type": self.binding_type.value,
            "priority": self.priority,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "allowed_users": list(self.allowed_users) if self.allowed_users else None,
            "blocked_users": list(self.blocked_users) if self.blocked_users else None,
            "config_overrides": self.config_overrides,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageBinding":
        return cls(
            provider=data["provider"],
            account_id=data["account_id"],
            peer_pattern=data["peer_pattern"],
            agent_binding=data["agent_binding"],
            binding_type=BindingType(data.get("binding_type", "default")),
            priority=data.get("priority", 0),
            time_window_start=data.get("time_window_start"),
            time_window_end=data.get("time_window_end"),
            allowed_users=set(data["allowed_users"]) if data.get("allowed_users") else None,
            blocked_users=set(data["blocked_users"]) if data.get("blocked_users") else None,
            config_overrides=data.get("config_overrides", {}),
            name=data.get("name"),
            description=data.get("description"),
            enabled=data.get("enabled", True),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now(timezone.utc)
            ),
        )


@dataclass
class BindingResolution:
    """Result of resolving a message binding."""

    matched: bool
    binding: Optional[MessageBinding] = None
    agent_binding: Optional[str] = None
    binding_type: BindingType = BindingType.DEFAULT
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Resolution metadata
    match_reason: Optional[str] = None
    candidates_checked: int = 0


@dataclass
class AgentSelection:
    """Result of selecting an agent based on a binding."""

    agent_name: str
    binding: MessageBinding
    config: Dict[str, Any] = field(default_factory=dict)
    selection_reason: str = "default"


class BindingRouter:
    """
    Routes messages to agents based on binding rules.

    Bindings are organized hierarchically:
    1. Provider (slack, telegram, discord)
    2. Account ID (workspace, team, server)
    3. Peer pattern (channel, DM, thread)

    Within each level, bindings are matched by priority (higher first).
    """

    def __init__(self):
        """Initialize the binding router."""
        # Bindings organized by provider -> account -> list
        self._bindings: Dict[str, Dict[str, List[MessageBinding]]] = {}

        # Default bindings per provider
        self._default_bindings: Dict[str, MessageBinding] = {}

        # Global default
        self._global_default = MessageBinding(
            provider="*",
            account_id="*",
            peer_pattern="*",
            agent_binding="default",
            binding_type=BindingType.DEFAULT,
            priority=-1000,
            name="global_default",
        )

        # Agent registry (for agent pool selection)
        self._agent_pools: Dict[str, List[str]] = {}

    def add_binding(self, binding: MessageBinding) -> None:
        """
        Add a binding to the router.

        Args:
            binding: The binding to add
        """
        provider = binding.provider
        account = binding.account_id

        if provider not in self._bindings:
            self._bindings[provider] = {}

        if account not in self._bindings[provider]:
            self._bindings[provider][account] = []

        self._bindings[provider][account].append(binding)

        # Sort by priority (descending)
        self._bindings[provider][account].sort(key=lambda b: b.priority, reverse=True)

        logger.debug(
            f"Added binding: {binding.name or binding.peer_pattern} "
            f"for {provider}/{account} -> {binding.agent_binding}"
        )

    def remove_binding(
        self,
        provider: str,
        account_id: str,
        peer_pattern: str,
    ) -> bool:
        """
        Remove a binding.

        Args:
            provider: Provider name
            account_id: Account ID
            peer_pattern: Peer pattern

        Returns:
            True if binding was found and removed
        """
        if provider not in self._bindings:
            return False

        if account_id not in self._bindings[provider]:
            return False

        bindings = self._bindings[provider][account_id]
        for binding in bindings:
            if binding.peer_pattern == peer_pattern:
                bindings.remove(binding)
                return True

        return False

    def set_default_binding(
        self,
        provider: str,
        binding: MessageBinding,
    ) -> None:
        """Set the default binding for a provider."""
        self._default_bindings[provider] = binding

    def set_global_default(self, binding: MessageBinding) -> None:
        """Set the global default binding."""
        self._global_default = binding

    def register_agent_pool(
        self,
        pool_name: str,
        agents: List[str],
    ) -> None:
        """
        Register an agent pool for pool-based routing.

        Args:
            pool_name: Name of the pool
            agents: List of agent names in the pool
        """
        self._agent_pools[pool_name] = agents

    def resolve(
        self,
        provider: str,
        account_id: str,
        peer_id: str,
        user_id: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> BindingResolution:
        """
        Resolve which binding applies to a message.

        Args:
            provider: Platform provider (slack, telegram, etc.)
            account_id: Workspace/team ID
            peer_id: Channel/DM/thread identifier
            user_id: Optional user ID for user-based filtering
            hour: Optional hour for time-based filtering

        Returns:
            BindingResolution with matched binding and configuration
        """
        candidates_checked = 0

        # Check exact provider/account bindings
        if provider in self._bindings and account_id in self._bindings[provider]:
            for binding in self._bindings[provider][account_id]:
                candidates_checked += 1

                if not binding.enabled:
                    continue

                if not binding.matches_peer(peer_id):
                    continue

                if not binding.matches_time(hour):
                    continue

                if not binding.matches_user(user_id):
                    continue

                return BindingResolution(
                    matched=True,
                    binding=binding,
                    agent_binding=binding.agent_binding,
                    binding_type=binding.binding_type,
                    config_overrides=binding.config_overrides,
                    match_reason=f"Matched pattern: {binding.peer_pattern}",
                    candidates_checked=candidates_checked,
                )

        # Check wildcard account bindings
        if provider in self._bindings and "*" in self._bindings[provider]:
            for binding in self._bindings[provider]["*"]:
                candidates_checked += 1

                if not binding.enabled:
                    continue

                if not binding.matches_peer(peer_id):
                    continue

                if not binding.matches_time(hour):
                    continue

                if not binding.matches_user(user_id):
                    continue

                return BindingResolution(
                    matched=True,
                    binding=binding,
                    agent_binding=binding.agent_binding,
                    binding_type=binding.binding_type,
                    config_overrides=binding.config_overrides,
                    match_reason=f"Matched wildcard account pattern: {binding.peer_pattern}",
                    candidates_checked=candidates_checked,
                )

        # Check provider default
        if provider in self._default_bindings:
            default = self._default_bindings[provider]
            return BindingResolution(
                matched=True,
                binding=default,
                agent_binding=default.agent_binding,
                binding_type=default.binding_type,
                config_overrides=default.config_overrides,
                match_reason="Provider default",
                candidates_checked=candidates_checked,
            )

        # Fall back to global default
        return BindingResolution(
            matched=True,
            binding=self._global_default,
            agent_binding=self._global_default.agent_binding,
            binding_type=self._global_default.binding_type,
            config_overrides=self._global_default.config_overrides,
            match_reason="Global default",
            candidates_checked=candidates_checked,
        )

    def get_agent_for_message(
        self,
        provider: str,
        account_id: str,
        peer_id: str,
        available_agents: List[Any],
        user_id: Optional[str] = None,
    ) -> AgentSelection:
        """
        Select an agent for a message based on bindings.

        Args:
            provider: Platform provider
            account_id: Workspace/team ID
            peer_id: Channel/DM identifier
            available_agents: List of available Agent objects
            user_id: Optional user ID

        Returns:
            AgentSelection with the selected agent
        """
        resolution = self.resolve(provider, account_id, peer_id, user_id)

        if not resolution.matched or not resolution.binding:
            # Should not happen due to global default
            if available_agents:
                return AgentSelection(
                    agent_name=available_agents[0].name,
                    binding=self._global_default,
                    selection_reason="No binding matched, using first available",
                )
            raise ValueError("No agents available")

        binding = resolution.binding
        agent_binding = resolution.agent_binding

        # Handle different binding types
        if resolution.binding_type == BindingType.SPECIFIC_AGENT:
            # Find specific agent
            for agent in available_agents:
                if agent.name == agent_binding:
                    return AgentSelection(
                        agent_name=agent.name,
                        binding=binding,
                        config=resolution.config_overrides,
                        selection_reason=f"Specific agent binding: {agent_binding}",
                    )

            # Fall back to first available
            logger.warning(f"Bound agent {agent_binding} not available, using fallback")

        elif resolution.binding_type == BindingType.AGENT_POOL:
            # Select from pool
            pool = self._agent_pools.get(agent_binding or "", [])
            for agent in available_agents:
                if agent.name in pool:
                    return AgentSelection(
                        agent_name=agent.name,
                        binding=binding,
                        config=resolution.config_overrides,
                        selection_reason=f"Selected from pool: {agent_binding}",
                    )

        elif resolution.binding_type == BindingType.DEBATE_TEAM:
            # For debate teams, return the binding info
            # The caller should use this to construct the full team
            if available_agents:
                return AgentSelection(
                    agent_name=available_agents[0].name,
                    binding=binding,
                    config={
                        **resolution.config_overrides,
                        "team_config": agent_binding,
                    },
                    selection_reason=f"Debate team: {agent_binding}",
                )

        # Default: return first available agent
        if available_agents:
            return AgentSelection(
                agent_name=available_agents[0].name,
                binding=binding,
                config=resolution.config_overrides,
                selection_reason="Default selection",
            )

        raise ValueError("No agents available for selection")

    def list_bindings(
        self,
        provider: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> List[MessageBinding]:
        """
        List all bindings, optionally filtered.

        Args:
            provider: Optional provider filter
            account_id: Optional account filter

        Returns:
            List of matching bindings
        """
        result: List[MessageBinding] = []

        for prov, accounts in self._bindings.items():
            if provider and prov != provider:
                continue

            for acc, bindings in accounts.items():
                if account_id and acc != account_id:
                    continue

                result.extend(bindings)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        total_bindings = sum(
            len(bindings) for accounts in self._bindings.values() for bindings in accounts.values()
        )

        return {
            "total_bindings": total_bindings,
            "providers": list(self._bindings.keys()),
            "agent_pools": list(self._agent_pools.keys()),
            "has_global_default": self._global_default is not None,
        }


# Global router singleton
_default_router: Optional[BindingRouter] = None


def get_binding_router() -> BindingRouter:
    """Get the default binding router instance."""
    global _default_router
    if _default_router is None:
        _default_router = BindingRouter()
    return _default_router


def reset_binding_router() -> None:
    """Reset the default router (for testing)."""
    global _default_router
    _default_router = None
