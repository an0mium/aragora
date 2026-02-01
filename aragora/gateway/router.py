"""
Agent Router - Per-channel/account agent assignment.

Routes incoming messages to the appropriate agent based on configurable
rules. Supports channel-specific, sender-specific, and content-based
routing.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.gateway.persistence import GatewayStore

from aragora.gateway.inbox import InboxMessage

logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    """A rule for routing messages to agents."""

    rule_id: str
    agent_id: str
    channel_pattern: str = "*"  # glob pattern for channel matching
    sender_pattern: str = "*"  # glob pattern for sender matching
    content_pattern: str | None = None  # optional keyword match
    priority: int = 0  # higher priority rules match first
    enabled: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


class AgentRouter:
    """
    Routes messages to agents based on configurable rules.

    Features:
    - Channel-based routing (e.g., Slack → claude, Telegram → gemini)
    - Sender-based routing (e.g., boss@co.com → priority agent)
    - Content-based routing (keyword matching)
    - Priority-ordered rule evaluation
    - Default agent fallback
    """

    def __init__(
        self,
        default_agent: str = "default",
        store: "GatewayStore | None" = None,
    ) -> None:
        self._rules: dict[str, RoutingRule] = {}
        self._default_agent = default_agent
        self._store = store

    async def hydrate(self) -> None:
        """Load persisted routing rules into memory."""
        if not self._store:
            return
        rules = await self._store.load_rules()
        self._rules = {rule.rule_id: rule for rule in rules}

    async def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        self._rules[rule.rule_id] = rule
        if self._store:
            await self._store.save_rule(rule)

    async def remove_rule(self, rule_id: str) -> bool:
        """Remove a routing rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            if self._store:
                await self._store.delete_rule(rule_id)
            return True
        return False

    async def get_rule(self, rule_id: str) -> RoutingRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    async def list_rules(self) -> list[RoutingRule]:
        """List all routing rules sorted by priority."""
        rules = list(self._rules.values())
        rules.sort(key=lambda r: -r.priority)
        return rules

    async def route(
        self,
        channel: str,
        message: InboxMessage,
    ) -> str:
        """
        Route a message to an agent.

        Evaluates rules in priority order and returns the agent ID
        of the first matching rule, or the default agent.

        Args:
            channel: Source channel.
            message: The incoming message.

        Returns:
            Agent ID to handle the message.
        """
        sorted_rules = sorted(
            [r for r in self._rules.values() if r.enabled],
            key=lambda r: -r.priority,
        )

        for rule in sorted_rules:
            if self._matches(rule, channel, message):
                logger.debug(
                    f"Routed message {message.message_id} to {rule.agent_id} "
                    f"via rule {rule.rule_id}"
                )
                return rule.agent_id

        return self._default_agent

    def _matches(
        self,
        rule: RoutingRule,
        channel: str,
        message: InboxMessage,
    ) -> bool:
        """Check if a message matches a routing rule."""
        if not fnmatch.fnmatch(channel, rule.channel_pattern):
            return False
        if not fnmatch.fnmatch(message.sender, rule.sender_pattern):
            return False
        if rule.content_pattern and rule.content_pattern.lower() not in message.content.lower():
            return False
        return True

    async def count_rules(self) -> int:
        """Get the number of routing rules."""
        return len(self._rules)
