"""Reasoning-domain event-subscriber home (P4a EventBus inversion, Batch E3).

The belief-network cross-subsystem reaction, relocated here from infrastructure
``aragora.events.cross_subscribers.handlers.basic`` so the reasoning-coupled
reaction lives in its DOMAIN home. The module self-registers via the domain-free
registry (``aragora.events.cross_subscribers.register_subscriber`` - domain ->
infrastructure, downward = legal); the layered bootstraps import it so
``CrossSubscriberManager.apply_registered_subscribers`` wires it in.

Per the relocate-UP no-shim exemption (AGENTS.md "P4a Contracts-Thread Shared Rules"
and docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §8) there is NO re-export shim at
the old path; every consumer is repointed instead.

Handles:
- Vote → Belief: Agent vote endorsements update the belief network
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.events.cross_subscribers import get_registered_subscribers, register_subscriber
from aragora.events.types import StreamEventType

if TYPE_CHECKING:
    from aragora.events.cross_subscribers import CrossSubscriberManager
    from aragora.events.types import StreamEvent

logger = logging.getLogger(__name__)

REASONING_EVENT_SUBSCRIBER_HANDLER_NAMES = frozenset(
    {
        "vote_to_belief",
    }
)


class ReasoningEventSubscriber:
    """Reasoning-domain cross-subscriber: vote -> belief network reactions."""

    def _handle_vote_to_belief(self, event: "StreamEvent") -> None:
        """Vote → Belief network update.

        When an agent casts a vote, update the belief network
        with the position endorsement.
        """
        data = event.data
        agent_name = data.get("agent", "")
        position = data.get("position", "")
        confidence = data.get("confidence", 0.5)
        debate_id = data.get("debate_id", "")

        if not position:
            return

        try:
            from aragora.reasoning.belief import BeliefNetwork

            network = BeliefNetwork()
            if hasattr(network, "update_belief"):
                network.update_belief(
                    agent=agent_name,
                    position=position,
                    confidence=confidence,
                    debate_id=debate_id,
                )
        except ImportError:
            pass  # BeliefNetwork not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Belief network update failed: %s", e)

    def register(self, manager: "CrossSubscriberManager") -> None:
        """Wire the reasoning-domain reaction into ``manager`` (keyed/idempotent)."""
        manager.register(
            "vote_to_belief",
            StreamEventType.VOTE,
            self._handle_vote_to_belief,
        )


def get_reasoning_event_subscriber() -> ReasoningEventSubscriber:
    """Return the ``ReasoningEventSubscriber`` currently wired into the registry.

    Registers a fresh instance first if none is present yet, reusing the
    existing one otherwise so repeated calls resolve to the same instance
    (mirrors ``aragora.knowledge.event_subscribers.get_knowledge_event_subscriber``).
    """
    subscriber = get_registered_subscribers().get("reasoning")
    if not isinstance(subscriber, ReasoningEventSubscriber):
        subscriber = ReasoningEventSubscriber()
        register_subscriber("reasoning", subscriber)
    return subscriber


def register() -> None:
    """(Re-)register this home's subscriber into the domain-free registry.

    Delegates to :func:`get_reasoning_event_subscriber`'s get-or-create so
    repeated calls reuse the existing instance instead of replacing it. Called
    explicitly (not just import side-effect) so registration survives a cached
    re-import after ``reset_registry`` in tests.
    """
    get_reasoning_event_subscriber()


register()
