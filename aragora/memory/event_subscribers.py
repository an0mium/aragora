"""Memory-domain event-subscriber home (P4a EventBus inversion, Batch E3).

Memory-coupled cross-subsystem reactions, relocated here from infrastructure
``aragora.events.cross_subscribers.handlers.basic`` so the memory-coupled
reactions live in their DOMAIN home. The module self-registers via the
domain-free registry (``aragora.events.cross_subscribers.register_subscriber`` -
domain -> infrastructure, downward = legal); the layered bootstraps import it so
``CrossSubscriberManager.apply_registered_subscribers`` wires these reactions in.

Per the relocate-UP no-shim exemption (AGENTS.md "P4a Contracts-Thread Shared Rules"
and docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §8) there is NO re-export shim at
the old path; every consumer is repointed instead.

Handles:
- Knowledge → Memory: Index updates sync to memory as cross-referencing entries
- Evidence → Insight: High-confidence evidence stored as memory insights
- Mound → Memory: Structure updates (culture patterns, node deletions) sync
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aragora.events.cross_subscribers import get_registered_subscribers, register_subscriber
from aragora.events.types import StreamEventType

if TYPE_CHECKING:
    from aragora.events.cross_subscribers import CrossSubscriberManager
    from aragora.events.types import StreamEvent
    from aragora.memory.continuum import ContinuumMemory

logger = logging.getLogger(__name__)

MEMORY_EVENT_SUBSCRIBER_HANDLER_NAMES = frozenset(
    {
        "knowledge_to_memory",
        "evidence_to_insight",
        "mound_to_memory",
    }
)


class MemoryEventSubscriber:
    """Memory-domain cross-subscriber: knowledge/evidence/mound -> memory reactions."""

    def _handle_knowledge_to_memory(self, event: "StreamEvent") -> None:
        """
        Knowledge indexed → Memory sync.

        When new knowledge is indexed, create corresponding
        memory entries for cross-referencing in debates.
        """
        data = event.data
        node_id = data.get("node_id", "")
        content = data.get("content", "")
        node_type = data.get("node_type", "fact")
        workspace_id = data.get("workspace_id", "default")

        logger.debug("Knowledge indexed: %s %s", node_type, node_id)

        # Create memory entry referencing knowledge node
        try:
            from aragora.memory import get_continuum_memory

            memory: ContinuumMemory | None = get_continuum_memory()
            if memory:
                # Store a reference to the knowledge node in memory
                memory_content = f"[Knowledge:{node_type}] {content[:500]}"
                entry_metadata: dict[str, Any] = {
                    "source": "knowledge_mound",
                    "node_id": node_id,
                    "node_type": node_type,
                    "workspace_id": workspace_id,
                }
                # Use synchronous add() since we're in a sync handler
                memory.add(
                    id=f"km_{node_id}",
                    content=memory_content,
                    importance=0.6,  # Default importance for knowledge references
                    metadata=entry_metadata,
                )
                logger.debug("Created memory reference for knowledge node %s", node_id)
        except ImportError:
            pass  # ContinuumMemory not available
        except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
            logger.debug("Memory sync for knowledge failed: %s", e)

    def _handle_evidence_to_insight(self, event: "StreamEvent") -> None:
        """
        Evidence found → Insight extraction.

        When new evidence is collected, attempt to extract
        insights that can be stored in memory for future debates.
        """
        data = event.data
        evidence_id = data.get("evidence_id", "")
        source = data.get("source", "")
        content = data.get("content", "")
        claim = data.get("claim", "")
        confidence = data.get("confidence", 0.5)

        logger.debug("Evidence collected: %s from %s", evidence_id, source)

        # Skip if no meaningful content
        if not content or len(content) < 50:
            return

        # Store evidence-backed insight in memory
        try:
            from aragora.memory import get_continuum_memory

            memory: ContinuumMemory | None = get_continuum_memory()
            if memory and confidence >= 0.7:  # Only store high-confidence evidence
                insight_content = (
                    f"[Evidence from {source}] "
                    f"Claim: {claim[:200] if claim else 'N/A'} | "
                    f"Evidence: {content[:300]}"
                )
                insight_metadata: dict[str, Any] = {
                    "source": source,
                    "evidence_id": evidence_id,
                    "confidence": confidence,
                    "type": "evidence_insight",
                }
                # Use synchronous add() since we're in a sync handler
                memory.add(
                    id=f"evidence_{evidence_id}",
                    content=insight_content,
                    importance=confidence,
                    metadata=insight_metadata,
                )
                logger.debug("Stored evidence insight from %s", source)
        except ImportError:
            pass  # ContinuumMemory not available
        except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
            logger.debug("Evidence insight storage failed: %s", e)

    def _handle_mound_to_memory(self, event: "StreamEvent") -> None:
        """
        Mound structure update → Memory/Debate sync.

        When the Knowledge Mound structure changes significantly,
        notify memory and debate systems to refresh their context.
        """
        data = event.data
        update_type = data.get("update_type", "unknown")
        workspace_id = data.get("workspace_id", "")

        logger.debug("Mound updated: type=%s, workspace=%s", update_type, workspace_id)

        # Handle culture pattern updates
        if update_type == "culture_patterns":
            patterns_count = data.get("patterns_count", 0)
            debate_id = data.get("debate_id", "")
            logger.info(
                "Culture patterns updated: %s patterns from debate %s", patterns_count, debate_id
            )

        # Handle node deletions
        elif update_type == "node_deleted":
            node_id = data.get("node_id", "")
            archived = data.get("archived", False)
            logger.debug("Knowledge node removed: %s (archived=%s)", node_id, archived)

            # Clear any cached references to this node
            try:
                from aragora.memory import get_continuum_memory

                memory = get_continuum_memory()
                if memory and hasattr(memory, "invalidate_reference"):
                    memory.invalidate_reference(node_id)
            except (ImportError, AttributeError):
                pass

    def register(self, manager: "CrossSubscriberManager") -> None:
        """Wire the memory-domain reactions into ``manager`` (keyed/idempotent)."""
        manager.register(
            "knowledge_to_memory",
            StreamEventType.KNOWLEDGE_INDEXED,
            self._handle_knowledge_to_memory,
        )
        manager.register(
            "evidence_to_insight",
            StreamEventType.EVIDENCE_FOUND,
            self._handle_evidence_to_insight,
        )
        manager.register(
            "mound_to_memory",
            StreamEventType.MOUND_UPDATED,
            self._handle_mound_to_memory,
        )


def get_memory_event_subscriber() -> MemoryEventSubscriber:
    """Return the ``MemoryEventSubscriber`` currently wired into the registry.

    Registers a fresh instance first if none is present yet, reusing the
    existing one otherwise so repeated calls resolve to the same instance
    (mirrors ``aragora.knowledge.event_subscribers.get_knowledge_event_subscriber``).
    """
    subscriber = get_registered_subscribers().get("memory")
    if not isinstance(subscriber, MemoryEventSubscriber):
        subscriber = MemoryEventSubscriber()
        register_subscriber("memory", subscriber)
    return subscriber


def register() -> None:
    """(Re-)register this home's subscriber into the domain-free registry.

    Delegates to :func:`get_memory_event_subscriber`'s get-or-create so repeated
    calls reuse the existing instance instead of replacing it. Called explicitly
    (not just import side-effect) so registration survives a cached re-import
    after ``reset_registry`` in tests.
    """
    get_memory_event_subscriber()


register()
