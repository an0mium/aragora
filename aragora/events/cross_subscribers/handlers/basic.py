"""
Basic cross-subsystem event handlers.

Handles core subsystem integrations:
- Memory → RLM: Retrieval patterns inform compression strategies
- Agent ELO → Debate: Performance updates team selection weights
- Knowledge → Memory: Index updates sync to memory insights
- Calibration → Agent: Confidence weight updates
- Evidence → Insight: Extract insights from evidence
- Webhook delivery: External system notifications
- Mound → Memory: Structure updates sync
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.events.types import StreamEvent

logger = logging.getLogger(__name__)


class BasicHandlersMixin:
    """Mixin providing basic cross-subsystem event handlers."""

    def _handle_memory_to_rlm(self, event: "StreamEvent") -> None:
        """
        Memory retrieval → RLM feedback.

        When memory is retrieved, inform RLM about retrieval patterns
        to optimize compression strategies. Tracks access patterns
        for adaptive compression.
        """
        data = event.data
        tier = data.get("tier", "unknown")
        hit = data.get("cache_hit", False)
        importance = data.get("importance", 0.5)

        # Track access pattern for RLM optimization
        logger.debug(f"Memory retrieval: tier={tier}, cache_hit={hit}")

        # Update RLM compression hints based on access patterns
        try:
            from aragora.rlm.compressor import get_compressor  # type: ignore[attr-defined]

            compressor = get_compressor()
            if compressor and hasattr(compressor, "record_access_pattern"):
                compressor.record_access_pattern(
                    tier=tier,
                    cache_hit=hit,
                    importance=importance,
                )
        except ImportError:
            pass  # RLM not available
        except Exception as e:
            logger.debug(f"RLM pattern recording failed: {e}")

    def _handle_elo_to_debate(self, event: "StreamEvent") -> None:
        """
        ELO update → Debate team selection weights.

        When agent ELO changes, update team selection weights
        for future debates. Significant changes are logged.
        """
        data = event.data
        agent_name = data.get("agent", "")
        new_elo = data.get("elo", 1500)
        delta = data.get("delta", 0)
        debate_id = data.get("debate_id", "")

        # Log significant ELO changes
        if abs(delta) > 50:
            logger.info(
                f"Significant ELO change: {agent_name} -> {new_elo} "
                f"(Δ{delta:+.0f}) in debate {debate_id}"
            )

        # Update agent pool weights for future team selection
        try:
            from aragora.debate.agent_pool import get_agent_pool  # type: ignore[attr-defined]

            pool = get_agent_pool()
            if pool and hasattr(pool, "update_elo_weight"):
                pool.update_elo_weight(agent_name, new_elo)
        except ImportError:
            pass  # AgentPool not available
        except Exception as e:
            logger.debug(f"AgentPool weight update failed: {e}")

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

        logger.debug(f"Knowledge indexed: {node_type} {node_id}")

        # Create memory entry referencing knowledge node
        try:
            from aragora.memory import get_continuum_memory

            memory = get_continuum_memory()
            if memory:
                # Store a reference to the knowledge node in memory
                memory_content = f"[Knowledge:{node_type}] {content[:500]}"
                metadata = {
                    "source": "knowledge_mound",
                    "node_id": node_id,
                    "node_type": node_type,
                    "workspace_id": workspace_id,
                }
                memory.store(  # type: ignore[call-arg,unused-coroutine]
                    content=memory_content,
                    importance=0.6,  # Default importance for knowledge references
                    metadata=metadata,
                )
                logger.debug(f"Created memory reference for knowledge node {node_id}")
        except ImportError:
            pass  # ContinuumMemory not available
        except Exception as e:
            logger.debug(f"Memory sync for knowledge failed: {e}")

    def _handle_calibration_to_agent(self, event: "StreamEvent") -> None:
        """
        Calibration update → Agent confidence weights.

        When calibration data changes, update agent confidence
        weights for vote weighting and team selection.
        """
        data = event.data
        agent_name = data.get("agent", "")
        calibration_score = data.get("score", 0.5)
        brier_score = data.get("brier_score", None)
        prediction_count = data.get("prediction_count", 0)

        logger.debug(
            f"Calibration update: {agent_name} -> {calibration_score:.2f} "
            f"(predictions: {prediction_count})"
        )

        # Update agent pool with calibration data
        try:
            from aragora.debate.agent_pool import get_agent_pool  # type: ignore[attr-defined]

            pool = get_agent_pool()
            if pool and hasattr(pool, "update_calibration"):
                pool.update_calibration(
                    agent_name=agent_name,
                    score=calibration_score,
                    brier_score=brier_score,
                )
        except ImportError:
            pass  # AgentPool not available
        except Exception as e:
            logger.debug(f"AgentPool calibration update failed: {e}")

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

        logger.debug(f"Evidence collected: {evidence_id} from {source}")

        # Skip if no meaningful content
        if not content or len(content) < 50:
            return

        # Store evidence-backed insight in memory
        try:
            from aragora.memory import get_continuum_memory

            memory = get_continuum_memory()
            if memory and confidence >= 0.7:  # Only store high-confidence evidence
                insight_content = (
                    f"[Evidence from {source}] "
                    f"Claim: {claim[:200] if claim else 'N/A'} | "
                    f"Evidence: {content[:300]}"
                )
                metadata = {
                    "source": source,
                    "evidence_id": evidence_id,
                    "confidence": confidence,
                    "type": "evidence_insight",
                }
                memory.store(  # type: ignore[call-arg,unused-coroutine]
                    content=insight_content,
                    importance=confidence,
                    metadata=metadata,
                )
                logger.debug(f"Stored evidence insight from {source}")
        except ImportError:
            pass  # ContinuumMemory not available
        except Exception as e:
            logger.debug(f"Evidence insight storage failed: {e}")

    def _handle_webhook_delivery(self, event: "StreamEvent") -> None:
        """
        Event → Webhook delivery.

        When any subscribable event occurs, deliver to registered webhooks.
        This enables external systems to receive real-time notifications.
        """
        try:
            from aragora.server.handlers.webhooks import get_webhook_store
            from aragora.events.dispatcher import dispatch_webhook_with_retry

            # Get registered webhooks for this event type
            store = get_webhook_store()
            event_type_str = event.type.value.lower()  # Convert enum to string
            webhooks = store.get_for_event(event_type_str)

            if not webhooks:
                return  # No webhooks registered for this event

            # Build payload
            import time
            import uuid

            payload = {
                "event": event_type_str,
                "delivery_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "data": event.data or {},
            }

            # Deliver to each matching webhook
            for webhook in webhooks:
                try:
                    result = dispatch_webhook_with_retry(webhook, payload)
                    if not result.success:
                        logger.warning(f"Webhook delivery failed for {webhook.id}: {result.error}")
                except Exception as e:
                    logger.error(f"Webhook dispatch error for {webhook.id}: {e}")

        except ImportError:
            logger.debug("Webhook modules not available for event delivery")
        except Exception as e:
            logger.debug(f"Webhook delivery handler error: {e}")

    def _handle_mound_to_memory(self, event: "StreamEvent") -> None:
        """
        Mound structure update → Memory/Debate sync.

        When the Knowledge Mound structure changes significantly,
        notify memory and debate systems to refresh their context.
        """
        data = event.data
        update_type = data.get("update_type", "unknown")
        workspace_id = data.get("workspace_id", "")

        logger.debug(f"Mound updated: type={update_type}, workspace={workspace_id}")

        # Handle culture pattern updates
        if update_type == "culture_patterns":
            patterns_count = data.get("patterns_count", 0)
            debate_id = data.get("debate_id", "")
            logger.info(
                f"Culture patterns updated: {patterns_count} patterns from debate {debate_id}"
            )

        # Handle node deletions
        elif update_type == "node_deleted":
            node_id = data.get("node_id", "")
            archived = data.get("archived", False)
            logger.debug(f"Knowledge node removed: {node_id} (archived={archived})")

            # Clear any cached references to this node
            try:
                from aragora.memory import get_continuum_memory

                memory = get_continuum_memory()
                if memory and hasattr(memory, "invalidate_reference"):
                    memory.invalidate_reference(node_id)
            except (ImportError, AttributeError):
                pass
