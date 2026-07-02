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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from aragora.events.types import StreamEvent
    from aragora.memory.continuum import ContinuumMemory

logger = logging.getLogger(__name__)


class CompressorProtocol(Protocol):
    """Protocol for RLM compressor with access pattern recording."""

    def record_access_pattern(
        self,
        tier: str,
        cache_hit: bool,
        importance: float,
    ) -> None:
        """Record a memory access pattern for compression optimization."""
        ...


class AgentPoolProtocol(Protocol):
    """Protocol for agent pool with ELO and calibration updates."""

    def update_elo_weight(self, agent_name: str, elo: float) -> None:
        """Update the ELO weight for an agent."""
        ...

    def update_calibration(
        self,
        agent_name: str,
        score: float,
        brier_score: float | None = None,
    ) -> None:
        """Update calibration data for an agent."""
        ...


class BasicHandlersMixin:
    """Mixin providing basic cross-subsystem event handlers."""

    def _handle_memory_to_rlm(self, event: StreamEvent) -> None:
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
        logger.debug("Memory retrieval: tier=%s, cache_hit=%s", tier, hit)

        # Update RLM compression hints based on access patterns
        try:
            import aragora.rlm.compressor as compressor_module

            # get_compressor may not exist yet (planned feature)
            get_compressor = getattr(compressor_module, "get_compressor", None)
            if get_compressor is None:
                return

            compressor: CompressorProtocol | None = get_compressor()
            if compressor and hasattr(compressor, "record_access_pattern"):
                compressor.record_access_pattern(
                    tier=tier,
                    cache_hit=hit,
                    importance=importance,
                )
        except ImportError:
            pass  # RLM module not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("RLM pattern recording failed: %s", e)

    def _handle_elo_to_debate(self, event: StreamEvent) -> None:
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
            import aragora.debate.agent_pool as agent_pool_module

            # get_agent_pool may not exist yet (planned feature)
            get_agent_pool = getattr(agent_pool_module, "get_agent_pool", None)
            if get_agent_pool is None:
                return

            pool: AgentPoolProtocol | None = get_agent_pool()
            if pool and hasattr(pool, "update_elo_weight"):
                pool.update_elo_weight(agent_name, new_elo)
        except ImportError:
            pass  # AgentPool module not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("AgentPool weight update failed: %s", e)

    def _handle_knowledge_to_memory(self, event: StreamEvent) -> None:
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

    def _handle_calibration_to_agent(self, event: StreamEvent) -> None:
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
            from aragora.debate.agent_pool import get_agent_pool

            pool = cast("AgentPoolProtocol | None", get_agent_pool())
            if pool and hasattr(pool, "update_calibration"):
                pool.update_calibration(
                    agent_name=agent_name,
                    score=calibration_score,
                    brier_score=brier_score,
                )
        except (ImportError, AttributeError):
            pass  # AgentPool or get_agent_pool not available
        except (RuntimeError, TypeError, ValueError) as e:
            logger.debug("AgentPool calibration update failed: %s", e)

    def _handle_evidence_to_insight(self, event: StreamEvent) -> None:
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

    def _handle_webhook_delivery(self, event: StreamEvent) -> None:
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
                        logger.warning(
                            "Webhook delivery failed for %s: %s", webhook.id, result.error
                        )
                except (OSError, ConnectionError, RuntimeError, ValueError, TypeError) as e:
                    logger.error("Webhook dispatch error for %s: %s", webhook.id, e)

        except ImportError:
            logger.debug("Webhook modules not available for event delivery")
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.debug("Webhook delivery handler error: %s", e)

    def _handle_mound_to_memory(self, event: StreamEvent) -> None:
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

    def _handle_gauntlet_complete_to_notification(self, event: StreamEvent) -> None:
        """Gauntlet complete → Notification dispatch.

        When a gauntlet stress-test finishes, notify stakeholders with
        the verdict and finding counts.
        """
        data = event.data
        gauntlet_id = data.get("gauntlet_id", "")
        verdict = data.get("verdict", "unknown")
        confidence = data.get("confidence", 0.0)
        total_findings = data.get("total_findings", 0)
        critical_count = data.get("critical_count", 0)

        logger.debug("Gauntlet complete: %s verdict=%s", gauntlet_id, verdict)

        try:
            import asyncio

            from aragora.notifications.service import notify_gauntlet_completed

            coro = notify_gauntlet_completed(
                gauntlet_id=gauntlet_id,
                verdict=verdict,
                confidence=confidence,
                total_findings=total_findings,
                critical_count=critical_count,
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                asyncio.run(coro)
        except ImportError:
            pass  # Notification service not available
        except (RuntimeError, TypeError, ValueError, OSError) as e:
            logger.debug("Gauntlet notification failed: %s", e)

    def _handle_debate_end_to_cost_tracking(self, event: StreamEvent) -> None:
        """Debate end → Cost tracking record.

        When a debate ends, record the total cost for billing
        and usage analytics.
        """
        data = event.data
        debate_id = data.get("debate_id", "")
        total_cost = data.get("total_cost", 0.0)
        total_tokens = data.get("total_tokens", 0)

        if not total_cost:
            return

        logger.debug(f"Recording debate cost: {debate_id} ${total_cost:.4f}")

        try:
            from aragora.billing.cost_tracker import get_cost_tracker

            tracker = get_cost_tracker()
            if tracker and hasattr(tracker, "record_debate_total"):
                tracker.record_debate_total(
                    debate_id=debate_id,
                    total_cost=total_cost,
                    total_tokens=total_tokens,
                )
        except ImportError:
            pass  # CostTracker not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Cost tracking record failed: %s", e)

    def _handle_consensus_to_learning(self, event: StreamEvent) -> None:
        """Consensus → Selection feedback learning.

        When consensus is reached, feed the outcome to the
        SelectionFeedbackLoop for performance-based agent selection.
        """
        data = event.data
        debate_id = data.get("debate_id", "")
        confidence = data.get("confidence", 0.0)
        agents_used = data.get("agents", [])

        if not agents_used or confidence < 0.5:
            return

        logger.debug(f"Learning from consensus: {debate_id} confidence={confidence:.2f}")

        try:
            from aragora.debate.selection_feedback import SelectionFeedbackLoop

            loop = SelectionFeedbackLoop()
            if hasattr(loop, "process_debate_outcome"):
                loop.process_debate_outcome(
                    debate_id=debate_id,
                    participants=agents_used,
                    winner=None,
                    confidence=confidence,
                )
        except ImportError:
            pass  # SelectionFeedbackLoop not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Selection feedback learning failed: %s", e)

    def _handle_agent_message_to_rhetorical(self, event: StreamEvent) -> None:
        """Agent message → Rhetorical analysis.

        When an agent sends a message, pass it to the RhetoricalObserver
        for argumentation quality analysis.
        """
        data = event.data
        agent_name = data.get("agent", "")
        content = data.get("content", "")

        if not content or len(content) < 20:
            return

        try:
            from aragora.debate.rhetorical_observer import get_rhetorical_observer

            observer = get_rhetorical_observer()
            if observer and hasattr(observer, "analyze_message"):
                observer.analyze_message(
                    agent_name=agent_name,
                    content=content,
                    metadata=data,
                )
        except ImportError:
            pass  # RhetoricalObserver not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Rhetorical analysis failed: %s", e)

    def _handle_vote_to_belief(self, event: StreamEvent) -> None:
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

    def _handle_debate_end_to_explainability(self, event: StreamEvent) -> None:
        """Debate end → Explainability auto-trigger.

        When a debate ends, log the event for downstream explainability
        processing. The actual explanation generation happens in
        ArenaExtensions._auto_generate_explanation.
        """
        data = event.data
        debate_id = data.get("debate_id", "")
        consensus = data.get("consensus_reached", False)
        confidence = data.get("confidence", 0.0)

        logger.debug(
            f"Debate ended for explainability: {debate_id} "
            f"consensus={consensus} confidence={confidence:.2f}"
        )

    def _handle_debate_end_to_workflow(self, event: StreamEvent) -> None:
        """Debate end -> post-debate workflow automation.

        Delegates to PostDebateWorkflowSubscriber to classify the debate
        outcome and trigger the appropriate workflow template.
        """
        try:
            from aragora.events.subscribers.workflow_automation import (
                PostDebateWorkflowSubscriber,
            )

            subscriber = PostDebateWorkflowSubscriber()
            subscriber.handle_debate_end(event)

            logger.debug(
                "Post-debate workflow processed: events=%d workflows=%d errors=%d",
                subscriber.stats["events_processed"],
                subscriber.stats["workflows_triggered"],
                subscriber.stats["errors"],
            )
        except ImportError:
            logger.debug("PostDebateWorkflowSubscriber not available")
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Debate end -> workflow handler error: %s", e)
