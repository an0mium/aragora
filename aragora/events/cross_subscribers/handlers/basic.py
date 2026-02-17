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
        logger.debug(f"Memory retrieval: tier={tier}, cache_hit={hit}")

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
            logger.debug(f"RLM pattern recording failed: {e}")

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
            logger.debug(f"AgentPool weight update failed: {e}")

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

        logger.debug(f"Knowledge indexed: {node_type} {node_id}")

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
                logger.debug(f"Created memory reference for knowledge node {node_id}")
        except ImportError:
            pass  # ContinuumMemory not available
        except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
            logger.debug(f"Memory sync for knowledge failed: {e}")

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
            logger.debug(f"AgentPool calibration update failed: {e}")

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

        logger.debug(f"Evidence collected: {evidence_id} from {source}")

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
                logger.debug(f"Stored evidence insight from {source}")
        except ImportError:
            pass  # ContinuumMemory not available
        except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
            logger.debug(f"Evidence insight storage failed: {e}")

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
                        logger.warning(f"Webhook delivery failed for {webhook.id}: {result.error}")
                except (OSError, ConnectionError, RuntimeError, ValueError, TypeError) as e:
                    logger.error(f"Webhook dispatch error for {webhook.id}: {e}")

        except ImportError:
            logger.debug("Webhook modules not available for event delivery")
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Webhook delivery handler error: {e}")

    def _handle_mound_to_memory(self, event: StreamEvent) -> None:
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

        logger.debug(f"Gauntlet complete: {gauntlet_id} verdict={verdict}")

        try:
            from aragora.notifications.service import notify_gauntlet_completed

            notify_gauntlet_completed(
                gauntlet_id=gauntlet_id,
                verdict=verdict,
                confidence=confidence,
                total_findings=total_findings,
                critical_count=critical_count,
            )
        except ImportError:
            pass  # Notification service not available
        except (RuntimeError, TypeError, ValueError, OSError) as e:
            logger.debug(f"Gauntlet notification failed: {e}")

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
            logger.debug(f"Cost tracking record failed: {e}")

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
            from aragora.debate.selection_feedback import get_selection_feedback_loop

            loop = get_selection_feedback_loop()
            if loop and hasattr(loop, "process_debate_outcome"):
                loop.process_debate_outcome(
                    debate_id=debate_id,
                    confidence=confidence,
                    agents=agents_used,
                )
        except ImportError:
            pass  # SelectionFeedbackLoop not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug(f"Selection feedback learning failed: {e}")

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
            logger.debug(f"Rhetorical analysis failed: {e}")

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
            from aragora.reasoning.belief import get_belief_network

            network = get_belief_network()
            if network and hasattr(network, "update_belief"):
                network.update_belief(
                    agent=agent_name,
                    position=position,
                    confidence=confidence,
                    debate_id=debate_id,
                )
        except ImportError:
            pass  # BeliefNetwork not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug(f"Belief network update failed: {e}")

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

    def _handle_debate_outcome_to_knowledge(self, event: StreamEvent) -> None:
        """Debate end → Knowledge Mound outcome persistence.

        When a debate ends, persist the outcome (winning position,
        key arguments, consensus strength) into the Knowledge Mound
        for future debate context enrichment.
        """
        data = event.data
        debate_id = data.get("debate_id", "")
        consensus = data.get("consensus_reached", False)
        confidence = data.get("confidence", 0.0)
        task = data.get("task", "")

        if not consensus or confidence < 0.6:
            return  # Only persist high-confidence outcomes

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if mound is None:
                return

            outcome_content = {
                "debate_id": debate_id,
                "task": task[:500] if task else "",
                "consensus_reached": consensus,
                "confidence": confidence,
                "winning_position": data.get("winning_position", ""),
                "key_arguments": data.get("key_arguments", [])[:10],
            }

            mound.ingest(
                content=str(outcome_content),
                source=f"debate:{debate_id}",
                node_type="debate_outcome",
                metadata=outcome_content,
            )
            logger.debug(f"Persisted debate outcome to KM: {debate_id}")
        except ImportError:
            pass  # Knowledge Mound not available
        except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
            logger.debug(f"KM outcome persistence failed: {e}")

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

    def _handle_workflow_outcome_to_supermemory(self, event: StreamEvent) -> None:
        """Workflow completion/failure → Supermemory persistence.

        When a workflow completes or fails, store the outcome in supermemory
        for cross-workflow learning. This creates institutional memory of
        what worked and what didn't, enabling future workflows to benefit
        from past experience.
        """
        data = event.data
        workflow_id = data.get("workflow_id", "")
        definition_id = data.get("definition_id", "")
        success = data.get("success", False)

        if not workflow_id:
            return

        logger.info(
            "Storing workflow outcome in supermemory: workflow=%s success=%s",
            workflow_id,
            success,
        )

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if mound is None:
                return

            outcome = {
                "workflow_id": workflow_id,
                "definition_id": definition_id,
                "success": success,
                "duration_ms": data.get("duration_ms", 0),
                "steps_executed": data.get("steps_executed", 0),
                "error": data.get("error", ""),
            }

            status = "completed successfully" if success else "failed"
            content = (
                f"Workflow {definition_id or workflow_id} {status}. "
                f"Steps: {outcome['steps_executed']}, "
                f"Duration: {outcome['duration_ms']}ms"
            )
            if not success and outcome["error"]:
                content += f". Error: {outcome['error'][:200]}"

            mound.ingest(
                content=content,
                source=f"workflow:{workflow_id}",
                node_type="workflow_outcome",
                metadata=outcome,
            )
            logger.debug("Workflow outcome stored in KM: %s", workflow_id)
        except ImportError:
            pass  # Knowledge Mound not available
        except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
            logger.debug(f"KM workflow storage failed: {e}")

    def _handle_tier_demotion_to_revalidation(self, event: StreamEvent) -> None:
        """Memory tier demotion → Re-validation trigger.

        When a memory entry is demoted to slow or glacial tier, trigger
        re-validation to ensure the content is still accurate before
        it becomes harder to access. This prevents stale or incorrect
        knowledge from persisting in lower tiers without review.
        """
        data = event.data
        memory_id = data.get("memory_id", "")
        to_tier = data.get("to_tier", "")
        from_tier = data.get("from_tier", "")

        if not memory_id:
            return

        # Only re-validate on demotion to slow or glacial tiers
        if to_tier not in ("slow", "glacial"):
            return

        logger.info(
            "Tier demotion re-validation: memory=%s from=%s to=%s",
            memory_id,
            from_tier,
            to_tier,
        )

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if mound is None:
                return

            # Mark for re-validation in KM
            if hasattr(mound, "mark_for_revalidation"):
                mound.mark_for_revalidation(
                    source=f"continuum:{memory_id}",
                    reason=f"tier_demotion:{from_tier}->{to_tier}",
                )
                logger.debug(
                    "Marked memory %s for KM re-validation after demotion",
                    memory_id,
                )
        except ImportError:
            pass  # Knowledge Mound not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug(f"KM re-validation trigger failed: {e}")

    def _handle_tier_promotion_to_knowledge(self, event: StreamEvent) -> None:
        """Memory tier promotion → Knowledge Mound notification.

        When a memory entry is promoted to a faster tier, notify KM
        so it can prioritize that knowledge for retrieval and ensure
        the entry's importance is reflected in search rankings.
        """
        data = event.data
        memory_id = data.get("memory_id", "")
        to_tier = data.get("to_tier", "")
        surprise_score = data.get("surprise_score", 0.0)

        if not memory_id:
            return

        logger.debug(
            "Tier promotion notification: memory=%s to=%s surprise=%.3f",
            memory_id,
            to_tier,
            surprise_score,
        )

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if mound is None:
                return

            # Boost importance in KM based on promotion
            if hasattr(mound, "boost_importance"):
                mound.boost_importance(
                    source=f"continuum:{memory_id}",
                    factor=1.0 + surprise_score,
                )
        except ImportError:
            pass  # Knowledge Mound not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug(f"KM importance boost failed: {e}")
