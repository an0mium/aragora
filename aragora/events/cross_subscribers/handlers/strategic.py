"""
Strategic cross-subsystem feedback loop handlers.

Closes feedback loops between subsystems that emit events (Tiers 1-4)
and subsystems that should consume them:
- Risk Warning → Health Registry: Degrade component health on security anomalies
- Agent Birth/Death → Control Plane: Sync genesis events to agent registry
- Approval Approved → KM Reinforcement: Human approvals boost knowledge confidence
- Budget Alert → Team Selection: Cost constraints limit debate composition
- Alert Escalated → Workflow Brake: Critical alerts pause active workflows
- Meta-Learning Adjusted → Team Selection: Hyperparameter changes inform selection
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.events.types import StreamEvent

logger = logging.getLogger(__name__)


class StrategicHandlersMixin:
    """Mixin providing strategic feedback loop handlers."""

    def _handle_risk_warning_to_health(self, event: StreamEvent) -> None:
        """Risk warning → Health registry degradation.

        When a security anomaly or domain risk is detected, record it
        in the health registry so that affected components are marked
        as degraded. This prevents compromised agents from being
        selected for future debates.
        """
        data = event.data
        risk_type = data.get("risk_type", "unknown")
        severity = data.get("severity", "low")
        component = data.get("component", data.get("agent", ""))
        description = data.get("description", "")[:200]

        if not component:
            return

        # Only degrade health for medium+ severity
        if severity in ("info", "low"):
            return

        logger.info(
            "Risk warning → health degradation: component=%s severity=%s type=%s",
            component,
            severity,
            risk_type,
        )

        try:
            from aragora.resilience.health import get_global_health_registry

            registry = get_global_health_registry()

            # get_or_create ensures the checker exists
            checker = registry.get_or_create(component)
            checker.record_failure(
                error=f"[{risk_type}] {description}",
            )
            logger.debug(
                "Recorded health degradation for %s from risk warning",
                component,
            )
        except ImportError:
            pass  # Health registry not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Health degradation from risk warning failed: %s", e)

    def _handle_genesis_to_control_plane(self, event: StreamEvent) -> None:
        """Agent birth/death/evolution → Control plane registry sync.

        When the genesis system creates, retires, or mutates an agent,
        update the control plane's agent registry so it reflects the
        current population. This ensures the control plane doesn't
        route tasks to dead agents or miss newly born ones.
        """
        data = event.data
        event_subtype = data.get("event_type", data.get("type", ""))
        agent_id = data.get("agent_id", data.get("genome_id", ""))

        if not agent_id:
            return

        logger.debug(
            "Genesis → control plane: event=%s agent=%s",
            event_subtype,
            agent_id,
        )

        try:
            from aragora.control_plane.registry import AgentRegistry

            import asyncio

            registry = AgentRegistry()

            if event_subtype in ("birth", "agent_birth"):
                capabilities = data.get("capabilities", [])
                agent_type = data.get("agent_type", "evolved")
                metadata = {"source": "genesis", "generation": data.get("generation", 0)}

                async def _register():
                    await registry.register(
                        agent_id=agent_id,
                        capabilities=capabilities,
                        model=agent_type,
                        metadata=metadata,
                    )

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_register())
                except RuntimeError:
                    pass  # No event loop; skip async registration
                logger.info("Scheduled born agent %s for control plane registration", agent_id)

            elif event_subtype in ("death", "agent_death"):

                async def _unregister():
                    await registry.unregister(agent_id)

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_unregister())
                except RuntimeError:
                    pass
                logger.info("Scheduled dead agent %s for control plane removal", agent_id)

            elif event_subtype in ("mutation", "evolution", "agent_evolution"):
                new_capabilities = data.get("capabilities", data.get("new_traits", []))

                async def _update():
                    await registry.register(
                        agent_id=agent_id,
                        capabilities=new_capabilities,
                        model=data.get("agent_type", "evolved"),
                        metadata={"evolved": True},
                    )

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_update())
                except RuntimeError:
                    pass
                logger.debug("Scheduled evolved agent %s for control plane update", agent_id)

        except ImportError:
            pass  # Control plane not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Genesis → control plane sync failed: %s", e)

    def _handle_approval_to_km_reinforcement(self, event: StreamEvent) -> None:
        """Human approval → KM confidence reinforcement.

        When a human approves a decision (via the approval flow),
        boost the confidence of related knowledge in the Knowledge Mound.
        This creates a feedback loop where human judgment improves
        the quality of future AI-driven decisions.
        """
        data = event.data
        decision_id = data.get("decision_id", data.get("request_id", ""))
        debate_id = data.get("debate_id", "")
        topic = data.get("topic", data.get("description", ""))

        if not topic:
            return

        logger.debug(
            "Approval → KM reinforcement: decision=%s debate=%s",
            decision_id,
            debate_id,
        )

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if mound is None:
                return

            # Boost importance of knowledge related to the approved decision
            if hasattr(mound, "boost_importance"):
                source = f"debate:{debate_id}" if debate_id else f"decision:{decision_id}"
                mound.boost_importance(
                    source=source,
                    factor=1.15,  # 15% confidence boost from human approval
                )
                logger.info(
                    "Boosted KM confidence for approved decision %s",
                    decision_id or debate_id,
                )
        except ImportError:
            pass  # Knowledge Mound not available
        except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
            logger.debug("KM reinforcement from approval failed: %s", e)

    def _handle_budget_alert_to_team_selection(self, event: StreamEvent) -> None:
        """Budget alert → Team selection constraint.

        When a budget threshold is exceeded, record the constraint
        so that future debate team selections prefer cheaper agents
        and smaller team sizes. This prevents cost overruns while
        maintaining decision quality.
        """
        data = event.data
        alert_type = data.get("alert_type", data.get("type", ""))
        threshold = data.get("threshold", 0.0)
        current_spend = data.get("current_spend", data.get("current", 0.0))
        workspace_id = data.get("workspace_id", "default")

        logger.info(
            "Budget alert → team selection: type=%s spend=%.2f threshold=%.2f workspace=%s",
            alert_type,
            current_spend,
            threshold,
            workspace_id,
        )

        try:
            from aragora.debate.team_selector import TeamSelector

            # Record budget constraint in TeamSelector's class-level state
            if hasattr(TeamSelector, "record_budget_constraint"):
                TeamSelector.record_budget_constraint(
                    workspace_id=workspace_id,
                    alert_type=alert_type,
                    threshold=threshold,
                    current_spend=current_spend,
                )
            else:
                # Fallback: store in module-level dict for TeamSelector to query
                if not hasattr(TeamSelector, "_budget_constraints"):
                    TeamSelector._budget_constraints = {}
                TeamSelector._budget_constraints[workspace_id] = {
                    "alert_type": alert_type,
                    "threshold": threshold,
                    "current_spend": current_spend,
                    "constrained": True,
                }
                logger.debug(
                    "Stored budget constraint for workspace %s (fallback)",
                    workspace_id,
                )
        except ImportError:
            pass  # TeamSelector not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Budget alert → team selection failed: %s", e)

    def _handle_alert_escalated_to_workflow_brake(self, event: StreamEvent) -> None:
        """Alert escalated → Workflow emergency brake.

        When an alert escalates to critical severity, pause all
        active workflows to prevent cascading failures. This is
        the safety valve that stops automated processes when
        something goes seriously wrong.
        """
        data = event.data
        severity = data.get("severity", data.get("new_severity", ""))
        alert_id = data.get("alert_id", "")
        reason = data.get("reason", data.get("message", ""))[:200]

        # Only brake on critical/emergency escalations
        if severity not in ("critical", "emergency", "fatal"):
            return

        logger.warning(
            "Alert escalated → workflow brake: alert=%s severity=%s reason=%s",
            alert_id,
            severity,
            reason,
        )

        try:
            from aragora.workflow.engine import get_workflow_engine

            engine = get_workflow_engine()
            if engine is None:
                return

            if hasattr(engine, "pause_all"):
                engine.pause_all(
                    reason=f"Emergency brake: {reason}",
                )
                logger.warning(
                    "Paused all workflows due to critical alert %s", alert_id
                )
            elif hasattr(engine, "emergency_stop"):
                engine.emergency_stop(reason=f"Alert escalation: {reason}")
                logger.warning(
                    "Emergency stopped workflows due to critical alert %s", alert_id
                )
        except ImportError:
            pass  # Workflow engine not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Workflow emergency brake failed: %s", e)

    def _handle_meta_learning_to_team_selection(self, event: StreamEvent) -> None:
        """Meta-learning adjustment → Team selection recalibration.

        When the MetaLearner auto-tunes hyperparameters based on
        debate outcomes, propagate the adjustments to the team
        selector so it can adapt its scoring weights accordingly.
        This creates a self-improving selection loop.
        """
        data = event.data
        adjustments = data.get("adjustments", {})
        learning_rate = data.get("learning_rate", 0.0)
        total_adjustments = data.get("total_adjustments", 0)

        if not adjustments:
            return

        logger.debug(
            "Meta-learning → team selection: %d adjustments, lr=%.4f",
            total_adjustments,
            learning_rate,
        )

        try:
            from aragora.debate.team_selector import TeamSelector

            # Propagate relevant hyperparameter adjustments
            if hasattr(TeamSelector, "apply_meta_learning"):
                TeamSelector.apply_meta_learning(
                    adjustments=adjustments,
                    learning_rate=learning_rate,
                )
            else:
                # Fallback: store adjustments for TeamSelector to query
                if not hasattr(TeamSelector, "_meta_learning_state"):
                    TeamSelector._meta_learning_state = {}
                TeamSelector._meta_learning_state.update({
                    "adjustments": adjustments,
                    "learning_rate": learning_rate,
                    "total_adjustments": total_adjustments,
                })
                logger.debug(
                    "Stored meta-learning state for team selection (fallback)"
                )
        except ImportError:
            pass  # TeamSelector not available
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("Meta-learning → team selection failed: %s", e)
