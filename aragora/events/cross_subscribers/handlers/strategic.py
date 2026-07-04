"""
Strategic cross-subsystem feedback loop handlers.

Closes feedback loops between subsystems that emit events (Tiers 1-4)
and subsystems that should consume them:
- Risk Warning → Health Registry: Degrade component health on security anomalies
- Agent Birth/Death → Control Plane: Sync genesis events to agent registry

The Budget Alert → Team Selection and Meta-Learning Adjusted → Team Selection
reactions relocated to ``aragora.debate.event_subscribers`` (P4a Batch E4
relocate-UP); see that module for those handlers. The Alert Escalated →
Workflow Brake reaction relocated to ``aragora.workflow.event_subscribers``
(P4a Batch E5 relocate-UP); see that module for that handler.
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
