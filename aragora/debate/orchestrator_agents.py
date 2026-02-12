"""Agent selection, quality filtering, hierarchy, and fabric helpers for Arena.

Extracted from orchestrator.py to reduce its size. These functions handle
ML-based agent delegation, performance-based team selection, quality gate
filtering, consensus estimation, critic selection, Gastown-style
hierarchy role assignment, and fabric agent retrieval.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from aragora.logging_config import get_logger as get_structured_logger

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.context import DebateContext
    from aragora.debate.hierarchy import AgentHierarchy, HierarchyConfig

logger = get_structured_logger(__name__)


def select_debate_team(
    agents: list[Agent],
    env: Any,
    extract_domain_fn: Any,
    enable_ml_delegation: bool,
    ml_delegation_strategy: Any,
    protocol: Any,
    use_performance_selection: bool,
    agent_pool: Any,
) -> list[Agent]:
    """Select debate team using ML delegation or AgentPool.

    Priority:
    1. ML delegation (if *enable_ml_delegation* is ``True``)
    2. Performance selection via AgentPool (if *use_performance_selection* is ``True``)
    3. Original requested agents

    Args:
        agents: The candidate agents.
        env: Debate environment.
        extract_domain_fn: Callable returning the debate domain string.
        enable_ml_delegation: Whether ML delegation is enabled.
        ml_delegation_strategy: The ML strategy instance (or ``None``).
        protocol: Debate protocol.
        use_performance_selection: Whether performance selection is enabled.
        agent_pool: The AgentPool instance.

    Returns:
        The selected list of agents.
    """
    if enable_ml_delegation and ml_delegation_strategy:
        try:
            selected = ml_delegation_strategy.select_agents(
                task=env.task,
                agents=agents,
                context={
                    "domain": extract_domain_fn(),
                    "protocol": protocol,
                },
                max_agents=len(agents),
            )
            logger.debug(
                f"[ml] Selected {len(selected)} agents via ML delegation: "
                f"{[a.name for a in selected]}"
            )
            return selected
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"[ml] ML delegation failed with data error, falling back: {e}")
        except Exception as e:
            logger.exception(f"[ml] Unexpected ML delegation error, falling back: {e}")

    if use_performance_selection:
        return agent_pool.select_team(
            domain=extract_domain_fn(),
            team_size=len(agents),
        )

    return agents


def filter_responses_by_quality(
    responses: list[tuple[str, str]],
    enable_quality_gates: bool,
    ml_quality_gate: Any,
    task: str,
    context: str = "",
) -> list[tuple[str, str]]:
    """Filter responses using ML quality gate if enabled.

    Args:
        responses: List of (agent_name, response_text) tuples.
        enable_quality_gates: Whether quality gates are enabled.
        ml_quality_gate: The quality gate instance (or ``None``).
        task: The debate task for fallback context.
        context: Optional task context for quality assessment.

    Returns:
        Filtered list containing only high-quality responses.
    """
    if not enable_quality_gates or not ml_quality_gate:
        return responses

    try:
        filtered = ml_quality_gate.filter_responses(responses, context=context or task)
        removed = len(responses) - len(filtered)
        if removed > 0:
            logger.debug(f"[ml] Quality gate filtered {removed} low-quality responses")
        return filtered
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning(f"[ml] Quality gate failed with data error, keeping all responses: {e}")
        return responses
    except Exception as e:
        logger.exception(f"[ml] Unexpected quality gate error, keeping all responses: {e}")
        return responses


def should_terminate_early(
    responses: list[tuple[str, str]],
    current_round: int,
    enable_consensus_estimation: bool,
    ml_consensus_estimator: Any,
    protocol: Any,
    task: str,
) -> bool:
    """Check if debate should terminate early based on consensus estimation.

    Args:
        responses: List of (agent_name, response_text) tuples.
        current_round: Current debate round number.
        enable_consensus_estimation: Whether consensus estimation is enabled.
        ml_consensus_estimator: The estimator instance (or ``None``).
        protocol: Debate protocol (provides ``rounds``).
        task: The debate task for context.

    Returns:
        ``True`` if consensus is highly likely and safe to terminate early.
    """
    if not enable_consensus_estimation or not ml_consensus_estimator:
        return False

    try:
        should_stop = ml_consensus_estimator.should_terminate_early(
            responses=responses,
            current_round=current_round,
            total_rounds=protocol.rounds,
            context=task,
        )
        if should_stop:
            logger.info(
                f"[ml] Consensus estimator recommends early termination at round "
                f"{current_round}/{protocol.rounds}"
            )
        return should_stop
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"[ml] Consensus estimation failed with data error: {e}")
        return False
    except Exception as e:
        logger.exception(f"[ml] Unexpected consensus estimation error: {e}")
        return False


def init_agent_hierarchy(
    enable_agent_hierarchy: bool,
    hierarchy_config: HierarchyConfig | None,
) -> AgentHierarchy | None:
    """Create an AgentHierarchy for Gastown-style role assignment.

    Args:
        enable_agent_hierarchy: Whether to enable role assignment.
        hierarchy_config: Optional custom configuration.

    Returns:
        An ``AgentHierarchy`` instance or ``None``.
    """
    from aragora.debate.hierarchy import AgentHierarchy, HierarchyConfig as HC

    if not enable_agent_hierarchy:
        return None

    config = hierarchy_config or HC()
    hierarchy = AgentHierarchy(config)
    logger.info(
        f"[hierarchy] AgentHierarchy initialized "
        f"(max_orchestrators={config.max_orchestrators}, "
        f"max_monitors={config.max_monitors})"
    )
    return hierarchy


def assign_hierarchy_roles(
    ctx: DebateContext,
    enable_agent_hierarchy: bool,
    hierarchy: AgentHierarchy | None,
    task_type: str | None = None,
) -> None:
    """Assign hierarchy roles to agents for this debate.

    Called during ``_run_inner`` after agent selection to assign roles.
    Results are stored in ``ctx.hierarchy_assignments``.

    Args:
        ctx: Debate context to update.
        enable_agent_hierarchy: Whether hierarchy is enabled.
        hierarchy: The AgentHierarchy instance (or ``None``).
        task_type: Optional task type for affinity matching.
    """
    if not enable_agent_hierarchy or hierarchy is None:
        return

    try:
        from aragora.routing.selection import AgentProfile

        profiles = []
        for agent in ctx.agents:
            profile = AgentProfile(
                name=agent.name,
                agent_type=getattr(agent, "provider", "unknown"),
                elo_rating=getattr(agent, "elo_rating", 1500.0),
                capabilities=getattr(agent, "capabilities", set()),
                task_affinity=getattr(agent, "task_affinity", {}),
            )
            profiles.append(profile)

        assignments = hierarchy.assign_roles(
            debate_id=ctx.debate_id,
            agents=profiles,
            task_type=task_type,
        )

        ctx.hierarchy_assignments = assignments

        role_summary = {
            role.value: [name for name, assign in assignments.items() if assign.role == role]
            for role in set(a.role for a in assignments.values())
        }
        logger.info(f"[hierarchy] Roles assigned for debate {ctx.debate_id}: {role_summary}")

    except (ImportError, ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning(f"[hierarchy] Role assignment failed: {e}")
        ctx.hierarchy_assignments = {}


def get_fabric_agents_sync(fabric: Any, fabric_config: Any) -> list:
    """Get agents from fabric pool synchronously.

    This is a sync helper for use during ``__init__``. For async contexts,
    use ``FabricDebateRunner`` instead.

    Args:
        fabric: AgentFabric instance.
        fabric_config: FabricDebateConfig with pool_id.

    Returns:
        List of FabricAgentAdapter instances that implement the Agent protocol.
    """
    from aragora.debate.fabric_integration import FabricAgentAdapter

    async def get_agents():
        pool = await fabric.get_pool(fabric_config.pool_id)
        if not pool:
            raise ValueError(f"Pool {fabric_config.pool_id} not found")

        max_agents = getattr(fabric_config, "max_agents", 10)
        agents = []
        for agent_id in pool.current_agents[:max_agents]:
            adapter = FabricAgentAdapter(
                fabric=fabric,
                agent_id=agent_id,
                model=pool.model,
            )
            agents.append(adapter)
        return agents

    try:
        asyncio.get_running_loop()
        # If we get here, there IS a running loop - we should not use sync helper
        raise RuntimeError("Cannot use sync helper in async context")
    except RuntimeError as e:
        # Only catch the "no running event loop" error, not our own error
        if "Cannot use sync helper" in str(e):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(get_agents())
        finally:
            loop.close()
