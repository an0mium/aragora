"""Strategy and workflow initialization for debate orchestration.

Handles fabric integration, debate strategy auto-creation, and
post-debate workflow initialization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.core import Agent

logger = logging.getLogger(__name__)


def init_fabric_integration(
    fabric: Optional[Any],
    fabric_config: Optional[Any],
    agents: list["Agent"],
    get_fabric_agents_fn: Callable,
) -> tuple[list["Agent"], Optional[Any], Optional[Any]]:
    """Initialize fabric integration for agent pool management.

    Returns (agents, fabric, fabric_config) tuple.
    """
    if fabric is not None and fabric_config is not None:
        if agents:
            raise ValueError(
                "Cannot specify both 'agents' and 'fabric'/'fabric_config'. "
                "Use either direct agents or fabric-managed agents."
            )
        agents = get_fabric_agents_fn(fabric, fabric_config)
        logger.info(
            f"[fabric] Arena using fabric pool {fabric_config.pool_id} with {len(agents)} agents"
        )
        return agents, fabric, fabric_config
    return agents, None, None


def init_debate_strategy(
    enable_adaptive_rounds: bool,
    debate_strategy: Optional[Any],
    continuum_memory: Optional[Any],
) -> Optional[Any]:
    """Initialize debate strategy for adaptive rounds.

    Auto-creates DebateStrategy if adaptive rounds enabled but no strategy provided.
    """
    if enable_adaptive_rounds and debate_strategy is None:
        try:
            from aragora.debate.strategy import DebateStrategy

            debate_strategy = DebateStrategy(
                continuum_memory=continuum_memory,
            )
            logger.info("debate_strategy auto-initialized for adaptive rounds")
        except ImportError:
            logger.debug("DebateStrategy not available")
            debate_strategy = None
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to initialize DebateStrategy: {e}")
            debate_strategy = None
        except Exception as e:
            logger.exception(f"Unexpected error initializing DebateStrategy: {e}")
            debate_strategy = None

    return debate_strategy


def init_post_debate_workflow(
    enable_post_debate_workflow: bool,
    post_debate_workflow: Optional[Any],
) -> Optional[Any]:
    """Initialize post-debate workflow automation.

    Auto-creates default post-debate workflow if enabled but not provided.
    """
    if enable_post_debate_workflow and post_debate_workflow is None:
        try:
            from aragora.workflow.patterns.post_debate import get_default_post_debate_workflow

            post_debate_workflow = get_default_post_debate_workflow()
            logger.debug("[arena] Auto-created default post-debate workflow")
        except ImportError:
            logger.warning("[arena] Post-debate workflow enabled but pattern not available")

    return post_debate_workflow
