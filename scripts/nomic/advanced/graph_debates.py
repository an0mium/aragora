"""
Graph-based debate topology for nomic loop.

Provides alternative debate structure using graph topology instead
of sequential rounds. Agents form nodes connected by argument edges.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GraphDebateRunner:
    """
    Runs graph-based debates using the DebateGraph topology.

    Unlike sequential Arena debates, graph debates allow:
    - Non-linear argument progression
    - Branch points for exploring alternatives
    - Dynamic node creation based on argument flow

    Usage:
        runner = GraphDebateRunner(
            enabled=True,
            available=DEBATE_GRAPH_AVAILABLE,
            orchestrator_class=GraphDebateOrchestrator,
            log_fn=loop._log,
        )

        if runner.is_enabled:
            result = await runner.run_debate(task, agents)
    """

    def __init__(
        self,
        enabled: bool = False,
        available: bool = False,
        orchestrator_class: Any = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize graph debate runner.

        Args:
            enabled: Whether graph debates are enabled
            available: Whether DebateGraph is available
            orchestrator_class: GraphDebateOrchestrator class
            log_fn: Optional logging function
        """
        self._enabled = enabled
        self._available = available
        self._orchestrator_class = orchestrator_class
        self._log = log_fn or (lambda msg: logger.info(msg))

    @property
    def is_enabled(self) -> bool:
        """Whether graph debates are enabled and available."""
        return self._enabled and self._available and self._orchestrator_class is not None

    async def run_debate(
        self,
        task: str,
        agents: list[Any],
    ) -> Optional[Any]:
        """
        Run a graph-based debate.

        Args:
            task: The debate topic/task
            agents: List of agents to participate

        Returns:
            DebateResult if successful, None otherwise
        """
        if not self.is_enabled:
            return None

        try:
            self._log("  [graph] Running graph-based debate...")

            # Create orchestrator on demand with specific agents
            orchestrator = self._orchestrator_class(agents=agents)
            result = await orchestrator.run_debate(task)

            # Verify result has required DebateResult interface
            if not hasattr(result, "consensus_reached") or not hasattr(result, "confidence"):
                self._log("  [graph] Incomplete result - falling back to arena")
                return None

            return result

        except Exception as e:
            self._log(f"  [graph] Debate error: {e}")
            return None


__all__ = ["GraphDebateRunner"]
