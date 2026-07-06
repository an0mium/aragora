"""
Default debate executor factory for the queue system.

Builds the async job-executor callable that ``DebateWorker``
(``aragora.queue.worker``) runs debate jobs through, wiring the domain-free
queue transport to the Arena. Lives in ``aragora.debate`` rather than
``aragora.queue`` because it imports ``aragora.agents.base`` and
``aragora.debate.orchestrator``, both domain-layer packages
(docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md section 10 Q2).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, cast

from aragora.exceptions import InfrastructureError
from aragora.queue.base import Job

if TYPE_CHECKING:
    from aragora.agents.base import AgentType


async def create_default_executor() -> Callable[[Job], Coroutine[Any, Any, dict[str, Any]]]:
    """
    Create a default debate executor.

    This imports the debate infrastructure and creates an executor
    that runs debates using the Arena.

    Returns:
        An async function that executes debate jobs
    """

    async def execute_debate(job: Job) -> dict[str, Any]:
        """Execute a debate from a job."""
        from aragora.queue.job import DebateResult, get_debate_payload

        payload = get_debate_payload(job)

        try:
            from aragora.agents.base import create_agent
            from aragora.core import DebateProtocol, Environment
            from aragora.debate.orchestrator import Arena
        except ImportError as e:
            raise InfrastructureError(f"Debate infrastructure not available: {e}")

        env = Environment(task=payload.question)
        protocol = cast(Any, DebateProtocol)(
            rounds=payload.rounds,
            consensus=cast(Any, payload.consensus),
        )

        agents_list = []
        for agent_type in payload.agents:
            agent = create_agent(cast("AgentType", agent_type))
            if agent is not None:
                agents_list.append(agent)

        start_time = time.time()
        arena = Arena(env, agents=agents_list, protocol=protocol)
        result = await arena.run()

        duration = time.time() - start_time

        debate_result = DebateResult(
            debate_id=result.debate_id if hasattr(result, "debate_id") else job.id,
            consensus_reached=(
                result.consensus_reached if hasattr(result, "consensus_reached") else False
            ),
            final_answer=result.final_answer if hasattr(result, "final_answer") else None,
            confidence=result.confidence if hasattr(result, "confidence") else 0.0,
            rounds_used=result.rounds_used if hasattr(result, "rounds_used") else payload.rounds,
            participants=payload.agents,
            duration_seconds=duration,
        )

        return debate_result.to_dict()

    return execute_debate
