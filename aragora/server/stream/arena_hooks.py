"""
Arena integration hooks for event emission.

Provides hooks that connect the Arena debate engine to the event streaming
system, enabling real-time WebSocket broadcasts of debate events.
"""

import contextvars
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Generator, Optional, cast

from aragora.server.error_utils import safe_error_message as _safe_error_message
from aragora.server.stream.emitter import SyncEventEmitter
from aragora.server.stream.events import StreamEvent, StreamEventType

logger = logging.getLogger(__name__)

# Context variable to track current task_id for streaming events
# This allows concurrent generate() calls from the same agent to be distinguished
_current_task_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_task_id", default=""
)


@contextmanager
def streaming_task_context(task_id: str) -> Generator[None, None, None]:
    """Context manager to set the current task_id for streaming events.

    Use this when calling agent methods that may stream, to ensure their
    TOKEN_* events include the task_id for proper grouping.

    Example:
        with streaming_task_context(f"{agent.name}:critique:{target}"):
            result = await agent.critique(proposal, task, context)
    """
    token = _current_task_id.set(task_id)
    try:
        yield
    finally:
        _current_task_id.reset(token)


def get_current_task_id() -> str:
    """Get the current task_id for streaming events."""
    return _current_task_id.get()


def wrap_agent_for_streaming(agent: Any, emitter: SyncEventEmitter, debate_id: str) -> Any:
    """Wrap an agent to emit token streaming events.

    If the agent has a generate_stream() method, we override its generate()
    to call generate_stream() and emit TOKEN_* events.

    Args:
        agent: The agent to wrap
        emitter: The SyncEventEmitter to emit events to
        debate_id: The debate ID to include in events

    Returns:
        The agent (possibly modified with streaming support)
    """
    # Check if agent supports streaming
    if not hasattr(agent, "generate_stream"):
        return agent

    # Store original generate method
    original_generate = agent.generate

    async def streaming_generate(prompt: str, context: Optional[Any] = None) -> str:
        """Streaming wrapper that emits TOKEN_* events."""
        # Get current task_id from context variable (set by streaming_task_context)
        task_id = get_current_task_id()

        # Fallback: generate unique ID if context not set (prevents text interleaving)
        if not task_id:
            # Use UUID for truly unique task_id (prevents collision in concurrent streams)
            task_id = f"{debate_id}:{agent.name}:{uuid.uuid4().hex[:8]}"
            logger.warning(
                f"Missing task_id for {agent.name}, using fallback: {task_id}. "
                "Consider wrapping the generate() call with streaming_task_context()."
            )

        # Emit start event
        emitter.emit(
            StreamEvent(
                type=StreamEventType.TOKEN_START,
                data={
                    "debate_id": debate_id,
                    "agent": agent.name,
                    "timestamp": datetime.now().isoformat(),
                },
                agent=agent.name,
                task_id=task_id,
            )
        )

        full_response = ""
        try:
            # Stream tokens from the agent
            async for token in agent.generate_stream(prompt, context):
                full_response += token
                # Emit token delta event
                emitter.emit(
                    StreamEvent(
                        type=StreamEventType.TOKEN_DELTA,
                        data={
                            "debate_id": debate_id,
                            "agent": agent.name,
                            "token": token,
                        },
                        agent=agent.name,
                        task_id=task_id,
                    )
                )

            # Emit end event
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.TOKEN_END,
                    data={
                        "debate_id": debate_id,
                        "agent": agent.name,
                        "full_response": full_response,
                    },
                    agent=agent.name,
                    task_id=task_id,
                )
            )

            return full_response

        except Exception as e:
            # Emit error as end event
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.TOKEN_END,
                    data={
                        "debate_id": debate_id,
                        "agent": agent.name,
                        "error": _safe_error_message(e, f"token streaming for {agent.name}"),
                        "full_response": full_response,
                    },
                    agent=agent.name,
                    task_id=task_id,
                )
            )
            # Fall back to non-streaming
            if full_response:
                return full_response
            return cast(str, await original_generate(prompt, context))

    # Replace the generate method
    agent.generate = streaming_generate
    return agent


def create_arena_hooks(emitter: SyncEventEmitter, loop_id: str = "") -> dict[str, Callable]:
    """
    Create hook functions for Arena event emission.

    These hooks are called synchronously by Arena at key points during debate.
    They emit events to the emitter queue for async WebSocket broadcast.

    Args:
        emitter: The SyncEventEmitter to emit events to
        loop_id: Debate ID to attach to all events (required for correct routing)

    Returns:
        dict of hook name -> callback function
    """

    def on_debate_start(task: str, agents: list[str]) -> None:
        emitter.emit(
            StreamEvent(
                type=StreamEventType.DEBATE_START,
                data={"task": task, "agents": agents},
                loop_id=loop_id,
            )
        )

    def on_agent_preview(agents: list[dict]) -> None:
        """Emit agent preview with roles, stances, and brief descriptions.

        Called early in debate initialization to show agent info while
        proposals are being generated.

        Args:
            agents: List of dicts with name, role, stance, description, strengths
        """
        emitter.emit(
            StreamEvent(
                type=StreamEventType.AGENT_PREVIEW,
                data={"agents": agents, "topology": "collaborative"},
                loop_id=loop_id,
            )
        )

    def on_context_preview(
        trending_topics: list[dict],
        research_status: str = "gathering context...",
        evidence_sources: list[str] | None = None,
    ) -> None:
        """Emit context preview with trending topics and research status.

        Called when context gathering begins to show relevant background info.
        """
        emitter.emit(
            StreamEvent(
                type=StreamEventType.CONTEXT_PREVIEW,
                data={
                    "trending_topics": trending_topics,
                    "research_status": research_status,
                    "evidence_sources": evidence_sources or [],
                },
                loop_id=loop_id,
            )
        )

    def on_round_start(round_num: int) -> None:
        emitter.emit(
            StreamEvent(
                type=StreamEventType.ROUND_START,
                data={"round": round_num},
                round=round_num,
                loop_id=loop_id,
            )
        )

    def on_message(agent: str, content: str, role: str, round_num: int) -> None:
        emitter.emit(
            StreamEvent(
                type=StreamEventType.AGENT_MESSAGE,
                data={"content": content, "role": role},
                round=round_num,
                agent=agent,
                loop_id=loop_id,
            )
        )

    def on_critique(
        agent: str,
        target: str,
        issues: list[str],
        severity: float,
        round_num: int,
        full_content: str | None = None,
    ) -> None:
        emitter.emit(
            StreamEvent(
                type=StreamEventType.CRITIQUE,
                data={
                    "target": target,
                    "issues": issues,  # Full issue list
                    "severity": severity,
                    "content": full_content or "\n".join(f"• {issue}" for issue in issues),
                },
                round=round_num,
                agent=agent,
                loop_id=loop_id,
            )
        )

    def on_vote(agent: str, vote: str, confidence: float) -> None:
        emitter.emit(
            StreamEvent(
                type=StreamEventType.VOTE,
                data={"vote": vote, "confidence": confidence},
                agent=agent,
                loop_id=loop_id,
            )
        )

    def on_consensus(reached: bool, confidence: float, answer: str, synthesis: str = "") -> None:
        emitter.emit(
            StreamEvent(
                type=StreamEventType.CONSENSUS,
                data={
                    "reached": reached,
                    "confidence": confidence,
                    "answer": answer,  # Full answer - no truncation
                    "synthesis": synthesis,  # Fallback synthesis in case SYNTHESIS event is missed
                },
                loop_id=loop_id,
            )
        )

    def on_synthesis(content: str, confidence: float = 0.0) -> None:
        """Emit explicit synthesis event for guaranteed delivery."""
        emitter.emit(
            StreamEvent(
                type=StreamEventType.SYNTHESIS,
                data={
                    "content": content,
                    "confidence": confidence,
                    "agent": "synthesis-agent",
                },
                agent="synthesis-agent",
                loop_id=loop_id,
            )
        )

    def on_debate_end(duration: float, rounds: int) -> None:
        emitter.emit(
            StreamEvent(
                type=StreamEventType.DEBATE_END,
                data={"duration": duration, "rounds": rounds},
                loop_id=loop_id,
            )
        )

    def on_agent_error(
        agent: str,
        error_type: str,
        message: str,
        recoverable: bool = True,
        phase: str = "",
    ) -> None:
        """Emit agent error event when an agent fails but debate continues.

        This helps frontends understand why an agent produced placeholder output.
        """
        emitter.emit(
            StreamEvent(
                type=StreamEventType.AGENT_ERROR,
                data={
                    "error_type": error_type,  # "timeout", "connection", "internal"
                    "message": message,
                    "recoverable": recoverable,
                    "phase": phase,
                },
                agent=agent,
                loop_id=loop_id,
            )
        )

    def on_phase_progress(
        phase: str,
        completed: int,
        total: int,
        current_agent: str = "",
    ) -> None:
        """Emit progress within a phase (e.g., 3/8 agents have generated proposals).

        This helps frontends show progress and detect stalls.
        """
        emitter.emit(
            StreamEvent(
                type=StreamEventType.PHASE_PROGRESS,
                data={
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "current_agent": current_agent,
                    "progress_pct": (completed / total * 100) if total > 0 else 0,
                },
                loop_id=loop_id,
            )
        )

    def on_heartbeat(phase: str = "", status: str = "alive") -> None:
        """Emit periodic heartbeat to indicate debate is still running.

        Should be called every ~30 seconds during long-running phases.
        """
        emitter.emit(
            StreamEvent(
                type=StreamEventType.HEARTBEAT,
                data={
                    "phase": phase,
                    "status": status,
                },
                loop_id=loop_id,
            )
        )

    return {
        "on_debate_start": on_debate_start,
        "on_agent_preview": on_agent_preview,
        "on_context_preview": on_context_preview,
        "on_round_start": on_round_start,
        "on_message": on_message,
        "on_critique": on_critique,
        "on_vote": on_vote,
        "on_consensus": on_consensus,
        "on_synthesis": on_synthesis,
        "on_debate_end": on_debate_end,
        "on_agent_error": on_agent_error,
        "on_phase_progress": on_phase_progress,
        "on_heartbeat": on_heartbeat,
    }


__all__ = [
    "create_arena_hooks",
    "wrap_agent_for_streaming",
    "streaming_task_context",
    "get_current_task_id",
]
