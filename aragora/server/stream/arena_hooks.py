"""
Arena integration hooks for event emission.

Provides hooks that connect the Arena debate engine to the event streaming
system, enabling real-time WebSocket broadcasts of debate events.
"""

import contextvars
from contextlib import contextmanager
from datetime import datetime
from typing import Callable

from aragora.server.error_utils import safe_error_message as _safe_error_message
from aragora.server.stream.emitter import SyncEventEmitter
from aragora.server.stream.events import StreamEvent, StreamEventType

# Context variable to track current task_id for streaming events
# This allows concurrent generate() calls from the same agent to be distinguished
_current_task_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_task_id", default=""
)


@contextmanager
def streaming_task_context(task_id: str):
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


def wrap_agent_for_streaming(agent, emitter: SyncEventEmitter, debate_id: str):
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

    async def streaming_generate(prompt: str, context=None):
        """Streaming wrapper that emits TOKEN_* events."""
        # Get current task_id from context variable (set by streaming_task_context)
        task_id = get_current_task_id()

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
            return await original_generate(prompt, context)

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

    return {
        "on_debate_start": on_debate_start,
        "on_round_start": on_round_start,
        "on_message": on_message,
        "on_critique": on_critique,
        "on_vote": on_vote,
        "on_consensus": on_consensus,
        "on_synthesis": on_synthesis,
        "on_debate_end": on_debate_end,
    }


__all__ = [
    "create_arena_hooks",
    "wrap_agent_for_streaming",
    "streaming_task_context",
    "get_current_task_id",
]
