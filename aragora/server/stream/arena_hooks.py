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

from aragora.debate.hooks import HookManager
from aragora.server.errors import safe_error_message as _safe_error_message
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


def create_hook_manager_from_emitter(
    emitter: SyncEventEmitter,
    loop_id: str = "",
) -> "HookManager":
    """
    Create a HookManager that bridges to WebSocket events.

    This function creates a HookManager and registers all standard hook types
    to emit events via the SyncEventEmitter, enabling real-time WebSocket
    broadcasts of debate lifecycle events.

    Args:
        emitter: The SyncEventEmitter to emit events to
        loop_id: Debate ID to attach to all events

    Returns:
        Configured HookManager with WebSocket bridges
    """
    from aragora.debate.hooks import HookManager, HookPriority, HookType

    manager = HookManager()

    # Bridge debate lifecycle hooks
    def on_pre_debate(**kwargs: Any) -> None:
        """Bridge PRE_DEBATE to DEBATE_START event."""
        task = kwargs.get("task", "")
        agents = [a.name if hasattr(a, "name") else str(a) for a in kwargs.get("agents", [])]
        emitter.emit(
            StreamEvent(
                type=StreamEventType.DEBATE_START,
                data={"task": task, "agents": agents},
                loop_id=loop_id,
            )
        )

    def on_post_debate(**kwargs: Any) -> None:
        """Bridge POST_DEBATE to DEBATE_END event."""
        result = kwargs.get("result")
        duration = getattr(result, "duration", 0.0) if result else 0.0
        rounds = getattr(result, "rounds_completed", 0) if result else 0
        emitter.emit(
            StreamEvent(
                type=StreamEventType.DEBATE_END,
                data={"duration": duration, "rounds": rounds},
                loop_id=loop_id,
            )
        )

    def on_pre_round(**kwargs: Any) -> None:
        """Bridge PRE_ROUND to ROUND_START event."""
        round_num = kwargs.get("round_num", 0)
        emitter.emit(
            StreamEvent(
                type=StreamEventType.ROUND_START,
                data={"round": round_num},
                round=round_num,
                loop_id=loop_id,
            )
        )

    def on_post_round(**kwargs: Any) -> None:
        """Bridge POST_ROUND - no direct event, but useful for logging."""
        round_num = kwargs.get("round_num", 0)
        proposals = kwargs.get("proposals", {})
        logger.debug(f"Round {round_num} complete with {len(proposals)} proposals")

    def on_post_consensus(**kwargs: Any) -> None:
        """Bridge POST_CONSENSUS to CONSENSUS event."""
        result = kwargs.get("result")
        if result:
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.CONSENSUS,
                    data={
                        "reached": getattr(result, "reached", False),
                        "confidence": getattr(result, "confidence", 0.0),
                        "answer": getattr(result, "answer", ""),
                    },
                    loop_id=loop_id,
                )
            )

    def on_finding(**kwargs: Any) -> None:
        """Bridge ON_FINDING to audit finding event."""
        finding = kwargs.get("finding")
        if finding:
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.PHASE_PROGRESS,  # Use PHASE_PROGRESS for audit
                    data={
                        "phase": "audit",
                        "event": "finding",
                        "finding_id": getattr(finding, "id", ""),
                        "title": getattr(finding, "title", ""),
                        "severity": getattr(finding, "severity", "info"),
                    },
                    loop_id=loop_id,
                )
            )

    def on_progress(**kwargs: Any) -> None:
        """Bridge ON_PROGRESS to PHASE_PROGRESS event."""
        phase = kwargs.get("phase", "")
        completed = kwargs.get("completed", 0)
        total = kwargs.get("total", 0)
        emitter.emit(
            StreamEvent(
                type=StreamEventType.PHASE_PROGRESS,
                data={
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "progress_pct": (completed / total * 100) if total > 0 else 0,
                },
                loop_id=loop_id,
            )
        )

    def on_error(**kwargs: Any) -> None:
        """Bridge ON_ERROR to AGENT_ERROR event."""
        agent = kwargs.get("agent", "unknown")
        error = kwargs.get("error")
        emitter.emit(
            StreamEvent(
                type=StreamEventType.AGENT_ERROR,
                data={
                    "error_type": type(error).__name__ if error else "unknown",
                    "message": str(error) if error else "Unknown error",
                    "recoverable": kwargs.get("recoverable", True),
                    "phase": kwargs.get("phase", ""),
                },
                agent=agent if isinstance(agent, str) else getattr(agent, "name", "unknown"),
                loop_id=loop_id,
            )
        )

    def on_cancellation(**kwargs: Any) -> None:
        """Bridge ON_CANCELLATION event."""
        reason = kwargs.get("reason", "User requested")
        emitter.emit(
            StreamEvent(
                type=StreamEventType.DEBATE_END,
                data={
                    "cancelled": True,
                    "reason": reason,
                },
                loop_id=loop_id,
            )
        )

    # Register all bridges
    manager.register(
        HookType.PRE_DEBATE, on_pre_debate, priority=HookPriority.LOW, name="ws_pre_debate"
    )
    manager.register(
        HookType.POST_DEBATE, on_post_debate, priority=HookPriority.LOW, name="ws_post_debate"
    )
    manager.register(
        HookType.PRE_ROUND, on_pre_round, priority=HookPriority.LOW, name="ws_pre_round"
    )
    manager.register(
        HookType.POST_ROUND, on_post_round, priority=HookPriority.LOW, name="ws_post_round"
    )
    manager.register(
        HookType.POST_CONSENSUS,
        on_post_consensus,
        priority=HookPriority.LOW,
        name="ws_post_consensus",
    )
    manager.register(
        HookType.ON_FINDING, on_finding, priority=HookPriority.LOW, name="ws_on_finding"
    )
    manager.register(
        HookType.ON_PROGRESS, on_progress, priority=HookPriority.LOW, name="ws_on_progress"
    )
    manager.register(HookType.ON_ERROR, on_error, priority=HookPriority.LOW, name="ws_on_error")
    manager.register(
        HookType.ON_CANCELLATION,
        on_cancellation,
        priority=HookPriority.LOW,
        name="ws_on_cancellation",
    )

    logger.debug(f"Created HookManager with {len(manager.stats)} hook types bridged to WebSocket")
    return manager


__all__ = [
    "create_arena_hooks",
    "create_hook_manager_from_emitter",
    "wrap_agent_for_streaming",
    "streaming_task_context",
    "get_current_task_id",
]
