"""
Arena↔Control Plane Bridge.

Bridges the Arena debate engine with Control Plane infrastructure for:
- Unified debate orchestration with SLA tracking
- Real-time event streaming to ControlPlaneStreamServer
- Agent performance extraction for ELO updates
- Shared state synchronization

Usage:
    bridge = ArenaControlPlaneBridge(
        stream_server=stream_server,
        shared_state=shared_state,
    )

    outcome = await bridge.execute_via_arena(
        task=deliberation_task,
        agents=selected_agents,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from aragora.control_plane.deliberation_events import DeliberationEventType
from aragora.control_plane.deliberation import (
    AgentPerformance,
    DeliberationOutcome,
    DeliberationTask,
    SLAComplianceLevel,
)

# Audit logging (optional)
try:
    from aragora.control_plane.audit import (
        log_deliberation_started,
        log_deliberation_completed,
        log_deliberation_sla_event,
    )

    HAS_AUDIT = True
except ImportError:
    HAS_AUDIT = False
    log_deliberation_started = None  # type: ignore
    log_deliberation_completed = None  # type: ignore
    log_deliberation_sla_event = None  # type: ignore

# Prometheus metrics (optional)
try:
    from aragora.server.prometheus_control_plane import (
        record_deliberation_complete,
        record_deliberation_sla,
        record_agent_utilization,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    record_deliberation_complete = None  # type: ignore
    record_deliberation_sla = None  # type: ignore
    record_agent_utilization = None  # type: ignore

if TYPE_CHECKING:
    from aragora.control_plane.shared_state import SharedControlPlaneState
    from aragora.core import Agent, DebateResult
    from aragora.server.stream.control_plane_stream import ControlPlaneStreamServer

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Intermediate metrics collected during debate for a single agent."""

    agent_id: str
    response_count: int = 0
    vote_count: int = 0
    critique_count: int = 0
    total_confidence: float = 0.0
    position_history: List[str] = field(default_factory=list)
    contributed_to_final: bool = False


class ArenaEventAdapter:
    """
    Adapts Arena event_hooks to ControlPlaneStreamServer broadcasts.

    Translates Arena events (on_round_start, on_agent_message, etc.) into
    control plane events for real-time monitoring.
    """

    def __init__(
        self,
        task_id: str,
        stream_server: Optional["ControlPlaneStreamServer"] = None,
        shared_state: Optional["SharedControlPlaneState"] = None,
        sla_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the event adapter.

        Args:
            task_id: Unique task identifier for events
            stream_server: Optional ControlPlaneStreamServer for broadcasting
            shared_state: Optional SharedControlPlaneState for persistence
            sla_callback: Optional callback for SLA notifications
        """
        self.task_id = task_id
        self.stream_server = stream_server
        self.shared_state = shared_state
        self.sla_callback = sla_callback

        # Track agent metrics during debate
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._current_round = 0
        self._total_rounds = 0
        self._start_time = time.time()

    async def _emit(self, event_type: DeliberationEventType, data: Dict[str, Any]) -> None:
        """Emit an event to the stream server."""
        if self.stream_server:
            try:
                from aragora.server.stream.control_plane_stream import (
                    ControlPlaneEvent,
                    ControlPlaneEventType,
                )

                # Map deliberation events to control plane events
                # We use a custom approach since these are deliberation-specific
                event = ControlPlaneEvent(
                    event_type=ControlPlaneEventType.METRICS_UPDATE,  # Reuse for deliberation
                    data={
                        "deliberation_event": event_type.value,
                        "task_id": self.task_id,
                        **data,
                    },
                )
                await self.stream_server.broadcast(event)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

        # Also update shared state if available
        if self.shared_state:
            try:
                await self.shared_state.update_task_progress(  # type: ignore[attr-defined]
                    self.task_id,
                    {
                        "event_type": event_type.value,
                        "timestamp": time.time(),
                        **data,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to update shared state: {e}")

    def _ensure_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Get or create agent metrics."""
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        return self._agent_metrics[agent_id]

    # =========================================================================
    # Event Hook Methods (called by Arena)
    # =========================================================================

    async def on_debate_start(self, task: str, agents: List[str], rounds: int) -> None:
        """Called when debate starts."""
        self._total_rounds = rounds
        self._start_time = time.time()

        for agent_id in agents:
            self._ensure_agent_metrics(agent_id)

        await self._emit(
            DeliberationEventType.DELIBERATION_STARTED,
            {
                "task_preview": task[:200] if task else "",
                "agents": agents,
                "total_rounds": rounds,
            },
        )

    async def on_round_start(self, round_num: int, total_rounds: int) -> None:
        """Called at the start of each debate round."""
        self._current_round = round_num
        self._total_rounds = total_rounds

        await self._emit(
            DeliberationEventType.ROUND_START,
            {
                "round": round_num,
                "total_rounds": total_rounds,
                "progress_pct": (round_num / total_rounds * 100) if total_rounds > 0 else 0,
            },
        )

    async def on_round_end(self, round_num: int, total_rounds: int) -> None:
        """Called at the end of each debate round."""
        await self._emit(
            DeliberationEventType.ROUND_END,
            {
                "round": round_num,
                "total_rounds": total_rounds,
                "elapsed_seconds": time.time() - self._start_time,
            },
        )

    async def on_agent_message(
        self,
        agent: str,
        content: str,
        role: str,
        round_num: int = 0,
    ) -> None:
        """Called when an agent produces a message."""
        metrics = self._ensure_agent_metrics(agent)
        metrics.response_count += 1

        # Track position changes
        content_preview = content[:100] if content else ""
        if metrics.position_history:
            if content_preview != metrics.position_history[-1]:
                metrics.position_history.append(content_preview)
        else:
            metrics.position_history.append(content_preview)

        await self._emit(
            DeliberationEventType.AGENT_MESSAGE,
            {
                "agent": agent,
                "role": role,
                "round": round_num or self._current_round,
                "content_preview": content_preview,
                "message_length": len(content) if content else 0,
            },
        )

    async def on_proposal(self, agent: str, proposal: str, round_num: int = 0) -> None:
        """Called when an agent submits a proposal."""
        metrics = self._ensure_agent_metrics(agent)
        metrics.response_count += 1

        await self._emit(
            DeliberationEventType.AGENT_PROPOSAL,
            {
                "agent": agent,
                "round": round_num or self._current_round,
                "proposal_preview": proposal[:200] if proposal else "",
            },
        )

    async def on_critique(
        self,
        critic: str,
        target: str,
        issues: List[str],
        severity: float,
    ) -> None:
        """Called when an agent critiques another."""
        metrics = self._ensure_agent_metrics(critic)
        metrics.critique_count += 1

        await self._emit(
            DeliberationEventType.AGENT_CRITIQUE,
            {
                "critic": critic,
                "target": target,
                "issue_count": len(issues),
                "severity": severity,
                "issues_preview": [i[:100] for i in issues[:3]],
            },
        )

    async def on_vote(
        self, agent: str, choice: str, confidence: float, reasoning: str = ""
    ) -> None:
        """Called when an agent casts a vote."""
        metrics = self._ensure_agent_metrics(agent)
        metrics.vote_count += 1
        metrics.total_confidence += confidence

        await self._emit(
            DeliberationEventType.VOTE,
            {
                "agent": agent,
                "choice": choice,
                "confidence": confidence,
                "reasoning_preview": reasoning[:100] if reasoning else "",
            },
        )

    async def on_consensus_check(
        self,
        reached: bool,
        confidence: float,
        votes: Dict[str, str],
    ) -> None:
        """Called when consensus is checked."""
        event_type = (
            DeliberationEventType.CONSENSUS_REACHED
            if reached
            else DeliberationEventType.CONSENSUS_CHECK
        )

        await self._emit(
            event_type,
            {
                "reached": reached,
                "confidence": confidence,
                "vote_distribution": self._summarize_votes(votes),
                "elapsed_seconds": time.time() - self._start_time,
            },
        )

    async def on_convergence(self, similarity: float, status: str) -> None:
        """Called when convergence is detected."""
        await self._emit(
            DeliberationEventType.CONVERGENCE_UPDATE,
            {
                "similarity": similarity,
                "status": status,
                "round": self._current_round,
            },
        )

    async def on_sla_warning(self, elapsed_seconds: float, timeout_seconds: float) -> None:
        """Called when SLA warning threshold is reached."""
        await self._emit(
            DeliberationEventType.SLA_WARNING,
            {
                "elapsed_seconds": elapsed_seconds,
                "timeout_seconds": timeout_seconds,
                "remaining_seconds": timeout_seconds - elapsed_seconds,
                "pct_used": (elapsed_seconds / timeout_seconds * 100)
                if timeout_seconds > 0
                else 100,
            },
        )

        if self.sla_callback:
            self.sla_callback(
                "sla_warning",
                {
                    "task_id": self.task_id,
                    "elapsed_seconds": elapsed_seconds,
                    "timeout_seconds": timeout_seconds,
                },
            )

    async def on_sla_critical(self, elapsed_seconds: float, timeout_seconds: float) -> None:
        """Called when SLA critical threshold is reached."""
        await self._emit(
            DeliberationEventType.SLA_CRITICAL,
            {
                "elapsed_seconds": elapsed_seconds,
                "timeout_seconds": timeout_seconds,
            },
        )

        if self.sla_callback:
            self.sla_callback(
                "sla_critical",
                {
                    "task_id": self.task_id,
                    "elapsed_seconds": elapsed_seconds,
                    "timeout_seconds": timeout_seconds,
                },
            )

    async def on_agent_error(self, agent: str, error: str, recovered: bool) -> None:
        """Called when an agent encounters an error."""
        await self._emit(
            DeliberationEventType.AGENT_ERROR,
            {
                "agent": agent,
                "error": error[:200],
                "recovered": recovered,
            },
        )

    async def on_debate_complete(
        self,
        result: "DebateResult",
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Called when debate completes."""
        event_type = (
            DeliberationEventType.DELIBERATION_COMPLETED
            if success
            else DeliberationEventType.DELIBERATION_FAILED
        )

        await self._emit(
            event_type,
            {
                "success": success,
                "consensus_reached": getattr(result, "consensus_reached", False),
                "confidence": getattr(result, "confidence", 0.0),
                "rounds_completed": getattr(result, "rounds_completed", 0),
                "duration_seconds": time.time() - self._start_time,
                "error": error,
                "winner": getattr(result, "winner", None),
            },
        )

    def _summarize_votes(self, votes: Dict[str, str]) -> Dict[str, int]:
        """Summarize vote distribution."""
        distribution: Dict[str, int] = {}
        for choice in votes.values():
            distribution[choice] = distribution.get(choice, 0) + 1
        return distribution

    def get_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get collected agent metrics."""
        return self._agent_metrics


class ArenaControlPlaneBridge:
    """
    Bridges Arena debate engine with Control Plane infrastructure.

    Provides:
    - Unified debate execution with SLA tracking
    - Real-time event streaming
    - Agent performance extraction for ELO updates
    - Shared state synchronization
    """

    def __init__(
        self,
        stream_server: Optional["ControlPlaneStreamServer"] = None,
        shared_state: Optional["SharedControlPlaneState"] = None,
        elo_callback: Optional[Callable[[Dict[str, AgentPerformance]], None]] = None,
        sla_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the bridge.

        Args:
            stream_server: ControlPlaneStreamServer for event broadcasting
            shared_state: SharedControlPlaneState for persistence
            elo_callback: Callback to update agent ELO scores
            sla_callback: Callback for SLA notifications
        """
        self.stream_server = stream_server
        self.shared_state = shared_state
        self.elo_callback = elo_callback
        self.sla_callback = sla_callback

    async def execute_via_arena(
        self,
        task: DeliberationTask,
        agents: List["Agent"],
        workspace_id: Optional[str] = None,
    ) -> DeliberationOutcome:
        """
        Execute a deliberation using the Arena with full event streaming.

        Args:
            task: The deliberation task to execute
            agents: List of Agent instances to participate
            workspace_id: Optional workspace for knowledge mound scoping

        Returns:
            DeliberationOutcome with results and agent performances
        """
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        task.metrics.started_at = time.time()

        # Create event adapter for this task
        adapter = ArenaEventAdapter(
            task_id=task.task_id,
            stream_server=self.stream_server,
            shared_state=self.shared_state,
            sla_callback=self.sla_callback,
        )

        # Build event_hooks dict that wraps adapter methods
        event_hooks = self._create_event_hooks(adapter, task)

        # Create environment and protocol
        environment = Environment(
            task=task.question,
            context=task.context or "",
        )

        protocol = DebateProtocol(
            rounds=task.sla.max_rounds,
            consensus="majority",
            min_confidence=0.6,  # type: ignore[call-arg]
        )

        try:
            # Create and run Arena
            arena = Arena(
                environment=environment,
                agents=agents,
                protocol=protocol,
                event_hooks=event_hooks,
                org_id=workspace_id or "",
            )

            # Start SLA monitoring
            sla_task = asyncio.create_task(self._monitor_sla(task, adapter))

            # Log deliberation start to audit trail
            if HAS_AUDIT and log_deliberation_started:
                try:
                    await log_deliberation_started(
                        task_id=task.task_id,
                        question=task.question,
                        agents=[a.name if hasattr(a, "name") else str(a) for a in agents],
                        sla_timeout_seconds=task.sla.timeout_seconds,
                        workspace_id=workspace_id,
                    )
                except Exception as e:
                    logger.debug(f"Failed to log deliberation start: {e}")

            try:
                # Run the debate
                result = await arena.run()

                # Extract metrics
                task.metrics.completed_at = time.time()
                task.metrics.rounds_completed = result.rounds_completed
                task.metrics.consensus_confidence = result.confidence

                # Determine final status
                if result.consensus_reached:
                    task.status = task.status.__class__.CONSENSUS_REACHED
                else:
                    task.status = task.status.__class__.NO_CONSENSUS

                # Extract agent performances
                agent_performances = self._extract_agent_performance(
                    result, adapter.get_agent_metrics()
                )

                # Check SLA compliance
                duration = task.metrics.duration_seconds or 0
                task.metrics.sla_compliance = task.sla.get_compliance_level(duration)

                # Record SLA compliance metric
                if HAS_PROMETHEUS and record_deliberation_sla:
                    try:
                        sla_level = (
                            task.metrics.sla_compliance.value
                            if task.metrics.sla_compliance
                            else "compliant"
                        )
                        record_deliberation_sla(sla_level)
                    except Exception as e:
                        logger.debug(f"Failed to record SLA compliance metric: {e}")

                # Build outcome
                outcome = DeliberationOutcome(
                    task_id=task.task_id,
                    request_id=task.request_id,
                    success=True,
                    consensus_reached=result.consensus_reached,
                    consensus_confidence=result.confidence,
                    winning_position=result.final_answer,
                    agent_performances=agent_performances,
                    duration_seconds=duration,
                    sla_compliant=task.metrics.sla_compliance != SLAComplianceLevel.VIOLATED,
                )

                # Log deliberation completion to audit trail
                if HAS_AUDIT and log_deliberation_completed:
                    try:
                        await log_deliberation_completed(
                            task_id=task.task_id,
                            success=True,
                            consensus_reached=result.consensus_reached,
                            confidence=result.confidence,
                            duration_seconds=duration,
                            sla_compliant=outcome.sla_compliant,
                            workspace_id=workspace_id,
                            winner=result.winner,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log deliberation completion: {e}")

                # Record Prometheus metrics
                if HAS_PROMETHEUS and record_deliberation_complete:
                    try:
                        record_deliberation_complete(
                            duration_seconds=duration,
                            status="completed",
                            consensus_reached=result.consensus_reached,
                            confidence=result.confidence,
                            round_count=result.rounds_completed,
                            agent_count=len(agents),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to record deliberation metrics: {e}")

                # Record agent utilization metrics
                if HAS_PROMETHEUS and record_agent_utilization and agent_performances:
                    try:
                        for agent_id, perf in agent_performances.items():
                            # Utilization based on response count relative to rounds
                            utilization = min(
                                1.0,
                                perf.response_count / max(1, result.rounds_completed),
                            )
                            record_agent_utilization(agent_id, utilization)
                    except Exception as e:
                        logger.debug(f"Failed to record agent utilization: {e}")

                # Fire ELO callback if provided
                if self.elo_callback and agent_performances:
                    try:
                        self.elo_callback(agent_performances)
                    except Exception as e:
                        logger.error(f"ELO callback failed: {e}")

                # Emit completion event
                await adapter.on_debate_complete(result, success=True)

                return outcome

            finally:
                sla_task.cancel()
                try:
                    await sla_task
                except asyncio.CancelledError:
                    pass

        except asyncio.TimeoutError:
            task.metrics.completed_at = time.time()
            task.metrics.sla_compliance = SLAComplianceLevel.VIOLATED
            duration = task.metrics.duration_seconds or 0

            # Record timeout metrics
            if HAS_PROMETHEUS and record_deliberation_complete:
                try:
                    record_deliberation_complete(
                        duration_seconds=duration,
                        status="timeout",
                        consensus_reached=False,
                        confidence=0.0,
                        round_count=0,
                        agent_count=len(agents),
                    )
                except Exception as e:
                    logger.debug(f"Failed to record timeout metrics: {e}")

            if HAS_PROMETHEUS and record_deliberation_sla:
                try:
                    record_deliberation_sla("violated")
                except Exception as e:
                    logger.debug(f"Failed to record SLA violation: {e}")

            await adapter.on_debate_complete(
                None,  # type: ignore
                success=False,
                error=f"Timeout after {task.sla.timeout_seconds}s",
            )

            return DeliberationOutcome(
                task_id=task.task_id,
                request_id=task.request_id,
                success=False,
                consensus_reached=False,
                duration_seconds=duration,
                sla_compliant=False,
            )

        except Exception as e:
            task.metrics.completed_at = time.time()
            error_msg = str(e)
            duration = task.metrics.duration_seconds or 0

            # Record failure metrics
            if HAS_PROMETHEUS and record_deliberation_complete:
                try:
                    record_deliberation_complete(
                        duration_seconds=duration,
                        status="failed",
                        consensus_reached=False,
                        confidence=0.0,
                        round_count=0,
                        agent_count=len(agents),
                    )
                except Exception as exc:
                    logger.debug(f"Failed to record failure metrics: {exc}")

            await adapter.on_debate_complete(
                None,  # type: ignore
                success=False,
                error=error_msg,
            )

            return DeliberationOutcome(
                task_id=task.task_id,
                request_id=task.request_id,
                success=False,
                consensus_reached=False,
                duration_seconds=duration,
                sla_compliant=task.metrics.sla_compliance != SLAComplianceLevel.VIOLATED,
            )

    def _create_event_hooks(
        self,
        adapter: ArenaEventAdapter,
        task: DeliberationTask,
    ) -> Dict[str, Callable]:
        """Create event_hooks dict for Arena initialization."""

        def sync_wrapper(coro_func):
            """Wrap async function for sync event hooks."""

            def wrapper(*args, **kwargs):
                try:
                    asyncio.get_running_loop()  # Check if loop exists
                    asyncio.ensure_future(coro_func(*args, **kwargs))
                except RuntimeError:
                    # No running loop, run synchronously
                    asyncio.run(coro_func(*args, **kwargs))

            return wrapper

        return {
            "on_debate_start": sync_wrapper(
                lambda task_str, agents, rounds: adapter.on_debate_start(task_str, agents, rounds)
            ),
            "on_round_start": sync_wrapper(
                lambda round_num, total: adapter.on_round_start(round_num, total)
            ),
            "on_round_end": sync_wrapper(
                lambda round_num, total: adapter.on_round_end(round_num, total)
            ),
            "on_agent_message": sync_wrapper(
                lambda agent, content, role, **kw: adapter.on_agent_message(
                    agent, content, role, kw.get("round", 0)
                )
            ),
            "on_proposal": sync_wrapper(
                lambda agent, proposal, **kw: adapter.on_proposal(
                    agent, proposal, kw.get("round", 0)
                )
            ),
            "on_critique": sync_wrapper(
                lambda critic, target, issues, severity: adapter.on_critique(
                    critic, target, issues, severity
                )
            ),
            "on_vote": sync_wrapper(
                lambda agent, choice, confidence, reasoning="": adapter.on_vote(
                    agent, choice, confidence, reasoning
                )
            ),
            "on_consensus": sync_wrapper(
                lambda reached, confidence, votes: adapter.on_consensus_check(
                    reached, confidence, votes
                )
            ),
            "on_convergence": sync_wrapper(
                lambda similarity, status: adapter.on_convergence(similarity, status)
            ),
            "on_agent_error": sync_wrapper(
                lambda agent, error, recovered=False: adapter.on_agent_error(
                    agent, error, recovered
                )
            ),
        }

    async def _monitor_sla(
        self,
        task: DeliberationTask,
        adapter: ArenaEventAdapter,
    ) -> None:
        """Monitor SLA compliance during execution."""
        start_time = task.metrics.started_at or time.time()
        warning_sent = False
        critical_sent = False

        while True:
            await asyncio.sleep(1.0)  # Check every second

            elapsed = time.time() - start_time
            compliance = task.sla.get_compliance_level(elapsed)
            task.metrics.sla_compliance = compliance

            if compliance == SLAComplianceLevel.WARNING and not warning_sent:
                await adapter.on_sla_warning(elapsed, task.sla.timeout_seconds)
                # Log to audit trail
                if HAS_AUDIT and log_deliberation_sla_event:
                    try:
                        await log_deliberation_sla_event(
                            task_id=task.task_id,
                            level="warning",
                            elapsed_seconds=elapsed,
                            timeout_seconds=task.sla.timeout_seconds,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log SLA warning: {e}")
                # Record Prometheus metric
                if HAS_PROMETHEUS and record_deliberation_sla:
                    try:
                        record_deliberation_sla("warning")
                    except Exception as e:
                        logger.debug(f"Failed to record SLA warning metric: {e}")
                warning_sent = True

            elif compliance == SLAComplianceLevel.CRITICAL and not critical_sent:
                await adapter.on_sla_critical(elapsed, task.sla.timeout_seconds)
                # Log to audit trail
                if HAS_AUDIT and log_deliberation_sla_event:
                    try:
                        await log_deliberation_sla_event(
                            task_id=task.task_id,
                            level="critical",
                            elapsed_seconds=elapsed,
                            timeout_seconds=task.sla.timeout_seconds,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log SLA critical: {e}")
                # Record Prometheus metric
                if HAS_PROMETHEUS and record_deliberation_sla:
                    try:
                        record_deliberation_sla("critical")
                    except Exception as e:
                        logger.debug(f"Failed to record SLA critical metric: {e}")
                critical_sent = True

            elif compliance == SLAComplianceLevel.VIOLATED:
                await adapter._emit(
                    DeliberationEventType.SLA_VIOLATED,
                    {
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": task.sla.timeout_seconds,
                    },
                )
                # Log to audit trail
                if HAS_AUDIT and log_deliberation_sla_event:
                    try:
                        await log_deliberation_sla_event(
                            task_id=task.task_id,
                            level="violated",
                            elapsed_seconds=elapsed,
                            timeout_seconds=task.sla.timeout_seconds,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log SLA violation: {e}")
                # Record Prometheus metric
                if HAS_PROMETHEUS and record_deliberation_sla:
                    try:
                        record_deliberation_sla("violated")
                    except Exception as e:
                        logger.debug(f"Failed to record SLA violation metric: {e}")
                break

    def _extract_agent_performance(
        self,
        result: "DebateResult",
        adapter_metrics: Dict[str, AgentMetrics],
    ) -> Dict[str, AgentPerformance]:
        """Extract per-agent performance metrics from DebateResult."""
        performances: Dict[str, AgentPerformance] = {}

        # Get winner from result
        winner = getattr(result, "winner", None)
        final_answer = getattr(result, "final_answer", "")

        for agent_id, metrics in adapter_metrics.items():
            # Calculate average confidence
            avg_confidence = (
                metrics.total_confidence / metrics.vote_count if metrics.vote_count > 0 else 0.0
            )

            # Check if position changed during debate
            position_changed = len(metrics.position_history) > 1

            # Check if final position matches consensus
            final_position_correct: bool = agent_id == winner or bool(
                metrics.position_history
                and final_answer
                and metrics.position_history[-1][:50] in final_answer[:100]
            )

            performances[agent_id] = AgentPerformance(
                agent_id=agent_id,
                contributed_to_consensus=final_position_correct,
                response_count=metrics.response_count,
                average_confidence=avg_confidence,
                position_changed=position_changed,
                final_position_correct=final_position_correct,
            )

        return performances


# Singleton bridge instance
_bridge: Optional[ArenaControlPlaneBridge] = None


def get_arena_bridge() -> Optional[ArenaControlPlaneBridge]:
    """Get the global arena bridge instance."""
    return _bridge


def set_arena_bridge(bridge: ArenaControlPlaneBridge) -> None:
    """Set the global arena bridge instance."""
    global _bridge
    _bridge = bridge


def init_arena_bridge(
    stream_server: Optional["ControlPlaneStreamServer"] = None,
    shared_state: Optional["SharedControlPlaneState"] = None,
    elo_callback: Optional[Callable[[Dict[str, AgentPerformance]], None]] = None,
) -> ArenaControlPlaneBridge:
    """Initialize and set the global arena bridge."""
    bridge = ArenaControlPlaneBridge(
        stream_server=stream_server,
        shared_state=shared_state,
        elo_callback=elo_callback,
    )
    set_arena_bridge(bridge)
    return bridge


__all__ = [
    "ArenaEventAdapter",
    "ArenaControlPlaneBridge",
    "AgentMetrics",
    "get_arena_bridge",
    "set_arena_bridge",
    "init_arena_bridge",
]
