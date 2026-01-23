"""
End-to-End Test Harness for Aragora.

Provides a complete integration testing environment that spins up:
- ControlPlaneCoordinator
- TaskScheduler
- Mock agents with configurable behaviors
- Optional Redis/PostgreSQL connections
- Metrics and tracing support

Usage:
    async with e2e_environment() as harness:
        # Register mock agents
        await harness.start()

        # Submit tasks
        task_id = await harness.submit_task("debate", {"topic": "AI safety"})

        # Wait for completion
        result = await harness.wait_for_task(task_id)

        # Run a full debate
        debate_result = await harness.run_debate("Should we use microservices?")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock

from aragora.control_plane import (
    ControlPlaneCoordinator,
    TaskScheduler,
    AgentRegistry,
    HealthMonitor,
    Task,
    TaskPriority,
    TaskStatus,
    AgentStatus,
)
from aragora.control_plane.coordinator import ControlPlaneConfig
from aragora.core import Environment, Message, Critique, Vote, DebateResult


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class E2ETestConfig:
    """Configuration for E2E test environment."""

    # Agent configuration
    num_agents: int = 3
    agent_capabilities: List[str] = field(default_factory=lambda: ["debate", "code", "analysis"])
    agent_response_delay: float = 0.05  # Seconds to simulate work

    # Infrastructure
    use_redis: bool = False  # Use in-memory by default
    redis_url: str = "redis://localhost:6379"
    use_postgres: bool = False
    postgres_url: str = "postgresql://localhost:5432/aragora_test"

    # Observability
    enable_metrics: bool = False
    enable_tracing: bool = False

    # Timeouts
    timeout_seconds: float = 30.0
    task_timeout_seconds: float = 10.0
    heartbeat_interval: float = 2.0

    # Debate configuration
    default_debate_rounds: int = 3
    consensus_threshold: float = 0.7

    # Test behavior
    fail_rate: float = 0.0  # Rate of simulated failures (0-1)
    latency_variance: float = 0.1  # Variance in response times


@dataclass
class MockAgentConfig:
    """Configuration for a mock agent."""

    id: str
    capabilities: List[str] = field(default_factory=lambda: ["general"])
    region: str = "us-east-1"
    model: str = "mock-model"
    provider: str = "mock-provider"
    response_template: str = "Response from {agent_id}: {task_type}"
    fail_rate: float = 0.0
    response_delay: float = 0.05


# ============================================================================
# Mock Agent Implementation
# ============================================================================


@dataclass
class MockAgent:
    """
    Mock agent for testing.

    Simulates an AI agent with configurable:
    - Response generation
    - Critique generation
    - Voting behavior
    - Failure modes
    """

    id: str
    capabilities: List[str] = field(default_factory=lambda: ["general"])
    region: str = "us-east-1"
    model: str = "mock-model"
    provider: str = "mock-provider"
    response_template: str = "Response from {agent_id}"
    fail_rate: float = 0.0
    response_delay: float = 0.05

    # Statistics
    tasks_processed: int = 0
    tasks_failed: int = 0
    total_latency_ms: float = 0.0

    # For compatibility with Arena
    name: str = field(default="")
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    metrics: Any = None

    # Arena role/stance attributes
    role: Optional[str] = None  # "proposer", "critic", "synthesizer", "judge"
    stance: Optional[str] = None  # "affirmative", "negative", "neutral"

    # System prompt for Arena compatibility
    system_prompt: str = ""
    agreement_intensity: float = 0.5  # 0-1 scale for how agreeable the agent is

    def __post_init__(self):
        if not self.name:
            self.name = self.id
        # Initialize system_prompt with default if empty
        if not self.system_prompt:
            self.system_prompt = f"You are {self.name}, a helpful AI assistant."

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a task and return result."""
        import random

        start = time.monotonic()

        # Simulate work
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay + random.uniform(0, 0.05))

        # Simulate failures
        if self.fail_rate > 0 and random.random() < self.fail_rate:
            self.tasks_failed += 1
            raise RuntimeError(f"Simulated failure in agent {self.id}")

        self.tasks_processed += 1
        latency_ms = (time.monotonic() - start) * 1000
        self.total_latency_ms += latency_ms

        return {
            "status": "completed",
            "result": self.response_template.format(
                agent_id=self.id,
                task_type=task.task_type if hasattr(task, "task_type") else "unknown",
            ),
            "agent_id": self.id,
            "latency_ms": latency_ms,
        }

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response (for Arena compatibility)."""
        await asyncio.sleep(self.response_delay)
        return self.response_template.format(agent_id=self.id, task_type="generate")

    async def critique(
        self, target_content: str, target_agent: str = "unknown", **kwargs
    ) -> Critique:
        """Generate a critique (for Arena compatibility)."""
        await asyncio.sleep(self.response_delay)
        return Critique(
            agent=self.id,
            target_agent=target_agent,
            target_content=target_content[:100],
            issues=["Minor issue found"],
            suggestions=["Consider improving clarity"],
            severity=3.0,
            reasoning=f"Critique from {self.id}",
        )

    async def vote(self, proposals: List[str], **kwargs) -> Vote:
        """Vote on proposals (for Arena compatibility)."""
        await asyncio.sleep(self.response_delay)
        # Vote for first proposal by default
        choice = proposals[0] if proposals else "abstain"
        return Vote(
            agent=self.id,
            choice=choice,
            reasoning=f"Vote from {self.id}",
            confidence=0.8,
            continue_debate=True,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "id": self.id,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "avg_latency_ms": (
                self.total_latency_ms / self.tasks_processed if self.tasks_processed > 0 else 0
            ),
        }


def create_mock_agent(
    name: str,
    response: str = "Default response",
    capabilities: Optional[List[str]] = None,
) -> MockAgent:
    """Create a mock agent with common defaults.

    This is a convenience function for creating agents in tests.
    """
    return MockAgent(
        id=name,
        name=name,
        capabilities=capabilities or ["debate", "code", "analysis"],
        response_template=response,
    )


# ============================================================================
# E2E Test Harness
# ============================================================================


class E2ETestHarness:
    """
    Full integration test harness.

    Spins up:
    - ControlPlaneCoordinator
    - TaskScheduler
    - Mock agents
    - Optional: Redis, PostgreSQL connections

    Usage:
        harness = E2ETestHarness()
        await harness.start()

        # Register custom agents
        agent = await harness.create_agent("custom-agent", ["special-cap"])

        # Submit and process tasks
        task_id = await harness.submit_task("debate", {"topic": "..."})
        result = await harness.wait_for_task(task_id)

        await harness.stop()
    """

    def __init__(self, config: Optional[E2ETestConfig] = None):
        """Initialize the harness with configuration."""
        self.config = config or E2ETestConfig()
        self.coordinator: Optional[ControlPlaneCoordinator] = None
        self.scheduler: Optional[TaskScheduler] = None
        self.registry: Optional[AgentRegistry] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.agents: List[MockAgent] = []
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._events: List[Dict[str, Any]] = []

    async def start(self) -> None:
        """Start the test environment."""
        if self._running:
            return

        # Create control plane config
        cp_config = ControlPlaneConfig(
            redis_url=self.config.redis_url if self.config.use_redis else "memory://",
            heartbeat_timeout=self.config.heartbeat_interval * 3,
            task_timeout=self.config.task_timeout_seconds,
            max_task_retries=2,
            cleanup_interval=1.0,
            enable_km_integration=False,  # Disable KM for tests
            enable_policy_sync=False,  # Disable policy sync for tests
        )

        # Initialize scheduler (with in-memory fallback)
        self.scheduler = TaskScheduler(
            redis_url=cp_config.redis_url,
            key_prefix="aragora:test:tasks:",
            stream_prefix="aragora:test:stream:",
        )
        await self.scheduler.connect()

        # Initialize registry
        self.registry = AgentRegistry(
            redis_url=cp_config.redis_url,
            key_prefix="aragora:test:agents:",
            heartbeat_timeout=self.config.heartbeat_interval * 3,
        )
        await self.registry.connect()

        # Initialize health monitor
        self.health_monitor = HealthMonitor(
            registry=self.registry,
            probe_interval=self.config.heartbeat_interval,
            probe_timeout=self.config.heartbeat_interval,
        )

        # Initialize coordinator with components
        self.coordinator = ControlPlaneCoordinator(
            config=cp_config,
            registry=self.registry,
            scheduler=self.scheduler,
            health_monitor=self.health_monitor,
        )
        await self.coordinator.connect()

        # Create default mock agents
        for i in range(self.config.num_agents):
            agent = MockAgent(
                id=f"test-agent-{i}",
                capabilities=self.config.agent_capabilities.copy(),
                response_delay=self.config.agent_response_delay,
                fail_rate=self.config.fail_rate,
            )
            self.agents.append(agent)

            # Register with coordinator
            await self.coordinator.register_agent(
                agent_id=agent.id,
                capabilities=agent.capabilities,
                model=agent.model,
                provider=agent.provider,
                metadata={"region": agent.region},
            )

        # Start health monitor
        await self.health_monitor.start()

        self._running = True

    async def stop(self) -> None:
        """Stop the test environment and clean up resources."""
        if not self._running:
            return

        self._running = False

        # Cancel any running worker tasks
        for task in self._worker_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._worker_tasks.clear()

        # Unregister agents
        for agent in self.agents:
            try:
                await self.coordinator.unregister_agent(agent.id)
            except Exception:
                pass

        self.agents.clear()

        # Stop health monitor
        if self.health_monitor:
            await self.health_monitor.stop()

        # Shutdown coordinator
        if self.coordinator:
            await self.coordinator.shutdown()

        # Close scheduler and registry
        if self.scheduler:
            await self.scheduler.close()
        if self.registry:
            await self.registry.close()

    # =========================================================================
    # Agent Management
    # =========================================================================

    async def create_agent(
        self,
        agent_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        config: Optional[MockAgentConfig] = None,
    ) -> MockAgent:
        """Create and register a new mock agent."""
        if config:
            agent = MockAgent(
                id=config.id,
                capabilities=config.capabilities,
                region=config.region,
                model=config.model,
                provider=config.provider,
                response_template=config.response_template,
                fail_rate=config.fail_rate,
                response_delay=config.response_delay,
            )
        else:
            agent = MockAgent(
                id=agent_id or f"agent-{uuid.uuid4().hex[:8]}",
                capabilities=capabilities or ["general"],
            )

        self.agents.append(agent)

        await self.coordinator.register_agent(
            agent_id=agent.id,
            capabilities=agent.capabilities,
            model=agent.model,
            provider=agent.provider,
            metadata={"region": agent.region},
        )

        return agent

    async def remove_agent(self, agent_id: str) -> bool:
        """Remove and unregister an agent."""
        agent = next((a for a in self.agents if a.id == agent_id), None)
        if not agent:
            return False

        self.agents.remove(agent)
        return await self.coordinator.unregister_agent(agent_id)

    def get_agent(self, agent_id: str) -> Optional[MockAgent]:
        """Get a mock agent by ID."""
        return next((a for a in self.agents if a.id == agent_id), None)

    # =========================================================================
    # Task Operations
    # =========================================================================

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
    ) -> str:
        """Submit a task and return task ID."""
        task_id = await self.coordinator.submit_task(
            task_type=task_type,
            payload=payload,
            required_capabilities=required_capabilities,
            priority=priority,
            timeout_seconds=timeout or self.config.task_timeout_seconds,
        )

        self._events.append(
            {
                "type": "task_submitted",
                "task_id": task_id,
                "task_type": task_type,
                "timestamp": time.time(),
            }
        )

        return task_id

    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
        process_with_agent: bool = True,
    ) -> Optional[Task]:
        """Wait for task completion.

        Args:
            task_id: Task to wait for
            timeout: Maximum wait time
            process_with_agent: If True, automatically process with a mock agent

        Returns:
            Completed task or None if timeout
        """
        timeout = timeout or self.config.timeout_seconds

        if process_with_agent:
            # Start a worker to process the task
            async def process():
                task = await self.coordinator.get_task(task_id)
                if not task:
                    return

                # Find an agent with matching capabilities
                for agent in self.agents:
                    caps_match = (
                        not task.required_capabilities
                        or task.required_capabilities.issubset(set(agent.capabilities))
                    )
                    if caps_match:
                        try:
                            result = await agent.process_task(task)
                            await self.coordinator.complete_task(
                                task_id=task_id,
                                result=result,
                                agent_id=agent.id,
                                latency_ms=result.get("latency_ms", 0),
                            )
                            return
                        except Exception as e:
                            await self.coordinator.fail_task(
                                task_id=task_id,
                                error=str(e),
                                agent_id=agent.id,
                                requeue=True,
                            )
                            return

            # Process in background
            asyncio.create_task(process())

        return await self.coordinator.wait_for_result(task_id, timeout=timeout)

    async def claim_and_process_task(
        self,
        agent: MockAgent,
        capabilities: Optional[List[str]] = None,
    ) -> Optional[Task]:
        """Have an agent claim and process a task."""
        caps = capabilities or agent.capabilities

        task = await self.coordinator.claim_task(
            agent_id=agent.id,
            capabilities=caps,
            block_ms=100,
        )

        if not task:
            return None

        try:
            result = await agent.process_task(task)
            await self.coordinator.complete_task(
                task_id=task.id,
                result=result,
                agent_id=agent.id,
                latency_ms=result.get("latency_ms", 0),
            )
        except Exception as e:
            await self.coordinator.fail_task(
                task_id=task.id,
                error=str(e),
                agent_id=agent.id,
                requeue=True,
            )

        return task

    # =========================================================================
    # Debate Operations
    # =========================================================================

    async def run_debate(
        self,
        topic: str,
        rounds: Optional[int] = None,
        agents: Optional[List[MockAgent]] = None,
        protocol_options: Optional[Dict[str, Any]] = None,
    ) -> DebateResult:
        """Run a full debate through the system.

        Args:
            topic: Debate topic
            rounds: Number of rounds (defaults to config)
            agents: Agents to use (defaults to all registered mock agents)
            protocol_options: Additional protocol configuration

        Returns:
            DebateResult from the completed debate
        """
        from aragora.debate import Arena, DebateProtocol

        env = Environment(task=topic)

        # Build protocol options
        proto_opts = {
            "rounds": rounds or self.config.default_debate_rounds,
            "consensus": "majority",
            "enable_calibration": False,
            "enable_rhetorical_observer": False,
            "enable_trickster": False,
        }
        if protocol_options:
            proto_opts.update(protocol_options)

        protocol = DebateProtocol(**proto_opts)

        # Use provided agents or default mock agents
        debate_agents = agents or self.agents

        # Create and run arena
        arena = Arena(env, debate_agents, protocol)
        result = await arena.run()

        self._events.append(
            {
                "type": "debate_completed",
                "topic": topic,
                "rounds": proto_opts["rounds"],
                "consensus_reached": result.consensus_reached
                if hasattr(result, "consensus_reached")
                else False,
                "timestamp": time.time(),
            }
        )

        return result

    async def run_debate_via_control_plane(
        self,
        topic: str,
        rounds: int = 3,
    ) -> Dict[str, Any]:
        """Run a debate by submitting it as a task to the control plane.

        This tests the full flow: submit task -> claim -> process -> complete.
        """
        # Submit debate task
        task_id = await self.submit_task(
            task_type="debate",
            payload={
                "topic": topic,
                "rounds": rounds,
            },
            required_capabilities=["debate"],
        )

        # Process with first available debate agent
        debate_agent = next(
            (a for a in self.agents if "debate" in a.capabilities),
            self.agents[0] if self.agents else None,
        )

        if not debate_agent:
            raise RuntimeError("No debate-capable agent available")

        # Claim and process
        task = await self.coordinator.claim_task(
            agent_id=debate_agent.id,
            capabilities=debate_agent.capabilities,
            block_ms=1000,
        )

        if not task:
            raise RuntimeError("Failed to claim debate task")

        # Simulate debate processing
        await asyncio.sleep(self.config.agent_response_delay * rounds)

        result = {
            "topic": topic,
            "rounds_completed": rounds,
            "consensus_reached": True,
            "final_answer": f"Consensus reached by {debate_agent.id}",
            "participants": [a.id for a in self.agents[:3]],
        }

        await self.coordinator.complete_task(
            task_id=task_id,
            result=result,
            agent_id=debate_agent.id,
            latency_ms=self.config.agent_response_delay * rounds * 1000,
        )

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive harness statistics."""
        cp_stats = await self.coordinator.get_stats() if self.coordinator else {}

        agent_stats = [agent.get_stats() for agent in self.agents]

        return {
            "control_plane": cp_stats,
            "agents": agent_stats,
            "events": len(self._events),
            "running": self._running,
        }

    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recorded events, optionally filtered by type."""
        if event_type:
            return [e for e in self._events if e.get("type") == event_type]
        return self._events.copy()

    def clear_events(self) -> None:
        """Clear recorded events."""
        self._events.clear()

    async def wait_for_agents_ready(self, timeout: float = 5.0) -> bool:
        """Wait for all agents to be ready."""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            all_ready = True
            for agent in self.agents:
                info = await self.coordinator.get_agent(agent.id)
                if not info or info.status != AgentStatus.READY:
                    all_ready = False
                    break

            if all_ready:
                return True

            await asyncio.sleep(0.1)

        return False


# ============================================================================
# Context Manager
# ============================================================================


@asynccontextmanager
async def e2e_environment(
    config: Optional[E2ETestConfig] = None,
) -> AsyncGenerator[E2ETestHarness, None]:
    """Context manager for E2E test environment.

    Usage:
        async with e2e_environment() as harness:
            task_id = await harness.submit_task("debate", {"topic": "..."})
            result = await harness.wait_for_task(task_id)
    """
    harness = E2ETestHarness(config)
    try:
        await harness.start()
        yield harness
    finally:
        await harness.stop()


# ============================================================================
# Specialized Harness Variants
# ============================================================================


class DebateTestHarness(E2ETestHarness):
    """Specialized harness for debate-focused testing.

    Provides additional helpers for debate scenarios:
    - Pre-configured debate agents
    - Consensus tracking
    - Vote aggregation
    """

    def __init__(self, config: Optional[E2ETestConfig] = None):
        debate_config = config or E2ETestConfig(
            num_agents=4,
            agent_capabilities=["debate", "critique", "vote"],
            default_debate_rounds=3,
        )
        super().__init__(debate_config)
        self._debate_results: List[DebateResult] = []

    async def run_debate_with_tracking(
        self,
        topic: str,
        rounds: int = 3,
    ) -> DebateResult:
        """Run a debate and track the result."""
        result = await self.run_debate(topic, rounds=rounds)
        self._debate_results.append(result)
        return result

    def get_debate_results(self) -> List[DebateResult]:
        """Get all tracked debate results."""
        return self._debate_results.copy()

    def get_consensus_rate(self) -> float:
        """Calculate the rate of debates that reached consensus."""
        if not self._debate_results:
            return 0.0
        consensus_count = sum(
            1 for r in self._debate_results if getattr(r, "consensus_reached", False)
        )
        return consensus_count / len(self._debate_results)


class LoadTestHarness(E2ETestHarness):
    """Specialized harness for load testing.

    Provides helpers for:
    - Concurrent task submission
    - Throughput measurement
    - Latency percentile tracking
    """

    def __init__(self, config: Optional[E2ETestConfig] = None):
        load_config = config or E2ETestConfig(
            num_agents=10,
            agent_response_delay=0.01,  # Fast responses for load testing
            timeout_seconds=60.0,
        )
        super().__init__(load_config)
        self._latencies: List[float] = []

    async def submit_concurrent_tasks(
        self,
        count: int,
        task_type: str = "test",
        payload_generator: Optional[Callable[[int], Dict[str, Any]]] = None,
    ) -> List[str]:
        """Submit multiple tasks concurrently."""

        async def submit_one(index: int) -> str:
            payload = payload_generator(index) if payload_generator else {"index": index}
            return await self.submit_task(task_type, payload)

        tasks = [submit_one(i) for i in range(count)]
        return await asyncio.gather(*tasks)

    async def process_all_tasks(self, task_ids: List[str]) -> List[Task]:
        """Process all submitted tasks and collect results."""
        results = await asyncio.gather(
            *[self.wait_for_task(tid) for tid in task_ids],
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, Task)]

    async def measure_throughput(
        self,
        task_count: int,
        task_type: str = "test",
    ) -> Dict[str, float]:
        """Measure task throughput."""
        start = time.monotonic()

        task_ids = await self.submit_concurrent_tasks(task_count, task_type)
        results = await self.process_all_tasks(task_ids)

        elapsed = time.monotonic() - start
        successful = len(results)

        return {
            "total_tasks": task_count,
            "successful_tasks": successful,
            "elapsed_seconds": elapsed,
            "tasks_per_second": successful / elapsed if elapsed > 0 else 0,
            "success_rate": successful / task_count if task_count > 0 else 0,
        }
