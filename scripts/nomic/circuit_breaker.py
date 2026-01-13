"""
Circuit breaker pattern for agent reliability in nomic loop.

Tracks consecutive failures per agent and temporarily disables
agents that fail repeatedly to prevent wasting cycles on broken agents.
"""

import logging


class AgentCircuitBreaker:
    """
    Circuit breaker pattern for agent reliability.

    Tracks consecutive failures per agent and temporarily disables
    agents that fail repeatedly to prevent wasting cycles on broken agents.

    Extended with task-scoped tracking (Jan 2026):
    - Tracks failures per task type (debate, design, implement, verify)
    - Agents can be disabled for specific task types while still usable for others
    - Success rates tracked for intelligent agent selection
    """

    def __init__(self, failure_threshold: int = 3, cooldown_cycles: int = 2):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before tripping
            cooldown_cycles: Number of cycles to skip after tripping
        """
        self.failure_threshold = failure_threshold
        self.cooldown_cycles = cooldown_cycles
        self.failures: dict[str, int] = {}  # agent_name -> consecutive failure count
        self.cooldowns: dict[str, int] = {}  # agent_name -> cycles remaining in cooldown

        # Task-scoped tracking (new)
        self.task_failures: dict[str, dict[str, int]] = {}  # agent -> task_type -> count
        self.task_success_rate: dict[str, dict[str, float]] = {}  # agent -> task_type -> rate
        self.task_cooldowns: dict[str, dict[str, int]] = {}  # agent -> task_type -> cooldown

    def record_success(self, agent_name: str) -> None:
        """Reset failure count and reduce cooldown on success."""
        self.failures[agent_name] = 0
        # If agent was in cooldown but succeeded (half-open state), reduce or close circuit
        if agent_name in self.cooldowns and self.cooldowns[agent_name] > 0:
            self.cooldowns[agent_name] = max(0, self.cooldowns[agent_name] - 1)
            if self.cooldowns[agent_name] == 0:
                del self.cooldowns[agent_name]
                logging.info(f"[circuit-breaker] {agent_name} recovered after success")

    def record_failure(self, agent_name: str) -> bool:
        """
        Record a failure and potentially trip the circuit.

        Returns:
            True if circuit just tripped (agent now in cooldown)
        """
        self.failures[agent_name] = self.failures.get(agent_name, 0) + 1
        if self.failures[agent_name] >= self.failure_threshold:
            self.cooldowns[agent_name] = self.cooldown_cycles
            self.failures[agent_name] = 0  # Reset for next time
            return True
        return False

    def record_task_success(self, agent_name: str, task_type: str) -> None:
        """Record success for specific task type and update running average."""
        # Initialize structures if needed
        if agent_name not in self.task_success_rate:
            self.task_success_rate[agent_name] = {}
        if agent_name not in self.task_failures:
            self.task_failures[agent_name] = {}

        # Reset task-specific failures
        self.task_failures[agent_name][task_type] = 0

        # Update running average (exponential moving average)
        current_rate = self.task_success_rate[agent_name].get(task_type, 0.5)
        self.task_success_rate[agent_name][task_type] = current_rate * 0.8 + 0.2

        # Also record agent-level success
        self.record_success(agent_name)

    def record_task_failure(self, agent_name: str, task_type: str) -> bool:
        """
        Record failure for specific task type.

        Returns:
            True if task-specific circuit just tripped
        """
        # Initialize structures if needed
        if agent_name not in self.task_failures:
            self.task_failures[agent_name] = {}
        if agent_name not in self.task_cooldowns:
            self.task_cooldowns[agent_name] = {}
        if agent_name not in self.task_success_rate:
            self.task_success_rate[agent_name] = {}

        # Increment task-specific failure count
        self.task_failures[agent_name][task_type] = (
            self.task_failures[agent_name].get(task_type, 0) + 1
        )

        # Update running average (exponential moving average toward 0)
        current_rate = self.task_success_rate[agent_name].get(task_type, 0.5)
        self.task_success_rate[agent_name][task_type] = current_rate * 0.8

        # Trip task-specific circuit if threshold reached
        if self.task_failures[agent_name][task_type] >= self.failure_threshold:
            self.task_cooldowns[agent_name][task_type] = self.cooldown_cycles
            self.task_failures[agent_name][task_type] = 0
            return True

        # Also record agent-level failure
        self.record_failure(agent_name)
        return False

    def get_task_success_rate(self, agent_name: str, task_type: str) -> float:
        """Get agent's success rate for specific task type (0.0 to 1.0)."""
        if agent_name not in self.task_success_rate:
            return 0.5  # Default neutral
        return self.task_success_rate[agent_name].get(task_type, 0.5)

    def is_available_for_task(self, agent_name: str, task_type: str) -> bool:
        """Check if agent is available for a specific task type."""
        # First check global availability
        if not self.is_available(agent_name):
            return False
        # Then check task-specific cooldown
        if agent_name in self.task_cooldowns:
            if self.task_cooldowns[agent_name].get(task_type, 0) > 0:
                return False
        return True

    def is_available(self, agent_name: str) -> bool:
        """Check if agent is available (not in cooldown)."""
        return self.cooldowns.get(agent_name, 0) <= 0

    def start_new_cycle(self) -> None:
        """Decrement cooldowns at start of each cycle."""
        # Decrement global cooldowns
        for agent_name in list(self.cooldowns.keys()):
            if self.cooldowns[agent_name] > 0:
                self.cooldowns[agent_name] -= 1

        # Decrement task-specific cooldowns
        for agent_name in list(self.task_cooldowns.keys()):
            for task_type in list(self.task_cooldowns[agent_name].keys()):
                if self.task_cooldowns[agent_name][task_type] > 0:
                    self.task_cooldowns[agent_name][task_type] -= 1

    def get_status(self) -> dict:
        """Get circuit breaker status for all agents."""
        return {
            "failures": dict(self.failures),
            "cooldowns": dict(self.cooldowns),
            "task_failures": dict(self.task_failures),
            "task_cooldowns": dict(self.task_cooldowns),
            "task_success_rates": dict(self.task_success_rate),
        }
