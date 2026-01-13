"""
Debate termination checking.

Extracted from Arena orchestrator to improve code organization and testability.
Handles judge-based termination and early stopping via agent consensus.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.core import Agent, Message
    from aragora.debate.protocol import DebateProtocol

logger = logging.getLogger(__name__)


class TerminationChecker:
    """Checks if a debate should terminate early.

    Implements two termination strategies:
    1. Judge termination: A judge agent evaluates if debate is conclusive
    2. Early stopping: Agents vote on whether more debate would help

    Both strategies respect minimum round requirements and protocol settings.
    """

    def __init__(
        self,
        protocol: "DebateProtocol",
        agents: list["Agent"],
        generate_fn: Callable[["Agent", str, list["Message"]], Any],
        task: str,
        select_judge_fn: Optional[Callable[[dict[str, str], list["Message"]], Any]] = None,
        hooks: Optional[dict[str, Callable]] = None,
    ) -> None:
        """Initialize the termination checker.

        Args:
            protocol: Debate protocol with termination settings
            agents: List of participating agents
            generate_fn: Async function to generate agent responses
            task: The debate task description
            select_judge_fn: Async function to select a judge agent
            hooks: Optional hooks for termination events
        """
        self.protocol = protocol
        self.agents = agents
        self.generate_fn = generate_fn
        self.task = task
        self.select_judge_fn = select_judge_fn
        self.hooks = hooks or {}

    async def check_judge_termination(
        self,
        round_num: int,
        proposals: dict[str, str],
        context: list["Message"],
    ) -> tuple[bool, str]:
        """Have a judge evaluate if the debate is conclusive.

        Args:
            round_num: Current round number
            proposals: Dict of agent name to proposal text
            context: Recent message context

        Returns:
            Tuple of (should_continue: bool, reason: str)
            - (True, "") means continue the debate
            - (False, reason) means stop with the given reason
        """
        if not self.protocol.judge_termination:
            return True, ""

        if round_num < self.protocol.min_rounds_before_judge_check:
            return True, ""

        if not self.select_judge_fn:
            logger.warning("Judge termination enabled but no judge selector provided")
            return True, ""

        # Select a judge
        judge = await self.select_judge_fn(proposals, context)

        prompt = f"""You are evaluating whether this multi-agent debate (decision stress-test) has reached a conclusive state.

Task: {self.task[:300]}

After {round_num} rounds of debate, the proposals are:
{chr(10).join(f"- {agent}: {prop[:200]}..." for agent, prop in proposals.items())}

Evaluate:
1. Have the key issues been thoroughly discussed?
2. Are there major unresolved disagreements that more debate could resolve?
3. Would additional rounds likely produce meaningful improvements?

Respond with:
CONCLUSIVE: <yes/no>
REASON: <brief explanation>"""

        try:
            response = await self.generate_fn(judge, prompt, context[-5:])
            lines = str(response).strip().split("\n")

            conclusive = False
            reason = ""

            for line in lines:
                if line.upper().startswith("CONCLUSIVE:"):
                    val = line.split(":", 1)[1].strip().lower()
                    conclusive = val in ("yes", "true", "1")
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            if conclusive:
                logger.info(f"judge_termination judge={judge.name} reason={reason[:100]}")
                # Emit event
                if "on_judge_termination" in self.hooks:
                    self.hooks["on_judge_termination"](judge.name, reason)
                return False, reason

        except Exception as e:
            logger.warning(f"Judge termination check failed: {e}")

        return True, ""

    async def check_early_stopping(
        self,
        round_num: int,
        proposals: dict[str, str],
        context: list["Message"],
    ) -> bool:
        """Check if agents want to stop debate early.

        Args:
            round_num: Current round number
            proposals: Dict of agent name to proposal text (unused, for consistency)
            context: Recent message context

        Returns:
            True if debate should continue, False if it should stop.
        """
        if not self.protocol.early_stopping:
            return True  # Continue

        if round_num < self.protocol.min_rounds_before_early_stop:
            return True  # Continue - haven't met minimum rounds

        # Ask each agent if they think more debate would help
        prompt = f"""After {round_num} round(s) of debate on this task:
Task: {self.task[:200]}

Current proposals have been critiqued and revised. Do you think additional debate
rounds would significantly improve the answer quality?

Respond with only: CONTINUE or STOP
- CONTINUE: More debate rounds would help refine the answer
- STOP: The proposals are mature enough, further debate is unlikely to help"""

        stop_votes = 0
        total_votes = 0

        tasks = [self.generate_fn(agent, prompt, context[-5:]) for agent in self.agents]
        try:
            # Use wait_for for Python 3.10 compatibility (asyncio.timeout is 3.11+)
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.protocol.round_timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Timeout during early stopping check - continue debate (safe default)
            logger.warning(
                f"Early stopping check timed out after {self.protocol.round_timeout_seconds}s"
            )
            return True

        for agent, result in zip(self.agents, results):
            if isinstance(result, BaseException):
                continue
            total_votes += 1
            response = str(result).strip().upper()
            if "STOP" in response and "CONTINUE" not in response:
                stop_votes += 1

        if total_votes == 0:
            return True  # Continue if voting failed

        stop_ratio = stop_votes / total_votes
        should_stop = stop_ratio >= self.protocol.early_stop_threshold

        if should_stop:
            logger.info(f"early_stopping votes={stop_votes}/{total_votes}")
            # Emit early stop event
            if "on_early_stop" in self.hooks:
                self.hooks["on_early_stop"](round_num, stop_votes, total_votes)

        return not should_stop  # Return True to continue, False to stop

    async def should_terminate(
        self,
        round_num: int,
        proposals: dict[str, str],
        context: list["Message"],
    ) -> tuple[bool, str]:
        """Check both termination conditions.

        Convenience method that checks both judge termination and early stopping.

        Args:
            round_num: Current round number
            proposals: Dict of agent name to proposal text
            context: Recent message context

        Returns:
            Tuple of (should_stop: bool, reason: str)
            - (True, reason) means stop the debate
            - (False, "") means continue
        """
        # Check judge termination first
        should_continue, reason = await self.check_judge_termination(round_num, proposals, context)
        if not should_continue:
            return True, reason

        # Check early stopping
        should_continue = await self.check_early_stopping(round_num, proposals, context)
        if not should_continue:
            return True, "Agents voted to stop early"

        return False, ""
