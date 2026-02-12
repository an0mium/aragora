"""
Budget coordination for debate orchestration.

Provides budget enforcement before, during, and after debates to ensure
organizations stay within their allocated spending limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aragora.logging_config import get_logger

if TYPE_CHECKING:
    from aragora.core import DebateResult

logger = get_logger(__name__)


class BudgetCoordinator:
    """Coordinates budget checks and cost recording for debates.

    This coordinator handles:
    - Pre-debate budget validation
    - Mid-debate budget checks for graceful pauses
    - Post-debate cost recording

    Args:
        org_id: Organization identifier for budget tracking
        user_id: Optional user identifier for per-user tracking
    """

    # Cost estimates for budget planning
    ESTIMATED_DEBATE_COST_USD = 0.10  # Conservative estimate for 3-round debate
    ESTIMATED_ROUND_COST_USD = 0.03  # Per-round estimate for mid-debate checks
    ESTIMATED_MESSAGE_COST_USD = 0.01  # Fallback per-message estimate

    def __init__(
        self,
        org_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Initialize budget coordinator.

        Args:
            org_id: Organization identifier for budget tracking
            user_id: Optional user identifier for per-user tracking
        """
        self.org_id = org_id
        self.user_id = user_id

    def check_budget_before_debate(self, debate_id: str) -> None:
        """Check if organization has sufficient budget before starting debate.

        Args:
            debate_id: Debate identifier for logging

        Raises:
            BudgetExceededError: If budget is exhausted and hard-stop is enforced.
        """
        if not self.org_id:
            return  # No org context - skip budget check

        try:
            from aragora.billing.budget_manager import BudgetAction, get_budget_manager

            manager = get_budget_manager()

            allowed, reason, action = manager.check_budget(
                org_id=self.org_id,
                estimated_cost_usd=self.ESTIMATED_DEBATE_COST_USD,
                user_id=self.user_id or None,
            )

            if not allowed:
                logger.warning(
                    f"budget_check_failed org_id={self.org_id} debate_id={debate_id} reason={reason}"
                )
                from aragora.exceptions import BudgetExceededError

                raise BudgetExceededError(f"Budget limit reached for organization: {reason}")

            if action == BudgetAction.SOFT_LIMIT:
                logger.warning(
                    f"budget_soft_limit_warning org_id={self.org_id} debate_id={debate_id} reason={reason}"
                )

        except ImportError:
            # Budget manager not available - proceed without check
            logger.debug("Budget manager not available, skipping pre-debate check")

    def check_budget_mid_debate(self, debate_id: str, round_num: int) -> tuple[bool, str]:
        """Check if organization has sufficient budget to continue debate mid-execution.

        Unlike check_budget_before_debate(), this method returns a tuple instead of
        raising an exception, allowing the debate to pause gracefully rather than
        fail abruptly.

        Args:
            debate_id: Debate identifier for logging
            round_num: Current round number for logging

        Returns:
            Tuple of (allowed: bool, reason: str)
            - allowed: True if debate can continue, False if budget exceeded
            - reason: Human-readable reason if budget check failed
        """
        if not self.org_id:
            return True, ""  # No org context - allow continuation

        try:
            from aragora.billing.budget_manager import BudgetAction, get_budget_manager

            manager = get_budget_manager()

            allowed, reason, action = manager.check_budget(
                org_id=self.org_id,
                estimated_cost_usd=self.ESTIMATED_ROUND_COST_USD,
                user_id=self.user_id or None,
            )

            if not allowed:
                logger.warning(
                    f"budget_exceeded_mid_debate org_id={self.org_id} "
                    f"debate_id={debate_id} round={round_num} reason={reason}"
                )
                return False, reason

            if action == BudgetAction.SOFT_LIMIT:
                logger.info(
                    f"budget_soft_limit_mid_debate org_id={self.org_id} "
                    f"debate_id={debate_id} round={round_num} reason={reason}"
                )
                # Continue but warn - could be logged for alerting

            return True, ""

        except ImportError:
            # Budget manager not available - allow continuation
            return True, ""
        except (ConnectionError, OSError, ValueError, TypeError, AttributeError) as e:
            # On any error, allow continuation (fail open for availability)
            logger.debug(f"Budget check error (continuing): {e}")
            return True, ""

    def record_debate_cost(
        self,
        debate_id: str,
        result: DebateResult,
        extensions: Any | None = None,
    ) -> None:
        """Record actual debate cost against organization budget.

        Args:
            debate_id: Debate identifier
            result: Completed debate result with token usage info
            extensions: Optional extensions object with cost tracking
        """
        if not self.org_id:
            return  # No org context - skip cost recording

        try:
            from aragora.billing.budget_manager import get_budget_manager

            manager = get_budget_manager()

            # Calculate actual cost from usage metrics
            actual_cost_usd = self._calculate_actual_cost(result, extensions)

            if actual_cost_usd > 0:
                manager.record_spend(
                    org_id=self.org_id,
                    amount_usd=actual_cost_usd,
                    description=f"Debate: {result.task[:50] if result.task else 'Unknown'}",
                    debate_id=debate_id,
                    user_id=self.user_id or None,
                )
                logger.info(
                    f"debate_cost_recorded org_id={self.org_id} debate_id={debate_id} cost=${actual_cost_usd:.4f}"
                )

        except ImportError:
            logger.debug("Budget manager not available, skipping cost recording")

    def _calculate_actual_cost(
        self,
        result: DebateResult,
        extensions: Any | None = None,
    ) -> float:
        """Calculate actual cost from debate result and extensions.

        Priority:
        1. Extensions total_cost_usd attribute
        2. Result metadata total_cost_usd
        3. Fallback: message count * per-message estimate

        Args:
            result: Completed debate result
            extensions: Optional extensions with cost tracking

        Returns:
            Estimated cost in USD
        """
        # Check if extensions recorded usage
        if extensions is not None and hasattr(extensions, "total_cost_usd"):
            cost = getattr(extensions, "total_cost_usd", 0.0)
            if cost > 0:
                return cost

        # Check result metadata
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            cost = float(result.metadata.get("total_cost_usd", 0.0))
            if cost > 0:
                return cost

        # Fallback: estimate from message count
        message_count = len(result.messages) if result.messages else 0
        critique_count = len(result.critiques) if result.critiques else 0
        return (message_count + critique_count) * self.ESTIMATED_MESSAGE_COST_USD
