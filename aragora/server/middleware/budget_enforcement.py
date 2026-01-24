"""
Budget Enforcement Middleware.

Provides decorators and middleware for enforcing budget limits
on operations before they execute.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class BudgetExceededError(Exception):
    """Raised when an operation would exceed budget limits."""

    def __init__(self, message: str, budget_id: Optional[str] = None, action: Optional[str] = None):
        super().__init__(message)
        self.budget_id = budget_id
        self.action = action


class BudgetWarning:
    """Warning about approaching budget limits."""

    def __init__(self, message: str, usage_percentage: float, action: str):
        self.message = message
        self.usage_percentage = usage_percentage
        self.action = action


def check_budget(
    estimated_cost_usd: Optional[float] = None,
    cost_estimator: Optional[Callable[..., float]] = None,
    on_warning: Optional[Callable[[BudgetWarning], bool]] = None,
) -> Callable[[F], F]:
    """Decorator to check budget before executing an operation.

    Args:
        estimated_cost_usd: Static estimated cost in USD
        cost_estimator: Function to dynamically estimate cost from args
        on_warning: Callback for budget warnings (return False to cancel)

    Example:
        @check_budget(estimated_cost_usd=0.10)
        async def run_debate(topic: str):
            ...

        @check_budget(cost_estimator=lambda rounds, agents: rounds * len(agents) * 0.02)
        async def run_debate_dynamic(rounds: int, agents: list):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get org_id from kwargs or context
            org_id = kwargs.get("org_id")
            user_id = kwargs.get("user_id")

            if not org_id:
                # Try to extract from first arg if it's a context object
                if args and hasattr(args[0], "org_id"):
                    org_id = args[0].org_id
                if args and hasattr(args[0], "user_id"):
                    user_id = args[0].user_id

            if not org_id:
                # No org context - skip budget check
                return await func(*args, **kwargs)

            # Calculate estimated cost
            cost = estimated_cost_usd or 0.0
            if cost_estimator:
                try:
                    cost = cost_estimator(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Cost estimator failed: {e}")

            if cost <= 0:
                # No cost - skip budget check
                return await func(*args, **kwargs)

            # Check budget
            try:
                from aragora.billing.budget_manager import get_budget_manager, BudgetAction

                manager = get_budget_manager()
                allowed, reason, action = manager.check_budget(
                    org_id=org_id,
                    estimated_cost_usd=cost,
                    user_id=user_id,
                )

                if not allowed:
                    raise BudgetExceededError(
                        reason,
                        action=action.value if action else None,
                    )

                # Handle soft limits (warnings)
                if action == BudgetAction.SOFT_LIMIT or action == BudgetAction.WARN:
                    warning = BudgetWarning(
                        message=reason,
                        usage_percentage=0.0,  # Would need to get from budget
                        action=action.value,
                    )

                    if on_warning:
                        proceed = on_warning(warning)
                        if not proceed:
                            raise BudgetExceededError(
                                "Operation cancelled due to budget warning",
                                action=action.value,
                            )

            except ImportError:
                # Budget manager not available - allow
                pass
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.warning(f"Budget check failed: {e}")
                # Fail open - allow operation if check fails

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get org_id from kwargs or context
            org_id = kwargs.get("org_id")
            user_id = kwargs.get("user_id")

            if not org_id:
                if args and hasattr(args[0], "org_id"):
                    org_id = args[0].org_id
                if args and hasattr(args[0], "user_id"):
                    user_id = args[0].user_id

            if not org_id:
                return func(*args, **kwargs)

            cost = estimated_cost_usd or 0.0
            if cost_estimator:
                try:
                    cost = cost_estimator(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Cost estimator failed: {e}")

            if cost <= 0:
                return func(*args, **kwargs)

            try:
                from aragora.billing.budget_manager import get_budget_manager, BudgetAction

                manager = get_budget_manager()
                allowed, reason, action = manager.check_budget(
                    org_id=org_id,
                    estimated_cost_usd=cost,
                    user_id=user_id,
                )

                if not allowed:
                    raise BudgetExceededError(
                        reason,
                        action=action.value if action else None,
                    )

                if action == BudgetAction.SOFT_LIMIT or action == BudgetAction.WARN:
                    warning = BudgetWarning(
                        message=reason,
                        usage_percentage=0.0,
                        action=action.value,
                    )

                    if on_warning:
                        proceed = on_warning(warning)
                        if not proceed:
                            raise BudgetExceededError(
                                "Operation cancelled due to budget warning",
                                action=action.value,
                            )

            except ImportError:
                pass
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.warning(f"Budget check failed: {e}")

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio_iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator


def asyncio_iscoroutinefunction(func: Any) -> bool:
    """Check if function is async."""
    import asyncio
    import inspect

    return asyncio.iscoroutinefunction(func) or inspect.iscoroutinefunction(func)


def record_spend(
    cost_calculator: Optional[Callable[..., float]] = None,
    description_template: str = "Operation",
) -> Callable[[F], F]:
    """Decorator to record spending after successful operation.

    Args:
        cost_calculator: Function to calculate actual cost from result
        description_template: Template for spend description

    Example:
        @record_spend(
            cost_calculator=lambda result: result.total_tokens * 0.00001,
            description_template="Debate: {topic}"
        )
        async def run_debate(topic: str):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)

            # Get context
            org_id = kwargs.get("org_id")
            user_id = kwargs.get("user_id")
            debate_id = kwargs.get("debate_id")

            if not org_id:
                if args and hasattr(args[0], "org_id"):
                    org_id = args[0].org_id
                if args and hasattr(args[0], "user_id"):
                    user_id = args[0].user_id
                if args and hasattr(args[0], "debate_id"):
                    debate_id = args[0].debate_id

            if not org_id:
                return result

            # Calculate cost
            cost = 0.0
            if cost_calculator:
                try:
                    cost = cost_calculator(result)
                except Exception as e:
                    logger.warning(f"Cost calculator failed: {e}")

            if cost <= 0:
                return result

            # Build description
            description = description_template
            try:
                description = description_template.format(**kwargs)
            except KeyError:
                pass

            # Record spend
            try:
                from aragora.billing.budget_manager import get_budget_manager

                manager = get_budget_manager()
                manager.record_spend(
                    org_id=org_id,
                    amount_usd=cost,
                    description=description,
                    debate_id=debate_id,
                    user_id=user_id,
                )
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to record spend: {e}")

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            org_id = kwargs.get("org_id")
            user_id = kwargs.get("user_id")
            debate_id = kwargs.get("debate_id")

            if not org_id:
                if args and hasattr(args[0], "org_id"):
                    org_id = args[0].org_id
                if args and hasattr(args[0], "user_id"):
                    user_id = args[0].user_id
                if args and hasattr(args[0], "debate_id"):
                    debate_id = args[0].debate_id

            if not org_id:
                return result

            cost = 0.0
            if cost_calculator:
                try:
                    cost = cost_calculator(result)
                except Exception as e:
                    logger.warning(f"Cost calculator failed: {e}")

            if cost <= 0:
                return result

            description = description_template
            try:
                description = description_template.format(**kwargs)
            except KeyError:
                pass

            try:
                from aragora.billing.budget_manager import get_budget_manager

                manager = get_budget_manager()
                manager.record_spend(
                    org_id=org_id,
                    amount_usd=cost,
                    description=description,
                    debate_id=debate_id,
                    user_id=user_id,
                )
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to record spend: {e}")

            return result

        if asyncio_iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator


def estimate_debate_cost(
    rounds: int = 3,
    agents: int = 2,
    avg_tokens_per_round: int = 2000,
    cost_per_1k_tokens: float = 0.003,
) -> float:
    """Estimate cost for a debate.

    Args:
        rounds: Number of debate rounds
        agents: Number of agents
        avg_tokens_per_round: Average tokens per agent per round
        cost_per_1k_tokens: Cost per 1000 tokens

    Returns:
        Estimated cost in USD
    """
    total_tokens = rounds * agents * avg_tokens_per_round * 2  # Input + output
    return (total_tokens / 1000) * cost_per_1k_tokens


def estimate_gauntlet_cost(
    probes: int = 10,
    attacks: int = 5,
    avg_tokens_per_operation: int = 3000,
    cost_per_1k_tokens: float = 0.003,
) -> float:
    """Estimate cost for a gauntlet run.

    Args:
        probes: Number of capability probes
        attacks: Number of red team attacks
        avg_tokens_per_operation: Average tokens per operation
        cost_per_1k_tokens: Cost per 1000 tokens

    Returns:
        Estimated cost in USD
    """
    total_operations = probes + attacks
    total_tokens = total_operations * avg_tokens_per_operation * 2
    return (total_tokens / 1000) * cost_per_1k_tokens


# Pre-built cost estimators for common operations
def DEBATE_COST_ESTIMATOR(*args: Any, **kwargs: Any) -> float:
    """Estimate cost for debate operations."""
    return estimate_debate_cost(
        rounds=kwargs.get("rounds", 3),
        agents=len(kwargs.get("agents", [])) or 2,
    )


def GAUNTLET_COST_ESTIMATOR(*args: Any, **kwargs: Any) -> float:
    """Estimate cost for gauntlet operations."""
    return estimate_gauntlet_cost(
        probes=kwargs.get("probes", 10),
        attacks=kwargs.get("attacks", 5),
    )


__all__ = [
    "BudgetExceededError",
    "BudgetWarning",
    "check_budget",
    "record_spend",
    "estimate_debate_cost",
    "estimate_gauntlet_cost",
    "DEBATE_COST_ESTIMATOR",
    "GAUNTLET_COST_ESTIMATOR",
]
