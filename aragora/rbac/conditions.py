"""
ABAC Condition Evaluator.

Provides advanced attribute-based access control (ABAC) condition evaluation
beyond simple equality matching. Supports:
- Time-based access (business hours, expiration)
- IP-based restrictions (allowlist, CIDR)
- Resource attribute matching (owner, status, tags)
- Custom condition functions

Usage:
    from aragora.rbac.conditions import ConditionEvaluator, TimeCondition

    evaluator = ConditionEvaluator()

    # Register conditions
    evaluator.register_condition("business_hours", TimeCondition(
        allowed_hours=(9, 17),
        allowed_days={0, 1, 2, 3, 4},  # Mon-Fri
        timezone="America/New_York",
    ))

    # Check conditions
    result = evaluator.evaluate(
        conditions={"business_hours": True, "ip_allowlist": ["10.0.0.0/8"]},
        context={"ip_address": "10.1.2.3", "current_time": datetime.now()},
    )
"""

from __future__ import annotations

import ipaddress
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ConditionError(Exception):
    """Error evaluating a condition."""

    pass


@dataclass
class ConditionResult:
    """Result of condition evaluation."""

    satisfied: bool
    condition_name: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class Condition(ABC):
    """Abstract base class for conditions."""

    @abstractmethod
    def evaluate(self, expected: Any, context: Dict[str, Any]) -> ConditionResult:
        """Evaluate the condition.

        Args:
            expected: Expected value from the permission's conditions
            context: Runtime context with actual values

        Returns:
            ConditionResult indicating if condition is satisfied
        """
        pass


class EqualityCondition(Condition):
    """Simple equality condition (default behavior)."""

    def __init__(self, context_key: str):
        self.context_key = context_key

    def evaluate(self, expected: Any, context: Dict[str, Any]) -> ConditionResult:
        actual = context.get(self.context_key)
        satisfied = actual == expected
        return ConditionResult(
            satisfied=satisfied,
            condition_name=f"equality:{self.context_key}",
            reason=f"{'Matched' if satisfied else 'Mismatched'}: {actual} {'==' if satisfied else '!='} {expected}",
            details={"expected": expected, "actual": actual},
        )


class TimeCondition(Condition):
    """Time-based access condition.

    Supports:
    - Business hours (start/end time)
    - Allowed days of week
    - Timezone handling
    - Date ranges
    """

    def __init__(
        self,
        allowed_hours: Optional[tuple[int, int]] = None,  # (start_hour, end_hour)
        allowed_days: Optional[Set[int]] = None,  # 0=Mon, 6=Sun
        timezone_name: str = "UTC",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        self.allowed_hours = allowed_hours
        self.allowed_days = allowed_days
        self.timezone_name = timezone_name
        self.start_date = start_date
        self.end_date = end_date

    def evaluate(self, expected: Any, context: Dict[str, Any]) -> ConditionResult:
        # Get current time from context or use now
        current = context.get("current_time")
        if current is None:
            current = datetime.now(timezone.utc)
        elif not current.tzinfo:
            current = current.replace(tzinfo=timezone.utc)

        # Check date range
        if self.start_date and current < self.start_date:
            return ConditionResult(
                satisfied=False,
                condition_name="time:date_range",
                reason=f"Access not yet available (starts {self.start_date})",
            )
        if self.end_date and current > self.end_date:
            return ConditionResult(
                satisfied=False,
                condition_name="time:date_range",
                reason=f"Access expired (ended {self.end_date})",
            )

        # Check day of week
        if self.allowed_days is not None:
            day_of_week = current.weekday()
            if day_of_week not in self.allowed_days:
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                allowed_names = [day_names[d] for d in sorted(self.allowed_days)]
                return ConditionResult(
                    satisfied=False,
                    condition_name="time:day_of_week",
                    reason=f"Access not allowed on {day_names[day_of_week]} (allowed: {', '.join(allowed_names)})",
                )

        # Check hours
        if self.allowed_hours is not None:
            start_hour, end_hour = self.allowed_hours
            current_hour = current.hour
            if not (start_hour <= current_hour < end_hour):
                return ConditionResult(
                    satisfied=False,
                    condition_name="time:hours",
                    reason=f"Access not allowed at {current_hour}:00 (allowed: {start_hour}:00-{end_hour}:00)",
                )

        return ConditionResult(
            satisfied=True,
            condition_name="time",
            reason="Time conditions satisfied",
        )


class IPCondition(Condition):
    """IP-based access condition.

    Supports:
    - Single IP allowlist/blocklist
    - CIDR ranges
    - Private network checks
    """

    def __init__(self, context_key: str = "ip_address"):
        self.context_key = context_key

    def evaluate(self, expected: Any, context: Dict[str, Any]) -> ConditionResult:
        ip_str = context.get(self.context_key)
        if not ip_str:
            return ConditionResult(
                satisfied=False,
                condition_name="ip:missing",
                reason="No IP address in context",
            )

        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return ConditionResult(
                satisfied=False,
                condition_name="ip:invalid",
                reason=f"Invalid IP address: {ip_str}",
            )

        # Handle different expected value types
        if isinstance(expected, bool):
            # True = any IP allowed, False = no IP allowed
            return ConditionResult(
                satisfied=expected,
                condition_name="ip:boolean",
                reason="IP check always " + ("passes" if expected else "fails"),
            )

        if isinstance(expected, str):
            expected = [expected]

        if isinstance(expected, list):
            for pattern in expected:
                if self._matches_pattern(ip, pattern):
                    return ConditionResult(
                        satisfied=True,
                        condition_name="ip:allowlist",
                        reason=f"IP {ip_str} matches {pattern}",
                        details={"ip": ip_str, "matched_pattern": pattern},
                    )
            return ConditionResult(
                satisfied=False,
                condition_name="ip:allowlist",
                reason=f"IP {ip_str} not in allowlist",
                details={"ip": ip_str, "allowlist": expected},
            )

        if isinstance(expected, dict):
            # Handle complex conditions
            if "allowlist" in expected:
                for pattern in expected["allowlist"]:
                    if self._matches_pattern(ip, pattern):
                        return ConditionResult(
                            satisfied=True,
                            condition_name="ip:allowlist",
                            reason=f"IP {ip_str} in allowlist",
                        )
                return ConditionResult(
                    satisfied=False,
                    condition_name="ip:allowlist",
                    reason=f"IP {ip_str} not in allowlist",
                )

            if "blocklist" in expected:
                for pattern in expected["blocklist"]:
                    if self._matches_pattern(ip, pattern):
                        return ConditionResult(
                            satisfied=False,
                            condition_name="ip:blocklist",
                            reason=f"IP {ip_str} is blocklisted",
                        )

            if expected.get("require_private", False):
                if not ip.is_private:
                    return ConditionResult(
                        satisfied=False,
                        condition_name="ip:private",
                        reason=f"IP {ip_str} is not a private address",
                    )

        return ConditionResult(
            satisfied=True,
            condition_name="ip",
            reason="IP conditions satisfied",
        )

    def _matches_pattern(
        self, ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address], pattern: str
    ) -> bool:
        """Check if IP matches a pattern (single IP or CIDR)."""
        try:
            if "/" in pattern:
                network = ipaddress.ip_network(pattern, strict=False)
                return ip in network
            else:
                return ip == ipaddress.ip_address(pattern)
        except ValueError:
            return False


class ResourceOwnerCondition(Condition):
    """Resource ownership condition.

    Checks if the actor owns the resource or is in the owner group.
    """

    def __init__(
        self,
        actor_key: str = "actor_id",
        owner_key: str = "resource_owner",
        group_key: str = "resource_owner_group",
    ):
        self.actor_key = actor_key
        self.owner_key = owner_key
        self.group_key = group_key

    def evaluate(self, expected: Any, context: Dict[str, Any]) -> ConditionResult:
        actor = context.get(self.actor_key)
        owner = context.get(self.owner_key)
        owner_group = context.get(self.group_key, [])

        if not actor:
            return ConditionResult(
                satisfied=False,
                condition_name="owner:no_actor",
                reason="No actor in context",
            )

        # Direct ownership
        if actor == owner:
            return ConditionResult(
                satisfied=True,
                condition_name="owner:direct",
                reason=f"Actor {actor} owns the resource",
            )

        # Group ownership
        if actor in owner_group:
            return ConditionResult(
                satisfied=True,
                condition_name="owner:group",
                reason=f"Actor {actor} is in owner group",
            )

        return ConditionResult(
            satisfied=False,
            condition_name="owner",
            reason=f"Actor {actor} is not the owner ({owner}) or in group",
        )


class ResourceStatusCondition(Condition):
    """Resource status condition.

    Checks if resource is in an allowed status.
    """

    def __init__(self, status_key: str = "resource_status"):
        self.status_key = status_key

    def evaluate(self, expected: Any, context: Dict[str, Any]) -> ConditionResult:
        status = context.get(self.status_key)

        if status is None:
            return ConditionResult(
                satisfied=False,
                condition_name="status:missing",
                reason="No resource status in context",
            )

        # Handle single value or list
        if isinstance(expected, str):
            expected = [expected]

        if status in expected:
            return ConditionResult(
                satisfied=True,
                condition_name="status",
                reason=f"Resource status '{status}' is allowed",
            )

        return ConditionResult(
            satisfied=False,
            condition_name="status",
            reason=f"Resource status '{status}' not in allowed: {expected}",
        )


class TagCondition(Condition):
    """Resource tag condition.

    Checks if resource has required tags.
    """

    def __init__(self, tags_key: str = "resource_tags"):
        self.tags_key = tags_key

    def evaluate(self, expected: Any, context: Dict[str, Any]) -> ConditionResult:
        tags = context.get(self.tags_key, [])
        if isinstance(tags, str):
            tags = [tags]

        # Handle different expected formats
        if isinstance(expected, str):
            # Single required tag
            if expected in tags:
                return ConditionResult(
                    satisfied=True,
                    condition_name="tags:has",
                    reason=f"Resource has required tag '{expected}'",
                )
            return ConditionResult(
                satisfied=False,
                condition_name="tags:missing",
                reason=f"Resource missing required tag '{expected}'",
            )

        if isinstance(expected, list):
            # All tags required
            missing = [t for t in expected if t not in tags]
            if not missing:
                return ConditionResult(
                    satisfied=True,
                    condition_name="tags:all",
                    reason="Resource has all required tags",
                )
            return ConditionResult(
                satisfied=False,
                condition_name="tags:missing",
                reason=f"Resource missing tags: {missing}",
            )

        if isinstance(expected, dict):
            # Complex tag conditions
            if "any" in expected:
                # At least one tag required
                if any(t in tags for t in expected["any"]):
                    return ConditionResult(
                        satisfied=True,
                        condition_name="tags:any",
                        reason="Resource has at least one required tag",
                    )
                return ConditionResult(
                    satisfied=False,
                    condition_name="tags:any",
                    reason=f"Resource missing all of: {expected['any']}",
                )

            if "all" in expected:
                missing = [t for t in expected["all"] if t not in tags]
                if not missing:
                    return ConditionResult(
                        satisfied=True,
                        condition_name="tags:all",
                        reason="Resource has all required tags",
                    )
                return ConditionResult(
                    satisfied=False,
                    condition_name="tags:all",
                    reason=f"Resource missing tags: {missing}",
                )

            if "none" in expected:
                # Forbidden tags
                forbidden = [t for t in expected["none"] if t in tags]
                if forbidden:
                    return ConditionResult(
                        satisfied=False,
                        condition_name="tags:forbidden",
                        reason=f"Resource has forbidden tags: {forbidden}",
                    )

        return ConditionResult(
            satisfied=True,
            condition_name="tags",
            reason="Tag conditions satisfied",
        )


class ConditionEvaluator:
    """
    Evaluates ABAC conditions against runtime context.

    Supports built-in condition types and custom conditions.
    """

    def __init__(self):
        self._conditions: Dict[str, Condition] = {
            # Built-in conditions
            "time": TimeCondition(),
            "business_hours": TimeCondition(
                allowed_hours=(9, 17),
                allowed_days={0, 1, 2, 3, 4},  # Mon-Fri
            ),
            "ip": IPCondition(),
            "ip_address": IPCondition(),
            "ip_allowlist": IPCondition(),
            "owner": ResourceOwnerCondition(),
            "resource_owner": ResourceOwnerCondition(),
            "status": ResourceStatusCondition(),
            "resource_status": ResourceStatusCondition(),
            "tags": TagCondition(),
            "resource_tags": TagCondition(),
        }

        # Custom condition functions
        self._custom_conditions: Dict[str, Callable[[Any, Dict[str, Any]], bool]] = {}

    def register_condition(self, name: str, condition: Condition) -> None:
        """Register a condition evaluator."""
        self._conditions[name] = condition

    def register_custom(
        self,
        name: str,
        evaluator: Callable[[Any, Dict[str, Any]], bool],
    ) -> None:
        """Register a custom condition function.

        Args:
            name: Condition name
            evaluator: Function that takes (expected_value, context) and returns bool
        """
        self._custom_conditions[name] = evaluator

    def evaluate(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any],
    ) -> tuple[bool, List[ConditionResult]]:
        """Evaluate all conditions.

        Args:
            conditions: Conditions from permission definition
            context: Runtime context with actual values

        Returns:
            Tuple of (all_satisfied, list_of_results)
        """
        results: List[ConditionResult] = []
        all_satisfied = True

        for name, expected in conditions.items():
            result = self._evaluate_single(name, expected, context)
            results.append(result)
            if not result.satisfied:
                all_satisfied = False

        return all_satisfied, results

    def _evaluate_single(
        self,
        name: str,
        expected: Any,
        context: Dict[str, Any],
    ) -> ConditionResult:
        """Evaluate a single condition."""
        # Check built-in conditions
        if name in self._conditions:
            try:
                return self._conditions[name].evaluate(expected, context)
            except Exception as e:
                logger.error(f"Error evaluating condition {name}: {e}")
                return ConditionResult(
                    satisfied=False,
                    condition_name=name,
                    reason=f"Evaluation error: {e}",
                )

        # Check custom conditions
        if name in self._custom_conditions:
            try:
                satisfied = self._custom_conditions[name](expected, context)
                return ConditionResult(
                    satisfied=satisfied,
                    condition_name=f"custom:{name}",
                    reason="Custom condition " + ("passed" if satisfied else "failed"),
                )
            except Exception as e:
                logger.error(f"Error evaluating custom condition {name}: {e}")
                return ConditionResult(
                    satisfied=False,
                    condition_name=f"custom:{name}",
                    reason=f"Evaluation error: {e}",
                )

        # Fall back to equality check
        actual = context.get(name)
        satisfied = actual == expected
        return ConditionResult(
            satisfied=satisfied,
            condition_name=f"equality:{name}",
            reason=f"{'Matched' if satisfied else 'Mismatched'}: {actual} {'==' if satisfied else '!='} {expected}",
        )


# Global instance
_condition_evaluator: Optional[ConditionEvaluator] = None


def get_condition_evaluator() -> ConditionEvaluator:
    """Get the global condition evaluator."""
    global _condition_evaluator
    if _condition_evaluator is None:
        _condition_evaluator = ConditionEvaluator()
    return _condition_evaluator


__all__ = [
    "Condition",
    "ConditionResult",
    "ConditionError",
    "EqualityCondition",
    "TimeCondition",
    "IPCondition",
    "ResourceOwnerCondition",
    "ResourceStatusCondition",
    "TagCondition",
    "ConditionEvaluator",
    "get_condition_evaluator",
]
