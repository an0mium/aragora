"""
Declarative Condition Evaluation Engine.

Evaluates hook conditions against trigger context to determine
whether a hook should fire.

Supports:
- Field path traversal (e.g., "result.confidence")
- Multiple comparison operators
- Type coercion for comparisons
- Logical AND of multiple conditions
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from aragora.hooks.config import ConditionConfig

__all__ = [
    "ConditionEvaluator",
    "Operator",
]

logger = logging.getLogger(__name__)


class Operator(str, Enum):
    """Comparison operators for conditions."""

    # Equality
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal

    # Numeric comparison
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal

    # String/collection
    CONTAINS = "contains"  # String contains or collection includes
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex match

    # Type checks
    IS_NULL = "is_null"  # Value is None
    IS_NOT_NULL = "is_not_null"
    IS_EMPTY = "is_empty"  # Empty string/list/dict
    IS_NOT_EMPTY = "is_not_empty"

    # Collection
    IN = "in"  # Value in list
    NOT_IN = "not_in"
    HAS_KEY = "has_key"  # Dict has key

    # Boolean
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


class ConditionEvaluator:
    """
    Evaluates conditions against a context dictionary.

    Usage:
        evaluator = ConditionEvaluator()
        context = {"result": {"confidence": 0.85, "consensus": True}}

        condition = ConditionConfig(
            field="result.confidence",
            operator="gte",
            value=0.8,
        )

        if evaluator.evaluate(condition, context):
            # Hook should fire
    """

    def evaluate(
        self,
        condition: ConditionConfig,
        context: dict[str, Any],
    ) -> bool:
        """
        Evaluate a single condition against context.

        Args:
            condition: The condition to evaluate
            context: The trigger context dictionary

        Returns:
            True if condition passes, False otherwise
        """
        try:
            # Get field value from context
            field_value = self._get_field_value(condition.field, context)

            # Evaluate based on operator
            result = self._evaluate_operator(
                condition.operator,
                field_value,
                condition.value,
            )

            # Apply negation if specified
            if condition.negate:
                result = not result

            return result

        except Exception as e:
            logger.warning(f"Condition evaluation error for {condition.field}: {e}")
            return False

    def evaluate_all(
        self,
        conditions: list[ConditionConfig],
        context: dict[str, Any],
    ) -> bool:
        """
        Evaluate all conditions (logical AND).

        Args:
            conditions: List of conditions to evaluate
            context: The trigger context dictionary

        Returns:
            True if all conditions pass, False otherwise
        """
        if not conditions:
            return True  # No conditions = always pass

        return all(self.evaluate(cond, context) for cond in conditions)

    def _get_field_value(
        self,
        field_path: str,
        context: dict[str, Any],
    ) -> Any:
        """
        Get value from context using dot notation path.

        Args:
            field_path: Dot-separated path (e.g., "result.confidence")
            context: The context dictionary

        Returns:
            The value at the path, or None if not found
        """
        parts = field_path.split(".")
        value: Any = context

        for part in parts:
            if value is None:
                return None

            # Handle dict access
            if isinstance(value, dict):
                value = value.get(part)
            # Handle object attribute access
            elif hasattr(value, part):
                value = getattr(value, part)
            # Handle list index access
            elif isinstance(value, (list, tuple)) and part.isdigit():
                idx = int(part)
                value = value[idx] if 0 <= idx < len(value) else None
            else:
                return None

        return value

    def _evaluate_operator(
        self,
        operator: str,
        field_value: Any,
        condition_value: Any,
    ) -> bool:
        """
        Evaluate a comparison operation.

        Args:
            operator: The operator name
            field_value: Value from context
            condition_value: Value to compare against

        Returns:
            True if comparison succeeds
        """
        op = operator.lower()

        # Equality operators
        if op == Operator.EQ:
            return self._equals(field_value, condition_value)
        elif op == Operator.NE:
            return not self._equals(field_value, condition_value)

        # Numeric comparison
        elif op == Operator.GT:
            return self._compare_numeric(field_value, condition_value) > 0
        elif op == Operator.GTE:
            return self._compare_numeric(field_value, condition_value) >= 0
        elif op == Operator.LT:
            return self._compare_numeric(field_value, condition_value) < 0
        elif op == Operator.LTE:
            return self._compare_numeric(field_value, condition_value) <= 0

        # String/collection contains
        elif op == Operator.CONTAINS:
            return self._contains(field_value, condition_value)
        elif op == Operator.NOT_CONTAINS:
            return not self._contains(field_value, condition_value)
        elif op == Operator.STARTS_WITH:
            return str(field_value).startswith(str(condition_value)) if field_value else False
        elif op == Operator.ENDS_WITH:
            return str(field_value).endswith(str(condition_value)) if field_value else False
        elif op == Operator.MATCHES:
            return bool(re.search(str(condition_value), str(field_value))) if field_value else False

        # Null/empty checks
        elif op == Operator.IS_NULL:
            return field_value is None
        elif op == Operator.IS_NOT_NULL:
            return field_value is not None
        elif op == Operator.IS_EMPTY:
            return self._is_empty(field_value)
        elif op == Operator.IS_NOT_EMPTY:
            return not self._is_empty(field_value)

        # Collection membership
        elif op == Operator.IN:
            if isinstance(condition_value, (list, tuple, set)):
                return field_value in condition_value
            return False
        elif op == Operator.NOT_IN:
            if isinstance(condition_value, (list, tuple, set)):
                return field_value not in condition_value
            return True
        elif op == Operator.HAS_KEY:
            return isinstance(field_value, dict) and condition_value in field_value

        # Boolean
        elif op == Operator.IS_TRUE:
            return bool(field_value) is True
        elif op == Operator.IS_FALSE:
            return bool(field_value) is False

        else:
            logger.warning(f"Unknown operator: {operator}")
            return False

    def _equals(self, a: Any, b: Any) -> bool:
        """Check equality with type coercion."""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False

        # Direct comparison first
        if a == b:
            return True

        # Try numeric comparison
        try:
            return float(a) == float(b)
        except (ValueError, TypeError):
            pass

        # Try string comparison
        return str(a).lower() == str(b).lower()

    def _compare_numeric(self, a: Any, b: Any) -> int:
        """
        Compare two values numerically.

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        try:
            a_num = float(a) if a is not None else 0
            b_num = float(b) if b is not None else 0

            if a_num < b_num:
                return -1
            elif a_num > b_num:
                return 1
            else:
                return 0
        except (ValueError, TypeError):
            # Fall back to string comparison
            a_str = str(a) if a is not None else ""
            b_str = str(b) if b is not None else ""

            if a_str < b_str:
                return -1
            elif a_str > b_str:
                return 1
            else:
                return 0

    def _contains(self, container: Any, item: Any) -> bool:
        """Check if container contains item."""
        if container is None:
            return False

        if isinstance(container, str):
            return str(item) in container

        if isinstance(container, (list, tuple, set)):
            return item in container

        if isinstance(container, dict):
            return item in container.values()

        return False

    def _is_empty(self, value: Any) -> bool:
        """Check if value is empty."""
        if value is None:
            return True
        if isinstance(value, (str, list, tuple, dict, set)):
            return len(value) == 0
        return False
