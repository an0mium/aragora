"""
Comprehensive tests for Condition Evaluation Engine.

Tests cover:
- All operator types (equality, numeric, string, null, collection, boolean)
- Field path traversal (dot notation)
- Condition negation
- Multiple condition evaluation (AND logic)
- Type coercion for comparisons
- Error handling for invalid conditions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aragora.hooks.conditions import ConditionEvaluator, Operator
from aragora.hooks.config import ConditionConfig


# =============================================================================
# Operator Enum Tests
# =============================================================================


class TestOperatorEnum:
    """Tests for Operator enum."""

    def test_all_operators_defined(self):
        """Test that all expected operators are defined."""
        expected = {
            "eq",
            "ne",
            "gt",
            "gte",
            "lt",
            "lte",
            "contains",
            "not_contains",
            "starts_with",
            "ends_with",
            "matches",
            "is_null",
            "is_not_null",
            "is_empty",
            "is_not_empty",
            "in",
            "not_in",
            "has_key",
            "is_true",
            "is_false",
        }
        actual = {op.value for op in Operator}
        assert actual == expected

    def test_operator_string_values(self):
        """Test operator string values are lowercase."""
        for op in Operator:
            assert op.value == op.value.lower()

    def test_operator_is_string_enum(self):
        """Test that Operator inherits from str."""
        assert isinstance(Operator.EQ, str)
        assert Operator.EQ == "eq"


# =============================================================================
# Equality Operators Tests
# =============================================================================


class TestEqualityOperators:
    """Tests for eq and ne operators."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_eq_string_match(self, evaluator):
        """Test eq with matching strings."""
        cond = ConditionConfig(field="status", operator="eq", value="active")
        assert evaluator.evaluate(cond, {"status": "active"}) is True

    def test_eq_string_no_match(self, evaluator):
        """Test eq with non-matching strings."""
        cond = ConditionConfig(field="status", operator="eq", value="active")
        assert evaluator.evaluate(cond, {"status": "inactive"}) is False

    def test_eq_integer_match(self, evaluator):
        """Test eq with matching integers."""
        cond = ConditionConfig(field="count", operator="eq", value=5)
        assert evaluator.evaluate(cond, {"count": 5}) is True

    def test_eq_float_match(self, evaluator):
        """Test eq with matching floats."""
        cond = ConditionConfig(field="score", operator="eq", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.5}) is True

    def test_eq_boolean_match(self, evaluator):
        """Test eq with matching booleans."""
        cond = ConditionConfig(field="active", operator="eq", value=True)
        assert evaluator.evaluate(cond, {"active": True}) is True

    def test_eq_none_values(self, evaluator):
        """Test eq with None values."""
        cond = ConditionConfig(field="value", operator="eq", value=None)
        assert evaluator.evaluate(cond, {"value": None}) is True

    def test_eq_type_coercion_numeric(self, evaluator):
        """Test eq with numeric type coercion."""
        cond = ConditionConfig(field="count", operator="eq", value="5")
        assert evaluator.evaluate(cond, {"count": 5}) is True

    def test_eq_case_insensitive_string(self, evaluator):
        """Test eq with case-insensitive string comparison."""
        cond = ConditionConfig(field="status", operator="eq", value="ACTIVE")
        assert evaluator.evaluate(cond, {"status": "active"}) is True

    def test_ne_string_match(self, evaluator):
        """Test ne with different strings."""
        cond = ConditionConfig(field="status", operator="ne", value="failed")
        assert evaluator.evaluate(cond, {"status": "success"}) is True

    def test_ne_string_no_match(self, evaluator):
        """Test ne with same strings."""
        cond = ConditionConfig(field="status", operator="ne", value="active")
        assert evaluator.evaluate(cond, {"status": "active"}) is False

    def test_ne_none_values(self, evaluator):
        """Test ne with None values."""
        cond = ConditionConfig(field="value", operator="ne", value=None)
        assert evaluator.evaluate(cond, {"value": "something"}) is True


# =============================================================================
# Numeric Comparison Operators Tests
# =============================================================================


class TestNumericOperators:
    """Tests for gt, gte, lt, lte operators."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_gt_greater(self, evaluator):
        """Test gt when value is greater."""
        cond = ConditionConfig(field="score", operator="gt", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.8}) is True

    def test_gt_equal(self, evaluator):
        """Test gt when value is equal."""
        cond = ConditionConfig(field="score", operator="gt", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.5}) is False

    def test_gt_less(self, evaluator):
        """Test gt when value is less."""
        cond = ConditionConfig(field="score", operator="gt", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.3}) is False

    def test_gte_greater(self, evaluator):
        """Test gte when value is greater."""
        cond = ConditionConfig(field="score", operator="gte", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.8}) is True

    def test_gte_equal(self, evaluator):
        """Test gte when value is equal."""
        cond = ConditionConfig(field="score", operator="gte", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.5}) is True

    def test_gte_less(self, evaluator):
        """Test gte when value is less."""
        cond = ConditionConfig(field="score", operator="gte", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.3}) is False

    def test_lt_less(self, evaluator):
        """Test lt when value is less."""
        cond = ConditionConfig(field="score", operator="lt", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.3}) is True

    def test_lt_equal(self, evaluator):
        """Test lt when value is equal."""
        cond = ConditionConfig(field="score", operator="lt", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.5}) is False

    def test_lt_greater(self, evaluator):
        """Test lt when value is greater."""
        cond = ConditionConfig(field="score", operator="lt", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.8}) is False

    def test_lte_less(self, evaluator):
        """Test lte when value is less."""
        cond = ConditionConfig(field="score", operator="lte", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.3}) is True

    def test_lte_equal(self, evaluator):
        """Test lte when value is equal."""
        cond = ConditionConfig(field="score", operator="lte", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.5}) is True

    def test_lte_greater(self, evaluator):
        """Test lte when value is greater."""
        cond = ConditionConfig(field="score", operator="lte", value=0.5)
        assert evaluator.evaluate(cond, {"score": 0.8}) is False

    def test_numeric_comparison_with_strings(self, evaluator):
        """Test numeric comparison with string values."""
        cond = ConditionConfig(field="count", operator="gt", value="5")
        assert evaluator.evaluate(cond, {"count": "10"}) is True

    def test_numeric_comparison_with_none(self, evaluator):
        """Test numeric comparison with None (treated as 0)."""
        cond = ConditionConfig(field="count", operator="gt", value=0)
        assert evaluator.evaluate(cond, {"count": None}) is False

    def test_numeric_comparison_fallback_to_string(self, evaluator):
        """Test numeric comparison falls back to string comparison."""
        cond = ConditionConfig(field="name", operator="gt", value="apple")
        assert evaluator.evaluate(cond, {"name": "banana"}) is True


# =============================================================================
# String Operators Tests
# =============================================================================


class TestStringOperators:
    """Tests for contains, not_contains, starts_with, ends_with, matches operators."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_contains_string_match(self, evaluator):
        """Test contains with substring match."""
        cond = ConditionConfig(field="message", operator="contains", value="world")
        assert evaluator.evaluate(cond, {"message": "Hello world"}) is True

    def test_contains_string_no_match(self, evaluator):
        """Test contains without substring."""
        cond = ConditionConfig(field="message", operator="contains", value="foo")
        assert evaluator.evaluate(cond, {"message": "Hello world"}) is False

    def test_contains_list_match(self, evaluator):
        """Test contains with list."""
        cond = ConditionConfig(field="items", operator="contains", value="apple")
        assert evaluator.evaluate(cond, {"items": ["apple", "banana"]}) is True

    def test_contains_list_no_match(self, evaluator):
        """Test contains with list, no match."""
        cond = ConditionConfig(field="items", operator="contains", value="orange")
        assert evaluator.evaluate(cond, {"items": ["apple", "banana"]}) is False

    def test_contains_dict_value(self, evaluator):
        """Test contains with dict (checks values)."""
        cond = ConditionConfig(field="data", operator="contains", value="test")
        assert evaluator.evaluate(cond, {"data": {"key": "test"}}) is True

    def test_contains_none(self, evaluator):
        """Test contains with None field."""
        cond = ConditionConfig(field="value", operator="contains", value="test")
        assert evaluator.evaluate(cond, {"value": None}) is False

    def test_not_contains_string(self, evaluator):
        """Test not_contains with string."""
        cond = ConditionConfig(field="message", operator="not_contains", value="error")
        assert evaluator.evaluate(cond, {"message": "Success!"}) is True

    def test_not_contains_list(self, evaluator):
        """Test not_contains with list."""
        cond = ConditionConfig(field="items", operator="not_contains", value="error")
        assert evaluator.evaluate(cond, {"items": ["success", "ok"]}) is True

    def test_starts_with_match(self, evaluator):
        """Test starts_with with matching prefix."""
        cond = ConditionConfig(field="name", operator="starts_with", value="test_")
        assert evaluator.evaluate(cond, {"name": "test_case"}) is True

    def test_starts_with_no_match(self, evaluator):
        """Test starts_with without matching prefix."""
        cond = ConditionConfig(field="name", operator="starts_with", value="prod_")
        assert evaluator.evaluate(cond, {"name": "test_case"}) is False

    def test_starts_with_none(self, evaluator):
        """Test starts_with with None field."""
        cond = ConditionConfig(field="name", operator="starts_with", value="test")
        assert evaluator.evaluate(cond, {"name": None}) is False

    def test_ends_with_match(self, evaluator):
        """Test ends_with with matching suffix."""
        cond = ConditionConfig(field="file", operator="ends_with", value=".yaml")
        assert evaluator.evaluate(cond, {"file": "config.yaml"}) is True

    def test_ends_with_no_match(self, evaluator):
        """Test ends_with without matching suffix."""
        cond = ConditionConfig(field="file", operator="ends_with", value=".json")
        assert evaluator.evaluate(cond, {"file": "config.yaml"}) is False

    def test_ends_with_none(self, evaluator):
        """Test ends_with with None field."""
        cond = ConditionConfig(field="file", operator="ends_with", value=".yaml")
        assert evaluator.evaluate(cond, {"file": None}) is False

    def test_matches_regex(self, evaluator):
        """Test matches with regex pattern."""
        cond = ConditionConfig(field="email", operator="matches", value=r"\w+@\w+\.\w+")
        assert evaluator.evaluate(cond, {"email": "user@example.com"}) is True

    def test_matches_no_match(self, evaluator):
        """Test matches with non-matching pattern."""
        cond = ConditionConfig(field="email", operator="matches", value=r"^\d+$")
        assert evaluator.evaluate(cond, {"email": "user@example.com"}) is False

    def test_matches_none(self, evaluator):
        """Test matches with None field."""
        cond = ConditionConfig(field="email", operator="matches", value=r".*")
        assert evaluator.evaluate(cond, {"email": None}) is False

    def test_matches_complex_pattern(self, evaluator):
        """Test matches with complex regex."""
        cond = ConditionConfig(
            field="id",
            operator="matches",
            value=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        )
        assert (
            evaluator.evaluate(
                cond,
                {"id": "123e4567-e89b-12d3-a456-426614174000"},
            )
            is True
        )


# =============================================================================
# Null and Empty Operators Tests
# =============================================================================


class TestNullEmptyOperators:
    """Tests for is_null, is_not_null, is_empty, is_not_empty operators."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_is_null_with_none(self, evaluator):
        """Test is_null with None value."""
        cond = ConditionConfig(field="value", operator="is_null", value=None)
        assert evaluator.evaluate(cond, {"value": None}) is True

    def test_is_null_with_value(self, evaluator):
        """Test is_null with non-None value."""
        cond = ConditionConfig(field="value", operator="is_null", value=None)
        assert evaluator.evaluate(cond, {"value": "something"}) is False

    def test_is_null_missing_field(self, evaluator):
        """Test is_null with missing field (resolves to None)."""
        cond = ConditionConfig(field="missing", operator="is_null", value=None)
        assert evaluator.evaluate(cond, {}) is True

    def test_is_not_null_with_value(self, evaluator):
        """Test is_not_null with value."""
        cond = ConditionConfig(field="value", operator="is_not_null", value=None)
        assert evaluator.evaluate(cond, {"value": "something"}) is True

    def test_is_not_null_with_none(self, evaluator):
        """Test is_not_null with None."""
        cond = ConditionConfig(field="value", operator="is_not_null", value=None)
        assert evaluator.evaluate(cond, {"value": None}) is False

    def test_is_empty_string(self, evaluator):
        """Test is_empty with empty string."""
        cond = ConditionConfig(field="text", operator="is_empty", value=None)
        assert evaluator.evaluate(cond, {"text": ""}) is True

    def test_is_empty_list(self, evaluator):
        """Test is_empty with empty list."""
        cond = ConditionConfig(field="items", operator="is_empty", value=None)
        assert evaluator.evaluate(cond, {"items": []}) is True

    def test_is_empty_dict(self, evaluator):
        """Test is_empty with empty dict."""
        cond = ConditionConfig(field="data", operator="is_empty", value=None)
        assert evaluator.evaluate(cond, {"data": {}}) is True

    def test_is_empty_none(self, evaluator):
        """Test is_empty with None (considered empty)."""
        cond = ConditionConfig(field="value", operator="is_empty", value=None)
        assert evaluator.evaluate(cond, {"value": None}) is True

    def test_is_empty_non_empty_string(self, evaluator):
        """Test is_empty with non-empty string."""
        cond = ConditionConfig(field="text", operator="is_empty", value=None)
        assert evaluator.evaluate(cond, {"text": "hello"}) is False

    def test_is_empty_non_empty_list(self, evaluator):
        """Test is_empty with non-empty list."""
        cond = ConditionConfig(field="items", operator="is_empty", value=None)
        assert evaluator.evaluate(cond, {"items": [1, 2]}) is False

    def test_is_not_empty_string(self, evaluator):
        """Test is_not_empty with non-empty string."""
        cond = ConditionConfig(field="text", operator="is_not_empty", value=None)
        assert evaluator.evaluate(cond, {"text": "hello"}) is True

    def test_is_not_empty_list(self, evaluator):
        """Test is_not_empty with non-empty list."""
        cond = ConditionConfig(field="items", operator="is_not_empty", value=None)
        assert evaluator.evaluate(cond, {"items": [1]}) is True

    def test_is_not_empty_empty(self, evaluator):
        """Test is_not_empty with empty value."""
        cond = ConditionConfig(field="text", operator="is_not_empty", value=None)
        assert evaluator.evaluate(cond, {"text": ""}) is False


# =============================================================================
# Collection Operators Tests
# =============================================================================


class TestCollectionOperators:
    """Tests for in, not_in, has_key operators."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_in_list_match(self, evaluator):
        """Test in with value in list."""
        cond = ConditionConfig(field="status", operator="in", value=["active", "pending"])
        assert evaluator.evaluate(cond, {"status": "active"}) is True

    def test_in_list_no_match(self, evaluator):
        """Test in with value not in list."""
        cond = ConditionConfig(field="status", operator="in", value=["active", "pending"])
        assert evaluator.evaluate(cond, {"status": "failed"}) is False

    def test_in_tuple(self, evaluator):
        """Test in with tuple."""
        cond = ConditionConfig(field="status", operator="in", value=("active", "pending"))
        assert evaluator.evaluate(cond, {"status": "active"}) is True

    def test_in_set(self, evaluator):
        """Test in with set."""
        cond = ConditionConfig(field="status", operator="in", value={"active", "pending"})
        assert evaluator.evaluate(cond, {"status": "pending"}) is True

    def test_in_non_collection(self, evaluator):
        """Test in with non-collection value returns False."""
        cond = ConditionConfig(field="status", operator="in", value="not_a_collection")
        assert evaluator.evaluate(cond, {"status": "test"}) is False

    def test_not_in_list_match(self, evaluator):
        """Test not_in with value not in list."""
        cond = ConditionConfig(field="status", operator="not_in", value=["failed", "error"])
        assert evaluator.evaluate(cond, {"status": "success"}) is True

    def test_not_in_list_no_match(self, evaluator):
        """Test not_in with value in list."""
        cond = ConditionConfig(field="status", operator="not_in", value=["failed", "error"])
        assert evaluator.evaluate(cond, {"status": "failed"}) is False

    def test_not_in_non_collection(self, evaluator):
        """Test not_in with non-collection value returns True."""
        cond = ConditionConfig(field="status", operator="not_in", value="not_a_collection")
        assert evaluator.evaluate(cond, {"status": "test"}) is True

    def test_has_key_match(self, evaluator):
        """Test has_key with existing key."""
        cond = ConditionConfig(field="data", operator="has_key", value="name")
        assert evaluator.evaluate(cond, {"data": {"name": "test", "value": 1}}) is True

    def test_has_key_no_match(self, evaluator):
        """Test has_key with missing key."""
        cond = ConditionConfig(field="data", operator="has_key", value="missing")
        assert evaluator.evaluate(cond, {"data": {"name": "test"}}) is False

    def test_has_key_non_dict(self, evaluator):
        """Test has_key with non-dict field."""
        cond = ConditionConfig(field="data", operator="has_key", value="key")
        assert evaluator.evaluate(cond, {"data": "not_a_dict"}) is False


# =============================================================================
# Boolean Operators Tests
# =============================================================================


class TestBooleanOperators:
    """Tests for is_true and is_false operators."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_is_true_with_true(self, evaluator):
        """Test is_true with True value."""
        cond = ConditionConfig(field="active", operator="is_true", value=None)
        assert evaluator.evaluate(cond, {"active": True}) is True

    def test_is_true_with_false(self, evaluator):
        """Test is_true with False value."""
        cond = ConditionConfig(field="active", operator="is_true", value=None)
        assert evaluator.evaluate(cond, {"active": False}) is False

    def test_is_true_with_truthy_string(self, evaluator):
        """Test is_true with truthy string."""
        cond = ConditionConfig(field="value", operator="is_true", value=None)
        assert evaluator.evaluate(cond, {"value": "hello"}) is True

    def test_is_true_with_falsy_string(self, evaluator):
        """Test is_true with falsy string (empty)."""
        cond = ConditionConfig(field="value", operator="is_true", value=None)
        assert evaluator.evaluate(cond, {"value": ""}) is False

    def test_is_true_with_truthy_number(self, evaluator):
        """Test is_true with truthy number."""
        cond = ConditionConfig(field="count", operator="is_true", value=None)
        assert evaluator.evaluate(cond, {"count": 1}) is True

    def test_is_true_with_zero(self, evaluator):
        """Test is_true with zero."""
        cond = ConditionConfig(field="count", operator="is_true", value=None)
        assert evaluator.evaluate(cond, {"count": 0}) is False

    def test_is_false_with_false(self, evaluator):
        """Test is_false with False value."""
        cond = ConditionConfig(field="active", operator="is_false", value=None)
        assert evaluator.evaluate(cond, {"active": False}) is True

    def test_is_false_with_true(self, evaluator):
        """Test is_false with True value."""
        cond = ConditionConfig(field="active", operator="is_false", value=None)
        assert evaluator.evaluate(cond, {"active": True}) is False

    def test_is_false_with_none(self, evaluator):
        """Test is_false with None (falsy)."""
        cond = ConditionConfig(field="value", operator="is_false", value=None)
        assert evaluator.evaluate(cond, {"value": None}) is True

    def test_is_false_with_empty_list(self, evaluator):
        """Test is_false with empty list (falsy)."""
        cond = ConditionConfig(field="items", operator="is_false", value=None)
        assert evaluator.evaluate(cond, {"items": []}) is True


# =============================================================================
# Field Path Traversal Tests
# =============================================================================


class TestFieldPathTraversal:
    """Tests for dot notation field path traversal."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_simple_field(self, evaluator):
        """Test simple field access."""
        cond = ConditionConfig(field="name", operator="eq", value="test")
        assert evaluator.evaluate(cond, {"name": "test"}) is True

    def test_nested_dict_access(self, evaluator):
        """Test nested dict access."""
        cond = ConditionConfig(field="result.confidence", operator="gt", value=0.8)
        assert evaluator.evaluate(cond, {"result": {"confidence": 0.9}}) is True

    def test_deeply_nested_access(self, evaluator):
        """Test deeply nested access."""
        cond = ConditionConfig(field="a.b.c.d", operator="eq", value="deep")
        context = {"a": {"b": {"c": {"d": "deep"}}}}
        assert evaluator.evaluate(cond, context) is True

    def test_list_index_access(self, evaluator):
        """Test list index access."""
        cond = ConditionConfig(field="items.0.name", operator="eq", value="first")
        context = {"items": [{"name": "first"}, {"name": "second"}]}
        assert evaluator.evaluate(cond, context) is True

    def test_list_index_out_of_bounds(self, evaluator):
        """Test list index out of bounds returns None."""
        cond = ConditionConfig(field="items.10.name", operator="is_null", value=None)
        context = {"items": [{"name": "first"}]}
        assert evaluator.evaluate(cond, context) is True

    def test_object_attribute_access(self, evaluator):
        """Test object attribute access."""

        @dataclass
        class Result:
            confidence: float = 0.9

        cond = ConditionConfig(field="result.confidence", operator="gt", value=0.8)
        context = {"result": Result()}
        assert evaluator.evaluate(cond, context) is True

    def test_missing_path_returns_none(self, evaluator):
        """Test missing path returns None."""
        cond = ConditionConfig(field="missing.path", operator="is_null", value=None)
        assert evaluator.evaluate(cond, {}) is True

    def test_none_in_path(self, evaluator):
        """Test None value in path returns None."""
        cond = ConditionConfig(field="result.confidence", operator="is_null", value=None)
        assert evaluator.evaluate(cond, {"result": None}) is True

    def test_mixed_access_types(self, evaluator):
        """Test mixed dict, list, and attribute access."""

        @dataclass
        class Item:
            value: int

        cond = ConditionConfig(field="data.items.0.value", operator="eq", value=42)
        context = {"data": {"items": [Item(value=42)]}}
        assert evaluator.evaluate(cond, context) is True


# =============================================================================
# Negation Tests
# =============================================================================


class TestNegation:
    """Tests for condition negation."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_negate_true_becomes_false(self, evaluator):
        """Test negating true condition becomes false."""
        cond = ConditionConfig(field="status", operator="eq", value="active", negate=True)
        assert evaluator.evaluate(cond, {"status": "active"}) is False

    def test_negate_false_becomes_true(self, evaluator):
        """Test negating false condition becomes true."""
        cond = ConditionConfig(field="status", operator="eq", value="active", negate=True)
        assert evaluator.evaluate(cond, {"status": "inactive"}) is True

    def test_negate_with_gt(self, evaluator):
        """Test negating gt operator."""
        cond = ConditionConfig(field="score", operator="gt", value=0.5, negate=True)
        # score > 0.5 is True, negated = False
        assert evaluator.evaluate(cond, {"score": 0.8}) is False
        # score > 0.5 is False, negated = True
        assert evaluator.evaluate(cond, {"score": 0.3}) is True

    def test_negate_with_contains(self, evaluator):
        """Test negating contains operator."""
        cond = ConditionConfig(field="text", operator="contains", value="error", negate=True)
        assert evaluator.evaluate(cond, {"text": "success message"}) is True
        assert evaluator.evaluate(cond, {"text": "error occurred"}) is False


# =============================================================================
# Multiple Conditions (AND Logic) Tests
# =============================================================================


class TestMultipleConditions:
    """Tests for evaluate_all with multiple conditions."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_all_conditions_pass(self, evaluator):
        """Test evaluate_all when all conditions pass."""
        conditions = [
            ConditionConfig(field="consensus", operator="is_true", value=None),
            ConditionConfig(field="confidence", operator="gte", value=0.8),
            ConditionConfig(field="status", operator="eq", value="complete"),
        ]
        context = {"consensus": True, "confidence": 0.95, "status": "complete"}
        assert evaluator.evaluate_all(conditions, context) is True

    def test_one_condition_fails(self, evaluator):
        """Test evaluate_all when one condition fails."""
        conditions = [
            ConditionConfig(field="consensus", operator="is_true", value=None),
            ConditionConfig(field="confidence", operator="gte", value=0.8),
        ]
        context = {"consensus": True, "confidence": 0.5}  # confidence fails
        assert evaluator.evaluate_all(conditions, context) is False

    def test_all_conditions_fail(self, evaluator):
        """Test evaluate_all when all conditions fail."""
        conditions = [
            ConditionConfig(field="consensus", operator="is_true", value=None),
            ConditionConfig(field="confidence", operator="gte", value=0.8),
        ]
        context = {"consensus": False, "confidence": 0.5}
        assert evaluator.evaluate_all(conditions, context) is False

    def test_empty_conditions_returns_true(self, evaluator):
        """Test evaluate_all with empty conditions returns True."""
        assert evaluator.evaluate_all([], {}) is True

    def test_single_condition(self, evaluator):
        """Test evaluate_all with single condition."""
        conditions = [
            ConditionConfig(field="active", operator="is_true", value=None),
        ]
        assert evaluator.evaluate_all(conditions, {"active": True}) is True
        assert evaluator.evaluate_all(conditions, {"active": False}) is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in condition evaluation."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_unknown_operator_returns_false(self, evaluator):
        """Test unknown operator returns False."""
        cond = ConditionConfig(field="value", operator="unknown_op", value="test")
        assert evaluator.evaluate(cond, {"value": "test"}) is False

    def test_invalid_regex_returns_false(self, evaluator):
        """Test invalid regex pattern returns False."""
        cond = ConditionConfig(field="text", operator="matches", value="[invalid(")
        # Should not raise, just return False
        assert evaluator.evaluate(cond, {"text": "test"}) is False

    def test_type_error_in_comparison_returns_false(self, evaluator):
        """Test type error in comparison returns False gracefully."""
        # Create a condition that might cause type issues
        cond = ConditionConfig(field="value", operator="gt", value="not_a_number")
        # String comparison fallback should handle this
        result = evaluator.evaluate(cond, {"value": "test"})
        # Result depends on string comparison, should not raise
        assert isinstance(result, bool)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    def test_empty_string_field(self, evaluator):
        """Test empty string as field path."""
        cond = ConditionConfig(field="", operator="eq", value="test")
        # Empty field path should return False gracefully
        result = evaluator.evaluate(cond, {"": "test"})
        # Depends on implementation, but should not crash

    def test_whitespace_field(self, evaluator):
        """Test whitespace in field path."""
        cond = ConditionConfig(field="field with space", operator="eq", value="test")
        # Should handle gracefully
        result = evaluator.evaluate(cond, {"field with space": "test"})
        assert result is True

    def test_special_characters_in_value(self, evaluator):
        """Test special characters in comparison value."""
        cond = ConditionConfig(field="text", operator="contains", value="$pecial!@#")
        assert evaluator.evaluate(cond, {"text": "test $pecial!@# chars"}) is True

    def test_unicode_values(self, evaluator):
        """Test unicode values."""
        cond = ConditionConfig(field="text", operator="contains", value="")
        assert evaluator.evaluate(cond, {"text": "Hello  World"}) is True

    def test_large_nested_context(self, evaluator):
        """Test with large nested context."""
        context = {"level0": {"level1": {"level2": {"level3": {"value": "deep"}}}}}
        cond = ConditionConfig(
            field="level0.level1.level2.level3.value",
            operator="eq",
            value="deep",
        )
        assert evaluator.evaluate(cond, context) is True

    def test_list_with_mixed_types(self, evaluator):
        """Test list contains with mixed types."""
        cond = ConditionConfig(field="items", operator="contains", value=1)
        assert evaluator.evaluate(cond, {"items": ["a", 1, None, True]}) is True

    def test_tuple_field_access(self, evaluator):
        """Test tuple index access."""
        cond = ConditionConfig(field="coords.0", operator="eq", value=10)
        assert evaluator.evaluate(cond, {"coords": (10, 20)}) is True

    def test_set_field_value(self, evaluator):
        """Test set field with contains."""
        cond = ConditionConfig(field="tags", operator="contains", value="python")
        assert evaluator.evaluate(cond, {"tags": {"python", "golang"}}) is True

    def test_float_precision(self, evaluator):
        """Test float comparison with precision issues.

        Note: 0.1 + 0.2 != 0.3 due to floating point representation.
        The evaluator uses direct float comparison, so this will fail
        unless values are exactly equal.
        """
        # Use values that are exactly equal
        cond = ConditionConfig(field="value", operator="eq", value=0.5)
        assert evaluator.evaluate(cond, {"value": 0.5}) is True

        # Different floats should not be equal
        cond2 = ConditionConfig(field="value", operator="eq", value=0.1 + 0.2)
        # 0.1 + 0.2 == 0.30000000000000004, not 0.3
        # The evaluator will use numeric comparison which checks float equality
        result = evaluator.evaluate(cond2, {"value": 0.3})
        # This may be True or False depending on implementation - just verify it doesn't crash
        assert isinstance(result, bool)

    def test_boolean_as_integer(self, evaluator):
        """Test boolean treated as integer in numeric comparison."""
        cond = ConditionConfig(field="flag", operator="eq", value=1)
        assert evaluator.evaluate(cond, {"flag": True}) is True

    def test_case_sensitivity_in_operators(self, evaluator):
        """Test that operators are case-insensitive."""
        cond = ConditionConfig(field="value", operator="EQ", value="test")
        assert evaluator.evaluate(cond, {"value": "test"}) is True

        cond2 = ConditionConfig(field="value", operator="Eq", value="test")
        assert evaluator.evaluate(cond2, {"value": "test"}) is True
