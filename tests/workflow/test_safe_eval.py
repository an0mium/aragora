"""
Tests for safe_eval module - AST-based expression evaluation.

These tests verify:
1. Basic expression evaluation works correctly
2. Security: dangerous operations are blocked
3. All supported node types work
4. Edge cases are handled
"""

import pytest

from aragora.workflow.safe_eval import SafeEvalError, safe_eval, safe_eval_bool


class TestBasicExpressions:
    """Test basic expression evaluation."""

    def test_numeric_constants(self):
        """Should evaluate numeric constants."""
        assert safe_eval("42") == 42
        assert safe_eval("3.14") == 3.14
        assert safe_eval("-5") == -5

    def test_string_constants(self):
        """Should evaluate string constants."""
        assert safe_eval("'hello'") == "hello"
        assert safe_eval('"world"') == "world"

    def test_boolean_constants(self):
        """Should evaluate boolean constants."""
        assert safe_eval("True") is True
        assert safe_eval("False") is False
        assert safe_eval("None") is None

    def test_variable_access(self):
        """Should access variables from namespace."""
        ns = {"x": 5, "name": "test"}
        assert safe_eval("x", ns) == 5
        assert safe_eval("name", ns) == "test"

    def test_unknown_variable_raises(self):
        """Should raise error for unknown variables."""
        with pytest.raises(SafeEvalError, match="Unknown name"):
            safe_eval("undefined_var")


class TestArithmetic:
    """Test arithmetic operations."""

    def test_addition(self):
        """Should evaluate addition."""
        assert safe_eval("2 + 3") == 5
        assert safe_eval("x + y", {"x": 10, "y": 20}) == 30

    def test_subtraction(self):
        """Should evaluate subtraction."""
        assert safe_eval("10 - 4") == 6

    def test_multiplication(self):
        """Should evaluate multiplication."""
        assert safe_eval("3 * 4") == 12

    def test_division(self):
        """Should evaluate division."""
        assert safe_eval("10 / 4") == 2.5
        assert safe_eval("10 // 3") == 3

    def test_modulo(self):
        """Should evaluate modulo."""
        assert safe_eval("10 % 3") == 1

    def test_power(self):
        """Should evaluate exponentiation."""
        assert safe_eval("2 ** 8") == 256

    def test_unary_operations(self):
        """Should evaluate unary operations."""
        assert safe_eval("-5") == -5
        assert safe_eval("+5") == 5
        assert safe_eval("not True") is False
        assert safe_eval("~0") == -1


class TestComparisons:
    """Test comparison operations."""

    def test_equality(self):
        """Should evaluate equality comparisons."""
        assert safe_eval("5 == 5") is True
        assert safe_eval("5 != 3") is True
        assert safe_eval("'a' == 'a'") is True

    def test_ordering(self):
        """Should evaluate ordering comparisons."""
        assert safe_eval("5 > 3") is True
        assert safe_eval("3 < 5") is True
        assert safe_eval("5 >= 5") is True
        assert safe_eval("5 <= 5") is True

    def test_chained_comparisons(self):
        """Should evaluate chained comparisons."""
        assert safe_eval("1 < 2 < 3") is True
        assert safe_eval("1 < 2 > 3") is False

    def test_membership(self):
        """Should evaluate membership tests."""
        assert safe_eval("3 in [1, 2, 3]") is True
        assert safe_eval("4 not in [1, 2, 3]") is True
        assert safe_eval("'a' in 'abc'") is True

    def test_identity(self):
        """Should evaluate identity tests."""
        assert safe_eval("None is None") is True
        assert safe_eval("None is not False") is True


class TestBooleanLogic:
    """Test boolean logic operations."""

    def test_and(self):
        """Should evaluate 'and' expressions."""
        assert safe_eval("True and True") is True
        assert safe_eval("True and False") is False
        assert safe_eval("1 and 2") == 2

    def test_or(self):
        """Should evaluate 'or' expressions."""
        assert safe_eval("True or False") is True
        assert safe_eval("False or False") is False
        assert safe_eval("0 or 'default'") == "default"

    def test_not(self):
        """Should evaluate 'not' expressions."""
        assert safe_eval("not False") is True
        assert safe_eval("not True") is False

    def test_complex_boolean(self):
        """Should evaluate complex boolean expressions."""
        ns = {"a": True, "b": False, "c": True}
        assert safe_eval("a and not b", ns) is True
        assert safe_eval("(a or b) and c", ns) is True


class TestAttributeAccess:
    """Test attribute access."""

    def test_simple_attribute(self):
        """Should access simple attributes."""

        class Obj:
            value = 42

        assert safe_eval("obj.value", {"obj": Obj()}) == 42

    def test_nested_attribute(self):
        """Should access nested attributes."""

        class Inner:
            data = "nested"

        class Outer:
            inner = Inner()

        assert safe_eval("outer.inner.data", {"outer": Outer()}) == "nested"

    def test_blocked_attribute_raises(self):
        """Should block access to dangerous attributes."""
        with pytest.raises(SafeEvalError, match="not allowed"):
            safe_eval("obj.__class__", {"obj": object()})

        with pytest.raises(SafeEvalError, match="not allowed"):
            safe_eval("obj.__builtins__", {"obj": {}})

        with pytest.raises(SafeEvalError, match="not allowed"):
            safe_eval("obj.__globals__", {"obj": lambda: None})


class TestSubscriptAccess:
    """Test subscript and slice access."""

    def test_list_index(self):
        """Should access list by index."""
        assert safe_eval("items[0]", {"items": [1, 2, 3]}) == 1
        assert safe_eval("items[-1]", {"items": [1, 2, 3]}) == 3

    def test_dict_key(self):
        """Should access dict by key."""
        assert safe_eval("data['key']", {"data": {"key": "value"}}) == "value"

    def test_slice(self):
        """Should evaluate slice expressions."""
        assert safe_eval("items[1:3]", {"items": [1, 2, 3, 4]}) == [2, 3]
        assert safe_eval("items[::2]", {"items": [1, 2, 3, 4]}) == [1, 3]


class TestCollections:
    """Test collection literals."""

    def test_list_literal(self):
        """Should evaluate list literals."""
        assert safe_eval("[1, 2, 3]") == [1, 2, 3]
        assert safe_eval("[x, y]", {"x": 1, "y": 2}) == [1, 2]

    def test_tuple_literal(self):
        """Should evaluate tuple literals."""
        assert safe_eval("(1, 2, 3)") == (1, 2, 3)

    def test_set_literal(self):
        """Should evaluate set literals."""
        assert safe_eval("{1, 2, 3}") == {1, 2, 3}

    def test_dict_literal(self):
        """Should evaluate dict literals."""
        assert safe_eval("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}


class TestBuiltinFunctions:
    """Test allowed builtin functions."""

    def test_len(self):
        """Should evaluate len()."""
        assert safe_eval("len([1, 2, 3])") == 3
        assert safe_eval("len('hello')") == 5

    def test_type_conversions(self):
        """Should evaluate type conversion functions."""
        assert safe_eval("int('42')") == 42
        assert safe_eval("float('3.14')") == 3.14
        assert safe_eval("str(42)") == "42"
        assert safe_eval("bool(1)") is True

    def test_min_max(self):
        """Should evaluate min/max."""
        assert safe_eval("min(3, 1, 2)") == 1
        assert safe_eval("max([1, 2, 3])") == 3

    def test_sum(self):
        """Should evaluate sum()."""
        assert safe_eval("sum([1, 2, 3])") == 6

    def test_all_any(self):
        """Should evaluate all/any."""
        assert safe_eval("all([True, True])") is True
        assert safe_eval("any([False, True])") is True

    def test_sorted(self):
        """Should evaluate sorted()."""
        assert safe_eval("sorted([3, 1, 2])") == [1, 2, 3]

    def test_abs_round(self):
        """Should evaluate abs/round."""
        assert safe_eval("abs(-5)") == 5
        assert safe_eval("round(3.14159, 2)") == 3.14


class TestComprehensions:
    """Test list/set/dict comprehensions."""

    def test_list_comprehension(self):
        """Should evaluate list comprehensions."""
        assert safe_eval("[x * 2 for x in [1, 2, 3]]") == [2, 4, 6]
        assert safe_eval("[x for x in [1, 2, 3] if x > 1]") == [2, 3]

    def test_set_comprehension(self):
        """Should evaluate set comprehensions."""
        result = safe_eval("{x * 2 for x in [1, 2, 2, 3]}")
        assert result == {2, 4, 6}

    def test_dict_comprehension(self):
        """Should evaluate dict comprehensions."""
        assert safe_eval("{x: x * 2 for x in [1, 2, 3]}") == {1: 2, 2: 4, 3: 6}


class TestTernary:
    """Test ternary expressions."""

    def test_ternary_true(self):
        """Should evaluate ternary when condition is true."""
        assert safe_eval("'yes' if True else 'no'") == "yes"

    def test_ternary_false(self):
        """Should evaluate ternary when condition is false."""
        assert safe_eval("'yes' if False else 'no'") == "no"

    def test_ternary_with_variables(self):
        """Should evaluate ternary with variables."""
        ns = {"x": 5}
        assert safe_eval("'big' if x > 3 else 'small'", ns) == "big"


class TestSecurityBlocking:
    """Test that dangerous operations are blocked."""

    def test_import_blocked(self):
        """Should block import statements."""
        # __import__ is not in safe builtins, so it's "Unknown name"
        with pytest.raises(SafeEvalError, match="Unknown name"):
            safe_eval("__import__('os')")

    def test_exec_blocked(self):
        """Should not allow exec-like calls."""
        # exec is not in the safe builtins
        with pytest.raises(SafeEvalError, match="Unknown name"):
            safe_eval("exec('print(1)')")

    def test_lambda_blocked(self):
        """Should block lambda expressions."""
        with pytest.raises(SafeEvalError, match="Unsupported"):
            safe_eval("lambda x: x")

    def test_class_introspection_blocked(self):
        """Should block class introspection attributes."""
        with pytest.raises(SafeEvalError, match="not allowed"):
            safe_eval("''.__class__.__mro__")

    def test_builtins_access_blocked(self):
        """Should block __builtins__ access."""
        with pytest.raises(SafeEvalError, match="not allowed"):
            safe_eval("{}.__class__.__bases__[0].__subclasses__()")

    def test_code_object_blocked(self):
        """Should block __code__ access."""

        def func():
            pass

        with pytest.raises(SafeEvalError, match="not allowed"):
            safe_eval("f.__code__", {"f": func})


class TestSafeEvalBool:
    """Test safe_eval_bool convenience function."""

    def test_bool_conversion(self):
        """Should convert results to boolean."""
        assert safe_eval_bool("5 > 3") is True
        assert safe_eval_bool("5 < 3") is False

    def test_truthy_values(self):
        """Should handle truthy/falsy values."""
        assert safe_eval_bool("1") is True
        assert safe_eval_bool("0") is False
        assert safe_eval_bool("'hello'") is True
        assert safe_eval_bool("''") is False


class TestWorkflowUseCases:
    """Test realistic workflow condition scenarios."""

    def test_step_output_check(self):
        """Should check step outputs like in workflows."""
        ns = {"outputs": {"analysis": {"score": 0.8, "status": "complete"}}}
        assert safe_eval_bool("outputs['analysis']['score'] > 0.5", ns) is True
        assert (
            safe_eval_bool("outputs['analysis']['status'] == 'complete'", ns) is True
        )

    def test_input_validation(self):
        """Should validate workflow inputs."""
        ns = {"inputs": {"document": "text content", "options": ["a", "b"]}}
        assert safe_eval_bool("len(inputs['document']) > 0", ns) is True
        assert safe_eval_bool("'a' in inputs['options']", ns) is True

    def test_state_conditions(self):
        """Should check workflow state."""
        ns = {"state": {"iteration": 3, "converged": False, "errors": []}}
        assert safe_eval_bool("state['iteration'] < 10", ns) is True
        assert safe_eval_bool("not state['converged']", ns) is True
        assert safe_eval_bool("len(state['errors']) == 0", ns) is True

    def test_complex_workflow_condition(self):
        """Should evaluate complex workflow conditions."""
        ns = {
            "step": {"risk_assessment": {"score": 0.85, "flags": ["high_value"]}},
            "inputs": {"threshold": 0.8},
        }
        condition = (
            "step['risk_assessment']['score'] > inputs['threshold'] "
            "or 'high_value' in step['risk_assessment']['flags']"
        )
        assert safe_eval_bool(condition, ns) is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_expression(self):
        """Should handle empty expression."""
        with pytest.raises(SafeEvalError, match="Invalid expression"):
            safe_eval("")

    def test_invalid_syntax(self):
        """Should raise on invalid syntax."""
        with pytest.raises(SafeEvalError, match="Invalid expression"):
            safe_eval("if True:")

    def test_empty_namespace(self):
        """Should work with empty/None namespace."""
        assert safe_eval("5 + 3", None) == 8
        assert safe_eval("5 + 3", {}) == 8

    def test_nested_function_calls(self):
        """Should handle nested function calls."""
        assert safe_eval("len(str(123))") == 3
        assert safe_eval("max(min(5, 10), 3)") == 5

    def test_method_calls(self):
        """Should allow method calls on safe types."""
        assert safe_eval("'hello'.upper()") == "HELLO"
        assert safe_eval("[1, 2, 3].count(2)") == 1
