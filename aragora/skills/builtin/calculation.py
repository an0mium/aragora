"""
Calculation Skill.

Provides safe mathematical calculation capabilities.
Uses AST-based evaluation to prevent code injection while supporting
complex mathematical expressions.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Callable

from ..base import (
    SyncSkill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


# Type aliases for operators
BinaryOpFunc = Callable[[Any, Any], Any]
UnaryOpFunc = Callable[[Any], Any]

# Safe binary operators for evaluation
SAFE_BINARY_OPERATORS: dict[type[ast.operator], BinaryOpFunc] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Safe unary operators
SAFE_UNARY_OPERATORS: dict[type[ast.unaryop], UnaryOpFunc] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions
SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    # Math module functions
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
}

# Safe constants
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


class SafeEvaluator(ast.NodeVisitor):
    """
    AST-based safe expression evaluator.

    Only allows mathematical operations, preventing code injection.
    """

    def __init__(self, variables: dict[str, Any] | None = None):
        self.variables = variables or {}

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Num(self, node: ast.Num) -> Any:
        # For Python < 3.8 compatibility
        return node.n

    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id
        # Check variables first
        if name in self.variables:
            return self.variables[name]
        # Check constants
        if name in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[name]
        raise ValueError(f"Unknown variable: {name}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)

        if op_type not in SAFE_BINARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        op = SAFE_BINARY_OPERATORS[op_type]

        # Prevent dangerous operations
        if op_type == ast.Pow:
            # Limit exponent to prevent memory issues
            if isinstance(right, (int, float)) and abs(right) > 1000:
                raise ValueError("Exponent too large (max 1000)")

        return op(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        op_type = type(node.op)

        if op_type not in SAFE_UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        return SAFE_UNARY_OPERATORS[op_type](operand)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")

        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Unknown function: {func_name}")

        func = SAFE_FUNCTIONS[func_name]
        args = [self.visit(arg) for arg in node.args]

        return func(*args)

    def visit_List(self, node: ast.List) -> list:
        return [self.visit(elem) for elem in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        return tuple(self.visit(elem) for elem in node.elts)

    def visit_Compare(self, node: ast.Compare) -> bool:
        """Handle comparison operations."""
        left = self.visit(node.left)

        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)

            if isinstance(op, ast.Eq):
                result = left == right
            elif isinstance(op, ast.NotEq):
                result = left != right
            elif isinstance(op, ast.Lt):
                result = left < right
            elif isinstance(op, ast.LtE):
                result = left <= right
            elif isinstance(op, ast.Gt):
                result = left > right
            elif isinstance(op, ast.GtE):
                result = left >= right
            else:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")

            if not result:
                return False
            left = right

        return True

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Handle ternary expressions (a if condition else b)."""
        condition = self.visit(node.test)
        if condition:
            return self.visit(node.body)
        return self.visit(node.orelse)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def safe_eval(expression: str, variables: dict[str, Any] | None = None) -> Any:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression string
        variables: Optional variable bindings

    Returns:
        Result of the calculation

    Raises:
        ValueError: If expression contains unsafe operations
        SyntaxError: If expression has invalid syntax
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise SyntaxError(f"Invalid expression: {e}") from e

    evaluator = SafeEvaluator(variables)
    return evaluator.visit(tree)


class CalculationSkill(SyncSkill):
    """
    Skill for performing mathematical calculations.

    Supports:
    - Basic arithmetic (+, -, *, /, //, %, **)
    - Mathematical functions (sqrt, sin, cos, log, etc.)
    - Constants (pi, e, tau)
    - Variables
    - Comparisons and conditionals
    - Unit conversions (common units)
    """

    def __init__(self, precision: int = 10):
        """
        Initialize calculation skill.

        Args:
            precision: Number of decimal places for results
        """
        self._precision = precision

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="calculation",
            version="1.0.0",
            description="Perform safe mathematical calculations",
            capabilities=[],  # No special capabilities needed
            input_schema={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                    "required": True,
                },
                "variables": {
                    "type": "object",
                    "description": "Variable bindings for the expression",
                },
                "precision": {
                    "type": "number",
                    "description": "Decimal precision for result",
                    "default": 10,
                },
                "unit_conversion": {
                    "type": "object",
                    "description": "Unit conversion: {from: unit, to: unit, value: number}",
                },
            },
            tags=["math", "calculation", "compute"],
            debate_compatible=True,
            max_execution_time_seconds=5.0,  # Calculations should be fast
        )

    def execute_sync(
        self,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute calculation."""
        # Handle unit conversion if requested
        if "unit_conversion" in input_data:
            return self._handle_unit_conversion(input_data["unit_conversion"])

        expression = input_data.get("expression", "")
        if not expression:
            return SkillResult.create_failure(
                "Expression is required",
                error_code="missing_expression",
            )

        variables = input_data.get("variables", {})
        precision = input_data.get("precision", self._precision)

        try:
            # Validate variables are numeric
            for name, value in variables.items():
                if not isinstance(value, (int, float, complex)):
                    return SkillResult.create_failure(
                        f"Variable '{name}' must be numeric",
                        error_code="invalid_variable",
                    )

            result = safe_eval(expression, variables)

            # Format result
            if isinstance(result, float):
                formatted = round(result, precision)
            elif isinstance(result, complex):
                formatted = complex(round(result.real, precision), round(result.imag, precision))
            else:
                formatted = result

            return SkillResult.create_success(
                {
                    "expression": expression,
                    "result": formatted,
                    "result_type": type(result).__name__,
                    "variables_used": list(variables.keys()) if variables else [],
                }
            )

        except SyntaxError as e:
            return SkillResult.create_failure(
                f"Invalid expression syntax: {e}",
                error_code="syntax_error",
            )
        except ValueError as e:
            return SkillResult.create_failure(
                f"Calculation error: {e}",
                error_code="value_error",
            )
        except ZeroDivisionError:
            return SkillResult.create_failure(
                "Division by zero",
                error_code="division_by_zero",
            )
        except OverflowError:
            return SkillResult.create_failure(
                "Result too large (overflow)",
                error_code="overflow",
            )
        except Exception as e:
            logger.exception(f"Calculation failed: {e}")
            return SkillResult.create_failure(f"Calculation failed: {e}")

    def _handle_unit_conversion(self, conversion: dict[str, Any]) -> SkillResult:
        """Handle unit conversion requests."""
        from_unit = conversion.get("from", "").lower()
        to_unit = conversion.get("to", "").lower()
        value = conversion.get("value")

        if value is None:
            return SkillResult.create_failure(
                "Value is required for unit conversion",
                error_code="missing_value",
            )

        # Unit conversion factors (to base unit)
        conversions: dict[str, dict[str, int | float]] = {
            # Length (base: meters)
            "length": {
                "m": 1,
                "meter": 1,
                "meters": 1,
                "km": 1000,
                "kilometer": 1000,
                "kilometers": 1000,
                "cm": 0.01,
                "centimeter": 0.01,
                "centimeters": 0.01,
                "mm": 0.001,
                "millimeter": 0.001,
                "millimeters": 0.001,
                "mi": 1609.344,
                "mile": 1609.344,
                "miles": 1609.344,
                "ft": 0.3048,
                "foot": 0.3048,
                "feet": 0.3048,
                "in": 0.0254,
                "inch": 0.0254,
                "inches": 0.0254,
                "yd": 0.9144,
                "yard": 0.9144,
                "yards": 0.9144,
            },
            # Weight (base: kilograms)
            "weight": {
                "kg": 1,
                "kilogram": 1,
                "kilograms": 1,
                "g": 0.001,
                "gram": 0.001,
                "grams": 0.001,
                "mg": 0.000001,
                "milligram": 0.000001,
                "milligrams": 0.000001,
                "lb": 0.453592,
                "pound": 0.453592,
                "pounds": 0.453592,
                "oz": 0.0283495,
                "ounce": 0.0283495,
                "ounces": 0.0283495,
                "ton": 1000,
                "tons": 1000,
            },
            # Time (base: seconds)
            "time": {
                "s": 1,
                "sec": 1,
                "second": 1,
                "seconds": 1,
                "ms": 0.001,
                "millisecond": 0.001,
                "milliseconds": 0.001,
                "min": 60,
                "minute": 60,
                "minutes": 60,
                "h": 3600,
                "hr": 3600,
                "hour": 3600,
                "hours": 3600,
                "day": 86400,
                "days": 86400,
                "week": 604800,
                "weeks": 604800,
            },
            # Data (base: bytes)
            "data": {
                "b": 1,
                "byte": 1,
                "bytes": 1,
                "kb": 1024,
                "kilobyte": 1024,
                "kilobytes": 1024,
                "mb": 1024**2,
                "megabyte": 1024**2,
                "megabytes": 1024**2,
                "gb": 1024**3,
                "gigabyte": 1024**3,
                "gigabytes": 1024**3,
                "tb": 1024**4,
                "terabyte": 1024**4,
                "terabytes": 1024**4,
            },
        }

        # Temperature units (handled specially)
        temperature_units: set[str] = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}

        # Find which category the units belong to
        category: str | None = None

        # Check temperature first
        if from_unit in temperature_units and to_unit in temperature_units:
            category = "temperature"
        else:
            for cat, units in conversions.items():
                if from_unit in units and to_unit in units:
                    category = cat
                    break

        if not category:
            return SkillResult.create_failure(
                f"Cannot convert between '{from_unit}' and '{to_unit}'",
                error_code="incompatible_units",
            )

        # Handle temperature specially
        if category == "temperature":
            result = self._convert_temperature(value, from_unit, to_unit)
            if result is None:
                return SkillResult.create_failure(
                    "Temperature conversion failed",
                    error_code="conversion_error",
                )
        else:
            # Standard conversion via base unit
            from_factor = conversions[category][from_unit]
            to_factor = conversions[category][to_unit]
            result = value * from_factor / to_factor

        return SkillResult.create_success(
            {
                "original_value": value,
                "original_unit": from_unit,
                "converted_value": round(result, self._precision),
                "converted_unit": to_unit,
                "category": category,
            }
        )

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float | None:
        """Convert between temperature units."""
        temp_map = {
            "c": "celsius",
            "celsius": "celsius",
            "f": "fahrenheit",
            "fahrenheit": "fahrenheit",
            "k": "kelvin",
            "kelvin": "kelvin",
        }

        from_type = temp_map.get(from_unit.lower())
        to_type = temp_map.get(to_unit.lower())

        if not from_type or not to_type:
            return None

        # Convert to Celsius first
        if from_type == "celsius":
            celsius = value
        elif from_type == "fahrenheit":
            celsius = (value - 32) * 5 / 9
        elif from_type == "kelvin":
            celsius = value - 273.15
        else:
            return None

        # Convert from Celsius to target
        if to_type == "celsius":
            return celsius
        elif to_type == "fahrenheit":
            return celsius * 9 / 5 + 32
        elif to_type == "kelvin":
            return celsius + 273.15
        return None


# Skill instance for registration
SKILLS = [CalculationSkill()]
