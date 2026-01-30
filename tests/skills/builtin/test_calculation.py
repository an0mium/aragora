"""
Tests for aragora.skills.builtin.calculation module.

Covers:
- CalculationSkill manifest and initialization
- Safe expression evaluation
- Mathematical functions
- Unit conversions
- Security (blocking dangerous operations)
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from aragora.skills.base import SkillContext, SkillStatus
from aragora.skills.builtin.calculation import (
    CalculationSkill,
    SafeEvaluator,
    safe_eval,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def skill() -> CalculationSkill:
    """Create a calculation skill for testing."""
    return CalculationSkill()


@pytest.fixture
def context() -> SkillContext:
    """Create a context for testing."""
    return SkillContext(user_id="user123")


# =============================================================================
# CalculationSkill Manifest Tests
# =============================================================================


class TestCalculationSkillManifest:
    """Tests for CalculationSkill manifest."""

    def test_manifest_name(self, skill: CalculationSkill):
        """Test manifest name."""
        assert skill.manifest.name == "calculation"

    def test_manifest_version(self, skill: CalculationSkill):
        """Test manifest version."""
        assert skill.manifest.version == "1.0.0"

    def test_manifest_input_schema(self, skill: CalculationSkill):
        """Test manifest input schema."""
        schema = skill.manifest.input_schema

        assert "expression" in schema
        assert schema["expression"]["type"] == "string"
        assert schema["expression"]["required"] is True

        assert "variables" in schema
        assert "precision" in schema
        assert "unit_conversion" in schema

    def test_manifest_debate_compatible(self, skill: CalculationSkill):
        """Test skill is debate compatible."""
        assert skill.manifest.debate_compatible is True

    def test_manifest_fast_timeout(self, skill: CalculationSkill):
        """Test manifest has fast timeout."""
        assert skill.manifest.max_execution_time_seconds == 5.0


# =============================================================================
# Safe Evaluation Tests
# =============================================================================


class TestSafeEval:
    """Tests for safe_eval function."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert safe_eval("2 + 3") == 5
        assert safe_eval("10 - 4") == 6
        assert safe_eval("3 * 4") == 12
        assert safe_eval("15 / 3") == 5
        assert safe_eval("17 // 5") == 3
        assert safe_eval("17 % 5") == 2
        assert safe_eval("2 ** 3") == 8

    def test_unary_operators(self):
        """Test unary operators."""
        assert safe_eval("-5") == -5
        assert safe_eval("+5") == 5
        assert safe_eval("--5") == 5

    def test_parentheses(self):
        """Test parentheses for order of operations."""
        assert safe_eval("(2 + 3) * 4") == 20
        assert safe_eval("2 + 3 * 4") == 14
        assert safe_eval("((2 + 3) * 4) / 2") == 10

    def test_constants(self):
        """Test mathematical constants."""
        assert safe_eval("pi") == math.pi
        assert safe_eval("e") == math.e
        assert safe_eval("tau") == math.tau

    def test_math_functions(self):
        """Test mathematical functions."""
        assert safe_eval("sqrt(16)") == 4
        assert safe_eval("abs(-5)") == 5
        assert safe_eval("round(3.7)") == 4
        assert safe_eval("floor(3.7)") == 3
        assert safe_eval("ceil(3.2)") == 4

    def test_trigonometric_functions(self):
        """Test trigonometric functions."""
        assert abs(safe_eval("sin(0)")) < 0.0001
        assert abs(safe_eval("cos(0)") - 1) < 0.0001
        assert abs(safe_eval("tan(0)")) < 0.0001

    def test_logarithmic_functions(self):
        """Test logarithmic functions."""
        assert abs(safe_eval("log(e)") - 1) < 0.0001
        assert safe_eval("log10(100)") == 2
        assert safe_eval("log2(8)") == 3

    def test_variables(self):
        """Test variable substitution."""
        assert safe_eval("x + y", {"x": 3, "y": 5}) == 8
        assert safe_eval("a * b + c", {"a": 2, "b": 3, "c": 4}) == 10

    def test_comparison_operators(self):
        """Test comparison operators."""
        assert safe_eval("3 < 5") is True
        assert safe_eval("3 > 5") is False
        assert safe_eval("3 == 3") is True
        assert safe_eval("3 != 5") is True
        assert safe_eval("3 <= 3") is True
        assert safe_eval("3 >= 5") is False

    def test_ternary_expression(self):
        """Test ternary expressions."""
        assert safe_eval("5 if 3 > 2 else 10") == 5
        assert safe_eval("5 if 3 < 2 else 10") == 10

    def test_list_operations(self):
        """Test list operations."""
        assert safe_eval("min([1, 2, 3])") == 1
        assert safe_eval("max([1, 2, 3])") == 3
        assert safe_eval("sum([1, 2, 3])") == 6


# =============================================================================
# Security Tests
# =============================================================================


class TestSafeEvalSecurity:
    """Tests for safe_eval security."""

    def test_blocks_import(self):
        """Test that import is blocked."""
        with pytest.raises(ValueError):
            safe_eval("__import__('os')")

    def test_blocks_exec(self):
        """Test that exec-like operations are blocked."""
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval("exec('print(1)')")

    def test_blocks_unknown_functions(self):
        """Test that unknown functions are blocked."""
        with pytest.raises(ValueError):
            safe_eval("dangerous_function()")

    def test_blocks_unknown_variables(self):
        """Test that unknown variables raise error."""
        with pytest.raises(ValueError):
            safe_eval("unknown_var + 5")

    def test_limits_exponent(self):
        """Test that large exponents are limited."""
        with pytest.raises(ValueError):
            safe_eval("2 ** 10000")

    def test_syntax_error_handling(self):
        """Test syntax error handling."""
        with pytest.raises(SyntaxError):
            safe_eval("2 +")


# =============================================================================
# CalculationSkill Execution Tests
# =============================================================================


class TestCalculationSkillExecution:
    """Tests for CalculationSkill execution."""

    @pytest.mark.asyncio
    async def test_execute_missing_expression(self, skill: CalculationSkill, context: SkillContext):
        """Test execution fails without expression."""
        result = skill.execute_sync({}, context)

        assert result.success is False
        assert "expression" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_basic_calculation(self, skill: CalculationSkill, context: SkillContext):
        """Test basic calculation execution."""
        result = skill.execute_sync({"expression": "2 + 3 * 4"}, context)

        assert result.success is True
        assert result.data["result"] == 14
        assert result.data["expression"] == "2 + 3 * 4"

    @pytest.mark.asyncio
    async def test_execute_with_variables(self, skill: CalculationSkill, context: SkillContext):
        """Test calculation with variables."""
        result = skill.execute_sync(
            {"expression": "x + y * z", "variables": {"x": 1, "y": 2, "z": 3}},
            context,
        )

        assert result.success is True
        assert result.data["result"] == 7
        assert "x" in result.data["variables_used"]

    @pytest.mark.asyncio
    async def test_execute_with_precision(self, skill: CalculationSkill, context: SkillContext):
        """Test calculation with custom precision."""
        result = skill.execute_sync({"expression": "pi * 2", "precision": 3}, context)

        assert result.success is True
        assert result.data["result"] == round(math.pi * 2, 3)

    @pytest.mark.asyncio
    async def test_execute_division_by_zero(self, skill: CalculationSkill, context: SkillContext):
        """Test division by zero handling."""
        result = skill.execute_sync({"expression": "5 / 0"}, context)

        assert result.success is False
        assert result.error_code == "division_by_zero"

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self, skill: CalculationSkill, context: SkillContext):
        """Test syntax error handling."""
        result = skill.execute_sync({"expression": "2 +"}, context)

        assert result.success is False
        assert result.error_code == "syntax_error"

    @pytest.mark.asyncio
    async def test_execute_invalid_variable(self, skill: CalculationSkill, context: SkillContext):
        """Test invalid variable handling."""
        result = skill.execute_sync(
            {"expression": "x + 1", "variables": {"x": "not a number"}},
            context,
        )

        assert result.success is False
        assert result.error_code == "invalid_variable"


# =============================================================================
# Unit Conversion Tests
# =============================================================================


class TestUnitConversion:
    """Tests for unit conversion functionality."""

    @pytest.mark.asyncio
    async def test_length_conversion_meters_to_feet(
        self, skill: CalculationSkill, context: SkillContext
    ):
        """Test meters to feet conversion."""
        result = skill.execute_sync(
            {"unit_conversion": {"from": "m", "to": "ft", "value": 1}},
            context,
        )

        assert result.success is True
        assert abs(result.data["converted_value"] - 3.28084) < 0.01

    @pytest.mark.asyncio
    async def test_weight_conversion_kg_to_lb(self, skill: CalculationSkill, context: SkillContext):
        """Test kilograms to pounds conversion."""
        result = skill.execute_sync(
            {"unit_conversion": {"from": "kg", "to": "lb", "value": 1}},
            context,
        )

        assert result.success is True
        assert abs(result.data["converted_value"] - 2.20462) < 0.01

    @pytest.mark.asyncio
    async def test_temperature_celsius_to_fahrenheit(
        self, skill: CalculationSkill, context: SkillContext
    ):
        """Test Celsius to Fahrenheit conversion."""
        result = skill.execute_sync(
            {"unit_conversion": {"from": "c", "to": "f", "value": 0}},
            context,
        )

        assert result.success is True
        assert result.data["converted_value"] == 32

    @pytest.mark.asyncio
    async def test_temperature_fahrenheit_to_celsius(
        self, skill: CalculationSkill, context: SkillContext
    ):
        """Test Fahrenheit to Celsius conversion."""
        result = skill.execute_sync(
            {"unit_conversion": {"from": "f", "to": "c", "value": 212}},
            context,
        )

        assert result.success is True
        assert result.data["converted_value"] == 100

    @pytest.mark.asyncio
    async def test_data_conversion_mb_to_gb(self, skill: CalculationSkill, context: SkillContext):
        """Test megabytes to gigabytes conversion."""
        result = skill.execute_sync(
            {"unit_conversion": {"from": "mb", "to": "gb", "value": 1024}},
            context,
        )

        assert result.success is True
        assert result.data["converted_value"] == 1

    @pytest.mark.asyncio
    async def test_time_conversion_hours_to_minutes(
        self, skill: CalculationSkill, context: SkillContext
    ):
        """Test hours to minutes conversion."""
        result = skill.execute_sync(
            {"unit_conversion": {"from": "h", "to": "min", "value": 2}},
            context,
        )

        assert result.success is True
        assert result.data["converted_value"] == 120

    @pytest.mark.asyncio
    async def test_incompatible_units(self, skill: CalculationSkill, context: SkillContext):
        """Test incompatible unit conversion fails."""
        result = skill.execute_sync(
            {"unit_conversion": {"from": "kg", "to": "meters", "value": 1}},
            context,
        )

        assert result.success is False
        assert result.error_code == "incompatible_units"

    @pytest.mark.asyncio
    async def test_missing_conversion_value(self, skill: CalculationSkill, context: SkillContext):
        """Test missing conversion value fails."""
        result = skill.execute_sync(
            {"unit_conversion": {"from": "m", "to": "ft"}},
            context,
        )

        assert result.success is False
        assert result.error_code == "missing_value"


# =============================================================================
# SKILLS Registration Tests
# =============================================================================


class TestSkillsRegistration:
    """Tests for SKILLS module-level list."""

    def test_skills_list_exists(self):
        """Test SKILLS list exists in module."""
        from aragora.skills.builtin import calculation

        assert hasattr(calculation, "SKILLS")

    def test_skills_list_contains_skill(self):
        """Test SKILLS list contains CalculationSkill."""
        from aragora.skills.builtin.calculation import SKILLS

        assert len(SKILLS) == 1
        assert isinstance(SKILLS[0], CalculationSkill)
