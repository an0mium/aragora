"""Tests for SpecValidator: adversarial spec review with five validator roles."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from aragora.prompt_engine.spec_validator import (
    SpecValidator,
    ValidationResult,
    ValidatorRole,
)


@dataclass
class _FakeRisk:
    description: str = ""
    mitigation: str = ""


@dataclass
class _FakeSpec:
    problem_statement: str = ""
    proposed_solution: str = ""
    implementation_plan: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    success_criteria: list = field(default_factory=list)


class TestValidatorRole:
    def test_all_roles_defined(self):
        assert len(ValidatorRole) == 5

    def test_role_values(self):
        assert ValidatorRole.DEVILS_ADVOCATE.value == "devils_advocate"
        assert ValidatorRole.SCOPE_DETECTOR.value == "scope_detector"
        assert ValidatorRole.SECURITY_REVIEWER.value == "security_reviewer"
        assert ValidatorRole.UX_ADVOCATE.value == "ux_advocate"
        assert ValidatorRole.TECH_DEBT_AUDITOR.value == "tech_debt_auditor"


class TestValidationResult:
    def test_to_dict(self):
        result = ValidationResult(
            role_results={
                ValidatorRole.DEVILS_ADVOCATE: {"passed": True, "confidence": 0.8, "issues": []}
            },
            overall_confidence=0.8,
            passed=True,
            dissenting_opinions=[],
        )
        d = result.to_dict()
        assert "devils_advocate" in d["role_results"]
        assert d["overall_confidence"] == 0.8
        assert d["passed"] is True

    def test_to_dict_with_dissenting_opinions(self):
        result = ValidationResult(
            role_results={},
            overall_confidence=0.3,
            passed=False,
            dissenting_opinions=["[devils_advocate] Missing problem statement"],
        )
        d = result.to_dict()
        assert len(d["dissenting_opinions"]) == 1


class TestSpecValidatorHeuristic:
    def setup_method(self):
        self.validator = SpecValidator()

    def test_complete_spec_passes(self):
        spec = _FakeSpec(
            problem_statement="Users can't find settings",
            proposed_solution="Add settings search",
            implementation_plan=["step1", "step2"],
            risks=[_FakeRisk(description="Complexity", mitigation="Incremental rollout")],
            success_criteria=["Users find settings 50% faster"],
        )
        result = self.validator.validate_heuristic(spec)
        assert result.passed is True
        assert result.overall_confidence > 0.7
        assert len(result.dissenting_opinions) == 0

    def test_missing_problem_statement(self):
        spec = _FakeSpec(
            proposed_solution="Add search",
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        assert result.passed is False
        da = result.role_results[ValidatorRole.DEVILS_ADVOCATE]
        assert da["passed"] is False
        assert any("Missing problem statement" in i for i in da["issues"])

    def test_missing_proposed_solution(self):
        spec = _FakeSpec(
            problem_statement="Problem exists",
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        assert result.passed is False
        da = result.role_results[ValidatorRole.DEVILS_ADVOCATE]
        assert any("Missing proposed solution" in i for i in da["issues"])

    def test_scope_creep_detection(self):
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
            implementation_plan=list(range(25)),
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        scope = result.role_results[ValidatorRole.SCOPE_DETECTOR]
        assert scope["passed"] is False
        assert any("scope creep" in i for i in scope["issues"])

    def test_scope_ok_with_20_steps(self):
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
            implementation_plan=list(range(20)),
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        scope = result.role_results[ValidatorRole.SCOPE_DETECTOR]
        assert scope["passed"] is True

    def test_security_risk_without_mitigation(self):
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
            risks=[_FakeRisk(description="Security vulnerability in auth", mitigation="")],
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        sec = result.role_results[ValidatorRole.SECURITY_REVIEWER]
        assert sec["passed"] is False
        assert any("Security risk without mitigation" in i for i in sec["issues"])

    def test_security_risk_with_mitigation_passes(self):
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
            risks=[
                _FakeRisk(description="Security vulnerability", mitigation="Add input validation")
            ],
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        sec = result.role_results[ValidatorRole.SECURITY_REVIEWER]
        assert sec["passed"] is True

    def test_no_success_criteria(self):
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
        )
        result = self.validator.validate_heuristic(spec)
        ux = result.role_results[ValidatorRole.UX_ADVOCATE]
        assert ux["passed"] is False
        assert any("No success criteria" in i for i in ux["issues"])

    def test_unmitigated_risk_flagged_by_tech_debt(self):
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
            risks=[_FakeRisk(description="Performance regression", mitigation="")],
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        td = result.role_results[ValidatorRole.TECH_DEBT_AUDITOR]
        assert td["passed"] is False
        assert any("Unmitigated risk" in i for i in td["issues"])

    def test_all_roles_present_in_results(self):
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        for role in ValidatorRole:
            assert role in result.role_results

    def test_overall_confidence_is_average(self):
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
            success_criteria=["criterion"],
        )
        result = self.validator.validate_heuristic(spec)
        confidences = [r["confidence"] for r in result.role_results.values()]
        expected = sum(confidences) / len(confidences)
        assert abs(result.overall_confidence - expected) < 1e-9


class TestSpecValidatorAsync:
    def test_async_validate_delegates_to_heuristic(self):
        validator = SpecValidator()
        spec = _FakeSpec(
            problem_statement="Problem",
            proposed_solution="Solution",
            success_criteria=["criterion"],
        )
        result = asyncio.run(validator.validate(spec))
        assert isinstance(result, ValidationResult)
        assert result.passed is True


class TestPublicImports:
    def test_importable_from_package(self):
        from aragora.prompt_engine import SpecValidator, ValidationResult, ValidatorRole

        assert SpecValidator is not None
        assert ValidationResult is not None
        assert ValidatorRole is not None
