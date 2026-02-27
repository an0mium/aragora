"""SpecValidator: adversarial review of specifications.

Validates specifications through five validator roles, each assessing
the spec from a different angle. Supports heuristic (no LLM) and
debate-driven (Arena) validation modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aragora.prompt_engine.types import Specification

logger = logging.getLogger(__name__)


class ValidatorRole(str, Enum):
    """Adversarial reviewer roles for spec validation."""

    DEVILS_ADVOCATE = "devils_advocate"
    SCOPE_DETECTOR = "scope_detector"
    SECURITY_REVIEWER = "security_reviewer"
    UX_ADVOCATE = "ux_advocate"
    TECH_DEBT_AUDITOR = "tech_debt_auditor"


@dataclass
class ValidationResult:
    """Result of spec validation across all roles."""

    role_results: dict[ValidatorRole, dict[str, Any]]
    overall_confidence: float
    passed: bool
    dissenting_opinions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_results": {k.value: v for k, v in self.role_results.items()},
            "overall_confidence": self.overall_confidence,
            "passed": self.passed,
            "dissenting_opinions": self.dissenting_opinions,
        }


class SpecValidator:
    """Validates specifications through adversarial review.

    Five validator roles each assess the spec from a different angle:
    1. Devil's Advocate -- challenges completeness and feasibility
    2. Scope Detector -- flags scope creep and unnecessary complexity
    3. Security Reviewer -- checks security implications
    4. UX Advocate -- evaluates user value and success criteria
    5. Tech Debt Auditor -- assesses maintenance burden and risk mitigation
    """

    def validate_heuristic(self, spec: Specification) -> ValidationResult:
        """Structural validation without LLM calls."""
        issues: list[str] = []
        role_results: dict[ValidatorRole, dict[str, Any]] = {}

        # Devil's advocate: completeness
        da_issues: list[str] = []
        problem = getattr(spec, "problem_statement", None) or ""
        solution = getattr(spec, "proposed_solution", None) or ""
        if not problem:
            da_issues.append("Missing problem statement")
        if not solution:
            da_issues.append("Missing proposed solution")
        da_passed = not da_issues
        role_results[ValidatorRole.DEVILS_ADVOCATE] = {
            "passed": da_passed,
            "confidence": 0.8 if da_passed else 0.3,
            "issues": da_issues,
        }
        issues.extend(f"[devils_advocate] {i}" for i in da_issues)

        # Scope detector: plan size
        plan = getattr(spec, "implementation_plan", None) or []
        if not isinstance(plan, list):
            plan = []
        scope_issues: list[str] = []
        if len(plan) > 20:
            scope_issues.append(
                f"Implementation plan has {len(plan)} steps -- possible scope creep"
            )
        scope_passed = not scope_issues
        role_results[ValidatorRole.SCOPE_DETECTOR] = {
            "passed": scope_passed,
            "confidence": 0.9 if scope_passed else 0.4,
            "issues": scope_issues,
        }
        issues.extend(f"[scope_detector] {i}" for i in scope_issues)

        # Security reviewer: risk mitigations
        risks = getattr(spec, "risks", None) or getattr(spec, "risk_register", None) or []
        sec_issues: list[str] = []
        for risk in risks:
            desc = getattr(risk, "description", "") or ""
            mitigation = getattr(risk, "mitigation", "") or ""
            if "security" in desc.lower() and not mitigation:
                sec_issues.append(f"Security risk without mitigation: {desc}")
        sec_passed = not sec_issues
        role_results[ValidatorRole.SECURITY_REVIEWER] = {
            "passed": sec_passed,
            "confidence": 0.85 if sec_passed else 0.4,
            "issues": sec_issues,
        }
        issues.extend(f"[security_reviewer] {i}" for i in sec_issues)

        # UX advocate: success criteria
        criteria = getattr(spec, "success_criteria", None) or []
        ux_issues: list[str] = []
        if not criteria:
            ux_issues.append("No success criteria defined")
        ux_passed = not ux_issues
        role_results[ValidatorRole.UX_ADVOCATE] = {
            "passed": ux_passed,
            "confidence": 0.8 if ux_passed else 0.3,
            "issues": ux_issues,
        }
        issues.extend(f"[ux_advocate] {i}" for i in ux_issues)

        # Tech debt auditor: risk mitigation coverage
        td_issues: list[str] = []
        for risk in risks:
            desc = getattr(risk, "description", "") or ""
            mitigation = getattr(risk, "mitigation", "") or ""
            if not mitigation:
                td_issues.append(f"Unmitigated risk: {desc}")
        td_passed = not td_issues
        role_results[ValidatorRole.TECH_DEBT_AUDITOR] = {
            "passed": td_passed,
            "confidence": 0.8 if td_passed else 0.4,
            "issues": td_issues,
        }
        issues.extend(f"[tech_debt_auditor] {i}" for i in td_issues)

        # Overall
        confidences = [r["confidence"] for r in role_results.values()]
        overall = sum(confidences) / len(confidences) if confidences else 0
        all_passed = all(r["passed"] for r in role_results.values())

        return ValidationResult(
            role_results=role_results,
            overall_confidence=overall,
            passed=all_passed,
            dissenting_opinions=issues,
        )

    async def validate(self, spec: Specification) -> ValidationResult:
        """Full validation. Currently uses heuristic; extensible to Arena debate."""
        return self.validate_heuristic(spec)
