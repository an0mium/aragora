"""
Pipeline module for aragora - Decision-to-PR generation.

Transforms debate outcomes into actionable development artifacts:
- DecisionMemo: Summary of debate conclusions
- RiskRegister: Identified risks and mitigations
- VerificationPlan: Verification strategy
- PatchPlan: Implementation steps
"""

from aragora.pipeline.pr_generator import PRGenerator, DecisionMemo, PatchPlan
from aragora.pipeline.risk_register import RiskRegister, Risk
from aragora.pipeline.test_plan import (
    VerificationPlan,
    VerificationCase,
    VerificationPlanGenerator,
)

# Backward compatibility aliases (old names triggered pytest discovery)
TestPlan = VerificationPlan
TestCase = VerificationCase
TestPlanGenerator = VerificationPlanGenerator

__all__ = [
    "PRGenerator",
    "DecisionMemo",
    "PatchPlan",
    "RiskRegister",
    "Risk",
    "VerificationPlan",
    "VerificationCase",
    # Backward compatibility
    "TestPlan",
    "TestCase",
]
