"""
Pipeline module for aragora - Decision-to-PR generation.

Transforms debate outcomes into actionable development artifacts:
- DecisionMemo: Summary of debate conclusions
- RiskRegister: Identified risks and mitigations
- VerificationPlan: Verification strategy
- PatchPlan: Implementation steps
"""

from aragora.pipeline.pr_generator import DecisionMemo, PatchPlan, PRGenerator
from aragora.pipeline.risk_register import Risk, RiskRegister
from aragora.pipeline.verification_plan import (
    VerificationCase,
    VerificationPlan,
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
