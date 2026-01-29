"""
Onboarding Wizard - Guided setup for Aragora (Moltbot parity).

Provides a structured onboarding experience for new users and devices,
walking them through:
- Account/device registration
- Agent provider configuration
- Channel setup (Slack, Teams, etc.)
- Permission/role assignment
- Initial preferences

This module implements the Moltbot onboarding model adapted for
enterprise deployment with SSO/RBAC integration.

Usage:
    from aragora.onboarding import OnboardingWizard

    wizard = OnboardingWizard()
    session = await wizard.start_session(user_id="user-123")
    result = await wizard.complete_step(session.session_id, "welcome", {...})
"""

from aragora.onboarding.wizard import (
    OnboardingWizard,
    OnboardingSession,
    OnboardingStep,
    StepStatus,
    WizardConfig,
)

__all__ = [
    "OnboardingWizard",
    "OnboardingSession",
    "OnboardingStep",
    "StepStatus",
    "WizardConfig",
]
