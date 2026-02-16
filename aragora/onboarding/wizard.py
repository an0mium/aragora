"""
Onboarding Wizard implementation.

Provides guided setup experience for new users and devices.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of an onboarding step."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class OnboardingStep:
    """A single step in the onboarding process."""

    step_id: str
    name: str
    description: str
    required: bool = True
    order: int = 0
    status: StepStatus = StepStatus.PENDING
    data: dict[str, Any] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    completed_at: float | None = None
    # Optional fields that guide the step
    fields: list[dict[str, Any]] = field(default_factory=list)
    help_text: str = ""
    next_step: str | None = None
    skip_condition: str | None = None  # Expression to evaluate for auto-skip

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "order": self.order,
            "status": self.status.value,
            "data": self.data,
            "validation_errors": self.validation_errors,
            "completed_at": self.completed_at,
            "fields": self.fields,
            "help_text": self.help_text,
            "next_step": self.next_step,
            "skip_condition": self.skip_condition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnboardingStep:
        """Deserialize from dictionary."""
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            description=data["description"],
            required=data.get("required", True),
            order=data.get("order", 0),
            status=StepStatus(data.get("status", "pending")),
            data=data.get("data", {}),
            validation_errors=data.get("validation_errors", []),
            completed_at=data.get("completed_at"),
            fields=data.get("fields", []),
            help_text=data.get("help_text", ""),
            next_step=data.get("next_step"),
            skip_condition=data.get("skip_condition"),
        )


@dataclass
class OnboardingSession:
    """An active onboarding session."""

    user_id: str
    session_id: str = ""
    device_id: str | None = None
    tenant_id: str | None = None
    steps: list[OnboardingStep] = field(default_factory=list)
    current_step: str = ""
    started_at: float = 0.0
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"onboard-{uuid.uuid4().hex[:8]}"
        if not self.started_at:
            self.started_at = time.time()

    @property
    def is_complete(self) -> bool:
        """Check if all required steps are complete."""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in self.steps if s.required
        )

    @property
    def progress(self) -> float:
        """Get completion progress as percentage."""
        if not self.steps:
            return 0.0
        completed = sum(
            1 for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        )
        return (completed / len(self.steps)) * 100

    def get_step(self, step_id: str) -> OnboardingStep | None:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "device_id": self.device_id,
            "tenant_id": self.tenant_id,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
            "is_complete": self.is_complete,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnboardingSession:
        """Deserialize from dictionary."""
        session = cls(
            user_id=data["user_id"],
            session_id=data.get("session_id", ""),
            device_id=data.get("device_id"),
            tenant_id=data.get("tenant_id"),
            current_step=data.get("current_step", ""),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )
        session.steps = [OnboardingStep.from_dict(s) for s in data.get("steps", [])]
        return session


@dataclass
class WizardConfig:
    """Configuration for the OnboardingWizard."""

    # Default steps to include
    include_welcome: bool = True
    include_device_setup: bool = True
    include_provider_config: bool = True
    include_channel_setup: bool = True
    include_preferences: bool = True
    include_permissions: bool = True

    # Behavior
    allow_skip_required: bool = False
    auto_advance: bool = True
    session_timeout_seconds: float = 3600.0  # 1 hour

    # Enterprise features
    require_sso: bool = False
    require_tenant: bool = False
    require_approval: bool = False


class OnboardingWizard:
    """
    Guided onboarding experience manager.

    Manages onboarding sessions, step progression, and data collection
    for new users and devices.
    """

    def __init__(self, config: WizardConfig | None = None):
        """Initialize the wizard."""
        self.config = config or WizardConfig()
        self._sessions: dict[str, OnboardingSession] = {}
        self._step_validators: dict[str, Callable] = {}
        self._step_handlers: dict[str, Callable] = {}

    def _create_default_steps(self) -> list[OnboardingStep]:
        """Create the default onboarding steps based on config."""
        steps = []
        order = 0

        if self.config.include_welcome:
            steps.append(
                OnboardingStep(
                    step_id="welcome",
                    name="Welcome",
                    description="Welcome to Aragora! Let's get you set up.",
                    order=order,
                    required=False,
                    help_text="This wizard will guide you through the initial setup.",
                )
            )
            order += 1

        if self.config.include_device_setup:
            steps.append(
                OnboardingStep(
                    step_id="device",
                    name="Device Setup",
                    description="Register your device for access.",
                    order=order,
                    fields=[
                        {"name": "device_name", "type": "text", "required": True},
                        {
                            "name": "device_type",
                            "type": "select",
                            "options": ["laptop", "desktop", "phone", "tablet"],
                        },
                    ],
                )
            )
            order += 1

        if self.config.include_provider_config:
            steps.append(
                OnboardingStep(
                    step_id="providers",
                    name="AI Providers",
                    description="Configure your AI provider API keys.",
                    order=order,
                    fields=[
                        {
                            "name": "anthropic_key",
                            "type": "password",
                            "label": "Anthropic API Key",
                        },
                        {
                            "name": "openai_key",
                            "type": "password",
                            "label": "OpenAI API Key",
                        },
                        {
                            "name": "openrouter_key",
                            "type": "password",
                            "label": "OpenRouter API Key (fallback)",
                        },
                    ],
                    help_text="At least one API key is required for agent functionality.",
                )
            )
            order += 1

        if self.config.include_channel_setup:
            steps.append(
                OnboardingStep(
                    step_id="channels",
                    name="Channels",
                    description="Connect your communication channels.",
                    order=order,
                    required=False,
                    fields=[
                        {"name": "slack_enabled", "type": "checkbox", "label": "Slack"},
                        {"name": "teams_enabled", "type": "checkbox", "label": "Teams"},
                        {"name": "email_enabled", "type": "checkbox", "label": "Email"},
                    ],
                    help_text="You can always add more channels later.",
                )
            )
            order += 1

        if self.config.include_preferences:
            steps.append(
                OnboardingStep(
                    step_id="preferences",
                    name="Preferences",
                    description="Set your default preferences.",
                    order=order,
                    required=False,
                    fields=[
                        {
                            "name": "default_agent",
                            "type": "select",
                            "options": ["claude", "gpt-4", "gemini"],
                        },
                        {"name": "notifications", "type": "checkbox", "default": True},
                        {"name": "timezone", "type": "timezone"},
                    ],
                )
            )
            order += 1

        if self.config.include_permissions:
            steps.append(
                OnboardingStep(
                    step_id="permissions",
                    name="Permissions",
                    description="Review and accept permissions.",
                    order=order,
                    fields=[
                        {
                            "name": "accept_terms",
                            "type": "checkbox",
                            "required": True,
                            "label": "I accept the Terms of Service",
                        },
                        {
                            "name": "accept_privacy",
                            "type": "checkbox",
                            "required": True,
                            "label": "I accept the Privacy Policy",
                        },
                    ],
                )
            )
            order += 1

        return steps

    async def start_session(
        self,
        user_id: str,
        device_id: str | None = None,
        tenant_id: str | None = None,
        custom_steps: list[OnboardingStep] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OnboardingSession:
        """
        Start a new onboarding session.

        Args:
            user_id: ID of the user being onboarded
            device_id: Optional device ID
            tenant_id: Optional tenant ID for multi-tenancy
            custom_steps: Optional custom steps (replaces defaults)
            metadata: Optional metadata to attach

        Returns:
            The created session
        """
        # Check for existing active session
        for session in self._sessions.values():
            if session.user_id == user_id and not session.is_complete:
                # Resume existing session
                return session

        steps = custom_steps or self._create_default_steps()

        session = OnboardingSession(
            user_id=user_id,
            device_id=device_id,
            tenant_id=tenant_id,
            steps=steps,
            current_step=steps[0].step_id if steps else "",
            metadata=metadata or {},
        )

        # Mark first step as active
        if session.steps:
            session.steps[0].status = StepStatus.ACTIVE

        self._sessions[session.session_id] = session
        return session

    async def get_session(self, session_id: str) -> OnboardingSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def get_user_session(self, user_id: str) -> OnboardingSession | None:
        """Get the active session for a user."""
        for session in self._sessions.values():
            if session.user_id == user_id and not session.is_complete:
                return session
        return None

    async def complete_step(
        self,
        session_id: str,
        step_id: str,
        data: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Complete an onboarding step.

        Args:
            session_id: Session ID
            step_id: Step to complete
            data: Data collected for this step

        Returns:
            Tuple of (success, validation_errors)
        """
        session = self._sessions.get(session_id)
        if not session:
            return False, ["Session not found"]

        step = session.get_step(step_id)
        if not step:
            return False, ["Step not found"]

        if step.status == StepStatus.COMPLETED:
            return False, ["Step already completed"]

        # Validate step data
        errors = await self._validate_step(step, data)
        if errors:
            step.validation_errors = errors
            step.status = StepStatus.FAILED
            return False, errors

        # Run step handler if registered
        if step_id in self._step_handlers:
            try:
                await self._step_handlers[step_id](session, step, data)
            except (RuntimeError, ValueError, TypeError, KeyError, OSError) as e:
                logger.warning("Step handler '%s' failed: %s", step_id, e)
                step.validation_errors = ["Step processing failed"]
                step.status = StepStatus.FAILED
                return False, ["Step processing failed"]

        # Mark complete
        step.data = data
        step.status = StepStatus.COMPLETED
        step.completed_at = time.time()
        step.validation_errors = []

        # Advance to next step
        if self.config.auto_advance:
            await self._advance_session(session, step)

        # Check if onboarding is complete
        if session.is_complete:
            session.completed_at = time.time()

        return True, []

    async def skip_step(self, session_id: str, step_id: str) -> bool:
        """
        Skip an onboarding step.

        Returns True if skipped, False if step is required and can't be skipped.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        step = session.get_step(step_id)
        if not step:
            return False

        if step.required and not self.config.allow_skip_required:
            return False

        step.status = StepStatus.SKIPPED
        step.completed_at = time.time()

        if self.config.auto_advance:
            await self._advance_session(session, step)

        return True

    async def go_back(self, session_id: str) -> OnboardingStep | None:
        """
        Go back to the previous step.

        Returns the previous step if available.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        current_idx = next(
            (i for i, s in enumerate(session.steps) if s.step_id == session.current_step),
            -1,
        )

        if current_idx <= 0:
            return None

        # Find previous non-skipped step
        for i in range(current_idx - 1, -1, -1):
            prev_step = session.steps[i]
            if prev_step.status != StepStatus.SKIPPED:
                # Reset current step
                current = session.get_step(session.current_step)
                if current:
                    current.status = StepStatus.PENDING

                # Make previous step active
                prev_step.status = StepStatus.ACTIVE
                session.current_step = prev_step.step_id
                return prev_step

        return None

    async def get_current_step(self, session_id: str) -> OnboardingStep | None:
        """Get the current active step."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.get_step(session.current_step)

    def register_validator(
        self, step_id: str, validator: Callable[[OnboardingStep, dict], list[str]]
    ):
        """Register a custom validator for a step."""
        self._step_validators[step_id] = validator

    def register_handler(
        self,
        step_id: str,
        handler: Callable[[OnboardingSession, OnboardingStep, dict], None],
    ):
        """Register a handler to run when a step is completed."""
        self._step_handlers[step_id] = handler

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an onboarding session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def get_stats(self) -> dict[str, Any]:
        """Get wizard statistics."""
        active = sum(1 for s in self._sessions.values() if not s.is_complete)
        completed = sum(1 for s in self._sessions.values() if s.is_complete)
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": active,
            "completed_sessions": completed,
        }

    async def _validate_step(self, step: OnboardingStep, data: dict[str, Any]) -> list[str]:
        """Validate step data."""
        errors = []

        # Check required fields
        for field_spec in step.fields:
            field_name = field_spec.get("name")
            if field_spec.get("required") and not data.get(field_name):
                errors.append(f"Field '{field_name}' is required")

        # Run custom validator if registered
        if step.step_id in self._step_validators:
            custom_errors = self._step_validators[step.step_id](step, data)
            if asyncio.iscoroutine(custom_errors):
                custom_errors = await custom_errors
            errors.extend(custom_errors)

        return errors

    async def _advance_session(self, session: OnboardingSession, completed_step: OnboardingStep):
        """Advance to the next step in the session."""
        # Find next step
        current_idx = next(
            (i for i, s in enumerate(session.steps) if s.step_id == completed_step.step_id),
            -1,
        )

        if current_idx < 0 or current_idx >= len(session.steps) - 1:
            session.current_step = ""
            return

        next_step = session.steps[current_idx + 1]

        # Check skip condition
        if next_step.skip_condition:
            # Simple skip condition evaluation (could be extended)
            if self._evaluate_skip_condition(session, next_step.skip_condition):
                next_step.status = StepStatus.SKIPPED
                await self._advance_session(session, next_step)
                return

        next_step.status = StepStatus.ACTIVE
        session.current_step = next_step.step_id

    def _evaluate_skip_condition(self, session: OnboardingSession, condition: str) -> bool:
        """Evaluate a skip condition. Simple implementation."""
        # For now, just check if a step is completed
        # Could be extended to support more complex expressions
        if condition.startswith("step_completed:"):
            step_id = condition.split(":")[1]
            step = session.get_step(step_id)
            return step is not None and step.status == StepStatus.COMPLETED
        return False

    # =========================================================================
    # Device-specific onboarding (Moltbot pattern)
    # =========================================================================

    async def create_device_session(
        self,
        device_id: str,
        capabilities: list[str],
        device_name: str = "",
        device_type: str = "unknown",
        user_id: str | None = None,
    ) -> OnboardingSession:
        """
        Create an onboarding session with device-specific steps.

        Pattern: Device Onboarding
        Inspired by: Moltbot (https://github.com/moltbot)
        Aragora adaptation: Capability-based step generation.

        Generates appropriate onboarding steps based on device capabilities.

        Args:
            device_id: Device ID to onboard
            capabilities: List of device capability strings
            device_name: Human-readable device name
            device_type: Device type (laptop, phone, etc.)
            user_id: Optional user ID (defaults to device_id)

        Returns:
            OnboardingSession with device-appropriate steps
        """
        # Build steps based on capabilities
        steps = self._get_steps_for_capabilities(capabilities)

        # Create session
        session = OnboardingSession(
            user_id=user_id or device_id,
            device_id=device_id,
            steps=steps,
            current_step=steps[0].step_id if steps else "",
            metadata={
                "device_type": device_type,
                "device_name": device_name,
                "capabilities": capabilities,
            },
        )

        # Mark first step as active
        if session.steps:
            session.steps[0].status = StepStatus.ACTIVE

        self._sessions[session.session_id] = session
        return session

    def _get_steps_for_capabilities(self, capabilities: list[str]) -> list[OnboardingStep]:
        """
        Generate onboarding steps based on device capabilities.

        Maps device capabilities to appropriate onboarding steps.

        Args:
            capabilities: List of device capability strings

        Returns:
            List of OnboardingStep objects
        """
        steps = []
        order = 0

        # Always include welcome
        steps.append(
            OnboardingStep(
                step_id="welcome",
                name="Welcome",
                description="Welcome to Aragora! Let's set up your device.",
                order=order,
                required=False,
            )
        )
        order += 1

        # Device pairing (always required for device sessions)
        steps.append(
            OnboardingStep(
                step_id="device_pairing",
                name="Device Pairing",
                description="Pair this device with your Aragora account.",
                order=order,
                required=True,
                fields=[
                    {"name": "pairing_code", "type": "text", "required": True},
                    {"name": "device_nickname", "type": "text", "required": False},
                ],
                help_text="Enter the pairing code shown in your Aragora dashboard.",
            )
        )
        order += 1

        # Voice setup if device has voice capability
        if "voice" in capabilities or "microphone" in capabilities:
            steps.append(
                OnboardingStep(
                    step_id="voice_setup",
                    name="Voice Setup",
                    description="Configure voice interaction settings.",
                    order=order,
                    required=False,
                    fields=[
                        {
                            "name": "wake_word",
                            "type": "select",
                            "options": ["hey aragora", "ok aragora", "computer"],
                        },
                        {"name": "voice_enabled", "type": "checkbox", "default": True},
                        {"name": "always_listening", "type": "checkbox", "default": False},
                    ],
                    help_text="Set up wake word detection for hands-free interaction.",
                )
            )
            order += 1

        # Display setup if device has display capability
        if "display" in capabilities or "screen" in capabilities:
            steps.append(
                OnboardingStep(
                    step_id="display_setup",
                    name="Display Settings",
                    description="Configure display preferences.",
                    order=order,
                    required=False,
                    fields=[
                        {"name": "theme", "type": "select", "options": ["light", "dark", "auto"]},
                        {"name": "canvas_enabled", "type": "checkbox", "default": True},
                        {"name": "notifications_visual", "type": "checkbox", "default": True},
                    ],
                )
            )
            order += 1

        # Computer use setup if device has automation capability
        if "automation" in capabilities or "computer_use" in capabilities:
            steps.append(
                OnboardingStep(
                    step_id="automation_setup",
                    name="Automation Setup",
                    description="Configure computer use and automation permissions.",
                    order=order,
                    required=True,
                    fields=[
                        {
                            "name": "allow_screenshots",
                            "type": "checkbox",
                            "default": True,
                            "required": True,
                        },
                        {"name": "allow_mouse_control", "type": "checkbox", "default": False},
                        {"name": "allow_keyboard_input", "type": "checkbox", "default": False},
                        {"name": "require_approval", "type": "checkbox", "default": True},
                    ],
                    help_text="These permissions control what actions agents can perform.",
                )
            )
            order += 1

        # Notification setup
        steps.append(
            OnboardingStep(
                step_id="notifications",
                name="Notifications",
                description="Configure how you receive notifications.",
                order=order,
                required=False,
                fields=[
                    {"name": "push_enabled", "type": "checkbox", "default": True},
                    {"name": "sound_enabled", "type": "checkbox", "default": True},
                    {"name": "quiet_hours", "type": "checkbox", "default": False},
                ],
            )
        )
        order += 1

        # Security setup
        steps.append(
            OnboardingStep(
                step_id="security",
                name="Security",
                description="Review and accept security settings.",
                order=order,
                required=True,
                fields=[
                    {
                        "name": "accept_device_policy",
                        "type": "checkbox",
                        "required": True,
                        "label": "I accept the device security policy",
                    },
                    {"name": "enable_encryption", "type": "checkbox", "default": True},
                    {"name": "auto_lock", "type": "checkbox", "default": True},
                ],
            )
        )

        return steps

    async def update_device_capabilities(
        self,
        session_id: str,
        new_capabilities: list[str],
    ) -> bool:
        """
        Update a session with new device capabilities.

        Can be used when a device's capabilities change mid-onboarding.

        Args:
            session_id: Session to update
            new_capabilities: Updated capability list

        Returns:
            True if session was updated
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        old_capabilities = session.metadata.get("capabilities", [])
        added = set(new_capabilities) - set(old_capabilities)

        if not added:
            return True  # No new capabilities

        # Generate new steps for added capabilities
        new_steps = self._get_steps_for_capabilities(list(added))

        # Filter to only capability-specific steps
        existing_ids = {s.step_id for s in session.steps}
        skip_ids = {"welcome", "device_pairing", "notifications", "security"}
        new_steps = [
            s for s in new_steps if s.step_id not in existing_ids and s.step_id not in skip_ids
        ]

        if new_steps:
            # Insert before the security step
            security_idx = next(
                (i for i, s in enumerate(session.steps) if s.step_id == "security"),
                len(session.steps),
            )
            for i, step in enumerate(new_steps):
                step.order = security_idx + i
                session.steps.insert(security_idx + i, step)

            # Update order for remaining steps
            for i, step in enumerate(session.steps):
                step.order = i

        session.metadata["capabilities"] = new_capabilities
        return True
