"""
Tests for Onboarding Wizard.

Tests the guided setup experience for new users and devices.
"""

from __future__ import annotations

import pytest

from aragora.onboarding.wizard import (
    OnboardingSession,
    OnboardingStep,
    OnboardingWizard,
    StepStatus,
    WizardConfig,
)


# =============================================================================
# OnboardingStep Tests
# =============================================================================


class TestOnboardingStep:
    """Test OnboardingStep dataclass."""

    def test_step_creation(self):
        step = OnboardingStep(
            step_id="welcome",
            name="Welcome",
            description="Welcome to the app",
        )
        assert step.step_id == "welcome"
        assert step.name == "Welcome"
        assert step.required is True
        assert step.status == StepStatus.PENDING

    def test_step_with_fields(self):
        step = OnboardingStep(
            step_id="profile",
            name="Profile",
            description="Set up your profile",
            fields=[
                {"name": "username", "type": "text", "required": True},
                {"name": "email", "type": "email", "required": True},
            ],
        )
        assert len(step.fields) == 2

    def test_step_to_dict(self):
        step = OnboardingStep(
            step_id="test",
            name="Test Step",
            description="Description",
            order=5,
        )
        data = step.to_dict()
        assert data["step_id"] == "test"
        assert data["order"] == 5
        assert data["status"] == "pending"

    def test_step_from_dict(self):
        data = {
            "step_id": "test",
            "name": "Test",
            "description": "Desc",
            "status": "completed",
            "order": 3,
        }
        step = OnboardingStep.from_dict(data)
        assert step.step_id == "test"
        assert step.status == StepStatus.COMPLETED
        assert step.order == 3

    def test_step_roundtrip(self):
        original = OnboardingStep(
            step_id="roundtrip",
            name="Roundtrip Test",
            description="Testing serialization",
            fields=[{"name": "field1", "type": "text"}],
            help_text="Some help",
        )
        data = original.to_dict()
        restored = OnboardingStep.from_dict(data)
        assert restored.step_id == original.step_id
        assert restored.help_text == original.help_text
        assert len(restored.fields) == 1


class TestStepStatus:
    """Test StepStatus enum."""

    def test_all_statuses(self):
        statuses = [
            StepStatus.PENDING,
            StepStatus.ACTIVE,
            StepStatus.COMPLETED,
            StepStatus.SKIPPED,
            StepStatus.FAILED,
        ]
        assert len(statuses) == 5


# =============================================================================
# OnboardingSession Tests
# =============================================================================


class TestOnboardingSession:
    """Test OnboardingSession dataclass."""

    def test_session_creation(self):
        session = OnboardingSession(user_id="user-123")
        assert session.user_id == "user-123"
        assert session.session_id.startswith("onboard-")
        assert session.started_at > 0

    def test_session_with_device(self):
        session = OnboardingSession(
            user_id="user-1",
            device_id="device-1",
            tenant_id="tenant-1",
        )
        assert session.device_id == "device-1"
        assert session.tenant_id == "tenant-1"

    def test_session_is_complete_empty(self):
        session = OnboardingSession(user_id="user-1")
        assert session.is_complete is True  # No steps = complete

    def test_session_is_complete_with_steps(self):
        session = OnboardingSession(user_id="user-1")
        session.steps = [
            OnboardingStep(
                step_id="s1",
                name="Step 1",
                description="",
                status=StepStatus.COMPLETED,
            ),
            OnboardingStep(
                step_id="s2",
                name="Step 2",
                description="",
                status=StepStatus.PENDING,
            ),
        ]
        assert session.is_complete is False

    def test_session_is_complete_all_done(self):
        session = OnboardingSession(user_id="user-1")
        session.steps = [
            OnboardingStep(
                step_id="s1",
                name="Step 1",
                description="",
                status=StepStatus.COMPLETED,
            ),
            OnboardingStep(
                step_id="s2",
                name="Step 2",
                description="",
                status=StepStatus.SKIPPED,
            ),
        ]
        assert session.is_complete is True

    def test_session_progress(self):
        session = OnboardingSession(user_id="user-1")
        session.steps = [
            OnboardingStep(step_id="s1", name="1", description="", status=StepStatus.COMPLETED),
            OnboardingStep(step_id="s2", name="2", description="", status=StepStatus.PENDING),
            OnboardingStep(step_id="s3", name="3", description="", status=StepStatus.PENDING),
            OnboardingStep(step_id="s4", name="4", description="", status=StepStatus.PENDING),
        ]
        assert session.progress == 25.0  # 1 of 4 = 25%

    def test_session_get_step(self):
        session = OnboardingSession(user_id="user-1")
        step = OnboardingStep(step_id="test", name="Test", description="")
        session.steps = [step]
        found = session.get_step("test")
        assert found is step

    def test_session_get_step_not_found(self):
        session = OnboardingSession(user_id="user-1")
        found = session.get_step("nonexistent")
        assert found is None

    def test_session_to_dict(self):
        session = OnboardingSession(
            user_id="user-1",
            session_id="onboard-test",
        )
        session.steps = [OnboardingStep(step_id="s1", name="S1", description="")]
        data = session.to_dict()
        assert data["user_id"] == "user-1"
        assert "is_complete" in data
        assert "progress" in data
        assert len(data["steps"]) == 1

    def test_session_from_dict(self):
        data = {
            "user_id": "user-1",
            "session_id": "onboard-test",
            "steps": [{"step_id": "s1", "name": "S1", "description": ""}],
        }
        session = OnboardingSession.from_dict(data)
        assert session.user_id == "user-1"
        assert len(session.steps) == 1


# =============================================================================
# WizardConfig Tests
# =============================================================================


class TestWizardConfig:
    """Test WizardConfig defaults."""

    def test_defaults(self):
        config = WizardConfig()
        assert config.include_welcome is True
        assert config.include_device_setup is True
        assert config.include_provider_config is True
        assert config.include_channel_setup is True
        assert config.include_preferences is True
        assert config.include_permissions is True
        assert config.allow_skip_required is False
        assert config.auto_advance is True

    def test_custom_config(self):
        config = WizardConfig(
            include_welcome=False,
            require_sso=True,
            require_tenant=True,
        )
        assert config.include_welcome is False
        assert config.require_sso is True
        assert config.require_tenant is True


# =============================================================================
# OnboardingWizard Tests
# =============================================================================


class TestWizardInit:
    """Test OnboardingWizard initialization."""

    def test_default_init(self):
        wizard = OnboardingWizard()
        assert wizard.config.include_welcome is True

    def test_init_with_config(self):
        config = WizardConfig(include_welcome=False)
        wizard = OnboardingWizard(config=config)
        assert wizard.config.include_welcome is False


class TestWizardSession:
    """Test session management."""

    @pytest.fixture
    def wizard(self):
        return OnboardingWizard()

    @pytest.mark.asyncio
    async def test_start_session(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        assert session.user_id == "user-1"
        assert len(session.steps) > 0
        assert session.current_step == session.steps[0].step_id

    @pytest.mark.asyncio
    async def test_start_session_resumes_existing(self, wizard):
        session1 = await wizard.start_session(user_id="user-1")
        session2 = await wizard.start_session(user_id="user-1")
        assert session1.session_id == session2.session_id

    @pytest.mark.asyncio
    async def test_start_session_with_device(self, wizard):
        session = await wizard.start_session(
            user_id="user-1",
            device_id="device-1",
            tenant_id="tenant-1",
        )
        assert session.device_id == "device-1"
        assert session.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_start_session_custom_steps(self, wizard):
        custom_steps = [
            OnboardingStep(step_id="custom1", name="Custom 1", description="First"),
            OnboardingStep(step_id="custom2", name="Custom 2", description="Second"),
        ]
        session = await wizard.start_session(
            user_id="user-1",
            custom_steps=custom_steps,
        )
        assert len(session.steps) == 2
        assert session.steps[0].step_id == "custom1"

    @pytest.mark.asyncio
    async def test_get_session(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        found = await wizard.get_session(session.session_id)
        assert found is session

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, wizard):
        found = await wizard.get_session("nonexistent")
        assert found is None

    @pytest.mark.asyncio
    async def test_get_user_session(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        found = await wizard.get_user_session("user-1")
        assert found is session


class TestWizardStepCompletion:
    """Test step completion."""

    @pytest.fixture
    def wizard(self):
        config = WizardConfig(
            include_welcome=True,
            include_device_setup=False,
            include_provider_config=False,
            include_channel_setup=False,
            include_preferences=False,
            include_permissions=False,
        )
        return OnboardingWizard(config=config)

    @pytest.mark.asyncio
    async def test_complete_step(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        success, errors = await wizard.complete_step(
            session.session_id,
            "welcome",
            {},
        )
        assert success is True
        assert len(errors) == 0

        step = session.get_step("welcome")
        assert step.status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_complete_step_invalid_session(self, wizard):
        success, errors = await wizard.complete_step(
            "nonexistent",
            "welcome",
            {},
        )
        assert success is False
        assert "Session not found" in errors

    @pytest.mark.asyncio
    async def test_complete_step_invalid_step(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        success, errors = await wizard.complete_step(
            session.session_id,
            "nonexistent",
            {},
        )
        assert success is False
        assert "Step not found" in errors

    @pytest.mark.asyncio
    async def test_complete_step_already_completed(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        await wizard.complete_step(session.session_id, "welcome", {})
        success, errors = await wizard.complete_step(
            session.session_id,
            "welcome",
            {},
        )
        assert success is False
        assert "already completed" in errors[0]


class TestWizardValidation:
    """Test step validation."""

    @pytest.fixture
    def wizard(self):
        config = WizardConfig(
            include_welcome=False,
            include_device_setup=True,
            include_provider_config=False,
            include_channel_setup=False,
            include_preferences=False,
            include_permissions=False,
        )
        return OnboardingWizard(config=config)

    @pytest.mark.asyncio
    async def test_validate_required_field(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        # Device step requires device_name
        success, errors = await wizard.complete_step(
            session.session_id,
            "device",
            {},  # Missing device_name
        )
        assert success is False
        assert any("device_name" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_with_required_field(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        success, errors = await wizard.complete_step(
            session.session_id,
            "device",
            {"device_name": "My Laptop"},
        )
        assert success is True


class TestWizardSkip:
    """Test step skipping."""

    @pytest.fixture
    def wizard(self):
        return OnboardingWizard()

    @pytest.mark.asyncio
    async def test_skip_optional_step(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        # Welcome is optional
        skipped = await wizard.skip_step(session.session_id, "welcome")
        assert skipped is True

        step = session.get_step("welcome")
        assert step.status == StepStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_skip_required_step_fails(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        # Permissions step is required
        skipped = await wizard.skip_step(session.session_id, "permissions")
        assert skipped is False


class TestWizardNavigation:
    """Test step navigation."""

    @pytest.fixture
    def wizard(self):
        config = WizardConfig(
            include_welcome=True,
            include_device_setup=True,
            include_provider_config=False,
            include_channel_setup=False,
            include_preferences=False,
            include_permissions=False,
        )
        return OnboardingWizard(config=config)

    @pytest.mark.asyncio
    async def test_auto_advance(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        assert session.current_step == "welcome"

        await wizard.complete_step(session.session_id, "welcome", {})
        assert session.current_step == "device"

    @pytest.mark.asyncio
    async def test_go_back(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        await wizard.complete_step(session.session_id, "welcome", {})
        assert session.current_step == "device"

        prev = await wizard.go_back(session.session_id)
        assert prev is not None
        assert prev.step_id == "welcome"
        assert session.current_step == "welcome"

    @pytest.mark.asyncio
    async def test_go_back_at_start(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        prev = await wizard.go_back(session.session_id)
        assert prev is None

    @pytest.mark.asyncio
    async def test_get_current_step(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        current = await wizard.get_current_step(session.session_id)
        assert current is not None
        assert current.step_id == "welcome"


class TestWizardHandlers:
    """Test custom handlers and validators."""

    @pytest.fixture
    def wizard(self):
        config = WizardConfig(
            include_welcome=True,
            include_device_setup=False,
            include_provider_config=False,
            include_channel_setup=False,
            include_preferences=False,
            include_permissions=False,
        )
        return OnboardingWizard(config=config)

    @pytest.mark.asyncio
    async def test_custom_validator(self, wizard):
        def validator(step, data):
            if data.get("name") == "bad":
                return ["Invalid name"]
            return []

        wizard.register_validator("welcome", validator)
        session = await wizard.start_session(user_id="user-1")
        success, errors = await wizard.complete_step(
            session.session_id,
            "welcome",
            {"name": "bad"},
        )
        assert success is False
        assert "Invalid name" in errors

    @pytest.mark.asyncio
    async def test_custom_handler(self, wizard):
        handler_called = []

        async def handler(session, step, data):
            handler_called.append(data)

        wizard.register_handler("welcome", handler)
        session = await wizard.start_session(user_id="user-1")
        await wizard.complete_step(
            session.session_id,
            "welcome",
            {"key": "value"},
        )
        assert len(handler_called) == 1
        assert handler_called[0]["key"] == "value"


class TestWizardCancel:
    """Test session cancellation."""

    @pytest.fixture
    def wizard(self):
        return OnboardingWizard()

    @pytest.mark.asyncio
    async def test_cancel_session(self, wizard):
        session = await wizard.start_session(user_id="user-1")
        cancelled = await wizard.cancel_session(session.session_id)
        assert cancelled is True

        found = await wizard.get_session(session.session_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, wizard):
        cancelled = await wizard.cancel_session("nonexistent")
        assert cancelled is False


class TestWizardStats:
    """Test wizard statistics."""

    @pytest.fixture
    def wizard(self):
        return OnboardingWizard()

    @pytest.mark.asyncio
    async def test_get_stats(self, wizard):
        await wizard.start_session(user_id="user-1")
        await wizard.start_session(user_id="user-2")

        stats = await wizard.get_stats()
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 2
        assert stats["completed_sessions"] == 0
