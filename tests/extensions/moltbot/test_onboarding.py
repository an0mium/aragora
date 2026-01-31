"""
Tests for Moltbot OnboardingOrchestrator - User Journey and Activation Flows.

Tests flow management, session handling, step progression, and analytics.
"""

import asyncio
import pytest
from pathlib import Path

from aragora.extensions.moltbot import OnboardingOrchestrator, ChannelType
from aragora.extensions.moltbot.models import OnboardingStep


class TestFlowManagement:
    """Tests for onboarding flow management."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.mark.asyncio
    async def test_create_flow(self, orchestrator: OnboardingOrchestrator):
        """Test creating an onboarding flow."""
        flow = await orchestrator.create_flow(
            name="User Registration",
            description="New user registration flow",
        )

        assert flow is not None
        assert flow.id is not None
        assert flow.name == "User Registration"
        assert flow.description == "New user registration flow"
        assert flow.status == "draft"
        assert flow.steps == []

    @pytest.mark.asyncio
    async def test_create_flow_with_targeting(self, orchestrator: OnboardingOrchestrator):
        """Test creating flow with targeting."""
        flow = await orchestrator.create_flow(
            name="Mobile Onboarding",
            target_segment="mobile_users",
            channels=[ChannelType.PUSH, ChannelType.SMS],
        )

        assert flow.target_segment == "mobile_users"
        assert ChannelType.PUSH in flow.channels
        assert ChannelType.SMS in flow.channels

    @pytest.mark.asyncio
    async def test_get_flow(self, orchestrator: OnboardingOrchestrator):
        """Test getting a flow by ID."""
        created = await orchestrator.create_flow(name="Test Flow")

        flow = await orchestrator.get_flow(created.id)

        assert flow is not None
        assert flow.id == created.id
        assert flow.name == "Test Flow"

    @pytest.mark.asyncio
    async def test_get_nonexistent_flow(self, orchestrator: OnboardingOrchestrator):
        """Test getting nonexistent flow."""
        flow = await orchestrator.get_flow("nonexistent")
        assert flow is None

    @pytest.mark.asyncio
    async def test_list_flows(self, orchestrator: OnboardingOrchestrator):
        """Test listing flows."""
        await orchestrator.create_flow(name="Flow 1")
        await orchestrator.create_flow(name="Flow 2")
        await orchestrator.create_flow(name="Flow 3")

        flows = await orchestrator.list_flows()
        assert len(flows) == 3

    @pytest.mark.asyncio
    async def test_list_flows_by_status(self, orchestrator: OnboardingOrchestrator):
        """Test listing flows by status."""
        flow1 = await orchestrator.create_flow(name="Flow 1")
        flow2 = await orchestrator.create_flow(name="Flow 2")

        # Add step and activate flow1
        await orchestrator.add_step(flow1.id, "Step 1", "info")
        await orchestrator.activate_flow(flow1.id)

        draft_flows = await orchestrator.list_flows(status="draft")
        active_flows = await orchestrator.list_flows(status="active")

        assert len(draft_flows) == 1
        assert len(active_flows) == 1

    @pytest.mark.asyncio
    async def test_list_flows_by_segment(self, orchestrator: OnboardingOrchestrator):
        """Test listing flows by target segment."""
        await orchestrator.create_flow(name="Enterprise", target_segment="enterprise")
        await orchestrator.create_flow(name="SMB", target_segment="smb")
        await orchestrator.create_flow(name="Consumer", target_segment="consumer")

        enterprise = await orchestrator.list_flows(target_segment="enterprise")
        assert len(enterprise) == 1
        assert enterprise[0].name == "Enterprise"


class TestStepManagement:
    """Tests for onboarding step management."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.mark.asyncio
    async def test_add_step(self, orchestrator: OnboardingOrchestrator):
        """Test adding a step to a flow."""
        flow = await orchestrator.create_flow(name="Test Flow")

        step = await orchestrator.add_step(
            flow_id=flow.id,
            name="Welcome",
            step_type="info",
            content={"message": "Welcome to our service!"},
        )

        assert step is not None
        assert step.id is not None
        assert step.name == "Welcome"
        assert step.type == "info"
        assert step.order == 0

    @pytest.mark.asyncio
    async def test_add_multiple_steps(self, orchestrator: OnboardingOrchestrator):
        """Test adding multiple steps."""
        flow = await orchestrator.create_flow(name="Multi-step Flow")

        step1 = await orchestrator.add_step(flow.id, "Step 1", "info")
        step2 = await orchestrator.add_step(flow.id, "Step 2", "input")
        step3 = await orchestrator.add_step(flow.id, "Step 3", "verification")

        updated_flow = await orchestrator.get_flow(flow.id)

        assert len(updated_flow.steps) == 3
        assert updated_flow.steps[0].order == 0
        assert updated_flow.steps[1].order == 1
        assert updated_flow.steps[2].order == 2

    @pytest.mark.asyncio
    async def test_add_step_with_validation(self, orchestrator: OnboardingOrchestrator):
        """Test adding step with validation rules."""
        flow = await orchestrator.create_flow(name="Validated Flow")

        step = await orchestrator.add_step(
            flow_id=flow.id,
            name="Email Input",
            step_type="input",
            validation={"email": ["required", "email"]},
        )

        assert step.validation is not None
        assert "email" in step.validation

    @pytest.mark.asyncio
    async def test_add_step_with_branching(self, orchestrator: OnboardingOrchestrator):
        """Test adding step with branching conditions."""
        flow = await orchestrator.create_flow(name="Branching Flow")

        step = await orchestrator.add_step(
            flow_id=flow.id,
            name="Plan Selection",
            step_type="decision",
            branch_conditions={
                "plan:basic": "basic-setup",
                "plan:premium": "premium-setup",
            },
        )

        assert len(step.branch_conditions) == 2
        assert step.branch_conditions["plan:basic"] == "basic-setup"

    @pytest.mark.asyncio
    async def test_add_step_to_nonexistent_flow(self, orchestrator: OnboardingOrchestrator):
        """Test adding step to nonexistent flow."""
        step = await orchestrator.add_step("nonexistent", "Step", "info")
        assert step is None


class TestFlowActivation:
    """Tests for flow activation and archival."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.mark.asyncio
    async def test_activate_flow(self, orchestrator: OnboardingOrchestrator):
        """Test activating a flow."""
        flow = await orchestrator.create_flow(name="To Activate")
        await orchestrator.add_step(flow.id, "Step 1", "info")

        activated = await orchestrator.activate_flow(flow.id)

        assert activated is not None
        assert activated.status == "active"

    @pytest.mark.asyncio
    async def test_activate_empty_flow_fails(self, orchestrator: OnboardingOrchestrator):
        """Test activating empty flow fails."""
        flow = await orchestrator.create_flow(name="Empty Flow")

        with pytest.raises(ValueError, match="no steps"):
            await orchestrator.activate_flow(flow.id)

    @pytest.mark.asyncio
    async def test_activate_nonexistent_flow(self, orchestrator: OnboardingOrchestrator):
        """Test activating nonexistent flow."""
        result = await orchestrator.activate_flow("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_archive_flow(self, orchestrator: OnboardingOrchestrator):
        """Test archiving a flow."""
        flow = await orchestrator.create_flow(name="To Archive")

        archived = await orchestrator.archive_flow(flow.id)

        assert archived is not None
        assert archived.status == "archived"


class TestSessionManagement:
    """Tests for onboarding session management."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.fixture
    async def active_flow(self, orchestrator: OnboardingOrchestrator):
        """Create an active flow with steps."""
        flow = await orchestrator.create_flow(name="Active Flow")
        await orchestrator.add_step(flow.id, "Welcome", "info")
        await orchestrator.add_step(flow.id, "Email", "input", validation={"email": ["required"]})
        await orchestrator.add_step(flow.id, "Verify", "verification")
        await orchestrator.activate_flow(flow.id)
        return await orchestrator.get_flow(flow.id)

    @pytest.mark.asyncio
    async def test_start_session(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test starting an onboarding session."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        assert session is not None
        assert session.id is not None
        assert session.flow_id == active_flow.id
        assert session.user_id == "user-1"
        assert session.status == "in_progress"
        assert session.current_step == active_flow.steps[0].id

    @pytest.mark.asyncio
    async def test_start_session_with_initial_data(
        self, orchestrator: OnboardingOrchestrator, active_flow
    ):
        """Test starting session with initial data."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
            initial_data={"referrer": "email_campaign"},
        )

        assert session.collected_data["referrer"] == "email_campaign"

    @pytest.mark.asyncio
    async def test_start_session_inactive_flow_fails(self, orchestrator: OnboardingOrchestrator):
        """Test starting session on inactive flow fails."""
        flow = await orchestrator.create_flow(name="Draft Flow")
        await orchestrator.add_step(flow.id, "Step", "info")

        with pytest.raises(ValueError, match="not active"):
            await orchestrator.start_session(
                flow_id=flow.id,
                user_id="user-1",
                channel_id="channel-1",
            )

    @pytest.mark.asyncio
    async def test_start_session_nonexistent_flow_fails(self, orchestrator: OnboardingOrchestrator):
        """Test starting session on nonexistent flow fails."""
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.start_session(
                flow_id="nonexistent",
                user_id="user-1",
                channel_id="channel-1",
            )

    @pytest.mark.asyncio
    async def test_start_session_increments_count(
        self, orchestrator: OnboardingOrchestrator, active_flow
    ):
        """Test starting session increments started count."""
        initial_count = active_flow.started_count

        await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        updated_flow = await orchestrator.get_flow(active_flow.id)
        assert updated_flow.started_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_get_session(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test getting a session by ID."""
        created = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        session = await orchestrator.get_session(created.id)

        assert session is not None
        assert session.id == created.id

    @pytest.mark.asyncio
    async def test_list_sessions(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test listing sessions."""
        for i in range(3):
            await orchestrator.start_session(
                flow_id=active_flow.id,
                user_id=f"user-{i}",
                channel_id="channel-1",
            )

        sessions = await orchestrator.list_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_by_flow(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test listing sessions by flow."""
        await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        sessions = await orchestrator.list_sessions(flow_id=active_flow.id)
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_get_current_step(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test getting current step."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        step, context = await orchestrator.get_current_step(session.id)

        assert step is not None
        assert step.name == "Welcome"
        assert context["session_id"] == session.id
        assert context["user_id"] == "user-1"


class TestStepProgression:
    """Tests for step progression."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.fixture
    async def active_flow(self, orchestrator: OnboardingOrchestrator):
        """Create an active flow with steps."""
        flow = await orchestrator.create_flow(name="Progressive Flow")
        await orchestrator.add_step(flow.id, "Step 1", "info")
        await orchestrator.add_step(
            flow.id, "Step 2", "input", validation={"email": ["required", "email"]}
        )
        await orchestrator.add_step(flow.id, "Step 3", "info")
        await orchestrator.activate_flow(flow.id)
        return await orchestrator.get_flow(flow.id)

    @pytest.mark.asyncio
    async def test_submit_step_advances(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test submitting step advances to next."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await orchestrator.submit_step(session.id, {})

        assert result["success"] is True
        assert result["completed_step"] == active_flow.steps[0].id
        assert result["next_step"]["id"] == active_flow.steps[1].id
        assert result["progress"] > 0

    @pytest.mark.asyncio
    async def test_submit_step_validation_failure(
        self, orchestrator: OnboardingOrchestrator, active_flow
    ):
        """Test submit step with validation failure."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        # Advance to step 2 (email input)
        await orchestrator.submit_step(session.id, {})

        # Submit invalid email
        result = await orchestrator.submit_step(session.id, {"email": "not-an-email"})

        assert result["success"] is False
        assert "validation_errors" in result

    @pytest.mark.asyncio
    async def test_submit_step_stores_data(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test submit step stores collected data."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        # Advance to step 2
        await orchestrator.submit_step(session.id, {})

        # Submit valid email
        await orchestrator.submit_step(session.id, {"email": "user@example.com"})

        updated = await orchestrator.get_session(session.id)
        assert updated.collected_data["email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_submit_step_completes_flow(
        self, orchestrator: OnboardingOrchestrator, active_flow
    ):
        """Test submitting final step completes flow."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        # Complete all steps
        await orchestrator.submit_step(session.id, {})  # Step 1
        await orchestrator.submit_step(session.id, {"email": "user@example.com"})  # Step 2
        result = await orchestrator.submit_step(session.id, {})  # Step 3

        assert result["success"] is True
        assert result["completed"] is True
        assert "collected_data" in result

        updated = await orchestrator.get_session(session.id)
        assert updated.status == "completed"

    @pytest.mark.asyncio
    async def test_submit_step_nonexistent_session(self, orchestrator: OnboardingOrchestrator):
        """Test submit step for nonexistent session."""
        result = await orchestrator.submit_step("nonexistent", {})

        assert result["success"] is False
        assert "not found" in result["error"]


class TestSessionControls:
    """Tests for session control operations."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.fixture
    async def active_flow(self, orchestrator: OnboardingOrchestrator):
        """Create an active flow with steps."""
        flow = await orchestrator.create_flow(name="Control Flow")
        await orchestrator.add_step(flow.id, "Step 1", "info")
        await orchestrator.add_step(flow.id, "Step 2", "input")
        await orchestrator.activate_flow(flow.id)
        return await orchestrator.get_flow(flow.id)

    @pytest.mark.asyncio
    async def test_abandon_session(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test abandoning a session."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        abandoned = await orchestrator.abandon_session(session.id, reason="user_closed")

        assert abandoned is not None
        assert abandoned.status == "abandoned"
        assert abandoned.metadata["abandon_reason"] == "user_closed"

    @pytest.mark.asyncio
    async def test_abandon_increments_count(
        self, orchestrator: OnboardingOrchestrator, active_flow
    ):
        """Test abandoning increments abandon count."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        await orchestrator.abandon_session(session.id)

        updated_flow = await orchestrator.get_flow(active_flow.id)
        assert updated_flow.abandoned_count == 1

    @pytest.mark.asyncio
    async def test_pause_session(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test pausing a session."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        paused = await orchestrator.pause_session(session.id)

        assert paused is not None
        assert paused.status == "paused"

    @pytest.mark.asyncio
    async def test_resume_session(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test resuming a paused session."""
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        await orchestrator.pause_session(session.id)
        resumed = await orchestrator.resume_session(session.id)

        assert resumed is not None
        assert resumed.status == "in_progress"


class TestValidation:
    """Tests for field validation."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    def test_email_validator(self, orchestrator: OnboardingOrchestrator):
        """Test email validator."""
        validator = orchestrator._validators["email"]

        assert validator("user@example.com") is True
        assert validator("test@domain.org") is True
        assert validator("invalid-email") is False
        assert validator("missing-at.com") is False

    def test_phone_validator(self, orchestrator: OnboardingOrchestrator):
        """Test phone validator."""
        validator = orchestrator._validators["phone"]

        assert validator("1234567890") is True
        assert validator("+1-234-567-8901") is True
        assert validator("123") is False
        assert validator("abc") is False

    def test_required_validator(self, orchestrator: OnboardingOrchestrator):
        """Test required validator."""
        validator = orchestrator._validators["required"]

        assert validator("value") is True
        assert validator("") is False
        assert validator(None) is False
        assert validator(0) is False  # 0 is falsy

    @pytest.mark.asyncio
    async def test_register_custom_validator(self, orchestrator: OnboardingOrchestrator):
        """Test registering custom validator."""

        def age_validator(value):
            return isinstance(value, int) and 18 <= value <= 120

        orchestrator.register_validator("age", age_validator)

        assert orchestrator._validators["age"](25) is True
        assert orchestrator._validators["age"](150) is False
        assert orchestrator._validators["age"](15) is False


class TestBranchingLogic:
    """Tests for branching logic in flows."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.mark.asyncio
    async def test_branch_conditions(self, orchestrator: OnboardingOrchestrator):
        """Test branching based on conditions."""
        flow = await orchestrator.create_flow(name="Branching Flow")

        # Decision step with branches
        await orchestrator.add_step(
            flow.id,
            "Plan Selection",
            "decision",
            branch_conditions={
                "plan:basic": "basic-step-id",
                "plan:premium": "premium-step-id",
            },
        )

        # Add target steps
        basic_step = await orchestrator.add_step(flow.id, "Basic Setup", "info")
        premium_step = await orchestrator.add_step(flow.id, "Premium Setup", "info")

        # Update branch conditions with actual IDs
        updated_flow = await orchestrator.get_flow(flow.id)
        updated_flow.steps[0].branch_conditions = {
            "plan:basic": basic_step.id,
            "plan:premium": premium_step.id,
        }

        await orchestrator.activate_flow(flow.id)

        # Start session and submit with "basic" choice
        session = await orchestrator.start_session(
            flow_id=flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await orchestrator.submit_step(session.id, {"plan": "basic"})

        assert result["success"] is True
        assert result["next_step"]["id"] == basic_step.id


class TestAnalytics:
    """Tests for onboarding analytics."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an orchestrator for testing."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.fixture
    async def active_flow(self, orchestrator: OnboardingOrchestrator):
        """Create an active flow with steps."""
        flow = await orchestrator.create_flow(name="Analytics Flow")
        await orchestrator.add_step(flow.id, "Step 1", "info")
        await orchestrator.add_step(flow.id, "Step 2", "info")
        await orchestrator.activate_flow(flow.id)
        return await orchestrator.get_flow(flow.id)

    @pytest.mark.asyncio
    async def test_get_flow_stats(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test getting flow statistics."""
        # Start and complete some sessions
        for i in range(3):
            session = await orchestrator.start_session(
                flow_id=active_flow.id,
                user_id=f"user-{i}",
                channel_id="channel-1",
            )
            await orchestrator.submit_step(session.id, {})
            await orchestrator.submit_step(session.id, {})

        # Abandon one
        session = await orchestrator.start_session(
            flow_id=active_flow.id,
            user_id="user-abandoned",
            channel_id="channel-1",
        )
        await orchestrator.abandon_session(session.id)

        stats = await orchestrator.get_flow_stats(active_flow.id)

        assert stats["flow_id"] == active_flow.id
        assert stats["started"] == 4
        assert stats["completed"] == 3
        assert stats["abandoned"] == 1
        assert stats["completion_rate"] == 0.75

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, orchestrator: OnboardingOrchestrator):
        """Test getting stats with no data."""
        stats = await orchestrator.get_stats()

        assert stats["flows_total"] == 0
        assert stats["flows_active"] == 0
        assert stats["sessions_total"] == 0
        assert stats["total_started"] == 0
        assert stats["total_completed"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, orchestrator: OnboardingOrchestrator, active_flow):
        """Test getting overall stats."""
        # Start sessions
        for i in range(5):
            await orchestrator.start_session(
                flow_id=active_flow.id,
                user_id=f"user-{i}",
                channel_id="channel-1",
            )

        stats = await orchestrator.get_stats()

        assert stats["flows_total"] == 1
        assert stats["flows_active"] == 1
        assert stats["sessions_total"] == 5
        assert stats["sessions_in_progress"] == 5
        assert stats["total_started"] == 5
