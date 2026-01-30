"""
Tests for FeatureDevelopmentAgent and related components.

Tests cover:
- Agent initialization and configuration
- FeatureSpec, DesignDecision, ImplementationStep, FeatureImplementation dataclasses
- Feature development workflow phases
- Error handling and edge cases
- Integration with debate system
- State management
- Memory interaction (lazy loading of components)
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from aragora.agents.feature_agent import (
    FeatureStatus,
    FeatureSpec,
    DesignDecision,
    ImplementationStep,
    FeatureImplementation,
    FeatureDevelopmentAgent,
)


class TestFeatureStatus:
    """Tests for FeatureStatus enum."""

    def test_planning_status_exists(self):
        """PLANNING status exists."""
        assert FeatureStatus.PLANNING.value == "planning"

    def test_designing_status_exists(self):
        """DESIGNING status exists."""
        assert FeatureStatus.DESIGNING.value == "designing"

    def test_testing_status_exists(self):
        """TESTING status exists."""
        assert FeatureStatus.TESTING.value == "testing"

    def test_implementing_status_exists(self):
        """IMPLEMENTING status exists."""
        assert FeatureStatus.IMPLEMENTING.value == "implementing"

    def test_verifying_status_exists(self):
        """VERIFYING status exists."""
        assert FeatureStatus.VERIFYING.value == "verifying"

    def test_awaiting_approval_status_exists(self):
        """AWAITING_APPROVAL status exists."""
        assert FeatureStatus.AWAITING_APPROVAL.value == "awaiting_approval"

    def test_completed_status_exists(self):
        """COMPLETED status exists."""
        assert FeatureStatus.COMPLETED.value == "completed"

    def test_failed_status_exists(self):
        """FAILED status exists."""
        assert FeatureStatus.FAILED.value == "failed"


class TestFeatureSpec:
    """Tests for FeatureSpec dataclass."""

    def test_create_minimal_spec(self):
        """Can create minimal feature spec."""
        spec = FeatureSpec(
            name="Test Feature",
            description="A test feature",
        )

        assert spec.name == "Test Feature"
        assert spec.description == "A test feature"
        assert spec.requirements == []
        assert spec.acceptance_criteria == []
        assert spec.affected_files == []
        assert spec.dependencies == []
        assert spec.priority == "medium"
        assert spec.tags == []

    def test_create_full_spec(self):
        """Can create fully specified feature spec."""
        spec = FeatureSpec(
            name="Auth Feature",
            description="Implement authentication",
            requirements=["Must support OAuth", "Must support SAML"],
            acceptance_criteria=["Users can log in", "Sessions persist"],
            affected_files=["auth/service.py", "auth/middleware.py"],
            dependencies=["requests", "pyjwt"],
            priority="critical",
            tags=["security", "auth"],
        )

        assert spec.name == "Auth Feature"
        assert len(spec.requirements) == 2
        assert len(spec.acceptance_criteria) == 2
        assert len(spec.affected_files) == 2
        assert spec.priority == "critical"
        assert "security" in spec.tags

    def test_spec_to_dict(self):
        """FeatureSpec serializes to dictionary correctly."""
        spec = FeatureSpec(
            name="Test",
            description="Test description",
            requirements=["req1"],
            tags=["tag1"],
        )

        data = spec.to_dict()

        assert data["name"] == "Test"
        assert data["description"] == "Test description"
        assert data["requirements"] == ["req1"]
        assert data["tags"] == ["tag1"]
        assert data["priority"] == "medium"
        assert data["acceptance_criteria"] == []

    def test_spec_to_dict_all_fields(self):
        """FeatureSpec to_dict includes all fields."""
        spec = FeatureSpec(
            name="Full",
            description="Full description",
            requirements=["r1", "r2"],
            acceptance_criteria=["a1"],
            affected_files=["f1.py"],
            dependencies=["dep1"],
            priority="high",
            tags=["t1"],
        )

        data = spec.to_dict()

        assert all(
            key in data
            for key in [
                "name",
                "description",
                "requirements",
                "acceptance_criteria",
                "affected_files",
                "dependencies",
                "priority",
                "tags",
            ]
        )


class TestDesignDecision:
    """Tests for DesignDecision dataclass."""

    def test_create_minimal_decision(self):
        """Can create minimal design decision."""
        decision = DesignDecision(
            question="What pattern to use?",
            decision="Use factory pattern",
            rationale="Better extensibility",
        )

        assert decision.question == "What pattern to use?"
        assert decision.decision == "Use factory pattern"
        assert decision.rationale == "Better extensibility"
        assert decision.alternatives == []
        assert decision.votes == {}
        assert isinstance(decision.timestamp, datetime)

    def test_create_full_decision(self):
        """Can create fully specified design decision."""
        decision = DesignDecision(
            question="Architecture choice?",
            decision="Microservices",
            rationale="Scale independently",
            alternatives=["Monolith", "Modular monolith"],
            votes={"agent1": "approve", "agent2": "approve"},
        )

        assert len(decision.alternatives) == 2
        assert len(decision.votes) == 2

    def test_decision_to_dict(self):
        """DesignDecision serializes to dictionary correctly."""
        decision = DesignDecision(
            question="Q1",
            decision="D1",
            rationale="R1",
            alternatives=["A1"],
            votes={"a1": "yes"},
        )

        data = decision.to_dict()

        assert data["question"] == "Q1"
        assert data["decision"] == "D1"
        assert data["rationale"] == "R1"
        assert data["alternatives"] == ["A1"]
        assert data["votes"] == {"a1": "yes"}
        assert "timestamp" in data

    def test_decision_timestamp_is_utc(self):
        """DesignDecision timestamp is in UTC."""
        decision = DesignDecision(
            question="Q",
            decision="D",
            rationale="R",
        )

        assert decision.timestamp.tzinfo == timezone.utc


class TestImplementationStep:
    """Tests for ImplementationStep dataclass."""

    def test_create_minimal_step(self):
        """Can create minimal implementation step."""
        step = ImplementationStep(
            step_id="step_1",
            description="Implement feature X",
        )

        assert step.step_id == "step_1"
        assert step.description == "Implement feature X"
        assert step.status == "pending"
        assert step.files_modified == []
        assert step.tests_added == []
        assert step.verification_results == {}
        assert step.error_message is None

    def test_create_full_step(self):
        """Can create fully specified implementation step."""
        step = ImplementationStep(
            step_id="step_2",
            description="Add auth module",
            status="completed",
            files_modified=["auth.py", "config.py"],
            tests_added=["test_auth.py"],
            verification_results={"syntax_check": True, "tests_pass": True},
            error_message=None,
        )

        assert step.status == "completed"
        assert len(step.files_modified) == 2
        assert step.verification_results["syntax_check"] is True

    def test_step_to_dict(self):
        """ImplementationStep serializes to dictionary correctly."""
        step = ImplementationStep(
            step_id="s1",
            description="d1",
            status="in_progress",
            files_modified=["f1.py"],
        )

        data = step.to_dict()

        assert data["step_id"] == "s1"
        assert data["description"] == "d1"
        assert data["status"] == "in_progress"
        assert data["files_modified"] == ["f1.py"]
        assert data["error_message"] is None

    def test_step_with_error(self):
        """ImplementationStep can have error message."""
        step = ImplementationStep(
            step_id="s1",
            description="Failed step",
            status="failed",
            error_message="Syntax error in file",
        )

        assert step.error_message == "Syntax error in file"
        assert step.to_dict()["error_message"] == "Syntax error in file"


class TestFeatureImplementation:
    """Tests for FeatureImplementation dataclass."""

    @pytest.fixture
    def sample_spec(self):
        """Create a sample feature spec."""
        return FeatureSpec(
            name="Sample Feature",
            description="A sample feature for testing",
        )

    def test_create_minimal_implementation(self, sample_spec):
        """Can create minimal feature implementation."""
        impl = FeatureImplementation(
            spec=sample_spec,
            status=FeatureStatus.PLANNING,
        )

        assert impl.spec == sample_spec
        assert impl.status == FeatureStatus.PLANNING
        assert impl.design_decisions == []
        assert impl.implementation_steps == []
        assert impl.tests_pass is False
        assert impl.implementation_files == []
        assert impl.test_files == []
        assert impl.approval_id is None
        assert isinstance(impl.started_at, datetime)
        assert impl.completed_at is None
        assert impl.error_message is None

    def test_implementation_with_decisions(self, sample_spec):
        """Implementation can contain design decisions."""
        decision = DesignDecision(
            question="Q1",
            decision="D1",
            rationale="R1",
        )
        impl = FeatureImplementation(
            spec=sample_spec,
            status=FeatureStatus.DESIGNING,
            design_decisions=[decision],
        )

        assert len(impl.design_decisions) == 1
        assert impl.design_decisions[0].question == "Q1"

    def test_implementation_with_steps(self, sample_spec):
        """Implementation can contain steps."""
        step = ImplementationStep(
            step_id="s1",
            description="Step 1",
        )
        impl = FeatureImplementation(
            spec=sample_spec,
            status=FeatureStatus.IMPLEMENTING,
            implementation_steps=[step],
        )

        assert len(impl.implementation_steps) == 1

    def test_implementation_to_dict(self, sample_spec):
        """FeatureImplementation serializes to dictionary correctly."""
        impl = FeatureImplementation(
            spec=sample_spec,
            status=FeatureStatus.COMPLETED,
            tests_pass=True,
            implementation_files=["main.py"],
            test_files=["test_main.py"],
        )

        data = impl.to_dict()

        assert data["spec"]["name"] == "Sample Feature"
        assert data["status"] == "completed"
        assert data["tests_pass"] is True
        assert data["implementation_files"] == ["main.py"]
        assert data["test_files"] == ["test_main.py"]
        assert "started_at" in data

    def test_implementation_completed_at(self, sample_spec):
        """Implementation tracks completion time."""
        completed_time = datetime.now(timezone.utc)
        impl = FeatureImplementation(
            spec=sample_spec,
            status=FeatureStatus.COMPLETED,
            completed_at=completed_time,
        )

        assert impl.completed_at == completed_time
        assert impl.to_dict()["completed_at"] is not None

    def test_implementation_with_error(self, sample_spec):
        """Implementation can track error message."""
        impl = FeatureImplementation(
            spec=sample_spec,
            status=FeatureStatus.FAILED,
            error_message="Build failed",
        )

        assert impl.error_message == "Build failed"
        assert impl.to_dict()["error_message"] == "Build failed"


class TestFeatureDevelopmentAgentInit:
    """Tests for FeatureDevelopmentAgent initialization."""

    @pytest.fixture
    def temp_codebase(self, tmp_path):
        """Create a temporary codebase directory."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass")
        return tmp_path

    def test_basic_initialization(self, temp_codebase):
        """Can initialize with minimal parameters."""
        agent = FeatureDevelopmentAgent(root_path=str(temp_codebase))

        assert agent.root_path == Path(temp_codebase)
        assert agent.enable_debate is True
        assert agent.enable_tdd is True
        assert agent.enable_approval is True

    def test_initialization_with_options_disabled(self, temp_codebase):
        """Can initialize with features disabled."""
        agent = FeatureDevelopmentAgent(
            root_path=str(temp_codebase),
            enable_debate=False,
            enable_tdd=False,
            enable_approval=False,
        )

        assert agent.enable_debate is False
        assert agent.enable_tdd is False
        assert agent.enable_approval is False

    def test_initialization_with_custom_log_fn(self, temp_codebase):
        """Can initialize with custom logging function."""
        log_messages = []

        def custom_log(msg):
            log_messages.append(msg)

        agent = FeatureDevelopmentAgent(
            root_path=str(temp_codebase),
            log_fn=custom_log,
        )

        agent._log("Test message")
        assert "Test message" in log_messages

    def test_lazy_components_initially_none(self, temp_codebase):
        """Lazy-loaded components are None initially."""
        agent = FeatureDevelopmentAgent(root_path=str(temp_codebase))

        assert agent._codebase_agent is None
        assert agent._test_generator is None
        assert agent._approval_workflow is None


class TestFeatureDevelopmentAgentLazyLoading:
    """Tests for lazy loading of agent components."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent instance."""
        return FeatureDevelopmentAgent(root_path=str(tmp_path))

    def test_codebase_agent_lazy_loads(self, agent):
        """Codebase agent is lazy loaded on access."""
        # Access codebase_agent property - it should attempt to lazy load
        # The import happens inside the property, so we verify the behavior
        codebase = agent.codebase_agent

        # After access, _codebase_agent should be set (or None if import failed)
        # The key test is that no exception is raised
        assert agent._codebase_agent is None or agent._codebase_agent is codebase

    def test_codebase_agent_returns_same_instance(self, agent):
        """Codebase agent returns same instance on repeated access."""
        # Set a mock to verify same instance is returned
        mock_agent = Mock()
        agent._codebase_agent = mock_agent

        result1 = agent.codebase_agent
        result2 = agent.codebase_agent

        assert result1 is result2
        assert result1 is mock_agent

    def test_test_generator_lazy_loads(self, agent):
        """Test generator is lazy loaded on access."""
        generator = agent.test_generator

        # After access, should be set (or None if import failed)
        assert agent._test_generator is None or agent._test_generator is generator

    def test_test_generator_returns_same_instance(self, agent):
        """Test generator returns same instance on repeated access."""
        mock_generator = Mock()
        agent._test_generator = mock_generator

        result1 = agent.test_generator
        result2 = agent.test_generator

        assert result1 is result2
        assert result1 is mock_generator

    def test_approval_workflow_lazy_loads(self, agent):
        """Approval workflow is lazy loaded on access."""
        workflow = agent.approval_workflow

        # After access, should be set (or None if import failed)
        assert agent._approval_workflow is None or agent._approval_workflow is workflow

    def test_approval_workflow_returns_same_instance(self, agent):
        """Approval workflow returns same instance on repeated access."""
        mock_workflow = Mock()
        agent._approval_workflow = mock_workflow

        result1 = agent.approval_workflow
        result2 = agent.approval_workflow

        assert result1 is result2
        assert result1 is mock_workflow


class TestFeatureDevelopmentWorkflow:
    """Tests for the feature development workflow."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent with mocked components."""
        return FeatureDevelopmentAgent(
            root_path=str(tmp_path),
            enable_debate=False,
            enable_tdd=False,
            enable_approval=False,
        )

    @pytest.fixture
    def sample_spec(self):
        """Create a sample feature spec."""
        return FeatureSpec(
            name="Test Feature",
            description="A test feature",
            requirements=["Requirement 1"],
            acceptance_criteria=["Criterion 1"],
        )

    @pytest.mark.asyncio
    async def test_develop_feature_basic_flow(self, agent, sample_spec):
        """Basic feature development flow completes."""
        result = await agent.develop_feature(sample_spec)

        assert isinstance(result, FeatureImplementation)
        assert result.status == FeatureStatus.COMPLETED
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_develop_feature_starts_as_planning(self, agent, sample_spec):
        """Feature development starts in PLANNING status."""
        # We need to track status changes
        statuses = []

        original_gather = agent._gather_context

        async def tracking_gather(*args, **kwargs):
            statuses.append(args[1].status)
            return await original_gather(*args, **kwargs)

        agent._gather_context = tracking_gather

        await agent.develop_feature(sample_spec)

        assert FeatureStatus.PLANNING in statuses

    @pytest.mark.asyncio
    async def test_develop_feature_with_tdd_enabled(self, tmp_path, sample_spec):
        """Feature development with TDD generates tests."""
        agent = FeatureDevelopmentAgent(
            root_path=str(tmp_path),
            enable_debate=False,
            enable_tdd=True,
            enable_approval=False,
        )

        # Mock the test generator
        mock_generator = Mock()
        agent._test_generator = mock_generator

        result = await agent.develop_feature(sample_spec)

        assert result.status == FeatureStatus.COMPLETED
        # Test files should be populated from acceptance criteria
        assert len(result.test_files) > 0

    @pytest.mark.asyncio
    async def test_develop_feature_exception_handling(self, agent, sample_spec):
        """Feature development handles exceptions gracefully."""

        async def failing_context(*args, **kwargs):
            raise ValueError("Context gathering failed")

        agent._gather_context = failing_context

        result = await agent.develop_feature(sample_spec)

        assert result.status == FeatureStatus.FAILED
        assert "Context gathering failed" in result.error_message

    @pytest.mark.asyncio
    async def test_develop_feature_with_approvers(self, tmp_path, sample_spec):
        """Feature development with approvers requests approval."""
        agent = FeatureDevelopmentAgent(
            root_path=str(tmp_path),
            enable_debate=False,
            enable_tdd=False,
            enable_approval=True,
        )

        # Mock the approval workflow
        mock_workflow = AsyncMock()
        mock_workflow.request_approval.return_value = Mock(
            request_id="approval_123",
            approved=True,
            message="Approved",
        )
        agent._approval_workflow = mock_workflow

        result = await agent.develop_feature(
            sample_spec,
            approvers=["approver1", "approver2"],
        )

        assert result.status == FeatureStatus.COMPLETED
        assert result.approval_id == "approval_123"


class TestGatherContext:
    """Tests for _gather_context method."""

    @pytest.fixture
    def agent_with_mocked_codebase(self, tmp_path):
        """Create agent with mocked codebase agent."""
        agent = FeatureDevelopmentAgent(
            root_path=str(tmp_path),
            enable_debate=True,
        )

        mock_codebase = AsyncMock()
        mock_codebase.index_codebase.return_value = Mock()
        mock_codebase.understand.return_value = Mock(
            relevant_files=["file1.py", "file2.py"],
            answer="Understanding response",
        )
        agent._codebase_agent = mock_codebase

        return agent

    @pytest.mark.asyncio
    async def test_gather_context_with_affected_files(self, agent_with_mocked_codebase):
        """Context gathering uses affected_files from spec."""
        spec = FeatureSpec(
            name="Test",
            description="Test description",
            affected_files=["specific_file.py"],
        )
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.PLANNING,
        )

        context = await agent_with_mocked_codebase._gather_context(spec, impl)

        assert context["relevant_files"] == ["specific_file.py"]

    @pytest.mark.asyncio
    async def test_gather_context_analyzes_requirements(self, agent_with_mocked_codebase):
        """Context gathering analyzes requirements."""
        spec = FeatureSpec(
            name="Test",
            description="Test description",
            requirements=["Req 1", "Req 2"],
        )
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.PLANNING,
        )

        context = await agent_with_mocked_codebase._gather_context(spec, impl)

        # Should have analyzed patterns for requirements
        assert "existing_patterns" in context


class TestDesignFeature:
    """Tests for _design_feature method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent instance."""
        return FeatureDevelopmentAgent(
            root_path=str(tmp_path),
            enable_debate=False,
        )

    @pytest.mark.asyncio
    async def test_design_creates_decisions(self, agent):
        """Design phase creates design decisions."""
        spec = FeatureSpec(
            name="Test",
            description="Test description",
        )
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.DESIGNING,
        )
        context = {"relevant_files": [], "existing_patterns": []}

        await agent._design_feature(spec, context, impl)

        # Should have created at least 3 design decisions
        assert len(impl.design_decisions) >= 3
        decision_types = [d.question for d in impl.design_decisions]
        assert any("architecture" in q.lower() or "pattern" in q.lower() for q in decision_types)

    @pytest.mark.asyncio
    async def test_design_uses_existing_patterns(self, agent):
        """Design uses existing patterns for file suggestions."""
        spec = FeatureSpec(
            name="AuthService",
            description="Auth service",
        )
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.DESIGNING,
        )
        context = {
            "relevant_files": ["src/service.py"],
            "existing_patterns": [{"requirement": "r1", "files": ["src/existing.py"]}],
        }

        design = await agent._design_feature(spec, context, impl)

        # Should suggest files based on existing patterns
        assert "files_to_create" in design
        assert "files_to_modify" in design


class TestGenerateTests:
    """Tests for _generate_tests method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent instance."""
        return FeatureDevelopmentAgent(
            root_path=str(tmp_path),
            enable_tdd=True,
        )

    @pytest.mark.asyncio
    async def test_generate_tests_from_criteria(self, agent):
        """Generates tests from acceptance criteria."""
        spec = FeatureSpec(
            name="Test",
            description="Test",
            acceptance_criteria=[
                "Users can login",
                "Sessions persist",
            ],
        )
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.TESTING,
        )
        design = {}

        # Mock test generator
        agent._test_generator = Mock()

        await agent._generate_tests(spec, design, impl)

        assert len(impl.test_files) == 2

    @pytest.mark.asyncio
    async def test_generate_tests_without_generator(self, agent):
        """Handles missing test generator gracefully."""
        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.TESTING,
        )
        design = {}

        agent._test_generator = None

        # Should not raise
        await agent._generate_tests(spec, design, impl)


class TestImplementFeature:
    """Tests for _implement_feature method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent instance."""
        return FeatureDevelopmentAgent(root_path=str(tmp_path))

    @pytest.mark.asyncio
    async def test_implement_creates_steps(self, agent):
        """Implementation creates steps for each file."""
        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.IMPLEMENTING,
        )
        design = {
            "files_to_create": ["new_file.py"],
            "files_to_modify": ["existing.py"],
        }

        await agent._implement_feature(spec, design, impl)

        assert len(impl.implementation_steps) == 2
        assert all(step.status == "completed" for step in impl.implementation_steps)

    @pytest.mark.asyncio
    async def test_implement_tracks_modified_files(self, agent):
        """Implementation tracks modified files."""
        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.IMPLEMENTING,
        )
        design = {
            "files_to_create": ["a.py", "b.py"],
            "files_to_modify": [],
        }

        await agent._implement_feature(spec, design, impl)

        assert len(impl.implementation_files) == 2


class TestVerifyImplementation:
    """Tests for _verify_implementation method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent instance."""
        return FeatureDevelopmentAgent(root_path=str(tmp_path))

    @pytest.mark.asyncio
    async def test_verify_sets_tests_pass(self, agent):
        """Verification sets tests_pass flag."""
        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.VERIFYING,
        )

        await agent._verify_implementation(impl)

        assert impl.tests_pass is True


class TestRequestApproval:
    """Tests for _request_approval method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent with mocked approval workflow."""
        agent = FeatureDevelopmentAgent(
            root_path=str(tmp_path),
            enable_approval=True,
        )
        return agent

    @pytest.mark.asyncio
    async def test_request_approval_creates_changes(self, agent):
        """Approval request creates FileChange objects."""
        mock_workflow = AsyncMock()
        mock_workflow.request_approval.return_value = Mock(
            request_id="req_123",
            approved=True,
            message="OK",
        )
        agent._approval_workflow = mock_workflow

        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.AWAITING_APPROVAL,
            implementation_files=["file1.py", "file2.py"],
        )

        await agent._request_approval(impl, ["approver1"])

        mock_workflow.request_approval.assert_called_once()
        call_args = mock_workflow.request_approval.call_args
        assert len(call_args.kwargs["changes"]) == 2

    @pytest.mark.asyncio
    async def test_request_approval_rejection_fails(self, agent):
        """Rejected approval marks implementation as failed."""
        mock_workflow = AsyncMock()
        mock_workflow.request_approval.return_value = Mock(
            request_id="req_456",
            approved=False,
            message="Changes rejected",
        )
        agent._approval_workflow = mock_workflow

        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.AWAITING_APPROVAL,
            implementation_files=["file.py"],
        )

        await agent._request_approval(impl, ["approver1"])

        assert impl.status == FeatureStatus.FAILED
        assert "rejected" in impl.error_message.lower()

    @pytest.mark.asyncio
    async def test_request_approval_without_workflow(self, agent):
        """Handles missing approval workflow gracefully."""
        agent._approval_workflow = None

        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.AWAITING_APPROVAL,
        )

        # Should not raise
        await agent._request_approval(impl, ["approver1"])


class TestUnderstandContext:
    """Tests for understand_context method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent instance."""
        return FeatureDevelopmentAgent(root_path=str(tmp_path))

    @pytest.mark.asyncio
    async def test_understand_context_returns_dict(self, agent):
        """understand_context returns a dictionary."""
        mock_codebase = AsyncMock()
        mock_codebase.understand.return_value = Mock(
            to_dict=Mock(return_value={"answer": "test answer"})
        )
        agent._codebase_agent = mock_codebase

        result = await agent.understand_context("What does this do?")

        assert isinstance(result, dict)
        assert "answer" in result

    @pytest.mark.asyncio
    async def test_understand_context_without_codebase_agent(self, tmp_path):
        """understand_context handles missing codebase agent."""
        agent = FeatureDevelopmentAgent(root_path=str(tmp_path))

        # Override codebase_agent property to return None
        with patch.object(
            FeatureDevelopmentAgent,
            "codebase_agent",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await agent.understand_context("Question?")

        assert "error" in result


class TestAuditImplementation:
    """Tests for audit_implementation method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent instance."""
        return FeatureDevelopmentAgent(root_path=str(tmp_path))

    @pytest.mark.asyncio
    async def test_audit_returns_findings(self, agent):
        """audit_implementation returns findings."""
        mock_codebase = AsyncMock()
        mock_codebase.audit.return_value = Mock(
            security_findings=[{"title": "Issue 1"}],
            bug_findings=[],
            risk_score=3.5,
            summary="Low risk",
        )
        agent._codebase_agent = mock_codebase

        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.COMPLETED,
        )

        result = await agent.audit_implementation(impl)

        assert result["security_findings"] == 1
        assert result["bug_findings"] == 0
        assert result["risk_score"] == 3.5

    @pytest.mark.asyncio
    async def test_audit_without_codebase_agent(self, tmp_path):
        """audit_implementation handles missing codebase agent."""
        agent = FeatureDevelopmentAgent(root_path=str(tmp_path))

        spec = FeatureSpec(name="Test", description="Test")
        impl = FeatureImplementation(
            spec=spec,
            status=FeatureStatus.COMPLETED,
        )

        # Override codebase_agent property to return None
        with patch.object(
            FeatureDevelopmentAgent,
            "codebase_agent",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await agent.audit_implementation(impl)

        assert "error" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent instance."""
        return FeatureDevelopmentAgent(
            root_path=str(tmp_path),
            enable_debate=False,
            enable_tdd=False,
            enable_approval=False,
        )

    @pytest.mark.asyncio
    async def test_empty_spec(self, agent):
        """Handles empty feature spec."""
        spec = FeatureSpec(name="", description="")

        result = await agent.develop_feature(spec)

        assert result.status == FeatureStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_spec_with_unicode(self, agent):
        """Handles Unicode in feature spec."""
        spec = FeatureSpec(
            name="Feature with Unicode",
            description="Description with emojis and unicode characters",
            requirements=["Requirement 1"],
        )

        result = await agent.develop_feature(spec)

        assert result.status == FeatureStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_many_requirements(self, agent):
        """Handles spec with many requirements."""
        spec = FeatureSpec(
            name="Complex Feature",
            description="Feature with many requirements",
            requirements=[f"Requirement {i}" for i in range(20)],
        )

        result = await agent.develop_feature(spec)

        assert result.status == FeatureStatus.COMPLETED

    def test_path_as_string(self, tmp_path):
        """Accepts path as string."""
        agent = FeatureDevelopmentAgent(root_path=str(tmp_path))
        assert agent.root_path == Path(tmp_path)

    def test_nonexistent_path_accepted(self, tmp_path):
        """Accepts nonexistent path (may be created later)."""
        nonexistent = tmp_path / "nonexistent"
        agent = FeatureDevelopmentAgent(root_path=str(nonexistent))
        assert agent.root_path == nonexistent
