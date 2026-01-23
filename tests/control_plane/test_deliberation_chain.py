"""
Tests for Control Plane Deliberation Chain.

Tests cover:
- ChainStatus enum
- StageStatus enum
- StageTransition enum
- DeliberationStage dataclass
- DeliberationChain dataclass
- StageResult dataclass
- ChainExecution dataclass
- ChainExecutor execution logic
"""

import pytest
from datetime import datetime, timezone

from aragora.control_plane.deliberation_chain import (
    ChainStatus,
    StageStatus,
    StageTransition,
    DeliberationStage,
    DeliberationChain,
    StageResult,
    ChainExecution,
)


class TestChainStatusEnum:
    """Tests for ChainStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all chain statuses are defined."""
        expected = ["pending", "running", "paused", "completed", "failed", "cancelled", "timeout"]
        for status in expected:
            assert ChainStatus(status) is not None

    def test_status_values(self):
        """Test chain status enum values."""
        assert ChainStatus.PENDING.value == "pending"
        assert ChainStatus.RUNNING.value == "running"
        assert ChainStatus.COMPLETED.value == "completed"
        assert ChainStatus.FAILED.value == "failed"
        assert ChainStatus.TIMEOUT.value == "timeout"


class TestStageStatusEnum:
    """Tests for StageStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all stage statuses are defined."""
        expected = ["pending", "running", "success", "failed", "skipped", "timeout"]
        for status in expected:
            assert StageStatus(status) is not None

    def test_status_values(self):
        """Test stage status enum values."""
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.SUCCESS.value == "success"
        assert StageStatus.SKIPPED.value == "skipped"


class TestStageTransitionEnum:
    """Tests for StageTransition enum."""

    def test_all_transitions_defined(self):
        """Test that all stage transitions are defined."""
        expected = ["success", "failure", "timeout", "error"]
        for transition in expected:
            assert StageTransition(transition) is not None

    def test_transition_values(self):
        """Test stage transition enum values."""
        assert StageTransition.SUCCESS.value == "success"
        assert StageTransition.FAILURE.value == "failure"
        assert StageTransition.TIMEOUT.value == "timeout"
        assert StageTransition.ERROR.value == "error"


class TestDeliberationStage:
    """Tests for DeliberationStage dataclass."""

    def test_stage_creation(self):
        """Test creating a deliberation stage."""
        stage = DeliberationStage(
            id="initial_review",
            topic_template="Review this code: {context.code}",
            agents=["claude", "gpt-4"],
            required_consensus=0.7,
            timeout_seconds=120,
            next_on_success="security_audit",
            next_on_failure="revise",
        )

        assert stage.id == "initial_review"
        assert "claude" in stage.agents
        assert stage.required_consensus == 0.7
        assert stage.timeout_seconds == 120
        assert stage.next_on_success == "security_audit"
        assert stage.next_on_failure == "revise"

    def test_stage_defaults(self):
        """Test stage default values."""
        stage = DeliberationStage(
            id="minimal",
            topic_template="Test topic",
        )

        assert stage.agents == []
        assert stage.required_consensus == 0.7
        assert stage.timeout_seconds == 300
        assert stage.next_on_success is None
        assert stage.next_on_failure is None
        assert stage.max_rounds == 5
        assert stage.min_agents == 2
        assert stage.retry_count == 0
        assert stage.retry_delay_seconds == 5

    def test_stage_to_dict(self):
        """Test stage serialization."""
        stage = DeliberationStage(
            id="test_stage",
            topic_template="Test: {context.input}",
            agents=["agent1", "agent2"],
            required_consensus=0.8,
            metadata={"custom": "value"},
        )

        data = stage.to_dict()
        assert data["id"] == "test_stage"
        assert data["agents"] == ["agent1", "agent2"]
        assert data["required_consensus"] == 0.8
        assert data["metadata"]["custom"] == "value"

    def test_stage_with_retries(self):
        """Test stage with retry configuration."""
        stage = DeliberationStage(
            id="retry_stage",
            topic_template="Retry test",
            retry_count=3,
            retry_delay_seconds=10,
        )

        assert stage.retry_count == 3
        assert stage.retry_delay_seconds == 10


class TestDeliberationChain:
    """Tests for DeliberationChain dataclass."""

    def test_chain_creation(self):
        """Test creating a deliberation chain."""
        stages = [
            DeliberationStage(
                id="stage1",
                topic_template="First stage",
                next_on_success="stage2",
            ),
            DeliberationStage(
                id="stage2",
                topic_template="Second stage: {previous.output}",
            ),
        ]

        chain = DeliberationChain(
            name="Test Pipeline",
            stages=stages,
            initial_context={"code": "def example(): pass"},
        )

        assert chain.name == "Test Pipeline"
        assert len(chain.stages) == 2
        assert chain.initial_context["code"] == "def example(): pass"

    def test_chain_defaults(self):
        """Test chain default values."""
        # Chain requires at least one stage
        chain = DeliberationChain(
            name="Minimal Chain",
            stages=[DeliberationStage(id="s1", topic_template="Test")],
        )

        assert chain.initial_context == {}
        assert chain.overall_timeout_seconds == 1800  # 30 minutes default
        # entry_stage_id defaults to first stage
        assert chain.entry_stage_id == "s1"

    def test_chain_validation_empty_stages(self):
        """Test that empty stages raises ValueError."""
        with pytest.raises(ValueError, match="at least one stage"):
            DeliberationChain(
                name="Invalid Chain",
                stages=[],
            )

    def test_chain_validation_invalid_stage_reference(self):
        """Test that invalid next stage reference raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            DeliberationChain(
                name="Invalid Chain",
                stages=[
                    DeliberationStage(
                        id="s1",
                        topic_template="Test",
                        next_on_success="nonexistent",
                    ),
                ],
            )

    def test_chain_to_dict(self):
        """Test chain serialization."""
        chain = DeliberationChain(
            name="Serialization Test",
            stages=[
                DeliberationStage(id="s1", topic_template="Topic 1"),
            ],
            initial_context={"key": "value"},
        )

        data = chain.to_dict()
        assert data["name"] == "Serialization Test"
        assert len(data["stages"]) == 1
        assert data["initial_context"]["key"] == "value"

    def test_chain_get_stage_by_id(self):
        """Test getting a stage by ID."""
        stages = [
            DeliberationStage(id="first", topic_template="First"),
            DeliberationStage(id="second", topic_template="Second"),
            DeliberationStage(id="third", topic_template="Third"),
        ]

        chain = DeliberationChain(name="Test", stages=stages)

        assert chain.get_stage("first").topic_template == "First"
        assert chain.get_stage("second").topic_template == "Second"
        assert chain.get_stage("nonexistent") is None


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_result_creation(self):
        """Test creating a stage result."""
        import time

        now = time.time()
        result = StageResult(
            stage_id="test_stage",
            status=StageStatus.SUCCESS,
            output="Stage completed successfully",
            confidence=0.85,
            started_at=now - 45.5,
            completed_at=now,
        )

        assert result.stage_id == "test_stage"
        assert result.status == StageStatus.SUCCESS
        assert result.output == "Stage completed successfully"
        assert result.confidence == 0.85
        # duration_seconds is a computed property
        assert result.duration_seconds is not None
        assert abs(result.duration_seconds - 45.5) < 0.1

    def test_result_failed(self):
        """Test failed stage result."""
        result = StageResult(
            stage_id="failed_stage",
            status=StageStatus.FAILED,
            output=None,
            confidence=0.0,
            error="Timeout waiting for agents",
        )

        assert result.status == StageStatus.FAILED
        assert result.error == "Timeout waiting for agents"

    def test_result_to_dict(self):
        """Test stage result serialization."""
        import time

        now = time.time()
        result = StageResult(
            stage_id="dict_test",
            status=StageStatus.SUCCESS,
            output="Test output",
            confidence=0.9,
            started_at=now - 30.0,
            completed_at=now,
        )

        data = result.to_dict()
        assert data["stage_id"] == "dict_test"
        assert data["status"] == "success"
        assert data["confidence"] == 0.9
        assert data["duration_seconds"] is not None


class TestChainExecution:
    """Tests for ChainExecution dataclass."""

    def test_execution_creation(self):
        """Test creating a chain execution."""
        import time

        now = time.time()

        # Create a chain first
        chain = DeliberationChain(
            name="Test Chain",
            stages=[
                DeliberationStage(id="s1", topic_template="Stage 1", next_on_success="s2"),
                DeliberationStage(id="s2", topic_template="Stage 2"),
            ],
        )

        execution = ChainExecution(
            chain=chain,
            status=ChainStatus.COMPLETED,
            stage_results={
                "s1": StageResult(
                    stage_id="s1", status=StageStatus.SUCCESS, output="Out1", confidence=0.8
                ),
                "s2": StageResult(
                    stage_id="s2", status=StageStatus.SUCCESS, output="Out2", confidence=0.9
                ),
            },
            started_at=now - 120.5,
            completed_at=now,
        )

        assert execution.chain.name == "Test Chain"
        assert execution.status == ChainStatus.COMPLETED
        assert len(execution.stage_results) == 2
        # duration_seconds is a computed property
        assert execution.duration_seconds is not None
        assert abs(execution.duration_seconds - 120.5) < 0.1

    def test_execution_failed_chain(self):
        """Test failed chain execution."""
        chain = DeliberationChain(
            name="Failed Chain",
            stages=[
                DeliberationStage(id="s1", topic_template="Stage 1", next_on_success="s2"),
                DeliberationStage(id="s2", topic_template="Stage 2"),
            ],
        )

        execution = ChainExecution(
            chain=chain,
            status=ChainStatus.FAILED,
            stage_results={
                "s1": StageResult(
                    stage_id="s1", status=StageStatus.SUCCESS, output="Ok", confidence=0.8
                ),
                "s2": StageResult(
                    stage_id="s2",
                    status=StageStatus.FAILED,
                    output=None,
                    confidence=0.0,
                    error="Agent timeout",
                ),
            },
            error="Chain failed at stage s2",
        )

        assert execution.status == ChainStatus.FAILED
        assert execution.error == "Chain failed at stage s2"

    def test_execution_to_dict(self):
        """Test chain execution serialization."""
        import time

        now = time.time()

        chain = DeliberationChain(
            name="Serialize Test",
            stages=[
                DeliberationStage(id="s1", topic_template="Stage 1"),
            ],
        )

        execution = ChainExecution(
            chain=chain,
            status=ChainStatus.COMPLETED,
            stage_results={},
            started_at=now - 60.0,
            completed_at=now,
        )

        data = execution.to_dict()
        assert data["chain_name"] == "Serialize Test"
        assert data["status"] == "completed"

    def test_get_last_result(self):
        """Test getting the last stage result."""
        import time

        now = time.time()

        chain = DeliberationChain(
            name="Test Chain",
            stages=[
                DeliberationStage(id="s1", topic_template="Stage 1", next_on_success="s2"),
                DeliberationStage(id="s2", topic_template="Stage 2"),
            ],
        )

        execution = ChainExecution(
            chain=chain,
            status=ChainStatus.COMPLETED,
            stage_results={
                "s1": StageResult(
                    stage_id="s1",
                    status=StageStatus.SUCCESS,
                    output="Out1",
                    confidence=0.8,
                    completed_at=now - 60.0,
                ),
                "s2": StageResult(
                    stage_id="s2",
                    status=StageStatus.SUCCESS,
                    output="Out2",
                    confidence=0.9,
                    completed_at=now,
                ),
            },
        )

        last_result = execution.get_last_result()
        assert last_result is not None
        assert last_result.stage_id == "s2"


class TestChainRouting:
    """Tests for chain stage routing logic."""

    def test_linear_chain(self):
        """Test a simple linear chain (s1 -> s2 -> s3)."""
        stages = [
            DeliberationStage(id="s1", topic_template="Stage 1", next_on_success="s2"),
            DeliberationStage(id="s2", topic_template="Stage 2", next_on_success="s3"),
            DeliberationStage(id="s3", topic_template="Stage 3"),
        ]

        chain = DeliberationChain(name="Linear", stages=stages)

        assert chain.get_stage("s1").next_on_success == "s2"
        assert chain.get_stage("s2").next_on_success == "s3"
        assert chain.get_stage("s3").next_on_success is None

    def test_branching_chain(self):
        """Test a branching chain with success/failure paths."""
        stages = [
            DeliberationStage(
                id="review",
                topic_template="Review",
                next_on_success="approve",
                next_on_failure="revise",
            ),
            DeliberationStage(id="approve", topic_template="Approve"),
            DeliberationStage(
                id="revise",
                topic_template="Revise",
                next_on_success="review",  # Loop back
            ),
        ]

        chain = DeliberationChain(name="Branching", stages=stages)

        review = chain.get_stage("review")
        assert review.next_on_success == "approve"
        assert review.next_on_failure == "revise"

        revise = chain.get_stage("revise")
        assert revise.next_on_success == "review"  # Can loop


class TestTopicTemplates:
    """Tests for topic template placeholder handling."""

    def test_template_with_context_placeholder(self):
        """Test template with context.* placeholder."""
        stage = DeliberationStage(
            id="ctx_test",
            topic_template="Review this code: {context.code}",
        )

        assert "{context.code}" in stage.topic_template

    def test_template_with_previous_placeholder(self):
        """Test template with previous.* placeholder."""
        stage = DeliberationStage(
            id="prev_test",
            topic_template="Previous output: {previous.output}. Confidence: {previous.confidence}",
        )

        assert "{previous.output}" in stage.topic_template
        assert "{previous.confidence}" in stage.topic_template

    def test_template_with_stage_reference(self):
        """Test template with specific stage reference."""
        stage = DeliberationStage(
            id="ref_test",
            topic_template="Initial review said: {initial_review.output}",
        )

        assert "{initial_review.output}" in stage.topic_template
