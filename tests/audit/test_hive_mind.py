"""
Tests for Hive-Mind Multi-Document Auditing.

Tests the hive_mind module that provides:
- Queen-Worker model for coordinated audits
- Task decomposition and delegation
- Worker pool management
- Byzantine consensus verification
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_session():
    """Create a mock AuditSession."""
    from aragora.audit.document_auditor import AuditSession, AuditType

    return AuditSession(
        id="session-test-123",
        document_ids=["doc-1"],
        audit_types=[AuditType.SECURITY, AuditType.QUALITY],
    )


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        {"id": "chunk-1", "content": "This is regular content about software."},
        {"id": "chunk-2", "content": "Contains password: secret123 in the code."},
        {"id": "chunk-3", "content": "User authentication flow with access control."},
    ]


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "mock_agent"
    agent.generate = AsyncMock(return_value="NO FINDINGS")
    return agent


@pytest.fixture
def mock_workers():
    """Create a list of mock worker agents."""
    workers = []
    for i in range(3):
        worker = MagicMock()
        worker.name = f"worker_{i}"
        worker.generate = AsyncMock(return_value="NO FINDINGS")
        workers.append(worker)
    return workers


@pytest.fixture
def hive_mind_config():
    """Create a HiveMindConfig for testing."""
    from aragora.audit.hive_mind import HiveMindConfig

    return HiveMindConfig(
        max_concurrent_workers=4,
        task_timeout_seconds=10,
        verify_critical_findings=False,  # Disable for unit tests
        max_retries_per_task=1,
    )


# ===========================================================================
# Tests: TaskPriority Enum
# ===========================================================================


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_values(self):
        """Test TaskPriority enum values."""
        from aragora.audit.hive_mind import TaskPriority

        assert TaskPriority.CRITICAL.value == "critical"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.NORMAL.value == "normal"
        assert TaskPriority.LOW.value == "low"

    def test_priority_ordering(self):
        """Test all priority levels exist."""
        from aragora.audit.hive_mind import TaskPriority

        assert len(TaskPriority) == 4


# ===========================================================================
# Tests: TaskStatus Enum
# ===========================================================================


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test TaskStatus enum values."""
        from aragora.audit.hive_mind import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.ASSIGNED.value == "assigned"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


# ===========================================================================
# Tests: WorkerTask Dataclass
# ===========================================================================


class TestWorkerTask:
    """Tests for WorkerTask dataclass."""

    def test_creation(self):
        """Test WorkerTask creation."""
        from aragora.audit.hive_mind import TaskPriority, TaskStatus, WorkerTask

        task = WorkerTask(
            id="task-1",
            chunk_id="chunk-1",
            chunk_content="Test content",
            audit_types=["security", "quality"],
        )

        assert task.id == "task-1"
        assert task.chunk_id == "chunk-1"
        assert task.chunk_content == "Test content"
        assert len(task.audit_types) == 2

    def test_default_values(self):
        """Test WorkerTask default values."""
        from aragora.audit.hive_mind import TaskPriority, TaskStatus, WorkerTask

        task = WorkerTask(
            id="task-1",
            chunk_id="chunk-1",
            chunk_content="Content",
            audit_types=[],
        )

        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert task.assigned_to is None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.metadata == {}

    def test_created_at_auto_set(self):
        """Test created_at is automatically set."""
        from aragora.audit.hive_mind import WorkerTask

        task = WorkerTask(
            id="task-1",
            chunk_id="chunk-1",
            chunk_content="Content",
            audit_types=[],
        )

        assert task.created_at is not None
        assert isinstance(task.created_at, datetime)


# ===========================================================================
# Tests: WorkerResult Dataclass
# ===========================================================================


class TestWorkerResult:
    """Tests for WorkerResult dataclass."""

    def test_creation(self):
        """Test WorkerResult creation."""
        from aragora.audit.hive_mind import WorkerResult

        result = WorkerResult(
            task_id="task-1",
            worker_name="worker-1",
            findings=[],
            raw_response="test response",
            duration_seconds=1.5,
            success=True,
        )

        assert result.task_id == "task-1"
        assert result.worker_name == "worker-1"
        assert result.success is True
        assert result.duration_seconds == 1.5

    def test_default_values(self):
        """Test WorkerResult default values."""
        from aragora.audit.hive_mind import WorkerResult

        result = WorkerResult(
            task_id="task-1",
            worker_name="worker-1",
        )

        assert result.findings == []
        assert result.raw_response == ""
        assert result.duration_seconds == 0.0
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test WorkerResult for failed task."""
        from aragora.audit.hive_mind import WorkerResult

        result = WorkerResult(
            task_id="task-1",
            worker_name="worker-1",
            success=False,
            error="Task timed out",
        )

        assert result.success is False
        assert result.error == "Task timed out"


# ===========================================================================
# Tests: HiveMindConfig Dataclass
# ===========================================================================


class TestHiveMindConfig:
    """Tests for HiveMindConfig dataclass."""

    def test_creation(self):
        """Test HiveMindConfig creation."""
        from aragora.audit.hive_mind import HiveMindConfig

        config = HiveMindConfig(
            max_concurrent_workers=10,
            task_timeout_seconds=600,
        )

        assert config.max_concurrent_workers == 10
        assert config.task_timeout_seconds == 600

    def test_default_values(self):
        """Test HiveMindConfig default values."""
        from aragora.audit.hive_mind import HiveMindConfig

        config = HiveMindConfig()

        assert config.max_concurrent_workers == 8
        assert config.task_timeout_seconds == 300
        assert config.verify_critical_findings is True
        assert config.min_verifications_per_finding == 2
        assert config.consensus_threshold == 0.66
        assert config.max_retries_per_task == 2
        assert config.retry_delay_seconds == 1.0
        assert config.chunk_batch_size == 10
        assert config.progress_callback is None

    def test_progress_callback(self):
        """Test HiveMindConfig with progress callback."""
        from aragora.audit.hive_mind import HiveMindConfig

        callback_calls = []

        def my_callback(stage, current, total):
            callback_calls.append((stage, current, total))

        config = HiveMindConfig(progress_callback=my_callback)
        config.progress_callback("test", 5, 10)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("test", 5, 10)


# ===========================================================================
# Tests: HiveMindResult Dataclass
# ===========================================================================


class TestHiveMindResult:
    """Tests for HiveMindResult dataclass."""

    def test_creation(self):
        """Test HiveMindResult creation."""
        from aragora.audit.hive_mind import HiveMindResult

        result = HiveMindResult(
            session_id="session-1",
            total_tasks=10,
            completed_tasks=8,
            failed_tasks=2,
        )

        assert result.session_id == "session-1"
        assert result.total_tasks == 10
        assert result.completed_tasks == 8
        assert result.failed_tasks == 2

    def test_default_values(self):
        """Test HiveMindResult default values."""
        from aragora.audit.hive_mind import HiveMindResult

        result = HiveMindResult(
            session_id="session-1",
            total_tasks=10,
            completed_tasks=10,
            failed_tasks=0,
        )

        assert result.findings == []
        assert result.verified_findings == []
        assert result.worker_stats == {}
        assert result.duration_seconds == 0.0
        assert result.success is True
        assert result.errors == []

    def test_critical_findings_property(self):
        """Test critical_findings property."""
        from aragora.audit.document_auditor import AuditFinding, FindingSeverity
        from aragora.audit.hive_mind import HiveMindResult

        findings = [
            MagicMock(severity=MagicMock(value="critical")),
            MagicMock(severity=MagicMock(value="high")),
            MagicMock(severity=MagicMock(value="critical")),
        ]

        result = HiveMindResult(
            session_id="session-1",
            total_tasks=3,
            completed_tasks=3,
            failed_tasks=0,
            findings=findings,
        )

        assert len(result.critical_findings) == 2

    def test_high_findings_property(self):
        """Test high_findings property."""
        from aragora.audit.hive_mind import HiveMindResult

        findings = [
            MagicMock(severity=MagicMock(value="critical")),
            MagicMock(severity=MagicMock(value="high")),
            MagicMock(severity=MagicMock(value="high")),
            MagicMock(severity=MagicMock(value="medium")),
        ]

        result = HiveMindResult(
            session_id="session-1",
            total_tasks=4,
            completed_tasks=4,
            failed_tasks=0,
            findings=findings,
        )

        assert len(result.high_findings) == 2


# ===========================================================================
# Tests: QueenOrchestrator Priority Assessment
# ===========================================================================


class TestQueenOrchestratorPriority:
    """Tests for QueenOrchestrator priority assessment."""

    def test_assess_priority_critical_password(self, mock_agent):
        """Test critical priority for password content."""
        from aragora.audit.hive_mind import QueenOrchestrator, TaskPriority

        queen = QueenOrchestrator(queen_agent=mock_agent)

        priority = queen._assess_priority("Contains password: secret123")

        assert priority == TaskPriority.CRITICAL

    def test_assess_priority_critical_api_key(self, mock_agent):
        """Test critical priority for API key content."""
        from aragora.audit.hive_mind import QueenOrchestrator, TaskPriority

        queen = QueenOrchestrator(queen_agent=mock_agent)

        priority = queen._assess_priority("api_key=abc123xyz")

        assert priority == TaskPriority.CRITICAL

    def test_assess_priority_high_auth(self, mock_agent):
        """Test high priority for auth-related content."""
        from aragora.audit.hive_mind import QueenOrchestrator, TaskPriority

        queen = QueenOrchestrator(queen_agent=mock_agent)

        priority = queen._assess_priority("User authentication module")

        assert priority == TaskPriority.HIGH

    def test_assess_priority_high_encryption(self, mock_agent):
        """Test high priority for encryption-related content."""
        from aragora.audit.hive_mind import QueenOrchestrator, TaskPriority

        queen = QueenOrchestrator(queen_agent=mock_agent)

        priority = queen._assess_priority("Data encryption handler")

        assert priority == TaskPriority.HIGH

    def test_assess_priority_normal(self, mock_agent):
        """Test normal priority for regular content."""
        from aragora.audit.hive_mind import QueenOrchestrator, TaskPriority

        queen = QueenOrchestrator(queen_agent=mock_agent)

        priority = queen._assess_priority("Regular documentation text")

        assert priority == TaskPriority.NORMAL


# ===========================================================================
# Tests: QueenOrchestrator Decomposition
# ===========================================================================


class TestQueenOrchestratorDecompose:
    """Tests for QueenOrchestrator task decomposition."""

    @pytest.mark.asyncio
    async def test_decompose_audit_creates_tasks(self, mock_agent, mock_session, sample_chunks):
        """Test decompose_audit creates tasks for each chunk."""
        from aragora.audit.hive_mind import QueenOrchestrator

        queen = QueenOrchestrator(queen_agent=mock_agent)

        tasks = await queen.decompose_audit(mock_session, sample_chunks)

        assert len(tasks) == 3
        assert all(t.id.startswith(mock_session.id) for t in tasks)

    @pytest.mark.asyncio
    async def test_decompose_audit_assigns_priorities(
        self, mock_agent, mock_session, sample_chunks
    ):
        """Test decompose_audit assigns correct priorities."""
        from aragora.audit.hive_mind import QueenOrchestrator, TaskPriority

        queen = QueenOrchestrator(queen_agent=mock_agent)

        tasks = await queen.decompose_audit(mock_session, sample_chunks)

        # chunk-2 has "password" so should be critical
        critical_tasks = [t for t in tasks if t.priority == TaskPriority.CRITICAL]
        assert len(critical_tasks) == 1

        # chunk-3 has "auth" and "access" so should be high
        high_tasks = [t for t in tasks if t.priority == TaskPriority.HIGH]
        assert len(high_tasks) == 1

    @pytest.mark.asyncio
    async def test_decompose_audit_includes_metadata(self, mock_agent, mock_session, sample_chunks):
        """Test decompose_audit includes metadata."""
        from aragora.audit.hive_mind import QueenOrchestrator

        queen = QueenOrchestrator(queen_agent=mock_agent)

        tasks = await queen.decompose_audit(mock_session, sample_chunks)

        for i, task in enumerate(tasks):
            assert task.metadata["session_id"] == mock_session.id
            assert task.metadata["chunk_index"] == i


# ===========================================================================
# Tests: QueenOrchestrator Prompt Building
# ===========================================================================


class TestQueenOrchestratorPromptBuilding:
    """Tests for QueenOrchestrator prompt building."""

    def test_build_audit_prompt(self, mock_agent):
        """Test _build_audit_prompt creates proper prompt."""
        from aragora.audit.hive_mind import QueenOrchestrator, WorkerTask

        queen = QueenOrchestrator(queen_agent=mock_agent)

        task = WorkerTask(
            id="task-1",
            chunk_id="chunk-1",
            chunk_content="Test content here",
            audit_types=["security", "quality"],
        )

        prompt = queen._build_audit_prompt(task)

        assert "security, quality" in prompt
        assert "chunk-1" in prompt
        assert "Test content here" in prompt
        assert "SEVERITY:" in prompt
        assert "EVIDENCE:" in prompt


# ===========================================================================
# Tests: QueenOrchestrator Finding Parsing
# ===========================================================================


class TestQueenOrchestratorFindingParsing:
    """Tests for QueenOrchestrator finding parsing."""

    def test_parse_findings_no_findings(self, mock_agent):
        """Test parsing response with no findings."""
        from aragora.audit.hive_mind import QueenOrchestrator, WorkerTask

        queen = QueenOrchestrator(queen_agent=mock_agent)
        task = WorkerTask(
            id="task-1",
            chunk_id="chunk-1",
            chunk_content="Test",
            audit_types=["security"],
        )

        findings = queen._parse_findings("NO FINDINGS", task, "worker", "session-1")

        assert findings == []

    def test_parse_findings_with_findings(self, mock_agent):
        """Test parsing response with findings."""
        from aragora.audit.hive_mind import QueenOrchestrator, WorkerTask

        queen = QueenOrchestrator(queen_agent=mock_agent)
        task = WorkerTask(
            id="task-1",
            chunk_id="chunk-1",
            chunk_content="Test",
            audit_types=["security"],
            metadata={"document_id": "doc-1"},
        )

        response = """FINDING: high
CATEGORY: credential_exposure
DESCRIPTION: Password found in plaintext
EVIDENCE: password=secret123
RECOMMENDATION: Use environment variables
---"""

        findings = queen._parse_findings(response, task, "worker", "session-1")

        assert len(findings) == 1
        assert findings[0].category == "credential_exposure"
        assert "Password" in findings[0].description

    def test_parse_findings_handles_malformed(self, mock_agent):
        """Test parsing handles malformed response gracefully."""
        from aragora.audit.hive_mind import QueenOrchestrator, WorkerTask

        queen = QueenOrchestrator(queen_agent=mock_agent)
        task = WorkerTask(
            id="task-1",
            chunk_id="chunk-1",
            chunk_content="Test",
            audit_types=["security"],
        )

        response = """Random text without proper formatting
This should not parse as a finding"""

        findings = queen._parse_findings(response, task, "worker", "session-1")

        assert findings == []


# ===========================================================================
# Tests: QueenOrchestrator Task Dispatch
# ===========================================================================


class TestQueenOrchestratorDispatch:
    """Tests for QueenOrchestrator task dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_tasks_sorts_by_priority(self, mock_agent, mock_workers):
        """Test dispatch_tasks sorts tasks by priority."""
        from aragora.audit.hive_mind import (
            QueenOrchestrator,
            TaskPriority,
            WorkerTask,
        )

        queen = QueenOrchestrator(queen_agent=mock_agent)

        tasks = [
            WorkerTask(
                id="low",
                chunk_id="1",
                chunk_content="low",
                audit_types=[],
                priority=TaskPriority.LOW,
            ),
            WorkerTask(
                id="critical",
                chunk_id="2",
                chunk_content="critical",
                audit_types=[],
                priority=TaskPriority.CRITICAL,
            ),
            WorkerTask(
                id="normal",
                chunk_id="3",
                chunk_content="normal",
                audit_types=[],
                priority=TaskPriority.NORMAL,
            ),
        ]

        await queen.dispatch_tasks(tasks, mock_workers)

        # Verify queue has items in priority order
        first = await queen._task_queue.get()
        assert first.id == "critical"

        second = await queen._task_queue.get()
        assert second.id == "normal"

        third = await queen._task_queue.get()
        assert third.id == "low"


# ===========================================================================
# Tests: AuditHiveMind Initialization
# ===========================================================================


class TestAuditHiveMindInit:
    """Tests for AuditHiveMind initialization."""

    def test_init_creates_orchestrator(self, mock_agent, mock_workers, hive_mind_config):
        """Test __post_init__ creates QueenOrchestrator."""
        from aragora.audit.hive_mind import AuditHiveMind

        hive = AuditHiveMind(
            queen=mock_agent,
            workers=mock_workers,
            config=hive_mind_config,
        )

        assert hasattr(hive, "_orchestrator")
        assert hive._orchestrator.queen_agent is mock_agent
        assert hive._orchestrator.config is hive_mind_config

    def test_init_with_delegation_strategy(self, mock_agent, mock_workers):
        """Test initialization with delegation strategy."""
        from aragora.audit.hive_mind import AuditHiveMind

        mock_strategy = MagicMock()

        hive = AuditHiveMind(
            queen=mock_agent,
            workers=mock_workers,
            delegation_strategy=mock_strategy,
        )

        assert hive._orchestrator.delegation_strategy is mock_strategy

    def test_init_with_hook_manager(self, mock_agent, mock_workers):
        """Test initialization with hook manager."""
        from aragora.audit.hive_mind import AuditHiveMind

        mock_hooks = MagicMock()

        hive = AuditHiveMind(
            queen=mock_agent,
            workers=mock_workers,
            hook_manager=mock_hooks,
        )

        assert hive._orchestrator.hook_manager is mock_hooks


# ===========================================================================
# Tests: AuditHiveMind Document Auditing
# ===========================================================================


class TestAuditHiveMindAudit:
    """Tests for AuditHiveMind document auditing."""

    @pytest.mark.asyncio
    async def test_audit_documents_returns_result(
        self, mock_agent, mock_workers, mock_session, sample_chunks, hive_mind_config
    ):
        """Test audit_documents returns HiveMindResult."""
        from aragora.audit.hive_mind import AuditHiveMind, HiveMindResult

        hive = AuditHiveMind(
            queen=mock_agent,
            workers=mock_workers,
            config=hive_mind_config,
        )

        result = await hive.audit_documents(mock_session, sample_chunks)

        assert isinstance(result, HiveMindResult)
        assert result.session_id == mock_session.id
        assert result.total_tasks == len(sample_chunks)

    @pytest.mark.asyncio
    async def test_audit_documents_tracks_duration(
        self, mock_agent, mock_workers, mock_session, sample_chunks, hive_mind_config
    ):
        """Test audit_documents tracks duration."""
        from aragora.audit.hive_mind import AuditHiveMind

        hive = AuditHiveMind(
            queen=mock_agent,
            workers=mock_workers,
            config=hive_mind_config,
        )

        result = await hive.audit_documents(mock_session, sample_chunks)

        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_audit_documents_collects_worker_stats(
        self, mock_agent, mock_workers, mock_session, sample_chunks, hive_mind_config
    ):
        """Test audit_documents collects worker statistics."""
        from aragora.audit.hive_mind import AuditHiveMind

        hive = AuditHiveMind(
            queen=mock_agent,
            workers=mock_workers,
            config=hive_mind_config,
        )

        result = await hive.audit_documents(mock_session, sample_chunks)

        # Should have stats for workers that processed tasks
        assert isinstance(result.worker_stats, dict)


# ===========================================================================
# Tests: AuditHiveMind Consensus Verification
# ===========================================================================


class TestAuditHiveMindConsensus:
    """Tests for AuditHiveMind consensus verification."""

    @pytest.mark.asyncio
    async def test_verify_findings_skips_with_few_workers(self, mock_agent):
        """Test consensus verification skips with < 3 workers."""
        from aragora.audit.hive_mind import AuditHiveMind, HiveMindConfig

        # Only 2 workers - not enough for Byzantine consensus
        workers = [MagicMock(), MagicMock()]
        for w in workers:
            w.name = "worker"
            w.generate = AsyncMock(return_value="NO FINDINGS")

        config = HiveMindConfig(verify_critical_findings=True)
        hive = AuditHiveMind(queen=mock_agent, workers=workers, config=config)

        mock_findings = [MagicMock()]
        result = await hive._verify_findings_consensus(mock_findings, "session-1")

        # Should return original findings since verification skipped
        assert result == mock_findings

    @pytest.mark.asyncio
    async def test_verify_findings_disabled_in_config(
        self, mock_agent, mock_workers, mock_session, sample_chunks
    ):
        """Test verification can be disabled in config."""
        from aragora.audit.hive_mind import AuditHiveMind, HiveMindConfig

        config = HiveMindConfig(verify_critical_findings=False)
        hive = AuditHiveMind(queen=mock_agent, workers=mock_workers, config=config)

        result = await hive.audit_documents(mock_session, sample_chunks)

        # verified_findings should be empty when disabled
        assert result.verified_findings == []
