"""
Tests for Finding Workflow State Machine.

Comprehensive test suite for the findings workflow module covering:
- WorkflowState enum and transitions
- WorkflowEvent creation and serialization
- FindingWorkflowData lifecycle
- FindingWorkflow state machine operations
- State transitions and validations
- Assignment and commenting
- Priority and due date management
- Linked findings and duplicates
- Time tracking and metrics
- Transition hooks
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.audit.findings.workflow import (
    FindingWorkflow,
    FindingWorkflowData,
    InvalidTransitionError,
    VALID_TRANSITIONS,
    WorkflowError,
    WorkflowEvent,
    WorkflowEventType,
    WorkflowState,
    WorkflowTransition,
    get_workflow_diagram,
    map_legacy_status,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def finding_workflow_data():
    """Create a FindingWorkflowData instance for testing."""
    now = datetime.now(timezone.utc)
    return FindingWorkflowData(
        finding_id="finding_123",
        created_at=now,
        updated_at=now,
        state_entered_at=now,
    )


@pytest.fixture
def finding_workflow(finding_workflow_data):
    """Create a FindingWorkflow instance for testing."""
    return FindingWorkflow(data=finding_workflow_data)


@pytest.fixture
def workflow_event():
    """Create a sample WorkflowEvent for testing."""
    return WorkflowEvent(
        finding_id="finding_123",
        event_type=WorkflowEventType.STATE_CHANGE,
        user_id="user_456",
        user_name="John Doe",
        from_state=WorkflowState.OPEN,
        to_state=WorkflowState.TRIAGING,
        comment="Starting triage",
    )


def make_workflow_data(
    finding_id: str = "test",
    current_state: WorkflowState = WorkflowState.OPEN,
    **kwargs,
) -> FindingWorkflowData:
    """Helper to create FindingWorkflowData with timezone-aware datetimes."""
    now = datetime.now(timezone.utc)
    return FindingWorkflowData(
        finding_id=finding_id,
        current_state=current_state,
        created_at=kwargs.get("created_at", now),
        updated_at=kwargs.get("updated_at", now),
        state_entered_at=kwargs.get("state_entered_at", now),
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ("created_at", "updated_at", "state_entered_at")
        },
    )


def make_workflow(
    finding_id: str = "test",
    current_state: WorkflowState = WorkflowState.OPEN,
    **kwargs,
) -> FindingWorkflow:
    """Helper to create FindingWorkflow with timezone-aware datetimes."""
    data = make_workflow_data(finding_id=finding_id, current_state=current_state, **kwargs)
    return FindingWorkflow(data=data)


# ===========================================================================
# Tests: WorkflowState Enum
# ===========================================================================


class TestWorkflowState:
    """Tests for WorkflowState enum."""

    def test_all_states_exist(self):
        """Test all workflow states exist."""
        assert WorkflowState.OPEN.value == "open"
        assert WorkflowState.TRIAGING.value == "triaging"
        assert WorkflowState.INVESTIGATING.value == "investigating"
        assert WorkflowState.REMEDIATING.value == "remediating"
        assert WorkflowState.RESOLVED.value == "resolved"
        assert WorkflowState.FALSE_POSITIVE.value == "false_positive"
        assert WorkflowState.ACCEPTED_RISK.value == "accepted_risk"
        assert WorkflowState.DUPLICATE.value == "duplicate"

    def test_legacy_states_exist(self):
        """Test legacy states exist for backward compatibility."""
        assert WorkflowState.ACKNOWLEDGED.value == "acknowledged"
        assert WorkflowState.WONT_FIX.value == "wont_fix"

    def test_state_is_string_enum(self):
        """Test that states are string enums."""
        assert isinstance(WorkflowState.OPEN, str)
        assert WorkflowState.RESOLVED == "resolved"


# ===========================================================================
# Tests: WorkflowEventType Enum
# ===========================================================================


class TestWorkflowEventType:
    """Tests for WorkflowEventType enum."""

    def test_all_event_types_exist(self):
        """Test all event types exist."""
        assert WorkflowEventType.STATE_CHANGE.value == "state_change"
        assert WorkflowEventType.ASSIGNMENT.value == "assignment"
        assert WorkflowEventType.COMMENT.value == "comment"
        assert WorkflowEventType.PRIORITY_CHANGE.value == "priority_change"
        assert WorkflowEventType.DUE_DATE_CHANGE.value == "due_date_change"
        assert WorkflowEventType.TAG_ADDED.value == "tag_added"
        assert WorkflowEventType.TAG_REMOVED.value == "tag_removed"
        assert WorkflowEventType.LINKED.value == "linked"
        assert WorkflowEventType.UNLINKED.value == "unlinked"
        assert WorkflowEventType.SEVERITY_CHANGE.value == "severity_change"


# ===========================================================================
# Tests: VALID_TRANSITIONS
# ===========================================================================


class TestValidTransitions:
    """Tests for VALID_TRANSITIONS mapping."""

    def test_open_transitions(self):
        """Test valid transitions from OPEN state."""
        valid = VALID_TRANSITIONS[WorkflowState.OPEN]
        assert WorkflowState.TRIAGING in valid
        assert WorkflowState.INVESTIGATING in valid
        assert WorkflowState.FALSE_POSITIVE in valid
        assert WorkflowState.DUPLICATE in valid

    def test_triaging_transitions(self):
        """Test valid transitions from TRIAGING state."""
        valid = VALID_TRANSITIONS[WorkflowState.TRIAGING]
        assert WorkflowState.INVESTIGATING in valid
        assert WorkflowState.FALSE_POSITIVE in valid
        assert WorkflowState.ACCEPTED_RISK in valid
        assert WorkflowState.OPEN in valid  # Can reopen

    def test_investigating_transitions(self):
        """Test valid transitions from INVESTIGATING state."""
        valid = VALID_TRANSITIONS[WorkflowState.INVESTIGATING]
        assert WorkflowState.REMEDIATING in valid
        assert WorkflowState.FALSE_POSITIVE in valid
        assert WorkflowState.ACCEPTED_RISK in valid
        assert WorkflowState.TRIAGING in valid  # Can go back

    def test_remediating_transitions(self):
        """Test valid transitions from REMEDIATING state."""
        valid = VALID_TRANSITIONS[WorkflowState.REMEDIATING]
        assert WorkflowState.RESOLVED in valid
        assert WorkflowState.INVESTIGATING in valid  # Go back if fix didn't work
        assert WorkflowState.ACCEPTED_RISK in valid

    def test_terminal_states_can_reopen(self):
        """Test that terminal states can transition back to OPEN."""
        assert WorkflowState.OPEN in VALID_TRANSITIONS[WorkflowState.RESOLVED]
        assert WorkflowState.OPEN in VALID_TRANSITIONS[WorkflowState.FALSE_POSITIVE]
        assert WorkflowState.OPEN in VALID_TRANSITIONS[WorkflowState.ACCEPTED_RISK]
        assert WorkflowState.OPEN in VALID_TRANSITIONS[WorkflowState.DUPLICATE]


# ===========================================================================
# Tests: WorkflowEvent Dataclass
# ===========================================================================


class TestWorkflowEvent:
    """Tests for WorkflowEvent dataclass."""

    def test_creation_with_defaults(self):
        """Test creating event with default values."""
        event = WorkflowEvent()

        assert event.id is not None
        assert event.finding_id == ""
        assert event.event_type == WorkflowEventType.STATE_CHANGE
        assert event.timestamp is not None
        assert event.user_id == ""
        assert event.comment == ""

    def test_creation_with_values(self, workflow_event):
        """Test creating event with values."""
        assert workflow_event.finding_id == "finding_123"
        assert workflow_event.event_type == WorkflowEventType.STATE_CHANGE
        assert workflow_event.user_id == "user_456"
        assert workflow_event.user_name == "John Doe"
        assert workflow_event.from_state == WorkflowState.OPEN
        assert workflow_event.to_state == WorkflowState.TRIAGING
        assert workflow_event.comment == "Starting triage"

    def test_to_dict(self, workflow_event):
        """Test converting event to dictionary."""
        result = workflow_event.to_dict()

        assert result["finding_id"] == "finding_123"
        assert result["event_type"] == "state_change"
        assert result["user_id"] == "user_456"
        assert result["from_state"] == "open"
        assert result["to_state"] == "triaging"
        assert result["comment"] == "Starting triage"
        assert "timestamp" in result
        assert "id" in result

    def test_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "id": "event_123",
            "finding_id": "finding_456",
            "event_type": "state_change",
            "timestamp": "2024-01-15T10:30:00+00:00",
            "user_id": "user_789",
            "user_name": "Jane Smith",
            "from_state": "open",
            "to_state": "investigating",
            "comment": "Starting investigation",
        }

        event = WorkflowEvent.from_dict(data)

        assert event.id == "event_123"
        assert event.finding_id == "finding_456"
        assert event.event_type == WorkflowEventType.STATE_CHANGE
        assert event.user_id == "user_789"
        assert event.from_state == WorkflowState.OPEN
        assert event.to_state == WorkflowState.INVESTIGATING

    def test_from_dict_with_minimal_data(self):
        """Test creating event from minimal dictionary."""
        data = {}

        event = WorkflowEvent.from_dict(data)

        assert event.id is not None
        assert event.finding_id == ""
        assert event.event_type == WorkflowEventType.STATE_CHANGE

    def test_to_dict_with_none_states(self):
        """Test to_dict with None states."""
        event = WorkflowEvent(
            finding_id="finding_123",
            event_type=WorkflowEventType.COMMENT,
            user_id="user_456",
            from_state=None,
            to_state=None,
            comment="Just a comment",
        )

        result = event.to_dict()

        assert result["from_state"] is None
        assert result["to_state"] is None


# ===========================================================================
# Tests: WorkflowTransition Dataclass
# ===========================================================================


class TestWorkflowTransition:
    """Tests for WorkflowTransition dataclass."""

    def test_creation(self):
        """Test creating a workflow transition."""
        now = datetime.now(timezone.utc)
        transition = WorkflowTransition(
            from_state=WorkflowState.OPEN,
            to_state=WorkflowState.TRIAGING,
            timestamp=now,
            user_id="user_123",
            comment="Starting triage",
            metadata={"priority": "high"},
        )

        assert transition.from_state == WorkflowState.OPEN
        assert transition.to_state == WorkflowState.TRIAGING
        assert transition.timestamp == now
        assert transition.user_id == "user_123"
        assert transition.comment == "Starting triage"
        assert transition.metadata == {"priority": "high"}


# ===========================================================================
# Tests: WorkflowError Exceptions
# ===========================================================================


class TestWorkflowErrors:
    """Tests for workflow error classes."""

    def test_workflow_error(self):
        """Test base WorkflowError."""
        error = WorkflowError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_invalid_transition_error(self):
        """Test InvalidTransitionError."""
        error = InvalidTransitionError(WorkflowState.OPEN, WorkflowState.RESOLVED)

        assert error.from_state == WorkflowState.OPEN
        assert error.to_state == WorkflowState.RESOLVED
        assert "open" in str(error).lower()
        assert "resolved" in str(error).lower()


# ===========================================================================
# Tests: FindingWorkflowData Dataclass
# ===========================================================================


class TestFindingWorkflowData:
    """Tests for FindingWorkflowData dataclass."""

    def test_creation_with_defaults(self):
        """Test creating workflow data with defaults."""
        data = FindingWorkflowData(finding_id="finding_123")

        assert data.finding_id == "finding_123"
        assert data.current_state == WorkflowState.OPEN
        assert data.history == []
        assert data.assigned_to is None
        assert data.priority == 3
        assert data.due_date is None
        assert data.linked_findings == []
        assert data.resolved_at is None
        assert data.time_in_states == {}

    def test_creation_with_values(self):
        """Test creating workflow data with values."""
        now = datetime.now(timezone.utc)
        data = FindingWorkflowData(
            finding_id="finding_123",
            current_state=WorkflowState.INVESTIGATING,
            assigned_to="user_456",
            priority=1,
            due_date=now + timedelta(days=7),
        )

        assert data.current_state == WorkflowState.INVESTIGATING
        assert data.assigned_to == "user_456"
        assert data.priority == 1
        assert data.due_date is not None

    def test_to_dict(self, finding_workflow_data):
        """Test converting workflow data to dictionary."""
        result = finding_workflow_data.to_dict()

        assert result["finding_id"] == "finding_123"
        assert result["current_state"] == "open"
        assert result["history"] == []
        assert result["assigned_to"] is None
        assert result["priority"] == 3
        assert "created_at" in result
        assert "updated_at" in result

    def test_from_dict(self):
        """Test creating workflow data from dictionary."""
        data = {
            "finding_id": "finding_456",
            "current_state": "investigating",
            "assigned_to": "user_789",
            "priority": 2,
            "linked_findings": ["finding_a", "finding_b"],
        }

        workflow_data = FindingWorkflowData.from_dict(data)

        assert workflow_data.finding_id == "finding_456"
        assert workflow_data.current_state == WorkflowState.INVESTIGATING
        assert workflow_data.assigned_to == "user_789"
        assert workflow_data.priority == 2
        assert len(workflow_data.linked_findings) == 2

    def test_from_dict_with_history(self):
        """Test creating workflow data from dict with history."""
        data = {
            "finding_id": "finding_123",
            "current_state": "triaging",
            "history": [
                {
                    "event_type": "state_change",
                    "user_id": "user_1",
                    "from_state": "open",
                    "to_state": "triaging",
                }
            ],
        }

        workflow_data = FindingWorkflowData.from_dict(data)

        assert len(workflow_data.history) == 1
        assert workflow_data.history[0].from_state == WorkflowState.OPEN


# ===========================================================================
# Tests: FindingWorkflow Initialization
# ===========================================================================


class TestFindingWorkflowInit:
    """Tests for FindingWorkflow initialization."""

    def test_init_with_data(self, finding_workflow_data):
        """Test initializing with existing data."""
        workflow = FindingWorkflow(data=finding_workflow_data)

        assert workflow.data == finding_workflow_data
        assert workflow.state == WorkflowState.OPEN

    def test_init_with_finding_id(self):
        """Test initializing with finding_id."""
        workflow = FindingWorkflow(finding_id="new_finding_123")

        assert workflow.data.finding_id == "new_finding_123"
        assert workflow.state == WorkflowState.OPEN

    def test_init_requires_data_or_finding_id(self):
        """Test that either data or finding_id is required."""
        with pytest.raises(ValueError):
            FindingWorkflow()


# ===========================================================================
# Tests: FindingWorkflow Properties
# ===========================================================================


class TestFindingWorkflowProperties:
    """Tests for FindingWorkflow properties."""

    def test_state_property(self, finding_workflow):
        """Test state property."""
        assert finding_workflow.state == WorkflowState.OPEN

    def test_history_property(self, finding_workflow):
        """Test history property."""
        assert finding_workflow.history == []

        # Add an event
        finding_workflow.add_comment("Test comment", user_id="user_1")

        assert len(finding_workflow.history) == 1

    def test_is_terminal_property(self, finding_workflow):
        """Test is_terminal property."""
        assert finding_workflow.is_terminal is False

        # Transition to terminal state
        finding_workflow.transition_to(
            WorkflowState.FALSE_POSITIVE,
            user_id="user_1",
        )

        assert finding_workflow.is_terminal is True

    def test_is_terminal_for_all_terminal_states(self):
        """Test is_terminal for all terminal states."""
        terminal_states = [
            WorkflowState.RESOLVED,
            WorkflowState.FALSE_POSITIVE,
            WorkflowState.ACCEPTED_RISK,
            WorkflowState.DUPLICATE,
        ]

        for state in terminal_states:
            workflow = make_workflow(finding_id="test", current_state=state)
            assert workflow.is_terminal is True, f"{state} should be terminal"


# ===========================================================================
# Tests: State Transitions
# ===========================================================================


class TestStateTransitions:
    """Tests for state transition functionality."""

    def test_can_transition_to_valid_state(self, finding_workflow):
        """Test can_transition_to for valid transitions."""
        assert finding_workflow.can_transition_to(WorkflowState.TRIAGING) is True
        assert finding_workflow.can_transition_to(WorkflowState.INVESTIGATING) is True
        assert finding_workflow.can_transition_to(WorkflowState.FALSE_POSITIVE) is True

    def test_can_transition_to_invalid_state(self, finding_workflow):
        """Test can_transition_to for invalid transitions."""
        assert finding_workflow.can_transition_to(WorkflowState.RESOLVED) is False
        assert finding_workflow.can_transition_to(WorkflowState.REMEDIATING) is False

    def test_get_valid_transitions(self, finding_workflow):
        """Test get_valid_transitions."""
        valid = finding_workflow.get_valid_transitions()

        assert WorkflowState.TRIAGING in valid
        assert WorkflowState.INVESTIGATING in valid
        assert WorkflowState.RESOLVED not in valid

    def test_transition_to_valid_state(self, finding_workflow):
        """Test transitioning to a valid state."""
        event = finding_workflow.transition_to(
            WorkflowState.TRIAGING,
            user_id="user_123",
            user_name="John Doe",
            comment="Starting triage",
        )

        assert finding_workflow.state == WorkflowState.TRIAGING
        assert event.event_type == WorkflowEventType.STATE_CHANGE
        assert event.from_state == WorkflowState.OPEN
        assert event.to_state == WorkflowState.TRIAGING
        assert event.user_id == "user_123"
        assert event.comment == "Starting triage"

    def test_transition_to_invalid_state_raises(self, finding_workflow):
        """Test that transitioning to invalid state raises error."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            finding_workflow.transition_to(
                WorkflowState.RESOLVED,
                user_id="user_123",
            )

        assert exc_info.value.from_state == WorkflowState.OPEN
        assert exc_info.value.to_state == WorkflowState.RESOLVED

    def test_transition_updates_timestamps(self, finding_workflow):
        """Test that transition updates timestamps."""
        original_updated = finding_workflow.data.updated_at

        finding_workflow.transition_to(
            WorkflowState.TRIAGING,
            user_id="user_123",
        )

        assert finding_workflow.data.updated_at >= original_updated

    def test_transition_adds_to_history(self, finding_workflow):
        """Test that transition adds event to history."""
        assert len(finding_workflow.history) == 0

        finding_workflow.transition_to(
            WorkflowState.TRIAGING,
            user_id="user_123",
        )

        assert len(finding_workflow.history) == 1
        assert finding_workflow.history[0].event_type == WorkflowEventType.STATE_CHANGE

    def test_transition_tracks_time_in_state(self, finding_workflow):
        """Test that transition tracks time in state."""
        # Transition to create time tracking
        finding_workflow.transition_to(
            WorkflowState.TRIAGING,
            user_id="user_123",
        )

        assert "open" in finding_workflow.data.time_in_states

    def test_transition_to_resolved_sets_resolved_at(self):
        """Test that transitioning to RESOLVED sets resolved_at."""
        workflow = make_workflow(finding_id="test", current_state=WorkflowState.REMEDIATING)

        assert workflow.data.resolved_at is None

        workflow.transition_to(
            WorkflowState.RESOLVED,
            user_id="user_123",
        )

        assert workflow.data.resolved_at is not None

    def test_full_workflow_path(self):
        """Test a complete workflow path from OPEN to RESOLVED."""
        workflow = make_workflow(finding_id="test_finding")

        # OPEN -> TRIAGING
        workflow.transition_to(WorkflowState.TRIAGING, user_id="user_1")
        assert workflow.state == WorkflowState.TRIAGING

        # TRIAGING -> INVESTIGATING
        workflow.transition_to(WorkflowState.INVESTIGATING, user_id="user_2")
        assert workflow.state == WorkflowState.INVESTIGATING

        # INVESTIGATING -> REMEDIATING
        workflow.transition_to(WorkflowState.REMEDIATING, user_id="user_3")
        assert workflow.state == WorkflowState.REMEDIATING

        # REMEDIATING -> RESOLVED
        workflow.transition_to(WorkflowState.RESOLVED, user_id="user_4")
        assert workflow.state == WorkflowState.RESOLVED
        assert workflow.is_terminal is True

        # Verify history
        assert len(workflow.history) == 4


# ===========================================================================
# Tests: Assignment
# ===========================================================================


class TestAssignment:
    """Tests for assignment functionality."""

    def test_assign_finding(self, finding_workflow):
        """Test assigning a finding to a user."""
        event = finding_workflow.assign(
            user_id="assignee_123",
            assigned_by="assigner_456",
            assigned_by_name="Manager",
            comment="Assigned for investigation",
        )

        assert finding_workflow.data.assigned_to == "assignee_123"
        assert finding_workflow.data.assigned_by == "assigner_456"
        assert finding_workflow.data.assigned_at is not None
        assert event.event_type == WorkflowEventType.ASSIGNMENT
        assert event.new_value == "assignee_123"

    def test_reassign_finding(self, finding_workflow):
        """Test reassigning a finding to a different user."""
        finding_workflow.assign(user_id="user_1", assigned_by="manager")

        event = finding_workflow.assign(
            user_id="user_2",
            assigned_by="manager",
        )

        assert finding_workflow.data.assigned_to == "user_2"
        assert event.old_value == "user_1"
        assert event.new_value == "user_2"

    def test_unassign_finding(self, finding_workflow):
        """Test unassigning a finding."""
        finding_workflow.assign(user_id="user_1", assigned_by="manager")

        event = finding_workflow.unassign(user_id="manager")

        assert finding_workflow.data.assigned_to is None
        assert finding_workflow.data.assigned_by is None
        assert finding_workflow.data.assigned_at is None
        assert event.old_value == "user_1"
        assert event.new_value is None


# ===========================================================================
# Tests: Comments
# ===========================================================================


class TestComments:
    """Tests for commenting functionality."""

    def test_add_comment(self, finding_workflow):
        """Test adding a comment."""
        event = finding_workflow.add_comment(
            "This is a test comment",
            user_id="user_123",
            user_name="John Doe",
        )

        assert event.event_type == WorkflowEventType.COMMENT
        assert event.comment == "This is a test comment"
        assert event.user_id == "user_123"
        assert event.user_name == "John Doe"

    def test_add_multiple_comments(self, finding_workflow):
        """Test adding multiple comments."""
        finding_workflow.add_comment("Comment 1", user_id="user_1")
        finding_workflow.add_comment("Comment 2", user_id="user_2")
        finding_workflow.add_comment("Comment 3", user_id="user_1")

        comments = finding_workflow.get_comments()

        assert len(comments) == 3

    def test_get_comments_empty(self, finding_workflow):
        """Test getting comments when there are none."""
        comments = finding_workflow.get_comments()

        assert comments == []


# ===========================================================================
# Tests: Priority Management
# ===========================================================================


class TestPriorityManagement:
    """Tests for priority management."""

    def test_set_priority(self, finding_workflow):
        """Test setting priority."""
        event = finding_workflow.set_priority(
            priority=1,
            user_id="user_123",
            comment="Escalated to highest priority",
        )

        assert finding_workflow.data.priority == 1
        assert event.event_type == WorkflowEventType.PRIORITY_CHANGE
        assert event.old_value == 3
        assert event.new_value == 1

    def test_set_priority_invalid_value(self, finding_workflow):
        """Test setting invalid priority raises error."""
        with pytest.raises(ValueError):
            finding_workflow.set_priority(priority=0, user_id="user_123")

        with pytest.raises(ValueError):
            finding_workflow.set_priority(priority=6, user_id="user_123")

    def test_priority_range(self, finding_workflow):
        """Test valid priority range."""
        for priority in [1, 2, 3, 4, 5]:
            finding_workflow.set_priority(priority=priority, user_id="user_123")
            assert finding_workflow.data.priority == priority


# ===========================================================================
# Tests: Due Date Management
# ===========================================================================


class TestDueDateManagement:
    """Tests for due date management."""

    def test_set_due_date(self, finding_workflow):
        """Test setting due date."""
        due_date = datetime.now(timezone.utc) + timedelta(days=7)

        event = finding_workflow.set_due_date(
            due_date=due_date,
            user_id="user_123",
            comment="Due in one week",
        )

        assert finding_workflow.data.due_date == due_date
        assert event.event_type == WorkflowEventType.DUE_DATE_CHANGE
        assert event.old_value is None
        assert event.new_value is not None

    def test_clear_due_date(self, finding_workflow):
        """Test clearing due date."""
        due_date = datetime.now(timezone.utc) + timedelta(days=7)
        finding_workflow.set_due_date(due_date=due_date, user_id="user_123")

        event = finding_workflow.set_due_date(
            due_date=None,
            user_id="user_123",
            comment="Removing deadline",
        )

        assert finding_workflow.data.due_date is None
        assert event.new_value is None


# ===========================================================================
# Tests: Linked Findings
# ===========================================================================


class TestLinkedFindings:
    """Tests for linked findings functionality."""

    def test_link_finding(self, finding_workflow):
        """Test linking to another finding."""
        event = finding_workflow.link_finding(
            linked_finding_id="related_finding_456",
            user_id="user_123",
            comment="Related issue",
        )

        assert "related_finding_456" in finding_workflow.data.linked_findings
        assert event.event_type == WorkflowEventType.LINKED
        assert event.new_value == "related_finding_456"

    def test_link_multiple_findings(self, finding_workflow):
        """Test linking to multiple findings."""
        finding_workflow.link_finding(linked_finding_id="finding_a", user_id="user_1")
        finding_workflow.link_finding(linked_finding_id="finding_b", user_id="user_1")
        finding_workflow.link_finding(linked_finding_id="finding_c", user_id="user_1")

        assert len(finding_workflow.data.linked_findings) == 3

    def test_link_same_finding_twice_no_duplicate(self, finding_workflow):
        """Test that linking same finding twice doesn't create duplicate."""
        finding_workflow.link_finding(linked_finding_id="finding_a", user_id="user_1")
        finding_workflow.link_finding(linked_finding_id="finding_a", user_id="user_1")

        assert len(finding_workflow.data.linked_findings) == 1


# ===========================================================================
# Tests: Mark as Duplicate
# ===========================================================================


class TestMarkDuplicate:
    """Tests for marking findings as duplicates."""

    def test_mark_duplicate(self, finding_workflow):
        """Test marking as duplicate of another finding."""
        event = finding_workflow.mark_duplicate(
            parent_finding_id="parent_finding_789",
            user_id="user_123",
            comment="Duplicate of existing issue",
        )

        assert finding_workflow.state == WorkflowState.DUPLICATE
        assert finding_workflow.data.parent_finding_id == "parent_finding_789"
        assert "parent_finding_789" in finding_workflow.data.linked_findings

    def test_mark_duplicate_is_terminal(self, finding_workflow):
        """Test that marking as duplicate moves to terminal state."""
        finding_workflow.mark_duplicate(
            parent_finding_id="parent_123",
            user_id="user_1",
        )

        assert finding_workflow.is_terminal is True


# ===========================================================================
# Tests: Severity Change
# ===========================================================================


class TestSeverityChange:
    """Tests for severity change tracking."""

    def test_change_severity(self, finding_workflow):
        """Test recording severity change."""
        event = finding_workflow.change_severity(
            new_severity="critical",
            old_severity="medium",
            user_id="user_123",
            comment="Upgraded after further analysis",
        )

        assert event.event_type == WorkflowEventType.SEVERITY_CHANGE
        assert event.old_value == "medium"
        assert event.new_value == "critical"
        assert event.field_name == "severity"


# ===========================================================================
# Tests: Transition Hooks
# ===========================================================================


class TestTransitionHooks:
    """Tests for transition hook functionality."""

    def test_add_transition_hook(self, finding_workflow):
        """Test adding a transition hook."""
        hook_called = []

        def my_hook(transition):
            hook_called.append(transition)

        finding_workflow.add_transition_hook(my_hook)

        finding_workflow.transition_to(
            WorkflowState.TRIAGING,
            user_id="user_123",
        )

        assert len(hook_called) == 1
        assert hook_called[0].from_state == WorkflowState.OPEN
        assert hook_called[0].to_state == WorkflowState.TRIAGING

    def test_multiple_transition_hooks(self, finding_workflow):
        """Test multiple transition hooks."""
        results = []

        def hook1(t):
            results.append("hook1")

        def hook2(t):
            results.append("hook2")

        finding_workflow.add_transition_hook(hook1)
        finding_workflow.add_transition_hook(hook2)

        finding_workflow.transition_to(WorkflowState.TRIAGING, user_id="user_1")

        assert results == ["hook1", "hook2"]

    def test_transition_hook_receives_metadata(self, finding_workflow):
        """Test that hooks receive metadata."""
        received_metadata = []

        def my_hook(transition):
            received_metadata.append(transition.metadata)

        finding_workflow.add_transition_hook(my_hook)

        finding_workflow.transition_to(
            WorkflowState.TRIAGING,
            user_id="user_123",
            metadata={"reason": "urgent"},
        )

        assert received_metadata[0] == {"reason": "urgent"}

    def test_hook_exception_does_not_prevent_transition(self, finding_workflow):
        """Test that hook exception doesn't prevent transition."""

        def failing_hook(transition):
            raise RuntimeError("Hook failed")

        finding_workflow.add_transition_hook(failing_hook)

        # Should not raise, transition should still happen
        finding_workflow.transition_to(WorkflowState.TRIAGING, user_id="user_1")

        assert finding_workflow.state == WorkflowState.TRIAGING


# ===========================================================================
# Tests: Time Tracking
# ===========================================================================


class TestTimeTracking:
    """Tests for time tracking functionality."""

    def test_get_time_to_resolution_not_resolved(self, finding_workflow):
        """Test getting time to resolution when not resolved."""
        result = finding_workflow.get_time_to_resolution()

        assert result is None

    def test_get_time_to_resolution_resolved(self):
        """Test getting time to resolution when resolved."""
        workflow = make_workflow(finding_id="test", current_state=WorkflowState.REMEDIATING)

        workflow.transition_to(WorkflowState.RESOLVED, user_id="user_1")

        result = workflow.get_time_to_resolution()

        assert result is not None
        assert result >= 0

    def test_time_in_states_tracked(self, finding_workflow):
        """Test that time in states is tracked."""
        finding_workflow.transition_to(WorkflowState.TRIAGING, user_id="user_1")

        assert "open" in finding_workflow.data.time_in_states
        assert finding_workflow.data.time_in_states["open"] >= 0


# ===========================================================================
# Tests: Get State Changes
# ===========================================================================


class TestGetStateChanges:
    """Tests for getting state change events."""

    def test_get_state_changes(self, finding_workflow):
        """Test getting state change events."""
        finding_workflow.transition_to(WorkflowState.TRIAGING, user_id="user_1")
        finding_workflow.add_comment("A comment", user_id="user_1")
        finding_workflow.transition_to(WorkflowState.INVESTIGATING, user_id="user_2")

        state_changes = finding_workflow.get_state_changes()

        assert len(state_changes) == 2
        assert all(e.event_type == WorkflowEventType.STATE_CHANGE for e in state_changes)


# ===========================================================================
# Tests: Workflow Diagram
# ===========================================================================


class TestWorkflowDiagram:
    """Tests for workflow diagram function."""

    def test_get_workflow_diagram(self):
        """Test getting workflow diagram."""
        diagram = get_workflow_diagram()

        assert "OPEN" in diagram
        assert "TRIAGING" in diagram
        assert "INVESTIGATING" in diagram
        # REMEDIATING might be wrapped across lines as "REMEDIAT" + "ING"
        assert "REMEDIAT" in diagram
        assert "RESOLVED" in diagram
        assert "FALSE_POSITIVE" in diagram or "FALSE" in diagram
        assert "Terminal States" in diagram or "terminal" in diagram.lower()


# ===========================================================================
# Tests: Legacy Status Mapping
# ===========================================================================


class TestLegacyStatusMapping:
    """Tests for legacy status mapping."""

    def test_map_legacy_status_open(self):
        """Test mapping legacy 'open' status."""
        assert map_legacy_status("open") == WorkflowState.OPEN

    def test_map_legacy_status_acknowledged(self):
        """Test mapping legacy 'acknowledged' status."""
        assert map_legacy_status("acknowledged") == WorkflowState.TRIAGING

    def test_map_legacy_status_resolved(self):
        """Test mapping legacy 'resolved' status."""
        assert map_legacy_status("resolved") == WorkflowState.RESOLVED

    def test_map_legacy_status_false_positive(self):
        """Test mapping legacy 'false_positive' status."""
        assert map_legacy_status("false_positive") == WorkflowState.FALSE_POSITIVE

    def test_map_legacy_status_wont_fix(self):
        """Test mapping legacy 'wont_fix' status."""
        assert map_legacy_status("wont_fix") == WorkflowState.ACCEPTED_RISK

    def test_map_legacy_status_unknown_defaults_to_open(self):
        """Test that unknown status defaults to OPEN."""
        assert map_legacy_status("unknown_status") == WorkflowState.OPEN

    def test_map_legacy_status_case_insensitive(self):
        """Test that mapping is case insensitive."""
        assert map_legacy_status("OPEN") == WorkflowState.OPEN
        assert map_legacy_status("Resolved") == WorkflowState.RESOLVED
        assert map_legacy_status("FALSE_POSITIVE") == WorkflowState.FALSE_POSITIVE


# ===========================================================================
# Tests: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_finding_id(self):
        """Test workflow with empty finding ID."""
        workflow = make_workflow(finding_id="")

        assert workflow.data.finding_id == ""
        assert workflow.state == WorkflowState.OPEN

    def test_long_comment(self, finding_workflow):
        """Test handling very long comments."""
        long_comment = "A" * 10000

        event = finding_workflow.add_comment(long_comment, user_id="user_1")

        assert event.comment == long_comment

    def test_special_characters_in_comment(self, finding_workflow):
        """Test handling special characters in comments."""
        comment = "Test <script>alert('xss')</script> & \"quotes\" 'apostrophes'"

        event = finding_workflow.add_comment(comment, user_id="user_1")

        assert event.comment == comment

    def test_transition_back_and_forth(self, finding_workflow):
        """Test transitioning back and forth between states."""
        finding_workflow.transition_to(WorkflowState.TRIAGING, user_id="user_1")
        finding_workflow.transition_to(WorkflowState.OPEN, user_id="user_1")
        finding_workflow.transition_to(WorkflowState.TRIAGING, user_id="user_1")

        assert finding_workflow.state == WorkflowState.TRIAGING
        assert len(finding_workflow.history) == 3

    def test_reopen_from_resolved(self):
        """Test reopening from resolved state."""
        workflow = make_workflow(finding_id="test", current_state=WorkflowState.RESOLVED)

        workflow.transition_to(WorkflowState.OPEN, user_id="user_1", comment="Issue recurred")

        assert workflow.state == WorkflowState.OPEN
        assert workflow.is_terminal is False

    def test_serialize_deserialize_roundtrip(self, finding_workflow):
        """Test serialization and deserialization roundtrip."""
        # Add some state
        finding_workflow.assign(user_id="user_1", assigned_by="manager")
        finding_workflow.transition_to(WorkflowState.TRIAGING, user_id="user_2")
        finding_workflow.add_comment("Test comment", user_id="user_3")
        finding_workflow.set_priority(1, user_id="user_4")

        # Serialize
        data_dict = finding_workflow.data.to_dict()

        # Deserialize
        new_data = FindingWorkflowData.from_dict(data_dict)
        new_workflow = FindingWorkflow(data=new_data)

        # Verify
        assert new_workflow.state == WorkflowState.TRIAGING
        assert new_workflow.data.assigned_to == "user_1"
        assert new_workflow.data.priority == 1
        assert len(new_workflow.history) == 4


# ===========================================================================
# Tests: Module Exports
# ===========================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_imports_work(self):
        """Test that all expected items can be imported."""
        from aragora.audit.findings.workflow import (
            FindingWorkflow,
            FindingWorkflowData,
            InvalidTransitionError,
            VALID_TRANSITIONS,
            WorkflowError,
            WorkflowEvent,
            WorkflowEventType,
            WorkflowState,
            WorkflowTransition,
            get_workflow_diagram,
            map_legacy_status,
        )

        assert WorkflowState is not None
        assert FindingWorkflow is not None
        assert WorkflowEvent is not None
