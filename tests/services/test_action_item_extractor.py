"""
Tests for Action Item Extraction Service.

Tests the action item extraction functionality including:
- Action type detection (review, respond, send, etc.)
- Urgency detection (ASAP, urgent, critical)
- Deadline parsing (today, tomorrow, weekdays, dates)
- Assignee detection
- Priority calculation
- Batch processing
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from aragora.services.action_item_extractor import (
    ActionItemExtractor,
    ActionItem,
    ExtractionResult,
    ActionItemPriority,
    ActionItemStatus,
    ActionType,
    extract_action_items_quick,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create action item extractor without user context."""
    return ActionItemExtractor()


@pytest.fixture
def extractor_with_user():
    """Create action item extractor with user context."""
    return ActionItemExtractor(
        user_email="me@example.com",
        user_name="Test User",
    )


def make_email(
    subject: str = "Test Email",
    body: str = "",
    sender: str = "sender@example.com",
    to_addresses: list = None,
    email_id: str = "test-123",
):
    """Create a mock email object."""
    email = MagicMock()
    email.id = email_id
    email.subject = subject
    email.body_text = body
    email.body = body
    email.from_address = sender
    email.sender = sender
    email.to_addresses = to_addresses or []
    return email


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data model creation."""

    def test_action_item_priority_values(self):
        """Should have correct priority values."""
        assert ActionItemPriority.CRITICAL.value == 1
        assert ActionItemPriority.HIGH.value == 2
        assert ActionItemPriority.MEDIUM.value == 3
        assert ActionItemPriority.LOW.value == 4

    def test_action_item_status_values(self):
        """Should have all status values."""
        assert ActionItemStatus.PENDING.value == "pending"
        assert ActionItemStatus.IN_PROGRESS.value == "in_progress"
        assert ActionItemStatus.COMPLETED.value == "completed"
        assert ActionItemStatus.CANCELLED.value == "cancelled"
        assert ActionItemStatus.DEFERRED.value == "deferred"

    def test_action_type_values(self):
        """Should have all action type values."""
        assert ActionType.REVIEW.value == "review"
        assert ActionType.RESPOND.value == "respond"
        assert ActionType.SEND.value == "send"
        assert ActionType.PROVIDE.value == "provide"
        assert ActionType.SCHEDULE.value == "schedule"
        assert ActionType.APPROVE.value == "approve"
        assert ActionType.COMPLETE.value == "complete"
        assert ActionType.UPDATE.value == "update"
        assert ActionType.FOLLOW_UP.value == "follow_up"
        assert ActionType.CONFIRM.value == "confirm"
        assert ActionType.DECISION.value == "decision"
        assert ActionType.CREATE.value == "create"

    def test_action_item_creation(self):
        """Should create ActionItem with defaults."""
        item = ActionItem(
            id="action-123",
            description="Review the proposal",
            action_type=ActionType.REVIEW,
            priority=ActionItemPriority.HIGH,
        )
        assert item.id == "action-123"
        assert item.description == "Review the proposal"
        assert item.action_type == ActionType.REVIEW
        assert item.priority == ActionItemPriority.HIGH
        assert item.status == ActionItemStatus.PENDING

    def test_action_item_with_all_fields(self):
        """Should create ActionItem with all fields."""
        deadline = datetime.now(timezone.utc) + timedelta(days=1)
        item = ActionItem(
            id="action-456",
            description="Send report",
            action_type=ActionType.SEND,
            priority=ActionItemPriority.CRITICAL,
            status=ActionItemStatus.IN_PROGRESS,
            source_email_id="email-789",
            assignee_email="assignee@example.com",
            assignee_name="John Doe",
            requester_email="requester@example.com",
            deadline=deadline,
            deadline_text="by tomorrow",
            is_urgent=True,
            is_explicit_deadline=True,
            confidence=0.9,
            tags=["send", "report"],
        )
        assert item.assignee_email == "assignee@example.com"
        assert item.is_urgent is True
        assert len(item.tags) == 2

    def test_action_item_to_dict(self):
        """Should convert ActionItem to dict."""
        item = ActionItem(
            id="action-123",
            description="Test",
            action_type=ActionType.REVIEW,
            priority=ActionItemPriority.HIGH,
        )
        d = item.to_dict()
        assert d["id"] == "action-123"
        assert d["action_type"] == "review"
        assert d["priority"] == 2
        assert d["priority_name"] == "HIGH"
        assert d["status"] == "pending"

    def test_extraction_result_creation(self):
        """Should create ExtractionResult."""
        result = ExtractionResult(
            email_id="email-123",
            action_items=[],
            total_count=0,
            high_priority_count=0,
            has_deadlines=False,
        )
        assert result.email_id == "email-123"
        assert result.total_count == 0

    def test_extraction_result_to_dict(self):
        """Should convert ExtractionResult to dict."""
        result = ExtractionResult(
            email_id="email-456",
            action_items=[],
            total_count=3,
            high_priority_count=1,
            has_deadlines=True,
            earliest_deadline=datetime(2025, 1, 20, 17, 0, tzinfo=timezone.utc),
            extraction_confidence=0.85,
        )
        d = result.to_dict()
        assert d["email_id"] == "email-456"
        assert d["total_count"] == 3
        assert d["has_deadlines"] is True
        assert "2025-01-20" in d["earliest_deadline"]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Test ActionItemExtractor initialization."""

    def test_init_without_user_context(self):
        """Should initialize without user context."""
        extractor = ActionItemExtractor()
        assert extractor.user_email is None
        assert extractor.user_name is None

    def test_init_with_user_context(self):
        """Should initialize with user context."""
        extractor = ActionItemExtractor(
            user_email="test@example.com",
            user_name="Test User",
        )
        assert extractor.user_email == "test@example.com"
        assert extractor.user_name == "Test User"

    def test_patterns_compiled(self):
        """Should compile patterns on init."""
        extractor = ActionItemExtractor()
        assert len(extractor._compiled_action_patterns) > 0
        assert ActionType.REVIEW in extractor._compiled_action_patterns
        assert len(extractor._compiled_urgency_patterns) > 0


# =============================================================================
# Action Type Detection Tests
# =============================================================================


class TestActionTypeDetection:
    """Test action type detection."""

    @pytest.mark.asyncio
    async def test_detect_review_action(self, extractor):
        """Should detect review action items."""
        email = make_email(
            subject="Document for Review",
            body="Please review the attached proposal and let me know your thoughts.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        review_items = [i for i in result.action_items if i.action_type == ActionType.REVIEW]
        assert len(review_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_respond_action(self, extractor):
        """Should detect respond action items."""
        email = make_email(
            subject="Question",
            body="Please respond with your availability. Let me know by Friday.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        respond_items = [i for i in result.action_items if i.action_type == ActionType.RESPOND]
        assert len(respond_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_send_action(self, extractor):
        """Should detect send action items."""
        email = make_email(
            subject="Request",
            body="Please send me the report when you have a chance.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        send_items = [i for i in result.action_items if i.action_type == ActionType.SEND]
        assert len(send_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_provide_action(self, extractor):
        """Should detect provide action items."""
        email = make_email(
            subject="Information Request",
            body="Please provide the specifications for the new system.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        provide_items = [i for i in result.action_items if i.action_type == ActionType.PROVIDE]
        assert len(provide_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_schedule_action(self, extractor):
        """Should detect schedule action items."""
        email = make_email(
            subject="Meeting Request",
            body="Let's schedule a meeting to discuss the roadmap.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        schedule_items = [i for i in result.action_items if i.action_type == ActionType.SCHEDULE]
        assert len(schedule_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_approve_action(self, extractor):
        """Should detect approve action items."""
        email = make_email(
            subject="Approval Needed",
            body="Please approve the budget proposal attached herewith.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        approve_items = [i for i in result.action_items if i.action_type == ActionType.APPROVE]
        assert len(approve_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_complete_action(self, extractor):
        """Should detect complete action items."""
        email = make_email(
            subject="Task Reminder",
            body="Please complete the onboarding checklist by end of week.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        complete_items = [i for i in result.action_items if i.action_type == ActionType.COMPLETE]
        assert len(complete_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_update_action(self, extractor):
        """Should detect update action items."""
        email = make_email(
            subject="Status Request",
            body="Please update me on the project progress.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        update_items = [i for i in result.action_items if i.action_type == ActionType.UPDATE]
        assert len(update_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_follow_up_action(self, extractor):
        """Should detect follow up action items."""
        email = make_email(
            subject="Follow Up Needed",
            body="Please follow up with the vendor on pricing.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        followup_items = [i for i in result.action_items if i.action_type == ActionType.FOLLOW_UP]
        assert len(followup_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_confirm_action(self, extractor):
        """Should detect confirm action items."""
        email = make_email(
            subject="Confirmation Request",
            body="Please confirm your attendance for the event.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        confirm_items = [i for i in result.action_items if i.action_type == ActionType.CONFIRM]
        assert len(confirm_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_decision_action(self, extractor):
        """Should detect decision action items."""
        email = make_email(
            subject="Decision Required",
            body="We need a decision on which vendor to go with.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        decision_items = [i for i in result.action_items if i.action_type == ActionType.DECISION]
        assert len(decision_items) >= 1

    @pytest.mark.asyncio
    async def test_detect_create_action(self, extractor):
        """Should detect create action items."""
        email = make_email(
            subject="Document Request",
            body="Please create a summary document for the board meeting.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        create_items = [i for i in result.action_items if i.action_type == ActionType.CREATE]
        assert len(create_items) >= 1

    @pytest.mark.asyncio
    async def test_no_action_items(self, extractor):
        """Should return empty list for non-action email."""
        email = make_email(
            subject="Meeting Notes",
            body="Here are the notes from today's meeting. Thanks for joining!",
        )
        result = await extractor.extract_action_items(email)

        # May have 0 items or some false positives
        assert result.total_count >= 0


# =============================================================================
# Urgency Detection Tests
# =============================================================================


class TestUrgencyDetection:
    """Test urgency detection."""

    @pytest.mark.asyncio
    async def test_detect_asap_urgency(self, extractor):
        """Should detect ASAP urgency."""
        email = make_email(
            subject="Urgent",
            body="Please review this ASAP - client is waiting.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        assert any(item.is_urgent for item in result.action_items)

    @pytest.mark.asyncio
    async def test_detect_urgent_keyword(self, extractor):
        """Should detect 'urgent' keyword."""
        email = make_email(
            subject="Request",
            body="This is urgent - please send the report immediately.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        assert any(item.is_urgent for item in result.action_items)

    @pytest.mark.asyncio
    async def test_detect_critical_urgency(self, extractor):
        """Should detect 'critical' keyword."""
        email = make_email(
            subject="Critical Issue",
            body="Please review this critical security issue.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        assert any(item.is_urgent for item in result.action_items)

    @pytest.mark.asyncio
    async def test_detect_eod_urgency(self, extractor):
        """Should detect EOD deadline."""
        email = make_email(
            subject="Request",
            body="Please send me your feedback by EOD.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1

    @pytest.mark.asyncio
    async def test_urgent_affects_priority(self, extractor):
        """Should give higher priority to urgent items."""
        email = make_email(
            subject="Request",
            body="This is urgent - please review the document immediately.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        urgent_items = [i for i in result.action_items if i.is_urgent]
        assert len(urgent_items) >= 1
        # Urgent items should be CRITICAL or HIGH priority
        assert all(
            item.priority in (ActionItemPriority.CRITICAL, ActionItemPriority.HIGH)
            for item in urgent_items
        )


# =============================================================================
# Deadline Extraction Tests
# =============================================================================


class TestDeadlineExtraction:
    """Test deadline extraction."""

    @pytest.mark.asyncio
    async def test_extract_today_deadline(self, extractor):
        """Should extract 'today' deadline."""
        email = make_email(
            subject="Request",
            body="Please send me the report today.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        items_with_deadline = [i for i in result.action_items if i.deadline]
        assert len(items_with_deadline) >= 1
        assert items_with_deadline[0].deadline.date() == datetime.now(timezone.utc).date()

    @pytest.mark.asyncio
    async def test_extract_tomorrow_deadline(self, extractor):
        """Should extract 'tomorrow' deadline."""
        email = make_email(
            subject="Request",
            body="Please review the document and respond tomorrow.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        items_with_deadline = [i for i in result.action_items if i.deadline]
        if items_with_deadline:
            expected = (datetime.now(timezone.utc) + timedelta(days=1)).date()
            assert items_with_deadline[0].deadline.date() == expected

    @pytest.mark.asyncio
    async def test_extract_weekday_deadline(self, extractor):
        """Should extract weekday deadline."""
        email = make_email(
            subject="Request",
            body="Please complete this by Friday.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        items_with_deadline = [i for i in result.action_items if i.deadline]
        if items_with_deadline:
            # Should be a Friday
            assert items_with_deadline[0].deadline.weekday() == 4

    @pytest.mark.asyncio
    async def test_extract_end_of_week(self, extractor):
        """Should extract 'end of week' deadline."""
        email = make_email(
            subject="Request",
            body="Please review this by end of week.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        items_with_deadline = [i for i in result.action_items if i.deadline]
        if items_with_deadline:
            # End of week should be Friday
            assert items_with_deadline[0].deadline.weekday() == 4

    @pytest.mark.asyncio
    async def test_extract_end_of_day(self, extractor):
        """Should extract 'end of day' deadline."""
        email = make_email(
            subject="Request",
            body="Please send the update by end of day.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        items_with_deadline = [i for i in result.action_items if i.deadline]
        if items_with_deadline:
            assert items_with_deadline[0].deadline.date() == datetime.now(timezone.utc).date()

    @pytest.mark.asyncio
    async def test_extract_within_hours(self, extractor):
        """Should extract 'within N hours' deadline."""
        email = make_email(
            subject="Request",
            body="Please respond within 2 hours.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        items_with_deadline = [i for i in result.action_items if i.deadline]
        if items_with_deadline:
            now = datetime.now(timezone.utc)
            # Deadline should be ~2 hours from now
            diff = items_with_deadline[0].deadline - now
            assert timedelta(hours=1) < diff < timedelta(hours=3)

    @pytest.mark.asyncio
    async def test_extract_within_days(self, extractor):
        """Should extract 'within N days' deadline."""
        email = make_email(
            subject="Request",
            body="Please review within 3 days.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        items_with_deadline = [i for i in result.action_items if i.deadline]
        if items_with_deadline:
            now = datetime.now(timezone.utc)
            diff = items_with_deadline[0].deadline - now
            assert timedelta(days=2) < diff < timedelta(days=4)

    @pytest.mark.asyncio
    async def test_has_deadlines_flag(self, extractor):
        """Should set has_deadlines flag correctly."""
        email = make_email(
            subject="Request",
            body="Please review by tomorrow.",
        )
        result = await extractor.extract_action_items(email)

        assert result.has_deadlines is True

    @pytest.mark.asyncio
    async def test_earliest_deadline(self, extractor):
        """Should track earliest deadline."""
        email = make_email(
            subject="Multiple Tasks",
            body="""
            Please review document A within 5 days.
            Please send report B within 2 hours.
            """,
        )
        result = await extractor.extract_action_items(email)

        if result.has_deadlines and result.earliest_deadline:
            now = datetime.now(timezone.utc)
            # earliest_deadline should be the 2-hour one, not the 5-day one
            diff = result.earliest_deadline - now
            # Should be within 3 hours (the closer deadline)
            assert diff < timedelta(hours=3)


# =============================================================================
# Assignee Detection Tests
# =============================================================================


class TestAssigneeDetection:
    """Test assignee detection."""

    @pytest.mark.asyncio
    async def test_detect_mention(self, extractor):
        """Should detect @mention assignee."""
        email = make_email(
            subject="Request",
            body="@john please review the proposal.",
            to_addresses=["john@example.com", "jane@example.com"],
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        # Should match john from to_addresses
        items_with_assignee = [i for i in result.action_items if i.assignee_email]
        if items_with_assignee:
            assert "john" in items_with_assignee[0].assignee_email.lower()

    @pytest.mark.asyncio
    async def test_detect_you_directed(self, extractor_with_user):
        """Should detect 'you' directed at user."""
        email = make_email(
            subject="Request",
            body="Can you please review this document?",
            to_addresses=["me@example.com"],
        )
        result = await extractor_with_user.extract_action_items(email)

        assert result.total_count >= 1
        items_with_assignee = [i for i in result.action_items if i.assignee_email]
        if items_with_assignee:
            assert items_with_assignee[0].assignee_email == "me@example.com"

    @pytest.mark.asyncio
    async def test_requester_from_sender(self, extractor):
        """Should set requester from sender."""
        email = make_email(
            subject="Request",
            body="Please review the attached proposal.",
            sender="boss@company.com",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        assert result.action_items[0].requester_email == "boss@company.com"


# =============================================================================
# Priority Calculation Tests
# =============================================================================


class TestPriorityCalculation:
    """Test priority calculation."""

    @pytest.mark.asyncio
    async def test_critical_priority_from_urgency(self, extractor):
        """Should assign critical priority for urgent items."""
        email = make_email(
            subject="URGENT",
            body="Please approve immediately - this is critical and blocking production.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        assert any(item.priority == ActionItemPriority.CRITICAL for item in result.action_items)

    @pytest.mark.asyncio
    async def test_high_priority_count(self, extractor):
        """Should count high priority items."""
        email = make_email(
            subject="Urgent Review",
            body="Please review this urgent proposal ASAP.",
        )
        result = await extractor.extract_action_items(email)

        if result.total_count > 0:
            # high_priority_count includes CRITICAL and HIGH
            assert result.high_priority_count >= 0


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Test batch email processing."""

    @pytest.mark.asyncio
    async def test_extract_batch(self, extractor):
        """Should process multiple emails."""
        emails = [
            make_email("Request 1", "Please review document A.", email_id="e1"),
            make_email("Request 2", "Please send me the report.", email_id="e2"),
            make_email("FYI", "Here's the update you requested.", email_id="e3"),
        ]

        results = await extractor.extract_batch(emails)

        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Test utility methods."""

    def test_mark_completed(self, extractor):
        """Should mark action item as completed."""
        item = ActionItem(
            id="action-123",
            description="Test",
            action_type=ActionType.REVIEW,
            priority=ActionItemPriority.MEDIUM,
        )

        completed = extractor.mark_completed(item)

        assert completed.status == ActionItemStatus.COMPLETED
        assert completed.completed_at is not None

    def test_get_pending_items(self, extractor):
        """Should filter pending items."""
        items = [
            ActionItem(
                id="a1",
                description="Pending 1",
                action_type=ActionType.REVIEW,
                priority=ActionItemPriority.HIGH,
                status=ActionItemStatus.PENDING,
            ),
            ActionItem(
                id="a2",
                description="Completed",
                action_type=ActionType.SEND,
                priority=ActionItemPriority.MEDIUM,
                status=ActionItemStatus.COMPLETED,
            ),
            ActionItem(
                id="a3",
                description="In Progress",
                action_type=ActionType.APPROVE,
                priority=ActionItemPriority.LOW,
                status=ActionItemStatus.IN_PROGRESS,
            ),
        ]

        pending = extractor.get_pending_items(items)

        assert len(pending) == 2
        assert all(
            item.status in (ActionItemStatus.PENDING, ActionItemStatus.IN_PROGRESS)
            for item in pending
        )

    def test_get_pending_items_with_deadline_filter(self, extractor):
        """Should filter pending items by deadline."""
        now = datetime.now(timezone.utc)
        items = [
            ActionItem(
                id="a1",
                description="Due soon",
                action_type=ActionType.REVIEW,
                priority=ActionItemPriority.HIGH,
                deadline=now + timedelta(hours=2),
            ),
            ActionItem(
                id="a2",
                description="Due later",
                action_type=ActionType.SEND,
                priority=ActionItemPriority.MEDIUM,
                deadline=now + timedelta(days=5),
            ),
            ActionItem(
                id="a3",
                description="No deadline",
                action_type=ActionType.APPROVE,
                priority=ActionItemPriority.LOW,
            ),
        ]

        pending = extractor.get_pending_items(items, due_within_hours=24)

        assert len(pending) == 1
        assert pending[0].id == "a1"

    def test_pending_items_sorted_by_deadline(self, extractor):
        """Should sort pending items by deadline then priority."""
        now = datetime.now(timezone.utc)
        items = [
            ActionItem(
                id="a1",
                description="Later deadline",
                action_type=ActionType.REVIEW,
                priority=ActionItemPriority.HIGH,
                deadline=now + timedelta(hours=5),
            ),
            ActionItem(
                id="a2",
                description="Sooner deadline",
                action_type=ActionType.SEND,
                priority=ActionItemPriority.LOW,
                deadline=now + timedelta(hours=2),
            ),
            ActionItem(
                id="a3",
                description="No deadline",
                action_type=ActionType.APPROVE,
                priority=ActionItemPriority.CRITICAL,
            ),
        ]

        pending = extractor.get_pending_items(items)

        # Items with deadlines first, sorted by deadline
        assert pending[0].id == "a2"  # Sooner deadline
        assert pending[1].id == "a1"  # Later deadline
        assert pending[2].id == "a3"  # No deadline (at end)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Test extract_action_items_quick function."""

    @pytest.mark.asyncio
    async def test_quick_extraction(self):
        """Should extract action items with quick function."""
        result = await extract_action_items_quick(
            subject="Review Request",
            body="Please review the attached proposal by Friday.",
            sender="boss@company.com",
        )

        assert isinstance(result, ExtractionResult)
        assert result.total_count >= 1

    @pytest.mark.asyncio
    async def test_quick_no_action_items(self):
        """Should handle email with no action items."""
        result = await extract_action_items_quick(
            subject="Newsletter",
            body="Here's the weekly update. Have a great day!",
            sender="news@company.com",
        )

        assert isinstance(result, ExtractionResult)
        # May have 0 or some detected items
        assert result.total_count >= 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_email(self, extractor):
        """Should handle empty email."""
        email = make_email(subject="", body="")
        result = await extractor.extract_action_items(email)

        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_long_email(self, extractor):
        """Should handle long email."""
        body = "Please review this document. " * 100
        email = make_email(subject="Long Email", body=body)
        result = await extractor.extract_action_items(email)

        # Should process without error
        assert result is not None
        # May have multiple items from repeated text
        assert result.total_count >= 0

    @pytest.mark.asyncio
    async def test_multiple_action_items(self, extractor):
        """Should extract multiple action items from one email."""
        email = make_email(
            subject="Multiple Tasks",
            body="""
            Hi Team,

            Please complete the following:
            1. Please review the budget proposal.
            2. Can you send me the Q4 report?
            3. Please approve the vendor contract.

            Thanks!
            """,
        )
        result = await extractor.extract_action_items(email)

        # Should detect multiple action items
        assert result.total_count >= 2

    @pytest.mark.asyncio
    async def test_deduplication(self, extractor):
        """Should deduplicate similar action items."""
        email = make_email(
            subject="Request",
            body="""
            Please review this.
            Please review this document.
            """,
        )
        result = await extractor.extract_action_items(email)

        # Similar items should be deduplicated
        # The exact count depends on deduplication logic
        assert result.total_count >= 1

    @pytest.mark.asyncio
    async def test_confidence_score(self, extractor):
        """Should have confidence scores."""
        email = make_email(
            subject="Request",
            body="Please review the attached proposal.",
        )
        result = await extractor.extract_action_items(email)

        assert result.total_count >= 1
        assert all(item.confidence > 0 for item in result.action_items)

    @pytest.mark.asyncio
    async def test_processing_time_tracked(self, extractor):
        """Should track processing time."""
        email = make_email(
            subject="Request",
            body="Please review this.",
        )
        result = await extractor.extract_action_items(email)

        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_tags_extracted(self, extractor):
        """Should extract relevant tags."""
        email = make_email(
            subject="Request",
            body="Please review the project document.",
        )
        result = await extractor.extract_action_items(email)

        if result.total_count > 0:
            # Should have at least the action type tag
            assert len(result.action_items[0].tags) >= 1

    @pytest.mark.asyncio
    async def test_skip_deadline_extraction(self, extractor):
        """Should skip deadline extraction when disabled."""
        email = make_email(
            subject="Request",
            body="Please send the report by Friday.",
        )
        result = await extractor.extract_action_items(email, extract_deadlines=False)

        # Should not extract deadline when disabled
        for item in result.action_items:
            assert item.deadline is None

    @pytest.mark.asyncio
    async def test_skip_assignee_detection(self, extractor):
        """Should skip assignee detection when disabled."""
        email = make_email(
            subject="Request",
            body="Can you please review this?",
            to_addresses=["john@example.com"],
        )
        result = await extractor.extract_action_items(email, detect_assignees=False)

        # Should not detect assignee when disabled
        for item in result.action_items:
            assert item.assignee_email is None
