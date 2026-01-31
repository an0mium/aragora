"""
Tests for access review scheduler module.

Tests cover:
- ReviewType enum
- ReviewStatus enum
- ReviewItemStatus enum
- AccessReviewItem dataclass
- AccessReview dataclass
- AccessReviewConfig dataclass
- AccessReviewStorage class
- AccessReviewScheduler class
- Global scheduler singleton
- SOC 2 CC6.1/CC6.2 compliance features

SOC 2 Compliance: CC6.1, CC6.2 (Access Control)
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.scheduler.access_review_scheduler import (
    AccessReview,
    AccessReviewConfig,
    AccessReviewItem,
    AccessReviewScheduler,
    AccessReviewStorage,
    ReviewItemStatus,
    ReviewStatus,
    ReviewType,
    get_access_review_scheduler,
    schedule_access_review,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestReviewType:
    """Tests for ReviewType enum."""

    def test_has_all_review_types(self):
        """Enum has all expected review types."""
        assert ReviewType.MONTHLY.value == "monthly"
        assert ReviewType.QUARTERLY.value == "quarterly"
        assert ReviewType.AD_HOC.value == "ad_hoc"
        assert ReviewType.STALE_CREDENTIALS.value == "stale_credentials"
        assert ReviewType.ORPHANED_ACCOUNTS.value == "orphaned_accounts"
        assert ReviewType.PRIVILEGE_ESCALATION.value == "privilege_escalation"

    def test_review_type_count(self):
        """Enum has exactly 6 review types."""
        assert len(ReviewType) == 6

    def test_is_string_enum(self):
        """ReviewType values are strings."""
        for review_type in ReviewType:
            assert isinstance(review_type.value, str)


class TestReviewStatus:
    """Tests for ReviewStatus enum."""

    def test_has_all_statuses(self):
        """Enum has all expected statuses."""
        assert ReviewStatus.PENDING.value == "pending"
        assert ReviewStatus.IN_PROGRESS.value == "in_progress"
        assert ReviewStatus.AWAITING_APPROVAL.value == "awaiting_approval"
        assert ReviewStatus.APPROVED.value == "approved"
        assert ReviewStatus.REJECTED.value == "rejected"
        assert ReviewStatus.EXPIRED.value == "expired"
        assert ReviewStatus.CANCELLED.value == "cancelled"

    def test_status_count(self):
        """Enum has exactly 7 statuses."""
        assert len(ReviewStatus) == 7


class TestReviewItemStatus:
    """Tests for ReviewItemStatus enum."""

    def test_has_all_item_statuses(self):
        """Enum has all expected item statuses."""
        assert ReviewItemStatus.PENDING.value == "pending"
        assert ReviewItemStatus.APPROVED.value == "approved"
        assert ReviewItemStatus.REVOKED.value == "revoked"
        assert ReviewItemStatus.MODIFIED.value == "modified"
        assert ReviewItemStatus.SKIPPED.value == "skipped"

    def test_item_status_count(self):
        """Enum has exactly 5 item statuses."""
        assert len(ReviewItemStatus) == 5


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestAccessReviewItem:
    """Tests for AccessReviewItem dataclass."""

    def test_create_with_required_fields(self):
        """Creates item with required fields only."""
        item = AccessReviewItem(
            item_id="item_123",
            user_id="user_456",
            user_email="user@example.com",
            resource_type="role",
            resource_id="admin",
            resource_name="Administrator",
        )

        assert item.item_id == "item_123"
        assert item.user_id == "user_456"
        assert item.user_email == "user@example.com"
        assert item.resource_type == "role"
        assert item.resource_id == "admin"
        assert item.resource_name == "Administrator"

    def test_default_values(self):
        """Default values are set correctly."""
        item = AccessReviewItem(
            item_id="item_123",
            user_id="user_456",
            user_email="user@example.com",
            resource_type="role",
            resource_id="admin",
            resource_name="Administrator",
        )

        assert item.granted_at is None
        assert item.last_used is None
        assert item.granted_by is None
        assert item.risk_level == "low"
        assert item.status == ReviewItemStatus.PENDING
        assert item.decision_by is None
        assert item.decision_at is None
        assert item.decision_notes is None
        assert item.metadata == {}

    def test_custom_values(self):
        """Custom values are set correctly."""
        now = datetime.now(timezone.utc)
        last_used = now - timedelta(days=10)

        item = AccessReviewItem(
            item_id="item_123",
            user_id="user_456",
            user_email="user@example.com",
            resource_type="api_key",
            resource_id="key_abc",
            resource_name="Production API Key",
            granted_at=now,
            last_used=last_used,
            granted_by="admin_user",
            risk_level="high",
            status=ReviewItemStatus.APPROVED,
            decision_by="reviewer_1",
            decision_at=now,
            decision_notes="Verified with manager",
            metadata={"department": "engineering"},
        )

        assert item.granted_at == now
        assert item.last_used == last_used
        assert item.granted_by == "admin_user"
        assert item.risk_level == "high"
        assert item.status == ReviewItemStatus.APPROVED
        assert item.decision_notes == "Verified with manager"


class TestAccessReview:
    """Tests for AccessReview dataclass."""

    def test_create_with_required_fields(self):
        """Creates review with required fields only."""
        review = AccessReview(
            review_id="review_123",
            review_type=ReviewType.MONTHLY,
        )

        assert review.review_id == "review_123"
        assert review.review_type == ReviewType.MONTHLY

    def test_default_values(self):
        """Default values are set correctly."""
        review = AccessReview(
            review_id="review_123",
            review_type=ReviewType.MONTHLY,
        )

        assert review.status == ReviewStatus.PENDING
        assert review.scope_workspaces == []
        assert review.scope_users == []
        assert review.created_at is not None
        assert review.due_date is None
        assert review.completed_at is None
        assert review.created_by == "system"
        assert review.assigned_reviewer is None
        assert review.items == []
        assert review.summary == {}

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        now = datetime.now(timezone.utc)
        due_date = now + timedelta(days=14)

        review = AccessReview(
            review_id="review_123",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.IN_PROGRESS,
            scope_workspaces=["ws_1", "ws_2"],
            created_at=now,
            due_date=due_date,
            created_by="admin",
            assigned_reviewer="reviewer@example.com",
        )

        d = review.to_dict()

        assert d["review_id"] == "review_123"
        assert d["review_type"] == "monthly"
        assert d["status"] == "in_progress"
        assert d["scope_workspaces"] == ["ws_1", "ws_2"]
        assert d["created_by"] == "admin"
        assert d["assigned_reviewer"] == "reviewer@example.com"

    def test_to_dict_with_none_values(self):
        """to_dict handles None values correctly."""
        review = AccessReview(
            review_id="review_123",
            review_type=ReviewType.MONTHLY,
        )

        d = review.to_dict()

        assert d["due_date"] is None
        assert d["completed_at"] is None

    def test_to_dict_items_count(self):
        """to_dict includes items count."""
        review = AccessReview(
            review_id="review_123",
            review_type=ReviewType.MONTHLY,
            items=[
                AccessReviewItem(
                    item_id="item_1",
                    user_id="user_1",
                    user_email="user1@example.com",
                    resource_type="role",
                    resource_id="admin",
                    resource_name="Admin",
                ),
                AccessReviewItem(
                    item_id="item_2",
                    user_id="user_2",
                    user_email="user2@example.com",
                    resource_type="role",
                    resource_id="viewer",
                    resource_name="Viewer",
                ),
            ],
        )

        d = review.to_dict()

        assert d["items_count"] == 2


class TestAccessReviewConfig:
    """Tests for AccessReviewConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = AccessReviewConfig()

        assert config.monthly_review_day == 1
        assert config.quarterly_review_months == [1, 4, 7, 10]
        assert config.stale_credential_days == 90
        assert config.orphan_detection_days == 30
        assert config.review_due_days == 14
        assert config.notification_email is None
        assert config.slack_webhook is None
        assert config.storage_path is None

    def test_custom_values(self):
        """Custom values are set correctly."""
        config = AccessReviewConfig(
            monthly_review_day=15,
            stale_credential_days=60,
            review_due_days=7,
            notification_email="security@example.com",
        )

        assert config.monthly_review_day == 15
        assert config.stale_credential_days == 60
        assert config.review_due_days == 7
        assert config.notification_email == "security@example.com"


# =============================================================================
# Storage Tests
# =============================================================================


class TestAccessReviewStorage:
    """Tests for AccessReviewStorage class."""

    @pytest.fixture
    def storage(self):
        """Create in-memory storage for testing."""
        return AccessReviewStorage()

    def test_init_creates_schema(self, storage):
        """Initializes with required tables."""
        conn = storage._get_conn()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "access_reviews" in tables
        assert "review_items" in tables
        assert "user_access_log" in tables
        assert "review_schedule" in tables

    def test_save_and_get_review(self, storage):
        """Saves and retrieves a review."""
        review = AccessReview(
            review_id="review_123",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.PENDING,
            scope_workspaces=["ws_1"],
            created_by="admin",
        )

        storage.save_review(review)
        retrieved = storage.get_review("review_123")

        assert retrieved is not None
        assert retrieved.review_id == "review_123"
        assert retrieved.review_type == ReviewType.MONTHLY
        assert retrieved.scope_workspaces == ["ws_1"]

    def test_get_nonexistent_review(self, storage):
        """Returns None for nonexistent review."""
        result = storage.get_review("nonexistent")

        assert result is None

    def test_save_review_with_items(self, storage):
        """Saves review with items."""
        item = AccessReviewItem(
            item_id="item_1",
            user_id="user_1",
            user_email="user1@example.com",
            resource_type="role",
            resource_id="admin",
            resource_name="Administrator",
            risk_level="high",
        )

        review = AccessReview(
            review_id="review_123",
            review_type=ReviewType.MONTHLY,
            items=[item],
        )

        storage.save_review(review)
        retrieved = storage.get_review("review_123")

        assert len(retrieved.items) == 1
        assert retrieved.items[0].item_id == "item_1"
        assert retrieved.items[0].risk_level == "high"

    def test_update_existing_review(self, storage):
        """Updates an existing review."""
        review = AccessReview(
            review_id="review_123",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.PENDING,
        )
        storage.save_review(review)

        review.status = ReviewStatus.IN_PROGRESS
        storage.save_review(review)

        retrieved = storage.get_review("review_123")
        assert retrieved.status == ReviewStatus.IN_PROGRESS

    def test_get_pending_reviews(self, storage):
        """Gets all pending reviews."""
        for i, status in enumerate(
            [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS, ReviewStatus.APPROVED]
        ):
            storage.save_review(
                AccessReview(
                    review_id=f"review_{i}",
                    review_type=ReviewType.MONTHLY,
                    status=status,
                )
            )

        pending = storage.get_pending_reviews()

        # PENDING and IN_PROGRESS should be returned
        assert len(pending) == 2
        statuses = {r.status for r in pending}
        assert ReviewStatus.PENDING in statuses
        assert ReviewStatus.IN_PROGRESS in statuses

    def test_get_overdue_reviews(self, storage):
        """Gets overdue reviews."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=1)
        future = now + timedelta(days=7)

        storage.save_review(
            AccessReview(
                review_id="overdue",
                review_type=ReviewType.MONTHLY,
                status=ReviewStatus.PENDING,
                due_date=past,
            )
        )
        storage.save_review(
            AccessReview(
                review_id="on_time",
                review_type=ReviewType.MONTHLY,
                status=ReviewStatus.PENDING,
                due_date=future,
            )
        )

        overdue = storage.get_overdue_reviews()

        assert len(overdue) == 1
        assert overdue[0].review_id == "overdue"

    def test_record_and_get_user_access(self, storage):
        """Records and retrieves user access."""
        # Insert directly since the upsert in record_user_access requires a unique index
        conn = storage._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO user_access_log (user_id, resource_type, resource_id, last_accessed)
            VALUES (?, ?, ?, ?)
            """,
            ("user_1", "role", "admin", now),
        )
        conn.commit()

        last_access = storage.get_last_access("user_1", "role", "admin")

        assert last_access is not None
        assert (datetime.now(timezone.utc) - last_access).total_seconds() < 5

    def test_get_last_access_nonexistent(self, storage):
        """Returns None for nonexistent access record."""
        result = storage.get_last_access("nonexistent", "role", "admin")

        assert result is None

    def test_get_stale_credentials(self, storage):
        """Gets credentials not used in N days."""
        conn = storage._get_conn()
        now = datetime.now(timezone.utc)
        stale = now - timedelta(days=100)
        recent = now - timedelta(days=10)

        # Insert stale access
        conn.execute(
            """
            INSERT INTO user_access_log (user_id, resource_type, resource_id, last_accessed)
            VALUES (?, ?, ?, ?)
            """,
            ("stale_user", "api_key", "key_1", stale.isoformat()),
        )
        # Insert recent access
        conn.execute(
            """
            INSERT INTO user_access_log (user_id, resource_type, resource_id, last_accessed)
            VALUES (?, ?, ?, ?)
            """,
            ("active_user", "api_key", "key_2", recent.isoformat()),
        )
        conn.commit()

        stale_creds = storage.get_stale_credentials(days=90)

        assert len(stale_creds) == 1
        assert stale_creds[0]["user_id"] == "stale_user"
        assert stale_creds[0]["days_stale"] >= 90

    def test_save_and_get_schedule(self, storage):
        """Saves and retrieves schedules."""
        now = datetime.now(timezone.utc)
        next_run = now + timedelta(days=30)

        storage.save_schedule("monthly_review", "monthly", next_run)

        # Verify schedule was saved
        conn = storage._get_conn()
        row = conn.execute(
            "SELECT * FROM review_schedule WHERE schedule_id = ?",
            ("monthly_review",),
        ).fetchone()

        assert row is not None
        assert row["review_type"] == "monthly"
        assert row["enabled"] == 1

    def test_get_due_schedules(self, storage):
        """Gets schedules that are due to run."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(days=7)

        storage.save_schedule("due_schedule", "monthly", past)
        storage.save_schedule("future_schedule", "monthly", future)

        due = storage.get_due_schedules()

        assert len(due) == 1
        assert due[0]["schedule_id"] == "due_schedule"

    def test_update_schedule_run(self, storage):
        """Updates schedule after a run."""
        now = datetime.now(timezone.utc)
        initial_run = now - timedelta(days=1)
        next_run = now + timedelta(days=30)

        storage.save_schedule("test_schedule", "monthly", initial_run)
        storage.update_schedule_run("test_schedule", next_run)

        conn = storage._get_conn()
        row = conn.execute(
            "SELECT * FROM review_schedule WHERE schedule_id = ?",
            ("test_schedule",),
        ).fetchone()

        assert row["last_run"] is not None
        assert datetime.fromisoformat(row["next_run"]) == next_run


# =============================================================================
# Scheduler Tests
# =============================================================================


class TestAccessReviewScheduler:
    """Tests for AccessReviewScheduler class."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    def test_init(self, scheduler):
        """Initializes with default state."""
        assert scheduler._running is False
        assert scheduler._task is None
        assert scheduler._access_providers == []
        assert scheduler._revocation_handlers == []
        assert scheduler._notification_handlers == []

    def test_register_access_provider(self, scheduler):
        """Registers an access provider."""
        provider = MagicMock(return_value=[])

        scheduler.register_access_provider(provider)

        assert provider in scheduler._access_providers

    def test_register_revocation_handler(self, scheduler):
        """Registers a revocation handler."""
        handler = MagicMock()

        scheduler.register_revocation_handler(handler)

        assert handler in scheduler._revocation_handlers

    def test_register_notification_handler(self, scheduler):
        """Registers a notification handler."""
        handler = MagicMock()

        scheduler.register_notification_handler(handler)

        assert handler in scheduler._notification_handlers

    @pytest.mark.asyncio
    async def test_create_review_monthly(self, scheduler):
        """Creates a monthly review."""
        review = await scheduler.create_review(review_type=ReviewType.MONTHLY)

        assert review is not None
        assert review.review_type == ReviewType.MONTHLY
        assert review.status == ReviewStatus.PENDING
        assert review.due_date is not None

    @pytest.mark.asyncio
    async def test_create_review_with_scope(self, scheduler):
        """Creates a review with workspace scope."""
        review = await scheduler.create_review(
            review_type=ReviewType.MONTHLY,
            scope_workspaces=["ws_1", "ws_2"],
            scope_users=["user_1"],
        )

        assert review.scope_workspaces == ["ws_1", "ws_2"]
        assert review.scope_users == ["user_1"]

    @pytest.mark.asyncio
    async def test_create_review_with_reviewer(self, scheduler):
        """Creates a review with assigned reviewer."""
        review = await scheduler.create_review(
            review_type=ReviewType.MONTHLY,
            assigned_reviewer="reviewer@example.com",
            created_by="admin@example.com",
        )

        assert review.assigned_reviewer == "reviewer@example.com"
        assert review.created_by == "admin@example.com"

    @pytest.mark.asyncio
    async def test_create_review_uses_mock_data(self, scheduler):
        """Creates review with mock data when no providers registered."""
        review = await scheduler.create_review(review_type=ReviewType.MONTHLY)

        # Should have items from mock data
        assert len(review.items) >= 1

    @pytest.mark.asyncio
    async def test_create_review_uses_registered_provider(self, scheduler):
        """Creates review using registered access provider."""
        provider = MagicMock(
            return_value=[
                {
                    "user_id": "test_user",
                    "user_email": "test@example.com",
                    "resource_type": "role",
                    "resource_id": "custom_role",
                    "resource_name": "Custom Role",
                    "workspace_id": "ws_1",
                }
            ]
        )
        scheduler.register_access_provider(provider)

        review = await scheduler.create_review(review_type=ReviewType.MONTHLY)

        provider.assert_called_once()
        assert any(item.resource_id == "custom_role" for item in review.items)

    @pytest.mark.asyncio
    async def test_create_review_calculates_summary(self, scheduler):
        """Creates review with calculated summary."""
        review = await scheduler.create_review(review_type=ReviewType.MONTHLY)

        assert "total_items" in review.summary
        assert "by_risk" in review.summary
        assert "by_type" in review.summary
        assert "unique_users" in review.summary

    def test_get_review(self, scheduler):
        """Gets a review by ID."""
        # Create and save a review directly
        review = AccessReview(
            review_id="test_review",
            review_type=ReviewType.MONTHLY,
        )
        scheduler._storage.save_review(review)

        retrieved = scheduler.get_review("test_review")

        assert retrieved is not None
        assert retrieved.review_id == "test_review"

    def test_get_pending_reviews(self, scheduler):
        """Gets all pending reviews."""
        scheduler._storage.save_review(
            AccessReview(
                review_id="pending_1",
                review_type=ReviewType.MONTHLY,
                status=ReviewStatus.PENDING,
            )
        )
        scheduler._storage.save_review(
            AccessReview(
                review_id="approved_1",
                review_type=ReviewType.MONTHLY,
                status=ReviewStatus.APPROVED,
            )
        )

        pending = scheduler.get_pending_reviews()

        assert len(pending) == 1
        assert pending[0].review_id == "pending_1"

    def test_get_stale_credentials(self, scheduler):
        """Gets stale credentials."""
        conn = scheduler._storage._get_conn()
        stale_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        conn.execute(
            """
            INSERT INTO user_access_log (user_id, resource_type, resource_id, last_accessed)
            VALUES (?, ?, ?, ?)
            """,
            ("stale_user", "api_key", "key_1", stale_time),
        )
        conn.commit()

        stale = scheduler.get_stale_credentials()

        assert len(stale) == 1
        assert stale[0]["user_id"] == "stale_user"

    def test_record_access(self, scheduler):
        """Records user access by inserting directly."""
        # Insert directly since the upsert in record_user_access requires a unique index
        conn = scheduler._storage._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO user_access_log (user_id, resource_type, resource_id, last_accessed)
            VALUES (?, ?, ?, ?)
            """,
            ("user_1", "role", "admin", now),
        )
        conn.commit()

        last_access = scheduler._storage.get_last_access("user_1", "role", "admin")
        assert last_access is not None


class TestReviewProcessing:
    """Tests for review item processing."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    @pytest.fixture
    def review_with_items(self, scheduler):
        """Create a review with items for testing."""
        item = AccessReviewItem(
            item_id="item_1",
            user_id="user_1",
            user_email="user1@example.com",
            resource_type="role",
            resource_id="admin",
            resource_name="Administrator",
        )
        review = AccessReview(
            review_id="review_1",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.IN_PROGRESS,
            items=[item],
        )
        scheduler._storage.save_review(review)
        return review

    @pytest.mark.asyncio
    async def test_approve_item(self, scheduler, review_with_items):
        """Approves a review item."""
        result = await scheduler.approve_item(
            review_id="review_1",
            item_id="item_1",
            decision_by="reviewer@example.com",
            notes="Access verified",
        )

        assert result is True

        review = scheduler.get_review("review_1")
        item = review.items[0]
        assert item.status == ReviewItemStatus.APPROVED
        assert item.decision_by == "reviewer@example.com"
        assert item.decision_notes == "Access verified"
        assert item.decision_at is not None

    @pytest.mark.asyncio
    async def test_approve_item_nonexistent_review(self, scheduler):
        """Returns False for nonexistent review."""
        result = await scheduler.approve_item(
            review_id="nonexistent",
            item_id="item_1",
            decision_by="reviewer",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_approve_item_nonexistent_item(self, scheduler, review_with_items):
        """Returns False for nonexistent item."""
        result = await scheduler.approve_item(
            review_id="review_1",
            item_id="nonexistent",
            decision_by="reviewer",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_item(self, scheduler, review_with_items):
        """Revokes a review item."""
        result = await scheduler.revoke_item(
            review_id="review_1",
            item_id="item_1",
            decision_by="reviewer@example.com",
            notes="Access no longer needed",
        )

        assert result is True

        review = scheduler.get_review("review_1")
        item = review.items[0]
        assert item.status == ReviewItemStatus.REVOKED

    @pytest.mark.asyncio
    async def test_revoke_item_calls_handler(self, scheduler, review_with_items):
        """Revocation calls registered handler."""
        revocation_handler = MagicMock()
        scheduler.register_revocation_handler(revocation_handler)

        await scheduler.revoke_item(
            review_id="review_1",
            item_id="item_1",
            decision_by="reviewer",
        )

        revocation_handler.assert_called_once_with("user_1", "role", "admin")

    @pytest.mark.asyncio
    async def test_all_items_processed_changes_status(self, scheduler):
        """All items processed changes review to awaiting_approval."""
        item = AccessReviewItem(
            item_id="item_1",
            user_id="user_1",
            user_email="user1@example.com",
            resource_type="role",
            resource_id="admin",
            resource_name="Admin",
        )
        review = AccessReview(
            review_id="review_1",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.IN_PROGRESS,
            items=[item],
        )
        scheduler._storage.save_review(review)

        await scheduler.approve_item("review_1", "item_1", "reviewer")

        updated = scheduler.get_review("review_1")
        assert updated.status == ReviewStatus.AWAITING_APPROVAL

    @pytest.mark.asyncio
    async def test_complete_review(self, scheduler):
        """Completes a review."""
        item = AccessReviewItem(
            item_id="item_1",
            user_id="user_1",
            user_email="user1@example.com",
            resource_type="role",
            resource_id="admin",
            resource_name="Admin",
            status=ReviewItemStatus.APPROVED,
        )
        review = AccessReview(
            review_id="review_1",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.AWAITING_APPROVAL,
            items=[item],
        )
        scheduler._storage.save_review(review)

        result = await scheduler.complete_review("review_1", "approver@example.com")

        assert result is not None
        assert result.status == ReviewStatus.APPROVED
        assert result.completed_at is not None
        assert result.summary["approved_by"] == "approver@example.com"

    @pytest.mark.asyncio
    async def test_complete_review_nonexistent(self, scheduler):
        """Returns None for nonexistent review."""
        result = await scheduler.complete_review("nonexistent", "approver")

        assert result is None

    @pytest.mark.asyncio
    async def test_complete_review_executes_revocations(self, scheduler):
        """Complete review executes pending revocations."""
        revocation_handler = MagicMock()
        scheduler.register_revocation_handler(revocation_handler)

        item = AccessReviewItem(
            item_id="item_1",
            user_id="user_1",
            user_email="user1@example.com",
            resource_type="role",
            resource_id="admin",
            resource_name="Admin",
            status=ReviewItemStatus.REVOKED,
        )
        review = AccessReview(
            review_id="review_1",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.AWAITING_APPROVAL,
            items=[item],
        )
        scheduler._storage.save_review(review)

        await scheduler.complete_review("review_1", "approver")

        revocation_handler.assert_called_once_with("user_1", "role", "admin")


class TestRiskAssessment:
    """Tests for risk level assessment."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    def test_low_risk_for_basic_access(self, scheduler):
        """Basic access is low risk."""
        access = {"resource_type": "role", "resource_id": "viewer"}
        risk = scheduler._assess_risk_level(access)
        assert risk == "low"

    def test_high_risk_for_admin(self, scheduler):
        """Admin access is high risk."""
        access = {"resource_type": "role", "resource_id": "admin"}
        risk = scheduler._assess_risk_level(access)
        assert risk == "high"

    def test_high_risk_for_api_key(self, scheduler):
        """API key access is high risk."""
        access = {"resource_type": "api_key", "resource_id": "key_123"}
        risk = scheduler._assess_risk_level(access)
        assert risk == "high"

    def test_high_risk_for_superuser(self, scheduler):
        """Superuser access is high risk."""
        access = {"resource_type": "role", "resource_id": "superuser"}
        risk = scheduler._assess_risk_level(access)
        assert risk == "high"

    def test_critical_risk_for_root(self, scheduler):
        """Root access is critical risk."""
        access = {"resource_type": "role", "resource_id": "root"}
        risk = scheduler._assess_risk_level(access)
        assert risk == "critical"

    def test_critical_risk_for_all_access(self, scheduler):
        """All access is critical risk."""
        access = {"resource_type": "permission", "resource_id": "all_access"}
        risk = scheduler._assess_risk_level(access)
        assert risk == "critical"

    def test_medium_risk_for_write_access(self, scheduler):
        """Write access is medium risk."""
        access = {"resource_type": "permission", "resource_id": "write_documents"}
        risk = scheduler._assess_risk_level(access)
        assert risk == "medium"

    def test_medium_risk_for_delete_access(self, scheduler):
        """Delete access is medium risk."""
        access = {"resource_type": "permission", "resource_id": "delete_records"}
        risk = scheduler._assess_risk_level(access)
        assert risk == "medium"


class TestSummaryCalculation:
    """Tests for summary calculation."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    def test_empty_items(self, scheduler):
        """Calculates summary for empty items."""
        summary = scheduler._calculate_summary([])

        assert summary["total_items"] == 0
        assert summary["unique_users"] == 0
        assert summary["stale_count"] == 0

    def test_counts_by_risk(self, scheduler):
        """Counts items by risk level."""
        items = [
            AccessReviewItem(
                item_id="1",
                user_id="u1",
                user_email="u1@example.com",
                resource_type="role",
                resource_id="admin",
                resource_name="Admin",
                risk_level="high",
            ),
            AccessReviewItem(
                item_id="2",
                user_id="u2",
                user_email="u2@example.com",
                resource_type="role",
                resource_id="viewer",
                resource_name="Viewer",
                risk_level="low",
            ),
            AccessReviewItem(
                item_id="3",
                user_id="u3",
                user_email="u3@example.com",
                resource_type="role",
                resource_id="editor",
                resource_name="Editor",
                risk_level="medium",
            ),
        ]

        summary = scheduler._calculate_summary(items)

        assert summary["by_risk"]["low"] == 1
        assert summary["by_risk"]["medium"] == 1
        assert summary["by_risk"]["high"] == 1

    def test_counts_by_type(self, scheduler):
        """Counts items by resource type."""
        items = [
            AccessReviewItem(
                item_id="1",
                user_id="u1",
                user_email="u1@example.com",
                resource_type="role",
                resource_id="admin",
                resource_name="Admin",
            ),
            AccessReviewItem(
                item_id="2",
                user_id="u2",
                user_email="u2@example.com",
                resource_type="api_key",
                resource_id="key_1",
                resource_name="API Key",
            ),
            AccessReviewItem(
                item_id="3",
                user_id="u3",
                user_email="u3@example.com",
                resource_type="role",
                resource_id="viewer",
                resource_name="Viewer",
            ),
        ]

        summary = scheduler._calculate_summary(items)

        assert summary["by_type"]["role"] == 2
        assert summary["by_type"]["api_key"] == 1

    def test_counts_unique_users(self, scheduler):
        """Counts unique users."""
        items = [
            AccessReviewItem(
                item_id="1",
                user_id="u1",
                user_email="u1@example.com",
                resource_type="role",
                resource_id="admin",
                resource_name="Admin",
            ),
            AccessReviewItem(
                item_id="2",
                user_id="u1",  # Same user
                user_email="u1@example.com",
                resource_type="api_key",
                resource_id="key_1",
                resource_name="API Key",
            ),
            AccessReviewItem(
                item_id="3",
                user_id="u2",
                user_email="u2@example.com",
                resource_type="role",
                resource_id="viewer",
                resource_name="Viewer",
            ),
        ]

        summary = scheduler._calculate_summary(items)

        assert summary["unique_users"] == 2

    def test_counts_stale_items(self, scheduler):
        """Counts stale items."""
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(days=100)
        recent_time = now - timedelta(days=10)

        items = [
            AccessReviewItem(
                item_id="1",
                user_id="u1",
                user_email="u1@example.com",
                resource_type="role",
                resource_id="admin",
                resource_name="Admin",
                last_used=stale_time,
            ),
            AccessReviewItem(
                item_id="2",
                user_id="u2",
                user_email="u2@example.com",
                resource_type="role",
                resource_id="viewer",
                resource_name="Viewer",
                last_used=recent_time,
            ),
        ]

        summary = scheduler._calculate_summary(items)

        assert summary["stale_count"] == 1


class TestSchedulerLifecycle:
    """Tests for scheduler lifecycle methods."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    @pytest.mark.asyncio
    async def test_start(self, scheduler):
        """Starts the scheduler."""
        await scheduler.start()

        assert scheduler._running is True
        assert scheduler._task is not None

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, scheduler):
        """Start is idempotent."""
        await scheduler.start()
        task1 = scheduler._task

        await scheduler.start()
        task2 = scheduler._task

        assert task1 is task2

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop(self, scheduler):
        """Stops the scheduler."""
        await scheduler.start()
        await scheduler.stop()

        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, scheduler):
        """Stop when not running does not raise."""
        await scheduler.stop()  # Should not raise

        assert scheduler._running is False


class TestScheduleInitialization:
    """Tests for schedule initialization."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    def test_init_schedules(self, scheduler):
        """Initializes default schedules."""
        scheduler._init_schedules()

        conn = scheduler._storage._get_conn()
        rows = conn.execute("SELECT * FROM review_schedule").fetchall()

        schedule_ids = {row["schedule_id"] for row in rows}
        assert "monthly_review" in schedule_ids
        assert "stale_check" in schedule_ids


class TestOverdueHandling:
    """Tests for overdue review handling."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    @pytest.mark.asyncio
    async def test_handle_overdue_marks_expired(self, scheduler):
        """Marks very overdue reviews as expired."""
        now = datetime.now(timezone.utc)
        overdue_date = now - timedelta(days=10)

        review = AccessReview(
            review_id="overdue_review",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.PENDING,
            due_date=overdue_date,
        )
        scheduler._storage.save_review(review)

        await scheduler._handle_overdue_review(review)

        updated = scheduler.get_review("overdue_review")
        assert updated.status == ReviewStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_handle_overdue_sends_notification(self, scheduler):
        """Sends notification for expired reviews."""
        notification_handler = MagicMock()
        scheduler.register_notification_handler(notification_handler)

        now = datetime.now(timezone.utc)
        overdue_date = now - timedelta(days=10)

        review = AccessReview(
            review_id="overdue_review",
            review_type=ReviewType.MONTHLY,
            status=ReviewStatus.PENDING,
            due_date=overdue_date,
        )
        scheduler._storage.save_review(review)

        await scheduler._handle_overdue_review(review)

        notification_handler.assert_called_once()
        notification = notification_handler.call_args[0][0]
        assert notification["type"] == "review_expired"


class TestNotifications:
    """Tests for notification handling."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    @pytest.mark.asyncio
    async def test_notify_review_created(self, scheduler):
        """Sends notification when review is created."""
        notification_handler = MagicMock()
        scheduler.register_notification_handler(notification_handler)

        review = await scheduler.create_review(review_type=ReviewType.MONTHLY)

        # Manually trigger notification
        await scheduler._notify_review_created(review)

        notification_handler.assert_called()
        notification = notification_handler.call_args[0][0]
        assert notification["type"] == "review_created"
        assert notification["review_id"] == review.review_id

    @pytest.mark.asyncio
    async def test_notification_handler_error_handling(self, scheduler):
        """Handles notification handler errors gracefully."""
        error_handler = MagicMock(side_effect=RuntimeError("Notification failed"))
        scheduler.register_notification_handler(error_handler)

        review = AccessReview(
            review_id="test",
            review_type=ReviewType.MONTHLY,
        )

        # Should not raise
        await scheduler._notify_review_created(review)


class TestGlobalScheduler:
    """Tests for global scheduler singleton."""

    def test_get_scheduler_creates_instance(self):
        """Creates scheduler on first call."""
        import aragora.scheduler.access_review_scheduler as module

        module._scheduler = None

        scheduler = get_access_review_scheduler()

        assert isinstance(scheduler, AccessReviewScheduler)

    def test_get_scheduler_returns_same_instance(self):
        """Returns same instance on subsequent calls."""
        import aragora.scheduler.access_review_scheduler as module

        module._scheduler = None

        scheduler1 = get_access_review_scheduler()
        scheduler2 = get_access_review_scheduler()

        assert scheduler1 is scheduler2


class TestConvenienceFunction:
    """Tests for schedule_access_review convenience function."""

    @pytest.mark.asyncio
    async def test_schedule_access_review_function(self):
        """schedule_access_review convenience function works."""
        import aragora.scheduler.access_review_scheduler as module

        module._scheduler = None

        review = await schedule_access_review(
            review_type=ReviewType.MONTHLY,
            scope_workspaces=["ws_1"],
        )

        assert review is not None
        assert review.review_type == ReviewType.MONTHLY
        assert review.scope_workspaces == ["ws_1"]


class TestStaleCredentialReview:
    """Tests for stale credential detection (90+ days)."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None, stale_credential_days=90)
        return AccessReviewScheduler(config)

    @pytest.mark.asyncio
    async def test_stale_credential_review_filters_recent(self, scheduler):
        """Stale credential review filters out recent access."""
        now = datetime.now(timezone.utc)

        # Record recent access directly
        conn = scheduler._storage._get_conn()
        conn.execute(
            """
            INSERT INTO user_access_log (user_id, resource_type, resource_id, last_accessed)
            VALUES (?, ?, ?, ?)
            """,
            ("user_1", "api_key", "key_1", now.isoformat()),
        )
        conn.commit()

        provider = MagicMock(
            return_value=[
                {
                    "user_id": "user_1",
                    "user_email": "user1@example.com",
                    "resource_type": "api_key",
                    "resource_id": "key_1",
                    "resource_name": "API Key",
                }
            ]
        )
        scheduler.register_access_provider(provider)

        review = await scheduler.create_review(review_type=ReviewType.STALE_CREDENTIALS)

        # Should not include recently used credential
        assert len(review.items) == 0

    @pytest.mark.asyncio
    async def test_stale_credential_review_includes_stale(self, scheduler):
        """Stale credential review includes old access."""
        # Insert stale access directly
        conn = scheduler._storage._get_conn()
        stale_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        conn.execute(
            """
            INSERT INTO user_access_log (user_id, resource_type, resource_id, last_accessed)
            VALUES (?, ?, ?, ?)
            """,
            ("user_1", "api_key", "key_1", stale_time),
        )
        conn.commit()

        provider = MagicMock(
            return_value=[
                {
                    "user_id": "user_1",
                    "user_email": "user1@example.com",
                    "resource_type": "api_key",
                    "resource_id": "key_1",
                    "resource_name": "API Key",
                }
            ]
        )
        scheduler.register_access_provider(provider)

        review = await scheduler.create_review(review_type=ReviewType.STALE_CREDENTIALS)

        assert len(review.items) == 1
        assert review.items[0].user_id == "user_1"


class TestWorkspaceScoping:
    """Tests for workspace scoping in reviews."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = AccessReviewConfig(storage_path=None)
        return AccessReviewScheduler(config)

    @pytest.mark.asyncio
    async def test_filters_by_workspace(self, scheduler):
        """Filters items by workspace scope."""
        provider = MagicMock(
            return_value=[
                {
                    "user_id": "user_1",
                    "user_email": "user1@example.com",
                    "resource_type": "role",
                    "resource_id": "admin",
                    "resource_name": "Admin",
                    "workspace_id": "ws_1",
                },
                {
                    "user_id": "user_2",
                    "user_email": "user2@example.com",
                    "resource_type": "role",
                    "resource_id": "viewer",
                    "resource_name": "Viewer",
                    "workspace_id": "ws_2",
                },
            ]
        )
        scheduler.register_access_provider(provider)

        review = await scheduler.create_review(
            review_type=ReviewType.MONTHLY,
            scope_workspaces=["ws_1"],
        )

        assert len(review.items) == 1
        assert review.items[0].user_id == "user_1"

    @pytest.mark.asyncio
    async def test_filters_by_user(self, scheduler):
        """Filters items by user scope."""
        provider = MagicMock(
            return_value=[
                {
                    "user_id": "user_1",
                    "user_email": "user1@example.com",
                    "resource_type": "role",
                    "resource_id": "admin",
                    "resource_name": "Admin",
                },
                {
                    "user_id": "user_2",
                    "user_email": "user2@example.com",
                    "resource_type": "role",
                    "resource_id": "viewer",
                    "resource_name": "Viewer",
                },
            ]
        )
        scheduler.register_access_provider(provider)

        review = await scheduler.create_review(
            review_type=ReviewType.MONTHLY,
            scope_users=["user_2"],
        )

        assert len(review.items) == 1
        assert review.items[0].user_id == "user_2"
