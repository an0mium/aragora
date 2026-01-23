"""
Tests for the email threading engine.

Tests:
- Thread reconstruction from headers
- Subject normalization
- Participant overlap matching
- Thread merging and splitting
- Summary generation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.services.email_threading import (
    EmailMessage,
    EmailThread,
    EmailThreader,
    ThreadSummary,
)


class TestEmailMessage:
    """Tests for EmailMessage dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = EmailMessage(
            message_id="msg-123",
            subject="Test Subject",
            sender="alice@example.com",
            recipients=["bob@example.com"],
        )
        assert msg.message_id == "msg-123"
        assert msg.sender == "alice@example.com"

    def test_message_to_dict(self):
        """Test serialization."""
        now = datetime.now(timezone.utc)
        msg = EmailMessage(
            message_id="msg-456",
            subject="Hello",
            sender="test@example.com",
            date=now,
        )
        data = msg.to_dict()
        assert data["message_id"] == "msg-456"
        assert data["subject"] == "Hello"
        assert data["date"] == now.isoformat()


class TestEmailThread:
    """Tests for EmailThread dataclass."""

    def test_create_thread(self):
        """Test creating a thread."""
        thread = EmailThread(
            thread_id="thread-123",
            subject="test subject",
        )
        assert thread.thread_id == "thread-123"
        assert thread.message_count == 0

    def test_thread_to_dict(self):
        """Test serialization."""
        thread = EmailThread(
            thread_id="thread-456",
            subject="discussion",
            participants={"alice@example.com", "bob@example.com"},
            message_count=3,
        )
        data = thread.to_dict()
        assert data["thread_id"] == "thread-456"
        assert len(data["participants"]) == 2
        assert data["message_count"] == 3


class TestThreadSummary:
    """Tests for ThreadSummary dataclass."""

    def test_create_summary(self):
        """Test creating a summary."""
        summary = ThreadSummary(
            thread_id="thread-123",
            summary="Discussion about project timeline",
            key_points=["Deadline extended", "Budget approved"],
        )
        assert summary.thread_id == "thread-123"
        assert len(summary.key_points) == 2

    def test_summary_to_dict(self):
        """Test serialization."""
        summary = ThreadSummary(
            thread_id="thread-456",
            summary="Review meeting notes",
            urgency="high",
        )
        data = summary.to_dict()
        assert data["urgency"] == "high"


class TestEmailThreader:
    """Tests for EmailThreader."""

    @pytest.fixture
    def threader(self):
        """Create a threader for testing."""
        return EmailThreader()

    def test_thread_single_message(self, threader):
        """Test threading a single message."""
        msg = EmailMessage(
            message_id="msg-1",
            subject="Hello",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=datetime.now(timezone.utc),
        )
        threads = threader.thread_emails([msg])
        assert len(threads) == 1
        assert threads[0].message_count == 1

    def test_thread_by_in_reply_to(self, threader):
        """Test threading using In-Reply-To header."""
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Hello",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=now,
        )
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Re: Hello",
            sender="bob@example.com",
            recipients=["alice@example.com"],
            in_reply_to="msg-1",
            date=now + timedelta(hours=1),
        )
        threads = threader.thread_emails([msg1, msg2])
        assert len(threads) == 1
        assert threads[0].message_count == 2

    def test_thread_by_references(self, threader):
        """Test threading using References header."""
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Project Discussion",
            sender="alice@example.com",
            recipients=["team@example.com"],
            date=now,
        )
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Re: Project Discussion",
            sender="bob@example.com",
            recipients=["team@example.com"],
            references=["msg-1"],
            date=now + timedelta(hours=1),
        )
        msg3 = EmailMessage(
            message_id="msg-3",
            subject="Re: Re: Project Discussion",
            sender="charlie@example.com",
            recipients=["team@example.com"],
            references=["msg-1", "msg-2"],
            date=now + timedelta(hours=2),
        )
        threads = threader.thread_emails([msg1, msg2, msg3])
        assert len(threads) == 1
        assert threads[0].message_count == 3

    def test_thread_by_subject(self, threader):
        """Test threading by normalized subject."""
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Meeting Tomorrow",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=now,
        )
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Re: Meeting Tomorrow",
            sender="bob@example.com",
            recipients=["alice@example.com"],
            date=now + timedelta(hours=1),
        )
        threads = threader.thread_emails([msg1, msg2])
        assert len(threads) == 1

    def test_subject_normalization(self, threader):
        """Test subject normalization."""
        assert threader._normalize_subject("RE: Hello") == "hello"
        assert threader._normalize_subject("Re: Re: Hello") == "hello"
        assert threader._normalize_subject("Fwd: Hello") == "hello"
        assert threader._normalize_subject("FW: Hello") == "hello"
        assert threader._normalize_subject("[dev] Project Update") == "project update"

    def test_separate_threads_different_subjects(self, threader):
        """Test that different subjects create separate threads."""
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Topic A",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=now,
        )
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Topic B",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=now + timedelta(hours=1),
        )
        threads = threader.thread_emails([msg1, msg2])
        assert len(threads) == 2

    def test_participant_tracking(self, threader):
        """Test that participants are tracked."""
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Discussion",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            cc=["charlie@example.com"],
            date=now,
        )
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Re: Discussion",
            sender="bob@example.com",
            recipients=["alice@example.com", "dave@example.com"],
            in_reply_to="msg-1",
            date=now + timedelta(hours=1),
        )
        threads = threader.thread_emails([msg1, msg2])
        assert len(threads) == 1
        participants = threads[0].participants
        assert "alice@example.com" in participants
        assert "bob@example.com" in participants
        assert "charlie@example.com" in participants
        assert "dave@example.com" in participants

    def test_unread_count(self, threader):
        """Test unread message counting."""
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Test",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=now,
            labels=["UNREAD"],
        )
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Re: Test",
            sender="bob@example.com",
            recipients=["alice@example.com"],
            in_reply_to="msg-1",
            date=now + timedelta(hours=1),
            labels=[],
        )
        threads = threader.thread_emails([msg1, msg2])
        assert threads[0].unread_count == 1

    def test_date_tracking(self, threader):
        """Test first/last message date tracking."""
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Thread",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=now,
        )
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Re: Thread",
            sender="bob@example.com",
            recipients=["alice@example.com"],
            in_reply_to="msg-1",
            date=now + timedelta(hours=5),
        )
        threads = threader.thread_emails([msg1, msg2])
        assert threads[0].first_message_date == now
        assert threads[0].last_message_date == now + timedelta(hours=5)


class TestThreadMergeAndSplit:
    """Tests for merge and split operations."""

    @pytest.fixture
    def threader(self):
        """Create a threader with some threads."""
        threader = EmailThreader()
        now = datetime.now(timezone.utc)

        # Create thread 1
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Topic A",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=now,
        )
        # Create thread 2
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Topic B",
            sender="charlie@example.com",
            recipients=["dave@example.com"],
            date=now + timedelta(hours=1),
        )
        threader.thread_emails([msg1, msg2])
        return threader

    def test_merge_threads(self, threader):
        """Test merging two threads."""
        thread_ids = list(threader._threads.keys())
        assert len(thread_ids) == 2

        merged = threader.merge_threads(thread_ids[0], thread_ids[1])
        assert merged is not None
        assert merged.message_count == 2
        assert len(threader._threads) == 1

    def test_merge_nonexistent_thread(self, threader):
        """Test merging with nonexistent thread."""
        thread_ids = list(threader._threads.keys())
        result = threader.merge_threads(thread_ids[0], "nonexistent")
        assert result is None

    def test_split_thread(self, threader):
        """Test splitting a thread."""
        # First create a multi-message thread
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="split-msg-1",
            subject="Long Thread",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date=now,
        )
        msg2 = EmailMessage(
            message_id="split-msg-2",
            subject="Re: Long Thread",
            sender="bob@example.com",
            recipients=["alice@example.com"],
            in_reply_to="split-msg-1",
            date=now + timedelta(hours=1),
        )
        msg3 = EmailMessage(
            message_id="split-msg-3",
            subject="Re: Re: Long Thread",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            in_reply_to="split-msg-2",
            date=now + timedelta(hours=2),
        )
        threader.thread_emails([msg1, msg2, msg3])

        # Find the thread with these messages
        thread_id = None
        for tid, thread in threader._threads.items():
            if any(m.message_id == "split-msg-1" for m in thread.messages):
                thread_id = tid
                break

        assert thread_id is not None
        initial_count = len(threader._threads)

        # Split out the last message
        new_thread = threader.split_thread(thread_id, ["split-msg-3"])
        assert new_thread is not None
        assert len(threader._threads) == initial_count + 1

    def test_split_nonexistent_thread(self, threader):
        """Test splitting nonexistent thread."""
        result = threader.split_thread("nonexistent", ["msg-1"])
        assert result is None


class TestThreadSummaryGeneration:
    """Tests for thread summary generation."""

    @pytest.fixture
    def threader(self):
        """Create a threader for testing."""
        return EmailThreader()

    def test_heuristic_summary(self, threader):
        """Test heuristic summary generation."""
        now = datetime.now(timezone.utc)
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Project Planning",
            sender="alice@example.com",
            recipients=["bob@example.com", "charlie@example.com"],
            body_preview="Hi team, let's discuss the project timeline.",
            date=now,
        )
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Re: Project Planning",
            sender="bob@example.com",
            recipients=["alice@example.com", "charlie@example.com"],
            body_preview="Please review the attached doc. Could you send feedback by Friday?",
            in_reply_to="msg-1",
            date=now + timedelta(days=1),
        )
        threads = threader.thread_emails([msg1, msg2])
        summary = threader.get_thread_summary(threads[0])

        assert "project planning" in summary.summary.lower()
        assert summary.thread_id == threads[0].thread_id

    def test_empty_thread_summary(self, threader):
        """Test summary of empty thread."""
        thread = EmailThread(
            thread_id="empty-thread",
            subject="Empty",
        )
        summary = threader._heuristic_summarize(thread)
        assert "empty" in summary.summary.lower()

    def test_urgency_detection_in_summary(self, threader):
        """Test urgency detection in summary."""
        now = datetime.now(timezone.utc)
        msg = EmailMessage(
            message_id="urgent-msg",
            subject="URGENT: Need response",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            body_preview="This is urgent! We need this ASAP!",
            date=now,
        )
        threads = threader.thread_emails([msg])
        summary = threader.get_thread_summary(threads[0])
        assert summary.urgency == "high"


class TestRelatedThreads:
    """Tests for finding related threads."""

    @pytest.fixture
    def threader(self):
        """Create a threader with related threads."""
        threader = EmailThreader()
        now = datetime.now(timezone.utc)

        # Thread 1: Project Alpha
        msg1 = EmailMessage(
            message_id="msg-1",
            subject="Project Alpha Planning",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            labels=["project", "planning"],
            date=now,
        )
        # Thread 2: Related to Project Alpha (same participants)
        msg2 = EmailMessage(
            message_id="msg-2",
            subject="Project Alpha Budget",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            labels=["project", "budget"],
            date=now + timedelta(hours=1),
        )
        # Thread 3: Different participants
        msg3 = EmailMessage(
            message_id="msg-3",
            subject="Different Topic",
            sender="charlie@example.com",
            recipients=["dave@example.com"],
            date=now + timedelta(hours=2),
        )
        threader.thread_emails([msg1, msg2, msg3])
        return threader

    def test_find_related_by_participants(self, threader):
        """Test finding related threads by participant overlap."""
        thread_ids = list(threader._threads.keys())

        # Find thread with Alice
        alice_thread_id = None
        for tid, thread in threader._threads.items():
            if "alice@example.com" in thread.participants:
                alice_thread_id = tid
                break

        related = threader.find_related_threads(alice_thread_id, max_results=5)

        # Should find at least one related thread
        assert len(related) >= 1

    def test_find_related_excludes_self(self, threader):
        """Test that related threads excludes the source thread."""
        thread_ids = list(threader._threads.keys())
        related = threader.find_related_threads(thread_ids[0], max_results=5)

        # Should not include self
        assert not any(t.thread_id == thread_ids[0] for t in related)

    def test_find_related_nonexistent(self, threader):
        """Test finding related for nonexistent thread."""
        related = threader.find_related_threads("nonexistent")
        assert related == []
