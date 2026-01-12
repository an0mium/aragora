"""
Tests for aragora.debate.wisdom_injector - Audience wisdom injection.

Tests cover:
- WisdomSubmission dataclass
- WisdomInjection dataclass
- WisdomInjector initialization
- submit_wisdom() async method
- upvote_wisdom() method
- _calculate_relevance() internal method
- find_relevant_wisdom() method
- inject_wisdom() method
- format_for_prompt() method
- get_stats() method
- Global instance management
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from aragora.debate.wisdom_injector import (
    WisdomSubmission,
    WisdomInjection,
    WisdomInjector,
    get_wisdom_injector,
    close_wisdom_injector,
    _injectors,
)


# Fixtures
@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage directory."""
    storage = tmp_path / "wisdom"
    storage.mkdir()
    return storage


@pytest.fixture
def injector(temp_storage):
    """Create a WisdomInjector for testing."""
    return WisdomInjector(loop_id="test-loop-001", storage_path=temp_storage)


@pytest.fixture
def sample_wisdom():
    """Create a sample WisdomSubmission."""
    return WisdomSubmission(
        id="abc123",
        text="Consider performance implications",
        submitter_id="user-001",
        timestamp=time.time(),
        loop_id="test-loop",
        context_tags=["performance", "architecture"],
    )


@pytest.fixture(autouse=True)
def cleanup_global_injectors():
    """Clean up global injectors after each test."""
    yield
    _injectors.clear()


class TestWisdomSubmission:
    """Tests for WisdomSubmission dataclass."""

    def test_required_fields(self):
        """Should create with required fields."""
        ws = WisdomSubmission(
            id="test-id",
            text="Test wisdom",
            submitter_id="user-123",
            timestamp=1000.0,
            loop_id="loop-001",
        )

        assert ws.id == "test-id"
        assert ws.text == "Test wisdom"
        assert ws.submitter_id == "user-123"
        assert ws.timestamp == 1000.0
        assert ws.loop_id == "loop-001"

    def test_optional_fields_defaults(self):
        """Optional fields should have correct defaults."""
        ws = WisdomSubmission(
            id="test-id",
            text="Test",
            submitter_id="user",
            timestamp=0.0,
            loop_id="loop",
        )

        assert ws.context_tags == []
        assert ws.relevance_score == 0.0
        assert ws.used is False
        assert ws.used_at is None
        assert ws.upvotes == 0

    def test_optional_fields_set(self):
        """Should accept optional field values."""
        ws = WisdomSubmission(
            id="test",
            text="Test",
            submitter_id="user",
            timestamp=100.0,
            loop_id="loop",
            context_tags=["tag1", "tag2"],
            relevance_score=0.75,
            used=True,
            used_at=200.0,
            upvotes=5,
        )

        assert ws.context_tags == ["tag1", "tag2"]
        assert ws.relevance_score == 0.75
        assert ws.used is True
        assert ws.used_at == 200.0
        assert ws.upvotes == 5

    def test_to_dict(self):
        """Should serialize to dictionary."""
        ws = WisdomSubmission(
            id="test-id",
            text="Wisdom text",
            submitter_id="user-001",
            timestamp=1234.56,
            loop_id="loop-001",
            context_tags=["ai", "debate"],
            upvotes=3,
        )

        d = ws.to_dict()

        assert d["id"] == "test-id"
        assert d["text"] == "Wisdom text"
        assert d["submitter_id"] == "user-001"
        assert d["timestamp"] == 1234.56
        assert d["context_tags"] == ["ai", "debate"]
        assert d["upvotes"] == 3

    def test_generate_id_deterministic(self):
        """generate_id should produce consistent IDs for same input."""
        id1 = WisdomSubmission.generate_id("text", "user", 1000.0)
        id2 = WisdomSubmission.generate_id("text", "user", 1000.0)

        assert id1 == id2

    def test_generate_id_unique_for_different_inputs(self):
        """generate_id should produce different IDs for different inputs."""
        id1 = WisdomSubmission.generate_id("text1", "user", 1000.0)
        id2 = WisdomSubmission.generate_id("text2", "user", 1000.0)
        id3 = WisdomSubmission.generate_id("text1", "other", 1000.0)
        id4 = WisdomSubmission.generate_id("text1", "user", 2000.0)

        assert len({id1, id2, id3, id4}) == 4

    def test_generate_id_length(self):
        """generate_id should return 12-char hash."""
        id_ = WisdomSubmission.generate_id("test", "user", 123.0)
        assert len(id_) == 12


class TestWisdomInjection:
    """Tests for WisdomInjection dataclass."""

    def test_required_fields(self):
        """Should create with required fields."""
        inj = WisdomInjection(
            wisdom_id="wis-001",
            agent_context="Claude was considering architecture",
            injection_reason="timeout",
            timestamp=1234.5,
        )

        assert inj.wisdom_id == "wis-001"
        assert inj.agent_context == "Claude was considering architecture"
        assert inj.injection_reason == "timeout"
        assert inj.timestamp == 1234.5

    def test_impact_score_default(self):
        """impact_score should default to 0.0."""
        inj = WisdomInjection(
            wisdom_id="wis",
            agent_context="context",
            injection_reason="stall",
            timestamp=100.0,
        )

        assert inj.impact_score == 0.0

    def test_impact_score_set(self):
        """Should accept custom impact_score."""
        inj = WisdomInjection(
            wisdom_id="wis",
            agent_context="context",
            injection_reason="requested",
            timestamp=100.0,
            impact_score=0.85,
        )

        assert inj.impact_score == 0.85

    def test_to_dict(self):
        """Should serialize to dictionary."""
        inj = WisdomInjection(
            wisdom_id="wis-002",
            agent_context="test context",
            injection_reason="timeout",
            timestamp=999.9,
            impact_score=0.5,
        )

        d = inj.to_dict()

        assert d["wisdom_id"] == "wis-002"
        assert d["agent_context"] == "test context"
        assert d["injection_reason"] == "timeout"
        assert d["timestamp"] == 999.9
        assert d["impact_score"] == 0.5


class TestWisdomInjectorInit:
    """Tests for WisdomInjector initialization."""

    def test_init_with_loop_id(self, temp_storage):
        """Should initialize with loop_id."""
        inj = WisdomInjector(loop_id="test-loop", storage_path=temp_storage)

        assert inj.loop_id == "test-loop"

    def test_init_creates_storage_directory(self, tmp_path):
        """Should create storage directory if needed."""
        storage = tmp_path / "new" / "storage"
        assert not storage.exists()

        inj = WisdomInjector(loop_id="test", storage_path=storage)

        assert storage.exists()
        assert inj.storage_path == storage

    def test_init_default_storage_path(self, tmp_path, monkeypatch):
        """Should use default storage path if not provided."""
        monkeypatch.chdir(tmp_path)

        inj = WisdomInjector(loop_id="test")

        expected = Path(".nomic/wisdom")
        assert inj.storage_path == expected

    def test_init_empty_lists(self, temp_storage):
        """Should initialize with empty lists."""
        inj = WisdomInjector(loop_id="test", storage_path=temp_storage)

        assert inj.pending_wisdom == []
        assert inj.used_wisdom == []
        assert inj.injections == []

    def test_init_empty_stats(self, temp_storage):
        """Should initialize with empty submitter stats."""
        inj = WisdomInjector(loop_id="test", storage_path=temp_storage)

        assert len(inj.submitter_stats) == 0


class TestWisdomInjectorPersistence:
    """Tests for wisdom persistence (load/save)."""

    def test_load_pending_from_file(self, temp_storage):
        """Should load pending wisdom from storage file."""
        # Pre-create storage file
        pending_data = {
            "pending": [
                {
                    "id": "pre-existing",
                    "text": "Pre-saved wisdom",
                    "submitter_id": "user-old",
                    "timestamp": 1000.0,
                    "loop_id": "load-test",
                    "context_tags": ["test"],
                    "relevance_score": 0.0,
                    "used": False,
                    "used_at": None,
                    "upvotes": 2,
                }
            ],
            "updated_at": "2024-01-01T00:00:00",
        }
        with open(temp_storage / "load-test_pending.json", "w") as f:
            json.dump(pending_data, f)

        inj = WisdomInjector(loop_id="load-test", storage_path=temp_storage)

        assert len(inj.pending_wisdom) == 1
        assert inj.pending_wisdom[0].id == "pre-existing"
        assert inj.pending_wisdom[0].text == "Pre-saved wisdom"
        assert inj.pending_wisdom[0].upvotes == 2

    def test_load_handles_missing_file(self, temp_storage):
        """Should handle missing storage file gracefully."""
        inj = WisdomInjector(loop_id="missing", storage_path=temp_storage)

        assert inj.pending_wisdom == []

    def test_load_handles_corrupt_file(self, temp_storage):
        """Should handle corrupt JSON file gracefully."""
        with open(temp_storage / "corrupt_pending.json", "w") as f:
            f.write("not valid json {{{")

        inj = WisdomInjector(loop_id="corrupt", storage_path=temp_storage)

        assert inj.pending_wisdom == []

    @pytest.mark.asyncio
    async def test_save_pending_creates_file(self, temp_storage):
        """Should save pending wisdom to file."""
        inj = WisdomInjector(loop_id="save-test", storage_path=temp_storage)

        await inj.submit_wisdom("Test wisdom", "user-001")

        # Check file was created
        save_file = temp_storage / "save-test_pending.json"
        assert save_file.exists()

        with open(save_file) as f:
            data = json.load(f)

        assert len(data["pending"]) == 1
        assert data["pending"][0]["text"] == "Test wisdom"


class TestSubmitWisdom:
    """Tests for submit_wisdom() async method."""

    @pytest.mark.asyncio
    async def test_submit_wisdom_success(self, injector):
        """Should successfully submit wisdom."""
        wisdom = await injector.submit_wisdom(
            text="Good insight here",
            submitter_id="user-001",
            context_tags=["ai"],
        )

        assert wisdom is not None
        assert wisdom.text == "Good insight here"
        assert wisdom.submitter_id == "user-001"
        assert wisdom.context_tags == ["ai"]
        assert wisdom.loop_id == "test-loop-001"

    @pytest.mark.asyncio
    async def test_submit_wisdom_adds_to_pending(self, injector):
        """Should add to pending list."""
        assert len(injector.pending_wisdom) == 0

        await injector.submit_wisdom("Wisdom 1", "user")
        assert len(injector.pending_wisdom) == 1

        await injector.submit_wisdom("Wisdom 2", "user")
        assert len(injector.pending_wisdom) == 2

    @pytest.mark.asyncio
    async def test_submit_wisdom_generates_unique_id(self, injector):
        """Should generate unique IDs."""
        w1 = await injector.submit_wisdom("First", "user1")
        w2 = await injector.submit_wisdom("Second", "user2")

        assert w1.id != w2.id

    @pytest.mark.asyncio
    async def test_submit_wisdom_strips_whitespace(self, injector):
        """Should strip whitespace from text."""
        wisdom = await injector.submit_wisdom("  trimmed text  ", "user")

        assert wisdom.text == "trimmed text"

    @pytest.mark.asyncio
    async def test_submit_wisdom_rejects_empty(self, injector):
        """Should reject empty text."""
        result = await injector.submit_wisdom("", "user")
        assert result is None

        result = await injector.submit_wisdom("   ", "user")
        assert result is None

    @pytest.mark.asyncio
    async def test_submit_wisdom_truncates_long_text(self, injector):
        """Should truncate text over MAX_WISDOM_LENGTH."""
        long_text = "x" * 500

        wisdom = await injector.submit_wisdom(long_text, "user")

        assert len(wisdom.text) == WisdomInjector.MAX_WISDOM_LENGTH

    @pytest.mark.asyncio
    async def test_submit_wisdom_rejects_duplicates(self, injector):
        """Should reject duplicate submissions."""
        await injector.submit_wisdom("Same text", "user1")
        duplicate = await injector.submit_wisdom("Same text", "user2")

        assert duplicate is None
        assert len(injector.pending_wisdom) == 1

    @pytest.mark.asyncio
    async def test_submit_wisdom_case_insensitive_duplicate(self, injector):
        """Should detect case-insensitive duplicates."""
        await injector.submit_wisdom("Same Text", "user1")
        duplicate = await injector.submit_wisdom("same text", "user2")

        assert duplicate is None

    @pytest.mark.asyncio
    async def test_submit_wisdom_updates_submitter_stats(self, injector):
        """Should update submitter statistics."""
        await injector.submit_wisdom("First", "user-001")
        await injector.submit_wisdom("Second", "user-001")

        assert injector.submitter_stats["user-001"]["submissions"] == 2

    @pytest.mark.asyncio
    async def test_submit_wisdom_respects_max_pending(self, injector):
        """Should enforce MAX_PENDING_WISDOM limit."""
        # Submit more than max
        for i in range(WisdomInjector.MAX_PENDING_WISDOM + 10):
            await injector.submit_wisdom(f"Wisdom {i}", f"user-{i}")

        assert len(injector.pending_wisdom) <= WisdomInjector.MAX_PENDING_WISDOM

    @pytest.mark.asyncio
    async def test_submit_wisdom_empty_tags(self, injector):
        """Should handle None context_tags."""
        wisdom = await injector.submit_wisdom("No tags", "user", context_tags=None)

        assert wisdom.context_tags == []


class TestUpvoteWisdom:
    """Tests for upvote_wisdom() method."""

    @pytest.mark.asyncio
    async def test_upvote_success(self, injector):
        """Should successfully upvote wisdom."""
        wisdom = await injector.submit_wisdom("Upvotable", "user")
        original_upvotes = wisdom.upvotes

        result = injector.upvote_wisdom(wisdom.id, "voter-001")

        assert result is True
        assert wisdom.upvotes == original_upvotes + 1

    @pytest.mark.asyncio
    async def test_upvote_nonexistent(self, injector):
        """Should return False for nonexistent wisdom."""
        result = injector.upvote_wisdom("nonexistent-id", "voter")

        assert result is False

    @pytest.mark.asyncio
    async def test_upvote_updates_submitter_stats(self, injector):
        """Should update submitter's upvote count."""
        wisdom = await injector.submit_wisdom("Upvotable", "author-001")

        injector.upvote_wisdom(wisdom.id, "voter-001")
        injector.upvote_wisdom(wisdom.id, "voter-002")

        assert injector.submitter_stats["author-001"]["upvotes"] == 2

    @pytest.mark.asyncio
    async def test_upvote_multiple_times(self, injector):
        """Should allow multiple upvotes (no double-vote prevention yet)."""
        wisdom = await injector.submit_wisdom("Popular", "user")

        injector.upvote_wisdom(wisdom.id, "voter")
        injector.upvote_wisdom(wisdom.id, "voter")

        assert wisdom.upvotes == 2


class TestCalculateRelevance:
    """Tests for _calculate_relevance() internal method."""

    @pytest.mark.asyncio
    async def test_relevance_recency_factor(self, injector):
        """Newer wisdom should have higher relevance."""
        # Recent wisdom
        wisdom_new = WisdomSubmission(
            id="new",
            text="Recent insight",
            submitter_id="user",
            timestamp=time.time(),  # Now
            loop_id="test",
        )
        # Old wisdom (25 hours ago)
        wisdom_old = WisdomSubmission(
            id="old",
            text="Old insight",
            submitter_id="user",
            timestamp=time.time() - 25 * 3600,
            loop_id="test",
        )

        context = {}
        score_new = injector._calculate_relevance(wisdom_new, context)
        score_old = injector._calculate_relevance(wisdom_old, context)

        assert score_new > score_old

    @pytest.mark.asyncio
    async def test_relevance_upvote_factor(self, injector):
        """Upvoted wisdom should have higher relevance."""
        wisdom_popular = WisdomSubmission(
            id="popular",
            text="Popular insight",
            submitter_id="user",
            timestamp=time.time(),
            loop_id="test",
            upvotes=10,
        )
        wisdom_unpopular = WisdomSubmission(
            id="unpopular",
            text="Unpopular insight",
            submitter_id="user",
            timestamp=time.time(),
            loop_id="test",
            upvotes=0,
        )

        context = {}
        score_popular = injector._calculate_relevance(wisdom_popular, context)
        score_unpopular = injector._calculate_relevance(wisdom_unpopular, context)

        assert score_popular > score_unpopular

    @pytest.mark.asyncio
    async def test_relevance_tag_matching(self, injector):
        """Tag matching should increase relevance."""
        wisdom_matched = WisdomSubmission(
            id="matched",
            text="Insight about performance",
            submitter_id="user",
            timestamp=time.time(),
            loop_id="test",
            context_tags=["performance", "optimization"],
        )
        wisdom_unmatched = WisdomSubmission(
            id="unmatched",
            text="Unrelated insight",
            submitter_id="user",
            timestamp=time.time(),
            loop_id="test",
            context_tags=["unrelated"],
        )

        context = {"topic": "performance testing", "tags": ["optimization"]}
        score_matched = injector._calculate_relevance(wisdom_matched, context)
        score_unmatched = injector._calculate_relevance(wisdom_unmatched, context)

        assert score_matched > score_unmatched

    @pytest.mark.asyncio
    async def test_relevance_reputation_factor(self, injector):
        """Submitters with used wisdom should have higher relevance."""
        # Set up reputation
        injector.submitter_stats["trusted-user"]["used"] = 5

        wisdom_trusted = WisdomSubmission(
            id="trusted",
            text="Insight from trusted user",
            submitter_id="trusted-user",
            timestamp=time.time(),
            loop_id="test",
        )
        wisdom_new_user = WisdomSubmission(
            id="new",
            text="Insight from new user",
            submitter_id="new-user",
            timestamp=time.time(),
            loop_id="test",
        )

        context = {}
        score_trusted = injector._calculate_relevance(wisdom_trusted, context)
        score_new = injector._calculate_relevance(wisdom_new_user, context)

        assert score_trusted > score_new

    @pytest.mark.asyncio
    async def test_relevance_max_capped_at_one(self, injector):
        """Relevance score should be capped at 1.0."""
        # Create wisdom with all positive factors
        injector.submitter_stats["super-user"]["used"] = 10

        wisdom = WisdomSubmission(
            id="perfect",
            text="Perfect matching keyword optimization performance",
            submitter_id="super-user",
            timestamp=time.time(),
            loop_id="test",
            context_tags=["optimization", "performance", "architecture"],
            upvotes=100,
        )

        context = {"topic": "optimization performance architecture"}
        score = injector._calculate_relevance(wisdom, context)

        assert score <= 1.0


class TestFindRelevantWisdom:
    """Tests for find_relevant_wisdom() method."""

    @pytest.mark.asyncio
    async def test_find_empty_pending(self, injector):
        """Should return empty list when no pending wisdom."""
        result = injector.find_relevant_wisdom({})

        assert result == []

    @pytest.mark.asyncio
    async def test_find_filters_by_threshold(self, injector):
        """Should filter wisdom below relevance threshold."""
        # Add wisdom with very low relevance (old, no tags, no upvotes)
        old_wisdom = WisdomSubmission(
            id="old",
            text="Irrelevant old insight",
            submitter_id="user",
            timestamp=time.time() - 48 * 3600,  # 2 days old
            loop_id="test-loop-001",
        )
        injector.pending_wisdom.append(old_wisdom)

        result = injector.find_relevant_wisdom({})

        # Should be filtered out due to low relevance
        assert old_wisdom not in result or old_wisdom.relevance_score >= WisdomInjector.RELEVANCE_THRESHOLD

    @pytest.mark.asyncio
    async def test_find_sorts_by_relevance(self, injector):
        """Should sort results by relevance descending."""
        # Add wisdom with different relevance
        for i, upvotes in enumerate([1, 10, 5]):
            wisdom = WisdomSubmission(
                id=f"wis-{i}",
                text=f"Insight {i}",
                submitter_id="user",
                timestamp=time.time(),
                loop_id="test-loop-001",
                upvotes=upvotes,
            )
            injector.pending_wisdom.append(wisdom)

        result = injector.find_relevant_wisdom({})

        # Should be sorted by relevance
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i].relevance_score >= result[i + 1].relevance_score

    @pytest.mark.asyncio
    async def test_find_respects_limit(self, injector):
        """Should respect the limit parameter."""
        # Add many wisdoms
        for i in range(10):
            wisdom = WisdomSubmission(
                id=f"wis-{i}",
                text=f"Insight number {i}",
                submitter_id="user",
                timestamp=time.time(),
                loop_id="test-loop-001",
                upvotes=i,  # Different upvotes for different relevance
            )
            injector.pending_wisdom.append(wisdom)

        result = injector.find_relevant_wisdom({}, limit=2)

        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_find_excludes_used(self, injector):
        """Should exclude already-used wisdom."""
        used_wisdom = WisdomSubmission(
            id="used",
            text="Already used",
            submitter_id="user",
            timestamp=time.time(),
            loop_id="test-loop-001",
            used=True,
            upvotes=10,
        )
        injector.pending_wisdom.append(used_wisdom)

        result = injector.find_relevant_wisdom({})

        assert used_wisdom not in result

    @pytest.mark.asyncio
    async def test_find_calculates_relevance(self, injector):
        """Should calculate relevance score for pending wisdom."""
        wisdom = WisdomSubmission(
            id="test",
            text="Test insight",
            submitter_id="user",
            timestamp=time.time(),
            loop_id="test-loop-001",
            relevance_score=0.0,  # Initially 0
        )
        injector.pending_wisdom.append(wisdom)

        injector.find_relevant_wisdom({"topic": "test insight"})

        # Relevance should have been calculated
        assert wisdom.relevance_score > 0


class TestInjectWisdom:
    """Tests for inject_wisdom() method."""

    def test_inject_marks_as_used(self, injector, sample_wisdom):
        """Should mark wisdom as used."""
        injector.pending_wisdom.append(sample_wisdom)

        injector.inject_wisdom(sample_wisdom, "claude context", "timeout")

        assert sample_wisdom.used is True
        assert sample_wisdom.used_at is not None

    def test_inject_updates_stats(self, injector, sample_wisdom):
        """Should update submitter stats."""
        injector.pending_wisdom.append(sample_wisdom)

        injector.inject_wisdom(sample_wisdom, "context", "timeout")

        assert injector.submitter_stats[sample_wisdom.submitter_id]["used"] == 1

    def test_inject_moves_to_used_list(self, injector, sample_wisdom):
        """Should move wisdom from pending to used list."""
        injector.pending_wisdom.append(sample_wisdom)

        injector.inject_wisdom(sample_wisdom, "context", "timeout")

        assert sample_wisdom not in injector.pending_wisdom
        assert sample_wisdom in injector.used_wisdom

    def test_inject_creates_injection_record(self, injector, sample_wisdom):
        """Should create injection record."""
        injector.pending_wisdom.append(sample_wisdom)

        injection = injector.inject_wisdom(sample_wisdom, "test context", "stall")

        assert isinstance(injection, WisdomInjection)
        assert injection.wisdom_id == sample_wisdom.id
        assert injection.agent_context == "test context"
        assert injection.injection_reason == "stall"
        assert injection in injector.injections

    def test_inject_truncates_long_context(self, injector, sample_wisdom):
        """Should truncate long agent context."""
        injector.pending_wisdom.append(sample_wisdom)
        long_context = "x" * 500

        injection = injector.inject_wisdom(sample_wisdom, long_context, "timeout")

        assert len(injection.agent_context) == 200

    def test_inject_reasons(self, injector):
        """Should accept different injection reasons."""
        reasons = ["timeout", "stall", "requested"]

        for reason in reasons:
            wisdom = WisdomSubmission(
                id=f"wis-{reason}",
                text=f"Wisdom for {reason}",
                submitter_id="user",
                timestamp=time.time(),
                loop_id="test-loop-001",
            )
            injector.pending_wisdom.append(wisdom)

            injection = injector.inject_wisdom(wisdom, "context", reason)
            assert injection.injection_reason == reason


class TestFormatForPrompt:
    """Tests for format_for_prompt() method."""

    def test_format_empty_list(self, injector):
        """Should return empty string for empty list."""
        result = injector.format_for_prompt([])

        assert result == ""

    def test_format_single_wisdom(self, injector, sample_wisdom):
        """Should format single wisdom."""
        result = injector.format_for_prompt([sample_wisdom])

        assert "[Audience Insights]" in result
        assert sample_wisdom.text in result
        assert "audience member" in result

    def test_format_multiple_wisdoms(self, injector):
        """Should format multiple wisdoms with numbering."""
        wisdoms = [
            WisdomSubmission(
                id=f"wis-{i}",
                text=f"Insight number {i}",
                submitter_id="user",
                timestamp=time.time(),
                loop_id="test",
            )
            for i in range(3)
        ]

        result = injector.format_for_prompt(wisdoms)

        assert "1." in result
        assert "2." in result
        assert "3." in result
        assert "Insight number 0" in result
        assert "Insight number 1" in result
        assert "Insight number 2" in result

    def test_format_structure(self, injector, sample_wisdom):
        """Should have proper structure."""
        result = injector.format_for_prompt([sample_wisdom])
        lines = result.split("\n")

        assert lines[0] == "[Audience Insights]"
        assert lines[1].startswith("1.")
        assert '"' in lines[1]  # Quoted text


class TestGetStats:
    """Tests for get_stats() method."""

    def test_stats_initial(self, injector):
        """Should return correct initial stats."""
        stats = injector.get_stats()

        assert stats["loop_id"] == "test-loop-001"
        assert stats["pending_count"] == 0
        assert stats["used_count"] == 0
        assert stats["injection_count"] == 0
        assert stats["unique_submitters"] == 0
        assert stats["total_upvotes"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_submissions(self, injector):
        """Should reflect submissions in stats."""
        await injector.submit_wisdom("One", "user-1")
        await injector.submit_wisdom("Two", "user-2")
        await injector.submit_wisdom("Three", "user-1")

        stats = injector.get_stats()

        assert stats["pending_count"] == 3
        assert stats["unique_submitters"] == 2

    @pytest.mark.asyncio
    async def test_stats_after_injection(self, injector):
        """Should reflect injections in stats."""
        wisdom = await injector.submit_wisdom("Inject me", "user")
        injector.inject_wisdom(wisdom, "context", "timeout")

        stats = injector.get_stats()

        assert stats["pending_count"] == 0
        assert stats["used_count"] == 1
        assert stats["injection_count"] == 1

    @pytest.mark.asyncio
    async def test_stats_upvotes(self, injector):
        """Should count total upvotes."""
        w1 = await injector.submit_wisdom("First", "user-1")
        w2 = await injector.submit_wisdom("Second", "user-2")

        injector.upvote_wisdom(w1.id, "voter")
        injector.upvote_wisdom(w1.id, "voter2")
        injector.upvote_wisdom(w2.id, "voter")

        stats = injector.get_stats()

        assert stats["total_upvotes"] == 3


class TestGlobalInstanceManagement:
    """Tests for global instance management functions."""

    def test_get_creates_new_instance(self, tmp_path, monkeypatch):
        """Should create new instance for new loop_id."""
        monkeypatch.chdir(tmp_path)

        injector = get_wisdom_injector("new-loop-001")

        assert isinstance(injector, WisdomInjector)
        assert injector.loop_id == "new-loop-001"

    def test_get_returns_existing_instance(self, tmp_path, monkeypatch):
        """Should return same instance for same loop_id."""
        monkeypatch.chdir(tmp_path)

        inj1 = get_wisdom_injector("same-loop")
        inj2 = get_wisdom_injector("same-loop")

        assert inj1 is inj2

    def test_get_different_loops_different_instances(self, tmp_path, monkeypatch):
        """Should create different instances for different loop_ids."""
        monkeypatch.chdir(tmp_path)

        inj1 = get_wisdom_injector("loop-a")
        inj2 = get_wisdom_injector("loop-b")

        assert inj1 is not inj2

    def test_close_removes_instance(self, tmp_path, monkeypatch):
        """Should remove instance from global registry."""
        monkeypatch.chdir(tmp_path)

        get_wisdom_injector("to-close")
        assert "to-close" in _injectors

        close_wisdom_injector("to-close")

        assert "to-close" not in _injectors

    def test_close_nonexistent_safe(self):
        """Should handle closing nonexistent loop safely."""
        close_wisdom_injector("nonexistent-loop")
        # Should not raise


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_unicode_in_wisdom(self, injector):
        """Should handle unicode characters."""
        wisdom = await injector.submit_wisdom(
            "Consider the emoji factor 🤔 and special chars: é, ñ, 中文",
            "user"
        )

        assert wisdom is not None
        assert "🤔" in wisdom.text

    @pytest.mark.asyncio
    async def test_special_characters_in_submitter_id(self, injector):
        """Should handle special characters in submitter ID."""
        wisdom = await injector.submit_wisdom(
            "Test",
            "user@domain.com/path#hash"
        )

        assert wisdom is not None
        assert wisdom.submitter_id == "user@domain.com/path#hash"

    @pytest.mark.asyncio
    async def test_empty_context_tags_list(self, injector):
        """Should handle empty context tags."""
        wisdom = await injector.submit_wisdom("No tags", "user", context_tags=[])

        assert wisdom.context_tags == []

    @pytest.mark.asyncio
    async def test_very_long_submitter_id(self, injector):
        """Should handle very long submitter IDs."""
        long_id = "user-" + "x" * 500
        wisdom = await injector.submit_wisdom("Test", long_id)

        assert wisdom is not None
        assert wisdom.submitter_id == long_id

    def test_relevance_with_empty_context(self, injector, sample_wisdom):
        """Should calculate relevance with empty context."""
        score = injector._calculate_relevance(sample_wisdom, {})

        assert 0.0 <= score <= 1.0

    def test_relevance_with_empty_string_context(self, injector, sample_wisdom):
        """Should handle empty string values in context."""
        context = {"topic": "", "tags": []}

        # Should not raise
        score = injector._calculate_relevance(sample_wisdom, context)
        assert 0.0 <= score <= 1.0


class TestConcurrency:
    """Tests for concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_submissions(self, injector):
        """Should handle concurrent submissions."""
        async def submit(i):
            return await injector.submit_wisdom(f"Concurrent wisdom {i}", f"user-{i}")

        # Submit concurrently
        tasks = [submit(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # Count successful submissions (some may be duplicates)
        successful = [r for r in results if r is not None]
        assert len(successful) == 20
        assert len(injector.pending_wisdom) == 20
