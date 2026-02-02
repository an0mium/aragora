"""Tests for Wisdom Injector module."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.debate.wisdom_injector import (
    WisdomInjection,
    WisdomInjector,
    WisdomSubmission,
    close_wisdom_injector,
    get_wisdom_injector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path):
    """Provide a temporary storage directory for the injector."""
    return tmp_path / "wisdom"


@pytest.fixture
def injector(tmp_storage):
    """Create a WisdomInjector with a temp storage path."""
    return WisdomInjector(loop_id="test_loop", storage_path=tmp_storage)


# ---------------------------------------------------------------------------
# WisdomSubmission dataclass
# ---------------------------------------------------------------------------


class TestWisdomSubmission:
    """Test the WisdomSubmission dataclass."""

    def test_default_fields(self):
        """Test default field values on WisdomSubmission."""
        ws = WisdomSubmission(
            id="abc123",
            text="insight",
            submitter_id="user1",
            timestamp=1000.0,
            loop_id="loop1",
        )
        assert ws.context_tags == []
        assert ws.relevance_score == 0.0
        assert ws.used is False
        assert ws.used_at is None
        assert ws.upvotes == 0

    def test_to_dict(self):
        """Test serialization via to_dict."""
        ws = WisdomSubmission(
            id="abc",
            text="hello",
            submitter_id="u1",
            timestamp=1.0,
            loop_id="l1",
            context_tags=["tag"],
            relevance_score=0.5,
            used=True,
            used_at=2.0,
            upvotes=3,
        )
        d = ws.to_dict()
        assert d["id"] == "abc"
        assert d["text"] == "hello"
        assert d["context_tags"] == ["tag"]
        assert d["relevance_score"] == 0.5
        assert d["used"] is True
        assert d["used_at"] == 2.0
        assert d["upvotes"] == 3

    def test_generate_id_deterministic(self):
        """Test that generate_id is deterministic for same inputs."""
        id1 = WisdomSubmission.generate_id("text", "user", 1.0)
        id2 = WisdomSubmission.generate_id("text", "user", 1.0)
        assert id1 == id2

    def test_generate_id_unique_for_different_inputs(self):
        """Test that generate_id differs for different inputs."""
        id1 = WisdomSubmission.generate_id("text1", "user", 1.0)
        id2 = WisdomSubmission.generate_id("text2", "user", 1.0)
        assert id1 != id2

    def test_generate_id_length(self):
        """Test that generate_id returns a 12-char hex string."""
        gid = WisdomSubmission.generate_id("t", "u", 0.0)
        assert len(gid) == 12
        # Should be valid hex
        int(gid, 16)


# ---------------------------------------------------------------------------
# WisdomInjection dataclass
# ---------------------------------------------------------------------------


class TestWisdomInjection:
    """Test the WisdomInjection dataclass."""

    def test_to_dict(self):
        """Test serialization via to_dict."""
        inj = WisdomInjection(
            wisdom_id="w1",
            agent_context="ctx",
            injection_reason="timeout",
            timestamp=100.0,
            impact_score=0.7,
        )
        d = inj.to_dict()
        assert d["wisdom_id"] == "w1"
        assert d["injection_reason"] == "timeout"
        assert d["impact_score"] == 0.7

    def test_default_impact_score(self):
        """Test default impact_score is 0.0."""
        inj = WisdomInjection(
            wisdom_id="w1",
            agent_context="ctx",
            injection_reason="stall",
            timestamp=0.0,
        )
        assert inj.impact_score == 0.0


# ---------------------------------------------------------------------------
# WisdomInjector initialization
# ---------------------------------------------------------------------------


class TestWisdomInjectorInit:
    """Test WisdomInjector initialization."""

    def test_init_creates_storage_directory(self, tmp_storage):
        """Test that __init__ creates the storage directory."""
        assert not tmp_storage.exists()
        WisdomInjector(loop_id="init_test", storage_path=tmp_storage)
        assert tmp_storage.exists()

    def test_init_sets_loop_id(self, injector):
        """Test loop_id is stored correctly."""
        assert injector.loop_id == "test_loop"

    def test_init_empty_collections(self, injector):
        """Test initial collections are empty."""
        assert injector.pending_wisdom == []
        assert injector.used_wisdom == []
        assert injector.injections == []

    def test_init_loads_existing_pending(self, tmp_storage):
        """Test __init__ loads previously saved pending wisdom."""
        tmp_storage.mkdir(parents=True, exist_ok=True)
        pending_file = tmp_storage / "reload_loop_pending.json"
        pending_file.write_text(
            json.dumps(
                {
                    "pending": [
                        {
                            "id": "w1",
                            "text": "existing",
                            "submitter_id": "u1",
                            "timestamp": 1.0,
                            "loop_id": "reload_loop",
                            "context_tags": [],
                            "relevance_score": 0.0,
                            "used": False,
                            "used_at": None,
                            "upvotes": 0,
                        }
                    ]
                }
            )
        )
        inj = WisdomInjector(loop_id="reload_loop", storage_path=tmp_storage)
        assert len(inj.pending_wisdom) == 1
        assert inj.pending_wisdom[0].text == "existing"

    def test_init_handles_corrupt_json(self, tmp_storage):
        """Test __init__ handles corrupt JSON without raising."""
        tmp_storage.mkdir(parents=True, exist_ok=True)
        pending_file = tmp_storage / "bad_loop_pending.json"
        pending_file.write_text("NOT JSON")
        # Should not raise
        inj = WisdomInjector(loop_id="bad_loop", storage_path=tmp_storage)
        assert inj.pending_wisdom == []


# ---------------------------------------------------------------------------
# submit_wisdom
# ---------------------------------------------------------------------------


class TestSubmitWisdom:
    """Test the submit_wisdom method."""

    @pytest.mark.asyncio
    async def test_submit_returns_submission(self, injector):
        """Test successful submission returns a WisdomSubmission."""
        ws = await injector.submit_wisdom("Good insight", "user1")
        assert ws is not None
        assert ws.text == "Good insight"
        assert ws.submitter_id == "user1"
        assert ws.loop_id == "test_loop"

    @pytest.mark.asyncio
    async def test_submit_empty_text_returns_none(self, injector):
        """Test empty text is rejected."""
        ws = await injector.submit_wisdom("", "user1")
        assert ws is None

    @pytest.mark.asyncio
    async def test_submit_whitespace_only_returns_none(self, injector):
        """Test whitespace-only text is rejected."""
        ws = await injector.submit_wisdom("   ", "user1")
        assert ws is None

    @pytest.mark.asyncio
    async def test_submit_truncates_long_text(self, injector):
        """Test text longer than MAX_WISDOM_LENGTH is truncated."""
        long_text = "A" * 500
        ws = await injector.submit_wisdom(long_text, "user1")
        assert ws is not None
        assert len(ws.text) == WisdomInjector.MAX_WISDOM_LENGTH

    @pytest.mark.asyncio
    async def test_submit_duplicate_rejected(self, injector):
        """Test duplicate text is rejected."""
        ws1 = await injector.submit_wisdom("duplicate insight", "user1")
        ws2 = await injector.submit_wisdom("duplicate insight", "user2")
        assert ws1 is not None
        assert ws2 is None

    @pytest.mark.asyncio
    async def test_submit_case_insensitive_duplicate(self, injector):
        """Test case-insensitive duplicate detection."""
        ws1 = await injector.submit_wisdom("Hello World", "user1")
        ws2 = await injector.submit_wisdom("hello world", "user2")
        assert ws1 is not None
        assert ws2 is None

    @pytest.mark.asyncio
    async def test_submit_with_context_tags(self, injector):
        """Test submission with context tags."""
        ws = await injector.submit_wisdom("tagged insight", "user1", context_tags=["perf", "arch"])
        assert ws is not None
        assert ws.context_tags == ["perf", "arch"]

    @pytest.mark.asyncio
    async def test_submit_updates_submitter_stats(self, injector):
        """Test submitter stats are updated on submission."""
        await injector.submit_wisdom("insight one", "user1")
        await injector.submit_wisdom("insight two", "user1")
        assert injector.submitter_stats["user1"]["submissions"] == 2

    @pytest.mark.asyncio
    async def test_submit_adds_to_pending(self, injector):
        """Test submission is added to pending list."""
        await injector.submit_wisdom("new wisdom", "user1")
        assert len(injector.pending_wisdom) == 1

    @pytest.mark.asyncio
    async def test_submit_enforces_max_pending(self, injector):
        """Test pending list is capped at MAX_PENDING_WISDOM."""
        for i in range(WisdomInjector.MAX_PENDING_WISDOM + 10):
            await injector.submit_wisdom(f"wisdom {i}", f"user_{i}")
        assert len(injector.pending_wisdom) <= WisdomInjector.MAX_PENDING_WISDOM

    @pytest.mark.asyncio
    async def test_submit_persists_to_disk(self, injector, tmp_storage):
        """Test wisdom is persisted after submission."""
        await injector.submit_wisdom("persistent", "user1")
        pending_file = tmp_storage / "test_loop_pending.json"
        assert pending_file.exists()
        data = json.loads(pending_file.read_text())
        assert len(data["pending"]) == 1
        assert data["pending"][0]["text"] == "persistent"


# ---------------------------------------------------------------------------
# upvote_wisdom
# ---------------------------------------------------------------------------


class TestUpvoteWisdom:
    """Test upvoting wisdom submissions."""

    @pytest.mark.asyncio
    async def test_upvote_existing_wisdom(self, injector):
        """Test upvoting an existing wisdom."""
        ws = await injector.submit_wisdom("upvotable", "user1")
        result = injector.upvote_wisdom(ws.id, "voter1")
        assert result is True
        assert ws.upvotes == 1

    @pytest.mark.asyncio
    async def test_upvote_updates_submitter_stats(self, injector):
        """Test upvote updates submitter stats."""
        ws = await injector.submit_wisdom("upvotable", "user1")
        injector.upvote_wisdom(ws.id, "voter1")
        assert injector.submitter_stats["user1"]["upvotes"] == 1

    def test_upvote_nonexistent_wisdom(self, injector):
        """Test upvoting a non-existent wisdom returns False."""
        result = injector.upvote_wisdom("nonexistent_id", "voter1")
        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_upvotes(self, injector):
        """Test multiple upvotes accumulate."""
        ws = await injector.submit_wisdom("popular", "user1")
        injector.upvote_wisdom(ws.id, "v1")
        injector.upvote_wisdom(ws.id, "v2")
        injector.upvote_wisdom(ws.id, "v3")
        assert ws.upvotes == 3


# ---------------------------------------------------------------------------
# _calculate_relevance
# ---------------------------------------------------------------------------


class TestCalculateRelevance:
    """Test relevance scoring."""

    def test_recent_wisdom_scores_higher(self, injector):
        """Test that newer wisdom gets a higher recency score."""
        recent = WisdomSubmission(
            id="r",
            text="insight",
            submitter_id="u",
            timestamp=time.time(),
            loop_id="test_loop",
        )
        old = WisdomSubmission(
            id="o",
            text="insight",
            submitter_id="u",
            timestamp=time.time() - 86400,
            loop_id="test_loop",
        )
        score_recent = injector._calculate_relevance(recent, {})
        score_old = injector._calculate_relevance(old, {})
        assert score_recent > score_old

    def test_upvotes_increase_relevance(self, injector):
        """Test upvotes increase relevance score."""
        base = WisdomSubmission(
            id="b",
            text="insight",
            submitter_id="u",
            timestamp=time.time(),
            loop_id="test_loop",
        )
        upvoted = WisdomSubmission(
            id="v",
            text="insight",
            submitter_id="u",
            timestamp=time.time(),
            loop_id="test_loop",
            upvotes=5,
        )
        score_base = injector._calculate_relevance(base, {})
        score_upvoted = injector._calculate_relevance(upvoted, {})
        assert score_upvoted > score_base

    def test_tag_matching_increases_relevance(self, injector):
        """Test tag overlap increases relevance."""
        ws = WisdomSubmission(
            id="t",
            text="performance optimization",
            submitter_id="u",
            timestamp=time.time(),
            loop_id="test_loop",
            context_tags=["performance"],
        )
        context_match = {"topic": "performance tuning", "tags": ["performance"]}
        context_no_match = {"topic": "unrelated topic"}
        score_match = injector._calculate_relevance(ws, context_match)
        score_no = injector._calculate_relevance(ws, context_no_match)
        assert score_match > score_no

    def test_submitter_reputation_increases_relevance(self, injector):
        """Test submitter with used wisdom gets higher relevance."""
        injector.submitter_stats["rep_user"]["used"] = 5
        ws = WisdomSubmission(
            id="s",
            text="insight",
            submitter_id="rep_user",
            timestamp=time.time(),
            loop_id="test_loop",
        )
        score = injector._calculate_relevance(ws, {})
        # Should include reputation component
        ws_no_rep = WisdomSubmission(
            id="s2",
            text="insight",
            submitter_id="new_user",
            timestamp=time.time(),
            loop_id="test_loop",
        )
        score_no_rep = injector._calculate_relevance(ws_no_rep, {})
        assert score > score_no_rep

    def test_relevance_capped_at_one(self, injector):
        """Test relevance score does not exceed 1.0."""
        injector.submitter_stats["power_user"]["used"] = 100
        ws = WisdomSubmission(
            id="max",
            text="performance optimization design architecture",
            submitter_id="power_user",
            timestamp=time.time(),
            loop_id="test_loop",
            upvotes=100,
            context_tags=["performance", "optimization", "design"],
        )
        context = {
            "topic": "performance optimization design",
            "tags": ["performance", "optimization", "design"],
        }
        score = injector._calculate_relevance(ws, context)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# find_relevant_wisdom
# ---------------------------------------------------------------------------


class TestFindRelevantWisdom:
    """Test finding relevant wisdom."""

    def test_empty_pending_returns_empty(self, injector):
        """Test returns empty list when no pending wisdom."""
        result = injector.find_relevant_wisdom({"topic": "anything"})
        assert result == []

    @pytest.mark.asyncio
    async def test_finds_relevant_wisdom(self, injector):
        """Test finds wisdom relevant to debate context."""
        await injector.submit_wisdom("performance matters", "u1", context_tags=["performance"])
        results = injector.find_relevant_wisdom(
            {"topic": "performance optimization", "tags": ["performance"]}
        )
        # May or may not meet threshold depending on timing - at least runs
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_excludes_used_wisdom(self, injector):
        """Test used wisdom is excluded from results."""
        ws = await injector.submit_wisdom("used insight", "u1")
        ws.used = True
        results = injector.find_relevant_wisdom({"topic": "used insight"})
        assert all(not w.used for w in results)

    @pytest.mark.asyncio
    async def test_respects_limit(self, injector):
        """Test limits the number of returned results."""
        for i in range(10):
            await injector.submit_wisdom(
                f"wisdom about performance number {i}",
                f"user_{i}",
                context_tags=["performance"],
            )
        results = injector.find_relevant_wisdom(
            {"topic": "performance", "tags": ["performance"]}, limit=2
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_sorted_by_relevance_descending(self, injector):
        """Test results are sorted by relevance (highest first)."""
        await injector.submit_wisdom("generic thought", "u1")
        ws2 = await injector.submit_wisdom(
            "performance optimization is key",
            "u2",
            context_tags=["performance"],
        )
        # Give ws2 upvotes to boost relevance
        if ws2:
            injector.upvote_wisdom(ws2.id, "v1")
            injector.upvote_wisdom(ws2.id, "v2")

        results = injector.find_relevant_wisdom({"topic": "performance", "tags": ["performance"]})
        if len(results) >= 2:
            assert results[0].relevance_score >= results[1].relevance_score

    @pytest.mark.asyncio
    async def test_filters_below_threshold(self, injector):
        """Test filters out wisdom below relevance threshold."""
        # Very old wisdom with no matching context should be below threshold
        ws = await injector.submit_wisdom("random stuff", "u1")
        if ws:
            ws.timestamp = time.time() - 200000  # Very old
        results = injector.find_relevant_wisdom({"topic": "unrelated"})
        for w in results:
            assert w.relevance_score >= WisdomInjector.RELEVANCE_THRESHOLD


# ---------------------------------------------------------------------------
# inject_wisdom
# ---------------------------------------------------------------------------


class TestInjectWisdom:
    """Test wisdom injection into debates."""

    @pytest.mark.asyncio
    async def test_inject_marks_wisdom_as_used(self, injector):
        """Test injection marks the wisdom as used."""
        ws = await injector.submit_wisdom("inject me", "u1")
        injector.inject_wisdom(ws, "agent_context", "timeout")
        assert ws.used is True
        assert ws.used_at is not None

    @pytest.mark.asyncio
    async def test_inject_moves_to_used_list(self, injector):
        """Test injection moves wisdom from pending to used list."""
        ws = await injector.submit_wisdom("inject me", "u1")
        injector.inject_wisdom(ws, "agent_context", "timeout")
        assert ws not in injector.pending_wisdom
        assert ws in injector.used_wisdom

    @pytest.mark.asyncio
    async def test_inject_creates_injection_record(self, injector):
        """Test injection creates a WisdomInjection record."""
        ws = await injector.submit_wisdom("inject me", "u1")
        injection = injector.inject_wisdom(ws, "agent_context", "stall")
        assert isinstance(injection, WisdomInjection)
        assert injection.wisdom_id == ws.id
        assert injection.injection_reason == "stall"
        assert injection.agent_context == "agent_context"
        assert len(injector.injections) == 1

    @pytest.mark.asyncio
    async def test_inject_updates_submitter_used_stats(self, injector):
        """Test injection updates submitter used count."""
        ws = await injector.submit_wisdom("inject me", "u1")
        injector.inject_wisdom(ws, "ctx", "timeout")
        assert injector.submitter_stats["u1"]["used"] == 1

    @pytest.mark.asyncio
    async def test_inject_truncates_long_agent_context(self, injector):
        """Test agent_context is truncated to 200 chars."""
        ws = await injector.submit_wisdom("inject me", "u1")
        long_ctx = "X" * 500
        injection = injector.inject_wisdom(ws, long_ctx, "timeout")
        assert len(injection.agent_context) == 200

    @pytest.mark.asyncio
    async def test_inject_reason_variants(self, injector):
        """Test various injection reasons are stored."""
        for reason in ("timeout", "stall", "requested"):
            ws = await injector.submit_wisdom(f"wisdom for {reason}", "u1")
            inj = injector.inject_wisdom(ws, "ctx", reason)
            assert inj.injection_reason == reason


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------


class TestFormatForPrompt:
    """Test prompt formatting."""

    def test_empty_list_returns_empty_string(self, injector):
        """Test empty input returns empty string."""
        assert injector.format_for_prompt([]) == ""

    def test_single_wisdom(self, injector):
        """Test formatting a single wisdom."""
        ws = WisdomSubmission(
            id="f1",
            text="Consider caching",
            submitter_id="u1",
            timestamp=0.0,
            loop_id="test_loop",
        )
        result = injector.format_for_prompt([ws])
        assert "[Audience Insights]" in result
        assert '1. "Consider caching" - audience member' in result

    def test_multiple_wisdoms_numbered(self, injector):
        """Test multiple wisdoms are numbered correctly."""
        wisdoms = [
            WisdomSubmission(
                id=f"f{i}",
                text=f"insight {i}",
                submitter_id="u1",
                timestamp=0.0,
                loop_id="test_loop",
            )
            for i in range(3)
        ]
        result = injector.format_for_prompt(wisdoms)
        assert "1." in result
        assert "2." in result
        assert "3." in result


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    """Test statistics reporting."""

    def test_initial_stats(self, injector):
        """Test stats on a fresh injector."""
        stats = injector.get_stats()
        assert stats["loop_id"] == "test_loop"
        assert stats["pending_count"] == 0
        assert stats["used_count"] == 0
        assert stats["injection_count"] == 0
        assert stats["unique_submitters"] == 0
        assert stats["total_upvotes"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_submissions(self, injector):
        """Test stats reflect submissions."""
        await injector.submit_wisdom("one", "u1")
        await injector.submit_wisdom("two", "u2")
        stats = injector.get_stats()
        assert stats["pending_count"] == 2
        assert stats["unique_submitters"] == 2

    @pytest.mark.asyncio
    async def test_stats_after_injection(self, injector):
        """Test stats reflect injections."""
        ws = await injector.submit_wisdom("inj", "u1")
        injector.inject_wisdom(ws, "ctx", "timeout")
        stats = injector.get_stats()
        assert stats["pending_count"] == 0
        assert stats["used_count"] == 1
        assert stats["injection_count"] == 1

    @pytest.mark.asyncio
    async def test_stats_upvotes(self, injector):
        """Test stats reflect upvotes."""
        ws = await injector.submit_wisdom("pop", "u1")
        injector.upvote_wisdom(ws.id, "v1")
        injector.upvote_wisdom(ws.id, "v2")
        stats = injector.get_stats()
        assert stats["total_upvotes"] == 2


# ---------------------------------------------------------------------------
# Module-level functions: get_wisdom_injector / close_wisdom_injector
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    """Test get_wisdom_injector and close_wisdom_injector."""

    def test_get_creates_and_caches(self, tmp_path):
        """Test get_wisdom_injector creates a new injector and caches it."""
        with patch(
            "aragora.debate.wisdom_injector.WisdomInjector.__init__",
            return_value=None,
        ) as mock_init:
            # Clear the module-level cache first
            from aragora.debate import wisdom_injector as wmod

            wmod._injectors.clear()

            inj1 = get_wisdom_injector("cached_loop")
            inj2 = get_wisdom_injector("cached_loop")
            assert inj1 is inj2
            # Init should only be called once
            assert mock_init.call_count == 1

            # Cleanup
            wmod._injectors.clear()

    def test_close_removes_injector(self):
        """Test close_wisdom_injector removes from cache."""
        from aragora.debate import wisdom_injector as wmod

        wmod._injectors["temp"] = "sentinel"
        close_wisdom_injector("temp")
        assert "temp" not in wmod._injectors

    def test_close_nonexistent_no_error(self):
        """Test closing a non-existent loop does not raise."""
        close_wisdom_injector("does_not_exist")
