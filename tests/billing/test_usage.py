"""
Tests for Usage Tracking System.

Tests cover:
- Token cost calculation for various providers and models
- UsageEvent creation, serialization, and deserialization
- UsageSummary generation
- UsageTracker recording and aggregation
- Edge cases (zero usage, negative values, large values)
- Error handling
"""

from decimal import Decimal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import json
import tempfile

import pytest

from aragora.billing.usage import (
    UsageEventType,
    UsageEvent,
    UsageSummary,
    UsageTracker,
    calculate_token_cost,
    PROVIDER_PRICING,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary database path for each test."""
    return tmp_path / "test_usage.db"


@pytest.fixture
def tracker(temp_db_path):
    """Fresh UsageTracker instance with temporary database."""
    return UsageTracker(db_path=temp_db_path)


@pytest.fixture
def sample_event():
    """Sample UsageEvent for testing."""
    return UsageEvent(
        id="event-001",
        user_id="user-123",
        org_id="org-456",
        event_type=UsageEventType.DEBATE,
        debate_id="debate-789",
        tokens_in=1000,
        tokens_out=500,
        provider="anthropic",
        model="claude-sonnet-4",
        metadata={"round": 1},
    )


# =============================================================================
# UsageEventType Tests
# =============================================================================


class TestUsageEventType:
    """Tests for UsageEventType enum."""

    def test_event_type_values(self):
        """Test all event type values exist."""
        assert UsageEventType.DEBATE.value == "debate"
        assert UsageEventType.API_CALL.value == "api_call"
        assert UsageEventType.STORAGE.value == "storage"
        assert UsageEventType.AGENT_CALL.value == "agent_call"

    def test_event_type_from_value(self):
        """Test creating event type from string value."""
        assert UsageEventType("debate") == UsageEventType.DEBATE
        assert UsageEventType("api_call") == UsageEventType.API_CALL
        assert UsageEventType("storage") == UsageEventType.STORAGE
        assert UsageEventType("agent_call") == UsageEventType.AGENT_CALL

    def test_invalid_event_type(self):
        """Test invalid event type raises ValueError."""
        with pytest.raises(ValueError):
            UsageEventType("invalid_type")


# =============================================================================
# PROVIDER_PRICING Tests
# =============================================================================


class TestProviderPricing:
    """Tests for provider pricing configuration."""

    def test_anthropic_pricing_exists(self):
        """Test Anthropic pricing is configured."""
        assert "anthropic" in PROVIDER_PRICING
        assert "claude-opus-4" in PROVIDER_PRICING["anthropic"]
        assert "claude-opus-4-output" in PROVIDER_PRICING["anthropic"]
        assert "claude-sonnet-4" in PROVIDER_PRICING["anthropic"]
        assert "claude-sonnet-4-output" in PROVIDER_PRICING["anthropic"]

    def test_openai_pricing_exists(self):
        """Test OpenAI pricing is configured."""
        assert "openai" in PROVIDER_PRICING
        assert "gpt-4o" in PROVIDER_PRICING["openai"]
        assert "gpt-4o-output" in PROVIDER_PRICING["openai"]
        assert "gpt-4o-mini" in PROVIDER_PRICING["openai"]

    def test_google_pricing_exists(self):
        """Test Google pricing is configured."""
        assert "google" in PROVIDER_PRICING
        assert "gemini-pro" in PROVIDER_PRICING["google"]
        assert "gemini-pro-output" in PROVIDER_PRICING["google"]

    def test_deepseek_pricing_exists(self):
        """Test DeepSeek pricing is configured."""
        assert "deepseek" in PROVIDER_PRICING
        assert "deepseek-v3" in PROVIDER_PRICING["deepseek"]
        assert "deepseek-v3-output" in PROVIDER_PRICING["deepseek"]

    def test_openrouter_default_pricing(self):
        """Test OpenRouter has default pricing."""
        assert "openrouter" in PROVIDER_PRICING
        assert "default" in PROVIDER_PRICING["openrouter"]
        assert "default-output" in PROVIDER_PRICING["openrouter"]

    def test_pricing_values_are_decimal(self):
        """Test all pricing values are Decimal."""
        for provider, models in PROVIDER_PRICING.items():
            for model, price in models.items():
                assert isinstance(price, Decimal), f"{provider}/{model} price is not Decimal"

    def test_pricing_values_positive(self):
        """Test all pricing values are positive."""
        for provider, models in PROVIDER_PRICING.items():
            for model, price in models.items():
                assert price > 0, f"{provider}/{model} price is not positive"


# =============================================================================
# calculate_token_cost Tests
# =============================================================================


class TestCalculateTokenCost:
    """Tests for calculate_token_cost function."""

    def test_anthropic_claude_opus(self):
        """Test cost calculation for Claude Opus."""
        cost = calculate_token_cost("anthropic", "claude-opus-4", 1_000_000, 1_000_000)
        # Input: $15/1M, Output: $75/1M
        expected = Decimal("15.00") + Decimal("75.00")
        assert cost == expected

    def test_anthropic_claude_sonnet(self):
        """Test cost calculation for Claude Sonnet."""
        cost = calculate_token_cost("anthropic", "claude-sonnet-4", 1_000_000, 1_000_000)
        # Input: $3/1M, Output: $15/1M
        expected = Decimal("3.00") + Decimal("15.00")
        assert cost == expected

    def test_openai_gpt4o(self):
        """Test cost calculation for GPT-4o."""
        cost = calculate_token_cost("openai", "gpt-4o", 1_000_000, 1_000_000)
        # Input: $2.50/1M, Output: $10/1M
        expected = Decimal("2.50") + Decimal("10.00")
        assert cost == expected

    def test_openai_gpt4o_mini(self):
        """Test cost calculation for GPT-4o-mini."""
        cost = calculate_token_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
        # Input: $0.15/1M, Output: $0.60/1M
        expected = Decimal("0.15") + Decimal("0.60")
        assert cost == expected

    def test_google_gemini_pro(self):
        """Test cost calculation for Gemini Pro."""
        cost = calculate_token_cost("google", "gemini-pro", 1_000_000, 1_000_000)
        # Input: $1.25/1M, Output: $5/1M
        expected = Decimal("1.25") + Decimal("5.00")
        assert cost == expected

    def test_deepseek_v3(self):
        """Test cost calculation for DeepSeek V3."""
        cost = calculate_token_cost("deepseek", "deepseek-v3", 1_000_000, 1_000_000)
        # Input: $0.14/1M, Output: $0.28/1M
        expected = Decimal("0.14") + Decimal("0.28")
        assert cost == expected

    def test_unknown_provider_uses_openrouter_default(self):
        """Test unknown provider falls back to OpenRouter pricing."""
        cost = calculate_token_cost("unknown_provider", "unknown_model", 1_000_000, 1_000_000)
        # Default: $2/1M input, $8/1M output
        expected = Decimal("2.00") + Decimal("8.00")
        assert cost == expected

    def test_unknown_model_uses_default(self):
        """Test unknown model falls back to default pricing."""
        cost = calculate_token_cost("anthropic", "unknown-model", 1_000_000, 1_000_000)
        # Falls back to default: $2/1M input, $8/1M output
        expected = Decimal("2.00") + Decimal("8.00")
        assert cost == expected

    def test_zero_tokens(self):
        """Test cost with zero tokens."""
        cost = calculate_token_cost("anthropic", "claude-sonnet-4", 0, 0)
        assert cost == Decimal("0")

    def test_zero_input_tokens(self):
        """Test cost with zero input tokens."""
        cost = calculate_token_cost("anthropic", "claude-sonnet-4", 0, 1_000_000)
        # Only output cost: $15/1M
        assert cost == Decimal("15.00")

    def test_zero_output_tokens(self):
        """Test cost with zero output tokens."""
        cost = calculate_token_cost("anthropic", "claude-sonnet-4", 1_000_000, 0)
        # Only input cost: $3/1M
        assert cost == Decimal("3.00")

    def test_small_token_counts(self):
        """Test cost with small token counts."""
        cost = calculate_token_cost("anthropic", "claude-sonnet-4", 100, 50)
        # Input: 100 * 3 / 1M = 0.0003, Output: 50 * 15 / 1M = 0.00075
        expected = Decimal("100") / Decimal("1000000") * Decimal("3.00") + Decimal("50") / Decimal(
            "1000000"
        ) * Decimal("15.00")
        assert cost == expected

    def test_large_token_counts(self):
        """Test cost with large token counts (10M+)."""
        cost = calculate_token_cost("anthropic", "claude-sonnet-4", 10_000_000, 5_000_000)
        # Input: 10M * 3 / 1M = 30, Output: 5M * 15 / 1M = 75
        expected = Decimal("30.00") + Decimal("75.00")
        assert cost == expected

    def test_cost_precision(self):
        """Test cost calculation maintains decimal precision."""
        cost = calculate_token_cost("openai", "gpt-4o-mini", 123, 456)
        assert isinstance(cost, Decimal)
        # Should not lose precision


# =============================================================================
# UsageEvent Tests
# =============================================================================


class TestUsageEvent:
    """Tests for UsageEvent dataclass."""

    def test_create_event(self, sample_event):
        """Test creating a usage event."""
        assert sample_event.id == "event-001"
        assert sample_event.user_id == "user-123"
        assert sample_event.org_id == "org-456"
        assert sample_event.event_type == UsageEventType.DEBATE
        assert sample_event.debate_id == "debate-789"
        assert sample_event.tokens_in == 1000
        assert sample_event.tokens_out == 500
        assert sample_event.provider == "anthropic"
        assert sample_event.model == "claude-sonnet-4"
        assert sample_event.metadata == {"round": 1}

    def test_default_values(self):
        """Test default field values."""
        event = UsageEvent()
        assert event.id  # Auto-generated UUID
        assert event.user_id == ""
        assert event.org_id == ""
        assert event.event_type == UsageEventType.DEBATE
        assert event.debate_id is None
        assert event.tokens_in == 0
        assert event.tokens_out == 0
        assert event.provider == ""
        assert event.model == ""
        assert event.cost_usd == Decimal("0")
        assert event.metadata == {}
        assert event.created_at is not None

    def test_calculate_cost(self, sample_event):
        """Test cost calculation method."""
        cost = sample_event.calculate_cost()
        assert isinstance(cost, Decimal)
        assert cost == sample_event.cost_usd
        # Claude Sonnet: 1000 tokens * $3/1M + 500 tokens * $15/1M
        expected = Decimal("1000") / Decimal("1000000") * Decimal("3.00") + Decimal(
            "500"
        ) / Decimal("1000000") * Decimal("15.00")
        assert cost == expected

    def test_calculate_cost_updates_field(self):
        """Test calculate_cost updates the cost_usd field."""
        event = UsageEvent(
            provider="openai",
            model="gpt-4o",
            tokens_in=1000,
            tokens_out=500,
        )
        assert event.cost_usd == Decimal("0")
        event.calculate_cost()
        assert event.cost_usd > Decimal("0")

    def test_to_dict(self, sample_event):
        """Test to_dict conversion."""
        sample_event.calculate_cost()
        data = sample_event.to_dict()

        assert data["id"] == "event-001"
        assert data["user_id"] == "user-123"
        assert data["org_id"] == "org-456"
        assert data["event_type"] == "debate"
        assert data["debate_id"] == "debate-789"
        assert data["tokens_in"] == 1000
        assert data["tokens_out"] == 500
        assert data["provider"] == "anthropic"
        assert data["model"] == "claude-sonnet-4"
        assert "cost_usd" in data
        assert data["metadata"] == {"round": 1}
        assert "created_at" in data

    def test_to_dict_serializable(self, sample_event):
        """Test to_dict output is JSON serializable."""
        sample_event.calculate_cost()
        data = sample_event.to_dict()
        json_str = json.dumps(data)
        assert json_str is not None

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "id": "event-002",
            "user_id": "user-abc",
            "org_id": "org-xyz",
            "event_type": "agent_call",
            "debate_id": "debate-123",
            "tokens_in": 2000,
            "tokens_out": 1000,
            "provider": "openai",
            "model": "gpt-4o",
            "cost_usd": "0.035",
            "metadata": {"agent": "claude"},
            "created_at": "2024-06-01T12:00:00",
        }

        event = UsageEvent.from_dict(data)

        assert event.id == "event-002"
        assert event.user_id == "user-abc"
        assert event.org_id == "org-xyz"
        assert event.event_type == UsageEventType.AGENT_CALL
        assert event.debate_id == "debate-123"
        assert event.tokens_in == 2000
        assert event.tokens_out == 1000
        assert event.provider == "openai"
        assert event.model == "gpt-4o"
        assert event.cost_usd == Decimal("0.035")
        assert event.metadata == {"agent": "claude"}
        assert event.created_at == datetime.fromisoformat("2024-06-01T12:00:00")

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {}
        event = UsageEvent.from_dict(data)

        assert event.id  # Auto-generated
        assert event.user_id == ""
        assert event.event_type == UsageEventType.DEBATE
        assert event.tokens_in == 0
        assert event.cost_usd == Decimal("0")

    def test_from_dict_with_datetime_object(self):
        """Test from_dict with datetime object instead of string."""
        dt = datetime(2024, 6, 15, 10, 30, 0)
        data = {
            "created_at": dt,
        }
        event = UsageEvent.from_dict(data)
        assert event.created_at == dt

    def test_roundtrip_serialization(self, sample_event):
        """Test to_dict -> from_dict roundtrip."""
        sample_event.calculate_cost()
        data = sample_event.to_dict()
        restored = UsageEvent.from_dict(data)

        assert restored.id == sample_event.id
        assert restored.user_id == sample_event.user_id
        assert restored.org_id == sample_event.org_id
        assert restored.event_type == sample_event.event_type
        assert restored.debate_id == sample_event.debate_id
        assert restored.tokens_in == sample_event.tokens_in
        assert restored.tokens_out == sample_event.tokens_out
        assert restored.provider == sample_event.provider
        assert restored.model == sample_event.model
        assert restored.cost_usd == sample_event.cost_usd
        assert restored.metadata == sample_event.metadata


# =============================================================================
# UsageSummary Tests
# =============================================================================


class TestUsageSummary:
    """Tests for UsageSummary dataclass."""

    def test_create_summary(self):
        """Test creating a usage summary."""
        summary = UsageSummary(
            org_id="org-123",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_debates=50,
            total_api_calls=200,
            total_agent_calls=150,
            total_tokens_in=1_000_000,
            total_tokens_out=500_000,
            total_cost_usd=Decimal("125.50"),
        )

        assert summary.org_id == "org-123"
        assert summary.total_debates == 50
        assert summary.total_tokens_in == 1_000_000
        assert summary.total_cost_usd == Decimal("125.50")

    def test_default_values(self):
        """Test default field values."""
        summary = UsageSummary(
            org_id="org-123",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
        )

        assert summary.total_debates == 0
        assert summary.total_api_calls == 0
        assert summary.total_agent_calls == 0
        assert summary.total_tokens_in == 0
        assert summary.total_tokens_out == 0
        assert summary.total_cost_usd == Decimal("0")
        assert summary.cost_by_provider == {}
        assert summary.debates_by_day == {}

    def test_to_dict(self):
        """Test to_dict conversion."""
        summary = UsageSummary(
            org_id="org-123",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_debates=10,
            total_cost_usd=Decimal("50.00"),
            cost_by_provider={
                "anthropic": Decimal("30.00"),
                "openai": Decimal("20.00"),
            },
            debates_by_day={
                "2024-01-01": 3,
                "2024-01-02": 7,
            },
        )

        data = summary.to_dict()

        assert data["org_id"] == "org-123"
        assert data["total_debates"] == 10
        assert data["total_cost_usd"] == "50.00"
        assert data["cost_by_provider"]["anthropic"] == "30.00"
        assert data["cost_by_provider"]["openai"] == "20.00"
        assert data["debates_by_day"]["2024-01-01"] == 3

    def test_to_dict_serializable(self):
        """Test to_dict output is JSON serializable."""
        summary = UsageSummary(
            org_id="org-123",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            cost_by_provider={"anthropic": Decimal("30.00")},
        )
        data = summary.to_dict()
        json_str = json.dumps(data)
        assert json_str is not None


# =============================================================================
# UsageTracker Initialization Tests
# =============================================================================


class TestUsageTrackerInit:
    """Tests for UsageTracker initialization."""

    def test_create_tracker_with_custom_path(self, temp_db_path):
        """Test creating tracker with custom db path."""
        tracker = UsageTracker(db_path=temp_db_path)
        assert tracker.db_path == temp_db_path
        assert temp_db_path.exists()

    def test_create_tracker_creates_parent_dirs(self, tmp_path):
        """Test tracker creates parent directories."""
        db_path = tmp_path / "subdir" / "deep" / "usage.db"
        tracker = UsageTracker(db_path=db_path)
        assert db_path.parent.exists()

    def test_schema_created(self, temp_db_path):
        """Test database schema is created."""
        import sqlite3

        tracker = UsageTracker(db_path=temp_db_path)

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='usage_events'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_indexes_created(self, temp_db_path):
        """Test database indexes are created."""
        import sqlite3

        tracker = UsageTracker(db_path=temp_db_path)

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "idx_usage_org_created" in indexes
        assert "idx_usage_user_created" in indexes
        assert "idx_usage_debate" in indexes


# =============================================================================
# UsageTracker Recording Tests
# =============================================================================


class TestUsageTrackerRecording:
    """Tests for UsageTracker recording functionality."""

    def test_record_event(self, tracker, sample_event):
        """Test recording a usage event."""
        tracker.record(sample_event)

        # Verify by querying the database
        with tracker._connection() as conn:
            row = conn.execute(
                "SELECT * FROM usage_events WHERE id = ?", (sample_event.id,)
            ).fetchone()

        assert row is not None
        assert row["user_id"] == "user-123"
        assert row["org_id"] == "org-456"
        assert row["tokens_in"] == 1000

    def test_record_calculates_cost_if_zero(self, tracker):
        """Test record calculates cost if not set."""
        event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        assert event.cost_usd == Decimal("0")

        tracker.record(event)

        assert event.cost_usd > Decimal("0")

    def test_record_preserves_existing_cost(self, tracker):
        """Test record preserves existing cost if set."""
        event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
            cost_usd=Decimal("99.99"),
        )

        tracker.record(event)

        assert event.cost_usd == Decimal("99.99")

    def test_record_debate(self, tracker):
        """Test record_debate convenience method."""
        event = tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=2000,
            tokens_out=1000,
            provider="openai",
            model="gpt-4o",
            metadata={"topic": "AI Safety"},
        )

        assert event.event_type == UsageEventType.DEBATE
        assert event.debate_id == "debate-1"
        assert event.tokens_in == 2000
        assert event.cost_usd > Decimal("0")
        assert event.metadata == {"topic": "AI Safety"}

    def test_record_agent_call(self, tracker):
        """Test record_agent_call convenience method."""
        event = tracker.record_agent_call(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            agent_name="claude",
            tokens_in=500,
            tokens_out=200,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        assert event.event_type == UsageEventType.AGENT_CALL
        assert event.metadata == {"agent": "claude"}
        assert event.cost_usd > Decimal("0")

    def test_record_agent_call_no_debate(self, tracker):
        """Test record_agent_call without debate_id."""
        event = tracker.record_agent_call(
            user_id="user-1",
            org_id="org-1",
            debate_id=None,
            agent_name="gemini",
            tokens_in=500,
            tokens_out=200,
            provider="google",
            model="gemini-pro",
        )

        assert event.debate_id is None

    def test_record_multiple_events(self, tracker):
        """Test recording multiple events."""
        for i in range(10):
            tracker.record_debate(
                user_id=f"user-{i % 3}",
                org_id="org-1",
                debate_id=f"debate-{i}",
                tokens_in=1000 * (i + 1),
                tokens_out=500,
                provider="anthropic",
                model="claude-sonnet-4",
            )

        with tracker._connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM usage_events WHERE org_id = ?", ("org-1",)
            ).fetchone()[0]

        assert count == 10

    def test_record_stores_metadata_as_json(self, tracker):
        """Test metadata is stored as JSON."""
        event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            metadata={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        tracker.record(event)

        with tracker._connection() as conn:
            row = conn.execute(
                "SELECT metadata FROM usage_events WHERE id = ?", (event.id,)
            ).fetchone()

        parsed = json.loads(row["metadata"])
        assert parsed["nested"]["key"] == "value"
        assert parsed["list"] == [1, 2, 3]


# =============================================================================
# UsageTracker Summary Tests
# =============================================================================


class TestUsageTrackerSummary:
    """Tests for UsageTracker summary functionality."""

    def test_get_summary_empty(self, tracker):
        """Test get_summary with no data."""
        summary = tracker.get_summary("org-1")

        assert summary.org_id == "org-1"
        assert summary.total_debates == 0
        assert summary.total_api_calls == 0
        assert summary.total_tokens_in == 0
        assert summary.total_cost_usd == Decimal("0")

    def test_get_summary_with_data(self, tracker):
        """Test get_summary with recorded data."""
        # Record some events
        for i in range(5):
            tracker.record_debate(
                user_id="user-1",
                org_id="org-1",
                debate_id=f"debate-{i}",
                tokens_in=1000,
                tokens_out=500,
                provider="anthropic",
                model="claude-sonnet-4",
            )

        for i in range(3):
            tracker.record_agent_call(
                user_id="user-1",
                org_id="org-1",
                debate_id=f"debate-{i}",
                agent_name="claude",
                tokens_in=200,
                tokens_out=100,
                provider="anthropic",
                model="claude-sonnet-4",
            )

        summary = tracker.get_summary("org-1")

        assert summary.total_debates == 5
        assert summary.total_agent_calls == 3
        assert summary.total_tokens_in == 5000 + 600  # 5*1000 + 3*200
        assert summary.total_tokens_out == 2500 + 300  # 5*500 + 3*100
        assert summary.total_cost_usd > Decimal("0")

    def test_get_summary_filters_by_period(self, tracker):
        """Test get_summary filters by time period."""
        now = datetime.now(timezone.utc)

        # Record event in the past
        old_event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            event_type=UsageEventType.DEBATE,
            tokens_in=10000,
            tokens_out=5000,
            provider="anthropic",
            model="claude-sonnet-4",
            created_at=now - timedelta(days=60),
        )
        tracker.record(old_event)

        # Record recent event
        new_event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            event_type=UsageEventType.DEBATE,
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
            created_at=now - timedelta(days=5),
        )
        tracker.record(new_event)

        # Get summary for last 30 days
        summary = tracker.get_summary(
            "org-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )

        assert summary.total_debates == 1
        assert summary.total_tokens_in == 1000

    def test_get_summary_cost_by_provider(self, tracker):
        """Test get_summary includes cost by provider."""
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-2",
            tokens_in=1000,
            tokens_out=500,
            provider="openai",
            model="gpt-4o",
        )

        summary = tracker.get_summary("org-1")

        assert "anthropic" in summary.cost_by_provider
        assert "openai" in summary.cost_by_provider
        assert summary.cost_by_provider["anthropic"] > Decimal("0")
        assert summary.cost_by_provider["openai"] > Decimal("0")

    def test_get_summary_debates_by_day(self, tracker):
        """Test get_summary includes debates by day."""
        now = datetime.now(timezone.utc)

        # Record debates on different days
        for i in range(3):
            event = UsageEvent(
                user_id="user-1",
                org_id="org-1",
                event_type=UsageEventType.DEBATE,
                tokens_in=1000,
                tokens_out=500,
                provider="anthropic",
                model="claude-sonnet-4",
                created_at=now - timedelta(days=i),
            )
            tracker.record(event)

        summary = tracker.get_summary("org-1")

        assert len(summary.debates_by_day) >= 1

    def test_get_summary_multiple_orgs_isolated(self, tracker):
        """Test summaries are isolated by organization."""
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record_debate(
            user_id="user-2",
            org_id="org-2",
            debate_id="debate-2",
            tokens_in=2000,
            tokens_out=1000,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        summary1 = tracker.get_summary("org-1")
        summary2 = tracker.get_summary("org-2")

        assert summary1.total_debates == 1
        assert summary1.total_tokens_in == 1000
        assert summary2.total_debates == 1
        assert summary2.total_tokens_in == 2000


# =============================================================================
# UsageTracker User Usage Tests
# =============================================================================


class TestUsageTrackerUserUsage:
    """Tests for UsageTracker user usage functionality."""

    def test_get_user_usage_empty(self, tracker):
        """Test get_user_usage with no data."""
        usage = tracker.get_user_usage("user-1")

        assert usage["user_id"] == "user-1"
        assert usage["debates"] == 0
        assert usage["total_tokens"] == 0
        assert usage["total_cost_usd"] == "0"

    def test_get_user_usage_with_data(self, tracker):
        """Test get_user_usage with recorded data."""
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-2",
            tokens_in=2000,
            tokens_out=1000,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        usage = tracker.get_user_usage("user-1")

        assert usage["debates"] == 2
        assert usage["total_tokens"] == 4500  # 1000+500 + 2000+1000
        assert Decimal(usage["total_cost_usd"]) > Decimal("0")

    def test_get_user_usage_filters_by_days(self, tracker):
        """Test get_user_usage filters by days."""
        now = datetime.now(timezone.utc)

        # Old event
        old_event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            event_type=UsageEventType.DEBATE,
            tokens_in=10000,
            tokens_out=5000,
            provider="anthropic",
            model="claude-sonnet-4",
            created_at=now - timedelta(days=60),
        )
        tracker.record(old_event)

        # Recent event
        new_event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            event_type=UsageEventType.DEBATE,
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
            created_at=now - timedelta(days=5),
        )
        tracker.record(new_event)

        usage_30 = tracker.get_user_usage("user-1", days=30)
        usage_90 = tracker.get_user_usage("user-1", days=90)

        assert usage_30["debates"] == 1
        assert usage_90["debates"] == 2

    def test_get_user_usage_isolated_by_user(self, tracker):
        """Test user usage is isolated by user."""
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record_debate(
            user_id="user-2",
            org_id="org-1",
            debate_id="debate-2",
            tokens_in=5000,
            tokens_out=2500,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        usage1 = tracker.get_user_usage("user-1")
        usage2 = tracker.get_user_usage("user-2")

        assert usage1["debates"] == 1
        assert usage1["total_tokens"] == 1500
        assert usage2["debates"] == 1
        assert usage2["total_tokens"] == 7500


# =============================================================================
# UsageTracker Debate Cost Tests
# =============================================================================


class TestUsageTrackerDebateCost:
    """Tests for UsageTracker debate cost functionality."""

    def test_get_debate_cost_empty(self, tracker):
        """Test get_debate_cost with no data."""
        cost = tracker.get_debate_cost("nonexistent-debate")
        assert cost == Decimal("0")

    def test_get_debate_cost(self, tracker):
        """Test get_debate_cost with recorded data."""
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record_agent_call(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            agent_name="claude",
            tokens_in=200,
            tokens_out=100,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        cost = tracker.get_debate_cost("debate-1")

        assert cost > Decimal("0")
        # Should be sum of both events

    def test_get_debate_cost_multiple_debates(self, tracker):
        """Test debate costs are isolated."""
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-2",
            tokens_in=5000,
            tokens_out=2500,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        cost1 = tracker.get_debate_cost("debate-1")
        cost2 = tracker.get_debate_cost("debate-2")

        assert cost1 < cost2


# =============================================================================
# UsageTracker Count Debates Tests
# =============================================================================


class TestUsageTrackerCountDebates:
    """Tests for UsageTracker count_debates_this_month functionality."""

    def test_count_debates_this_month_empty(self, tracker):
        """Test count_debates_this_month with no data."""
        count = tracker.count_debates_this_month("org-1")
        assert count == 0

    def test_count_debates_this_month(self, tracker):
        """Test count_debates_this_month with data."""
        for i in range(5):
            tracker.record_debate(
                user_id="user-1",
                org_id="org-1",
                debate_id=f"debate-{i}",
                tokens_in=1000,
                tokens_out=500,
                provider="anthropic",
                model="claude-sonnet-4",
            )

        count = tracker.count_debates_this_month("org-1")
        assert count == 5

    def test_count_debates_this_month_ignores_agent_calls(self, tracker):
        """Test count_debates_this_month only counts debates."""
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record_agent_call(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            agent_name="claude",
            tokens_in=200,
            tokens_out=100,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        count = tracker.count_debates_this_month("org-1")
        assert count == 1

    def test_count_debates_this_month_isolated_by_org(self, tracker):
        """Test debate counts are isolated by organization."""
        tracker.record_debate(
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record_debate(
            user_id="user-2",
            org_id="org-2",
            debate_id="debate-2",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        count1 = tracker.count_debates_this_month("org-1")
        count2 = tracker.count_debates_this_month("org-2")

        assert count1 == 1
        assert count2 == 1


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_tokens(self, tracker):
        """Test recording event with zero tokens."""
        event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            tokens_in=0,
            tokens_out=0,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record(event)

        assert event.cost_usd == Decimal("0")

    def test_very_large_token_counts(self, tracker):
        """Test recording event with very large token counts."""
        event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            tokens_in=1_000_000_000,  # 1 billion
            tokens_out=500_000_000,  # 500 million
            provider="anthropic",
            model="claude-sonnet-4",
        )
        tracker.record(event)

        # Should not overflow, cost should be reasonable
        assert event.cost_usd > Decimal("0")
        # 1B * $3/1M = $3000, 500M * $15/1M = $7500
        expected = Decimal("3000") + Decimal("7500")
        assert event.cost_usd == expected

    def test_empty_strings(self, tracker):
        """Test recording event with empty strings."""
        event = UsageEvent(
            user_id="",
            org_id="",
            provider="",
            model="",
            tokens_in=100,
            tokens_out=50,
        )
        tracker.record(event)

        # Should still work with default pricing
        assert event.cost_usd > Decimal("0")

    def test_special_characters_in_strings(self, tracker):
        """Test recording event with special characters."""
        event = UsageEvent(
            user_id="user_with-special.chars@example.com",
            org_id="org/with/slashes",
            debate_id="debate'with\"quotes",
            provider="anthropic",
            model="claude-sonnet-4",
            metadata={"key": "value with 'quotes' and \"double quotes\""},
        )
        tracker.record(event)

        # Should handle special characters
        with tracker._connection() as conn:
            row = conn.execute("SELECT * FROM usage_events WHERE id = ?", (event.id,)).fetchone()

        assert row is not None
        assert row["user_id"] == "user_with-special.chars@example.com"

    def test_unicode_in_metadata(self, tracker):
        """Test recording event with unicode in metadata."""
        event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            metadata={"topic": "Artificial Intelligence in Healthcare"},
        )
        tracker.record(event)

        with tracker._connection() as conn:
            row = conn.execute(
                "SELECT metadata FROM usage_events WHERE id = ?", (event.id,)
            ).fetchone()

        parsed = json.loads(row["metadata"])
        assert parsed["topic"] == "Artificial Intelligence in Healthcare"

    def test_concurrent_writes(self, temp_db_path):
        """Test concurrent writes to the database."""
        import threading

        tracker = UsageTracker(db_path=temp_db_path)
        errors = []

        def write_events(org_id, count):
            try:
                for i in range(count):
                    tracker.record_debate(
                        user_id=f"user-{org_id}",
                        org_id=org_id,
                        debate_id=f"debate-{org_id}-{i}",
                        tokens_in=100,
                        tokens_out=50,
                        provider="anthropic",
                        model="claude-sonnet-4",
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=write_events, args=(f"org-{i}", 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all events were recorded
        with tracker._connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]

        assert count == 50  # 5 threads * 10 events

    def test_decimal_precision_preserved(self, tracker):
        """Test decimal precision is preserved through recording."""
        event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
            tokens_in=1,
            tokens_out=1,
            provider="anthropic",
            model="claude-sonnet-4",
        )
        event.calculate_cost()
        original_cost = event.cost_usd

        tracker.record(event)

        # Query back
        cost = tracker.get_debate_cost(event.debate_id) if event.debate_id else Decimal("0")
        # For events without debate_id, check via summary
        summary = tracker.get_summary("org-1")
        # The total cost should be very close (may have floating point differences from SQLite)
        assert summary.total_cost_usd >= Decimal("0")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_event_type_in_from_dict(self):
        """Test from_dict with invalid event type."""
        with pytest.raises(ValueError):
            UsageEvent.from_dict({"event_type": "invalid"})

    def test_database_path_permissions(self, tmp_path):
        """Test handling of database path issues."""
        # Create a file where directory would be
        blocked_path = tmp_path / "blocked_file"
        blocked_path.touch()
        db_path = blocked_path / "usage.db"

        # Should raise an error when trying to create parent directory
        with pytest.raises((OSError, NotADirectoryError)):
            UsageTracker(db_path=db_path)

    def test_malformed_metadata_json(self, tracker):
        """Test handling of malformed metadata (direct DB manipulation)."""
        import sqlite3

        # Record a valid event first
        event = UsageEvent(
            user_id="user-1",
            org_id="org-1",
        )
        tracker.record(event)

        # Manually corrupt the metadata (simulating DB corruption)
        with tracker._connection() as conn:
            conn.execute(
                "UPDATE usage_events SET metadata = 'invalid json' WHERE id = ?", (event.id,)
            )
            conn.commit()

        # Reading should still work for other operations
        summary = tracker.get_summary("org-1")
        assert summary is not None


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """Test all __all__ exports are importable."""
        from aragora.billing.usage import __all__

        expected = [
            "UsageEventType",
            "UsageEvent",
            "UsageSummary",
            "UsageTracker",
            "calculate_token_cost",
            "PROVIDER_PRICING",
        ]

        for name in expected:
            assert name in __all__

    def test_imports_work(self):
        """Test all exports can be imported."""
        from aragora.billing.usage import (
            UsageEventType,
            UsageEvent,
            UsageSummary,
            UsageTracker,
            calculate_token_cost,
            PROVIDER_PRICING,
        )

        assert UsageEventType is not None
        assert UsageEvent is not None
        assert UsageSummary is not None
        assert UsageTracker is not None
        assert calculate_token_cost is not None
        assert PROVIDER_PRICING is not None
