"""
Comprehensive tests for the FactRegistry module.

Tests cover:
- RegisteredFact dataclass creation and defaults
- Staleness calculation (days since last_verified)
- Confidence decay calculation (decay_rate, min_confidence)
- needs_reverification property (threshold logic)
- Supersession tracking (is_superseded property)
- RegisteredFact methods (refresh, add_contradiction, supersede, to_dict, from_dict)
- FactRegistry initialization (with/without vector store)
- register() - basic registration, auto-duplicate detection, vertical classification
- query() - filtering by workspace, vertical, confidence, source
- get_fact() - retrieval by ID, not found case
- refresh_fact() - update last_verified timestamp
- add_contradiction() - relationship tracking
- supersede_fact() - mark as superseded
- get_stale_facts() - stale detection with vertical-specific rates
- get_stats() - statistics gathering
- Memory fallback when vector store unavailable
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.fact_registry import (
    FactRegistry,
    RegisteredFact,
)
from aragora.knowledge.mound_core import ProvenanceType
from aragora.knowledge.types import ValidationStatus


# ============================================================================
# Mock Classes for Testing
# ============================================================================


@dataclass
class MockSearchResult:
    """Mock search result from vector store."""

    id: str
    content: str
    score: float
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockVectorStore:
    """Mock vector store for testing FactRegistry."""

    def __init__(self):
        self.data: dict[str, dict[str, Any]] = {}
        self.connected = False
        self.connect_called = False
        self.upsert_calls: list[dict] = []
        self.search_calls: list[dict] = []

    async def connect(self) -> None:
        self.connected = True
        self.connect_called = True

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any],
        namespace: str = "default",
    ) -> None:
        self.upsert_calls.append(
            {
                "id": id,
                "embedding": embedding,
                "content": content,
                "metadata": metadata,
                "namespace": namespace,
            }
        )
        self.data[id] = {
            "id": id,
            "embedding": embedding,
            "content": content,
            "metadata": metadata,
            "namespace": namespace,
        }

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        namespace: str = "default",
        min_score: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MockSearchResult]:
        self.search_calls.append(
            {
                "embedding": embedding,
                "limit": limit,
                "namespace": namespace,
                "min_score": min_score,
                "filters": filters,
            }
        )
        # Return matching results from data
        results = []
        for item_id, item in self.data.items():
            if item["namespace"] == namespace:
                # Apply filters if any
                if filters:
                    match = True
                    for key, value in filters.items():
                        if item["metadata"].get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                results.append(
                    MockSearchResult(
                        id=item_id,
                        content=item["content"],
                        score=0.98,
                        embedding=item["embedding"],
                        metadata=item["metadata"],
                    )
                )
        return results[:limit]


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self, embedding: list[float] | None = None):
        self.default_embedding = embedding or [0.1] * 384
        self.embed_calls: list[str] = []
        self.should_fail = False

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        if self.should_fail:
            raise RuntimeError("Embedding service unavailable")
        return self.default_embedding


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    """Create a mock vector store."""
    return MockVectorStore()


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    """Create a mock embedding service."""
    return MockEmbeddingService()


@pytest.fixture
def fact_registry(
    mock_vector_store: MockVectorStore,
    mock_embedding_service: MockEmbeddingService,
) -> FactRegistry:
    """Create a FactRegistry with mocked dependencies."""
    return FactRegistry(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )


@pytest.fixture
def fact_registry_no_vector() -> FactRegistry:
    """Create a FactRegistry without vector store."""
    return FactRegistry()


@pytest.fixture
def sample_fact() -> RegisteredFact:
    """Create a sample RegisteredFact for testing."""
    return RegisteredFact(
        id="fact_test123",
        statement="API keys should never be committed to version control",
        vertical="software",
        category="best_practice",
        base_confidence=0.9,
        workspace_id="ws_test",
        topics=["security", "git"],
        source_ids=["doc_1", "doc_2"],
    )


@pytest.fixture
def stale_fact() -> RegisteredFact:
    """Create a stale fact (verified 30 days ago with fast decay)."""
    verification_date = datetime.now() - timedelta(days=30)
    return RegisteredFact(
        id="fact_stale123",
        statement="Old security advisory",
        vertical="software",
        category="vulnerability",
        base_confidence=0.8,
        verification_date=verification_date,
        decay_rate=0.05,  # Fast decay
        workspace_id="ws_test",
    )


# ============================================================================
# RegisteredFact Dataclass Tests
# ============================================================================


class TestRegisteredFactCreation:
    """Tests for RegisteredFact dataclass creation and defaults."""

    def test_basic_creation(self):
        """Test creating a RegisteredFact with minimal arguments."""
        fact = RegisteredFact(
            id="fact_001",
            statement="Test statement",
        )

        assert fact.id == "fact_001"
        assert fact.statement == "Test statement"
        assert fact.vertical == "general"
        assert fact.category == "general"
        assert fact.base_confidence == 0.5
        assert fact.decay_rate == 0.02
        assert fact.verification_count == 0
        assert fact.contradiction_count == 0
        assert fact.validation_status == ValidationStatus.UNVERIFIED
        assert fact.source_type == ProvenanceType.DOCUMENT
        assert fact.source_ids == []
        assert fact.contributing_agents == []
        assert fact.workspace_id == ""
        assert fact.topics == []
        assert fact.supports == []
        assert fact.contradicts == []
        assert fact.derived_from == []
        assert fact.embedding is None
        assert fact.superseded_at is None
        assert fact.superseded_by is None

    def test_creation_with_all_arguments(self):
        """Test creating a RegisteredFact with all arguments."""
        now = datetime.now()
        fact = RegisteredFact(
            id="fact_002",
            statement="Complete test statement",
            vertical="software",
            category="vulnerability",
            base_confidence=0.95,
            verification_date=now,
            decay_rate=0.05,
            validation_status=ValidationStatus.MAJORITY_AGREED,
            verification_count=3,
            contradiction_count=1,
            source_type=ProvenanceType.DEBATE,
            source_ids=["src1", "src2"],
            contributing_agents=["claude", "gpt4"],
            workspace_id="ws_prod",
            topics=["security", "web"],
            supports=["fact_100"],
            contradicts=["fact_101"],
            derived_from=["fact_102"],
            embedding=[0.1, 0.2, 0.3],
            created_at=now,
            updated_at=now,
            superseded_at=now,
            superseded_by="fact_200",
        )

        assert fact.id == "fact_002"
        assert fact.statement == "Complete test statement"
        assert fact.vertical == "software"
        assert fact.category == "vulnerability"
        assert fact.base_confidence == 0.95
        assert fact.verification_date == now
        assert fact.decay_rate == 0.05
        assert fact.validation_status == ValidationStatus.MAJORITY_AGREED
        assert fact.verification_count == 3
        assert fact.contradiction_count == 1
        assert fact.source_type == ProvenanceType.DEBATE
        assert fact.source_ids == ["src1", "src2"]
        assert fact.contributing_agents == ["claude", "gpt4"]
        assert fact.workspace_id == "ws_prod"
        assert fact.topics == ["security", "web"]
        assert fact.supports == ["fact_100"]
        assert fact.contradicts == ["fact_101"]
        assert fact.derived_from == ["fact_102"]
        assert fact.embedding == [0.1, 0.2, 0.3]
        assert fact.superseded_at == now
        assert fact.superseded_by == "fact_200"


class TestRegisteredFactStaleness:
    """Tests for staleness_days property calculation."""

    def test_fresh_fact_staleness(self):
        """Test staleness is near zero for fresh fact."""
        fact = RegisteredFact(
            id="fact_fresh",
            statement="Fresh fact",
            verification_date=datetime.now(),
        )

        # Should be very close to 0
        assert fact.staleness_days < 0.001

    def test_one_day_old_staleness(self):
        """Test staleness for 1 day old fact."""
        fact = RegisteredFact(
            id="fact_1day",
            statement="One day old fact",
            verification_date=datetime.now() - timedelta(days=1),
        )

        assert 0.99 < fact.staleness_days < 1.01

    def test_ten_days_old_staleness(self):
        """Test staleness for 10 days old fact."""
        fact = RegisteredFact(
            id="fact_10days",
            statement="Ten days old fact",
            verification_date=datetime.now() - timedelta(days=10),
        )

        assert 9.99 < fact.staleness_days < 10.01

    def test_staleness_with_hours(self):
        """Test staleness with fractional days."""
        fact = RegisteredFact(
            id="fact_hours",
            statement="Half day old fact",
            verification_date=datetime.now() - timedelta(hours=12),
        )

        assert 0.49 < fact.staleness_days < 0.51


class TestRegisteredFactConfidenceDecay:
    """Tests for current_confidence property with decay calculation."""

    def test_fresh_fact_no_decay(self):
        """Test confidence is at base for fresh fact."""
        fact = RegisteredFact(
            id="fact_fresh",
            statement="Fresh fact",
            base_confidence=0.9,
            verification_date=datetime.now(),
            decay_rate=0.02,
        )

        # Should be very close to base confidence
        assert abs(fact.current_confidence - 0.9) < 0.01

    def test_decay_after_10_days(self):
        """Test confidence decay after 10 days."""
        fact = RegisteredFact(
            id="fact_10days",
            statement="Ten days old fact",
            base_confidence=1.0,
            verification_date=datetime.now() - timedelta(days=10),
            decay_rate=0.02,
        )

        # decay = min(0.9, 10 * 0.02) = 0.2
        # current = 1.0 * (1.0 - 0.2) = 0.8
        expected = 1.0 * (1.0 - 0.2)
        assert abs(fact.current_confidence - expected) < 0.01

    def test_decay_caps_at_90_percent(self):
        """Test decay is capped at 90%."""
        fact = RegisteredFact(
            id="fact_very_old",
            statement="Very old fact",
            base_confidence=1.0,
            verification_date=datetime.now() - timedelta(days=100),
            decay_rate=0.02,  # Would be 200% decay without cap
        )

        # decay = min(0.9, 100 * 0.02) = 0.9
        # current = 1.0 * (1.0 - 0.9) = 0.1
        expected = 1.0 * (1.0 - 0.9)
        assert abs(fact.current_confidence - expected) < 0.01

    def test_verification_bonus(self):
        """Test verification count bonus to confidence."""
        fact = RegisteredFact(
            id="fact_verified",
            statement="Well verified fact",
            base_confidence=0.5,
            verification_date=datetime.now(),
            verification_count=5,  # 5 * 0.02 = 0.1 bonus
            decay_rate=0.02,
        )

        # base = 0.5, decay = 0, bonus = min(0.2, 0.1) = 0.1
        # current = 0.5 + 0.1 = 0.6
        assert abs(fact.current_confidence - 0.6) < 0.01

    def test_verification_bonus_caps_at_20_percent(self):
        """Test verification bonus is capped at 0.2."""
        fact = RegisteredFact(
            id="fact_many_verifications",
            statement="Many times verified fact",
            base_confidence=0.5,
            verification_date=datetime.now(),
            verification_count=20,  # Would be 0.4 bonus without cap
            decay_rate=0.02,
        )

        # bonus = min(0.2, 20 * 0.02) = 0.2
        # current = 0.5 + 0.2 = 0.7
        assert abs(fact.current_confidence - 0.7) < 0.01

    def test_contradiction_penalty(self):
        """Test contradiction count penalty to confidence."""
        fact = RegisteredFact(
            id="fact_contradicted",
            statement="Contradicted fact",
            base_confidence=0.8,
            verification_date=datetime.now(),
            contradiction_count=2,  # 2 * 0.05 = 0.1 penalty
            decay_rate=0.02,
        )

        # base = 0.8, decay = 0, bonus = 0, penalty = 0.1
        # current = 0.8 - 0.1 = 0.7
        assert abs(fact.current_confidence - 0.7) < 0.01

    def test_contradiction_penalty_caps_at_30_percent(self):
        """Test contradiction penalty is capped at 0.3."""
        fact = RegisteredFact(
            id="fact_many_contradictions",
            statement="Many times contradicted fact",
            base_confidence=0.8,
            verification_date=datetime.now(),
            contradiction_count=10,  # Would be 0.5 penalty without cap
            decay_rate=0.02,
        )

        # penalty = min(0.3, 10 * 0.05) = 0.3
        # current = 0.8 - 0.3 = 0.5
        assert abs(fact.current_confidence - 0.5) < 0.01

    def test_confidence_floor_at_zero(self):
        """Test confidence cannot go below 0."""
        fact = RegisteredFact(
            id="fact_very_low",
            statement="Very low confidence fact",
            base_confidence=0.1,
            verification_date=datetime.now() - timedelta(days=50),
            contradiction_count=6,  # 0.3 penalty
            decay_rate=0.02,
        )

        # decay = min(0.9, 50 * 0.02) = 0.9
        # adjusted = 0.1 * (1 - 0.9) = 0.01
        # penalty = 0.3
        # current = max(0, 0.01 - 0.3) = 0
        assert fact.current_confidence == 0.0

    def test_confidence_ceiling_at_one(self):
        """Test confidence cannot exceed 1."""
        fact = RegisteredFact(
            id="fact_boosted",
            statement="Highly boosted fact",
            base_confidence=1.0,
            verification_date=datetime.now(),
            verification_count=15,  # 0.2 bonus (capped)
            decay_rate=0.02,
        )

        # base = 1.0, bonus = 0.2
        # current = min(1.0, 1.0 + 0.2) = 1.0
        assert fact.current_confidence == 1.0


class TestNeedsReverification:
    """Tests for needs_reverification property."""

    def test_fresh_fact_no_reverification_needed(self):
        """Test fresh fact does not need reverification."""
        fact = RegisteredFact(
            id="fact_fresh",
            statement="Fresh fact",
            base_confidence=0.8,
            verification_date=datetime.now(),
        )

        assert not fact.needs_reverification

    def test_stale_fact_needs_reverification(self):
        """Test stale fact needs reverification."""
        # With decay_rate=0.02 and 30 days, decay = 0.6
        # current = 0.8 * (1 - 0.6) = 0.32
        # threshold = 0.5 * 0.8 = 0.4
        # 0.32 < 0.4, needs reverification
        fact = RegisteredFact(
            id="fact_stale",
            statement="Stale fact",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=30),
            decay_rate=0.02,
        )

        assert fact.needs_reverification

    def test_borderline_reverification_just_below(self):
        """Test fact just below reverification threshold needs it."""
        # With decay_rate=0.02 and 26 days:
        # decay = 0.52, current = 0.8 * (1 - 0.52) = 0.384
        # threshold = 0.5 * 0.8 = 0.4
        # 0.384 < 0.4, so needs reverification
        fact = RegisteredFact(
            id="fact_borderline",
            statement="Borderline fact",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=26),
            decay_rate=0.02,
        )

        assert fact.needs_reverification

    def test_borderline_reverification_just_above(self):
        """Test fact just above reverification threshold does not need it."""
        # With decay_rate=0.02 and 24 days:
        # decay = 0.48, current = 0.8 * (1 - 0.48) = 0.416
        # threshold = 0.5 * 0.8 = 0.4
        # 0.416 > 0.4, so does not need reverification
        fact = RegisteredFact(
            id="fact_ok",
            statement="OK fact",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=24),
            decay_rate=0.02,
        )

        assert not fact.needs_reverification


class TestIsSuperseded:
    """Tests for is_superseded property."""

    def test_not_superseded(self):
        """Test fact without supersession."""
        fact = RegisteredFact(
            id="fact_active",
            statement="Active fact",
        )

        assert not fact.is_superseded
        assert fact.superseded_at is None
        assert fact.superseded_by is None

    def test_is_superseded(self):
        """Test superseded fact."""
        fact = RegisteredFact(
            id="fact_old",
            statement="Old fact",
            superseded_at=datetime.now(),
            superseded_by="fact_new",
        )

        assert fact.is_superseded


# ============================================================================
# RegisteredFact Method Tests
# ============================================================================


class TestRegisteredFactRefresh:
    """Tests for RegisteredFact.refresh() method."""

    def test_refresh_updates_verification_date(self):
        """Test refresh updates verification date."""
        old_date = datetime.now() - timedelta(days=10)
        fact = RegisteredFact(
            id="fact_refresh",
            statement="Refreshable fact",
            verification_date=old_date,
            verification_count=2,
        )

        before_refresh = datetime.now()
        fact.refresh()
        after_refresh = datetime.now()

        assert before_refresh <= fact.verification_date <= after_refresh
        assert fact.verification_count == 3  # Incremented

    def test_refresh_updates_confidence(self):
        """Test refresh can update base confidence."""
        fact = RegisteredFact(
            id="fact_refresh_conf",
            statement="Refreshable fact",
            base_confidence=0.5,
        )

        fact.refresh(new_confidence=0.9)

        assert fact.base_confidence == 0.9

    def test_refresh_updates_updated_at(self):
        """Test refresh updates updated_at timestamp."""
        old_update = datetime.now() - timedelta(days=5)
        fact = RegisteredFact(
            id="fact_refresh_update",
            statement="Refreshable fact",
            updated_at=old_update,
        )

        before_refresh = datetime.now()
        fact.refresh()
        after_refresh = datetime.now()

        assert before_refresh <= fact.updated_at <= after_refresh


class TestRegisteredFactAddContradiction:
    """Tests for RegisteredFact.add_contradiction() method."""

    def test_add_contradiction(self):
        """Test adding a contradiction."""
        fact = RegisteredFact(
            id="fact_main",
            statement="Main fact",
        )

        fact.add_contradiction("fact_contradicting")

        assert "fact_contradicting" in fact.contradicts
        assert fact.contradiction_count == 1

    def test_add_multiple_contradictions(self):
        """Test adding multiple contradictions."""
        fact = RegisteredFact(
            id="fact_main",
            statement="Main fact",
        )

        fact.add_contradiction("fact_c1")
        fact.add_contradiction("fact_c2")
        fact.add_contradiction("fact_c3")

        assert len(fact.contradicts) == 3
        assert fact.contradiction_count == 3

    def test_add_duplicate_contradiction(self):
        """Test adding duplicate contradiction is ignored."""
        fact = RegisteredFact(
            id="fact_main",
            statement="Main fact",
        )

        fact.add_contradiction("fact_c1")
        fact.add_contradiction("fact_c1")  # Duplicate

        assert len(fact.contradicts) == 1
        assert fact.contradiction_count == 1

    def test_add_contradiction_updates_timestamp(self):
        """Test adding contradiction updates updated_at."""
        old_update = datetime.now() - timedelta(days=1)
        fact = RegisteredFact(
            id="fact_main",
            statement="Main fact",
            updated_at=old_update,
        )

        before_add = datetime.now()
        fact.add_contradiction("fact_c1")
        after_add = datetime.now()

        assert before_add <= fact.updated_at <= after_add


class TestRegisteredFactSupersede:
    """Tests for RegisteredFact.supersede() method."""

    def test_supersede_marks_fact(self):
        """Test supersede marks fact correctly."""
        fact = RegisteredFact(
            id="fact_old",
            statement="Old fact",
        )

        before_supersede = datetime.now()
        fact.supersede("fact_new")
        after_supersede = datetime.now()

        assert fact.is_superseded
        assert fact.superseded_by == "fact_new"
        assert before_supersede <= fact.superseded_at <= after_supersede

    def test_supersede_updates_timestamp(self):
        """Test supersede updates updated_at."""
        old_update = datetime.now() - timedelta(days=1)
        fact = RegisteredFact(
            id="fact_old",
            statement="Old fact",
            updated_at=old_update,
        )

        before_supersede = datetime.now()
        fact.supersede("fact_new")
        after_supersede = datetime.now()

        assert before_supersede <= fact.updated_at <= after_supersede


class TestRegisteredFactToDict:
    """Tests for RegisteredFact.to_dict() method."""

    def test_to_dict_basic(self, sample_fact: RegisteredFact):
        """Test to_dict includes all fields."""
        result = sample_fact.to_dict()

        assert result["id"] == sample_fact.id
        assert result["statement"] == sample_fact.statement
        assert result["vertical"] == sample_fact.vertical
        assert result["category"] == sample_fact.category
        assert result["base_confidence"] == sample_fact.base_confidence
        # Use approximate comparison for current_confidence (computed property)
        assert abs(result["current_confidence"] - sample_fact.current_confidence) < 0.0001
        assert result["decay_rate"] == sample_fact.decay_rate
        assert result["validation_status"] == sample_fact.validation_status.value
        assert result["verification_count"] == sample_fact.verification_count
        assert result["contradiction_count"] == sample_fact.contradiction_count
        assert result["source_type"] == sample_fact.source_type.value
        assert result["source_ids"] == sample_fact.source_ids
        assert result["contributing_agents"] == sample_fact.contributing_agents
        assert result["workspace_id"] == sample_fact.workspace_id
        assert result["topics"] == sample_fact.topics
        assert result["supports"] == sample_fact.supports
        assert result["contradicts"] == sample_fact.contradicts
        assert result["derived_from"] == sample_fact.derived_from
        assert result["needs_reverification"] == sample_fact.needs_reverification
        assert result["is_superseded"] == sample_fact.is_superseded

    def test_to_dict_dates_are_iso_format(self, sample_fact: RegisteredFact):
        """Test dates are converted to ISO format strings."""
        result = sample_fact.to_dict()

        # Verify date strings are valid ISO format
        datetime.fromisoformat(result["verification_date"])
        datetime.fromisoformat(result["created_at"])
        datetime.fromisoformat(result["updated_at"])

    def test_to_dict_superseded_fact(self):
        """Test to_dict with superseded fact."""
        fact = RegisteredFact(
            id="fact_superseded",
            statement="Superseded fact",
            superseded_at=datetime.now(),
            superseded_by="fact_new",
        )

        result = fact.to_dict()

        assert result["superseded_at"] is not None
        assert result["superseded_by"] == "fact_new"
        assert result["is_superseded"] is True


class TestRegisteredFactFromDict:
    """Tests for RegisteredFact.from_dict() class method."""

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "id": "fact_min",
            "statement": "Minimal fact",
        }

        fact = RegisteredFact.from_dict(data)

        assert fact.id == "fact_min"
        assert fact.statement == "Minimal fact"
        assert fact.vertical == "general"
        assert fact.category == "general"
        assert fact.base_confidence == 0.5
        assert fact.decay_rate == 0.02

    def test_from_dict_complete(self, sample_fact: RegisteredFact):
        """Test from_dict with complete data."""
        data = sample_fact.to_dict()
        reconstructed = RegisteredFact.from_dict(data)

        assert reconstructed.id == sample_fact.id
        assert reconstructed.statement == sample_fact.statement
        assert reconstructed.vertical == sample_fact.vertical
        assert reconstructed.category == sample_fact.category
        assert reconstructed.base_confidence == sample_fact.base_confidence
        assert reconstructed.validation_status == sample_fact.validation_status
        assert reconstructed.workspace_id == sample_fact.workspace_id

    def test_from_dict_parses_dates(self):
        """Test from_dict parses ISO date strings."""
        now = datetime.now()
        data = {
            "id": "fact_dates",
            "statement": "Fact with dates",
            "verification_date": now.isoformat(),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        fact = RegisteredFact.from_dict(data)

        # Dates should be datetime objects, not strings
        assert isinstance(fact.verification_date, datetime)
        assert isinstance(fact.created_at, datetime)
        assert isinstance(fact.updated_at, datetime)

    def test_from_dict_parses_enums(self):
        """Test from_dict parses enum values."""
        data = {
            "id": "fact_enums",
            "statement": "Fact with enums",
            "validation_status": "majority_agreed",
            "source_type": "debate",
        }

        fact = RegisteredFact.from_dict(data)

        assert fact.validation_status == ValidationStatus.MAJORITY_AGREED
        assert fact.source_type == ProvenanceType.DEBATE

    def test_from_dict_superseded(self):
        """Test from_dict with superseded fact."""
        now = datetime.now()
        data = {
            "id": "fact_superseded",
            "statement": "Superseded fact",
            "superseded_at": now.isoformat(),
            "superseded_by": "fact_new",
        }

        fact = RegisteredFact.from_dict(data)

        assert fact.is_superseded
        assert fact.superseded_by == "fact_new"
        assert isinstance(fact.superseded_at, datetime)


# ============================================================================
# FactRegistry Initialization Tests
# ============================================================================


class TestFactRegistryInit:
    """Tests for FactRegistry initialization."""

    def test_init_without_stores(self):
        """Test initialization without vector store or embedding service."""
        registry = FactRegistry()

        assert registry._vector_store is None
        assert registry._embedding_service is None
        assert registry._facts == {}
        assert not registry._initialized

    def test_init_with_vector_store(self, mock_vector_store: MockVectorStore):
        """Test initialization with vector store."""
        registry = FactRegistry(vector_store=mock_vector_store)

        assert registry._vector_store is mock_vector_store
        assert registry._embedding_service is None

    def test_init_with_embedding_service(self, mock_embedding_service: MockEmbeddingService):
        """Test initialization with embedding service."""
        registry = FactRegistry(embedding_service=mock_embedding_service)

        assert registry._vector_store is None
        assert registry._embedding_service is mock_embedding_service

    def test_init_with_both(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedding_service: MockEmbeddingService,
    ):
        """Test initialization with both stores."""
        registry = FactRegistry(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )

        assert registry._vector_store is mock_vector_store
        assert registry._embedding_service is mock_embedding_service


class TestFactRegistryInitialize:
    """Tests for FactRegistry.initialize() method."""

    @pytest.mark.asyncio
    async def test_initialize_connects_vector_store(self, mock_vector_store: MockVectorStore):
        """Test initialize connects to vector store."""
        registry = FactRegistry(vector_store=mock_vector_store)

        await registry.initialize()

        assert mock_vector_store.connect_called
        assert mock_vector_store.connected
        assert registry._initialized

    @pytest.mark.asyncio
    async def test_initialize_without_vector_store(self):
        """Test initialize works without vector store."""
        registry = FactRegistry()

        await registry.initialize()

        assert registry._initialized


# ============================================================================
# FactRegistry.register() Tests
# ============================================================================


class TestFactRegistryRegister:
    """Tests for FactRegistry.register() method."""

    @pytest.mark.asyncio
    async def test_register_basic(self, fact_registry: FactRegistry):
        """Test basic fact registration."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Python is a programming language",
            vertical="software",
            category="general",
            confidence=0.9,
        )

        assert fact.id.startswith("fact_")
        assert fact.statement == "Python is a programming language"
        assert fact.vertical == "software"
        assert fact.category == "general"
        assert fact.base_confidence == 0.9
        assert fact_registry._facts[fact.id] == fact

    @pytest.mark.asyncio
    async def test_register_with_source_info(self, fact_registry: FactRegistry):
        """Test registration with source information."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="API keys are secrets",
            source_type=ProvenanceType.DOCUMENT,
            source_ids=["doc_1", "doc_2"],
            workspace_id="ws_prod",
            topics=["security", "api"],
        )

        assert fact.source_type == ProvenanceType.DOCUMENT
        assert fact.source_ids == ["doc_1", "doc_2"]
        assert fact.workspace_id == "ws_prod"
        assert fact.topics == ["security", "api"]

    @pytest.mark.asyncio
    async def test_register_stores_in_vector_store(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedding_service: MockEmbeddingService,
    ):
        """Test registration stores in vector store."""
        registry = FactRegistry(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )
        await registry.initialize()

        fact = await registry.register(
            statement="Test fact for vector store",
            vertical="software",
        )

        # Verify upsert was called
        assert len(mock_vector_store.upsert_calls) == 1
        upsert = mock_vector_store.upsert_calls[0]
        assert upsert["id"] == fact.id
        assert upsert["content"] == fact.statement
        assert upsert["namespace"] == "software"
        assert upsert["metadata"]["vertical"] == "software"

    @pytest.mark.asyncio
    async def test_register_vertical_decay_rates(self, fact_registry: FactRegistry):
        """Test registration uses vertical-specific decay rates."""
        await fact_registry.initialize()

        # Software vulnerability - higher decay rate
        vuln_fact = await fact_registry.register(
            statement="CVE-2024-0001 vulnerability",
            vertical="software",
            category="vulnerability",
        )

        # Legal clause - stable decay rate
        # Note: Uses actual rates from LegalKnowledge vertical module
        legal_fact = await fact_registry.register(
            statement="Contract termination clause",
            vertical="legal",
            category="clause",
        )

        assert vuln_fact.decay_rate == 0.05  # Software vulnerability rate
        assert legal_fact.decay_rate == 0.005  # Legal clause rate (from LegalKnowledge)

    @pytest.mark.asyncio
    async def test_register_duplicate_detection(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedding_service: MockEmbeddingService,
    ):
        """Test near-duplicate detection on registration."""
        registry = FactRegistry(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )
        await registry.initialize()

        # Register first fact
        fact1 = await registry.register(
            statement="API keys should be kept secret",
            vertical="software",
        )

        # Mock vector store returns similar match
        mock_vector_store.data[fact1.id] = {
            "id": fact1.id,
            "embedding": mock_embedding_service.default_embedding,
            "content": fact1.statement,
            "metadata": {"vertical": "software"},
            "namespace": "software",
        }

        # Register similar fact - should update existing
        fact2 = await registry.register(
            statement="API keys should always be kept secret",
            vertical="software",
        )

        # Should return the existing fact (updated)
        assert fact2.id == fact1.id
        assert fact2.verification_count >= 1

    @pytest.mark.asyncio
    async def test_register_skip_duplicate_check(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedding_service: MockEmbeddingService,
    ):
        """Test skipping duplicate check on registration."""
        registry = FactRegistry(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )
        await registry.initialize()

        # Register first fact
        fact1 = await registry.register(
            statement="First statement",
            vertical="software",
        )

        # Register with duplicate check disabled
        fact2 = await registry.register(
            statement="Second statement",
            vertical="software",
            check_duplicates=False,
        )

        # Should create new fact even if similar
        assert fact1.id != fact2.id

    @pytest.mark.asyncio
    async def test_register_without_vector_store(self, fact_registry_no_vector: FactRegistry):
        """Test registration works without vector store."""
        await fact_registry_no_vector.initialize()

        fact = await fact_registry_no_vector.register(
            statement="Fact without vector store",
            vertical="general",
        )

        assert fact.id.startswith("fact_")
        assert fact.statement == "Fact without vector store"
        assert fact_registry_no_vector._facts[fact.id] == fact


# ============================================================================
# FactRegistry.query() Tests
# ============================================================================


class TestFactRegistryQuery:
    """Tests for FactRegistry.query() method."""

    @pytest.mark.asyncio
    async def test_query_basic(self, fact_registry_no_vector: FactRegistry):
        """Test basic query with in-memory search."""
        await fact_registry_no_vector.initialize()

        # Register some facts (no vector store, so uses memory)
        await fact_registry_no_vector.register(
            statement="Python is a programming language",
            vertical="software",
        )
        await fact_registry_no_vector.register(
            statement="JavaScript is also a programming language",
            vertical="software",
        )

        results = await fact_registry_no_vector.query("programming language")

        # In-memory fallback matches on keyword overlap
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_query_with_vertical_filter(self, fact_registry: FactRegistry):
        """Test query with vertical filtering."""
        await fact_registry.initialize()

        # Register facts in different verticals
        await fact_registry.register(
            statement="Python programming best practices",
            vertical="software",
        )
        await fact_registry.register(
            statement="GDPR compliance requirements",
            vertical="legal",
        )

        results = await fact_registry.query(
            "practices requirements",
            verticals=["software"],
        )

        for fact in results:
            assert fact.vertical == "software"

    @pytest.mark.asyncio
    async def test_query_with_workspace_filter(self, fact_registry: FactRegistry):
        """Test query with workspace filtering."""
        await fact_registry.initialize()

        await fact_registry.register(
            statement="Workspace A security policy",
            workspace_id="ws_a",
        )
        await fact_registry.register(
            statement="Workspace B security policy",
            workspace_id="ws_b",
        )

        results = await fact_registry.query(
            "security policy",
            workspace_id="ws_a",
        )

        for fact in results:
            assert fact.workspace_id == "ws_a"

    @pytest.mark.asyncio
    async def test_query_with_min_confidence(self, fact_registry: FactRegistry):
        """Test query filters by minimum confidence."""
        await fact_registry.initialize()

        await fact_registry.register(
            statement="High confidence fact",
            confidence=0.9,
        )
        await fact_registry.register(
            statement="Low confidence fact",
            confidence=0.2,
        )

        results = await fact_registry.query(
            "confidence fact",
            min_confidence=0.5,
        )

        for fact in results:
            assert fact.current_confidence >= 0.5

    @pytest.mark.asyncio
    async def test_query_excludes_stale_by_default(self, fact_registry: FactRegistry):
        """Test query excludes stale facts by default."""
        await fact_registry.initialize()

        # Register a fresh fact
        await fact_registry.register(
            statement="Fresh security update",
        )

        # Manually add a stale fact
        stale_fact = RegisteredFact(
            id="fact_stale",
            statement="Stale security update",
            base_confidence=0.6,
            verification_date=datetime.now() - timedelta(days=100),
            decay_rate=0.02,  # Will trigger needs_reverification
        )
        fact_registry._facts[stale_fact.id] = stale_fact

        results = await fact_registry.query("security update")

        # Should not include stale fact
        stale_ids = [f.id for f in results if f.id == "fact_stale"]
        assert len(stale_ids) == 0

    @pytest.mark.asyncio
    async def test_query_includes_stale_when_requested(self, fact_registry_no_vector: FactRegistry):
        """Test query includes stale facts when requested."""
        await fact_registry_no_vector.initialize()

        # Add stale fact directly to in-memory store
        stale_fact = RegisteredFact(
            id="fact_stale",
            statement="Stale security update",
            base_confidence=0.6,
            verification_date=datetime.now() - timedelta(days=100),
            decay_rate=0.02,
        )
        fact_registry_no_vector._facts[stale_fact.id] = stale_fact

        results = await fact_registry_no_vector.query(
            "security update",
            include_stale=True,
            min_confidence=0.0,  # Must lower since stale facts have low confidence
        )

        stale_ids = [f.id for f in results if f.id == "fact_stale"]
        assert len(stale_ids) == 1

    @pytest.mark.asyncio
    async def test_query_excludes_superseded_by_default(self, fact_registry: FactRegistry):
        """Test query excludes superseded facts by default."""
        await fact_registry.initialize()

        # Add superseded fact
        superseded = RegisteredFact(
            id="fact_old",
            statement="Old security policy",
            superseded_at=datetime.now(),
            superseded_by="fact_new",
        )
        fact_registry._facts[superseded.id] = superseded

        results = await fact_registry.query("security policy")

        superseded_ids = [f.id for f in results if f.id == "fact_old"]
        assert len(superseded_ids) == 0

    @pytest.mark.asyncio
    async def test_query_includes_superseded_when_requested(
        self, fact_registry_no_vector: FactRegistry
    ):
        """Test query includes superseded facts when requested."""
        await fact_registry_no_vector.initialize()

        # Add superseded fact directly to in-memory store
        superseded = RegisteredFact(
            id="fact_old",
            statement="Old security policy",
            base_confidence=0.8,
            superseded_at=datetime.now(),
            superseded_by="fact_new",
        )
        fact_registry_no_vector._facts[superseded.id] = superseded

        results = await fact_registry_no_vector.query(
            "security policy",
            include_superseded=True,
        )

        superseded_ids = [f.id for f in results if f.id == "fact_old"]
        assert len(superseded_ids) == 1

    @pytest.mark.asyncio
    async def test_query_memory_fallback(self, fact_registry_no_vector: FactRegistry):
        """Test query falls back to in-memory search."""
        await fact_registry_no_vector.initialize()

        await fact_registry_no_vector.register(
            statement="Python is great for automation",
        )
        await fact_registry_no_vector.register(
            statement="JavaScript is great for web development",
        )

        results = await fact_registry_no_vector.query("Python automation")

        # Should find the Python fact
        statements = [f.statement for f in results]
        assert any("Python" in s for s in statements)


# ============================================================================
# FactRegistry.get_fact() Tests
# ============================================================================


class TestFactRegistryGetFact:
    """Tests for FactRegistry.get_fact() method."""

    @pytest.mark.asyncio
    async def test_get_fact_exists(self, fact_registry: FactRegistry):
        """Test getting an existing fact."""
        await fact_registry.initialize()

        registered = await fact_registry.register(
            statement="Fact to retrieve",
        )

        fact = await fact_registry.get_fact(registered.id)

        assert fact is not None
        assert fact.id == registered.id
        assert fact.statement == "Fact to retrieve"

    @pytest.mark.asyncio
    async def test_get_fact_not_found(self, fact_registry: FactRegistry):
        """Test getting a non-existent fact."""
        await fact_registry.initialize()

        fact = await fact_registry.get_fact("nonexistent_id")

        assert fact is None


# ============================================================================
# FactRegistry.refresh_fact() Tests
# ============================================================================


class TestFactRegistryRefreshFact:
    """Tests for FactRegistry.refresh_fact() method."""

    @pytest.mark.asyncio
    async def test_refresh_fact_success(self, fact_registry: FactRegistry):
        """Test refreshing an existing fact."""
        await fact_registry.initialize()

        # Register a fact
        fact = await fact_registry.register(
            statement="Fact to refresh",
            confidence=0.7,
        )
        old_verification = fact.verification_date
        old_count = fact.verification_count

        # Wait a tiny bit
        await asyncio.sleep(0.01)

        result = await fact_registry.refresh_fact(fact.id)

        assert result is True
        assert fact.verification_date > old_verification
        assert fact.verification_count == old_count + 1

    @pytest.mark.asyncio
    async def test_refresh_fact_with_new_confidence(self, fact_registry: FactRegistry):
        """Test refreshing with new confidence."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Fact to update confidence",
            confidence=0.5,
        )

        await fact_registry.refresh_fact(fact.id, new_confidence=0.9)

        assert fact.base_confidence == 0.9

    @pytest.mark.asyncio
    async def test_refresh_fact_not_found(self, fact_registry: FactRegistry):
        """Test refreshing a non-existent fact."""
        await fact_registry.initialize()

        result = await fact_registry.refresh_fact("nonexistent_id")

        assert result is False


# ============================================================================
# FactRegistry.add_contradiction() Tests
# ============================================================================


class TestFactRegistryAddContradiction:
    """Tests for FactRegistry.add_contradiction() method."""

    @pytest.mark.asyncio
    async def test_add_contradiction_success(self, fact_registry: FactRegistry):
        """Test adding contradiction between facts."""
        await fact_registry.initialize()

        fact1 = await fact_registry.register(statement="Fact A")
        fact2 = await fact_registry.register(statement="Fact B (contradicts A)")

        result = await fact_registry.add_contradiction(fact1.id, fact2.id)

        assert result is True
        assert fact2.id in fact1.contradicts
        assert fact1.id in fact2.contradicts  # Bidirectional

    @pytest.mark.asyncio
    async def test_add_contradiction_updates_count(self, fact_registry: FactRegistry):
        """Test adding contradiction updates contradiction count."""
        await fact_registry.initialize()

        fact1 = await fact_registry.register(statement="Fact A")
        fact2 = await fact_registry.register(statement="Fact B")

        await fact_registry.add_contradiction(fact1.id, fact2.id)

        assert fact1.contradiction_count == 1
        assert fact2.contradiction_count == 1

    @pytest.mark.asyncio
    async def test_add_contradiction_fact_not_found(self, fact_registry: FactRegistry):
        """Test adding contradiction with non-existent fact."""
        await fact_registry.initialize()

        result = await fact_registry.add_contradiction("nonexistent", "also_nonexistent")

        assert result is False


# ============================================================================
# FactRegistry.supersede_fact() Tests
# ============================================================================


class TestFactRegistrySupersedeFact:
    """Tests for FactRegistry.supersede_fact() method."""

    @pytest.mark.asyncio
    async def test_supersede_fact_success(self, fact_registry: FactRegistry):
        """Test superseding a fact."""
        await fact_registry.initialize()

        old_fact = await fact_registry.register(statement="Old information")
        new_fact = await fact_registry.register(statement="Updated information")

        result = await fact_registry.supersede_fact(old_fact.id, new_fact.id)

        assert result is True
        assert old_fact.is_superseded
        assert old_fact.superseded_by == new_fact.id
        assert old_fact.id in new_fact.derived_from

    @pytest.mark.asyncio
    async def test_supersede_fact_not_found(self, fact_registry: FactRegistry):
        """Test superseding a non-existent fact."""
        await fact_registry.initialize()

        new_fact = await fact_registry.register(statement="New fact")

        result = await fact_registry.supersede_fact("nonexistent", new_fact.id)

        assert result is False


# ============================================================================
# FactRegistry.get_stale_facts() Tests
# ============================================================================


class TestFactRegistryGetStaleFacts:
    """Tests for FactRegistry.get_stale_facts() method."""

    @pytest.mark.asyncio
    async def test_get_stale_facts_returns_stale(self, fact_registry: FactRegistry):
        """Test getting stale facts."""
        await fact_registry.initialize()

        # Add a fresh fact
        await fact_registry.register(statement="Fresh fact")

        # Add stale facts manually
        stale1 = RegisteredFact(
            id="stale_1",
            statement="Stale fact 1",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=50),
            decay_rate=0.02,
        )
        stale2 = RegisteredFact(
            id="stale_2",
            statement="Stale fact 2",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=100),
            decay_rate=0.02,
        )
        fact_registry._facts[stale1.id] = stale1
        fact_registry._facts[stale2.id] = stale2

        stale_facts = await fact_registry.get_stale_facts()

        assert len(stale_facts) >= 2
        stale_ids = [f.id for f in stale_facts]
        assert "stale_1" in stale_ids
        assert "stale_2" in stale_ids

    @pytest.mark.asyncio
    async def test_get_stale_facts_sorted_by_staleness(self, fact_registry: FactRegistry):
        """Test stale facts are sorted by staleness (most stale first)."""
        await fact_registry.initialize()

        # Add stale facts with different staleness
        stale1 = RegisteredFact(
            id="stale_less",
            statement="Less stale",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=30),
            decay_rate=0.05,
        )
        stale2 = RegisteredFact(
            id="stale_more",
            statement="More stale",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=60),
            decay_rate=0.05,
        )
        fact_registry._facts[stale1.id] = stale1
        fact_registry._facts[stale2.id] = stale2

        stale_facts = await fact_registry.get_stale_facts()

        # Most stale (60 days) should come first
        assert stale_facts[0].id == "stale_more"

    @pytest.mark.asyncio
    async def test_get_stale_facts_filters_by_workspace(self, fact_registry: FactRegistry):
        """Test getting stale facts filtered by workspace."""
        await fact_registry.initialize()

        stale_ws_a = RegisteredFact(
            id="stale_ws_a",
            statement="Stale in workspace A",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=50),
            decay_rate=0.02,
            workspace_id="ws_a",
        )
        stale_ws_b = RegisteredFact(
            id="stale_ws_b",
            statement="Stale in workspace B",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=50),
            decay_rate=0.02,
            workspace_id="ws_b",
        )
        fact_registry._facts[stale_ws_a.id] = stale_ws_a
        fact_registry._facts[stale_ws_b.id] = stale_ws_b

        stale_facts = await fact_registry.get_stale_facts(workspace_id="ws_a")

        assert len(stale_facts) == 1
        assert stale_facts[0].workspace_id == "ws_a"

    @pytest.mark.asyncio
    async def test_get_stale_facts_excludes_superseded(self, fact_registry: FactRegistry):
        """Test superseded facts are not included in stale list."""
        await fact_registry.initialize()

        stale_superseded = RegisteredFact(
            id="stale_superseded",
            statement="Stale but superseded",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=50),
            decay_rate=0.02,
            superseded_at=datetime.now(),
            superseded_by="new_fact",
        )
        fact_registry._facts[stale_superseded.id] = stale_superseded

        stale_facts = await fact_registry.get_stale_facts()

        stale_ids = [f.id for f in stale_facts]
        assert "stale_superseded" not in stale_ids

    @pytest.mark.asyncio
    async def test_get_stale_facts_respects_limit(self, fact_registry: FactRegistry):
        """Test stale facts respects limit parameter."""
        await fact_registry.initialize()

        # Add many stale facts
        for i in range(10):
            stale = RegisteredFact(
                id=f"stale_{i}",
                statement=f"Stale fact {i}",
                base_confidence=0.8,
                verification_date=datetime.now() - timedelta(days=50 + i),
                decay_rate=0.02,
            )
            fact_registry._facts[stale.id] = stale

        stale_facts = await fact_registry.get_stale_facts(limit=3)

        assert len(stale_facts) == 3


# ============================================================================
# FactRegistry.get_stats() Tests
# ============================================================================


class TestFactRegistryGetStats:
    """Tests for FactRegistry.get_stats() method."""

    @pytest.mark.asyncio
    async def test_get_stats_empty_registry(self, fact_registry: FactRegistry):
        """Test stats for empty registry."""
        await fact_registry.initialize()

        stats = await fact_registry.get_stats()

        assert stats["total_facts"] == 0
        assert stats["by_vertical"] == {}
        assert stats["stale_count"] == 0
        assert stats["superseded_count"] == 0
        assert stats["average_confidence"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_facts(self, fact_registry_no_vector: FactRegistry):
        """Test stats with registered facts."""
        await fact_registry_no_vector.initialize()

        # Use registry without vector store to avoid duplicate detection
        await fact_registry_no_vector.register(
            statement="Software fact 1 about Python",
            vertical="software",
            confidence=0.9,
        )
        await fact_registry_no_vector.register(
            statement="Software fact 2 about JavaScript",
            vertical="software",
            confidence=0.8,
        )
        await fact_registry_no_vector.register(
            statement="Legal fact about contracts",
            vertical="legal",
            confidence=0.7,
        )

        stats = await fact_registry_no_vector.get_stats()

        assert stats["total_facts"] == 3
        assert stats["by_vertical"]["software"] == 2
        assert stats["by_vertical"]["legal"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_counts_stale(self, fact_registry: FactRegistry):
        """Test stats count stale facts."""
        await fact_registry.initialize()

        # Add fresh fact
        await fact_registry.register(statement="Fresh fact")

        # Add stale fact
        stale = RegisteredFact(
            id="stale_stat",
            statement="Stale for stats",
            base_confidence=0.8,
            verification_date=datetime.now() - timedelta(days=50),
            decay_rate=0.02,
        )
        fact_registry._facts[stale.id] = stale

        stats = await fact_registry.get_stats()

        assert stats["stale_count"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_counts_superseded(self, fact_registry: FactRegistry):
        """Test stats count superseded facts."""
        await fact_registry.initialize()

        await fact_registry.register(statement="Active fact")

        superseded = RegisteredFact(
            id="superseded_stat",
            statement="Superseded for stats",
            superseded_at=datetime.now(),
            superseded_by="new_fact",
        )
        fact_registry._facts[superseded.id] = superseded

        stats = await fact_registry.get_stats()

        assert stats["superseded_count"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_calculates_average_confidence(self, fact_registry: FactRegistry):
        """Test stats calculate average confidence."""
        await fact_registry.initialize()

        # Add facts with known confidences
        fact1 = RegisteredFact(
            id="conf_1",
            statement="Fact 1",
            base_confidence=0.8,
            verification_date=datetime.now(),
        )
        fact2 = RegisteredFact(
            id="conf_2",
            statement="Fact 2",
            base_confidence=0.6,
            verification_date=datetime.now(),
        )
        fact_registry._facts[fact1.id] = fact1
        fact_registry._facts[fact2.id] = fact2

        stats = await fact_registry.get_stats()

        # Average of 0.8 and 0.6 = 0.7
        assert abs(stats["average_confidence"] - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_get_stats_excludes_superseded_from_average(self, fact_registry: FactRegistry):
        """Test superseded facts are excluded from average confidence."""
        await fact_registry.initialize()

        active = RegisteredFact(
            id="active",
            statement="Active fact",
            base_confidence=0.8,
            verification_date=datetime.now(),
        )
        superseded = RegisteredFact(
            id="superseded",
            statement="Superseded fact",
            base_confidence=0.2,  # Low confidence, would skew average
            superseded_at=datetime.now(),
            superseded_by="new",
        )
        fact_registry._facts[active.id] = active
        fact_registry._facts[superseded.id] = superseded

        stats = await fact_registry.get_stats()

        # Average should only include active fact
        assert abs(stats["average_confidence"] - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_get_stats_filters_by_workspace(self, fact_registry: FactRegistry):
        """Test stats can be filtered by workspace."""
        await fact_registry.initialize()

        fact_ws_a = RegisteredFact(
            id="ws_a_fact",
            statement="Workspace A fact",
            workspace_id="ws_a",
        )
        fact_ws_b = RegisteredFact(
            id="ws_b_fact",
            statement="Workspace B fact",
            workspace_id="ws_b",
        )
        fact_registry._facts[fact_ws_a.id] = fact_ws_a
        fact_registry._facts[fact_ws_b.id] = fact_ws_b

        stats = await fact_registry.get_stats(workspace_id="ws_a")

        assert stats["total_facts"] == 1


# ============================================================================
# Memory Fallback Tests
# ============================================================================


class TestFactRegistryMemoryFallback:
    """Tests for memory fallback when vector store unavailable."""

    @pytest.mark.asyncio
    async def test_query_without_vector_store(self, fact_registry_no_vector: FactRegistry):
        """Test query works without vector store."""
        await fact_registry_no_vector.initialize()

        await fact_registry_no_vector.register(
            statement="Memory only fact about Python",
        )

        results = await fact_registry_no_vector.query("Python")

        assert len(results) >= 1
        assert any("Python" in f.statement for f in results)

    @pytest.mark.asyncio
    async def test_query_falls_back_on_embedding_failure(self, mock_vector_store: MockVectorStore):
        """Test query falls back to memory when embedding fails."""
        failing_embedding = MockEmbeddingService()
        failing_embedding.should_fail = True

        registry = FactRegistry(
            vector_store=mock_vector_store,
            embedding_service=failing_embedding,
        )
        await registry.initialize()

        # Add fact directly to memory
        fact = RegisteredFact(
            id="memory_fact",
            statement="Fact in memory only about databases",
        )
        registry._facts[fact.id] = fact

        # Query should still work via memory fallback
        results = await registry.query("databases")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_register_without_embedding_service(self, mock_vector_store: MockVectorStore):
        """Test registration works without embedding service."""
        registry = FactRegistry(vector_store=mock_vector_store)
        await registry.initialize()

        fact = await registry.register(
            statement="Fact without embeddings",
        )

        assert fact.id in registry._facts
        assert fact.embedding is None
        # Vector store upsert should not be called without embedding
        assert len(mock_vector_store.upsert_calls) == 0


# ============================================================================
# Vertical-Specific Decay Rate Tests
# ============================================================================


class TestVerticalDecayRates:
    """Tests for vertical-specific decay rates."""

    @pytest.mark.asyncio
    async def test_software_vulnerability_decay_rate(self, fact_registry: FactRegistry):
        """Test software vulnerability has fast decay rate."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="CVE vulnerability",
            vertical="software",
            category="vulnerability",
        )

        assert fact.decay_rate == 0.05

    @pytest.mark.asyncio
    async def test_software_secret_decay_rate(self, fact_registry: FactRegistry):
        """Test software secret has very fast decay rate."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="API key rotation",
            vertical="software",
            category="secret",
        )

        assert fact.decay_rate == 0.1

    @pytest.mark.asyncio
    async def test_software_best_practice_decay_rate(self, fact_registry: FactRegistry):
        """Test software best practice has slow decay rate."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Code review is important",
            vertical="software",
            category="best_practice",
        )

        assert fact.decay_rate == 0.01

    @pytest.mark.asyncio
    async def test_legal_clause_decay_rate(self, fact_registry: FactRegistry):
        """Test legal clause has very slow decay rate."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Termination clause terms",
            vertical="legal",
            category="clause",
        )

        # LegalKnowledge uses 0.005 for clauses
        assert fact.decay_rate == 0.005

    @pytest.mark.asyncio
    async def test_healthcare_clinical_guideline_decay_rate(self, fact_registry: FactRegistry):
        """Test healthcare clinical guideline decay rate."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Treatment protocol",
            vertical="healthcare",
            category="clinical_guideline",
        )

        assert fact.decay_rate == 0.02

    @pytest.mark.asyncio
    async def test_unknown_vertical_default_decay_rate(self, fact_registry: FactRegistry):
        """Test unknown vertical uses default decay rate."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Unknown domain fact",
            vertical="unknown_vertical",
            category="unknown_category",
        )

        assert fact.decay_rate == 0.02  # Default rate


# ============================================================================
# Edge Cases
# ============================================================================


class TestFactRegistryEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_statement(self, fact_registry: FactRegistry):
        """Test registering fact with empty statement."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="",
        )

        assert fact.statement == ""

    @pytest.mark.asyncio
    async def test_unicode_statement(self, fact_registry: FactRegistry):
        """Test registering fact with unicode characters."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Unicode test: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud83d\ude00",
        )

        assert "\u4e2d\u6587" in fact.statement

    @pytest.mark.asyncio
    async def test_very_long_statement(self, fact_registry: FactRegistry):
        """Test registering fact with very long statement."""
        await fact_registry.initialize()

        long_statement = "A" * 10000
        fact = await fact_registry.register(
            statement=long_statement,
        )

        assert len(fact.statement) == 10000

    @pytest.mark.asyncio
    async def test_zero_confidence(self, fact_registry: FactRegistry):
        """Test registering fact with zero confidence."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Zero confidence fact",
            confidence=0.0,
        )

        assert fact.base_confidence == 0.0
        assert fact.current_confidence == 0.0

    @pytest.mark.asyncio
    async def test_confidence_exactly_one(self, fact_registry: FactRegistry):
        """Test registering fact with confidence exactly 1.0."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Perfect confidence fact",
            confidence=1.0,
        )

        assert fact.base_confidence == 1.0

    @pytest.mark.asyncio
    async def test_many_topics(self, fact_registry: FactRegistry):
        """Test registering fact with many topics."""
        await fact_registry.initialize()

        topics = [f"topic_{i}" for i in range(100)]
        fact = await fact_registry.register(
            statement="Multi-topic fact",
            topics=topics,
        )

        assert len(fact.topics) == 100

    @pytest.mark.asyncio
    async def test_special_characters_in_ids(self, fact_registry: FactRegistry):
        """Test handling facts with special workspace IDs."""
        await fact_registry.initialize()

        fact = await fact_registry.register(
            statement="Fact with special workspace",
            workspace_id="ws:special/chars#test",
        )

        assert fact.workspace_id == "ws:special/chars#test"
