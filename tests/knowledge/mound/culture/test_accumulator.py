"""
Comprehensive tests for the Culture Accumulator module.

Tests CultureAccumulator, OrganizationCultureManager, CultureDocument,
CultureDocumentCategory, OrganizationCulture, and DebateObservation.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch
import uuid

from aragora.knowledge.mound.culture.accumulator import (
    CultureAccumulator,
    CultureDocument,
    CultureDocumentCategory,
    DebateObservation,
    OrganizationCulture,
    OrganizationCultureManager,
)
from aragora.knowledge.mound.types import (
    CulturePattern,
    CulturePatternType,
    CultureProfile,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    mound = MagicMock()
    mound._vector_store = None
    mound.event_emitter = None
    return mound


@pytest.fixture
def mock_mound_with_event_emitter():
    """Create a mock KnowledgeMound with event emitter."""
    mound = MagicMock()
    mound._vector_store = None
    mound.event_emitter = MagicMock()
    mound.event_emitter.emit_sync = MagicMock()
    return mound


@pytest.fixture
def mock_mound_with_vector_store():
    """Create a mock KnowledgeMound with vector store."""
    mound = MagicMock()
    mound._vector_store = AsyncMock()
    mound._vector_store.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mound.event_emitter = None
    return mound


@pytest.fixture
def culture_accumulator(mock_mound):
    """Create a CultureAccumulator instance."""
    return CultureAccumulator(mound=mock_mound, min_observations_for_pattern=3)


@pytest.fixture
def culture_accumulator_with_emitter(mock_mound_with_event_emitter):
    """Create a CultureAccumulator instance with event emitter."""
    return CultureAccumulator(mound=mock_mound_with_event_emitter, min_observations_for_pattern=3)


@pytest.fixture
def org_culture_manager(mock_mound, culture_accumulator):
    """Create an OrganizationCultureManager instance."""
    return OrganizationCultureManager(
        mound=mock_mound,
        culture_accumulator=culture_accumulator,
    )


@pytest.fixture
def org_culture_manager_with_vector(mock_mound_with_vector_store, culture_accumulator):
    """Create an OrganizationCultureManager with vector store."""
    return OrganizationCultureManager(
        mound=mock_mound_with_vector_store,
        culture_accumulator=culture_accumulator,
    )


@pytest.fixture
def org_culture_manager_no_accumulator(mock_mound):
    """Create an OrganizationCultureManager without accumulator."""
    return OrganizationCultureManager(
        mound=mock_mound,
        culture_accumulator=None,
    )


@pytest.fixture
def sample_debate_result():
    """Create a sample debate result object."""

    @dataclass
    class Proposal:
        agent_type: str
        content: str

    @dataclass
    class Critique:
        type: str
        content: str

    @dataclass
    class DebateResult:
        debate_id: str
        task: str
        proposals: list
        winner: Optional[str]
        consensus_reached: bool
        confidence: float
        rounds_used: int
        critiques: list

    return DebateResult(
        debate_id="debate_001",
        task="Review security architecture",
        proposals=[
            Proposal(agent_type="claude", content="Use OAuth2"),
            Proposal(agent_type="gpt4", content="Use JWT"),
        ],
        winner="claude",
        consensus_reached=True,
        confidence=0.85,
        rounds_used=3,
        critiques=[
            Critique(type="security", content="Consider token expiry"),
            Critique(type="scalability", content="Think about load balancing"),
        ],
    )


@pytest.fixture
def sample_debate_result_no_winner():
    """Create a debate result without a winner."""

    @dataclass
    class Proposal:
        agent_type: str
        content: str

    @dataclass
    class DebateResult:
        debate_id: str
        task: str
        proposals: list
        winner: Optional[str]
        consensus_reached: bool
        confidence: float
        rounds_used: int
        critiques: list

    return DebateResult(
        debate_id="debate_002",
        task="Discuss random topic with no domain",
        proposals=[
            Proposal(agent_type="claude", content="Option A"),
        ],
        winner=None,
        consensus_reached=False,
        confidence=0.3,
        rounds_used=5,
        critiques=[],
    )


@pytest.fixture
def sample_debate_result_high_confidence():
    """Create a debate result with very high confidence (unanimous)."""

    @dataclass
    class Proposal:
        agent_type: str
        content: str

    @dataclass
    class DebateResult:
        debate_id: str
        task: str
        proposals: list
        winner: Optional[str]
        consensus_reached: bool
        confidence: float
        rounds_used: int
        critiques: list

    return DebateResult(
        debate_id="debate_003",
        task="Review database schema",
        proposals=[
            Proposal(agent_type="gpt4", content="Normalize tables"),
        ],
        winner="gpt4",
        consensus_reached=True,
        confidence=0.95,
        rounds_used=1,
        critiques=[],
    )


# ============================================================================
# CultureDocumentCategory Tests
# ============================================================================


class TestCultureDocumentCategory:
    """Tests for CultureDocumentCategory enum."""

    def test_values_enum_value(self):
        """Test VALUES enum value."""
        assert CultureDocumentCategory.VALUES.value == "values"

    def test_practices_enum_value(self):
        """Test PRACTICES enum value."""
        assert CultureDocumentCategory.PRACTICES.value == "practices"

    def test_standards_enum_value(self):
        """Test STANDARDS enum value."""
        assert CultureDocumentCategory.STANDARDS.value == "standards"

    def test_policies_enum_value(self):
        """Test POLICIES enum value."""
        assert CultureDocumentCategory.POLICIES.value == "policies"

    def test_learnings_enum_value(self):
        """Test LEARNINGS enum value."""
        assert CultureDocumentCategory.LEARNINGS.value == "learnings"

    def test_category_is_string(self):
        """Test that category is a string enum."""
        assert isinstance(CultureDocumentCategory.VALUES, str)

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        expected = {"values", "practices", "standards", "policies", "learnings"}
        actual = {c.value for c in CultureDocumentCategory}
        assert actual == expected


# ============================================================================
# CultureDocument Tests
# ============================================================================


class TestCultureDocument:
    """Tests for CultureDocument dataclass."""

    def test_basic_creation(self):
        """Test creating a culture document."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Core Values",
            content="We believe in quality.",
            created_by="admin",
        )

        assert doc.id == "cd_001"
        assert doc.org_id == "org_001"
        assert doc.category == CultureDocumentCategory.VALUES
        assert doc.title == "Core Values"
        assert doc.content == "We believe in quality."
        assert doc.created_by == "admin"
        assert doc.version == 1
        assert doc.is_active is True

    def test_default_values(self):
        """Test default values are set correctly."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Test",
            content="Content",
        )

        assert doc.embeddings is None
        assert doc.created_by == ""
        assert doc.version == 1
        assert doc.supersedes is None
        assert doc.source_workspace_id is None
        assert doc.source_pattern_id is None
        assert doc.metadata == {}
        assert doc.is_active is True
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)

    def test_with_embeddings(self):
        """Test document with embeddings."""
        embeddings = [0.1, 0.2, 0.3, 0.4]
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Test",
            content="Content",
            embeddings=embeddings,
        )

        assert doc.embeddings == embeddings

    def test_with_metadata(self):
        """Test document with metadata."""
        metadata = {"source": "debate", "importance": "high"}
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.POLICIES,
            title="Test",
            content="Content",
            metadata=metadata,
        )

        assert doc.metadata == metadata
        assert doc.metadata["source"] == "debate"

    def test_with_supersedes(self):
        """Test document that supersedes another."""
        doc = CultureDocument(
            id="cd_002",
            org_id="org_001",
            category=CultureDocumentCategory.STANDARDS,
            title="Updated Standard",
            content="New content",
            version=2,
            supersedes="cd_001",
        )

        assert doc.version == 2
        assert doc.supersedes == "cd_001"

    def test_with_source_info(self):
        """Test document with source information."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.LEARNINGS,
            title="Learned Pattern",
            content="Content",
            source_workspace_id="ws_001",
            source_pattern_id="cp_001",
        )

        assert doc.source_workspace_id == "ws_001"
        assert doc.source_pattern_id == "cp_001"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Test",
            content="Content",
            created_by="user",
            metadata={"key": "value"},
        )

        d = doc.to_dict()

        assert d["id"] == "cd_001"
        assert d["org_id"] == "org_001"
        assert d["category"] == "practices"
        assert d["title"] == "Test"
        assert d["content"] == "Content"
        assert d["created_by"] == "user"
        assert d["version"] == 1
        assert d["supersedes"] is None
        assert d["source_workspace_id"] is None
        assert d["source_pattern_id"] is None
        assert d["metadata"]["key"] == "value"
        assert d["is_active"] is True
        assert "created_at" in d
        assert "updated_at" in d

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields populated."""
        doc = CultureDocument(
            id="cd_002",
            org_id="org_001",
            category=CultureDocumentCategory.LEARNINGS,
            title="Full Doc",
            content="All fields",
            embeddings=[0.1, 0.2],
            created_by="admin",
            version=3,
            supersedes="cd_001",
            source_workspace_id="ws_001",
            source_pattern_id="cp_001",
            metadata={"promoted": True},
            is_active=False,
        )

        d = doc.to_dict()

        assert d["version"] == 3
        assert d["supersedes"] == "cd_001"
        assert d["source_workspace_id"] == "ws_001"
        assert d["source_pattern_id"] == "cp_001"
        assert d["is_active"] is False

    def test_datetime_iso_format_in_to_dict(self):
        """Test that datetimes are in ISO format in to_dict."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Test",
            content="Content",
        )

        d = doc.to_dict()

        # Should be valid ISO format strings
        datetime.fromisoformat(d["created_at"])
        datetime.fromisoformat(d["updated_at"])


# ============================================================================
# OrganizationCulture Tests
# ============================================================================


class TestOrganizationCulture:
    """Tests for OrganizationCulture dataclass."""

    def test_basic_creation(self):
        """Test creating an organization culture."""
        culture = OrganizationCulture(
            org_id="org_001",
            documents=[],
            aggregated_patterns={},
            workspace_profiles={},
        )

        assert culture.org_id == "org_001"
        assert culture.documents == []
        assert culture.aggregated_patterns == {}
        assert culture.workspace_profiles == {}
        assert culture.total_observations == 0
        assert culture.workspace_count == 0
        assert culture.dominant_traits == {}
        assert isinstance(culture.generated_at, datetime)

    def test_with_documents(self):
        """Test organization culture with documents."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Values",
            content="Content",
        )

        culture = OrganizationCulture(
            org_id="org_001",
            documents=[doc],
            aggregated_patterns={},
            workspace_profiles={},
        )

        assert len(culture.documents) == 1
        assert culture.documents[0].id == "cd_001"

    def test_with_workspace_profiles(self):
        """Test organization culture with workspace profiles."""
        profile = CultureProfile(
            workspace_id="ws_001",
            patterns={},
            generated_at=datetime.now(),
            total_observations=10,
            dominant_traits={"top_agents": ["claude"]},
        )

        culture = OrganizationCulture(
            org_id="org_001",
            documents=[],
            aggregated_patterns={},
            workspace_profiles={"ws_001": profile},
            workspace_count=1,
            total_observations=10,
        )

        assert culture.workspace_count == 1
        assert "ws_001" in culture.workspace_profiles

    def test_with_dominant_traits(self):
        """Test organization culture with dominant traits."""
        traits = {
            "top_agents": ["claude", "gpt4"],
            "expertise_areas": ["security", "api"],
            "avg_consensus_rounds": 3.5,
            "consensus_rate": 0.85,
        }

        culture = OrganizationCulture(
            org_id="org_001",
            documents=[],
            aggregated_patterns={},
            workspace_profiles={},
            dominant_traits=traits,
        )

        assert culture.dominant_traits["top_agents"] == ["claude", "gpt4"]
        assert culture.dominant_traits["consensus_rate"] == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Values",
            content="Content",
        )

        culture = OrganizationCulture(
            org_id="org_001",
            documents=[doc],
            aggregated_patterns={},
            workspace_profiles={},
            workspace_count=2,
            total_observations=50,
            dominant_traits={"key": "value"},
        )

        d = culture.to_dict()

        assert d["org_id"] == "org_001"
        assert len(d["documents"]) == 1
        assert d["workspace_count"] == 2
        assert d["total_observations"] == 50
        assert d["dominant_traits"]["key"] == "value"
        assert "generated_at" in d

    def test_to_dict_datetime_format(self):
        """Test that generated_at is in ISO format."""
        culture = OrganizationCulture(
            org_id="org_001",
            documents=[],
            aggregated_patterns={},
            workspace_profiles={},
        )

        d = culture.to_dict()
        datetime.fromisoformat(d["generated_at"])


# ============================================================================
# DebateObservation Tests
# ============================================================================


class TestDebateObservation:
    """Tests for DebateObservation dataclass."""

    def test_basic_creation(self):
        """Test creating a debate observation."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Security review",
            participating_agents=["claude", "gpt4"],
            winning_agents=["claude"],
            rounds_to_consensus=3,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=["security", "scalability"],
            domain="security",
        )

        assert obs.debate_id == "debate_001"
        assert obs.topic == "Security review"
        assert obs.participating_agents == ["claude", "gpt4"]
        assert obs.winning_agents == ["claude"]
        assert obs.rounds_to_consensus == 3
        assert obs.consensus_reached is True
        assert obs.consensus_strength == "strong"
        assert obs.critique_patterns == ["security", "scalability"]
        assert obs.domain == "security"
        assert isinstance(obs.created_at, datetime)

    def test_with_no_consensus(self):
        """Test observation without consensus."""
        obs = DebateObservation(
            debate_id="debate_002",
            topic="Contentious topic",
            participating_agents=["claude", "gpt4", "gemini"],
            winning_agents=[],
            rounds_to_consensus=10,
            consensus_reached=False,
            consensus_strength="weak",
            critique_patterns=[],
            domain=None,
        )

        assert obs.consensus_reached is False
        assert obs.winning_agents == []
        assert obs.domain is None

    def test_with_no_domain(self):
        """Test observation with no inferred domain."""
        obs = DebateObservation(
            debate_id="debate_003",
            topic="Random discussion",
            participating_agents=["claude"],
            winning_agents=["claude"],
            rounds_to_consensus=1,
            consensus_reached=True,
            consensus_strength="unanimous",
            critique_patterns=[],
            domain=None,
        )

        assert obs.domain is None

    def test_default_created_at(self):
        """Test that created_at defaults to now."""
        before = datetime.now()
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Test",
            participating_agents=[],
            winning_agents=[],
            rounds_to_consensus=1,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=[],
            domain=None,
        )
        after = datetime.now()

        assert before <= obs.created_at <= after


# ============================================================================
# CultureAccumulator Tests - Initialization
# ============================================================================


class TestCultureAccumulatorInit:
    """Tests for CultureAccumulator initialization."""

    def test_basic_init(self, mock_mound):
        """Test basic initialization."""
        acc = CultureAccumulator(mound=mock_mound)

        assert acc._mound == mock_mound
        assert acc._min_observations == 3  # Default

    def test_custom_min_observations(self, mock_mound):
        """Test initialization with custom min observations."""
        acc = CultureAccumulator(mound=mock_mound, min_observations_for_pattern=5)

        assert acc._min_observations == 5

    def test_patterns_dict_initialized_empty(self, mock_mound):
        """Test that patterns dict is initialized empty."""
        acc = CultureAccumulator(mound=mock_mound)

        assert len(acc._patterns) == 0


# ============================================================================
# CultureAccumulator Tests - _infer_domain
# ============================================================================


class TestCultureAccumulatorInferDomain:
    """Tests for CultureAccumulator._infer_domain method."""

    def test_security_domain(self, culture_accumulator):
        """Test inferring security domain."""
        assert culture_accumulator._infer_domain("Review security architecture") == "security"
        assert culture_accumulator._infer_domain("Auth token implementation") == "security"
        assert culture_accumulator._infer_domain("Encrypt sensitive data") == "security"
        assert culture_accumulator._infer_domain("Vulnerability assessment") == "security"
        assert culture_accumulator._infer_domain("Attack vector analysis") == "security"

    def test_performance_domain(self, culture_accumulator):
        """Test inferring performance domain."""
        assert culture_accumulator._infer_domain("Performance testing") == "performance"
        assert culture_accumulator._infer_domain("Speed optimization") == "performance"
        assert culture_accumulator._infer_domain("Reduce latency") == "performance"
        assert culture_accumulator._infer_domain("Optimize algorithm") == "performance"
        assert culture_accumulator._infer_domain("Cache implementation") == "performance"

    def test_architecture_domain(self, culture_accumulator):
        """Test inferring architecture domain."""
        assert culture_accumulator._infer_domain("Architecture review") == "architecture"
        assert culture_accumulator._infer_domain("Design patterns") == "architecture"
        assert culture_accumulator._infer_domain("Module structure") == "architecture"

    def test_testing_domain(self, culture_accumulator):
        """Test inferring testing domain."""
        assert culture_accumulator._infer_domain("Write tests") == "testing"
        assert culture_accumulator._infer_domain("QA review") == "testing"
        assert culture_accumulator._infer_domain("Coverage report") == "testing"
        assert culture_accumulator._infer_domain("Add assertions") == "testing"
        assert culture_accumulator._infer_domain("Mock dependencies") == "testing"

    def test_api_domain(self, culture_accumulator):
        """Test inferring API domain."""
        assert culture_accumulator._infer_domain("API design") == "api"
        assert culture_accumulator._infer_domain("New endpoint") == "api"
        assert culture_accumulator._infer_domain("REST service") == "api"
        assert culture_accumulator._infer_domain("GraphQL schema") == "api"
        assert culture_accumulator._infer_domain("RPC implementation") == "api"

    def test_database_domain(self, culture_accumulator):
        """Test inferring database domain."""
        assert culture_accumulator._infer_domain("Database schema") == "database"
        assert culture_accumulator._infer_domain("SQL optimization") == "database"
        assert culture_accumulator._infer_domain("Query performance") == "database"
        assert culture_accumulator._infer_domain("Index creation") == "database"

    def test_frontend_domain(self, culture_accumulator):
        """Test inferring frontend domain."""
        assert culture_accumulator._infer_domain("UI improvements") == "frontend"
        assert culture_accumulator._infer_domain("Frontend refactor") == "frontend"
        assert culture_accumulator._infer_domain("React component") == "frontend"
        assert culture_accumulator._infer_domain("CSS styling") == "frontend"

    def test_legal_domain(self, culture_accumulator):
        """Test inferring legal domain."""
        assert culture_accumulator._infer_domain("Contract review") == "legal"
        assert culture_accumulator._infer_domain("Compliance check") == "legal"
        assert culture_accumulator._infer_domain("Regulation compliance") == "legal"
        assert culture_accumulator._infer_domain("Legal terms") == "legal"

    def test_financial_domain(self, culture_accumulator):
        """Test inferring financial domain."""
        assert culture_accumulator._infer_domain("Financial report") == "financial"
        assert culture_accumulator._infer_domain("Accounting review") == "financial"
        assert culture_accumulator._infer_domain("Audit preparation") == "financial"
        assert culture_accumulator._infer_domain("Budget planning") == "financial"
        assert culture_accumulator._infer_domain("Cost analysis") == "financial"

    def test_no_domain_match(self, culture_accumulator):
        """Test no domain match returns None."""
        assert culture_accumulator._infer_domain("Random topic") is None
        assert culture_accumulator._infer_domain("General discussion") is None
        assert culture_accumulator._infer_domain("") is None

    def test_case_insensitive(self, culture_accumulator):
        """Test domain inference is case insensitive."""
        assert culture_accumulator._infer_domain("SECURITY review") == "security"
        assert culture_accumulator._infer_domain("Security Review") == "security"
        assert culture_accumulator._infer_domain("sEcUrItY") == "security"


# ============================================================================
# CultureAccumulator Tests - _extract_observation
# ============================================================================


class TestCultureAccumulatorExtractObservation:
    """Tests for CultureAccumulator._extract_observation method."""

    def test_extract_basic_observation(self, culture_accumulator, sample_debate_result):
        """Test extracting basic observation from debate result."""
        obs = culture_accumulator._extract_observation(sample_debate_result)

        assert obs is not None
        assert obs.debate_id == "debate_001"
        assert obs.topic == "Review security architecture"
        assert "claude" in obs.participating_agents
        assert "gpt4" in obs.participating_agents
        assert "claude" in obs.winning_agents
        assert obs.consensus_reached is True
        assert obs.rounds_to_consensus == 3
        assert obs.consensus_strength == "strong"
        assert obs.domain == "security"

    def test_extract_critique_patterns(self, culture_accumulator, sample_debate_result):
        """Test that critique patterns are extracted."""
        obs = culture_accumulator._extract_observation(sample_debate_result)

        assert "security" in obs.critique_patterns
        assert "scalability" in obs.critique_patterns

    def test_consensus_strength_unanimous(self, culture_accumulator):
        """Test unanimous consensus strength."""

        @dataclass
        class DebateResult:
            debate_id: str = "test"
            task: str = "test"
            proposals: list = None
            winner: Optional[str] = None
            consensus_reached: bool = True
            confidence: float = 0.95
            rounds_used: int = 1
            critiques: list = None

        result = DebateResult(proposals=[], critiques=[])
        obs = culture_accumulator._extract_observation(result)

        assert obs.consensus_strength == "unanimous"

    def test_consensus_strength_strong(self, culture_accumulator):
        """Test strong consensus strength."""

        @dataclass
        class DebateResult:
            debate_id: str = "test"
            task: str = "test"
            proposals: list = None
            winner: Optional[str] = None
            consensus_reached: bool = True
            confidence: float = 0.75
            rounds_used: int = 1
            critiques: list = None

        result = DebateResult(proposals=[], critiques=[])
        obs = culture_accumulator._extract_observation(result)

        assert obs.consensus_strength == "strong"

    def test_consensus_strength_moderate(self, culture_accumulator):
        """Test moderate consensus strength."""

        @dataclass
        class DebateResult:
            debate_id: str = "test"
            task: str = "test"
            proposals: list = None
            winner: Optional[str] = None
            consensus_reached: bool = True
            confidence: float = 0.55
            rounds_used: int = 1
            critiques: list = None

        result = DebateResult(proposals=[], critiques=[])
        obs = culture_accumulator._extract_observation(result)

        assert obs.consensus_strength == "moderate"

    def test_consensus_strength_weak(self, culture_accumulator):
        """Test weak consensus strength."""

        @dataclass
        class DebateResult:
            debate_id: str = "test"
            task: str = "test"
            proposals: list = None
            winner: Optional[str] = None
            consensus_reached: bool = False
            confidence: float = 0.35
            rounds_used: int = 5
            critiques: list = None

        result = DebateResult(proposals=[], critiques=[])
        obs = culture_accumulator._extract_observation(result)

        assert obs.consensus_strength == "weak"

    def test_extract_with_no_winner(self, culture_accumulator, sample_debate_result_no_winner):
        """Test extraction when no winner."""
        obs = culture_accumulator._extract_observation(sample_debate_result_no_winner)

        assert obs.winning_agents == []

    def test_extract_limits_critiques(self, culture_accumulator):
        """Test that critique patterns are limited to 5."""

        @dataclass
        class Critique:
            type: str
            content: str

        @dataclass
        class DebateResult:
            debate_id: str = "test"
            task: str = "test"
            proposals: list = None
            winner: Optional[str] = None
            consensus_reached: bool = True
            confidence: float = 0.8
            rounds_used: int = 1
            critiques: list = None

        critiques = [Critique(type=f"type_{i}", content=f"content_{i}") for i in range(10)]
        result = DebateResult(proposals=[], critiques=critiques)
        obs = culture_accumulator._extract_observation(result)

        assert len(obs.critique_patterns) <= 5

    def test_extract_handles_missing_attributes(self, culture_accumulator):
        """Test extraction handles missing attributes gracefully."""

        class MinimalResult:
            pass

        result = MinimalResult()
        obs = culture_accumulator._extract_observation(result)

        assert obs is not None
        assert obs.participating_agents == []
        assert obs.winning_agents == []
        assert obs.consensus_reached is False

    def test_extract_uses_topic_fallback(self, culture_accumulator):
        """Test extraction uses 'topic' attribute if 'task' missing."""

        @dataclass
        class DebateResult:
            debate_id: str = "test"
            topic: str = "Topic from attribute"
            proposals: list = None
            winner: Optional[str] = None
            consensus_reached: bool = True
            confidence: float = 0.8
            rounds_used: int = 1
            critiques: list = None

        result = DebateResult(proposals=[], critiques=[])
        obs = culture_accumulator._extract_observation(result)

        assert obs.topic == "Topic from attribute"

    def test_extract_returns_none_on_exception(self, culture_accumulator):
        """Test extraction returns None when exception occurs."""

        class BrokenResult:
            @property
            def proposals(self):
                raise RuntimeError("Broken")

        result = BrokenResult()
        obs = culture_accumulator._extract_observation(result)

        # Should handle the exception and still return an observation
        # The actual behavior depends on exception handling in the code
        # It catches exceptions and returns None
        assert obs is None


# ============================================================================
# CultureAccumulator Tests - observe_debate
# ============================================================================


class TestCultureAccumulatorObserveDebate:
    """Tests for CultureAccumulator.observe_debate method."""

    @pytest.mark.asyncio
    async def test_observe_debate_returns_patterns(self, culture_accumulator, sample_debate_result):
        """Test that observing debate returns patterns."""
        patterns = await culture_accumulator.observe_debate(
            sample_debate_result,
            workspace_id="ws_001",
        )

        assert isinstance(patterns, list)
        assert len(patterns) >= 1

    @pytest.mark.asyncio
    async def test_observe_debate_creates_agent_preference_pattern(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that agent preference pattern is created."""
        patterns = await culture_accumulator.observe_debate(
            sample_debate_result,
            workspace_id="ws_001",
        )

        agent_patterns = [
            p for p in patterns if p.pattern_type == CulturePatternType.AGENT_PREFERENCES
        ]
        assert len(agent_patterns) >= 1

    @pytest.mark.asyncio
    async def test_observe_debate_creates_decision_style_pattern(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that decision style pattern is created."""
        patterns = await culture_accumulator.observe_debate(
            sample_debate_result,
            workspace_id="ws_001",
        )

        decision_patterns = [
            p for p in patterns if p.pattern_type == CulturePatternType.DECISION_STYLE
        ]
        assert len(decision_patterns) >= 1

    @pytest.mark.asyncio
    async def test_observe_debate_creates_debate_dynamics_pattern(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that debate dynamics pattern is created from critiques."""
        patterns = await culture_accumulator.observe_debate(
            sample_debate_result,
            workspace_id="ws_001",
        )

        dynamics_patterns = [
            p for p in patterns if p.pattern_type == CulturePatternType.DEBATE_DYNAMICS
        ]
        assert len(dynamics_patterns) >= 1

    @pytest.mark.asyncio
    async def test_observe_debate_creates_domain_expertise_pattern(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that domain expertise pattern is created."""
        patterns = await culture_accumulator.observe_debate(
            sample_debate_result,
            workspace_id="ws_001",
        )

        domain_patterns = [
            p for p in patterns if p.pattern_type == CulturePatternType.DOMAIN_EXPERTISE
        ]
        assert len(domain_patterns) >= 1

    @pytest.mark.asyncio
    async def test_observe_debate_updates_existing_patterns(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that observing multiple debates updates existing patterns."""
        # First observation
        await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        # Get initial pattern
        patterns_before = culture_accumulator.get_patterns("ws_001")
        initial_count = patterns_before[0].observation_count if patterns_before else 0

        # Second observation
        sample_debate_result.debate_id = "debate_002"
        await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        # Check pattern was updated
        patterns_after = culture_accumulator.get_patterns("ws_001")
        # At least one pattern should have increased observation count
        updated_patterns = [p for p in patterns_after if p.observation_count > 1]
        assert len(updated_patterns) >= 1

    @pytest.mark.asyncio
    async def test_observe_debate_confidence_increases(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that confidence increases with more observations."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(
                sample_debate_result,
                workspace_id="ws_001",
            )

        patterns = culture_accumulator.get_patterns("ws_001")
        high_confidence = [p for p in patterns if p.confidence > 0.3]
        assert len(high_confidence) >= 1

    @pytest.mark.asyncio
    async def test_observe_debate_with_no_domain(
        self, culture_accumulator, sample_debate_result_no_winner
    ):
        """Test observing debate with no inferred domain."""
        patterns = await culture_accumulator.observe_debate(
            sample_debate_result_no_winner,
            workspace_id="ws_001",
        )

        # Should still create decision style pattern
        decision_patterns = [
            p for p in patterns if p.pattern_type == CulturePatternType.DECISION_STYLE
        ]
        assert len(decision_patterns) >= 1

    @pytest.mark.asyncio
    async def test_observe_debate_returns_empty_on_invalid_result(self, culture_accumulator):
        """Test that invalid debate result returns empty list."""

        class BrokenResult:
            @property
            def proposals(self):
                raise RuntimeError("Broken")

        patterns = await culture_accumulator.observe_debate(BrokenResult(), workspace_id="ws_001")

        assert patterns == []

    @pytest.mark.asyncio
    async def test_observe_debate_emits_mound_updated_event(
        self, culture_accumulator_with_emitter, sample_debate_result
    ):
        """Test that observing debate emits MOUND_UPDATED event."""
        patterns = await culture_accumulator_with_emitter.observe_debate(
            sample_debate_result,
            workspace_id="ws_001",
        )

        # Event should have been emitted since patterns were created
        if patterns:
            culture_accumulator_with_emitter._mound.event_emitter.emit_sync.assert_called()


# ============================================================================
# CultureAccumulator Tests - _emit_mound_updated
# ============================================================================


class TestCultureAccumulatorEmitMoundUpdated:
    """Tests for CultureAccumulator._emit_mound_updated method."""

    def test_emit_with_no_event_emitter(self, culture_accumulator):
        """Test emit does nothing when no event emitter."""
        # Should not raise
        culture_accumulator._emit_mound_updated(
            workspace_id="ws_001",
            update_type="test",
        )

    def test_emit_with_event_emitter(self, culture_accumulator_with_emitter):
        """Test emit calls event emitter."""
        culture_accumulator_with_emitter._emit_mound_updated(
            workspace_id="ws_001",
            update_type="culture_patterns",
            patterns_count=5,
        )

        culture_accumulator_with_emitter._mound.event_emitter.emit_sync.assert_called_once()

    def test_emit_handles_attribute_error(self, mock_mound):
        """Test emit handles AttributeError gracefully."""
        mock_mound.event_emitter = MagicMock()
        mock_mound.event_emitter.emit_sync = MagicMock(side_effect=AttributeError())

        acc = CultureAccumulator(mound=mock_mound)
        # Should not raise
        acc._emit_mound_updated(workspace_id="ws_001", update_type="test")

    def test_emit_handles_type_error(self, mock_mound):
        """Test emit handles TypeError gracefully."""
        mock_mound.event_emitter = MagicMock()
        mock_mound.event_emitter.emit_sync = MagicMock(side_effect=TypeError())

        acc = CultureAccumulator(mound=mock_mound)
        # Should not raise
        acc._emit_mound_updated(workspace_id="ws_001", update_type="test")


# ============================================================================
# CultureAccumulator Tests - get_patterns
# ============================================================================


class TestCultureAccumulatorGetPatterns:
    """Tests for CultureAccumulator.get_patterns method."""

    @pytest.mark.asyncio
    async def test_get_patterns_empty_workspace(self, culture_accumulator):
        """Test getting patterns for empty workspace."""
        patterns = culture_accumulator.get_patterns("empty_ws")

        assert patterns == []

    @pytest.mark.asyncio
    async def test_get_patterns_all_types(self, culture_accumulator, sample_debate_result):
        """Test getting all patterns."""
        await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        patterns = culture_accumulator.get_patterns("ws_001")

        assert len(patterns) >= 1

    @pytest.mark.asyncio
    async def test_get_patterns_by_type(self, culture_accumulator, sample_debate_result):
        """Test getting patterns filtered by type."""
        await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        patterns = culture_accumulator.get_patterns(
            "ws_001",
            pattern_type=CulturePatternType.AGENT_PREFERENCES,
        )

        for p in patterns:
            assert p.pattern_type == CulturePatternType.AGENT_PREFERENCES

    @pytest.mark.asyncio
    async def test_get_patterns_min_confidence(self, culture_accumulator, sample_debate_result):
        """Test getting patterns with minimum confidence."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        patterns = culture_accumulator.get_patterns(
            "ws_001",
            min_confidence=0.5,
        )

        for p in patterns:
            assert p.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_get_patterns_min_observations(self, culture_accumulator, sample_debate_result):
        """Test getting patterns with minimum observations."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        patterns = culture_accumulator.get_patterns(
            "ws_001",
            min_observations=3,
        )

        for p in patterns:
            assert p.observation_count >= 3

    @pytest.mark.asyncio
    async def test_get_patterns_sorted_by_confidence(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that patterns are sorted by confidence."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        patterns = culture_accumulator.get_patterns("ws_001")

        if len(patterns) >= 2:
            for i in range(len(patterns) - 1):
                assert patterns[i].confidence >= patterns[i + 1].confidence

    @pytest.mark.asyncio
    async def test_get_patterns_nonexistent_type(self, culture_accumulator, sample_debate_result):
        """Test getting patterns for type that has no data."""
        await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        patterns = culture_accumulator.get_patterns(
            "ws_001",
            pattern_type=CulturePatternType.RISK_TOLERANCE,
        )

        assert patterns == []


# ============================================================================
# CultureAccumulator Tests - get_patterns_summary
# ============================================================================


class TestCultureAccumulatorGetPatternsSummary:
    """Tests for CultureAccumulator.get_patterns_summary method."""

    def test_summary_empty_workspace(self, culture_accumulator):
        """Test summary for empty workspace."""
        summary = culture_accumulator.get_patterns_summary("empty_ws")

        assert summary["workspace_id"] == "empty_ws"
        assert summary["total_patterns"] == 0
        assert summary["patterns_by_type"] == {}
        assert summary["total_observations"] == 0
        assert summary["top_patterns"] == []

    @pytest.mark.asyncio
    async def test_summary_with_patterns(self, culture_accumulator, sample_debate_result):
        """Test summary with patterns."""
        for i in range(3):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        summary = culture_accumulator.get_patterns_summary("ws_001")

        assert summary["workspace_id"] == "ws_001"
        assert summary["total_patterns"] >= 1
        assert summary["total_observations"] >= 1
        assert len(summary["patterns_by_type"]) >= 1
        assert len(summary["top_patterns"]) >= 1

    @pytest.mark.asyncio
    async def test_summary_top_patterns_format(self, culture_accumulator, sample_debate_result):
        """Test top patterns format in summary."""
        await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        summary = culture_accumulator.get_patterns_summary("ws_001")

        if summary["top_patterns"]:
            top = summary["top_patterns"][0]
            assert "id" in top
            assert "type" in top
            assert "key" in top
            assert "confidence" in top
            assert "observations" in top

    @pytest.mark.asyncio
    async def test_summary_top_patterns_limit(self, culture_accumulator, sample_debate_result):
        """Test that top patterns are limited to 10."""
        # Create many patterns by observing debates with different domains
        domains = [
            "security",
            "performance",
            "testing",
            "api",
            "database",
            "frontend",
            "architecture",
            "legal",
            "financial",
        ]
        for i, domain in enumerate(domains):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            sample_debate_result.task = f"Review {domain} design"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        summary = culture_accumulator.get_patterns_summary("ws_001")

        assert len(summary["top_patterns"]) <= 10


# ============================================================================
# CultureAccumulator Tests - get_profile
# ============================================================================


class TestCultureAccumulatorGetProfile:
    """Tests for CultureAccumulator.get_profile method."""

    @pytest.mark.asyncio
    async def test_profile_empty_workspace(self, culture_accumulator):
        """Test profile for empty workspace."""
        profile = await culture_accumulator.get_profile("empty_ws")

        assert profile.workspace_id == "empty_ws"
        assert profile.total_observations == 0
        assert profile.dominant_traits == {}

    @pytest.mark.asyncio
    async def test_profile_with_patterns(self, culture_accumulator, sample_debate_result):
        """Test profile with patterns."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        profile = await culture_accumulator.get_profile("ws_001")

        assert profile.workspace_id == "ws_001"
        assert profile.total_observations >= 1
        assert isinstance(profile.patterns, dict)
        assert isinstance(profile.generated_at, datetime)

    @pytest.mark.asyncio
    async def test_profile_dominant_traits_top_agents(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that top agents are extracted in dominant traits."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        profile = await culture_accumulator.get_profile("ws_001")

        if "top_agents" in profile.dominant_traits:
            assert isinstance(profile.dominant_traits["top_agents"], list)

    @pytest.mark.asyncio
    async def test_profile_dominant_traits_consensus_info(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that consensus info is extracted in dominant traits."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        profile = await culture_accumulator.get_profile("ws_001")

        # Decision style patterns should provide this
        if "avg_consensus_rounds" in profile.dominant_traits:
            assert isinstance(profile.dominant_traits["avg_consensus_rounds"], (int, float))

    @pytest.mark.asyncio
    async def test_profile_dominant_traits_expertise_areas(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that expertise areas are extracted in dominant traits."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        profile = await culture_accumulator.get_profile("ws_001")

        if "expertise_areas" in profile.dominant_traits:
            assert isinstance(profile.dominant_traits["expertise_areas"], list)

    @pytest.mark.asyncio
    async def test_profile_filters_confident_patterns(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that profile only includes confident patterns."""
        # Single observation - low confidence
        await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        profile = await culture_accumulator.get_profile("ws_001")

        # Patterns with confidence >= 0.3 or observation_count >= min_observations
        # are included


# ============================================================================
# CultureAccumulator Tests - recommend_agents
# ============================================================================


class TestCultureAccumulatorRecommendAgents:
    """Tests for CultureAccumulator.recommend_agents method."""

    @pytest.mark.asyncio
    async def test_recommend_agents_empty_workspace(self, culture_accumulator):
        """Test recommendations for empty workspace."""
        recommendations = await culture_accumulator.recommend_agents(
            task_type="security",
            workspace_id="empty_ws",
        )

        assert recommendations == []

    @pytest.mark.asyncio
    async def test_recommend_agents_with_data(self, culture_accumulator, sample_debate_result):
        """Test recommendations with accumulated data."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        recommendations = await culture_accumulator.recommend_agents(
            task_type="security",
            workspace_id="ws_001",
        )

        assert isinstance(recommendations, list)
        if recommendations:
            assert "claude" in recommendations

    @pytest.mark.asyncio
    async def test_recommend_agents_limit(self, culture_accumulator, sample_debate_result):
        """Test that recommendations are limited to 3."""
        for i in range(10):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        recommendations = await culture_accumulator.recommend_agents(
            task_type="security",
            workspace_id="ws_001",
        )

        assert len(recommendations) <= 3

    @pytest.mark.asyncio
    async def test_recommend_agents_partial_match(self, culture_accumulator, sample_debate_result):
        """Test recommendations with partial task type match."""
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        # "sec" should match "security"
        recommendations = await culture_accumulator.recommend_agents(
            task_type="sec",
            workspace_id="ws_001",
        )

        # May or may not match depending on implementation


# ============================================================================
# CultureAccumulator Tests - _update_* methods
# ============================================================================


class TestCultureAccumulatorUpdateMethods:
    """Tests for CultureAccumulator pattern update methods."""

    @pytest.mark.asyncio
    async def test_update_agent_preferences_creates_pattern(self, culture_accumulator):
        """Test that agent preferences pattern is created."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Security review",
            participating_agents=["claude", "gpt4"],
            winning_agents=["claude"],
            rounds_to_consensus=3,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=[],
            domain="security",
        )

        patterns = await culture_accumulator._update_agent_preferences(obs, "ws_001")

        assert len(patterns) >= 1
        assert patterns[0].pattern_type == CulturePatternType.AGENT_PREFERENCES
        assert patterns[0].pattern_value["agent"] == "claude"
        assert patterns[0].pattern_value["domain"] == "security"

    @pytest.mark.asyncio
    async def test_update_agent_preferences_no_winner(self, culture_accumulator):
        """Test agent preferences with no winner."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Random",
            participating_agents=["claude"],
            winning_agents=[],
            rounds_to_consensus=5,
            consensus_reached=False,
            consensus_strength="weak",
            critique_patterns=[],
            domain="security",
        )

        patterns = await culture_accumulator._update_agent_preferences(obs, "ws_001")

        assert patterns == []

    @pytest.mark.asyncio
    async def test_update_agent_preferences_no_domain(self, culture_accumulator):
        """Test agent preferences with no domain."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Random",
            participating_agents=["claude"],
            winning_agents=["claude"],
            rounds_to_consensus=1,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=[],
            domain=None,
        )

        patterns = await culture_accumulator._update_agent_preferences(obs, "ws_001")

        assert patterns == []

    @pytest.mark.asyncio
    async def test_update_decision_style_creates_pattern(self, culture_accumulator):
        """Test that decision style pattern is created."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Test",
            participating_agents=[],
            winning_agents=[],
            rounds_to_consensus=4,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=[],
            domain=None,
        )

        patterns = await culture_accumulator._update_decision_style(obs, "ws_001")

        assert len(patterns) == 1
        assert patterns[0].pattern_type == CulturePatternType.DECISION_STYLE
        assert patterns[0].pattern_value["avg_rounds"] == 4
        assert patterns[0].pattern_value["consensus_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_update_decision_style_updates_averages(self, culture_accumulator):
        """Test that decision style averages are updated correctly."""
        obs1 = DebateObservation(
            debate_id="debate_001",
            topic="Test",
            participating_agents=[],
            winning_agents=[],
            rounds_to_consensus=2,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=[],
            domain=None,
        )

        obs2 = DebateObservation(
            debate_id="debate_002",
            topic="Test",
            participating_agents=[],
            winning_agents=[],
            rounds_to_consensus=6,
            consensus_reached=False,
            consensus_strength="weak",
            critique_patterns=[],
            domain=None,
        )

        await culture_accumulator._update_decision_style(obs1, "ws_001")
        patterns = await culture_accumulator._update_decision_style(obs2, "ws_001")

        # Average should be (2 + 6) / 2 = 4
        assert patterns[0].pattern_value["avg_rounds"] == 4.0
        # Consensus rate should be (1.0 + 0.0) / 2 = 0.5
        assert patterns[0].pattern_value["consensus_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_update_debate_dynamics_creates_patterns(self, culture_accumulator):
        """Test that debate dynamics patterns are created for critiques."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Test",
            participating_agents=[],
            winning_agents=[],
            rounds_to_consensus=1,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=["security", "scalability"],
            domain=None,
        )

        patterns = await culture_accumulator._update_debate_dynamics(obs, "ws_001")

        assert len(patterns) == 2
        for p in patterns:
            assert p.pattern_type == CulturePatternType.DEBATE_DYNAMICS

    @pytest.mark.asyncio
    async def test_update_debate_dynamics_no_critiques(self, culture_accumulator):
        """Test debate dynamics with no critiques."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Test",
            participating_agents=[],
            winning_agents=[],
            rounds_to_consensus=1,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=[],
            domain=None,
        )

        patterns = await culture_accumulator._update_debate_dynamics(obs, "ws_001")

        assert patterns == []

    @pytest.mark.asyncio
    async def test_update_domain_expertise_creates_pattern(self, culture_accumulator):
        """Test that domain expertise pattern is created."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Test",
            participating_agents=[],
            winning_agents=[],
            rounds_to_consensus=1,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=[],
            domain="security",
        )

        patterns = await culture_accumulator._update_domain_expertise(obs, "ws_001")

        assert len(patterns) == 1
        assert patterns[0].pattern_type == CulturePatternType.DOMAIN_EXPERTISE
        assert patterns[0].pattern_key == "security"

    @pytest.mark.asyncio
    async def test_update_domain_expertise_no_domain(self, culture_accumulator):
        """Test domain expertise with no domain."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Test",
            participating_agents=[],
            winning_agents=[],
            rounds_to_consensus=1,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=[],
            domain=None,
        )

        patterns = await culture_accumulator._update_domain_expertise(obs, "ws_001")

        assert patterns == []


# ============================================================================
# OrganizationCultureManager Tests - Initialization
# ============================================================================


class TestOrganizationCultureManagerInit:
    """Tests for OrganizationCultureManager initialization."""

    def test_basic_init(self, mock_mound, culture_accumulator):
        """Test basic initialization."""
        manager = OrganizationCultureManager(
            mound=mock_mound,
            culture_accumulator=culture_accumulator,
        )

        assert manager._mound == mock_mound
        assert manager._accumulator == culture_accumulator

    def test_init_without_accumulator(self, mock_mound):
        """Test initialization without accumulator."""
        manager = OrganizationCultureManager(
            mound=mock_mound,
            culture_accumulator=None,
        )

        assert manager._accumulator is None


# ============================================================================
# OrganizationCultureManager Tests - register_workspace
# ============================================================================


class TestOrganizationCultureManagerRegisterWorkspace:
    """Tests for OrganizationCultureManager.register_workspace method."""

    def test_register_workspace(self, org_culture_manager):
        """Test registering a workspace."""
        org_culture_manager.register_workspace("ws_001", "org_001")

        assert org_culture_manager._workspace_orgs["ws_001"] == "org_001"

    def test_register_multiple_workspaces(self, org_culture_manager):
        """Test registering multiple workspaces."""
        org_culture_manager.register_workspace("ws_001", "org_001")
        org_culture_manager.register_workspace("ws_002", "org_001")
        org_culture_manager.register_workspace("ws_003", "org_002")

        assert org_culture_manager._workspace_orgs["ws_001"] == "org_001"
        assert org_culture_manager._workspace_orgs["ws_002"] == "org_001"
        assert org_culture_manager._workspace_orgs["ws_003"] == "org_002"

    def test_register_workspace_overwrites(self, org_culture_manager):
        """Test that re-registering overwrites org."""
        org_culture_manager.register_workspace("ws_001", "org_001")
        org_culture_manager.register_workspace("ws_001", "org_002")

        assert org_culture_manager._workspace_orgs["ws_001"] == "org_002"


# ============================================================================
# OrganizationCultureManager Tests - add_document
# ============================================================================


class TestOrganizationCultureManagerAddDocument:
    """Tests for OrganizationCultureManager.add_document method."""

    @pytest.mark.asyncio
    async def test_add_document_basic(self, org_culture_manager):
        """Test adding a basic document."""
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Core Values",
            content="We believe in quality.",
            created_by="admin",
        )

        assert doc.id.startswith("cd_")
        assert doc.org_id == "org_001"
        assert doc.category == CultureDocumentCategory.VALUES
        assert doc.title == "Core Values"
        assert doc.content == "We believe in quality."
        assert doc.created_by == "admin"
        assert doc.version == 1
        assert doc.is_active is True

    @pytest.mark.asyncio
    async def test_add_document_with_metadata(self, org_culture_manager):
        """Test adding document with metadata."""
        metadata = {"source": "manual", "reviewed": True}
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.POLICIES,
            title="Policy",
            content="Content",
            created_by="admin",
            metadata=metadata,
        )

        assert doc.metadata == metadata

    @pytest.mark.asyncio
    async def test_add_document_with_vector_store(self, org_culture_manager_with_vector):
        """Test adding document generates embeddings."""
        doc = await org_culture_manager_with_vector.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Test",
            content="Test content",
            created_by="admin",
        )

        assert doc.embeddings is not None
        assert doc.embeddings == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_add_document_stores_in_dict(self, org_culture_manager):
        """Test that document is stored in internal dict."""
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Test",
            content="Content",
            created_by="admin",
        )

        assert doc.id in org_culture_manager._documents["org_001"]

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, org_culture_manager):
        """Test adding multiple documents."""
        doc1 = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Values",
            content="Content 1",
            created_by="admin",
        )
        doc2 = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Practices",
            content="Content 2",
            created_by="admin",
        )

        assert len(org_culture_manager._documents["org_001"]) == 2
        assert doc1.id in org_culture_manager._documents["org_001"]
        assert doc2.id in org_culture_manager._documents["org_001"]


# ============================================================================
# OrganizationCultureManager Tests - update_document
# ============================================================================


class TestOrganizationCultureManagerUpdateDocument:
    """Tests for OrganizationCultureManager.update_document method."""

    @pytest.mark.asyncio
    async def test_update_document(self, org_culture_manager):
        """Test updating a document."""
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Code Review",
            content="All code must be reviewed.",
            created_by="admin",
        )

        updated = await org_culture_manager.update_document(
            doc_id=doc.id,
            org_id="org_001",
            content="All code must be reviewed by at least two people.",
            updated_by="manager",
        )

        assert updated.version == 2
        assert updated.supersedes == doc.id
        assert updated.is_active is True
        assert updated.content == "All code must be reviewed by at least two people."

    @pytest.mark.asyncio
    async def test_update_deactivates_old_doc(self, org_culture_manager):
        """Test that updating deactivates old document."""
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.POLICIES,
            title="Policy",
            content="Old content",
            created_by="admin",
        )

        await org_culture_manager.update_document(
            doc_id=doc.id,
            org_id="org_001",
            content="New content",
            updated_by="admin",
        )

        assert doc.is_active is False

    @pytest.mark.asyncio
    async def test_update_preserves_category(self, org_culture_manager):
        """Test that update preserves category."""
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.STANDARDS,
            title="Standard",
            content="Old content",
            created_by="admin",
        )

        updated = await org_culture_manager.update_document(
            doc_id=doc.id,
            org_id="org_001",
            content="New content",
            updated_by="admin",
        )

        assert updated.category == CultureDocumentCategory.STANDARDS

    @pytest.mark.asyncio
    async def test_update_preserves_title(self, org_culture_manager):
        """Test that update preserves title."""
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Important Title",
            content="Old content",
            created_by="admin",
        )

        updated = await org_culture_manager.update_document(
            doc_id=doc.id,
            org_id="org_001",
            content="New content",
            updated_by="admin",
        )

        assert updated.title == "Important Title"

    @pytest.mark.asyncio
    async def test_update_nonexistent_doc_raises(self, org_culture_manager):
        """Test that updating nonexistent doc raises error."""
        with pytest.raises(ValueError, match="Document not found"):
            await org_culture_manager.update_document(
                doc_id="nonexistent",
                org_id="org_001",
                content="New content",
                updated_by="admin",
            )

    @pytest.mark.asyncio
    async def test_update_with_vector_store(self, org_culture_manager_with_vector):
        """Test update generates new embeddings."""
        doc = await org_culture_manager_with_vector.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Test",
            content="Old content",
            created_by="admin",
        )

        updated = await org_culture_manager_with_vector.update_document(
            doc_id=doc.id,
            org_id="org_001",
            content="New content",
            updated_by="admin",
        )

        assert updated.embeddings is not None


# ============================================================================
# OrganizationCultureManager Tests - get_documents
# ============================================================================


class TestOrganizationCultureManagerGetDocuments:
    """Tests for OrganizationCultureManager.get_documents method."""

    @pytest.mark.asyncio
    async def test_get_documents_empty(self, org_culture_manager):
        """Test getting documents from empty org."""
        docs = await org_culture_manager.get_documents("org_001")

        assert docs == []

    @pytest.mark.asyncio
    async def test_get_all_documents(self, org_culture_manager):
        """Test getting all documents."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Values",
            content="Content",
            created_by="admin",
        )
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Practices",
            content="Content",
            created_by="admin",
        )

        docs = await org_culture_manager.get_documents("org_001")

        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_get_documents_by_category(self, org_culture_manager):
        """Test filtering documents by category."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Values",
            content="Content",
            created_by="admin",
        )
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Practices",
            content="Content",
            created_by="admin",
        )

        docs = await org_culture_manager.get_documents(
            "org_001",
            category=CultureDocumentCategory.VALUES,
        )

        assert len(docs) == 1
        assert docs[0].category == CultureDocumentCategory.VALUES

    @pytest.mark.asyncio
    async def test_get_documents_active_only(self, org_culture_manager):
        """Test getting only active documents."""
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Values",
            content="Old content",
            created_by="admin",
        )

        await org_culture_manager.update_document(
            doc_id=doc.id,
            org_id="org_001",
            content="New content",
            updated_by="admin",
        )

        active_docs = await org_culture_manager.get_documents("org_001", active_only=True)
        all_docs = await org_culture_manager.get_documents("org_001", active_only=False)

        assert len(active_docs) == 1
        assert len(all_docs) == 2

    @pytest.mark.asyncio
    async def test_get_documents_sorted_by_updated(self, org_culture_manager):
        """Test that documents are sorted by updated_at descending."""
        doc1 = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="First",
            content="Content",
            created_by="admin",
        )
        doc2 = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Second",
            content="Content",
            created_by="admin",
        )

        docs = await org_culture_manager.get_documents("org_001")

        # Most recently updated should be first
        assert docs[0].title == "Second"


# ============================================================================
# OrganizationCultureManager Tests - query_culture
# ============================================================================


class TestOrganizationCultureManagerQueryCulture:
    """Tests for OrganizationCultureManager.query_culture method."""

    @pytest.mark.asyncio
    async def test_query_empty_org(self, org_culture_manager):
        """Test querying empty org."""
        results = await org_culture_manager.query_culture("org_001", "test")

        assert results == []

    @pytest.mark.asyncio
    async def test_query_by_keyword_in_title(self, org_culture_manager):
        """Test querying by keyword in title."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Security Practices",
            content="General practices",
            created_by="admin",
        )

        results = await org_culture_manager.query_culture("org_001", "security")

        assert len(results) >= 1
        assert "security" in results[0].title.lower()

    @pytest.mark.asyncio
    async def test_query_by_keyword_in_content(self, org_culture_manager):
        """Test querying by keyword in content."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Best Practices",
            content="Use encryption for sensitive data.",
            created_by="admin",
        )

        results = await org_culture_manager.query_culture("org_001", "encryption")

        assert len(results) >= 1
        assert "encryption" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_query_with_limit(self, org_culture_manager):
        """Test query respects limit."""
        for i in range(5):
            await org_culture_manager.add_document(
                org_id="org_001",
                category=CultureDocumentCategory.VALUES,
                title=f"Value {i}",
                content=f"Important value {i}",
                created_by="admin",
            )

        results = await org_culture_manager.query_culture("org_001", "important", limit=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_query_no_match(self, org_culture_manager):
        """Test query with no matches."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Security",
            content="Security matters.",
            created_by="admin",
        )

        results = await org_culture_manager.query_culture("org_001", "xyz123")

        assert results == []

    @pytest.mark.asyncio
    async def test_query_title_scored_higher(self, org_culture_manager):
        """Test that title matches score higher than content."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Test Document",
            content="Random content",
            created_by="admin",
        )
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Random Title",
            content="Test content here",
            created_by="admin",
        )

        results = await org_culture_manager.query_culture("org_001", "test")

        # Title match should come first
        assert results[0].title == "Test Document"


# ============================================================================
# OrganizationCultureManager Tests - _cosine_similarity
# ============================================================================


class TestOrganizationCultureManagerCosineSimilarity:
    """Tests for OrganizationCultureManager._cosine_similarity method."""

    def test_identical_vectors(self, org_culture_manager):
        """Test similarity of identical vectors."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]

        assert org_culture_manager._cosine_similarity(a, b) == 1.0

    def test_orthogonal_vectors(self, org_culture_manager):
        """Test similarity of orthogonal vectors."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]

        assert org_culture_manager._cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self, org_culture_manager):
        """Test similarity of opposite vectors."""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]

        assert org_culture_manager._cosine_similarity(a, b) == -1.0

    def test_partial_similarity(self, org_culture_manager):
        """Test partial similarity."""
        a = [1.0, 0.0, 0.0]
        b = [0.5, 0.5, 0.0]

        # cos(45 degrees) = sqrt(2)/2 ≈ 0.707
        result = org_culture_manager._cosine_similarity(a, b)
        assert 0.7 < result < 0.72

    def test_zero_vector(self, org_culture_manager):
        """Test similarity with zero vector."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]

        assert org_culture_manager._cosine_similarity(a, b) == 0.0

    def test_different_length_vectors(self, org_culture_manager):
        """Test similarity of different length vectors."""
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]

        assert org_culture_manager._cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self, org_culture_manager):
        """Test similarity of both zero vectors."""
        a = [0.0, 0.0]
        b = [0.0, 0.0]

        assert org_culture_manager._cosine_similarity(a, b) == 0.0


# ============================================================================
# OrganizationCultureManager Tests - promote_pattern_to_culture
# ============================================================================


class TestOrganizationCultureManagerPromotePattern:
    """Tests for OrganizationCultureManager.promote_pattern_to_culture method."""

    @pytest.mark.asyncio
    async def test_promote_pattern(
        self, org_culture_manager, culture_accumulator, sample_debate_result
    ):
        """Test promoting a pattern to culture document."""
        org_culture_manager.register_workspace("ws_001", "org_001")

        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id="ws_001")

        profile = await culture_accumulator.get_profile("ws_001")
        all_patterns = []
        for patterns in profile.patterns.values():
            all_patterns.extend(patterns)

        if all_patterns:
            pattern = all_patterns[0]
            doc = await org_culture_manager.promote_pattern_to_culture(
                workspace_id="ws_001",
                pattern_id=pattern.id,
                promoted_by="admin",
                title="Promoted Pattern",
            )

            assert doc.category == CultureDocumentCategory.LEARNINGS
            assert doc.source_pattern_id == pattern.id
            assert doc.source_workspace_id == "ws_001"

    @pytest.mark.asyncio
    async def test_promote_unregistered_workspace_raises(self, org_culture_manager):
        """Test promoting from unregistered workspace raises error."""
        with pytest.raises(ValueError, match="not registered"):
            await org_culture_manager.promote_pattern_to_culture(
                workspace_id="unregistered",
                pattern_id="cp_001",
                promoted_by="admin",
            )

    @pytest.mark.asyncio
    async def test_promote_without_accumulator_raises(self, org_culture_manager_no_accumulator):
        """Test promoting without accumulator raises error."""
        org_culture_manager_no_accumulator.register_workspace("ws_001", "org_001")

        with pytest.raises(ValueError, match="accumulator not configured"):
            await org_culture_manager_no_accumulator.promote_pattern_to_culture(
                workspace_id="ws_001",
                pattern_id="cp_001",
                promoted_by="admin",
            )

    @pytest.mark.asyncio
    async def test_promote_nonexistent_pattern_raises(
        self, org_culture_manager, culture_accumulator
    ):
        """Test promoting nonexistent pattern raises error."""
        org_culture_manager.register_workspace("ws_001", "org_001")

        with pytest.raises(ValueError, match="Pattern not found"):
            await org_culture_manager.promote_pattern_to_culture(
                workspace_id="ws_001",
                pattern_id="nonexistent",
                promoted_by="admin",
            )


# ============================================================================
# OrganizationCultureManager Tests - _pattern_to_content
# ============================================================================


class TestOrganizationCultureManagerPatternToContent:
    """Tests for OrganizationCultureManager._pattern_to_content method."""

    def test_pattern_to_content_format(self, org_culture_manager):
        """Test pattern to content formatting."""
        pattern = CulturePattern(
            id="cp_001",
            workspace_id="ws_001",
            pattern_type=CulturePatternType.AGENT_PREFERENCES,
            pattern_key="security:claude",
            pattern_value={"agent": "claude", "domain": "security"},
            observation_count=5,
            confidence=0.85,
            first_observed_at=datetime.now(),
            last_observed_at=datetime.now(),
            contributing_debates=["d1", "d2"],
        )

        content = org_culture_manager._pattern_to_content(pattern)

        assert "Pattern Type: agent_preferences" in content
        assert "Pattern Key: security:claude" in content
        assert "Observations: 5" in content
        assert "Confidence: 85.00%" in content
        assert "agent: claude" in content
        assert "domain: security" in content
        assert "Contributing Debates: 2" in content


# ============================================================================
# OrganizationCultureManager Tests - get_organization_culture
# ============================================================================


class TestOrganizationCultureManagerGetOrganizationCulture:
    """Tests for OrganizationCultureManager.get_organization_culture method."""

    @pytest.mark.asyncio
    async def test_get_empty_org_culture(self, org_culture_manager):
        """Test getting culture for empty org."""
        culture = await org_culture_manager.get_organization_culture("org_001")

        assert culture.org_id == "org_001"
        assert culture.documents == []
        assert culture.workspace_count == 0

    @pytest.mark.asyncio
    async def test_get_org_culture_with_documents(self, org_culture_manager):
        """Test getting culture with documents."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Values",
            content="Content",
            created_by="admin",
        )

        culture = await org_culture_manager.get_organization_culture("org_001")

        assert len(culture.documents) == 1

    @pytest.mark.asyncio
    async def test_get_org_culture_with_workspaces(
        self, org_culture_manager, culture_accumulator, sample_debate_result
    ):
        """Test getting culture with workspace patterns."""
        org_culture_manager.register_workspace("ws_001", "org_001")
        org_culture_manager.register_workspace("ws_002", "org_001")

        for ws_id in ["ws_001", "ws_002"]:
            for i in range(2):
                sample_debate_result.debate_id = f"{ws_id}_debate_{i}"
                await culture_accumulator.observe_debate(sample_debate_result, workspace_id=ws_id)

        culture = await org_culture_manager.get_organization_culture("org_001")

        assert culture.workspace_count == 2
        assert "ws_001" in culture.workspace_profiles
        assert "ws_002" in culture.workspace_profiles

    @pytest.mark.asyncio
    async def test_get_org_culture_specific_workspaces(
        self, org_culture_manager, culture_accumulator, sample_debate_result
    ):
        """Test getting culture for specific workspaces."""
        org_culture_manager.register_workspace("ws_001", "org_001")
        org_culture_manager.register_workspace("ws_002", "org_001")
        org_culture_manager.register_workspace("ws_003", "org_001")

        for ws_id in ["ws_001", "ws_002", "ws_003"]:
            sample_debate_result.debate_id = f"{ws_id}_debate"
            await culture_accumulator.observe_debate(sample_debate_result, workspace_id=ws_id)

        culture = await org_culture_manager.get_organization_culture(
            "org_001",
            workspace_ids=["ws_001", "ws_002"],
        )

        # Should only include specified workspaces
        assert "ws_001" in culture.workspace_profiles
        assert "ws_002" in culture.workspace_profiles
        # ws_003 should not be included
        assert "ws_003" not in culture.workspace_profiles


# ============================================================================
# OrganizationCultureManager Tests - _extract_dominant_traits
# ============================================================================


class TestOrganizationCultureManagerExtractDominantTraits:
    """Tests for OrganizationCultureManager._extract_dominant_traits method."""

    def test_extract_top_agents(self, org_culture_manager):
        """Test extracting top agents from patterns."""
        patterns = {
            CulturePatternType.AGENT_PREFERENCES: [
                CulturePattern(
                    id="cp_001",
                    workspace_id="ws_001",
                    pattern_type=CulturePatternType.AGENT_PREFERENCES,
                    pattern_key="security:claude",
                    pattern_value={"agent": "claude"},
                    observation_count=10,
                    confidence=0.9,
                    first_observed_at=datetime.now(),
                    last_observed_at=datetime.now(),
                ),
                CulturePattern(
                    id="cp_002",
                    workspace_id="ws_001",
                    pattern_type=CulturePatternType.AGENT_PREFERENCES,
                    pattern_key="security:gpt4",
                    pattern_value={"agent": "gpt4"},
                    observation_count=5,
                    confidence=0.7,
                    first_observed_at=datetime.now(),
                    last_observed_at=datetime.now(),
                ),
            ],
            CulturePatternType.DOMAIN_EXPERTISE: [],
            CulturePatternType.DECISION_STYLE: [],
        }

        traits = org_culture_manager._extract_dominant_traits(patterns)

        assert "top_agents" in traits
        assert "claude" in traits["top_agents"]

    def test_extract_expertise_areas(self, org_culture_manager):
        """Test extracting expertise areas from patterns."""
        patterns = {
            CulturePatternType.AGENT_PREFERENCES: [],
            CulturePatternType.DOMAIN_EXPERTISE: [
                CulturePattern(
                    id="cp_001",
                    workspace_id="ws_001",
                    pattern_type=CulturePatternType.DOMAIN_EXPERTISE,
                    pattern_key="security",
                    pattern_value={},
                    observation_count=10,
                    confidence=0.9,
                    first_observed_at=datetime.now(),
                    last_observed_at=datetime.now(),
                ),
                CulturePattern(
                    id="cp_002",
                    workspace_id="ws_001",
                    pattern_type=CulturePatternType.DOMAIN_EXPERTISE,
                    pattern_key="api",
                    pattern_value={},
                    observation_count=5,
                    confidence=0.7,
                    first_observed_at=datetime.now(),
                    last_observed_at=datetime.now(),
                ),
            ],
            CulturePatternType.DECISION_STYLE: [],
        }

        traits = org_culture_manager._extract_dominant_traits(patterns)

        assert "expertise_areas" in traits
        assert "security" in traits["expertise_areas"]

    def test_extract_decision_patterns(self, org_culture_manager):
        """Test extracting decision patterns."""
        patterns = {
            CulturePatternType.AGENT_PREFERENCES: [],
            CulturePatternType.DOMAIN_EXPERTISE: [],
            CulturePatternType.DECISION_STYLE: [
                CulturePattern(
                    id="cp_001",
                    workspace_id="ws_001",
                    pattern_type=CulturePatternType.DECISION_STYLE,
                    pattern_key="consensus_pattern",
                    pattern_value={"avg_rounds": 3.5, "consensus_rate": 0.85},
                    observation_count=10,
                    confidence=0.9,
                    first_observed_at=datetime.now(),
                    last_observed_at=datetime.now(),
                ),
            ],
        }

        traits = org_culture_manager._extract_dominant_traits(patterns)

        assert "avg_consensus_rounds" in traits
        assert "consensus_rate" in traits

    def test_extract_empty_patterns(self, org_culture_manager):
        """Test extracting from empty patterns."""
        patterns = {
            CulturePatternType.AGENT_PREFERENCES: [],
            CulturePatternType.DOMAIN_EXPERTISE: [],
            CulturePatternType.DECISION_STYLE: [],
        }

        traits = org_culture_manager._extract_dominant_traits(patterns)

        assert traits == {}


# ============================================================================
# OrganizationCultureManager Tests - get_relevant_context
# ============================================================================


class TestOrganizationCultureManagerGetRelevantContext:
    """Tests for OrganizationCultureManager.get_relevant_context method."""

    @pytest.mark.asyncio
    async def test_get_context_empty(self, org_culture_manager):
        """Test getting context for empty org."""
        context = await org_culture_manager.get_relevant_context(
            org_id="org_001",
            task="test task",
        )

        assert context == ""

    @pytest.mark.asyncio
    async def test_get_context_with_match(self, org_culture_manager):
        """Test getting context with matching document."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="API Design",
            content="All APIs should be RESTful.",
            created_by="admin",
        )

        context = await org_culture_manager.get_relevant_context(
            org_id="org_001",
            task="api",
        )

        assert "Organizational Context" in context
        assert "API Design" in context

    @pytest.mark.asyncio
    async def test_get_context_respects_limit(self, org_culture_manager):
        """Test that context respects max_documents limit."""
        for i in range(5):
            await org_culture_manager.add_document(
                org_id="org_001",
                category=CultureDocumentCategory.VALUES,
                title=f"Value {i}",
                content=f"Test value {i}",
                created_by="admin",
            )

        context = await org_culture_manager.get_relevant_context(
            org_id="org_001",
            task="test",
            max_documents=2,
        )

        # Should only include 2 documents
        assert context.count("###") <= 2

    @pytest.mark.asyncio
    async def test_get_context_no_match(self, org_culture_manager):
        """Test getting context with no matching documents."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Security",
            content="Security matters.",
            created_by="admin",
        )

        context = await org_culture_manager.get_relevant_context(
            org_id="org_001",
            task="xyz123",
        )

        assert context == ""

    @pytest.mark.asyncio
    async def test_get_context_format(self, org_culture_manager):
        """Test context output format."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Test Practice",
            content="This is a test practice.",
            created_by="admin",
        )

        context = await org_culture_manager.get_relevant_context(
            org_id="org_001",
            task="test",
        )

        assert "## Organizational Context" in context
        assert "### Test Practice (practices)" in context
        assert "This is a test practice." in context
