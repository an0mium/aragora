"""
Shared mock store implementations for testing.

This module provides reusable mock store classes and factory functions
for common storage dependencies used across handler tests.

Usage:
    from tests.fixtures.shared.stores import (
        create_mock_server_context,
        create_mock_elo_system,
        MockMetaStore,
    )

    @pytest.fixture
    def server_context():
        return create_mock_server_context()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock


# ============================================================================
# Mock Debate Storage
# ============================================================================


def create_mock_debate_storage() -> MagicMock:
    """Create a mock DebateStorage for handler testing.

    Returns:
        Mock storage with common methods pre-configured.
    """
    storage = MagicMock()

    # List debates
    storage.list_debates.return_value = [
        {
            "id": "debate-001",
            "slug": "test-debate-one",
            "task": "Test task one",
            "created_at": "2026-01-15T10:00:00Z",
            "consensus_reached": False,
        },
        {
            "id": "debate-002",
            "slug": "test-debate-two",
            "task": "Test task two",
            "created_at": "2026-01-14T10:00:00Z",
            "consensus_reached": True,
        },
    ]

    # Get single debate
    storage.get_debate.return_value = {
        "id": "debate-001",
        "slug": "test-debate-one",
        "task": "Test task one",
        "messages": [
            {"agent": "claude", "content": "Initial proposal", "round": 1},
            {"agent": "gemini", "content": "Critique of proposal", "round": 1},
        ],
        "critiques": [],
        "votes": [],
        "consensus_reached": False,
        "rounds_used": 3,
        "created_at": "2026-01-15T10:00:00Z",
    }
    storage.get_debate_by_slug.return_value = storage.get_debate.return_value

    # Search
    storage.search.return_value = storage.list_debates.return_value

    # Public/private
    storage.is_public.return_value = True

    # Save operations
    storage.save_debate.return_value = "debate-new"
    storage.update_debate.return_value = True
    storage.delete_debate.return_value = True

    return storage


# ============================================================================
# Mock User Store
# ============================================================================


def create_mock_user_store() -> MagicMock:
    """Create a mock UserStore for handler testing.

    Returns:
        Mock user store with common authentication methods.
    """
    store = MagicMock()

    # User retrieval
    store.get_user.return_value = {
        "user_id": "user-001",
        "email": "test@example.com",
        "username": "testuser",
        "created_at": "2026-01-01T00:00:00Z",
        "is_active": True,
    }
    store.get_user_by_email.return_value = store.get_user.return_value
    store.get_user_by_username.return_value = store.get_user.return_value

    # Authentication
    store.verify_password.return_value = True
    store.create_session.return_value = "session-token-123"
    store.validate_session.return_value = store.get_user.return_value

    # User creation
    store.create_user.return_value = "user-new"
    store.update_user.return_value = True
    store.delete_user.return_value = True

    return store


# ============================================================================
# Mock ELO System
# ============================================================================


def create_mock_elo_system() -> MagicMock:
    """Create a mock ELO system for handler testing.

    Returns:
        Mock ELO system with leaderboard and rating methods.
    """
    elo = MagicMock()

    # Mock rating object
    mock_rating = MagicMock()
    mock_rating.agent_name = "claude"
    mock_rating.elo = 1650
    mock_rating.wins = 10
    mock_rating.losses = 5
    mock_rating.draws = 3
    mock_rating.games_played = 18
    mock_rating.win_rate = 0.56
    mock_rating.domain_elos = {}
    mock_rating.debates_count = 18
    mock_rating.critiques_accepted = 5
    mock_rating.critiques_total = 10

    # Methods
    elo.get_rating.return_value = mock_rating
    elo.get_leaderboard.return_value = [mock_rating]
    elo.get_cached_leaderboard.return_value = [
        {
            "agent_name": "claude",
            "elo": 1650,
            "wins": 10,
            "losses": 5,
            "draws": 3,
            "games_played": 18,
            "win_rate": 0.56,
        },
        {
            "agent_name": "gemini",
            "elo": 1580,
            "wins": 8,
            "losses": 7,
            "draws": 2,
            "games_played": 17,
            "win_rate": 0.47,
        },
    ]
    elo.get_recent_matches.return_value = []
    elo.get_cached_recent_matches.return_value = []
    elo.get_head_to_head.return_value = {
        "matches": 5,
        "agent_a_wins": 3,
        "agent_b_wins": 2,
        "draws": 0,
    }
    elo.get_stats.return_value = {
        "total_agents": 10,
        "total_matches": 50,
        "avg_elo": 1500,
    }
    elo.get_rivals.return_value = []
    elo.get_allies.return_value = []
    elo.record_match.return_value = None

    return elo


# ============================================================================
# Mock Knowledge Store
# ============================================================================


def create_mock_knowledge_store() -> MagicMock:
    """Create a mock KnowledgeStore for handler testing.

    Returns:
        Mock knowledge store with fact/mound operations.
    """
    store = MagicMock()

    # Facts
    store.list_facts.return_value = [
        {
            "fact_id": "fact-001",
            "content": "Test fact one",
            "source": "debate-001",
            "confidence": 0.95,
            "created_at": "2026-01-15T10:00:00Z",
        },
    ]
    store.get_fact.return_value = store.list_facts.return_value[0]
    store.add_fact.return_value = "fact-new"
    store.update_fact.return_value = True
    store.delete_fact.return_value = True

    # Search
    store.search_facts.return_value = store.list_facts.return_value

    return store


# ============================================================================
# Mock Workflow Store
# ============================================================================


def create_mock_workflow_store() -> MagicMock:
    """Create a mock WorkflowStore for handler testing."""
    store = MagicMock()

    store.list_workflows.return_value = [
        {
            "workflow_id": "wf-001",
            "name": "Test Workflow",
            "status": "active",
            "created_at": "2026-01-15T10:00:00Z",
        },
    ]
    store.get_workflow.return_value = store.list_workflows.return_value[0]
    store.create_workflow.return_value = "wf-new"
    store.update_workflow.return_value = True
    store.delete_workflow.return_value = True

    return store


# ============================================================================
# Mock Workspace Store
# ============================================================================


def create_mock_workspace_store() -> MagicMock:
    """Create a mock WorkspaceStore for handler testing."""
    store = MagicMock()

    store.list_workspaces.return_value = [
        {
            "workspace_id": "ws-001",
            "name": "Test Workspace",
            "owner_id": "user-001",
            "created_at": "2026-01-15T10:00:00Z",
        },
    ]
    store.get_workspace.return_value = store.list_workspaces.return_value[0]
    store.create_workspace.return_value = "ws-new"
    store.update_workspace.return_value = True
    store.delete_workspace.return_value = True

    return store


# ============================================================================
# Mock Audit Store
# ============================================================================


def create_mock_audit_store() -> MagicMock:
    """Create a mock AuditStore for handler testing."""
    store = MagicMock()

    store.list_events.return_value = [
        {
            "event_id": "evt-001",
            "event_type": "debate.created",
            "actor_id": "user-001",
            "resource_id": "debate-001",
            "timestamp": "2026-01-15T10:00:00Z",
        },
    ]
    store.get_event.return_value = store.list_events.return_value[0]
    store.log_event.return_value = "evt-new"

    return store


# ============================================================================
# Mock Critique Store
# ============================================================================


def create_mock_critique_store() -> MagicMock:
    """Create a mock CritiqueStore for handler testing."""
    store = MagicMock()

    store.get_critiques.return_value = [
        {
            "critique_id": "crit-001",
            "debate_id": "debate-001",
            "critic": "gemini",
            "target": "claude",
            "content": "The proposed solution lacks scalability.",
            "severity": "medium",
            "accepted": False,
        },
    ]
    store.add_critique.return_value = "crit-new"
    store.update_critique.return_value = True
    store.get_stats.return_value = {
        "total_critiques": 100,
        "accepted_count": 45,
        "acceptance_rate": 0.45,
    }

    return store


# ============================================================================
# Mock Server Context
# ============================================================================


def create_mock_server_context(
    storage: Optional[MagicMock] = None,
    user_store: Optional[MagicMock] = None,
    elo_system: Optional[MagicMock] = None,
    knowledge_store: Optional[MagicMock] = None,
    workflow_store: Optional[MagicMock] = None,
    workspace_store: Optional[MagicMock] = None,
    audit_store: Optional[MagicMock] = None,
    critique_store: Optional[MagicMock] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a complete server context with all mock dependencies.

    This provides the standard server_context dict used by handlers.

    Args:
        storage: Optional custom mock storage
        user_store: Optional custom mock user store
        elo_system: Optional custom mock ELO system
        knowledge_store: Optional custom mock knowledge store
        workflow_store: Optional custom mock workflow store
        workspace_store: Optional custom mock workspace store
        audit_store: Optional custom mock audit store
        critique_store: Optional custom mock critique store
        **kwargs: Additional context entries

    Returns:
        Complete server context dict
    """
    context = {
        "storage": storage or create_mock_debate_storage(),
        "user_store": user_store or create_mock_user_store(),
        "elo_system": elo_system or create_mock_elo_system(),
        "knowledge_store": knowledge_store or create_mock_knowledge_store(),
        "workflow_store": workflow_store or create_mock_workflow_store(),
        "workspace_store": workspace_store or create_mock_workspace_store(),
        "audit_store": audit_store or create_mock_audit_store(),
        "critique_store": critique_store or create_mock_critique_store(),
        "debate_embeddings": None,
        "nomic_dir": None,
    }
    context.update(kwargs)
    return context


# ============================================================================
# Additional Mock Classes
# ============================================================================


@dataclass
class MockMetaStore:
    """Mock meta store for knowledge management tests.

    Simulates a key-value store for metadata with in-memory storage.
    """

    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value by key."""
        self.data[key] = value

    def delete(self, key: str) -> bool:
        """Delete a key."""
        if key in self.data:
            del self.data[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data

    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (simplified, only supports *)."""
        if pattern == "*":
            return list(self.data.keys())
        # Simple prefix matching
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self.data.keys() if k.startswith(prefix)]
        return [k for k in self.data.keys() if k == pattern]

    def clear(self) -> None:
        """Clear all data."""
        self.data.clear()


@dataclass
class MockConnection:
    """Mock database connection for testing.

    Provides a simple mock connection interface.
    """

    connected: bool = True
    executed_queries: List[str] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)

    def execute(self, query: str, params: Optional[tuple] = None) -> "MockConnection":
        """Execute a query."""
        self.executed_queries.append(query)
        return self

    def fetchone(self) -> Optional[tuple]:
        """Fetch one result."""
        return self.results[0] if self.results else None

    def fetchall(self) -> List[tuple]:
        """Fetch all results."""
        return self.results

    def fetchmany(self, size: int = 100) -> List[tuple]:
        """Fetch many results."""
        return self.results[:size]

    def commit(self) -> None:
        """Commit transaction."""
        pass

    def rollback(self) -> None:
        """Rollback transaction."""
        pass

    def close(self) -> None:
        """Close connection."""
        self.connected = False


class MockSemanticStore:
    """Mock semantic store for vector similarity tests.

    Provides basic vector storage and search simulation.
    """

    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector with optional metadata."""
        self.vectors[id] = vector
        self.metadata[id] = metadata or {}

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get vector and metadata by ID."""
        if id not in self.vectors:
            return None
        return {
            "id": id,
            "vector": self.vectors[id],
            "metadata": self.metadata[id],
        }

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors (returns mock results)."""
        # Return first k items as mock results with mock scores
        results = []
        for i, (id, vec) in enumerate(list(self.vectors.items())[:k]):
            results.append(
                {
                    "id": id,
                    "score": 0.95 - (i * 0.05),  # Decreasing mock scores
                    "metadata": self.metadata[id],
                }
            )
        return results

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        if id in self.vectors:
            del self.vectors[id]
            del self.metadata[id]
            return True
        return False

    def clear(self) -> None:
        """Clear all vectors."""
        self.vectors.clear()
        self.metadata.clear()

    def count(self) -> int:
        """Get total count of vectors."""
        return len(self.vectors)


# ============================================================================
# Mock Calibration Tracker
# ============================================================================


def create_mock_calibration_tracker() -> MagicMock:
    """Create a mock CalibrationTracker for handler testing.

    Returns:
        Mock calibration tracker with calibration methods.
    """
    tracker = MagicMock()

    # Mock calibration summary
    mock_summary = MagicMock()
    mock_summary.agent = "test_agent"
    mock_summary.total_predictions = 100
    mock_summary.total_correct = 75
    mock_summary.brier_score = 0.15
    mock_summary.ece = 0.08
    mock_summary.adjust_confidence = MagicMock(side_effect=lambda c, domain=None: c)

    # Configure methods
    tracker.get_calibration_summary.return_value = mock_summary
    tracker.get_brier_score.return_value = 0.15
    tracker.get_expected_calibration_error.return_value = 0.08
    tracker.get_calibration_curve.return_value = []
    tracker.get_all_agents.return_value = ["test_agent"]
    tracker.record_prediction = MagicMock()
    tracker.record_outcome = MagicMock()
    tracker.get_temperature_params.return_value = MagicMock(
        temperature=1.0, get_temperature=MagicMock(return_value=1.0)
    )

    return tracker


# ============================================================================
# Mock Knowledge Mound
# ============================================================================


def create_mock_knowledge_mound() -> MagicMock:
    """Create a mock KnowledgeMound for handler testing.

    Returns:
        Mock knowledge mound with core operations.
    """
    mound = MagicMock()

    # Mock fact objects
    mock_fact = MagicMock()
    mock_fact.fact_id = "fact-001"
    mock_fact.content = "Test fact content"
    mock_fact.source = "debate-001"
    mock_fact.confidence = 0.95
    mock_fact.metadata = {}

    # Core operations
    mound.add_fact.return_value = "fact-new"
    mound.get_fact.return_value = mock_fact
    mound.search.return_value = [mock_fact]
    mound.delete_fact.return_value = True
    mound.update_fact.return_value = True

    # Stats
    mound.get_stats.return_value = {
        "total_facts": 100,
        "sources": 10,
        "avg_confidence": 0.85,
    }

    return mound


# ============================================================================
# HTTP Handler Mocks (for handler testing)
# ============================================================================


def create_mock_http_handler(
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    client_address: tuple = ("127.0.0.1", 12345),
) -> MagicMock:
    """Create a mock HTTP handler for handler testing.

    Args:
        method: HTTP method
        body: Request body as dict
        headers: Request headers
        client_address: Client address tuple

    Returns:
        Mock HTTP handler
    """
    import json

    mock = MagicMock()
    mock.command = method

    # Set up body reading
    if body is not None:
        body_bytes = json.dumps(body).encode()
    else:
        body_bytes = b"{}"

    mock.rfile = MagicMock()
    mock.rfile.read = MagicMock(return_value=body_bytes)

    # Set up headers
    mock.headers = headers or {}
    mock.headers.setdefault("Content-Length", str(len(body_bytes)))

    # Set up client address
    mock.client_address = client_address

    return mock
