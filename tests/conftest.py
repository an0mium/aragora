"""
Shared pytest fixtures for Aragora test suite.

This module provides common fixtures used across multiple test files,
reducing duplication and ensuring consistent test setup.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, MagicMock

import pytest


# ============================================================================
# Temporary File/Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary SQLite database file.

    Yields the path to a temporary .db file that is automatically
    cleaned up after the test completes.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory.

    Yields a Path to a temporary directory that is automatically
    cleaned up after the test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_nomic_dir() -> Generator[Path, None, None]:
    """Create a temporary nomic directory with state files.

    Creates a directory structure mimicking the nomic system:
    - nomic_state.json: Current nomic state
    - nomic_loop.log: Recent log entries

    Yields a Path to the directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)

        # Create nomic state file
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({
            "phase": "implement",
            "stage": "executing",
            "cycle": 1,
            "total_tasks": 5,
            "completed_tasks": 2,
        }))

        # Create nomic log file
        log_file = nomic_dir / "nomic_loop.log"
        log_file.write_text("\n".join([
            "2026-01-05 00:00:01 Starting cycle 1",
            "2026-01-05 00:00:02 Phase: context",
            "2026-01-05 00:00:03 Phase: debate",
            "2026-01-05 00:00:04 Phase: design",
            "2026-01-05 00:00:05 Phase: implement",
        ]))

        yield nomic_dir


# ============================================================================
# Mock Storage Fixtures
# ============================================================================

@pytest.fixture
def mock_storage() -> Mock:
    """Create a mock DebateStorage.

    Returns a Mock object with common storage methods pre-configured
    with sensible return values.
    """
    storage = Mock()
    storage.list_debates.return_value = [
        {
            "id": "debate-1",
            "slug": "test-debate",
            "task": "Test task",
            "created_at": "2026-01-05",
        },
        {
            "id": "debate-2",
            "slug": "another-debate",
            "task": "Another task",
            "created_at": "2026-01-04",
        },
    ]
    storage.get_debate.return_value = {
        "id": "debate-1",
        "slug": "test-debate",
        "task": "Test task",
        "messages": [{"agent": "claude", "content": "Hello"}],
        "critiques": [],
        "consensus_reached": False,
        "rounds_used": 3,
    }
    storage.get_debate_by_slug.return_value = storage.get_debate.return_value
    return storage


@pytest.fixture
def mock_elo_system() -> Mock:
    """Create a mock EloSystem.

    Returns a Mock object with common ELO system methods pre-configured.
    """
    elo = Mock()

    # Mock agent rating
    mock_rating = Mock()
    mock_rating.agent_name = "test_agent"
    mock_rating.elo = 1500
    mock_rating.wins = 5
    mock_rating.losses = 3
    mock_rating.draws = 2
    mock_rating.games_played = 10
    mock_rating.win_rate = 0.5
    mock_rating.domain_elos = {}
    mock_rating.debates_count = 10
    mock_rating.critiques_accepted = 5
    mock_rating.critiques_total = 10

    elo.get_rating.return_value = mock_rating
    elo.get_leaderboard.return_value = [mock_rating]
    elo.get_cached_leaderboard.return_value = [{
        "agent_name": "test_agent",
        "elo": 1500,
        "wins": 5,
        "losses": 3,
        "draws": 2,
        "games_played": 10,
        "win_rate": 0.5,
    }]
    elo.get_recent_matches.return_value = []
    elo.get_cached_recent_matches.return_value = []
    elo.get_head_to_head.return_value = {
        "matches": 5,
        "agent_a_wins": 2,
        "agent_b_wins": 2,
        "draws": 1,
    }
    elo.get_stats.return_value = {
        "total_agents": 10,
        "total_matches": 50,
        "avg_elo": 1500,
    }
    elo.get_rivals.return_value = []
    elo.get_allies.return_value = []

    return elo


# ============================================================================
# Mock Agent Fixtures
# ============================================================================

@pytest.fixture
def mock_agent() -> Mock:
    """Create a mock Agent.

    Returns a Mock object representing a debate agent.
    """
    agent = Mock()
    agent.name = "test_agent"
    agent.role = "proposer"
    agent.model = "claude-3-opus"

    async def mock_generate(*args, **kwargs):
        return "This is a test response from the agent."

    agent.generate = mock_generate
    return agent


@pytest.fixture
def mock_agents() -> list[Mock]:
    """Create a list of mock agents for multi-agent tests.

    Returns a list of 3 mock agents with different names.
    """
    agents = []
    for i, name in enumerate(["claude", "gemini", "gpt4"]):
        agent = Mock()
        agent.name = name
        agent.role = "proposer" if i == 0 else "critic"
        agent.model = f"model-{name}"
        agents.append(agent)
    return agents


# ============================================================================
# Mock Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_environment() -> Mock:
    """Create a mock Environment for arena testing.

    Returns a Mock object with environment properties.
    """
    env = Mock()
    env.task = "Test debate task"
    env.context = ""
    env.max_rounds = 5
    return env


# ============================================================================
# Event Emitter Fixtures
# ============================================================================

@pytest.fixture
def mock_emitter() -> Mock:
    """Create a mock event emitter.

    Returns a Mock object that can be used as an event emitter.
    """
    emitter = Mock()
    emitter.emit = Mock()
    emitter.subscribe = Mock()
    emitter.unsubscribe = Mock()
    return emitter


# ============================================================================
# Auth Fixtures
# ============================================================================

@pytest.fixture
def mock_auth_config() -> Mock:
    """Create a mock AuthConfig.

    Returns a Mock configured for authentication testing.
    """
    from aragora.server.auth import AuthConfig

    config = AuthConfig()
    config.api_token = "test_secret_key_12345"
    config.enabled = True
    config.rate_limit_per_minute = 60
    config.ip_rate_limit_per_minute = 120
    return config


# ============================================================================
# Handler Context Fixtures
# ============================================================================

@pytest.fixture
def handler_context(mock_storage, mock_elo_system, temp_nomic_dir) -> dict:
    """Create a complete handler context.

    Returns a dict with all common handler dependencies configured.
    """
    return {
        "storage": mock_storage,
        "elo_system": mock_elo_system,
        "nomic_dir": temp_nomic_dir,
        "debate_embeddings": None,
        "critique_store": None,
    }


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
def event_loop_policy():
    """Configure event loop policy for async tests.

    This fixture ensures consistent async behavior across platforms.
    """
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def elo_system(temp_db) -> "EloSystem":
    """Create a real EloSystem with a temporary database.

    Returns an EloSystem instance backed by a temp database
    that is cleaned up after the test.
    """
    from aragora.ranking.elo import EloSystem
    return EloSystem(db_path=temp_db)


@pytest.fixture
def continuum_memory(temp_db) -> "ContinuumMemory":
    """Create a real ContinuumMemory with a temporary database.

    Returns a ContinuumMemory instance backed by a temp database
    that is cleaned up after the test.
    """
    from aragora.memory.continuum import ContinuumMemory
    return ContinuumMemory(db_path=temp_db)


# ============================================================================
# Environment Variable Fixtures
# ============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """Clear API key environment variables for testing.

    Use this fixture when testing code that checks for API keys
    to ensure consistent behavior.
    """
    env_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ARAGORA_API_TOKEN",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Set mock API keys for testing.

    Use this fixture when testing code that requires API keys
    but shouldn't make real API calls.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    return monkeypatch


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_debate_messages() -> list[dict]:
    """Return sample debate messages for testing."""
    return [
        {
            "agent": "claude",
            "role": "proposer",
            "content": "I propose that we should implement feature X.",
            "round": 1,
        },
        {
            "agent": "gemini",
            "role": "critic",
            "content": "I have concerns about the scalability of feature X.",
            "round": 1,
        },
        {
            "agent": "claude",
            "role": "proposer",
            "content": "Addressing your concerns, we can add caching.",
            "round": 2,
        },
    ]


@pytest.fixture
def sample_critique() -> dict:
    """Return a sample critique for testing."""
    return {
        "critic": "gemini",
        "target": "claude",
        "content": "The proposed solution doesn't address edge cases.",
        "severity": "medium",
        "accepted": False,
    }


# ============================================================================
# Global State Reset Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_lazy_globals():
    """Reset lazy-loaded globals between tests.

    This fixture prevents test pollution from global state that persists
    between tests. It runs automatically after each test (autouse=True).

    Affected modules:
    - aragora.debate.orchestrator (9 globals)
    - aragora.server.handlers.* (2-4 globals each)
    """
    yield  # Run the test first

    # Reset orchestrator globals
    try:
        import aragora.debate.orchestrator as orch
        orch.PositionTracker = None
        orch.CalibrationTracker = None
        orch.InsightExtractor = None
        orch.InsightStore = None
        orch.CitationExtractor = None
        orch.BeliefNetwork = None
        orch.BeliefPropagationAnalyzer = None
        orch.CritiqueStore = None
        orch.ArgumentCartographer = None
    except (ImportError, AttributeError):
        pass

    # Reset handler globals (belief)
    try:
        import aragora.server.handlers.belief as belief_handler
        if hasattr(belief_handler, 'BeliefNetwork'):
            belief_handler.BeliefNetwork = None
        if hasattr(belief_handler, 'BeliefPropagationAnalyzer'):
            belief_handler.BeliefPropagationAnalyzer = None
        if hasattr(belief_handler, 'PersonaLaboratory'):
            belief_handler.PersonaLaboratory = None
        if hasattr(belief_handler, 'ProvenanceTracker'):
            belief_handler.ProvenanceTracker = None
    except (ImportError, AttributeError):
        pass

    # Reset handler globals (consensus)
    try:
        import aragora.server.handlers.consensus as consensus_handler
        if hasattr(consensus_handler, 'ConsensusMemory'):
            consensus_handler.ConsensusMemory = None
        if hasattr(consensus_handler, 'DissentRetriever'):
            consensus_handler.DissentRetriever = None
    except (ImportError, AttributeError):
        pass

    # Reset handler globals (critique)
    try:
        import aragora.server.handlers.critique as critique_handler
        if hasattr(critique_handler, 'CritiqueStore'):
            critique_handler.CritiqueStore = None
    except (ImportError, AttributeError):
        pass

    # Reset handler globals (calibration)
    try:
        import aragora.server.handlers.calibration as cal_handler
        if hasattr(cal_handler, 'CalibrationTracker'):
            cal_handler.CalibrationTracker = None
        if hasattr(cal_handler, 'EloSystem'):
            cal_handler.EloSystem = None
    except (ImportError, AttributeError):
        pass
