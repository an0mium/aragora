"""
Shared pytest fixtures for Aragora test suite.

This module provides common fixtures used across multiple test files,
reducing duplication and ensuring consistent test setup.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Generator
from unittest.mock import Mock, MagicMock, AsyncMock

import pytest

from aragora.resilience import reset_all_circuit_breakers
from tests.utils import managed_fixture

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem
    from aragora.memory.continuum import ContinuumMemory


# ============================================================================
# Test Tier Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom pytest markers for test tiers.

    Test Tiers:
    - smoke: Quick sanity tests for CI (<5 min total)
    - integration: Tests requiring external dependencies (APIs, DBs)
    - slow: Long-running tests (>30s each)

    CI Strategy:
    - PR CI: pytest -m "not slow and not integration" (~5 min)
    - Nightly: pytest (full suite)

    Usage:
        @pytest.mark.smoke
        def test_basic_import():
            ...

        @pytest.mark.slow
        def test_full_debate_with_all_agents():
            ...

        @pytest.mark.integration
        def test_supabase_connection():
            ...
    """
    config.addinivalue_line(
        "markers", "smoke: quick sanity tests for fast CI feedback"
    )
    config.addinivalue_line(
        "markers", "integration: tests requiring external dependencies (APIs, databases)"
    )
    config.addinivalue_line(
        "markers", "slow: long-running tests (>30 seconds)"
    )
    config.addinivalue_line(
        "markers", "unit: isolated unit tests with no external dependencies"
    )


# ============================================================================
# Global Test Setup
# ============================================================================


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset all circuit breakers before each test.

    This ensures tests don't affect each other through shared circuit breaker state.
    Auto-used so every test gets a clean circuit breaker state.
    """
    reset_all_circuit_breakers()
    yield
    # Also reset after test to ensure clean state for next test
    reset_all_circuit_breakers()


@pytest.fixture(autouse=True)
def clear_handler_cache():
    """Clear the handler cache before and after each test.

    This prevents test pollution from cached responses in handlers
    that use @ttl_cache decorator.
    """
    try:
        from aragora.server.handlers.base import clear_cache
        clear_cache()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.handlers.base import clear_cache
        clear_cache()
    except ImportError:
        pass


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
def elo_system(temp_db) -> Generator["EloSystem", None, None]:
    """Create a real EloSystem with a temporary database.

    Yields an EloSystem instance backed by a temp database.
    The database connection is properly closed after the test.
    """
    from aragora.ranking.elo import EloSystem

    system = EloSystem(db_path=temp_db)
    with managed_fixture(system, name="EloSystem"):
        yield system


@pytest.fixture
def continuum_memory(temp_db) -> Generator["ContinuumMemory", None, None]:
    """Create a real ContinuumMemory with a temporary database.

    Yields a ContinuumMemory instance backed by a temp database.
    The database connection is properly closed after the test.
    """
    from aragora.memory.continuum import ContinuumMemory

    memory = ContinuumMemory(db_path=temp_db)
    with managed_fixture(memory, name="ContinuumMemory"):
        yield memory


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
        "SUPABASE_URL",
        "SUPABASE_KEY",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture(autouse=True)
def reset_supabase_env(monkeypatch):
    """Reset Supabase environment variables between tests.

    This prevents test pollution where earlier tests set SUPABASE_URL/KEY
    that affect later tests expecting unconfigured clients.
    """
    # Clear Supabase env vars to ensure clean state
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    yield


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

def _reset_lazy_globals_impl():
    """Implementation of lazy globals reset.

    Extracted to allow calling before AND after tests.
    """
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

    # Clear DatabaseManager singleton instances
    try:
        from aragora.storage.schema import DatabaseManager
        DatabaseManager.clear_instances()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def reset_lazy_globals():
    """Reset lazy-loaded globals BEFORE and AFTER each test.

    This fixture prevents test pollution from global state that persists
    between tests. Running reset both before AND after ensures:
    1. Each test starts with clean state
    2. If a test hangs/times out, the next test still gets clean state

    Affected modules:
    - aragora.debate.orchestrator (9 globals)
    - aragora.server.handlers.* (2-4 globals each)
    - aragora.storage.schema.DatabaseManager (singleton cache)
    """
    _reset_lazy_globals_impl()  # Reset BEFORE test
    yield
    _reset_lazy_globals_impl()  # Reset AFTER test


# ============================================================================
# API Response Mocking Fixtures
# ============================================================================

@pytest.fixture
def mock_anthropic_response():
    """Create mock Anthropic API response.

    Returns a factory function that creates mock responses.
    Use with `unittest.mock.patch` to mock httpx or requests calls.

    Example:
        def test_anthropic_call(mock_anthropic_response):
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_post.return_value = mock_anthropic_response("Hello!")
                # ... test code
    """
    def _make_response(
        content: str = "Test response",
        model: str = "claude-sonnet-4-20250514",
        stop_reason: str = "end_turn",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
            "model": model,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp
    return _make_response


@pytest.fixture
def mock_openai_response():
    """Create mock OpenAI API response.

    Returns a factory function that creates mock responses.

    Example:
        def test_openai_call(mock_openai_response):
            with patch('openai.AsyncOpenAI') as mock_client:
                mock_client.return_value.chat.completions.create = AsyncMock(
                    return_value=mock_openai_response("Hello!")
                )
    """
    def _make_response(
        content: str = "Test response",
        model: str = "gpt-4o",
        finish_reason: str = "stop",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_choice.message.role = "assistant"
        mock_choice.finish_reason = finish_reason
        mock_choice.index = 0

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = prompt_tokens
        mock_usage.completion_tokens = completion_tokens
        mock_usage.total_tokens = prompt_tokens + completion_tokens

        mock_resp = MagicMock()
        mock_resp.id = "chatcmpl-test123"
        mock_resp.model = model
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        mock_resp.created = 1700000000

        return mock_resp
    return _make_response


@pytest.fixture
def mock_openrouter_response():
    """Create mock OpenRouter API response.

    OpenRouter uses OpenAI-compatible format.
    """
    def _make_response(
        content: str = "Test response",
        model: str = "anthropic/claude-3.5-sonnet",
        finish_reason: str = "stop",
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "gen-test123",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp
    return _make_response


@pytest.fixture
def mock_streaming_response():
    """Create mock streaming API response (SSE format).

    Returns a factory that creates an async generator for streaming responses.
    """
    def _make_stream(chunks: list[str] | None = None):
        if chunks is None:
            chunks = ["Hello", " world", "!"]

        async def _stream():
            for i, chunk in enumerate(chunks):
                yield {
                    "id": f"chunk-{i}",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None if i < len(chunks) - 1 else "stop",
                    }],
                }

        return _stream()
    return _make_stream


# ============================================================================
# Z3/Formal Verification Fixtures
# ============================================================================

@pytest.fixture
def z3_available() -> bool:
    """Check if Z3 solver is available.

    Returns True if Z3 can be imported and used.
    Use with pytest.mark.skipif for Z3-dependent tests.

    Example:
        @pytest.mark.skipif(not z3_available(), reason="Z3 not installed")
        def test_z3_proof(z3_available):
            ...
    """
    try:
        import z3
        # Quick sanity check that Z3 actually works
        solver = z3.Solver()
        x = z3.Int('x')
        solver.add(x > 0)
        return solver.check() == z3.sat
    except ImportError:
        return False
    except Exception:
        return False


# Helper function for use in skipif decorators
def _z3_installed() -> bool:
    """Check if Z3 is installed (for use in decorators)."""
    try:
        import z3
        return True
    except ImportError:
        return False


# Make this available at module level for skipif decorators
Z3_AVAILABLE = _z3_installed()


# ============================================================================
# HTTP Client Mocking Fixtures
# ============================================================================

@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient.

    Returns a configured mock client for HTTP request testing.
    """
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp.ClientSession.

    Returns a configured mock session for async HTTP testing.
    """
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    # Mock response context manager
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={})
    mock_response.text = AsyncMock(return_value="")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    session.get = MagicMock(return_value=mock_response)
    session.post = MagicMock(return_value=mock_response)

    return session


# ============================================================================
# Pulse/Trending Fixtures
# ============================================================================

@pytest.fixture
def mock_pulse_topics():
    """Create sample trending topics for Pulse tests.

    Returns a list of mock TrendingTopic-like dicts.
    """
    return [
        {
            "topic": "AI Safety Debate",
            "platform": "hackernews",
            "category": "tech",
            "volume": 500,
            "controversy_score": 0.8,
            "timestamp": "2026-01-12T00:00:00Z",
        },
        {
            "topic": "Climate Policy",
            "platform": "reddit",
            "category": "politics",
            "volume": 350,
            "controversy_score": 0.7,
            "timestamp": "2026-01-12T01:00:00Z",
        },
        {
            "topic": "Cryptocurrency Regulation",
            "platform": "twitter",
            "category": "finance",
            "volume": 200,
            "controversy_score": 0.6,
            "timestamp": "2026-01-12T02:00:00Z",
        },
    ]


@pytest.fixture
def mock_pulse_manager(mock_pulse_topics):
    """Create a mock PulseManager for scheduler tests.

    Returns a MagicMock with common PulseManager methods configured.
    """
    manager = MagicMock()
    manager.get_trending_topics = AsyncMock(return_value=mock_pulse_topics)
    manager.get_topic_history = AsyncMock(return_value=[])
    manager.refresh_topics = AsyncMock(return_value=None)
    return manager


# ============================================================================
# WebSocket Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection.

    Returns a MagicMock configured for WebSocket testing.
    """
    ws = MagicMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_json = AsyncMock(return_value={})
    ws.receive_text = AsyncMock(return_value="")
    ws.close = AsyncMock()
    ws.accept = AsyncMock()

    # Track sent messages for assertions
    ws.sent_messages = []

    async def track_send(data):
        ws.sent_messages.append(data)

    ws.send_json.side_effect = track_send

    return ws
