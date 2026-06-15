"""HTTP/WebSocket/Pulse client mock fixtures (pytest plugin)."""

from unittest.mock import AsyncMock, MagicMock

import pytest


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
        x = z3.Int("x")
        solver.add(x > 0)
        return solver.check() == z3.sat
    except ImportError:
        return False
    except Exception:
        return False


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
