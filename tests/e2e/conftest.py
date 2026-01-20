"""
Shared fixtures for E2E tests.

Provides:
- Server fixtures (test client, async client)
- Database fixtures (test database, cleanup)
- Tenant fixtures (test tenants, isolation verification)
- Connector fixtures (mock external services)
- Debate fixtures (agent mocks, debate setup)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class E2EConfig:
    """E2E test configuration."""

    test_port: int = 18080
    test_host: str = "127.0.0.1"
    use_temp_db: bool = True
    db_url: Optional[str] = None
    request_timeout: float = 30.0
    debate_timeout: float = 120.0
    sync_timeout: float = 60.0
    mock_external_apis: bool = True
    mock_llm_responses: bool = True


@pytest.fixture(scope="session")
def e2e_config() -> E2EConfig:
    """Provide E2E configuration."""
    return E2EConfig(
        test_port=int(os.environ.get("E2E_TEST_PORT", "18080")),
        mock_external_apis=os.environ.get("E2E_REAL_APIS", "false").lower() != "true",
        mock_llm_responses=os.environ.get("E2E_REAL_LLM", "false").lower() != "true",
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Test Client
# ============================================================================


@dataclass
class TestClient:
    """HTTP test client for E2E tests."""

    base_url: str

    async def get(self, path: str, headers: Optional[Dict[str, str]] = None,
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{path}"
            async with session.get(url, headers=headers, params=params) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "json": await response.json() if response.content_type == "application/json" else None,
                    "text": await response.text(),
                }

    async def post(self, path: str, json: Optional[Dict[str, Any]] = None,
                   headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{path}"
            async with session.post(url, json=json, headers=headers) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "json": await response.json() if response.content_type == "application/json" else None,
                    "text": await response.text(),
                }

    async def put(self, path: str, json: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{path}"
            async with session.put(url, json=json, headers=headers) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "json": await response.json() if response.content_type == "application/json" else None,
                    "text": await response.text(),
                }

    async def delete(self, path: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{path}"
            async with session.delete(url, headers=headers) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "json": await response.json() if response.content_type == "application/json" else None,
                    "text": await response.text(),
                }


@pytest_asyncio.fixture
async def test_client(e2e_config: E2EConfig) -> AsyncGenerator[TestClient, None]:
    """Provide test HTTP client."""
    client = TestClient(base_url=f"http://{e2e_config.test_host}:{e2e_config.test_port}")
    yield client


# ============================================================================
# Tenant Fixtures
# ============================================================================


@dataclass
class TestTenant:
    """Test tenant for isolation testing."""

    id: str
    name: str
    api_key: str
    tier: str = "standard"

    @property
    def auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "X-Tenant-ID": self.id}


@pytest.fixture
def tenant_a() -> TestTenant:
    return TestTenant(
        id=f"tenant-a-{uuid.uuid4().hex[:8]}",
        name="Acme Corp",
        api_key=f"ak_test_a_{uuid.uuid4().hex}",
        tier="enterprise",
    )


@pytest.fixture
def tenant_b() -> TestTenant:
    return TestTenant(
        id=f"tenant-b-{uuid.uuid4().hex[:8]}",
        name="Globex Inc",
        api_key=f"ak_test_b_{uuid.uuid4().hex}",
        tier="standard",
    )


@pytest.fixture
def isolated_tenants(tenant_a: TestTenant, tenant_b: TestTenant) -> List[TestTenant]:
    return [tenant_a, tenant_b]


# ============================================================================
# Connector Fixtures
# ============================================================================


@pytest.fixture
def mock_github_api():
    responses = {
        "repos": [{"id": 1, "name": "repo-1", "full_name": "org/repo-1"}],
        "contents": [{"name": "README.md", "path": "README.md", "type": "file"}],
    }
    with patch("aragora.connectors.enterprise.git.github.GitHubClient") as mock:
        instance = MagicMock()
        instance.get_repos = AsyncMock(return_value=responses["repos"])
        instance.get_contents = AsyncMock(return_value=responses["contents"])
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_slack_api():
    responses = {
        "channels": [{"id": "C001", "name": "general", "is_channel": True}],
        "messages": [{"ts": "1234567890.123456", "text": "Hello!", "user": "U001"}],
    }
    with patch("aragora.connectors.enterprise.collaboration.slack.SlackClient") as mock:
        instance = MagicMock()
        instance.list_channels = AsyncMock(return_value=responses["channels"])
        instance.get_messages = AsyncMock(return_value=responses["messages"])
        mock.return_value = instance
        yield mock


# ============================================================================
# Agent Fixtures
# ============================================================================


@dataclass
class MockAgentResponse:
    content: str
    model: str = "mock-model"
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 100, "completion_tokens": 50})


@pytest.fixture
def mock_llm_agents():
    """Provide mock LLM responses for tests that need them.

    Note: Most tests create their own mock agents inline.
    This fixture provides standard mock responses when needed.
    """
    responses = [
        MockAgentResponse(content="Thoughtful response to the debate topic.", model="claude-3-opus"),
        MockAgentResponse(content="Counter-argument with evidence.", model="gpt-4"),
        MockAgentResponse(content="Finding common ground.", model="gemini-pro"),
    ]
    response_idx = [0]

    def get_next(*args, **kwargs):
        idx = response_idx[0] % len(responses)
        response_idx[0] += 1
        return responses[idx]

    # Yield the response generator - tests create their own agent mocks
    yield {"responses": responses, "get_next": get_next}


# ============================================================================
# Debate Fixtures
# ============================================================================


@dataclass
class DebateSetup:
    topic: str
    agents: List[str]
    rounds: int
    protocol: str = "structured"
    consensus_threshold: float = 0.7

    def to_config(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "agents": self.agents,
            "protocol": {"type": self.protocol, "rounds": self.rounds,
                        "consensus": {"type": "majority", "threshold": self.consensus_threshold}},
        }


@pytest.fixture
def basic_debate() -> DebateSetup:
    return DebateSetup(
        topic="Should we adopt microservices architecture?",
        agents=["claude", "gpt4", "gemini"],
        rounds=3,
    )


@pytest.fixture
def extended_debate() -> DebateSetup:
    return DebateSetup(
        topic="Optimal approach to distributed system design?",
        agents=["claude", "gpt4", "gemini", "mistral"],
        rounds=55,
        protocol="extended",
        consensus_threshold=0.8,
    )


# ============================================================================
# Knowledge Fixtures
# ============================================================================


@pytest.fixture
def sample_facts() -> List[Dict[str, Any]]:
    return [
        {"content": "Python is a programming language.", "source": "e2e_test", "confidence": 0.99},
        {"content": "Machine learning requires data.", "source": "e2e_test", "confidence": 0.95},
        {"content": "APIs enable system integration.", "source": "e2e_test", "confidence": 0.97},
    ]


# ============================================================================
# Utilities
# ============================================================================


@pytest.fixture
def unique_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def assert_timing(max_seconds: float):
    import time
    start = time.monotonic()
    yield
    elapsed = time.monotonic() - start
    assert elapsed <= max_seconds, f"Operation took {elapsed:.2f}s, expected <= {max_seconds}s"


@pytest.fixture
def timing_context():
    return assert_timing
