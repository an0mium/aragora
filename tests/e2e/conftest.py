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
from typing import Any, Optional
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Import live server fixtures for use by tests
from tests.e2e.server_fixture import (  # noqa: F401
    LiveServerInfo,
    find_free_port,
    http_client,
    live_server,
)

pytestmark = pytest.mark.e2e

# Note: pytest-asyncio configured in pyproject.toml with asyncio_mode = "auto"


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


# ============================================================================
# RBAC Bypass for E2E Tests
# ============================================================================


@pytest.fixture(autouse=True)
def _bypass_rbac_for_e2e_tests(request, monkeypatch):
    """Auto-bypass RBAC permission checks in E2E tests.

    E2E tests call handler functions directly with mocked bridges.
    These handlers use @require_permission decorators that need an
    AuthorizationContext. This fixture provides a permissive context
    so the tests can exercise the business logic.
    """
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    try:
        from aragora.rbac import decorators
        from aragora.rbac.models import AuthorizationContext

        mock_auth_ctx = AuthorizationContext(
            user_id="e2e-test-user",
            org_id="e2e-test-org",
            roles={"admin", "owner"},
            permissions={"*"},
        )

        original_get_context = decorators._get_context_from_args

        def patched_get_context(args, kwargs, context_param):
            result = original_get_context(args, kwargs, context_param)
            if result is None:
                return mock_auth_ctx
            return result

        monkeypatch.setattr(decorators, "_get_context_from_args", patched_get_context)
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.rbac.checker import get_permission_checker
        from aragora.rbac.models import AuthorizationDecision

        checker = get_permission_checker()

        def _always_allow(context, permission_key, resource_id=None):
            return AuthorizationDecision(
                allowed=True,
                reason="E2E test bypass",
                permission_key=permission_key,
            )

        monkeypatch.setattr(checker, "check_permission", _always_allow)
    except (ImportError, AttributeError):
        pass

    yield


# ============================================================================
# Debate Isolation Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _isolate_e2e_databases(tmp_path, monkeypatch):
    """Isolate SQLite databases to a temp directory for each test.

    Arena initialization creates CalibrationTracker and other stores that
    open real SQLite database files.  If those files are locked by another
    process (e.g. the dev server), tests block indefinitely on the WAL
    mutex.  Pointing ARAGORA_DATA_DIR at a fresh tmp directory avoids
    contention entirely.

    Also forces the Jaccard similarity backend to prevent
    SentenceTransformer model downloads from HuggingFace, which can hang
    in CI or air-gapped environments.
    """
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ARAGORA_CONVERGENCE_BACKEND", "jaccard")
    monkeypatch.setenv("ARAGORA_SIMILARITY_BACKEND", "jaccard")
    monkeypatch.setenv("ARAGORA_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _disable_post_debate_external_calls(monkeypatch):
    """Disable post-debate pipeline steps that make external calls.

    The DEFAULT_POST_DEBATE_CONFIG enables gauntlet validation, explanation
    building, plan creation, and other steps that call _run_async_callable()
    which starts threads making real HTTP calls. In tests without real API
    keys, these threads block indefinitely.
    """
    try:
        import aragora.debate.post_debate_coordinator as pdc_mod

        patched = pdc_mod.PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_create_pr=False,
            auto_build_integrity_package=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_verify_arguments=False,
            auto_push_calibration=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            auto_settlement_tracking=False,
            auto_llm_judge=False,
        )
        monkeypatch.setattr(pdc_mod, "DEFAULT_POST_DEBATE_CONFIG", patched)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _mock_scan_code_markers(monkeypatch):
    """Prevent scan_code_markers from walking the entire repo.

    MetaPlanner.prioritize_work() -> NextStepsRunner.scan() ->
    scan_code_markers() does os.walk on up to 5000 files.
    This causes timeouts in long suite runs.
    """
    try:
        import aragora.compat.openclaw.next_steps_runner as nsr_mod

        monkeypatch.setattr(nsr_mod, "scan_code_markers", lambda repo_path: ([], 0))
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _mock_research_phase(monkeypatch):
    """Mock the pre-debate research phase to avoid real API calls.

    Arena's context_init phase calls research_for_debate() which uses
    Claude's web_search tool or Anthropic API for background research.
    Without mocking, this makes real HTTP calls that block indefinitely
    in test environments without API keys or network access.
    """

    async def _fake_research_for_debate(question: str) -> str:
        return ""

    async def _fake_research_question(
        question: str,
        summarize: bool = True,
        max_results: int = 5,
        force_research: bool = True,
    ):
        return None

    try:
        import aragora.server.research_phase as rp_mod

        monkeypatch.setattr(rp_mod, "research_for_debate", _fake_research_for_debate)
        monkeypatch.setattr(rp_mod, "research_question", _fake_research_question)
    except ImportError:
        pass

    # Also mock the context gatherer source that calls research_for_debate
    try:
        import aragora.debate.context_gatherer.sources as sources_mod

        monkeypatch.setattr(
            sources_mod,
            "research_for_debate",
            _fake_research_for_debate,
        )
    except (ImportError, AttributeError):
        pass

    try:
        import aragora.debate.context.sources as ctx_sources_mod

        monkeypatch.setattr(
            ctx_sources_mod,
            "research_for_debate",
            _fake_research_for_debate,
        )
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _mock_llm_synthesis(monkeypatch):
    """Mock LLM-based synthesis to avoid real API calls during debates.

    After debate rounds, the synthesis_generator creates an AnthropicAPIAgent
    or falls back to OpenRouter to generate a final synthesis. Without mocking,
    this makes real HTTP calls that block indefinitely.

    Also prevents the OpenRouter fallback agent creation that triggers on
    quota/rate-limit errors.
    """
    try:
        import aragora.agents.fallback as fallback_mod

        original_create = fallback_mod.create_openrouter_fallback

        def _mock_create_fallback(agent_name, model=None, **kwargs):
            """Return a mock agent instead of a real OpenRouter agent."""
            mock = MagicMock()
            mock.name = agent_name

            async def _generate(prompt, **kw):
                return f"Mock synthesis for {agent_name}"

            mock.generate = _generate
            return mock

        monkeypatch.setattr(fallback_mod, "create_openrouter_fallback", _mock_create_fallback)
    except (ImportError, AttributeError):
        pass

    # Mock the Anthropic API agent used for synthesis
    try:
        import aragora.debate.phases.synthesis_generator as synth_mod

        original_init = getattr(synth_mod, "_original_synth_init", None)

        # Patch AnthropicAPIAgent at the import point in synthesis_generator
        import aragora.agents.api_agents.anthropic as anthropic_mod

        OriginalAgent = anthropic_mod.AnthropicAPIAgent
        original_generate = OriginalAgent.generate if hasattr(OriginalAgent, "generate") else None

        class MockSynthesisAgent:
            def __init__(self, name="synthesis-agent", model="mock", **kwargs):
                self.name = name
                self.model = model
                self.total_input_tokens = 0
                self.total_output_tokens = 0

            async def generate(self, prompt, **kwargs):
                return "Mock synthesis: The agents agreed on a balanced approach."

        monkeypatch.setattr(anthropic_mod, "AnthropicAPIAgent", MockSynthesisAgent)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _mock_agent_registry(monkeypatch):
    """Mock AgentRegistry.create to return mock agents instead of real API agents.

    Tests that call AgentRegistry.create() (e.g. canvas execute_action with
    start_debate) would otherwise create real AnthropicAPIAgent or OpenAI agents
    that make HTTP calls to external APIs. This fixture intercepts the registry
    to return lightweight mock agents with all required methods.
    """

    def _make_mock_agent(model_type, name=None, role="proposer", model=None, **kwargs):
        from aragora.core_types import Agent, Critique, Vote

        class _TestAgent(Agent):
            def __init__(self):
                super().__init__(
                    name=name or f"mock-{model_type}",
                    model=model or "mock-model",
                    role=role,
                )
                self.total_input_tokens = 0
                self.total_output_tokens = 0
                self.input_tokens = 0
                self.output_tokens = 0
                self.total_tokens_in = 0
                self.total_tokens_out = 0
                self.metrics = None
                self.provider = None

            async def generate(self, prompt, **kw):
                return f"Mock response from {self.name}"

            async def critique(self, proposal, task="", **kw):
                return Critique(
                    agent=self.name,
                    target_agent=kw.get("target_agent", "unknown"),
                    target_content=str(proposal)[:100],
                    issues=[],
                    suggestions=[],
                    severity=0.2,
                    reasoning="Mock critique",
                )

            async def vote(self, proposals, task="", **kw):
                choice = (
                    list(proposals.keys())[0]
                    if isinstance(proposals, dict) and proposals
                    else (proposals[0] if proposals else "abstain")
                )
                return Vote(
                    agent=self.name,
                    choice=choice,
                    reasoning="Mock vote",
                    confidence=0.8,
                    continue_debate=False,
                )

        return _TestAgent()

    try:
        from aragora.agents.registry import AgentRegistry

        monkeypatch.setattr(
            AgentRegistry, "create", classmethod(lambda cls, *a, **kw: _make_mock_agent(*a, **kw))
        )
    except (ImportError, AttributeError):
        pass


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class E2EConfig:
    """E2E test configuration."""

    test_port: int = 18080
    test_host: str = "127.0.0.1"
    use_temp_db: bool = True
    db_url: str | None = None
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


# Note: Using default event loop from pytest-asyncio with function scope
# This matches the global asyncio_default_fixture_loop_scope = "function" setting


# ============================================================================
# Test Client
# ============================================================================


@dataclass
class TestClient:
    """HTTP test client for E2E tests."""

    base_url: str

    async def get(
        self,
        path: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{path}"
            async with session.get(url, headers=headers, params=params) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "json": (
                        await response.json()
                        if response.content_type == "application/json"
                        else None
                    ),
                    "text": await response.text(),
                }

    async def post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{path}"
            async with session.post(url, json=json, headers=headers) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "json": (
                        await response.json()
                        if response.content_type == "application/json"
                        else None
                    ),
                    "text": await response.text(),
                }

    async def put(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{path}"
            async with session.put(url, json=json, headers=headers) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "json": (
                        await response.json()
                        if response.content_type == "application/json"
                        else None
                    ),
                    "text": await response.text(),
                }

    async def delete(self, path: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{path}"
            async with session.delete(url, headers=headers) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "json": (
                        await response.json()
                        if response.content_type == "application/json"
                        else None
                    ),
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

    __test__ = False  # Not a pytest test class

    id: str
    name: str
    api_key: str
    tier: str = "standard"

    @property
    def auth_headers(self) -> dict[str, str]:
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
def isolated_tenants(tenant_a: TestTenant, tenant_b: TestTenant) -> list[TestTenant]:
    return [tenant_a, tenant_b]


# ============================================================================
# Connector Fixtures
# ============================================================================


@pytest.fixture
def mock_github_api():
    """Mock GitHub API responses via _run_gh method."""
    import json
    import base64

    # Mock responses for different gh CLI commands
    async def mock_run_gh(args):
        # Detect which API is being called
        args_str = " ".join(args)

        if "commits/" in args_str and ".sha" in args_str:
            # Get latest commit SHA
            return "abc123def456"

        if "/commits" in args_str:
            # Get commits list
            return json.dumps(
                [
                    {
                        "sha": "abc123def456",
                        "commit": {
                            "message": "Initial commit",
                            "author": {"name": "test", "date": "2024-01-01T00:00:00Z"},
                        },
                    }
                ]
            )

        if "/git/trees/" in args_str:
            # Get file tree
            return json.dumps(
                [
                    {"path": "README.md", "type": "blob", "size": 100, "sha": "file1"},
                    {"path": "src/main.py", "type": "blob", "size": 500, "sha": "file2"},
                ]
            )

        if "/contents/" in args_str:
            # Get file content (base64 encoded)
            return base64.b64encode(b"# Test README\nHello world").decode()

        if "issue list" in args_str:
            return json.dumps([])

        if "pr list" in args_str:
            return json.dumps([])

        return None

    with patch(
        "aragora.connectors.enterprise.git.github.GitHubEnterpriseConnector._run_gh",
        side_effect=mock_run_gh,
    ):
        with patch(
            "aragora.connectors.enterprise.git.github.GitHubEnterpriseConnector._check_gh_cli",
            return_value=True,
        ):
            yield


@pytest.fixture
def mock_slack_api():
    """Mock Slack API responses via _api_request method."""
    import json

    async def mock_api_request(endpoint, method="GET", params=None, json_data=None):
        """Mock Slack Web API responses."""
        if endpoint == "conversations.list":
            return {
                "ok": True,
                "channels": [
                    {
                        "id": "C001",
                        "name": "general",
                        "is_private": False,
                        "is_archived": False,
                        "topic": {"value": "General discussion"},
                        "purpose": {"value": "Company-wide announcements"},
                        "num_members": 50,
                        "created": 1609459200,
                    },
                    {
                        "id": "C002",
                        "name": "random",
                        "is_private": False,
                        "is_archived": False,
                        "topic": {"value": "Random chat"},
                        "purpose": {"value": "Off-topic discussion"},
                        "num_members": 30,
                        "created": 1609459200,
                    },
                ],
                "response_metadata": {},
            }
        elif endpoint == "conversations.history":
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": "1234567890.123456",
                        "text": "Hello team!",
                        "user": "U001",
                        "reactions": [],
                        "files": [],
                    },
                    {
                        "ts": "1234567890.123457",
                        "text": "Welcome!",
                        "user": "U002",
                        "reactions": [{"name": "wave", "count": 2}],
                        "files": [],
                    },
                ],
                "has_more": False,
            }
        elif endpoint == "users.info":
            user_id = params.get("user", "U001") if params else "U001"
            return {
                "ok": True,
                "user": {
                    "id": user_id,
                    "name": f"user_{user_id}",
                    "real_name": f"Test User {user_id}",
                    "profile": {
                        "display_name": f"testuser{user_id}",
                        "email": f"{user_id}@test.com",
                    },
                    "is_bot": False,
                },
            }
        elif endpoint == "conversations.replies":
            return {
                "ok": True,
                "messages": [
                    {"ts": "1234567890.123456", "text": "Parent message", "user": "U001"},
                    {"ts": "1234567890.123458", "text": "Reply 1", "user": "U002"},
                ],
            }
        elif endpoint == "search.messages":
            return {
                "ok": True,
                "messages": {"matches": []},
            }
        return {"ok": True}

    with patch(
        "aragora.connectors.enterprise.collaboration.slack.SlackConnector._api_request",
        side_effect=mock_api_request,
    ):
        with patch(
            "aragora.connectors.enterprise.collaboration.slack.SlackConnector._get_auth_header",
            return_value={"Authorization": "Bearer xoxb-mock-token"},
        ):
            yield


@pytest.fixture
def mock_notion_api():
    """Mock Notion API responses via _api_request method."""
    import json

    async def mock_api_request(endpoint, method="GET", params=None, json_data=None):
        """Mock Notion API responses."""
        if endpoint == "/search" and method == "POST":
            filter_type = json_data.get("filter", {}).get("value") if json_data else None
            if filter_type == "page":
                return {
                    "results": [
                        {
                            "object": "page",
                            "id": "page-001",
                            "url": "https://notion.so/page-001",
                            "properties": {
                                "title": {
                                    "type": "title",
                                    "title": [{"plain_text": "Test Page"}],
                                }
                            },
                            "parent": {"type": "workspace"},
                            "created_time": "2024-01-01T00:00:00.000Z",
                            "last_edited_time": "2024-01-15T00:00:00.000Z",
                            "created_by": {"id": "user-001"},
                            "last_edited_by": {"id": "user-001"},
                            "archived": False,
                        }
                    ],
                    "next_cursor": None,
                }
            elif filter_type == "database":
                return {
                    "results": [
                        {
                            "object": "database",
                            "id": "db-001",
                            "url": "https://notion.so/db-001",
                            "title": [{"plain_text": "Tasks Database"}],
                            "description": [],
                            "properties": {},
                            "created_time": "2024-01-01T00:00:00.000Z",
                            "last_edited_time": "2024-01-15T00:00:00.000Z",
                        }
                    ],
                    "next_cursor": None,
                }
            return {"results": [], "next_cursor": None}
        elif "/blocks/" in endpoint and "/children" in endpoint:
            return {
                "results": [
                    {
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"plain_text": "Test paragraph content"}]},
                        "has_children": False,
                    }
                ],
                "next_cursor": None,
            }
        elif endpoint.startswith("/pages/"):
            page_id = endpoint.split("/")[2]
            return {
                "object": "page",
                "id": page_id,
                "url": f"https://notion.so/{page_id}",
                "properties": {
                    "title": {
                        "type": "title",
                        "title": [{"plain_text": "Fetched Page"}],
                    }
                },
                "parent": {"type": "workspace"},
                "created_time": "2024-01-01T00:00:00.000Z",
                "last_edited_time": "2024-01-15T00:00:00.000Z",
                "created_by": {"id": "user-001"},
                "last_edited_by": {"id": "user-001"},
                "archived": False,
            }
        elif endpoint.startswith("/databases/") and "/query" in endpoint:
            return {
                "results": [
                    {
                        "object": "page",
                        "id": "entry-001",
                        "url": "https://notion.so/entry-001",
                        "properties": {
                            "Name": {
                                "type": "title",
                                "title": [{"plain_text": "Task 1"}],
                            },
                            "Status": {
                                "type": "select",
                                "select": {"name": "In Progress"},
                            },
                        },
                        "parent": {"type": "database_id", "database_id": "db-001"},
                        "created_time": "2024-01-01T00:00:00.000Z",
                        "last_edited_time": "2024-01-15T00:00:00.000Z",
                        "created_by": {"id": "user-001"},
                        "last_edited_by": {"id": "user-001"},
                        "archived": False,
                    }
                ],
                "next_cursor": None,
            }
        elif endpoint.startswith("/databases/"):
            db_id = endpoint.split("/")[2]
            return {
                "object": "database",
                "id": db_id,
                "url": f"https://notion.so/{db_id}",
                "title": [{"plain_text": "Test Database"}],
                "description": [],
                "properties": {},
                "created_time": "2024-01-01T00:00:00.000Z",
                "last_edited_time": "2024-01-15T00:00:00.000Z",
            }
        return {}

    with patch(
        "aragora.connectors.enterprise.collaboration.notion.NotionConnector._api_request",
        side_effect=mock_api_request,
    ):
        with patch(
            "aragora.connectors.enterprise.collaboration.notion.NotionConnector._get_auth_header",
            return_value={"Authorization": "Bearer secret_mock", "Notion-Version": "2022-06-28"},
        ):
            yield


# ============================================================================
# Agent Fixtures
# ============================================================================


@dataclass
class MockAgentResponse:
    content: str
    model: str = "mock-model"
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 100, "completion_tokens": 50}
    )


@pytest.fixture
def mock_llm_agents():
    """Provide mock LLM responses for tests that need them.

    Note: Most tests create their own mock agents inline.
    This fixture provides standard mock responses when needed.
    """
    responses = [
        MockAgentResponse(
            content="Thoughtful response to the debate topic.", model="claude-3-opus"
        ),
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
    agents: list[str]
    rounds: int
    protocol: str = "structured"
    consensus_threshold: float = 0.7

    def to_config(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "agents": self.agents,
            "protocol": {
                "type": self.protocol,
                "rounds": self.rounds,
                "consensus": {"type": "majority", "threshold": self.consensus_threshold},
            },
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
def sample_facts() -> list[dict[str, Any]]:
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


# ============================================================================
# E2E Test Harness Fixtures
# ============================================================================


# Import harness components
from tests.e2e.harness import (
    E2ETestConfig as HarnessConfig,
    E2ETestHarness,
    MockAgent,
    MockAgentConfig,
    DebateTestHarness,
    LoadTestHarness,
    e2e_environment,
    create_mock_agent,
)


@pytest_asyncio.fixture
async def e2e_harness() -> AsyncGenerator[E2ETestHarness, None]:
    """Provide a basic E2E test harness.

    Creates a test environment with:
    - 3 mock agents with general capabilities
    - In-memory storage (no Redis required)
    - Short timeouts for fast tests

    Usage:
        async def test_something(e2e_harness):
            task_id = await e2e_harness.submit_task("test", {"data": "..."})
            result = await e2e_harness.wait_for_task(task_id)
    """
    async with e2e_environment() as harness:
        yield harness


@pytest_asyncio.fixture
async def e2e_harness_with_redis() -> AsyncGenerator[E2ETestHarness, None]:
    """Provide E2E harness with Redis backend.

    Note: Requires Redis to be running at localhost:6379.
    Falls back to in-memory if Redis is unavailable.
    """
    config = HarnessConfig(
        use_redis=True,
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
    )
    async with e2e_environment(config) as harness:
        yield harness


@pytest_asyncio.fixture
async def debate_harness() -> AsyncGenerator[DebateTestHarness, None]:
    """Provide a debate-focused test harness.

    Pre-configured for debate testing with:
    - 4 agents with debate/critique/vote capabilities
    - Debate result tracking
    - Consensus rate calculation
    """
    harness = DebateTestHarness()
    try:
        await harness.start()
        yield harness
    finally:
        await harness.stop()


@pytest_asyncio.fixture
async def load_test_harness() -> AsyncGenerator[LoadTestHarness, None]:
    """Provide a load testing harness.

    Pre-configured for load testing with:
    - 10 agents
    - Fast response times
    - Concurrent task submission helpers
    - Throughput measurement
    """
    harness = LoadTestHarness()
    try:
        await harness.start()
        yield harness
    finally:
        await harness.stop()


@pytest.fixture
def harness_config() -> HarnessConfig:
    """Provide a customizable harness configuration.

    Usage:
        def test_custom(harness_config):
            harness_config.num_agents = 5
            harness_config.fail_rate = 0.1
            # Then use with e2e_environment(harness_config)
    """
    return HarnessConfig()


@pytest.fixture
def mock_agent_factory():
    """Factory for creating mock agents with custom configurations.

    Usage:
        def test_agents(mock_agent_factory):
            agent1 = mock_agent_factory("agent-1", ["debate"])
            agent2 = mock_agent_factory("agent-2", ["code"], fail_rate=0.5)
    """

    def factory(
        name: str,
        capabilities: list[str] | None = None,
        response: str = "Default response",
        fail_rate: float = 0.0,
        response_delay: float = 0.05,
    ) -> MockAgent:
        return MockAgent(
            id=name,
            name=name,
            capabilities=capabilities or ["general"],
            response_template=response,
            fail_rate=fail_rate,
            response_delay=response_delay,
        )

    return factory
