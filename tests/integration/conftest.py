"""
Integration test fixtures for Aragora.

Provides fixtures for full end-to-end testing with:
- Mock agents with predictable responses
- Temporary databases
- Memory tier management
- Debate environment setup
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest

from aragora.core import (
    Agent,
    Message,
    Critique,
    Vote,
    Environment,
    DebateResult,
)
from aragora.debate.orchestrator import Arena, DebateProtocol


# =============================================================================
# Auto-apply integration marker to all tests in this directory
# =============================================================================


def pytest_collection_modifyitems(items):
    """Automatically add integration marker to all tests in this directory."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# =============================================================================
# Mock Agent Classes
# =============================================================================


class MockAgent(Agent):
    """
    Mock agent for integration testing with configurable responses.

    Usage:
        agent = MockAgent("test", responses=["Response 1", "Response 2"])
        result = await agent.generate("prompt")  # Returns "Response 1"
        result = await agent.generate("prompt")  # Returns "Response 2"
    """

    def __init__(
        self,
        name: str = "mock_agent",
        model: str = "mock-model",
        role: str = "proposer",
        responses: list[str] | None = None,
        critiques: list[Critique] | None = None,
        votes: list[Vote] | None = None,
        delay: float = 0.0,
    ):
        super().__init__(name, model, role)
        self.agent_type = "mock"
        self._responses = responses or [f"Default response from {name}"]
        self._critiques = critiques or []
        self._votes = votes or []
        self._call_count = 0
        self._delay = delay

    async def generate(self, prompt: str, context: list | None = None) -> str:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response

    async def critique(self, proposal: str, task: str, context: list | None = None) -> Critique:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._critiques:
            critique = self._critiques[self._call_count % len(self._critiques)]
            self._call_count += 1
            return critique
        self._call_count += 1
        return Critique(
            agent=self.name,
            target_agent="target",
            target_content=proposal[:100],
            issues=["Minor issue identified"],
            suggestions=["Consider improvement"],
            severity=0.3,
            reasoning="Standard critique",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._votes:
            vote = self._votes[self._call_count % len(self._votes)]
            self._call_count += 1
            return vote
        self._call_count += 1
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"Vote from {self.name}",
            confidence=0.85,
            continue_debate=False,
        )

    def reset(self):
        """Reset call count for reuse."""
        self._call_count = 0


class FailingAgent(MockAgent):
    """Agent that fails after N successful calls."""

    def __init__(
        self,
        name: str = "failing_agent",
        fail_after: int = 2,
        error_type: type = Exception,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._fail_after = fail_after
        self._error_type = error_type

    async def generate(self, prompt: str, context: list | None = None) -> str:
        if self._call_count >= self._fail_after:
            raise self._error_type(f"Simulated failure in {self.name}")
        return await super().generate(prompt, context)


class SlowAgent(MockAgent):
    """Agent with configurable response delay for timeout testing."""

    def __init__(self, name: str = "slow_agent", delay: float = 5.0, **kwargs):
        super().__init__(name, delay=delay, **kwargs)


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Provide a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_db.sqlite"


@pytest.fixture
def temp_memory_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for memory storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_agent_factory():
    """Factory for creating mock agents with custom responses."""

    def create(
        name: str = "mock",
        role: str = "proposer",
        responses: list[str] | None = None,
    ) -> MockAgent:
        return MockAgent(name=name, role=role, responses=responses)

    return create


@pytest.fixture
def mock_agents() -> list[MockAgent]:
    """Standard set of mock agents for debate testing."""
    return [
        MockAgent(
            name="proposer_agent",
            role="proposer",
            responses=[
                "I propose we implement a rate limiter using the token bucket algorithm. "
                "This approach provides smooth rate limiting with burst support.",
                "Based on the critique, I'll refine my proposal to include sliding window "
                "rate limiting as a fallback mechanism.",
            ],
        ),
        MockAgent(
            name="critic_agent",
            role="critic",
            responses=[
                "The proposal is solid but needs to address edge cases around "
                "distributed rate limiting across multiple nodes.",
                "The refinement addresses my concerns. I support this approach.",
            ],
        ),
        MockAgent(
            name="synthesizer_agent",
            role="synthesizer",
            responses=[
                "Synthesizing the discussion: We should use token bucket with sliding "
                "window fallback, ensuring distributed coordination via Redis.",
            ],
        ),
    ]


@pytest.fixture
def consensus_agents() -> list[MockAgent]:
    """Agents configured to reach consensus quickly."""
    shared_vote = Vote(
        agent="",  # Will be overwritten
        choice="proposer_agent",
        reasoning="Strong proposal",
        confidence=0.9,
        continue_debate=False,
    )
    return [
        MockAgent(
            name="agent_1",
            role="proposer",
            votes=[Vote(**{**shared_vote.__dict__, "agent": "agent_1"})],
        ),
        MockAgent(
            name="agent_2",
            role="critic",
            votes=[Vote(**{**shared_vote.__dict__, "agent": "agent_2"})],
        ),
        MockAgent(
            name="agent_3",
            role="synthesizer",
            votes=[Vote(**{**shared_vote.__dict__, "agent": "agent_3"})],
        ),
    ]


@pytest.fixture
def split_vote_agents() -> list[MockAgent]:
    """Agents configured to produce a split vote (no consensus)."""
    return [
        MockAgent(
            name="agent_1",
            role="proposer",
            votes=[
                Vote(
                    agent="agent_1",
                    choice="agent_1",
                    reasoning="My proposal is best",
                    confidence=0.9,
                    continue_debate=True,
                )
            ],
        ),
        MockAgent(
            name="agent_2",
            role="critic",
            votes=[
                Vote(
                    agent="agent_2",
                    choice="agent_2",
                    reasoning="I disagree",
                    confidence=0.9,
                    continue_debate=True,
                )
            ],
        ),
        MockAgent(
            name="agent_3",
            role="synthesizer",
            votes=[
                Vote(
                    agent="agent_3",
                    choice="agent_3",
                    reasoning="Neither is complete",
                    confidence=0.6,
                    continue_debate=True,
                )
            ],
        ),
    ]


# =============================================================================
# Debate Environment Fixtures
# =============================================================================


@pytest.fixture
def simple_environment() -> Environment:
    """Simple environment for basic debate testing."""
    return Environment(
        task="Design a rate limiter API",
        context="Building a web API that needs rate limiting",
    )


@pytest.fixture
def complex_environment() -> Environment:
    """Complex environment with constraints."""
    return Environment(
        task="Design a distributed consensus algorithm for multi-region deployment",
        context="""
        Requirements:
        - Must handle network partitions gracefully
        - Eventual consistency is acceptable
        - Need to support 5 regions globally
        - Latency budget: 500ms for reads, 2s for writes
        """,
    )


@pytest.fixture
def standard_protocol() -> DebateProtocol:
    """Standard debate protocol for testing."""
    return DebateProtocol(
        rounds=3,
        consensus="majority",
        critique_required=True,
    )


@pytest.fixture
def quick_protocol() -> DebateProtocol:
    """Quick debate protocol for fast tests."""
    return DebateProtocol(
        rounds=1,
        consensus="any",
        critique_required=False,
    )


# =============================================================================
# RBAC Auto-Bypass Fixture
# =============================================================================


@pytest.fixture(autouse=True)
def mock_auth_for_integration_tests(request, monkeypatch):
    """Bypass RBAC authentication for integration tests.

    This autouse fixture patches get_auth_context to return an authenticated
    admin context by default, and patches _get_context_from_args to inject
    context into already-decorated functions.

    To test authentication/authorization behavior specifically, use the
    @pytest.mark.no_auto_auth marker to opt-out of auto-mocking:

        @pytest.mark.no_auto_auth
        def test_unauthenticated_returns_401(handler, mock_http_handler):
            # get_auth_context will NOT be mocked for this test
            result = handler.handle("/api/v1/resource", {}, mock_http_handler)
            assert result.status_code == 401
    """
    # Check if test has opted out of auto-auth
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    # Create a mock auth context with admin permissions
    try:
        from aragora.rbac.models import AuthorizationContext

        mock_auth_ctx = AuthorizationContext(
            user_id="test-user-001",
            user_email="test@example.com",
            org_id="test-org-001",
            roles={"admin", "owner"},
            permissions={"*"},  # Wildcard grants all permissions
        )

        # Patch _get_context_from_args to return mock context when no context found
        try:
            from aragora.rbac import decorators

            original_get_context = decorators._get_context_from_args

            def patched_get_context_from_args(args, kwargs, context_param):
                result = original_get_context(args, kwargs, context_param)
                if result is None:
                    return mock_auth_ctx
                return result

            monkeypatch.setattr(decorators, "_get_context_from_args", patched_get_context_from_args)
        except (ImportError, AttributeError):
            pass

        # Create a mock UserAuthContext for BaseHandler auth methods
        try:
            from aragora.billing.auth.context import UserAuthContext

            mock_user_ctx = UserAuthContext(
                authenticated=True,
                user_id="test-user-001",
                email="test@example.com",
                org_id="test-org-001",
                role="admin",
                token_type="access",
                client_ip="127.0.0.1",
            )
            # Add extra attributes that some handlers expect
            object.__setattr__(mock_user_ctx, "roles", ["admin", "owner"])
            object.__setattr__(
                mock_user_ctx,
                "permissions",
                ["*", "admin", "knowledge.read", "knowledge.write", "knowledge.delete"],
            )

            # Patch BaseHandler auth methods
            from aragora.server.handlers.base import BaseHandler

            def mock_require_auth_or_error(self, handler):
                """Mock require_auth_or_error that returns authenticated user."""
                return mock_user_ctx, None

            def mock_require_admin_or_error(self, handler):
                """Mock require_admin_or_error that returns admin user."""
                return mock_user_ctx, None

            def mock_get_current_user(self, handler):
                """Mock get_current_user that returns authenticated user."""
                return mock_user_ctx

            monkeypatch.setattr(BaseHandler, "require_auth_or_error", mock_require_auth_or_error)
            monkeypatch.setattr(BaseHandler, "require_admin_or_error", mock_require_admin_or_error)
            monkeypatch.setattr(BaseHandler, "get_current_user", mock_get_current_user)

        except (ImportError, AttributeError):
            pass

        yield mock_auth_ctx

    except ImportError:
        # RBAC not available, skip auth mocking
        yield


# =============================================================================
# External Dependency Mocks
# =============================================================================


@pytest.fixture(autouse=True)
def mock_external_apis():
    """Mock all external API calls to prevent network requests."""
    with patch.object(
        Arena,
        "_gather_trending_context",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch(
            "aragora.debate.phases.context_init.ContextInitializer.initialize",
            new_callable=AsyncMock,
            return_value=None,
        ):
            yield


@pytest.fixture
def mock_llm_calls():
    """Mock LLM API calls for deterministic testing."""
    with patch(
        "aragora.agents.api_agents.anthropic.AnthropicAgent.generate",
        new_callable=AsyncMock,
        return_value="Mocked LLM response",
    ):
        with patch(
            "aragora.agents.api_agents.openai.OpenAIAgent.generate",
            new_callable=AsyncMock,
            return_value="Mocked LLM response",
        ):
            yield


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
def critique_store(temp_db_path) -> Generator:
    """Provide a CritiqueStore with temporary database."""
    from aragora.memory.store import CritiqueStore

    store = CritiqueStore(str(temp_db_path))
    yield store


@pytest.fixture
def memory_store(temp_memory_dir) -> Generator:
    """Provide a ContinuumMemory with temporary storage."""
    from aragora.memory.continuum import ContinuumMemory

    memory = ContinuumMemory(str(temp_memory_dir / "memory.db"))
    yield memory


@pytest.fixture
def elo_system(temp_db_path) -> Generator:
    """Provide an ELO system with temporary database."""
    from aragora.ranking.elo import EloSystem

    system = EloSystem(str(temp_db_path))
    yield system


# =============================================================================
# Helper Functions
# =============================================================================


async def run_debate_to_completion(
    arena: Arena,
    timeout: float = 30.0,
) -> DebateResult:
    """Run a debate with timeout protection."""
    return await asyncio.wait_for(arena.run(), timeout=timeout)


def assert_debate_completed(result: DebateResult):
    """Assert that a debate completed successfully."""
    assert result is not None
    assert result.rounds_completed >= 1
    assert result.final_answer is not None or result.proposals


def assert_consensus_reached(result: DebateResult):
    """Assert that consensus was reached."""
    assert result.consensus_reached
    assert result.final_answer is not None


def assert_no_consensus(result: DebateResult):
    """Assert that consensus was not reached."""
    assert not result.consensus_reached


# =============================================================================
# PostgreSQL Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def postgres_url():
    """Get PostgreSQL URL from environment if available."""
    import os

    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgresql://"):
        return url
    return None


@pytest.fixture
def postgres_backend(postgres_url):
    """Create a PostgreSQL backend for testing (skips if not configured)."""
    if not postgres_url:
        pytest.skip("PostgreSQL not configured (set DATABASE_URL)")

    from aragora.db.backends import PostgreSQLBackend

    backend = PostgreSQLBackend(postgres_url)
    yield backend
    backend.close()


@pytest.fixture
def db_backend(request, temp_db_path, postgres_url):
    """
    Parametrized database backend fixture.

    Returns SQLite by default, or PostgreSQL if DATABASE_URL is set.
    Use with @pytest.mark.parametrize to test both backends:

        @pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
        def test_something(db_backend, backend_type):
            ...
    """
    backend_type = getattr(request, "param", "sqlite")

    if backend_type == "postgres":
        if not postgres_url:
            pytest.skip("PostgreSQL not configured")
        from aragora.db.backends import PostgreSQLBackend

        backend = PostgreSQLBackend(postgres_url)
    else:
        from aragora.db.backends import SQLiteBackend

        backend = SQLiteBackend(str(temp_db_path))

    yield backend
    backend.close()
