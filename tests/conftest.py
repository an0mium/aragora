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
# Optional Dependency Skip Markers
# ============================================================================
# These markers can be used to skip tests requiring optional dependencies.
# Usage: @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)


def _check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# Z3 solver for formal verification
HAS_Z3 = _check_import("z3")
REQUIRES_Z3 = "z3-solver not installed (pip install z3-solver)"
requires_z3 = not HAS_Z3

# Redis for caching and pub/sub
HAS_REDIS = _check_import("redis")
REQUIRES_REDIS = "redis not installed (pip install redis)"
requires_redis = not HAS_REDIS

# PostgreSQL async driver
HAS_ASYNCPG = _check_import("asyncpg")
REQUIRES_ASYNCPG = "asyncpg not installed (pip install asyncpg)"
requires_asyncpg = not HAS_ASYNCPG

# Supabase client
HAS_SUPABASE = _check_import("supabase")
REQUIRES_SUPABASE = "supabase not installed (pip install supabase)"
requires_supabase = not HAS_SUPABASE

# HTTPX async client
HAS_HTTPX = _check_import("httpx")
REQUIRES_HTTPX = "httpx not installed (pip install httpx)"
requires_httpx = not HAS_HTTPX

# WebSockets
HAS_WEBSOCKETS = _check_import("websockets")
REQUIRES_WEBSOCKETS = "websockets not installed (pip install websockets)"
requires_websockets = not HAS_WEBSOCKETS

# PyJWT
HAS_PYJWT = _check_import("jwt")
REQUIRES_PYJWT = "PyJWT not installed (pip install PyJWT)"
requires_pyjwt = not HAS_PYJWT

# Scikit-learn for ML features
HAS_SKLEARN = _check_import("sklearn")
REQUIRES_SKLEARN = "scikit-learn not installed (pip install scikit-learn)"
requires_sklearn = not HAS_SKLEARN

# SentenceTransformers for embeddings
HAS_SENTENCE_TRANSFORMERS = _check_import("sentence_transformers")
REQUIRES_SENTENCE_TRANSFORMERS = "sentence-transformers not installed"
requires_sentence_transformers = not HAS_SENTENCE_TRANSFORMERS

# MCP (Model Context Protocol)
HAS_MCP = _check_import("mcp")
REQUIRES_MCP = "mcp not installed (pip install mcp)"
requires_mcp = not HAS_MCP


def _check_aragora_module(module_path: str) -> bool:
    """Check if an Aragora module can be imported."""
    try:
        __import__(module_path)
        return True
    except (ImportError, AttributeError):
        return False


# Aragora optional modules
HAS_RLM = _check_aragora_module("aragora.rlm")
REQUIRES_RLM = "RLM module not available"
requires_rlm = not HAS_RLM

HAS_RBAC = _check_aragora_module("aragora.rbac")
REQUIRES_RBAC = "RBAC module not available"
requires_rbac = not HAS_RBAC

HAS_TRICKSTER = _check_aragora_module("aragora.debate.trickster")
REQUIRES_TRICKSTER = "Trickster module not available"
requires_trickster = not HAS_TRICKSTER

HAS_PLUGINS = _check_aragora_module("aragora.plugins")
REQUIRES_PLUGINS = "Plugins module not available"
requires_plugins = not HAS_PLUGINS


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
    config.addinivalue_line("markers", "smoke: quick sanity tests for fast CI feedback")
    config.addinivalue_line(
        "markers", "integration: tests requiring external dependencies (APIs, databases)"
    )
    config.addinivalue_line("markers", "slow: long-running tests (>30 seconds)")
    config.addinivalue_line("markers", "unit: isolated unit tests with no external dependencies")
    config.addinivalue_line(
        "markers", "network: tests requiring external network calls (skip with -m 'not network')"
    )


# ============================================================================
# Global Test Setup
# ============================================================================


@pytest.fixture(autouse=True)
def fast_convergence_backend(request):
    """Use fast Jaccard backend for convergence detection by default.

    This prevents slow ML model loading during tests. Tests that specifically
    need SentenceTransformer should use @pytest.mark.slow and the full backend.

    Set ARAGORA_CONVERGENCE_BACKEND=jaccard for fast tests (default).
    Tests marked @pytest.mark.slow will use the real ML backend.
    """
    # Skip this fixture for slow tests - they may need real ML backend
    if "slow" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    # Set fast backend for non-slow tests
    old_value = os.environ.get("ARAGORA_CONVERGENCE_BACKEND")
    os.environ["ARAGORA_CONVERGENCE_BACKEND"] = "jaccard"
    yield
    # Restore original value
    if old_value is None:
        os.environ.pop("ARAGORA_CONVERGENCE_BACKEND", None)
    else:
        os.environ["ARAGORA_CONVERGENCE_BACKEND"] = old_value


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
def mock_sentence_transformers(request, monkeypatch):
    """Mock SentenceTransformer to prevent HuggingFace model downloads.

    This prevents tests from making network calls to HuggingFace Hub,
    which can cause timeouts and flaky tests. Tests marked @pytest.mark.slow
    that need real embeddings are excluded.

    The mock returns deterministic embeddings based on input text hash,
    ensuring consistent behavior across test runs.
    """
    import numpy as np

    # Clear embedding service cache to ensure fresh instances per test
    try:
        import aragora.ml.embeddings as emb_module

        emb_module._embedding_services.clear()
    except (ImportError, AttributeError):
        pass

    # Skip for slow tests that may need real embeddings
    if "slow" in [m.name for m in request.node.iter_markers()]:
        yield
        # Clear cache after slow test too
        try:
            emb_module._embedding_services.clear()
        except (ImportError, AttributeError, NameError):
            pass
        return

    class MockSentenceTransformer:
        """Mock SentenceTransformer that returns deterministic embeddings."""

        def __init__(self, model_name_or_path=None, **kwargs):
            self.model_name = model_name_or_path or "mock-model"
            self._embedding_dim = 384  # Standard for many models

        def encode(
            self,
            sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=False,
            **kwargs,
        ):
            """Return deterministic embeddings with semantic-like similarity.

            Embeddings are based on word tokens, so texts with common words
            will have similar embeddings (mimicking real semantic similarity).
            """
            single_input = isinstance(sentences, str)
            if single_input:
                sentences = [sentences]

            embeddings = []
            for text in sentences:
                # Create embedding based on word tokens for semantic-like similarity
                emb = np.zeros(self._embedding_dim, dtype=np.float32)
                words = text.lower().split()
                for word in words:
                    # Add contribution from each word (deterministic)
                    word_seed = hash(word) % (2**32)
                    word_rng = np.random.RandomState(word_seed)
                    word_vec = word_rng.randn(self._embedding_dim).astype(np.float32)
                    emb += word_vec * 0.1
                # Add small unique component for exact text
                text_seed = hash(text) % (2**32)
                text_rng = np.random.RandomState(text_seed)
                emb += text_rng.randn(self._embedding_dim).astype(np.float32) * 0.01

                if normalize_embeddings:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                embeddings.append(emb)

            result = np.array(embeddings)

            # Return 1D array for single input (matches real SentenceTransformer behavior)
            if single_input:
                result = result[0]

            if convert_to_tensor:
                try:
                    import torch

                    return torch.tensor(result)
                except ImportError:
                    pass
            return result

        def get_sentence_embedding_dimension(self):
            return self._embedding_dim

    class MockCrossEncoder:
        """Mock CrossEncoder for NLI/contradiction detection."""

        def __init__(self, model_name=None, **kwargs):
            self.model_name = model_name or "mock-cross-encoder"

        def predict(self, sentence_pairs, **kwargs):
            """Return mock contradiction scores."""
            if not sentence_pairs:
                return np.array([])
            # Return neutral scores (entailment, neutral, contradiction)
            return np.array([[0.1, 0.8, 0.1]] * len(sentence_pairs))

    # Mock at the sentence_transformers module level
    try:
        import sentence_transformers

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", MockSentenceTransformer)
        if hasattr(sentence_transformers, "CrossEncoder"):
            monkeypatch.setattr(sentence_transformers, "CrossEncoder", MockCrossEncoder)
    except ImportError:
        pass

    # Also patch string-based imports
    try:
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            MockSentenceTransformer,
        )
        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            MockCrossEncoder,
        )
    except (ImportError, AttributeError):
        pass

    # Patch modules that do lazy imports
    modules_to_patch = [
        "aragora.debate.convergence",
        "aragora.debate.similarity.backends",
        "aragora.debate.similarity.factory",
        "aragora.knowledge.bridges",
        "aragora.memory.embeddings",
        "aragora.analysis.semantic",
        "aragora.ml.embeddings",
    ]
    for module_path in modules_to_patch:
        try:
            monkeypatch.setattr(
                f"{module_path}.SentenceTransformer",
                MockSentenceTransformer,
            )
        except (ImportError, AttributeError):
            pass
        try:
            monkeypatch.setattr(
                f"{module_path}.CrossEncoder",
                MockCrossEncoder,
            )
        except (ImportError, AttributeError):
            pass

    yield


@pytest.fixture(autouse=True)
def mock_external_apis(request, monkeypatch):
    """Mock external API clients to prevent network calls during tests.

    This prevents tests from making real API calls to:
    - OpenAI (openai.OpenAI, openai.AsyncOpenAI)
    - Anthropic (anthropic.Anthropic, anthropic.AsyncAnthropic)
    - Generic HTTP (httpx.Client, httpx.AsyncClient)

    Tests marked @pytest.mark.network or @pytest.mark.integration are excluded
    and will use real API clients (for tests that need actual network access).

    The mock returns deterministic responses based on input prompts,
    ensuring consistent behavior across test runs.
    """
    # Skip for tests that need real network access
    markers = [m.name for m in request.node.iter_markers()]
    if "network" in markers or "integration" in markers:
        yield
        return

    # =========================================================================
    # Mock OpenAI Client
    # =========================================================================

    class MockOpenAIMessage:
        """Mock OpenAI message object."""

        def __init__(self, content: str, role: str = "assistant"):
            self.content = content
            self.role = role

    class MockOpenAIChoice:
        """Mock OpenAI choice object."""

        def __init__(self, content: str, index: int = 0):
            self.message = MockOpenAIMessage(content)
            self.index = index
            self.finish_reason = "stop"

    class MockOpenAIUsage:
        """Mock OpenAI usage object."""

        def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens

    class MockOpenAICompletion:
        """Mock OpenAI chat completion response."""

        def __init__(self, content: str, model: str = "gpt-4o"):
            self.id = "chatcmpl-mock123"
            self.model = model
            self.choices = [MockOpenAIChoice(content)]
            self.usage = MockOpenAIUsage()
            self.created = 1700000000

    class MockOpenAIChatCompletions:
        """Mock OpenAI chat completions API."""

        def _generate_response(self, messages, **kwargs) -> str:
            """Generate deterministic response based on input."""
            # Extract the last user message for deterministic response
            last_msg = ""
            for msg in reversed(messages):
                content = (
                    msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                )
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                if role == "user":
                    last_msg = content
                    break

            # Generate deterministic response based on hash of input
            seed = hash(last_msg) % 1000
            responses = [
                f"I understand your query about '{last_msg[:50]}...'. Here's my analysis.",
                "Based on the information provided, I would suggest considering multiple perspectives.",
                "This is an interesting question. Let me provide a structured response.",
                "After careful consideration, here are my thoughts on the matter.",
                "I'll address your question systematically with supporting reasoning.",
            ]
            return responses[seed % len(responses)]

        def create(self, messages, model="gpt-4o", **kwargs):
            """Sync create method."""
            content = self._generate_response(messages, **kwargs)
            return MockOpenAICompletion(content, model)

        async def acreate(self, messages, model="gpt-4o", **kwargs):
            """Async create method (for compatibility)."""
            content = self._generate_response(messages, **kwargs)
            return MockOpenAICompletion(content, model)

    class MockOpenAIAsyncChatCompletions:
        """Mock async OpenAI chat completions API."""

        def _generate_response(self, messages, **kwargs) -> str:
            """Generate deterministic response based on input."""
            last_msg = ""
            for msg in reversed(messages):
                content = (
                    msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                )
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                if role == "user":
                    last_msg = content
                    break

            seed = hash(last_msg) % 1000
            responses = [
                f"I understand your query about '{last_msg[:50]}...'. Here's my analysis.",
                "Based on the information provided, I would suggest considering multiple perspectives.",
                "This is an interesting question. Let me provide a structured response.",
                "After careful consideration, here are my thoughts on the matter.",
                "I'll address your question systematically with supporting reasoning.",
            ]
            return responses[seed % len(responses)]

        async def create(self, messages, model="gpt-4o", **kwargs):
            """Async create method."""
            content = self._generate_response(messages, **kwargs)
            return MockOpenAICompletion(content, model)

    class MockOpenAIChat:
        """Mock OpenAI chat API."""

        def __init__(self, async_mode: bool = False):
            if async_mode:
                self.completions = MockOpenAIAsyncChatCompletions()
            else:
                self.completions = MockOpenAIChatCompletions()

    class MockOpenAIClient:
        """Mock OpenAI sync client."""

        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key or "mock-openai-key"
            self.chat = MockOpenAIChat(async_mode=False)

    class MockAsyncOpenAIClient:
        """Mock OpenAI async client."""

        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key or "mock-openai-key"
            self.chat = MockOpenAIChat(async_mode=True)

    # =========================================================================
    # Mock Anthropic Client
    # =========================================================================

    class MockAnthropicTextBlock:
        """Mock Anthropic text block."""

        def __init__(self, text: str):
            self.type = "text"
            self.text = text

    class MockAnthropicUsage:
        """Mock Anthropic usage object."""

        def __init__(self, input_tokens: int = 10, output_tokens: int = 20):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class MockAnthropicMessage:
        """Mock Anthropic message response."""

        def __init__(self, content: str, model: str = "claude-sonnet-4-20250514"):
            self.id = "msg_mock123"
            self.type = "message"
            self.role = "assistant"
            self.content = [MockAnthropicTextBlock(content)]
            self.model = model
            self.stop_reason = "end_turn"
            self.usage = MockAnthropicUsage()

    class MockAnthropicMessages:
        """Mock Anthropic messages API."""

        def _generate_response(self, messages, **kwargs) -> str:
            """Generate deterministic response based on input."""
            last_msg = ""
            for msg in reversed(messages):
                content = (
                    msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                )
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                if role == "user":
                    last_msg = content if isinstance(content, str) else str(content)
                    break

            seed = hash(last_msg) % 1000
            responses = [
                f"Thank you for your question. I'll provide a thorough analysis of '{last_msg[:40]}...'.",
                "Let me address this thoughtfully. There are several key considerations here.",
                "This is a nuanced topic that deserves careful examination.",
                "I appreciate the opportunity to discuss this. Here's my perspective.",
                "Based on my analysis, I can offer the following insights.",
            ]
            return responses[seed % len(responses)]

        def create(self, messages, model="claude-sonnet-4-20250514", max_tokens=1024, **kwargs):
            """Sync create method."""
            content = self._generate_response(messages, **kwargs)
            return MockAnthropicMessage(content, model)

    class MockAnthropicAsyncMessages:
        """Mock async Anthropic messages API."""

        def _generate_response(self, messages, **kwargs) -> str:
            """Generate deterministic response based on input."""
            last_msg = ""
            for msg in reversed(messages):
                content = (
                    msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                )
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                if role == "user":
                    last_msg = content if isinstance(content, str) else str(content)
                    break

            seed = hash(last_msg) % 1000
            responses = [
                f"Thank you for your question. I'll provide a thorough analysis of '{last_msg[:40]}...'.",
                "Let me address this thoughtfully. There are several key considerations here.",
                "This is a nuanced topic that deserves careful examination.",
                "I appreciate the opportunity to discuss this. Here's my perspective.",
                "Based on my analysis, I can offer the following insights.",
            ]
            return responses[seed % len(responses)]

        async def create(
            self, messages, model="claude-sonnet-4-20250514", max_tokens=1024, **kwargs
        ):
            """Async create method."""
            content = self._generate_response(messages, **kwargs)
            return MockAnthropicMessage(content, model)

    class MockAnthropicClient:
        """Mock Anthropic sync client."""

        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key or "mock-anthropic-key"
            self.messages = MockAnthropicMessages()

    class MockAsyncAnthropicClient:
        """Mock Anthropic async client."""

        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key or "mock-anthropic-key"
            self.messages = MockAnthropicAsyncMessages()

    # =========================================================================
    # Mock HTTPX Clients
    # =========================================================================

    class MockHTTPXResponse:
        """Mock httpx response object."""

        def __init__(self, status_code: int = 200, json_data: dict = None, text: str = ""):
            self.status_code = status_code
            self._json_data = json_data or {}
            self._text = text or json.dumps(self._json_data)
            self.headers = {"content-type": "application/json"}
            self.is_success = 200 <= status_code < 300

        def json(self):
            return self._json_data

        @property
        def text(self):
            return self._text

        def raise_for_status(self):
            if not self.is_success:
                raise Exception(f"HTTP {self.status_code}")

    class MockHTTPXClient:
        """Mock httpx sync client."""

        def __init__(self, **kwargs):
            self._base_url = kwargs.get("base_url", "")
            self._timeout = kwargs.get("timeout", 30)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def _make_response(self, url: str, **kwargs) -> MockHTTPXResponse:
            """Generate mock response based on URL."""
            # Return deterministic responses based on URL hash
            seed = hash(url) % 100
            return MockHTTPXResponse(
                status_code=200,
                json_data={
                    "status": "ok",
                    "url": url,
                    "mock": True,
                    "seed": seed,
                },
            )

        def get(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        def post(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        def put(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        def delete(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        def patch(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        def request(self, method, url, **kwargs):
            return self._make_response(url, **kwargs)

        def close(self):
            pass

    class MockAsyncHTTPXClient:
        """Mock httpx async client."""

        def __init__(self, **kwargs):
            self._base_url = kwargs.get("base_url", "")
            self._timeout = kwargs.get("timeout", 30)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def _make_response(self, url: str, **kwargs) -> MockHTTPXResponse:
            """Generate mock response based on URL."""
            seed = hash(url) % 100
            return MockHTTPXResponse(
                status_code=200,
                json_data={
                    "status": "ok",
                    "url": url,
                    "mock": True,
                    "seed": seed,
                },
            )

        async def get(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        async def post(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        async def put(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        async def delete(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        async def patch(self, url, **kwargs):
            return self._make_response(url, **kwargs)

        async def request(self, method, url, **kwargs):
            return self._make_response(url, **kwargs)

        async def aclose(self):
            pass

        def close(self):
            pass

    # =========================================================================
    # Apply Patches
    # =========================================================================

    # Patch OpenAI
    try:
        import openai

        monkeypatch.setattr(openai, "OpenAI", MockOpenAIClient)
        monkeypatch.setattr(openai, "AsyncOpenAI", MockAsyncOpenAIClient)
    except ImportError:
        pass

    # Also patch string-based imports for OpenAI
    try:
        monkeypatch.setattr("openai.OpenAI", MockOpenAIClient)
        monkeypatch.setattr("openai.AsyncOpenAI", MockAsyncOpenAIClient)
    except (ImportError, AttributeError):
        pass

    # Patch Anthropic
    try:
        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", MockAnthropicClient)
        monkeypatch.setattr(anthropic, "AsyncAnthropic", MockAsyncAnthropicClient)
    except ImportError:
        pass

    # Also patch string-based imports for Anthropic
    try:
        monkeypatch.setattr("anthropic.Anthropic", MockAnthropicClient)
        monkeypatch.setattr("anthropic.AsyncAnthropic", MockAsyncAnthropicClient)
    except (ImportError, AttributeError):
        pass

    # Patch httpx
    try:
        import httpx

        monkeypatch.setattr(httpx, "Client", MockHTTPXClient)
        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncHTTPXClient)
    except ImportError:
        pass

    # Also patch string-based imports for httpx
    try:
        monkeypatch.setattr("httpx.Client", MockHTTPXClient)
        monkeypatch.setattr("httpx.AsyncClient", MockAsyncHTTPXClient)
    except (ImportError, AttributeError):
        pass

    # Patch modules that may do lazy imports of API clients
    api_modules_to_patch = [
        "aragora.agents.api_agents.anthropic",
        "aragora.agents.api_agents.openai",
        "aragora.agents.api_agents.openrouter",
        "aragora.agents.fallback",
        "aragora.rlm.bridge",
    ]
    for module_path in api_modules_to_patch:
        # Patch OpenAI in module
        try:
            monkeypatch.setattr(f"{module_path}.OpenAI", MockOpenAIClient)
        except (ImportError, AttributeError):
            pass
        try:
            monkeypatch.setattr(f"{module_path}.AsyncOpenAI", MockAsyncOpenAIClient)
        except (ImportError, AttributeError):
            pass
        # Patch Anthropic in module
        try:
            monkeypatch.setattr(f"{module_path}.Anthropic", MockAnthropicClient)
        except (ImportError, AttributeError):
            pass
        try:
            monkeypatch.setattr(f"{module_path}.AsyncAnthropic", MockAsyncAnthropicClient)
        except (ImportError, AttributeError):
            pass

    yield


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
        state_file.write_text(
            json.dumps(
                {
                    "phase": "implement",
                    "stage": "executing",
                    "cycle": 1,
                    "total_tasks": 5,
                    "completed_tasks": 2,
                }
            )
        )

        # Create nomic log file
        log_file = nomic_dir / "nomic_loop.log"
        log_file.write_text(
            "\n".join(
                [
                    "2026-01-05 00:00:01 Starting cycle 1",
                    "2026-01-05 00:00:02 Phase: context",
                    "2026-01-05 00:00:03 Phase: debate",
                    "2026-01-05 00:00:04 Phase: design",
                    "2026-01-05 00:00:05 Phase: implement",
                ]
            )
        )

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
    elo.get_cached_leaderboard.return_value = [
        {
            "agent_name": "test_agent",
            "elo": 1500,
            "wins": 5,
            "losses": 3,
            "draws": 2,
            "games_played": 10,
            "win_rate": 0.5,
        }
    ]
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


@pytest.fixture
def mock_calibration_tracker() -> Mock:
    """Create a mock CalibrationTracker.

    Returns a Mock object with calibration methods that return
    fast, deterministic values suitable for testing.
    """
    tracker = Mock()

    # Mock calibration summary
    mock_summary = Mock()
    mock_summary.agent = "test_agent"
    mock_summary.total_predictions = 100
    mock_summary.total_correct = 75
    mock_summary.brier_score = 0.15
    mock_summary.ece = 0.08
    mock_summary.adjust_confidence = Mock(side_effect=lambda c, domain=None: c)

    # Configure methods
    tracker.get_calibration_summary.return_value = mock_summary
    tracker.get_brier_score.return_value = 0.15
    tracker.get_expected_calibration_error.return_value = 0.08
    tracker.get_calibration_curve.return_value = []
    tracker.get_all_agents.return_value = ["test_agent"]
    tracker.record_prediction = Mock()
    tracker.record_outcome = Mock()
    tracker.get_temperature_params.return_value = Mock(
        temperature=1.0, get_temperature=Mock(return_value=1.0)
    )

    return tracker


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

        if hasattr(belief_handler, "BeliefNetwork"):
            belief_handler.BeliefNetwork = None
        if hasattr(belief_handler, "BeliefPropagationAnalyzer"):
            belief_handler.BeliefPropagationAnalyzer = None
        if hasattr(belief_handler, "PersonaLaboratory"):
            belief_handler.PersonaLaboratory = None
        if hasattr(belief_handler, "ProvenanceTracker"):
            belief_handler.ProvenanceTracker = None
    except (ImportError, AttributeError):
        pass

    # Reset handler globals (consensus)
    try:
        import aragora.server.handlers.consensus as consensus_handler

        if hasattr(consensus_handler, "ConsensusMemory"):
            consensus_handler.ConsensusMemory = None
        if hasattr(consensus_handler, "DissentRetriever"):
            consensus_handler.DissentRetriever = None
    except (ImportError, AttributeError):
        pass

    # Reset handler globals (critique)
    try:
        import aragora.server.handlers.critique as critique_handler

        if hasattr(critique_handler, "CritiqueStore"):
            critique_handler.CritiqueStore = None
    except (ImportError, AttributeError):
        pass

    # Reset handler globals (calibration)
    try:
        import aragora.server.handlers.calibration as cal_handler

        if hasattr(cal_handler, "CalibrationTracker"):
            cal_handler.CalibrationTracker = None
        if hasattr(cal_handler, "EloSystem"):
            cal_handler.EloSystem = None
    except (ImportError, AttributeError):
        pass

    # Clear DatabaseManager singleton instances
    try:
        from aragora.storage.schema import DatabaseManager

        DatabaseManager.clear_instances()
    except (ImportError, AttributeError):
        pass

    # Clear rate limiters to prevent test pollution
    try:
        from aragora.server.handlers.utils.rate_limit import clear_all_limiters

        clear_all_limiters()
    except (ImportError, AttributeError):
        pass

    # Clear module-level rate limiters (not in registry)
    try:
        import aragora.server.handlers.analytics as analytics

        if hasattr(analytics, "_analytics_limiter"):
            analytics._analytics_limiter.clear()
    except (ImportError, AttributeError):
        pass

    # Reset event loop if closed (prevents "Event loop is closed" errors)
    try:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            # No event loop in current thread - create one
            asyncio.set_event_loop(asyncio.new_event_loop())
    except ImportError:
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
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
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
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None if i < len(chunks) - 1 else "stop",
                        }
                    ],
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
        x = z3.Int("x")
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


# ============================================================================
# Skip Count Monitoring
# ============================================================================
# Track skip counts to warn when threshold is exceeded.
# See tests/SKIP_AUDIT.md for skip marker inventory.

SKIP_THRESHOLD = 650  # Maximum allowed skips (conditional + unconditional)
UNCONDITIONAL_SKIP_THRESHOLD = 35  # Maximum unconditional @pytest.mark.skip


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Warn if skip count exceeds threshold."""
    skipped = len(terminalreporter.stats.get("skipped", []))

    if skipped > SKIP_THRESHOLD:
        terminalreporter.write_line("")
        terminalreporter.write_line(
            f"WARNING: Skip count ({skipped}) exceeds threshold ({SKIP_THRESHOLD})",
            yellow=True,
            bold=True,
        )
        terminalreporter.write_line(
            "  Review tests/SKIP_AUDIT.md and reduce skipped tests.", yellow=True
        )
        terminalreporter.write_line("")
