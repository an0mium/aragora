"""
Shared pytest fixtures for Aragora test suite.

This module provides common fixtures used across multiple test files,
reducing duplication and ensuring consistent test setup.
"""

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Generator
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
    """Check if a module is available without fully importing it.

    Uses ``importlib.util.find_spec`` to avoid executing module-level code
    that may trigger heavy side-effects (e.g. ``sentence_transformers``
    pulls in ``transformers`` which imports ``huggingface_hub``, potentially
    blocking on network downloads in CI).
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
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

# Scikit-learn for ML features - now always available
HAS_SKLEARN = True
REQUIRES_SKLEARN = "scikit-learn not installed (pip install scikit-learn)"
requires_sklearn = False  # sklearn is always installed

# SentenceTransformers for embeddings
HAS_SENTENCE_TRANSFORMERS = _check_import("sentence_transformers")
REQUIRES_SENTENCE_TRANSFORMERS = "sentence-transformers not installed"
requires_sentence_transformers = not HAS_SENTENCE_TRANSFORMERS

# MCP (Model Context Protocol)
HAS_MCP = _check_import("mcp")
REQUIRES_MCP = "mcp not installed (pip install mcp)"
requires_mcp = not HAS_MCP

# aiosqlite for async SQLite
HAS_AIOSQLITE = _check_import("aiosqlite")
REQUIRES_AIOSQLITE = "aiosqlite not installed (pip install aiosqlite)"
requires_aiosqlite = not HAS_AIOSQLITE

# Twilio for SMS/voice
HAS_TWILIO = _check_import("twilio")
REQUIRES_TWILIO = "twilio not installed (pip install twilio)"
requires_twilio = not HAS_TWILIO

# PyOTP for TOTP/HOTP
HAS_PYOTP = _check_import("pyotp")
REQUIRES_PYOTP = "pyotp not installed (pip install pyotp)"
requires_pyotp = not HAS_PYOTP

# psycopg2 for PostgreSQL
HAS_PSYCOPG2 = _check_import("psycopg2")
REQUIRES_PSYCOPG2 = "psycopg2 not installed (pip install psycopg2-binary)"
requires_psycopg2 = not HAS_PSYCOPG2

# NetworkX for graph operations
HAS_NETWORKX = _check_import("networkx")
REQUIRES_NETWORKX = "networkx not installed (pip install networkx)"
requires_networkx = not HAS_NETWORKX

# ============================================================================
# Composite Skip Markers
# ============================================================================
# These combine multiple requirements for common test scenarios

# Requires any database backend
HAS_DATABASE = HAS_ASYNCPG or HAS_PSYCOPG2 or HAS_AIOSQLITE
REQUIRES_DATABASE = "No database driver installed (asyncpg, psycopg2, or aiosqlite)"
requires_database = not HAS_DATABASE

# Requires async database support
HAS_ASYNC_DB = HAS_ASYNCPG or HAS_AIOSQLITE
REQUIRES_ASYNC_DB = "No async database driver installed (asyncpg or aiosqlite)"
requires_async_db = not HAS_ASYNC_DB


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

HAS_BROADCAST = _check_aragora_module("aragora.broadcast.pipeline")
REQUIRES_BROADCAST = "Broadcast module not available (see #134)"
requires_broadcast = not HAS_BROADCAST


# Broadcast E2E tests require specific APIs not yet implemented
def _check_broadcast_e2e_api() -> bool:
    """Check if broadcast E2E test API is available."""
    try:
        from aragora.broadcast.audio_engine import AudioEngine, get_voice_for_agent
        from aragora.broadcast.rss_gen import create_episode, generate_feed

        return True
    except ImportError:
        return False


HAS_BROADCAST_E2E_API = _check_broadcast_e2e_api()
REQUIRES_BROADCAST_E2E_API = "Broadcast E2E API not fully implemented (AudioEngine, create_episode)"
requires_broadcast_e2e_api = not HAS_BROADCAST_E2E_API

HAS_BROADCAST_STORAGE = _check_aragora_module("aragora.broadcast.storage")
REQUIRES_BROADCAST_STORAGE = "Broadcast storage not available (see #134)"
requires_broadcast_storage = not HAS_BROADCAST_STORAGE

# Security and encryption modules
HAS_ENCRYPTION = _check_aragora_module("aragora.security.encryption")
REQUIRES_ENCRYPTION = "Encryption service not available"
requires_encryption = not HAS_ENCRYPTION

HAS_INTEGRATION_STORE = _check_aragora_module("aragora.storage.integration_store")
REQUIRES_INTEGRATION_STORE = "IntegrationStore not available"
requires_integration_store = not HAS_INTEGRATION_STORE

HAS_GMAIL_TOKEN_STORE = _check_aragora_module("aragora.storage.gmail_token_store")
REQUIRES_GMAIL_TOKEN_STORE = "GmailTokenStore not available"
requires_gmail_token_store = not HAS_GMAIL_TOKEN_STORE

HAS_SYNC_STORE = _check_aragora_module("aragora.connectors.enterprise.sync_store")
REQUIRES_SYNC_STORE = "SyncStore not available"
requires_sync_store = not HAS_SYNC_STORE

HAS_KEY_ROTATION = _check_aragora_module("aragora.security.migration")
REQUIRES_KEY_ROTATION = "Key rotation not available"
requires_key_rotation = not HAS_KEY_ROTATION

HAS_SECURITY_HANDLER = _check_aragora_module("aragora.server.handlers.admin.security")
REQUIRES_SECURITY_HANDLER = "SecurityHandler not available"
requires_security_handler = not HAS_SECURITY_HANDLER

HAS_SECURITY_METRICS = _check_aragora_module("aragora.observability.metrics.security")
REQUIRES_SECURITY_METRICS = "Security metrics not available"
requires_security_metrics = not HAS_SECURITY_METRICS

# Debate and evolution modules (commonly skipped)
HAS_RHETORICAL_OBSERVER = _check_aragora_module("aragora.debate.rhetorical_observer")
REQUIRES_RHETORICAL_OBSERVER = "RhetoricalObserver module not available"
requires_rhetorical_observer = not HAS_RHETORICAL_OBSERVER

HAS_INTROSPECTION = _check_aragora_module("aragora.introspection")
REQUIRES_INTROSPECTION = "Introspection module not available"
requires_introspection = not HAS_INTROSPECTION

HAS_EVOLUTION = _check_aragora_module("aragora.evolution")
REQUIRES_EVOLUTION = "Evolution module not available"
requires_evolution = not HAS_EVOLUTION

HAS_BREEDING = _check_aragora_module("aragora.evolution.breeding")
REQUIRES_BREEDING = "Breeding module not available"
requires_breeding = not HAS_BREEDING

HAS_GENESIS = _check_aragora_module("aragora.genesis")
REQUIRES_GENESIS = "Genesis module not available"
requires_genesis = not HAS_GENESIS

HAS_PHASES = _check_aragora_module("aragora.debate.phases")
REQUIRES_PHASES = "Phase modules not available"
requires_phases = not HAS_PHASES

HAS_NOVELTY_TRACKER = _check_aragora_module("aragora.evolution.novelty")
REQUIRES_NOVELTY_TRACKER = "NoveltyTracker module not available"
requires_novelty_tracker = not HAS_NOVELTY_TRACKER

HAS_CULTURE_MANAGER = _check_aragora_module("aragora.organization.culture")
REQUIRES_CULTURE_MANAGER = "OrganizationCultureManager not available"
requires_culture_manager = not HAS_CULTURE_MANAGER

HAS_MEMORY_ANALYTICS = _check_aragora_module("aragora.server.handlers.memory")
REQUIRES_MEMORY_ANALYTICS = "MemoryAnalyticsHandler not available"
requires_memory_analytics = not HAS_MEMORY_ANALYTICS


def _check_handlers_available() -> bool:
    """Check if handler registry is available."""
    try:
        from aragora.server.handler_registry import HANDLERS_AVAILABLE

        return HANDLERS_AVAILABLE
    except ImportError:
        return False


HAS_HANDLERS = _check_handlers_available()
REQUIRES_HANDLERS = "Handlers not available"
requires_handlers = not HAS_HANDLERS


# ============================================================================
# CI Environment Detection
# ============================================================================
# Detect common CI environment variables
RUNNING_IN_CI = any(
    os.environ.get(var)
    for var in [
        "CI",  # Generic CI flag (GitHub Actions, GitLab CI, etc.)
        "GITHUB_ACTIONS",  # GitHub Actions
        "GITLAB_CI",  # GitLab CI
        "CIRCLECI",  # CircleCI
        "JENKINS_URL",  # Jenkins
        "TRAVIS",  # Travis CI
        "BUILDKITE",  # Buildkite
    ]
)
REQUIRES_NO_CI = "Test skipped in CI environment"
requires_no_ci = RUNNING_IN_CI


# ============================================================================
# Test Tier Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom pytest markers and configure test environment.

    Test Tiers:
    - smoke: Quick sanity tests for CI (<5 min total)
    - integration: Tests requiring external dependencies (APIs, DBs)
    - slow: Long-running tests (>30s each)

    CI Strategy:
    - PR CI: pytest -m "not slow and not integration" (~5 min)
    - Nightly: pytest (full suite)

    Environment Configuration:
    - Sets ARAGORA_AUTH_CLEANUP_INTERVAL to 1 second for fast test cleanup.
      This prevents the 300-second default from blocking test completion.

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
    # Set fast auth cleanup interval for tests (1 second instead of 300)
    # This prevents test timeouts caused by long cleanup waits
    if "ARAGORA_AUTH_CLEANUP_INTERVAL" not in os.environ:
        os.environ["ARAGORA_AUTH_CLEANUP_INTERVAL"] = "1"

    config.addinivalue_line("markers", "smoke: quick sanity tests for fast CI feedback")
    config.addinivalue_line(
        "markers", "integration: tests requiring external dependencies (APIs, databases)"
    )
    config.addinivalue_line("markers", "slow: long-running tests (>30 seconds)")
    config.addinivalue_line("markers", "unit: isolated unit tests with no external dependencies")
    config.addinivalue_line(
        "markers", "network: tests requiring external network calls (skip with -m 'not network')"
    )
    config.addinivalue_line(
        "markers", "no_auto_auth: disable automatic auth bypass for handler tests"
    )


# ============================================================================
# RBAC Bypass for Root-Level Handler Tests
# ============================================================================


@pytest.fixture(autouse=True)
def _bypass_rbac_for_root_handler_tests(request, monkeypatch):
    """Auto-bypass RBAC for root-level test_handlers_*.py files.

    The tests/server/handlers/ directory has its own conftest with comprehensive
    auth bypass. Root-level handler test files (tests/test_handlers_*.py) also
    call handler methods directly but lack RBAC context. This fixture provides
    a minimal bypass for those files only.
    """
    # Only activate for root-level handler test files
    test_file = request.fspath.basename
    if not test_file.startswith("test_handlers_") and not test_file.startswith(
        "test_agents_handler"
    ):
        yield
        return

    # Respect no_auto_auth marker
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    try:
        from aragora.rbac import decorators
        from aragora.rbac.models import AuthorizationContext

        mock_auth_ctx = AuthorizationContext(
            user_id="test-user-001",
            org_id="test-org-001",
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

    # Also bypass the PermissionChecker
    try:
        from aragora.rbac.checker import get_permission_checker
        from aragora.rbac.models import AuthorizationDecision

        checker = get_permission_checker()

        def _always_allow(context, permission_key, resource_id=None):
            return AuthorizationDecision(
                allowed=True,
                reason="Test bypass",
                permission_key=permission_key,
            )

        monkeypatch.setattr(checker, "check_permission", _always_allow)
    except (ImportError, AttributeError):
        pass

    # Bypass handler-level require_permission decorator (separate from RBAC).
    # The handlers.utils.decorators.require_permission uses _test_user_context_override
    # and extract_user_from_request from billing.jwt_auth, which must also be patched.
    try:
        from aragora.server.handlers.utils import decorators as handler_decorators
        from aragora.billing.auth.context import UserAuthContext

        mock_user_ctx = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id="test-org-001",
            role="admin",
            token_type="access",
        )

        monkeypatch.setattr(handler_decorators, "_test_user_context_override", mock_user_ctx)
        monkeypatch.setattr(handler_decorators, "has_permission", lambda role, perm: True)
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.billing.auth.context import UserAuthContext as _UAC

        _mock_user = _UAC(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id="test-org-001",
            role="admin",
            token_type="access",
        )

        monkeypatch.setattr(
            "aragora.billing.jwt_auth.extract_user_from_request",
            lambda handler, user_store=None: _mock_user,
        )
    except (ImportError, AttributeError):
        pass

    yield


# ============================================================================
# Global Test Setup
# ============================================================================


@pytest.fixture(autouse=True, scope="session")
def _preinstall_fake_sentence_transformers():
    """Install a lightweight fake sentence_transformers module into sys.modules.

    The real sentence_transformers package takes ~30s to import because it drags
    in the entire huggingface transformers library. This causes the very first
    test in any file to exceed pytest-timeout and hang.

    By pre-installing a fake module at session scope, we prevent the real import
    from ever happening. The per-test mock_sentence_transformers fixture then
    patches specific attributes on this fake module as needed.
    """
    import sys
    import types

    import numpy as np

    # Only install fake if the real module isn't already loaded
    if "sentence_transformers" in sys.modules:
        yield
        return

    class _FakeSentenceTransformer:
        def __init__(self, model_name_or_path=None, **kwargs):
            self.model_name = model_name_or_path or "mock-model"
            self._embedding_dim = 384

        def encode(self, sentences, **kwargs):
            single = isinstance(sentences, str)
            if single:
                sentences = [sentences]
            result = np.array(
                [np.random.RandomState(hash(t) % 2**32).randn(384).astype(np.float32)
                 for t in sentences]
            )
            return result[0] if single else result

        def get_sentence_embedding_dimension(self):
            return self._embedding_dim

    class _FakeCrossEncoder:
        def __init__(self, model_name=None, **kwargs):
            self.model_name = model_name or "mock-cross-encoder"

        def predict(self, sentence_pairs, **kwargs):
            if not sentence_pairs:
                return np.array([])
            return np.array([[0.1, 0.8, 0.1]] * len(sentence_pairs))

    # Create fake module hierarchy
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeSentenceTransformer
    fake_st.CrossEncoder = _FakeCrossEncoder
    fake_st.__version__ = "0.0.0-test-fake"

    # Also create submodules that might be imported
    for sub in ("cross_encoder", "backend", "models", "util"):
        fake_sub = types.ModuleType(f"sentence_transformers.{sub}")
        sys.modules[f"sentence_transformers.{sub}"] = fake_sub

    # The cross_encoder submodule needs CrossEncoder
    sys.modules["sentence_transformers.cross_encoder"].CrossEncoder = _FakeCrossEncoder

    saved = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_st

    yield

    # Restore original state
    if saved is not None:
        sys.modules["sentence_transformers"] = saved
    else:
        sys.modules.pop("sentence_transformers", None)
    for sub in ("cross_encoder", "backend", "models", "util"):
        sys.modules.pop(f"sentence_transformers.{sub}", None)


@pytest.fixture(autouse=True, scope="session")
def _suppress_auth_cleanup_threads():
    """Prevent AuthConfig from spawning background cleanup threads.

    AuthConfig.__init__ calls _start_cleanup_thread() which spawns a daemon
    thread. Many AuthConfig instances are created across tests (mock_auth_config
    fixture, direct instantiation, module-level singleton). Without this fix,
    dozens of daemon threads accumulate and can cause pytest shutdown to hang.

    This session-scoped autouse fixture patches _start_cleanup_thread to a
    no-op and stops any already-running thread on the module-level singleton.
    """
    # Ensure production-mode env vars don't leak into auth module import.
    # Earlier tests (or the outer shell) may set ARAGORA_ENV=production which
    # causes auth_config.configure_from_env() to raise at import time.
    saved_env = os.environ.get("ARAGORA_ENV")
    if saved_env == "production":
        os.environ["ARAGORA_ENV"] = "development"

    try:
        from aragora.server.auth import AuthConfig, auth_config
    except Exception:
        # If import still fails, nothing to suppress
        if saved_env is not None:
            os.environ["ARAGORA_ENV"] = saved_env
        yield
        return

    # Stop the thread on the module-level singleton (spawned at import time)
    auth_config.stop_cleanup_thread()

    # Patch the class method so future instances don't spawn threads
    original = AuthConfig._start_cleanup_thread
    AuthConfig._start_cleanup_thread = lambda self: None

    yield

    # Restore original method
    AuthConfig._start_cleanup_thread = original
    if saved_env is not None:
        os.environ["ARAGORA_ENV"] = saved_env


@pytest.fixture
def stop_auth_cleanup():
    """Fixture to stop auth cleanup threads after tests.

    Use this fixture for tests that create AuthConfig instances.
    It yields a function that stops the cleanup thread, and also
    cleans up on teardown.

    Usage:
        def test_auth_config(stop_auth_cleanup):
            from aragora.server.auth import AuthConfig
            config = AuthConfig()
            # ... test code ...
            stop_auth_cleanup(config)
    """
    configs = []

    def _stop(auth_config):
        configs.append(auth_config)
        if hasattr(auth_config, "stop_cleanup_thread"):
            auth_config.stop_cleanup_thread()

    yield _stop

    # Cleanup any remaining configs
    for config in configs:
        if hasattr(config, "stop_cleanup_thread"):
            try:
                config.stop_cleanup_thread()
            except Exception:
                pass


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
def reset_continuum_memory_singleton():
    """Reset ContinuumMemory singleton between tests.

    Prevents cross-test pollution via the global ContinuumMemory instance.
    """
    try:
        from aragora.memory.continuum.singleton import reset_continuum_memory
    except Exception:
        yield
        return

    reset_continuum_memory()
    yield
    reset_continuum_memory()


@pytest.fixture(autouse=True)
def mock_sentence_transformers(request, monkeypatch):
    """Mock SentenceTransformer to prevent HuggingFace model downloads.

    This prevents tests from making network calls to HuggingFace Hub,
    which can cause timeouts and flaky tests. Tests marked @pytest.mark.slow
    that need real embeddings are excluded.

    The mock returns deterministic embeddings based on input text hash,
    ensuring consistent behavior across test runs.
    """
    import sys

    import numpy as np

    # Clear embedding service cache to ensure fresh instances per test.
    # IMPORTANT: Use sys.modules lookup instead of import to avoid triggering
    # the heavy sentence_transformers/transformers import chain (~30s) which
    # causes pytest timeout failures.
    emb_module = sys.modules.get("aragora.ml.embeddings")
    if emb_module is not None:
        try:
            emb_module._embedding_services.clear()
        except AttributeError:
            pass

    # Skip for slow tests that may need real embeddings
    if "slow" in [m.name for m in request.node.iter_markers()]:
        yield
        # Clear cache after slow test too
        if emb_module is not None:
            try:
                emb_module._embedding_services.clear()
            except AttributeError:
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

    # Mock at the sentence_transformers module level.
    # IMPORTANT: Only patch if already imported. Do NOT trigger the heavy
    # sentence_transformers/transformers import chain (~30s) which exceeds
    # pytest-timeout and causes test hangs.
    st_mod = sys.modules.get("sentence_transformers")
    if st_mod is not None:
        monkeypatch.setattr(st_mod, "SentenceTransformer", MockSentenceTransformer)
        if hasattr(st_mod, "CrossEncoder"):
            monkeypatch.setattr(st_mod, "CrossEncoder", MockCrossEncoder)

    # Patch modules that have already imported SentenceTransformer/CrossEncoder
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
        mod = sys.modules.get(module_path)
        if mod is None:
            continue
        if hasattr(mod, "SentenceTransformer"):
            monkeypatch.setattr(mod, "SentenceTransformer", MockSentenceTransformer)
        if hasattr(mod, "CrossEncoder"):
            monkeypatch.setattr(mod, "CrossEncoder", MockCrossEncoder)

    yield


@pytest.fixture(autouse=True)
def mock_semantic_store_embeddings(request, monkeypatch):
    """Force SemanticStore to use hash-based EmbeddingProvider instead of API-based.

    Without this, SemanticStore._auto_detect_provider() picks OpenAI/Gemini when
    API keys are set, causing real HTTP calls via aiohttp. Under load (thousands of
    tests), these hit rate limits and the exponential backoff retries cause hangs
    that can't be interrupted by pytest-timeout (stuck in C-level asyncio selector).
    """
    markers = [m.name for m in request.node.iter_markers()]
    if "network" in markers or "integration" in markers or "slow" in markers:
        yield
        return

    try:
        from aragora.memory.embeddings import EmbeddingProvider

        monkeypatch.setattr(
            "aragora.knowledge.mound.semantic_store.SemanticStore._auto_detect_provider",
            lambda self: EmbeddingProvider(dimension=256),
        )
    except (ImportError, AttributeError):
        pass

    yield


@pytest.fixture(autouse=True)
def _disable_rate_limiting(request, monkeypatch):
    """Disable handler rate limiters to prevent xdist cross-test interference.

    Under xdist, rate limiter singletons accumulate state from tests running
    on the same worker process, causing unrelated tests to receive 429 instead
    of their expected status codes.

    Tests that specifically exercise rate-limiting behavior should use
    @pytest.mark.rate_limit_test to opt out and get real rate limiting.
    """
    markers = [m.name for m in request.node.iter_markers()]
    if "rate_limit_test" in markers:
        yield
        return

    try:
        import aragora.server.handlers.utils.rate_limit as rl_mod

        monkeypatch.setattr(rl_mod, "RATE_LIMITING_DISABLED", True)
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
    force_mock = os.environ.get("ARAGORA_FORCE_MOCK_APIS", "").lower() in ("1", "true", "yes")
    markers = [m.name for m in request.node.iter_markers()]
    if ("network" in markers or "integration" in markers) and not force_mock:
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
            self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
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
            self.request = type("Request", (), {"method": "GET", "url": ""})()

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
            self.headers: dict[str, str] = {}

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

        async def head(self, url, **kwargs):
            return self._make_response(url, **kwargs)

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
def mock_auth_config():
    """Create a mock AuthConfig.

    Returns an AuthConfig configured for authentication testing.
    Cleanup thread is suppressed by _suppress_auth_cleanup_threads.
    """
    from aragora.server.auth import AuthConfig

    config = AuthConfig()
    config.api_token = "test_secret_key_12345"
    config.enabled = True
    config.rate_limit_per_minute = 60
    config.ip_rate_limit_per_minute = 120
    yield config
    config.stop_cleanup_thread()


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


@pytest.fixture(autouse=True)
def test_environment(monkeypatch):
    """Set test environment variables for all tests.

    This fixture configures the environment for testing:
    - ARAGORA_API_TOKEN: Provides auth token to prevent AuthenticationError
    - ARAGORA_REQUIRE_DISTRIBUTED: Disables distributed mode requirement
    - ARAGORA_SSRF_ALLOW_LOCALHOST: Allows localhost URLs for integration tests
    """
    monkeypatch.setenv("ARAGORA_API_TOKEN", "test-token")
    monkeypatch.setenv("ARAGORA_REQUIRE_DISTRIBUTED", "false")
    monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")
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

    # Reset additional global singletons/caches that commonly pollute tests.
    # Keep this best-effort: optional modules may not be importable in all envs.
    try:
        from aragora.core.embeddings.cache import reset_caches

        reset_caches()
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.core.embeddings.service import reset_embedding_service

        reset_embedding_service()
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.rlm.factory import reset_singleton

        reset_singleton()
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.reasoning.evidence_bridge import reset_evidence_bridge

        reset_evidence_bridge()
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.observability.incident_store import reset_incident_store

        reset_incident_store()
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.observability.slo_history import reset_slo_history_store

        reset_slo_history_store()
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.events.cross_subscribers import reset_cross_subscriber_manager

        reset_cross_subscriber_manager()
    except (ImportError, AttributeError):
        pass

    # Clear rate limiters to prevent test pollution
    try:
        from aragora.server.handlers.utils.rate_limit import clear_all_limiters

        clear_all_limiters()
    except (ImportError, AttributeError):
        pass

    # Reset distributed rate limiter singleton to prevent cross-test pollution.
    # The distributed limiter has its own internal memory backend that accumulates
    # state independently of the _limiters registry cleared above.
    try:
        from aragora.server.middleware.rate_limit.distributed import reset_distributed_limiter

        reset_distributed_limiter()
    except (ImportError, AttributeError):
        pass

    # Clear ALL module-level RateLimiter instances across loaded aragora modules.
    # This replaces individual per-module cleanup blocks with a single loop that
    # discovers every RateLimiter in any loaded aragora.* module, preventing
    # order-dependent test failures when new handlers add limiters.
    try:
        import sys

        from aragora.server.handlers.utils.rate_limit import RateLimiter as _RL

        for mod in list(sys.modules.values()):
            mod_name = getattr(mod, "__name__", "") or ""
            if not mod_name.startswith("aragora."):
                continue
            for attr_name in dir(mod):
                if not attr_name.startswith("_") or attr_name.startswith("__"):
                    continue
                try:
                    obj = getattr(mod, attr_name, None)
                    if isinstance(obj, _RL):
                        obj.clear()
                except Exception:
                    pass
    except ImportError:
        pass

    # Clear all registered @lru_cache instances
    try:
        from aragora.utils.cache_registry import clear_all_lru_caches

        clear_all_lru_caches()
    except (ImportError, AttributeError):
        pass

    # Reset deletion coordinator singleton
    try:
        import aragora.deletion_coordinator as _dc

        _dc._coordinator_instance = None
    except (ImportError, AttributeError):
        pass

    # Reset global moderation singleton
    try:
        import aragora.moderation.spam_integration as _spam

        _spam._global_moderation = None
    except (ImportError, AttributeError):
        pass

    # Reset whisper backend instances
    try:
        import aragora.transcription.whisper_backend as _wb

        _wb._backend_instances = {}
    except (ImportError, AttributeError):
        pass

    # Reset RBAC PermissionChecker singleton
    try:
        import aragora.rbac.checker as _rbac_checker

        _rbac_checker._permission_checker = None
    except (ImportError, AttributeError):
        pass

    # Reset decision metrics singleton state
    try:
        import aragora.observability.decision_metrics as _dm

        _dm._initialized = False
        _dm.DECISION_REQUESTS = None
        _dm.DECISION_RESULTS = None
        _dm.DECISION_LATENCY = None
        _dm.DECISION_CONFIDENCE = None
        _dm.DECISION_CACHE_HITS = None
        _dm.DECISION_CACHE_MISSES = None
        _dm.DECISION_DEDUP_HITS = None
        _dm.DECISION_ACTIVE = None
        _dm.DECISION_ERRORS = None
        _dm.DECISION_CONSENSUS_RATE = None
        _dm.DECISION_AGENTS_USED = None
    except (ImportError, AttributeError):
        pass

    # Reset SLO metrics singleton state
    try:
        import aragora.observability.slo as _slo

        _slo._slo_metrics_initialized = False
        _slo.SLO_COMPLIANCE = None
        _slo.SLO_ERROR_BUDGET = None
        _slo.SLO_BURN_RATE = None
    except (ImportError, AttributeError):
        pass

    # Reset OTel tracing state
    try:
        import aragora.observability.otel as _otel

        _otel._initialized = False
        _otel._tracer_provider = None
        _otel._tracers.clear()
    except (ImportError, AttributeError):
        pass

    # Reset unified audit logger singleton
    try:
        import aragora.audit.unified as _unified_audit

        _unified_audit._unified_logger = None
    except (ImportError, AttributeError):
        pass

    # Reset event dispatcher singletons
    try:
        import aragora.events.dispatcher as _evt

        _evt._event_rate_limiter = None
        _evt._dispatcher = None
    except (ImportError, AttributeError):
        pass

    # Reset ELO system singleton and class-level caches to prevent
    # cross-test contamination via shared mutable state
    try:
        import aragora.ranking.elo as _elo_mod

        _elo_mod._elo_store = None
        _elo_mod.EloSystem._rating_cache.clear()
        _elo_mod.EloSystem._leaderboard_cache.clear()
        _elo_mod.EloSystem._stats_cache.clear()
        _elo_mod.EloSystem._calibration_cache.clear()
    except (ImportError, AttributeError):
        pass

    # Reset approval gate in-memory state to prevent cross-test pollution
    # via the module-level _pending_approvals dict and _last_cleanup_time
    try:
        import aragora.server.middleware.approval_gate as _approval_gate

        _approval_gate._pending_approvals.clear()
        _approval_gate._last_cleanup_time = 0.0
    except (ImportError, AttributeError):
        pass

    # Reset store metrics _initialized flag to prevent Prometheus
    # CollectorRegistry conflicts (ValueError: Duplicated timeseries)
    try:
        import aragora.observability.metrics.stores as _store_metrics

        _store_metrics._initialized = False
    except (ImportError, AttributeError):
        pass

    # Reset gauntlet signing singleton to prevent stale HMAC keys from one
    # test file leaking into another (each ReceiptSigner generates an
    # ephemeral key on creation, so a cached signer breaks verification).
    try:
        import aragora.gauntlet.signing as _signing

        _signing._default_signer = None
    except (ImportError, AttributeError):
        pass

    # Reset encryption service singleton and SecretManager cache so tests
    # that manipulate ARAGORA_ENCRYPTION_KEY or ARAGORA_ENV don't poison
    # other test files (e.g. test_service_generates_ephemeral_key_without_env).
    try:
        import aragora.security.encryption as _enc

        _enc._encryption_service = None
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.config.secrets import reset_secret_manager

        reset_secret_manager()
    except (ImportError, AttributeError):
        pass

    # Reset embedding provider singleton so tests that configure custom
    # providers don't leak into subsequent test files.
    try:
        import aragora.embeddings as _embed

        _embed._default_provider = None
    except (ImportError, AttributeError):
        pass

    # Reset connector registry singleton to prevent cross-test pollution.
    try:
        from aragora.connectors.runtime_registry import ConnectorRegistry

        ConnectorRegistry.reset()
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
    - Rate limiters (via clear_all_limiters, distributed reset, and universal cleanup):
      - _limiters registry (all auth_rate_limit and rate_limit decorators)
      - DistributedRateLimiter singleton (reset_distributed_limiter)
      - ALL module-level RateLimiter instances in loaded aragora.* modules
        (discovered dynamically via isinstance check, no manual enumeration needed)
    - aragora.rbac.checker._permission_checker (PermissionChecker singleton)
    - aragora.ranking.elo._elo_store (EloSystem singleton + class-level TTL caches)
    - aragora.observability.decision_metrics (11 metric globals + _initialized)
    - aragora.observability.slo (3 SLO metric globals + _slo_metrics_initialized)
    - aragora.observability.otel (_initialized, _tracer_provider, _tracers)
    - aragora.events.dispatcher (_event_rate_limiter, _dispatcher)
    - aragora.audit.unified._unified_logger (UnifiedAuditLogger singleton)
    - aragora.server.middleware.approval_gate (_pending_approvals, _last_cleanup_time)
    - aragora.observability.metrics.stores (_initialized flag for Prometheus re-registration)
    - aragora.gauntlet.signing._default_signer (ReceiptSigner singleton)
    - aragora.security.encryption._encryption_service (EncryptionService singleton)
    - aragora.config.secrets SecretManager (cached encryption keys)
    - aragora.embeddings._default_provider (EmbeddingProvider singleton)
    - aragora.connectors.runtime_registry.ConnectorRegistry._instance
    """
    _reset_lazy_globals_impl()  # Reset BEFORE test
    yield
    _reset_lazy_globals_impl()  # Reset AFTER test


@pytest.fixture(autouse=True)
def _clear_config_legacy_cache():
    """Clear any cached legacy constants from aragora.config globals.

    The config package's ``__getattr__`` previously cached legacy names
    (e.g. ``DEFAULT_CONSENSUS``) in ``globals()`` on first access, causing
    tests that modify the underlying environment variables to read stale
    values.  The caching has been removed, but this fixture acts as a
    safety belt: it scrubs any legacy names that may have leaked into the
    module's global dict between tests so that ``__getattr__`` is always
    invoked on the next access.
    """
    yield
    try:
        import aragora.config as _cfg

        _legacy = getattr(_cfg, "_LEGACY_NAMES", set())
        _slo = getattr(_cfg, "_SLO_NAMES", set())
        _to_clear = _legacy | _slo | {"DEFAULT_AGENT_LIST"}
        _g = vars(_cfg)
        for name in _to_clear:
            _g.pop(name, None)
    except Exception:
        pass


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
# Additional Skip Markers for Common Scenarios
# ============================================================================
# These markers consolidate common pytest.skip() patterns into proper skip markers.

# Cryptography library (used for JWT, encryption)
HAS_CRYPTOGRAPHY = _check_import("cryptography")
REQUIRES_CRYPTOGRAPHY = "cryptography not installed (pip install cryptography)"
requires_cryptography = not HAS_CRYPTOGRAPHY

# Tree-sitter for code parsing
HAS_TREE_SITTER = _check_import("tree_sitter")
REQUIRES_TREE_SITTER = "tree-sitter not installed"
requires_tree_sitter = not HAS_TREE_SITTER

# Whisper for transcription
HAS_WHISPER = _check_import("whisper")
REQUIRES_WHISPER = "whisper not installed"
requires_whisper = not HAS_WHISPER

# Z3 solver (expanded from existing)
# Note: HAS_Z3 defined earlier in file


def _has_z3_binary() -> bool:
    """Check if Z3 binary is available and working."""
    try:
        import z3

        solver = z3.Solver()
        x = z3.Int("x")
        solver.add(x > 0)
        return solver.check() == z3.sat
    except (ImportError, Exception):
        return False


HAS_Z3_WORKING = _has_z3_binary()
REQUIRES_Z3_WORKING = "Z3 solver not installed or not working"
requires_z3_working = not HAS_Z3_WORKING

# Lean theorem prover
HAS_LEAN = _check_import("lean")
REQUIRES_LEAN = "Lean theorem prover not installed"
requires_lean = not HAS_LEAN

# pydub for audio processing
HAS_PYDUB = _check_import("pydub")
REQUIRES_PYDUB = "pydub not installed (pip install pydub)"
requires_pydub = not HAS_PYDUB

# WeasyPrint for PDF generation
HAS_WEASYPRINT = _check_import("weasyprint")
REQUIRES_WEASYPRINT = "WeasyPrint not installed (pip install weasyprint)"
requires_weasyprint = not HAS_WEASYPRINT

# Milvus vector database
HAS_MILVUS = _check_import("pymilvus")
REQUIRES_MILVUS = "pymilvus not installed"
requires_milvus = not HAS_MILVUS

# aiohttp for async HTTP
HAS_AIOHTTP = _check_import("aiohttp")
REQUIRES_AIOHTTP = "aiohttp not installed (pip install aiohttp)"
requires_aiohttp = not HAS_AIOHTTP


# FFmpeg for video processing
def _has_ffmpeg() -> bool:
    """Check if FFmpeg is available on PATH."""
    import shutil

    return shutil.which("ffmpeg") is not None


HAS_FFMPEG = _has_ffmpeg()
REQUIRES_FFMPEG = "FFmpeg not available in PATH"
requires_ffmpeg = not HAS_FFMPEG


def _has_git() -> bool:
    """Check if git is available on PATH."""
    import shutil

    return shutil.which("git") is not None


HAS_GIT = _has_git()
REQUIRES_GIT = "git not available in PATH"
requires_git = not HAS_GIT


# Platform-specific capabilities
def _supports_symlinks() -> bool:
    """Check if the system supports symlinks."""
    import os
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test")
            link_path = os.path.join(tmpdir, "link")
            with open(test_file, "w") as f:
                f.write("test")
            os.symlink(test_file, link_path)
            return True
    except (OSError, NotImplementedError):
        return False


HAS_SYMLINKS = _supports_symlinks()
REQUIRES_SYMLINKS = "Symlink creation not supported on this platform"
requires_symlinks = not HAS_SYMLINKS


def _supports_signals() -> bool:
    """Check if the system supports signal-based timeouts (Unix-like)."""
    import os

    return os.name != "nt"  # Not Windows


HAS_SIGNALS = _supports_signals()
REQUIRES_SIGNALS = "Signal-based timeout not available on Windows"
requires_signals = not HAS_SIGNALS


# PostgreSQL database availability
def _has_postgres_configured() -> bool:
    """Check if PostgreSQL is configured via environment."""
    database_url = os.environ.get("DATABASE_URL", "")
    return "postgres" in database_url.lower()


HAS_POSTGRES_CONFIGURED = _has_postgres_configured()
REQUIRES_POSTGRES = "PostgreSQL not configured (set DATABASE_URL)"
requires_postgres = not HAS_POSTGRES_CONFIGURED


# ============================================================================
# Skip Count Monitoring
# ============================================================================
# Track skip counts to warn when threshold is exceeded.
# See tests/SKIP_AUDIT.md for skip marker inventory.

SKIP_THRESHOLD = 200  # Raised from 150 to accommodate contract matrix parametrized skips
UNCONDITIONAL_SKIP_THRESHOLD = (
    0  # No unconditional @pytest.mark.skip allowed (was 1, converted last one to xfail)
)


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


# ============================================================================
# Global Mock Pollution Guards
# ============================================================================
# These guards repair module-level attributes that tests may accidentally
# replace with mocks.  The more detailed handler-specific fixture is in
# tests/server/handlers/conftest.py; this lightweight version covers test
# files outside that directory tree.

# Capture real references at import time
try:
    from aragora.utils.async_utils import run_async as _global_real_run_async
except ImportError:
    _global_real_run_async = None

_global_real_extract_path_param = None
try:
    from aragora.server.handlers.base import BaseHandler as _GlobalBaseHandler

    _global_real_extract_path_param = getattr(_GlobalBaseHandler, "extract_path_param", None)
except ImportError:
    _GlobalBaseHandler = None

# Capture the side_effect property descriptor
from unittest.mock import NonCallableMock as _GlobalNCMock

_global_side_effect_descriptor = None
for _klass in _GlobalNCMock.__mro__:
    if "side_effect" in _klass.__dict__:
        _global_side_effect_descriptor = _klass.__dict__["side_effect"]
        break


@pytest.fixture(autouse=True)
def _global_mock_pollution_guard():
    """Repair mock pollution that can leak across test files."""
    import sys

    # Repair MagicMock.side_effect property descriptor
    if _global_side_effect_descriptor is not None:
        current = _GlobalNCMock.__dict__.get("side_effect")
        if current is not _global_side_effect_descriptor:
            _GlobalNCMock.side_effect = _global_side_effect_descriptor

    # Restore BaseHandler.extract_path_param
    if _GlobalBaseHandler is not None and _global_real_extract_path_param is not None:
        current = getattr(_GlobalBaseHandler, "extract_path_param", None)
        if current is not _global_real_extract_path_param:
            setattr(_GlobalBaseHandler, "extract_path_param", _global_real_extract_path_param)

    # Restore run_async in loaded modules
    if _global_real_run_async is not None:
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith(("aragora.server.", "aragora.utils.")):
                continue
            for attr in ("run_async", "_run_async"):
                current = getattr(mod, attr, None)
                if current is not None and current is not _global_real_run_async:
                    setattr(mod, attr, _global_real_run_async)

    yield

    # Teardown: same repairs
    if _global_side_effect_descriptor is not None:
        current = _GlobalNCMock.__dict__.get("side_effect")
        if current is not _global_side_effect_descriptor:
            _GlobalNCMock.side_effect = _global_side_effect_descriptor

    if _GlobalBaseHandler is not None and _global_real_extract_path_param is not None:
        current = getattr(_GlobalBaseHandler, "extract_path_param", None)
        if current is not _global_real_extract_path_param:
            setattr(_GlobalBaseHandler, "extract_path_param", _global_real_extract_path_param)

    if _global_real_run_async is not None:
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith(("aragora.server.", "aragora.utils.")):
                continue
            for attr in ("run_async", "_run_async"):
                current = getattr(mod, attr, None)
                if current is not None and current is not _global_real_run_async:
                    setattr(mod, attr, _global_real_run_async)
