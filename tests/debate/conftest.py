"""
Conftest for debate tests.

Re-synchronizes module class references after each test to prevent
isinstance() failures caused by importlib.reload() of the similarity
backends module in tests/debate/similarity/.
"""

import asyncio
import concurrent.futures
import os
import sys
import threading

import pytest

# ---------------------------------------------------------------------------
# Agent class pollution guard
# ---------------------------------------------------------------------------
# Capture the real Agent.__init__ at import time.  Mock pollution from tests
# in other directories can corrupt the Agent class (e.g. by destroying the
# NonCallableMock.side_effect descriptor, which cascades into failures that
# prevent Agent.__init__ from running properly).  This fixture restores
# Agent.__init__ before and after every debate test.
from aragora.core_types import Agent as _RealAgent

_real_agent_init = _RealAgent.__init__


@pytest.fixture(autouse=True)
def _protect_agent_class():
    """Guard against mock pollution that corrupts Agent.__init__.

    Without this, random test ordering can cause:
        AttributeError: 'Agent' object has no attribute 'role'
    in roles_manager.assign_initial_roles() because Agent.__init__
    never ran (or was replaced by a mock).
    """
    # Setup: restore before the test runs
    if _RealAgent.__init__ is not _real_agent_init:
        _RealAgent.__init__ = _real_agent_init

    yield

    # Teardown: restore after the test runs
    if _RealAgent.__init__ is not _real_agent_init:
        _RealAgent.__init__ = _real_agent_init


@pytest.fixture(autouse=True)
def _isolate_debate_databases(tmp_path, monkeypatch):
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
    # Prevent background LLM classification from making real API calls
    # (QuestionClassifier.classify() creates an AsyncAnthropic client that
    # opens TCP connections which keep the event loop alive).
    monkeypatch.setenv("ARAGORA_OFFLINE", "1")
    # Prevent Pulse ingestors from making real HTTP calls to Google Trends,
    # HackerNews, Reddit, GitHub during tests.  The ContextGatherer checks
    # this env var to skip trending topic fetching entirely.
    monkeypatch.setenv("ARAGORA_DISABLE_TRENDING", "1")
    # Prevent real Slack API calls from notification providers
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)


@pytest.fixture(autouse=True)
def _clear_similarity_backend_state():
    """Clear all similarity backend caches after each test.

    Prevents cross-test pollution from:
    - Cached similarity computations (JaccardBackend, TFIDFBackend)
    - Factory registry state (SimilarityFactory)
    - Cached ML models (SentenceTransformerBackend)

    Without this, pytest-randomly can cause failures when tests that populate
    caches run before tests that expect clean state.
    """
    yield

    try:
        from aragora.debate.similarity.backends import (
            JaccardBackend,
            SentenceTransformerBackend,
            TFIDFBackend,
        )

        JaccardBackend.clear_cache()
        TFIDFBackend.clear_cache()
        SentenceTransformerBackend.clear_cache()
        SentenceTransformerBackend._model_cache = None
        SentenceTransformerBackend._model_name_cache = None
        SentenceTransformerBackend._nli_model_cache = None
        SentenceTransformerBackend._nli_model_name_cache = None
    except ImportError:
        pass

    try:
        from aragora.debate.similarity.factory import SimilarityFactory

        # Re-initialize rather than just clear — other tests may depend on
        # the factory being populated with default backends.
        SimilarityFactory._registry.clear()
        SimilarityFactory._initialized = False
        SimilarityFactory._ensure_initialized()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _mock_scan_code_markers(request, monkeypatch):
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
            auto_gauntlet_validate=False,
            auto_verify_arguments=False,
            auto_push_calibration=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_persist_receipt=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            auto_llm_judge=False,
        )
        monkeypatch.setattr(pdc_mod, "DEFAULT_POST_DEBATE_CONFIG", patched)

        # Patch specific step methods that make real LLM/async calls and block
        # indefinitely in tests. These are the steps where tests create their
        # own PostDebateConfig, bypassing the DEFAULT_POST_DEBATE_CONFIG patch.
        def _noop_llm_judge(self, *args, **kwargs):
            return None

        def _noop_outcome_feedback(self, *args, **kwargs):
            return None

        monkeypatch.setattr(pdc_mod.PostDebateCoordinator, "_step_llm_judge", _noop_llm_judge)
        monkeypatch.setattr(
            pdc_mod.PostDebateCoordinator, "_step_outcome_feedback", _noop_outcome_feedback
        )
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _resync_all_backend_refs():
    """Re-synchronize ALL module class references after each test.

    Some tests call importlib.reload() on the backends module, which
    creates new class objects.  Other modules (convergence, test modules)
    still hold references to the old classes, causing isinstance()
    failures or stale class-level state (e.g. _similarity_cache) in
    tests that run later.

    This fixture is intentionally named differently from the child
    conftest fixture ``_resync_convergence_after_backend_reload`` in
    tests/debate/similarity/conftest.py.  Pytest picks the closest
    fixture when names collide, so a same-named parent fixture would
    be shadowed by the child.  Using a distinct name ensures both run.
    """
    yield

    backends_mod = sys.modules.get("aragora.debate.similarity.backends")
    if backends_mod is None:
        return

    _SYNCED_NAMES = [
        "JaccardBackend",
        "TFIDFBackend",
        "SentenceTransformerBackend",
        "SimilarityBackend",
        "get_similarity_backend",
    ]

    for mod_name in list(sys.modules):
        if not (mod_name.startswith("aragora.debate") or mod_name.startswith("tests.debate")):
            continue
        if mod_name == "aragora.debate.similarity.backends":
            continue
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for name in _SYNCED_NAMES:
            new_val = getattr(backends_mod, name, None)
            if new_val is not None and hasattr(mod, name):
                old_val = getattr(mod, name)
                if old_val is not new_val:
                    setattr(mod, name, new_val)


# ---------------------------------------------------------------------------
# Fake API keys to prevent slow fallback paths
# ---------------------------------------------------------------------------
# When no API keys are present, several Arena subsystems (context delegation,
# Supabase sync, etc.) attempt network connections that time out after ~5s
# each. This adds up to >180s across ~30 arena.run() tests.  Setting fake
# keys ensures those code paths skip quickly (ARAGORA_OFFLINE=1 prevents
# actual API calls, but some guards only check key presence).


@pytest.fixture(autouse=True)
def _ensure_fake_api_keys(monkeypatch):
    """Provide fake API keys so Arena subsystems skip slow fallback paths.

    Without keys, code paths that gate on key presence fall through to
    network-timeout fallbacks that add ~5 seconds per arena.run() call.
    Combined with ARAGORA_OFFLINE=1 (set by _isolate_debate_databases),
    these keys are never actually used for real API calls.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")


# ---------------------------------------------------------------------------
# Stub expensive context-gathering operations
# ---------------------------------------------------------------------------
# Arena.run() delegates to ContextInitializer which gathers Aragora project
# docs, evidence, trending context, and memory prefetch via run_in_executor.
# These are unnecessary for unit tests and add significant wall-clock time.


@pytest.fixture(autouse=True)
def _stub_expensive_arena_io(monkeypatch, request):
    """Stub out expensive I/O operations inside Arena.run().

    Tests marked with ``@pytest.mark.no_io_stubs`` skip these patches
    (e.g. delegation tests that verify the call chain).

    Targets:
    - Arena._gather_aragora_context (reads project docs via run_in_executor)
    - Arena._gather_evidence_context (evidence collection)
    - Arena._gather_trending_context (Pulse trending topics)
    - Arena._perform_research (ContextGatherer.gather_all -> PulseManager HTTP)
    - Arena._init_km_context (Knowledge Mound queries)
    - Arena._queue_for_supabase_sync (Supabase connection attempts)
    - Arena._create_debate_bead (bead creation I/O)
    - MemoryManager.prefetch_for_debate_async (thread pool prefetch)
    - webhook_config_store (AWS Secrets Manager calls)

    The _perform_research stub is critical: without it, background research
    tasks call ContextGatherer.gather_all() which creates PulseManager
    instances that make 9 real HTTP calls to Google Trends, HackerNews,
    Reddit, and GitHub APIs, adding ~4 seconds per arena.run() call.

    This reduces per-test arena.run() time from ~5s to <0.5s.
    """
    if request.node.get_closest_marker("no_io_stubs"):
        return

    from aragora.debate.orchestrator import Arena

    async def _noop_async_none(self, *args, **kwargs):
        return None

    async def _noop_async(self, *args, **kwargs):
        pass

    def _noop_sync(self, *args, **kwargs):
        pass

    def _noop_sync_none(self, *args, **kwargs):
        return None

    # Stub context delegation (file reads, evidence gathering, trending)
    monkeypatch.setattr(Arena, "_gather_aragora_context", _noop_async_none)
    monkeypatch.setattr(Arena, "_gather_evidence_context", _noop_async_none)
    monkeypatch.setattr(Arena, "_gather_trending_context", _noop_async_none)

    # Stub _perform_research to prevent background research tasks from
    # calling ContextGatherer.gather_all() which creates PulseManager
    # instances that make real HTTP calls to Google Trends, HN, Reddit,
    # and GitHub APIs (~4 seconds of network I/O per arena.run()).
    async def _noop_research(self, *args, **kwargs):
        return ""

    monkeypatch.setattr(Arena, "_perform_research", _noop_research)

    # Stub KM context initialization (Knowledge Mound queries)
    monkeypatch.setattr(Arena, "_init_km_context", _noop_async)

    # Note: _ingest_debate_outcome is NOT stubbed here because
    # TestKnowledgeMoundIntegration.test_ingest_debate_outcome_stores_high_confidence
    # directly calls arena._ingest_debate_outcome() and verifies KM store calls.
    # The retry/backoff loop in _km_ingest_background (1s + 2s sleep) is handled
    # by the cleanup drain timeout.  With ARAGORA_OFFLINE=1 and fake API keys,
    # the KM ingestion fails fast without network timeouts.

    # Stub Supabase sync and bead creation (prevent network I/O)
    monkeypatch.setattr(Arena, "_queue_for_supabase_sync", _noop_sync)
    monkeypatch.setattr(Arena, "_create_debate_bead", _noop_async_none)

    # Stub memory prefetching (run_in_executor calls)
    try:
        from aragora.debate.memory_manager import MemoryManager

        monkeypatch.setattr(MemoryManager, "prefetch_for_debate_async", _noop_async)
    except (ImportError, AttributeError):
        pass

    # Stub webhook config store creation (prevents AWS Secrets Manager calls
    # triggered by event dispatch -> webhook store -> connection_factory ->
    # secrets.get -> boto3 API call)
    try:
        from aragora.storage import webhook_config_store as wcs_mod

        def _fake_get_webhook_config_store():
            return None

        monkeypatch.setattr(wcs_mod, "get_webhook_config_store", _fake_get_webhook_config_store)
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Session-level cleanup: force-terminate dangling threads on exit
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):
    """Force-cleanup dangling non-daemon threads and executor pools.

    After all debate tests complete, leaked ThreadPoolExecutor workers
    (created by asyncio loop.run_in_executor(None, ...)) and stale
    pytest-timeout timer threads can keep the process alive indefinitely.

    This hook:
    1. Shuts down any default executors left on stale event loops
    2. Force-joins non-daemon threads with a short timeout
    """
    # Shut down any ThreadPoolExecutor instances attached to event loops
    # that weren't properly closed
    import gc

    gc.collect()
    for obj in gc.get_objects():
        if isinstance(obj, concurrent.futures.ThreadPoolExecutor):
            try:
                obj.shutdown(wait=False, cancel_futures=True)
            except (TypeError, RuntimeError):
                # Python <3.9 doesn't have cancel_futures
                try:
                    obj.shutdown(wait=False)
                except RuntimeError:
                    pass

    # Give threads a moment to finish, then force-join stragglers
    non_daemon_threads = [
        t
        for t in threading.enumerate()
        if t.is_alive()
        and not t.daemon
        and t is not threading.main_thread()
        and t.name != "MainThread"
    ]
    for t in non_daemon_threads:
        t.join(timeout=1.0)
        if t.is_alive():
            # Thread didn't stop; nothing more we can do from pure Python,
            # but at least the executor shutdown above should have signalled
            # its work queue to drain.
            pass
