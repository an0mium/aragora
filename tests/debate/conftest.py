"""
Conftest for debate tests.

Re-synchronizes module class references after each test to prevent
isinstance() failures caused by importlib.reload() of the similarity
backends module in tests/debate/similarity/.
"""

import sys

import pytest


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
            TFIDFBackend,
            SentenceTransformerBackend,
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

        SimilarityFactory._registry.clear()
        SimilarityFactory._initialized = False
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _mock_scan_code_markers(request, monkeypatch):
    """Prevent scan_code_markers from walking the entire repo.

    MetaPlanner.prioritize_work() → NextStepsRunner.scan() →
    scan_code_markers() does os.walk on up to 5000 files.
    This causes timeouts in long suite runs.
    """
    try:
        import aragora.compat.openclaw.next_steps_runner as nsr_mod

        monkeypatch.setattr(nsr_mod, "scan_code_markers", lambda repo_path: ([], 0))
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _resync_convergence_after_backend_reload():
    """Re-synchronize convergence module class references after each test.

    Some tests call importlib.reload() on the backends module, which
    creates new class objects.  Other modules (convergence) still hold
    references to the old classes, causing isinstance() failures in
    tests that run later.

    This fixture patches every module that re-exports backend classes
    so their references match the current backends module.
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

    # Update any module that re-exports these names (convergence, etc.)
    for mod_name in list(sys.modules):
        if not mod_name.startswith("aragora.debate"):
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
                # Only update if it's actually stale (different object)
                if old_val is not new_val:
                    setattr(mod, name, new_val)
