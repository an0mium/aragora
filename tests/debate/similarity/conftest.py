"""
Conftest for debate/similarity tests.

Ensures that importlib.reload() of aragora.debate.similarity.backends
(used by tfidf_backend fixture and import-error tests) does not leave
stale class references in aragora.debate.convergence, which re-exports
JaccardBackend, TFIDFBackend, SimilarityBackend, etc.

Without this fixture, isinstance() checks in downstream tests fail
because the convergence module holds references to the *original*
class objects while the backends module now contains *reloaded* (new)
class objects.
"""

import pytest


@pytest.fixture(autouse=True)
def _resync_convergence_after_backend_reload():
    """Re-synchronize convergence module class references after each test.

    Some tests in this directory call importlib.reload() on the backends
    module, which creates new class objects.  The convergence __init__
    still holds references to the old classes, causing isinstance()
    failures in tests that run later.

    This fixture runs after each test and patches the convergence
    module's exported names to match the current backends module.
    """
    yield

    try:
        import aragora.debate.similarity.backends as backends_mod
        import aragora.debate.convergence as convergence_mod

        # List of class/function names that convergence re-exports
        # from backends and that must stay in sync.
        _SYNCED_NAMES = [
            "JaccardBackend",
            "TFIDFBackend",
            "SentenceTransformerBackend",
            "SimilarityBackend",
            "get_similarity_backend",
            "_normalize_backend_name",
            "_ENV_CONVERGENCE_BACKEND",
        ]

        for name in _SYNCED_NAMES:
            current = getattr(backends_mod, name, None)
            if current is not None:
                setattr(convergence_mod, name, current)
    except ImportError:
        pass
