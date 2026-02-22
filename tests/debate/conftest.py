"""
Conftest for debate tests.

Re-synchronizes module class references after each test to prevent
isinstance() failures caused by importlib.reload() of the similarity
backends module in tests/debate/similarity/.
"""

import sys

import pytest


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
