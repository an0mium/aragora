"""Shared fixtures for the pipeline test suite.

``aragora/swarm/__init__.py`` defines a PEP 562 module ``__getattr__`` that only
serves names in its ``_EXPORTS`` allowlist. ``tranche_submit`` is a submodule,
not an export, so it is reachable as ``aragora.swarm.tranche_submit`` only after
it has been imported and bound onto the package namespace. Several pipeline
tests patch ``sys.modules`` (``patch.dict``) to simulate missing optional
dependencies; under ``pytest-randomly`` an ordering can leave the
``aragora.swarm`` package or the submodule unbound, after which
``monkeypatch.setattr("aragora.swarm.tranche_submit.submit_intake_bundle", ...)``
in ``test_stage_transitions.py`` raises ``AttributeError`` from that
``__getattr__``. Snapshot the clean module objects at import time and restore the
binding before every test so the suite is order-independent without touching
source behavior.
"""

from __future__ import annotations

import sys

import pytest

import aragora
import aragora.swarm
import aragora.swarm.tranche_submit

_SWARM_PACKAGE = sys.modules["aragora.swarm"]
_TRANCHE_SUBMIT = sys.modules["aragora.swarm.tranche_submit"]


@pytest.fixture(autouse=True)
def _restore_swarm_tranche_submit_binding():
    sys.modules["aragora.swarm"] = _SWARM_PACKAGE
    sys.modules["aragora.swarm.tranche_submit"] = _TRANCHE_SUBMIT
    aragora.swarm = _SWARM_PACKAGE
    _SWARM_PACKAGE.tranche_submit = _TRANCHE_SUBMIT
    yield
