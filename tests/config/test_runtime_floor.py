"""Regression guard for the declared base runtime dependency floor.

aiohttp (server handlers, agents, connectors -- the widest runtime footprint)
and websockets (the ``aragora/server/stream/*`` WebSocket surface) must stay in
the base ``[project.dependencies]`` floor alongside pydantic and PyYAML, so a
plain ``pip install aragora`` resolves the async runtime stack without ``[all]``.
These tests fail loudly if the floor is ever dropped back into the extras only
(the VAL-P3-003 packaging assertion), mirroring ``test_dateutil_floor.py``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_REQUIRED_BASE = {"aiohttp", "websockets", "pyyaml", "pydantic"}


def _base_dependency_names() -> set[str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    return {re.split(r"[<>=!~\[; ]", dep.strip(), 1)[0].lower().replace("-", "_") for dep in deps}


def test_runtime_floor_declared_in_base():
    missing = _REQUIRED_BASE - _base_dependency_names()
    assert not missing, f"base [project.dependencies] dropped the runtime floor: {sorted(missing)}"
