"""Regression guard for the declared base runtime dependency floor.

aiohttp (server handlers, agents, connectors -- the widest runtime footprint)
and websockets (the ``aragora/server/stream/*`` WebSocket surface) must stay in
the base ``[project.dependencies]`` floor alongside pydantic and PyYAML, so a
plain ``pip install aragora`` resolves the async runtime stack without ``[all]``
(the VAL-P3-003 packaging assertion).

These tests parse ``pyproject.toml`` and fail if a floor member is dropped to the
extras only, or if its declared lower bound is weakened below the audited floor
(so a silent ``>=3.13`` downgrade cannot slip through). A resolved/installed
version check (as in ``test_dateutil_floor.py``) is intentionally omitted here:
an older aiohttp already present in an ambient environment would false-fail it,
whereas the *declared* floor is what governs a fresh ``pip install aragora``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# name -> declared base lower bound, mirroring pyproject [project.dependencies].
_BASE_FLOORS: dict[str, tuple[int, ...]] = {
    "aiohttp": (3, 14, 1),
    "websockets": (13, 0),
    "pyyaml": (6, 0, 3),
    "pydantic": (2, 13, 4),
}


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _declared_base_dependencies() -> dict[str, str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    declared: dict[str, str] = {}
    for dep in data["project"]["dependencies"]:
        name = re.split(r"[<>=!~\[; ]", dep.strip(), maxsplit=1)[0]
        declared[name.lower().replace("-", "_")] = dep.strip()
    return declared


def test_runtime_floor_declared_in_base():
    declared = _declared_base_dependencies()
    missing = sorted(name for name in _BASE_FLOORS if name not in declared)
    assert not missing, f"base [project.dependencies] dropped the runtime floor: {missing}"


def test_declared_floor_not_weakened():
    declared = _declared_base_dependencies()
    for name, floor in _BASE_FLOORS.items():
        spec = declared.get(name, "")
        lower = re.search(r">=\s*([0-9][0-9.]*)", spec)
        assert lower, f"{name} base requirement {spec!r} is missing a lower-bound pin"
        assert _version_tuple(lower.group(1)) >= floor, (
            f"{name} lower bound {lower.group(1)} weakened below {'.'.join(map(str, floor))}"
        )
