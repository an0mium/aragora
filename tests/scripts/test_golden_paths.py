"""Tests for scripts/golden_paths.py."""

from __future__ import annotations

import importlib.util
from dataclasses import fields
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "golden_paths.py"
    spec = importlib.util.spec_from_file_location("golden_paths", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load golden_paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_overrides_only_include_supported_debate_protocol_fields() -> None:
    module = _load_script_module()

    overrides = module._protocol_overrides(mode="full", enable_trending=True)
    supported_fields = {field.name for field in fields(module.DebateProtocol)}

    assert set(overrides).issubset(supported_fields)
    assert overrides["use_structured_phases"] is True
    assert overrides["early_stopping"] is False
