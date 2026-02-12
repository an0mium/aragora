"""Shared SICA runtime settings parsing."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


@dataclass(frozen=True)
class SICASettings:
    """Resolved SICA settings from environment."""

    enabled: bool
    improvement_types_csv: str
    generator_model: str
    require_approval: bool
    run_tests: bool
    run_typecheck: bool
    run_lint: bool
    test_command: str
    typecheck_command: str
    lint_command: str
    validation_timeout: float
    max_opportunities: int
    max_rollbacks: int


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_sica_settings(env: Mapping[str, str] | None = None) -> SICASettings:
    """Load SICA settings from environment with consistent defaults."""
    source = env or os.environ
    return SICASettings(
        enabled=_get_bool(source, "NOMIC_SICA_ENABLED", False),
        improvement_types_csv=source.get(
            "NOMIC_SICA_IMPROVEMENT_TYPES",
            "reliability,testability,readability",
        ),
        generator_model=source.get("NOMIC_SICA_GENERATOR_MODEL", "codex"),
        require_approval=_get_bool(source, "NOMIC_SICA_REQUIRE_APPROVAL", True),
        run_tests=_get_bool(source, "NOMIC_SICA_RUN_TESTS", True),
        run_typecheck=_get_bool(source, "NOMIC_SICA_RUN_TYPECHECK", True),
        run_lint=_get_bool(source, "NOMIC_SICA_RUN_LINT", True),
        test_command=source.get("NOMIC_SICA_TEST_COMMAND", "pytest"),
        typecheck_command=source.get("NOMIC_SICA_TYPECHECK_COMMAND", "mypy"),
        lint_command=source.get("NOMIC_SICA_LINT_COMMAND", "ruff check"),
        validation_timeout=_get_float(source, "NOMIC_SICA_VALIDATION_TIMEOUT", 300.0),
        max_opportunities=_get_int(source, "NOMIC_SICA_MAX_OPPORTUNITIES", 5),
        max_rollbacks=_get_int(source, "NOMIC_SICA_MAX_ROLLBACKS", 3),
    )
