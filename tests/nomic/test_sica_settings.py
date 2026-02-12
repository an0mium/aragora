"""Tests for shared SICA settings parsing."""

from __future__ import annotations

from aragora.nomic.sica_settings import load_sica_settings


def test_load_sica_settings_bool_variants() -> None:
    settings = load_sica_settings(
        {
            "NOMIC_SICA_ENABLED": "true",
            "NOMIC_SICA_REQUIRE_APPROVAL": "off",
            "NOMIC_SICA_RUN_TESTS": "yes",
            "NOMIC_SICA_RUN_TYPECHECK": "no",
            "NOMIC_SICA_RUN_LINT": "on",
        }
    )
    assert settings.enabled is True
    assert settings.require_approval is False
    assert settings.run_tests is True
    assert settings.run_typecheck is False
    assert settings.run_lint is True


def test_load_sica_settings_numeric_fallbacks() -> None:
    settings = load_sica_settings(
        {
            "NOMIC_SICA_VALIDATION_TIMEOUT": "not-a-float",
            "NOMIC_SICA_MAX_OPPORTUNITIES": "x",
            "NOMIC_SICA_MAX_ROLLBACKS": "y",
        }
    )
    assert settings.validation_timeout == 300.0
    assert settings.max_opportunities == 5
    assert settings.max_rollbacks == 3
