"""Tests for scripts/analyze_openapi_coverage.py."""

from __future__ import annotations

import math
from typing import Any

import pytest

import scripts.analyze_openapi_coverage as coverage


def _spec_with_response_schema() -> dict[str, Any]:
    return {
        "paths": {
            "/health": {
                "get": {
                    "operationId": "getHealth",
                    "tags": ["Health"],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"ok": {"type": "boolean"}},
                                    }
                                }
                            }
                        }
                    },
                }
            }
        }
    }


@pytest.mark.parametrize("value", [0.0, 50.5, 100.0])
def test_validate_fail_threshold_accepts_percentages(value: float) -> None:
    assert coverage.validate_fail_threshold(value) == value


@pytest.mark.parametrize("value", [-0.1, 100.1, math.nan, math.inf, -math.inf, True])
def test_validate_fail_threshold_rejects_impossible_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite percentage between 0 and 100"):
        coverage.validate_fail_threshold(value)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1", "101"])
def test_main_refuses_invalid_fail_threshold_before_loading_spec(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_load() -> dict[str, Any]:
        raise AssertionError("spec should not be loaded for invalid threshold")

    monkeypatch.setattr(coverage, "load_openapi_spec", fail_load)

    rc = coverage.main([f"--fail-threshold={value}", "--json"])

    assert rc == 2
    assert "finite percentage between 0 and 100" in capsys.readouterr().err


def test_main_accepts_valid_fail_threshold(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(coverage, "load_openapi_spec", _spec_with_response_schema)

    rc = coverage.main(["--fail-threshold=100", "--json"])

    assert rc == 0
    assert '"response_coverage_pct": 100.0' in capsys.readouterr().out
