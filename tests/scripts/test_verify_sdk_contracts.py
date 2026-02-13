"""Tests for scripts/verify_sdk_contracts.py helpers."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.verify_sdk_contracts as verify_sdk_contracts


def test_normalize_collapses_version_and_params():
    assert (
        verify_sdk_contracts._normalize("/api/v1/policies/{policy_id}/")
        == "/api/policies/{param}"
    )


def test_load_baseline_reads_expected_sets(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "python_sdk_drift": ["GET /api/a"],
                "typescript_sdk_drift": ["POST /api/b"],
                "missing_stable": ["GET /api/c"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = verify_sdk_contracts._load_baseline(baseline)
    assert loaded["python_sdk_drift"] == {"GET /api/a"}
    assert loaded["typescript_sdk_drift"] == {"POST /api/b"}
    assert loaded["missing_stable"] == {"GET /api/c"}


def test_load_openapi_endpoints_multi_unions_specs(tmp_path: Path):
    spec_a = tmp_path / "a.json"
    spec_b = tmp_path / "b.json"
    spec_a.write_text(
        json.dumps({"paths": {"/api/v1/a": {"get": {}}}}) + "\n",
        encoding="utf-8",
    )
    spec_b.write_text(
        json.dumps({"paths": {"/api/v1/b": {"post": {}}}}) + "\n",
        encoding="utf-8",
    )

    endpoints = verify_sdk_contracts._load_openapi_endpoints_multi([spec_a, spec_b])
    assert ("get", "/api/a") in endpoints
    assert ("post", "/api/b") in endpoints
