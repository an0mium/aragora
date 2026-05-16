"""Tests for the Factory review benchmark smoke planner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import run_factory_review_benchmark_smoke as smoke  # noqa: E402


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "factory_review_droid_external_smoke",
        "source": {"benchmark_repo": "droid-code-review-evals/review-droid-benchmark"},
        "smoke_cases": [
            {
                "case_id": "droid-sentry-pr-6",
                "repo": "droid-code-review-evals/droid-sentry",
                "pr_number": 6,
                "pr_url": "https://github.com/droid-code-review-evals/droid-sentry/pull/6",
                "title": "Enhanced Pagination Performance for High-Volume Audit Logs",
                "base_ref": "master",
                "head_ref": "performance-enhancement-complete",
                "head_sha": "cb7212e11dbdbc1813237ad129c7bc108f944e3d",
                "validation_path": "validations/droid-sentry_pr_6_validation.json",
                "validation_url": "https://example.test/droid-sentry_pr_6_validation.json",
            }
        ],
    }


def test_build_run_plan_defaults_to_no_publish_dry_run(tmp_path: Path) -> None:
    plan = smoke.build_run_plan(_manifest(), artifact_root=tmp_path / "artifacts")

    assert plan["guardrails"]["mode"] == "dry_run"
    assert plan["guardrails"]["no_publish_review"] is True
    assert plan["guardrails"]["external_pr_comments"] is False
    assert plan["guardrails"]["live_routing_change"] is False
    assert len(plan["cases"]) == 1

    case = plan["cases"][0]
    assert case["execute_default"] is False
    assert case["head_sha"] == "cb7212e11dbdbc1813237ad129c7bc108f944e3d"
    assert case["expected_head_sha"] == "cb7212e11dbdbc1813237ad129c7bc108f944e3d"
    assert case["pre_execute_head_check"] == {
        "type": "github_pr_head_sha",
        "expected_head_sha": "cb7212e11dbdbc1813237ad129c7bc108f944e3d",
    }
    assert "--no-publish-review" in case["command"]
    assert "--json" in case["command"]
    assert str(tmp_path / "artifacts" / "droid-sentry" / "pr-6") in case["command"]


def test_build_run_plan_can_pin_reviewer(tmp_path: Path) -> None:
    plan = smoke.build_run_plan(
        _manifest(),
        artifact_root=tmp_path / "artifacts",
        reviewer="codex",
    )

    command = plan["cases"][0]["command"]
    assert command[-2:] == ["--reviewer", "codex"]


def test_manifest_rejects_missing_validation_url() -> None:
    manifest = _manifest()
    case = dict(manifest["smoke_cases"][0])
    case.pop("validation_url")
    manifest["smoke_cases"] = [case]

    try:
        smoke.build_run_plan(manifest)
    except ValueError as exc:
        assert "validation_url" in str(exc)
    else:
        raise AssertionError("missing validation_url should fail")


def test_manifest_rejects_malformed_case() -> None:
    manifest = _manifest()
    manifest["smoke_cases"] = ["not-a-case"]

    try:
        smoke.build_run_plan(manifest)
    except ValueError as exc:
        assert "must be a JSON object" in str(exc)
    else:
        raise AssertionError("malformed case should fail")


def test_verify_case_head_rejects_head_drift(monkeypatch: object) -> None:
    manifest = _manifest()
    plan = smoke.build_run_plan(manifest)
    case = plan["cases"][0]

    def fake_current_head_sha(pr_url: str) -> str:
        assert pr_url == "https://github.com/droid-code-review-evals/droid-sentry/pull/6"
        return "different-sha"

    monkeypatch.setattr(smoke, "_current_pr_head_sha", fake_current_head_sha)

    try:
        smoke._verify_case_head(case)
    except ValueError as exc:
        assert "head SHA drifted" in str(exc)
    else:
        raise AssertionError("head SHA drift should fail closed")


def test_main_writes_plan_without_execute(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "plan.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    assert smoke.main(["--manifest", str(manifest_path), "--output", str(output_path)]) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["guardrails"]["mode"] == "dry_run"
    assert "executions" not in payload
    assert payload["cases"][0]["command"][2:4] == ["aragora.cli.main", "review-pr"]
