"""Tests for ``scripts/collect_model_evidence.py``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HEAD = "b01e86f5915a85fc6a97ec44bf669b95734e0443"


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collector = _load_module("collect_model_evidence.py")


def _completed(args: list[str], stdout: str, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=args, returncode=returncode, stdout=stdout, stderr=stderr
    )


def _pr_payload(head: str = HEAD) -> str:
    return json.dumps(
        {
            "number": 7498,
            "headRefOid": head,
            "url": "https://github.com/synaptent/aragora/pull/7498",
            "title": "docs: sync generated docs after #7491",
            "files": [
                {"path": "docs-site/docs/deployment/secrets-management.md"},
                {"path": "docs-site/docs/operations/overview.md"},
            ],
        }
    )


def test_direct_gemini_failure_falls_back_to_droid_gemini() -> None:
    calls: list[list[str]] = []

    def fake_runner(
        args: list[str], *, input_text: str | None = None, timeout_seconds: float | None = None
    ):
        calls.append(args)
        if args[:3] == ["gh", "pr", "view"]:
            return _completed(args, _pr_payload())
        if args[:3] == ["gh", "pr", "diff"]:
            return _completed(args, "diff --git a/docs.md b/docs.md\n+safe docs update")
        if args[:3] == ["python3", "-m", "aragora.cli.main"]:
            body = args[args.index("--body") + 1]
            if "Gemini" in body:
                return _completed(
                    args,
                    json.dumps(
                        {"would_count": True, "counted_reviewer_ids": ["gemini"], "problems": []}
                    ),
                )
            return _completed(
                args,
                json.dumps({"would_count": False, "counted_reviewer_ids": [], "problems": []}),
            )
        if args and args[0] == "gemini":
            return _completed(args, "", returncode=1, stderr="API key expired")
        if args[:2] == ["droid", "exec"]:
            return _completed(
                args,
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "Verdict: CLEAN\nBlocking findings: None\nValidation confidence: high",
                    }
                ),
            )
        raise AssertionError(f"unexpected command: {args}")

    result = collector.collect_model_evidence(
        pr=7498,
        expected_head=HEAD,
        family_order=("gemini", "droid-gemini"),
        post_comment=False,
        runner=fake_runner,
    )

    assert result.status == "ready"
    assert result.selected_route == "droid-gemini"
    assert result.comment is not None
    assert result.comment.startswith("## Gemini via Droid focused adversarial dogfood")
    assert any(args and args[0] == "gemini" for args in calls)
    assert any(args[:3] == ["droid", "exec", "--model"] for args in calls)


def test_uncounted_route_is_skipped_before_model_execution() -> None:
    calls: list[list[str]] = []

    def fake_runner(
        args: list[str], *, input_text: str | None = None, timeout_seconds: float | None = None
    ):
        calls.append(args)
        if args[:3] == ["gh", "pr", "view"]:
            return _completed(args, _pr_payload())
        if args[:3] == ["gh", "pr", "diff"]:
            return _completed(args, "diff")
        if args[:3] == ["python3", "-m", "aragora.cli.main"]:
            return _completed(
                args,
                json.dumps(
                    {
                        "would_count": False,
                        "counted_reviewer_ids": [],
                        "problems": ["missing_known_model_reviewer_heading"],
                    }
                ),
            )
        if args[:2] == ["droid", "exec"]:
            raise AssertionError("uncounted route should be skipped before model execution")
        raise AssertionError(f"unexpected command: {args}")

    result = collector.collect_model_evidence(
        pr=7498,
        expected_head=HEAD,
        family_order=("droid-kimi",),
        post_comment=False,
        runner=fake_runner,
    )

    assert result.status == "no_countable_route"
    assert result.selected_route is None
    assert any(
        "preflight evidence-lint would not count" in attempt.error for attempt in result.attempts
    )


def test_post_comment_rechecks_head_and_refuses_on_drift() -> None:
    calls: list[list[str]] = []
    pr_views = [_pr_payload(), _pr_payload(head="drifted-head")]

    def fake_runner(
        args: list[str], *, input_text: str | None = None, timeout_seconds: float | None = None
    ):
        calls.append(args)
        if args[:3] == ["gh", "pr", "view"]:
            return _completed(args, pr_views.pop(0))
        if args[:3] == ["gh", "pr", "diff"]:
            return _completed(args, "diff")
        if args[:3] == ["python3", "-m", "aragora.cli.main"]:
            return _completed(args, json.dumps({"would_count": True, "problems": []}))
        if args[:2] == ["droid", "exec"]:
            return _completed(args, "Verdict: CLEAN\nBlocking findings: None")
        if args[:3] == ["gh", "pr", "comment"]:
            raise AssertionError("must not post after head drift")
        raise AssertionError(f"unexpected command: {args}")

    result = collector.collect_model_evidence(
        pr=7498,
        expected_head=HEAD,
        family_order=("droid-gemini",),
        post_comment=True,
        runner=fake_runner,
    )

    assert result.status == "head_drift"
    assert result.posted is False
    assert not any(args[:3] == ["gh", "pr", "comment"] for args in calls)


def test_blocking_findings_suppress_comment_post() -> None:
    calls: list[list[str]] = []

    def fake_runner(
        args: list[str], *, input_text: str | None = None, timeout_seconds: float | None = None
    ):
        calls.append(args)
        if args[:3] == ["gh", "pr", "view"]:
            return _completed(args, _pr_payload())
        if args[:3] == ["gh", "pr", "diff"]:
            return _completed(args, "diff")
        if args[:3] == ["python3", "-m", "aragora.cli.main"]:
            return _completed(args, json.dumps({"would_count": True, "problems": []}))
        if args[:2] == ["droid", "exec"]:
            return _completed(
                args,
                "Verdict: BLOCKED\nBlocking findings:\n- docs claim a generated secret value is safe",
            )
        if args[:3] == ["gh", "pr", "comment"]:
            raise AssertionError("must not post comments with blocking findings")
        raise AssertionError(f"unexpected command: {args}")

    result = collector.collect_model_evidence(
        pr=7498,
        expected_head=HEAD,
        family_order=("droid-gemini",),
        post_comment=True,
        runner=fake_runner,
    )

    assert result.status == "blocking_findings"
    assert result.posted is False
    assert "docs claim a generated secret value is safe" in result.blocking_findings[0]
