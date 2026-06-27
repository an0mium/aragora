"""Tests for ``scripts/anchor_intent_chain.py`` (TET phase T2 external anchor).

All network boundaries (gh runner, rekor submission) are injected fakes;
no test touches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from aragora.trail.intent_chain import append_intent


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "anchor_intent_chain.py"
    spec = importlib.util.spec_from_file_location("anchor_intent_chain_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


anchor = _load_module()

MAIN_SHA = "a" * 40
TARGET = {"repo": "synaptent/aragora", "pr": 99}


def _chain_with_records(tmp_path: Path, n: int = 3) -> Path:
    chain = tmp_path / "chain.jsonl"
    for i in range(n):
        append_intent(
            chain,
            actor_class="agent-claude",
            intent_type="publish_pr",
            target=TARGET,
            payload={"n": i},
            now=lambda: "2026-06-11T22:00:00+00:00",
        )
    return chain


class FakeGh:
    """Scripted gh runner recording every invocation."""

    def __init__(self, head_sha: str = MAIN_SHA, post_rc: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.head_sha = head_sha
        self.post_rc = post_rc

    def __call__(self, args: list[str]) -> tuple[int, str]:
        self.calls.append(args)
        if "--method" not in args:  # the head-resolution GET
            return 0, self.head_sha
        return self.post_rc, "" if self.post_rc == 0 else "boom"


def _run(chain: Path, gh: FakeGh, logs: list[str], **kwargs: Any) -> int:
    defaults: dict[str, Any] = {
        "chain_path": chain,
        "repo": "synaptent/aragora",
        "apply": False,
        "rekor": False,
        "max_anchors": 2,
        "run_gh": gh,
        "log": logs.append,
    }
    defaults.update(kwargs)
    return anchor.run_anchor(**defaults)


class TestHeadAndPlan:
    def test_empty_chain_is_clean_noop(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gh = FakeGh()
        rc = _run(tmp_path / "missing.jsonl", gh, logs)
        assert rc == anchor.EXIT_OK
        assert gh.calls == []  # nothing resolved, nothing posted
        assert json.loads(logs[0])["result"] == "no-op"

    def test_dry_run_plans_status_without_posting(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path, 3)
        logs: list[str] = []
        gh = FakeGh()
        rc = _run(chain, gh, logs)
        assert rc == anchor.EXIT_OK
        # Only the head resolution call ran; no POST left the machine.
        assert len(gh.calls) == 1
        plan = json.loads(logs[0])
        assert plan["mode"] == "dry-run"
        assert plan["seq"] == 2
        assert plan["anchor_target"]["context"] == "aragora/trail-anchor"
        assert f"repos/synaptent/aragora/statuses/{MAIN_SHA}" in plan["gh_args"]

    def test_status_args_carry_seq_and_head12(self, tmp_path: Path) -> None:
        head_hash = "f" * 64
        args = anchor.build_status_args("synaptent/aragora", MAIN_SHA, 7, head_hash)
        assert f"description=trail-anchor seq=7 head={head_hash[:12]}" in args
        assert "state=success" in args
        assert "context=aragora/trail-anchor" in args


class TestFailClosed:
    def test_broken_chain_is_never_anchored(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path, 3)
        lines = chain.read_text().splitlines()
        tampered = json.loads(lines[1])
        tampered["payload"]["n"] = 999
        lines[1] = json.dumps(tampered)
        chain.write_text("\n".join(lines) + "\n")
        logs: list[str] = []
        gh = FakeGh()
        rc = _run(chain, gh, logs, apply=True)
        assert rc == anchor.EXIT_FAILURE
        assert gh.calls == []
        assert "verification failed at seq=1" in json.loads(logs[0])["reason"]

    def test_head_resolution_failure_fails_closed(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path)
        logs: list[str] = []

        def broken_gh(args: list[str]) -> tuple[int, str]:
            return 1, "api down"

        rc = _run(chain, broken_gh, logs, apply=True)
        assert rc == anchor.EXIT_FAILURE

    def test_status_post_failure_fails_closed(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path)
        logs: list[str] = []
        gh = FakeGh(post_rc=1)
        rc = _run(chain, gh, logs, apply=True)
        assert rc == anchor.EXIT_FAILURE
        assert any("status post failed" in line for line in logs)

    def test_max_anchors_zero_blocks_apply(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path)
        logs: list[str] = []
        gh = FakeGh()
        rc = _run(chain, gh, logs, apply=True, max_anchors=0)
        assert rc == anchor.EXIT_FAILURE
        assert len(gh.calls) == 1  # head resolution only; POST blocked


class TestApply:
    def test_apply_posts_anchor_status(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path, 2)
        logs: list[str] = []
        gh = FakeGh()
        rc = _run(chain, gh, logs, apply=True)
        assert rc == anchor.EXIT_OK
        post = gh.calls[-1]
        assert "--method" in post and "POST" in post
        assert f"repos/synaptent/aragora/statuses/{MAIN_SHA}" in post
        assert any("trail-anchor seq=1" in part for part in post)
        assert json.loads(logs[-1])["result"] == "anchored"


class RecordingRekorSubmit:
    """Injected stand-in for ``aragora.trail.rekor.submit_hash``."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, head_hash: str) -> Any:
        from aragora.trail.rekor import RekorEntry, RekorError

        self.calls.append(head_hash)
        if self.fail:
            raise RekorError("rekor submission failed: HTTP 503")
        return RekorEntry(uuid="c" * 64, log_index=777, integrated_time=1760000099)


class TestRekor:
    def test_rekor_success_records_log_index_uuid_time(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path)
        logs: list[str] = []
        gh = FakeGh()
        submit = RecordingRekorSubmit()
        rc = _run(chain, gh, logs, apply=True, rekor=True, rekor_submit=submit)
        assert rc == anchor.EXIT_OK
        notes = [json.loads(line) for line in logs if '"rekor"' in line]
        assert notes[-1]["rekor"] == "anchored"
        assert notes[-1]["log_index"] == 777
        assert notes[-1]["uuid"] == "c" * 64
        assert notes[-1]["integrated_time"] == 1760000099
        # The submitted hash is the verified chain head, never anything else.
        plan = json.loads(logs[0])
        assert submit.calls == [plan["head"]]

    def test_rekor_failure_degrades_without_fabricating_a_record(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path)
        logs: list[str] = []
        gh = FakeGh()
        submit = RecordingRekorSubmit(fail=True)
        rc = _run(chain, gh, logs, apply=True, rekor=True, rekor_submit=submit)
        # Never blocks: the commit-status anchor stands, exit stays OK.
        assert rc == anchor.EXIT_OK
        assert any(json.loads(line).get("result") == "anchored" for line in logs)
        notes = [json.loads(line) for line in logs if '"rekor"' in line]
        assert notes[-1]["rekor"] == "degraded"
        # No fabricated anchor fields on failure.
        assert "log_index" not in notes[-1]
        assert "uuid" not in notes[-1]
        assert "integrated_time" not in notes[-1]

    def test_rekor_dry_run_plans_submission_without_network(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path)
        logs: list[str] = []
        gh = FakeGh()
        submit = RecordingRekorSubmit()
        rc = _run(chain, gh, logs, rekor=True, rekor_submit=submit)
        assert rc == anchor.EXIT_OK
        assert submit.calls == []  # dry run: nothing leaves the machine
        notes = [json.loads(line) for line in logs if '"rekor"' in line]
        assert notes[-1]["rekor"] == "dry-run"
        assert notes[-1]["url"].startswith("https://rekor.sigstore.dev")

    def test_broken_chain_is_never_submitted_to_rekor(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path, 3)
        lines = chain.read_text().splitlines()
        tampered = json.loads(lines[1])
        tampered["payload"]["n"] = 999
        lines[1] = json.dumps(tampered)
        chain.write_text("\n".join(lines) + "\n")
        logs: list[str] = []
        gh = FakeGh()
        submit = RecordingRekorSubmit()
        rc = _run(chain, gh, logs, apply=True, rekor=True, rekor_submit=submit)
        assert rc == anchor.EXIT_FAILURE
        assert submit.calls == []  # fail-closed gate fires before any anchor
        assert gh.calls == []

    def test_rekor_skipped_when_max_anchors_exhausted(self, tmp_path: Path) -> None:
        chain = _chain_with_records(tmp_path)
        logs: list[str] = []
        gh = FakeGh()
        submit = RecordingRekorSubmit()
        rc = _run(chain, gh, logs, apply=True, rekor=True, max_anchors=1, rekor_submit=submit)
        assert rc == anchor.EXIT_OK
        assert submit.calls == []
        notes = [json.loads(line) for line in logs if '"rekor"' in line]
        assert notes[-1]["rekor"] == "skipped"


class TestCli:
    def test_main_dry_run_on_empty_chain(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = anchor.main(["--chain", str(tmp_path / "missing.jsonl")])
        assert rc == anchor.EXIT_OK
        assert json.loads(capsys.readouterr().out.splitlines()[0])["result"] == "no-op"
