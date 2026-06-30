"""Tests for ``scripts/dogfood_evidence.py`` (bounded fail-closed dogfood step, #8219).

All I/O (gh head fetch, file list, worktree checkout/remove, validation,
evidence-lint, comment post) is injected; no test touches the network, git, or a
subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


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


df = _load_module("dogfood_evidence.py")


# --- trust guard ------------------------------------------------------------


def test_trusted_namespaces_accepted():
    assert df.is_trusted_head_ref("codex/foo")
    assert df.is_trusted_head_ref("elves/run-20260611-x")
    assert df.is_trusted_head_ref("aragora/thing")
    assert df.is_trusted_head_ref("dependabot/npm_and_yarn/react-19")
    assert df.is_trusted_head_ref("dependabot")


def test_untrusted_and_fork_refs_refused():
    assert not df.is_trusted_head_ref("feature/random")
    assert not df.is_trusted_head_ref("")
    # A fork can spoof the namespace; cross-repo is never trusted.
    assert not df.is_trusted_head_ref("codex/spoof", is_cross_repo=True)
    assert not df.is_trusted_head_ref("dependabot", is_cross_repo=True)


# --- validation discovery ---------------------------------------------------


def test_discover_prefers_touched_test_files():
    cmd, why = df.discover_validation_command(["aragora/foo.py", "tests/scripts/test_foo.py"])
    assert "pytest" in cmd
    assert "tests/scripts/test_foo.py" in cmd
    assert "test" in why


def test_discover_falls_back_to_compile_check():
    cmd, _why = df.discover_validation_command(["aragora/bar.py"])
    assert cmd[:2] == [sys.executable, "-c"]
    assert "aragora/bar.py" in cmd


def test_discover_empty_when_no_python():
    cmd, why = df.discover_validation_command(["aragora/live/package-lock.json"])
    assert cmd == []
    assert "no python" in why.lower()


# --- lint-counts-as-dogfood gate -------------------------------------------


def test_lint_counts_requires_dogfood_evidence():
    assert df.lint_counts_as_dogfood({"would_count": True, "dogfood_evidence": [{"x": 1}]})
    # would_count via model-review only (no dogfood entry) must NOT count.
    assert not df.lint_counts_as_dogfood(
        {"would_count": True, "dogfood_evidence": [], "reviewer_signals": [{"x": 1}]}
    )
    assert not df.lint_counts_as_dogfood({"would_count": False, "dogfood_evidence": [{"x": 1}]})


def test_composed_comment_is_recognizable():
    body = df.compose_dogfood_comment(
        pr=42,
        head_sha="abcdef1234567890",
        model_family="claude",
        command="pytest tests/x",
        passed=True,
        output_digest="3 passed",
    )
    assert "abcdef1234" in body
    assert "Model family: claude" in body
    assert "dogfood: yes" in body


# --- orchestration: dogfood_pr ---------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.posted: list[tuple[int, str]] = []
        self.removed: list[str] = []
        self.checked_out: list[str] = []


def _make_calls(
    rec: _Recorder,
    *,
    head_ref: str = "codex/x",
    head_sha: str = "deadbeef0123456789",
    is_cross: bool = False,
    files: list[str] | None = None,
    checkout_ok: bool = True,
    validation: tuple[bool, str] = (True, "3 passed"),
    lint: dict[str, Any] | None = None,
    post_ok: bool = True,
) -> dict[str, Any]:
    files = files if files is not None else ["aragora/foo.py", "tests/test_foo.py"]
    lint = lint if lint is not None else {"would_count": True, "dogfood_evidence": [{"x": 1}]}

    def fetch_head(repo: str, pr: int) -> dict[str, Any]:
        return {
            "headRefOid": head_sha,
            "headRefName": head_ref,
            "isCrossRepository": is_cross,
        }

    def changed_files(repo: str, pr: int) -> list[str]:
        return list(files)

    def checkout(repo: str, sha: str, dest: str) -> bool:
        rec.checked_out.append(dest)
        return checkout_ok

    def remove_worktree(dest: str) -> None:
        rec.removed.append(dest)

    def run_validation(cmd: list[str], cwd: str, timeout: int) -> tuple[bool, str]:
        return validation

    def lint_evidence(repo: str, pr: int, sha: str, body: str) -> dict[str, Any]:
        return dict(lint)

    def post_comment(repo: str, pr: int, body: str) -> bool:
        rec.posted.append((pr, body))
        return post_ok

    return {
        "fetch_head": fetch_head,
        "changed_files": changed_files,
        "checkout": checkout,
        "remove_worktree": remove_worktree,
        "run_validation": run_validation,
        "lint_evidence": lint_evidence,
        "post_comment": post_comment,
        "worktree_factory": lambda: "/tmp/disposable-wt",
    }


def _run(rec: _Recorder, calls: dict[str, Any], *, apply: bool = True):
    return df.dogfood_pr(
        repo="o/r",
        pr=7,
        model_family="claude",
        timeout=600,
        apply=apply,
        log=lambda _m: None,
        **calls,
    )


def test_dogfood_pass_posts_counting_evidence():
    rec = _Recorder()
    out = _run(rec, _make_calls(rec))
    assert out.status == "posted"
    assert rec.posted and rec.posted[0][0] == 7
    assert "Model family: claude" in rec.posted[0][1]
    assert rec.removed == ["/tmp/disposable-wt"]  # cleaned


def test_dogfood_fail_posts_nothing_records_skip():
    rec = _Recorder()
    out = _run(rec, _make_calls(rec, validation=(False, "1 failed")))
    assert out.status == "failed"
    assert rec.posted == []  # FAIL-CLOSED: never posts on a failing run
    assert rec.removed == ["/tmp/disposable-wt"]  # worktree still cleaned


def test_non_code_docs_pr_no_dogfood_attempted():
    rec = _Recorder()
    out = _run(rec, _make_calls(rec, files=["docs/X.md", "aragora/live/package-lock.json"]))
    assert out.status == "skipped"
    assert "no scoped validation" in out.reason
    assert rec.checked_out == []  # never checked out
    assert rec.posted == []


def test_untrusted_fork_skipped_before_checkout():
    rec = _Recorder()
    out = _run(rec, _make_calls(rec, is_cross=True))
    assert out.status == "skipped"
    assert "untrusted" in out.reason
    assert rec.checked_out == []  # never executed PR code
    assert rec.posted == []


def test_untrusted_branch_namespace_skipped():
    rec = _Recorder()
    out = _run(rec, _make_calls(rec, head_ref="feature/sketchy"))
    assert out.status == "skipped"
    assert "untrusted" in out.reason
    assert rec.posted == []


def test_lint_not_counting_does_not_post():
    rec = _Recorder()
    out = _run(
        rec,
        _make_calls(rec, lint={"would_count": True, "dogfood_evidence": []}),
    )
    assert out.status == "skipped"
    assert "would not count" in out.reason
    assert rec.posted == []
    assert rec.removed == ["/tmp/disposable-wt"]  # cleaned


def test_worktree_cleaned_even_when_checkout_fails():
    rec = _Recorder()
    out = _run(rec, _make_calls(rec, checkout_ok=False))
    assert out.status == "skipped"
    assert "checkout failed" in out.reason
    assert rec.removed == ["/tmp/disposable-wt"]  # finally-block cleanup


def test_worktree_cleaned_even_on_validation_exception():
    rec = _Recorder()
    calls = _make_calls(rec)

    def boom(cmd: list[str], cwd: str, timeout: int):
        raise RuntimeError("kaboom")

    calls["run_validation"] = boom
    try:
        _run(rec, calls)
    except RuntimeError:
        pass
    assert rec.removed == ["/tmp/disposable-wt"]  # finally still ran


def test_dry_run_does_not_post_even_on_pass():
    rec = _Recorder()
    out = _run(rec, _make_calls(rec), apply=False)
    assert out.status == "skipped"
    assert "dry-run" in out.reason
    assert out.would_count is True
    assert rec.posted == []
    assert rec.removed == ["/tmp/disposable-wt"]


def test_timeout_honored_in_default_run_validation(monkeypatch):
    import subprocess as _sp

    def fake_run(*args, **kwargs):
        assert kwargs.get("timeout") == 5
        raise _sp.TimeoutExpired(cmd="x", timeout=5)

    monkeypatch.setattr(df.subprocess, "run", fake_run)
    passed, out = df.default_run_validation([sys.executable, "-c", "pass"], cwd=".", timeout=5)
    assert passed is False
    assert "timed out" in out
