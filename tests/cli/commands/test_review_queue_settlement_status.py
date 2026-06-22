"""Atomic 'aragora/human-settlement' status posting for record-settlement.

Guards the integrity property: a green human-settlement commit status can only
exist when its backing receipt was durably written. The status POST is performed
by record-settlement itself (after the receipt), never as a decoupled
``gh api ... statuses`` command that runs regardless of receipt success.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from aragora.cli.commands import review_queue as rq


def _args(**over):
    base = dict(
        pr="7521",
        head_sha="83611936eed7d10d683c15b0af21411288e16b17",
        action="approve",
        reason="Operator risk settlement: Tier 4",
        repo="synaptent/aragora",
        review_queue_root=None,
        apply_post_merge_lane_audit=False,
        post_github_status=True,
        github_status_context="aragora/human-settlement",
        json=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _fake_result():
    receipt = SimpleNamespace(to_dict=lambda: {"head_sha": "83611936eed7", "pr_number": 7521})
    return SimpleNamespace(
        receipt=receipt,
        receipt_sha256="sha256:1dc0212113e65dd0c7c2d3101494e1aa739cfc34d",
        idempotent=False,
        written=True,
        post_merge_lane_audit_failed=False,
        to_dict=lambda: {
            "head_sha": "83611936eed7",
            "pr_number": 7521,
            "receipt_sha256": "sha256:1dc0212113e65dd0c7c2d3101494e1aa739cfc34d",
            "written": True,
        },
    )


@pytest.fixture
def gh_calls(monkeypatch):
    calls: list[list[str]] = []

    def fake_gh_json(args):
        calls.append(list(args))
        if args[:3] == ["repo", "view", "--json"]:
            return {"nameWithOwner": "synaptent/aragora"}
        if args[:1] == ["api"]:
            return {"state": "success", "context": "aragora/human-settlement"}
        return None

    monkeypatch.setattr(rq, "_gh_json", fake_gh_json)
    monkeypatch.setattr(rq, "_require_clean_worktree", lambda repo_root: None)
    return calls


def _statuses_posts(calls):
    return [c for c in calls if c[:1] == ["api"] and any("statuses/" in part for part in c)]


def test_status_posted_after_successful_receipt(gh_calls, monkeypatch):
    monkeypatch.setattr(rq, "_record_external_settlement", lambda **kw: _fake_result())

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = rq._cmd_record_settlement(_args())

    assert rc == 0
    posts = _statuses_posts(gh_calls)
    assert len(posts) == 1, gh_calls
    flat = " ".join(posts[0])
    assert "state=success" in flat
    assert "context=aragora/human-settlement" in flat
    # The receipt sha is embedded in the status description (status->receipt link).
    assert "sha256:1dc0212113e65dd0c7c2d3101494e1aa739cfc34d" in flat
    payload = json.loads(buf.getvalue())
    assert payload["github_status"]["posted"] is True


def test_status_NEVER_posted_when_receipt_write_fails(gh_calls, monkeypatch):
    """The integrity property: a failed/aborted receipt must not yield a status."""

    def boom(**kw):
        raise rq._GhError("settlement requires a clean worktree")

    monkeypatch.setattr(rq, "_record_external_settlement", boom)

    rc = rq._cmd_record_settlement(_args())

    assert rc == 1
    assert _statuses_posts(gh_calls) == [], "status POST must not run when receipt write fails"


def test_status_failure_after_receipt_returns_error_but_keeps_receipt(monkeypatch):
    """Safe direction: receipt written, status POST fails -> exit 1, no green status."""
    monkeypatch.setattr(rq, "_require_clean_worktree", lambda repo_root: None)
    monkeypatch.setattr(rq, "_record_external_settlement", lambda **kw: _fake_result())

    def fake_gh_json(args):
        if args[:3] == ["repo", "view", "--json"]:
            return {"nameWithOwner": "synaptent/aragora"}
        raise rq._GhError("status endpoint 422")

    monkeypatch.setattr(rq, "_gh_json", fake_gh_json)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = rq._cmd_record_settlement(_args())

    assert rc == 1
    payload = json.loads(buf.getvalue())
    assert payload["github_status"]["posted"] is False
    assert payload["written"] is True  # receipt survived


def test_no_status_posted_without_flag(gh_calls, monkeypatch):
    monkeypatch.setattr(rq, "_record_external_settlement", lambda **kw: _fake_result())

    rc = rq._cmd_record_settlement(_args(post_github_status=False))

    assert rc == 0
    assert _statuses_posts(gh_calls) == []


def test_resolve_repo_slug_prefers_override():
    assert rq._resolve_settlement_repo_slug("owner/name") == "owner/name"


def test_resolve_repo_slug_falls_back_to_gh(monkeypatch):
    monkeypatch.setattr(rq, "_gh_json", lambda args: {"nameWithOwner": "synaptent/aragora"})
    assert rq._resolve_settlement_repo_slug(None) == "synaptent/aragora"
