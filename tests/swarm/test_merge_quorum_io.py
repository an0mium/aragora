"""Tests for I/O helpers in aragora.swarm.merge_quorum_io.

Focused on ``fetch_pr_tier``, which must read the tier from the per-PR rows
under the merge-packet ``entries`` envelope (not the top-level object).
"""

from __future__ import annotations

import json
import subprocess

from aragora.swarm import merge_quorum_io as m


def _proc(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["gh"], returncode=returncode, stdout=stdout, stderr="")


def test_read_env_prefers_app_installation_token(monkeypatch) -> None:
    from aragora.swarm import github_app_auth

    monkeypatch.setattr(
        github_app_auth, "get_github_app_installation_token", lambda env=None: "fake-app-token"
    )
    env = m._read_env()
    assert env["GH_TOKEN"] == "fake-app-token"
    assert env["GITHUB_TOKEN"] == "fake-app-token"
    assert env["ARAGORA_GITHUB_AUTH_SOURCE"] == "github_app_installation"


def test_read_env_degrades_to_ambient_auth_without_app_config(monkeypatch) -> None:
    from aragora.swarm import github_app_auth

    # No App config -> mint returns None -> read env carries no App-source tag, so
    # gh falls back to the operator's ambient auth instead of crashing.
    monkeypatch.setattr(github_app_auth, "get_github_app_installation_token", lambda env=None: None)
    env = m._read_env()
    assert env.get("ARAGORA_GITHUB_AUTH_SOURCE") != "github_app_installation"


def test_fetch_pr_context_routes_reads_through_app_token(monkeypatch) -> None:
    from aragora.swarm import github_app_auth

    monkeypatch.setattr(
        github_app_auth, "get_github_app_installation_token", lambda env=None: "fake-app-token"
    )
    captured_envs: list[dict] = []

    def capture_run(args, *, env=None, timeout=m._GH_TIMEOUT):
        captured_envs.append(env or {})
        return _proc(json.dumps({"headRefOid": "abc", "commits": [], "statusCheckRollup": []}))

    monkeypatch.setattr(m, "run", capture_run)
    m.fetch_pr_context("o/r", 1)
    assert captured_envs, "expected fetch_pr_context to shell out to gh"
    assert captured_envs[0].get("GH_TOKEN") == "fake-app-token"
    assert captured_envs[0].get("ARAGORA_GITHUB_AUTH_SOURCE") == "github_app_installation"


def test_fetch_pr_context_uses_rest_fallback_after_pr_view_transport(monkeypatch) -> None:
    calls: list[list[str]] = []
    head = "f" * 40

    def fake_run(args, *, env=None, timeout=m._GH_TIMEOUT):
        del env, timeout
        calls.append(args)
        if args[:4] == ["gh", "pr", "view", "8532"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="error connecting to api.github.com",
            )
        if args == ["gh", "api", "repos/o/r/pulls/8532"]:
            return _proc(
                json.dumps(
                    {
                        "number": 8532,
                        "title": "deps",
                        "html_url": "https://github.com/o/r/pull/8532",
                        "head": {"sha": head, "ref": "dependabot/sdk"},
                        "base": {"sha": "b" * 40, "ref": "main"},
                        "state": "open",
                        "draft": False,
                        "mergeable": True,
                        "mergeable_state": "blocked",
                        "user": {"login": "dependabot[bot]"},
                    }
                )
            )
        if args == ["gh", "api", "repos/o/r/pulls/8532/files?per_page=100"]:
            return _proc(json.dumps([{"filename": "sdk/typescript/package.json"}]))
        if args == ["gh", "api", "repos/o/r/issues/8532/comments?per_page=100"]:
            return _proc(json.dumps([]))
        if args == ["gh", "api", "repos/o/r/pulls/8532/reviews?per_page=100"]:
            return _proc(json.dumps([]))
        if args == ["gh", "api", "repos/o/r/pulls/8532/commits?per_page=100"]:
            return _proc(
                json.dumps(
                    [
                        {
                            "sha": head,
                            "commit": {"author": {"date": "2026-06-23T01:02:03Z"}},
                        }
                    ]
                )
            )
        if args == ["gh", "api", f"repos/o/r/commits/{head}/statuses?per_page=100"]:
            return _proc(json.dumps([]))
        if args == ["gh", "api", f"repos/o/r/commits/{head}/status"]:
            return _proc(json.dumps({"statuses": []}))
        if args == ["gh", "api", f"repos/o/r/commits/{head}/check-runs?per_page=100"]:
            return _proc(
                json.dumps(
                    {
                        "total_count": 2,
                        "check_runs": [
                            {
                                "name": "lint",
                                "status": "completed",
                                "conclusion": "success",
                                "completed_at": "2026-06-23T01:04:00Z",
                            },
                            {
                                "name": "aragora-merge-quorum",
                                "status": "completed",
                                "conclusion": "failure",
                                "completed_at": "2026-06-23T01:05:00Z",
                            },
                        ],
                    }
                )
            )
        if args == [
            "gh",
            "api",
            "repos/o/r/branches/main/protection/required_status_checks",
        ]:
            return _proc(json.dumps({"strict": False, "checks": [{"context": "lint"}]}))
        raise AssertionError(args)

    monkeypatch.setattr(m, "run", fake_run)

    ctx = m.fetch_pr_context("o/r", 8532)

    assert ctx["head_sha"] == head
    assert ctx["head_committed_at"] == "2026-06-23T01:02:03Z"
    assert ctx["quorum_conclusion"] == "FAILURE"
    assert ctx["has_real_required_failure"] is False
    assert ctx["rest_fallback"]["enabled"] is True
    assert any(call[:4] == ["gh", "pr", "view", "8532"] for call in calls)
    assert any(call[:3] == ["gh", "api", "repos/o/r/pulls/8532"] for call in calls)


def test_fetch_pr_tier_reads_nested_entries(monkeypatch) -> None:
    payload = {
        "version": "merge_authorization_packet.v1",
        "entries": [{"pr_number": 7742, "tier": 4, "tier_name": "tier_4_preapproval_required"}],
    }
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc(json.dumps(payload)))
    assert m.fetch_pr_tier("o/r", 7742) == 4


def test_fetch_merge_packet_classification_reads_semantic_fields(monkeypatch) -> None:
    payload = {
        "version": "merge_authorization_packet.v1",
        "entries": [
            {
                "pr_number": 7754,
                "head_sha": "abc123",
                "tier": 2,
                "status": "repair_or_wait",
                "verdict": "not_ready_for_settlement",
                "requires_human_risk_settlement": False,
            }
        ],
    }
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc(json.dumps(payload)))

    packet = m.fetch_merge_packet_classification("o/r", 7754)

    assert packet is not None
    assert packet.pr_number == 7754
    assert packet.head_sha == "abc123"
    assert packet.tier == 2
    assert packet.status == "repair_or_wait"
    assert packet.verdict == "not_ready_for_settlement"
    assert packet.requires_human_risk_settlement is False


def test_fetch_quorum_run_packet_classification_parses_log(monkeypatch) -> None:
    log = (
        "PR #7754 | Tier 4 | status=human_preapproval_required | "
        "verdict=tier_4_human_preapproval_required\n"
    )
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc(log))

    packet = m.fetch_quorum_run_packet_classification("o/r", run_id=123, pr=7754, head_sha="abc123")

    assert packet is not None
    assert packet.source == "ci"
    assert packet.tier == 4
    assert packet.requires_human_risk_settlement is True


def test_fetch_pr_tier_filters_by_pr_number(monkeypatch) -> None:
    # A multi-PR envelope must resolve the requested PR, never the first row.
    payload = {"entries": [{"pr_number": 111, "tier": 1}, {"pr_number": 7742, "tier": 4}]}
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc(json.dumps(payload)))
    assert m.fetch_pr_tier("o/r", 7742) == 4
    assert m.fetch_pr_tier("o/r", 111) == 1


def test_fetch_pr_tier_coerces_string_pr_number(monkeypatch) -> None:
    payload = {"entries": [{"pr_number": "7742", "tier": "4"}]}
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc(json.dumps(payload)))
    assert m.fetch_pr_tier("o/r", 7742) == 4


def test_fetch_pr_tier_none_when_pr_number_absent_from_envelope(monkeypatch) -> None:
    # Rows disclose pr_number but none match the request -> no wrong-PR fallback.
    payload = {"entries": [{"pr_number": 111, "tier": 1}]}
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc(json.dumps(payload)))
    assert m.fetch_pr_tier("o/r", 999) is None


def test_fetch_pr_tier_accepts_bare_list(monkeypatch) -> None:
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc(json.dumps([{"tier": 2}])))
    assert m.fetch_pr_tier("o/r", 1) == 2


def test_fetch_pr_tier_accepts_single_entry_dict(monkeypatch) -> None:
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc(json.dumps({"tier": 1})))
    assert m.fetch_pr_tier("o/r", 1) == 1


def test_fetch_pr_tier_none_when_no_tier(monkeypatch) -> None:
    monkeypatch.setattr(
        m, "run", lambda *a, **k: _proc(json.dumps({"entries": [{"pr_number": 1}]}))
    )
    assert m.fetch_pr_tier("o/r", 1) is None


def test_fetch_pr_tier_none_on_bad_json(monkeypatch) -> None:
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc("not json"))
    assert m.fetch_pr_tier("o/r", 1) is None


def test_fetch_pr_tier_none_on_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(m, "run", lambda *a, **k: _proc("", returncode=1))
    assert m.fetch_pr_tier("o/r", 1) is None


def test_fetch_pr_tier_none_on_timeout(monkeypatch) -> None:
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="gh", timeout=1)

    monkeypatch.setattr(m, "run", _boom)
    assert m.fetch_pr_tier("o/r", 1) is None
