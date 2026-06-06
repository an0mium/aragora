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
