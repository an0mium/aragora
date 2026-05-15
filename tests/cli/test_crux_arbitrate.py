"""Tests for aragora.cli.commands.crux_arbitrate (DIC-27 / #6221).

All tests are hermetic: no live API calls, no queue mutation, no disk I/O
outside tmpdir fixtures.

Flag-gate coverage
------------------
- ``_dry_run_*``  — always works regardless of ARAGORA_CRUX_ARBITRATION_ENABLED
- ``_flag_off_*`` — flag absent → exits 1 with helpful message
- ``_flag_on_*``  — flag set → creates CruxArbitration via build_arbitration()
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aragora.cli.commands.crux_arbitrate import (
    _load_cruxes,
    _render_arbitration,
    _render_dry_run,
    cmd_crux_arbitrate,
)
from aragora.epistemic.arbitration import (
    PERSISTENT_CRUX_MIN_CONSECUTIVE,
    PERSISTENT_CRUX_MIN_SCORE,
    PersistentCrux,
    build_arbitration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_QUALIFYING_CRUX = PersistentCrux(
    crux_id="crux_aaa",
    statement="Three green soaks are required before expanding B2.",
    question_family_id="qfam_b2_expansion",
    consecutive_debate_count=PERSISTENT_CRUX_MIN_CONSECUTIVE,
    load_bearing_score=PERSISTENT_CRUX_MIN_SCORE,
    cruxset_receipt_ids=("rcpt_001", "rcpt_002", "rcpt_003"),
)

_NOT_QUALIFYING_CRUX = PersistentCrux(
    crux_id="crux_bbb",
    statement="Performance budget is the key constraint.",
    question_family_id="qfam_perf",
    consecutive_debate_count=1,  # below threshold
    load_bearing_score=0.2,  # below threshold
    cruxset_receipt_ids=(),
)


def _write_crux_file(tmp_path: Path, cruxes: list[PersistentCrux]) -> Path:
    p = tmp_path / "cruxes.json"
    p.write_text(json.dumps([c.to_dict() for c in cruxes]))
    return p


def _write_single_crux_file(tmp_path: Path, crux: PersistentCrux) -> Path:
    p = tmp_path / "single.json"
    p.write_text(json.dumps(crux.to_dict()))
    return p


def _make_args(**kwargs):
    """Build a minimal argparse.Namespace for cmd_crux_arbitrate."""
    import argparse

    defaults = {
        "input": None,
        "dry_run": False,
        "crux_id": None,
        "side": None,
        "rationale": None,
        "operator": "operator",
        "expires_days": 90,
        "evidence": None,
        "output": None,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _load_cruxes unit tests
# ---------------------------------------------------------------------------


def test_load_list_of_cruxes(tmp_path):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX, _NOT_QUALIFYING_CRUX])
    cruxes = _load_cruxes(str(p))
    assert len(cruxes) == 2
    assert cruxes[0].crux_id == "crux_aaa"
    assert cruxes[1].crux_id == "crux_bbb"


def test_load_single_crux_dict(tmp_path):
    p = _write_single_crux_file(tmp_path, _QUALIFYING_CRUX)
    cruxes = _load_cruxes(str(p))
    assert len(cruxes) == 1
    assert cruxes[0].crux_id == "crux_aaa"


def test_load_preserves_receipt_ids(tmp_path):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    cruxes = _load_cruxes(str(p))
    assert cruxes[0].cruxset_receipt_ids == ("rcpt_001", "rcpt_002", "rcpt_003")


def test_load_bad_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json}")
    with pytest.raises(Exception):
        _load_cruxes(str(p))


# ---------------------------------------------------------------------------
# _render_dry_run unit tests
# ---------------------------------------------------------------------------


def test_render_dry_run_text_shows_counts():
    out = _render_dry_run([_QUALIFYING_CRUX, _NOT_QUALIFYING_CRUX], json_output=False)
    assert "2 crux(es) loaded" in out
    assert "1 qualify" in out
    assert "crux_aaa" in out
    assert "crux_bbb" in out


def test_render_dry_run_json_structure():
    out = _render_dry_run([_QUALIFYING_CRUX, _NOT_QUALIFYING_CRUX], json_output=True)
    data = json.loads(out)
    assert len(data["qualifying"]) == 1
    assert len(data["not_qualifying"]) == 1
    assert data["qualifying"][0]["crux_id"] == "crux_aaa"


def test_render_dry_run_all_not_qualifying():
    out = _render_dry_run([_NOT_QUALIFYING_CRUX], json_output=False)
    assert "0 qualify" in out


# ---------------------------------------------------------------------------
# dry-run mode via cmd_crux_arbitrate (no flag needed)
# ---------------------------------------------------------------------------


def test_dry_run_prints_qualifying(tmp_path, capsys):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX, _NOT_QUALIFYING_CRUX])
    args = _make_args(input=str(p), dry_run=True)
    cmd_crux_arbitrate(args)
    out = capsys.readouterr().out
    assert "crux_aaa" in out
    assert "qualify" in out.lower()


def test_dry_run_json(tmp_path, capsys):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(input=str(p), dry_run=True, json=True)
    cmd_crux_arbitrate(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "qualifying" in data


# ---------------------------------------------------------------------------
# flag-OFF tests (ARAGORA_CRUX_ARBITRATION_ENABLED not set)
# ---------------------------------------------------------------------------


def test_flag_off_live_mode_exits_nonzero(tmp_path):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(
        input=str(p),
        crux_id="crux_aaa",
        side="accept",
        rationale="Test rationale",
    )
    env = {k: v for k, v in os.environ.items() if k != "ARAGORA_CRUX_ARBITRATION_ENABLED"}
    env["ARAGORA_CRUX_ARBITRATION_ENABLED"] = ""
    old = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        with pytest.raises(SystemExit) as exc_info:
            cmd_crux_arbitrate(args)
        assert exc_info.value.code == 1
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_flag_off_error_message_is_helpful(tmp_path, capsys):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(
        input=str(p),
        crux_id="crux_aaa",
        side="accept",
        rationale="Test",
    )
    old_val = os.environ.pop("ARAGORA_CRUX_ARBITRATION_ENABLED", None)
    try:
        with pytest.raises(SystemExit):
            cmd_crux_arbitrate(args)
        err = capsys.readouterr().err
        assert "ARAGORA_CRUX_ARBITRATION_ENABLED" in err
        assert "--dry-run" in err
    finally:
        if old_val is not None:
            os.environ["ARAGORA_CRUX_ARBITRATION_ENABLED"] = old_val


# ---------------------------------------------------------------------------
# flag-ON tests
# ---------------------------------------------------------------------------


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("ARAGORA_CRUX_ARBITRATION_ENABLED", "1")


def test_flag_on_happy_path(tmp_path, capsys, flag_on):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(
        input=str(p),
        crux_id="crux_aaa",
        side="accept",
        rationale="Soaks confirm the claim holds.",
        operator="alice",
    )
    cmd_crux_arbitrate(args)
    out = capsys.readouterr().out
    assert "Arbitration created" in out
    assert "crux_aaa" in out
    assert "alice" in out
    assert "accept" in out


def test_flag_on_json_output(tmp_path, capsys, flag_on):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(
        input=str(p),
        crux_id="crux_aaa",
        side="defer",
        rationale="Need more data.",
        json=True,
    )
    cmd_crux_arbitrate(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["crux"]["crux_id"] == "crux_aaa"
    assert data["side"] == "defer"
    assert "checksum" in data
    assert data["checksum"]


def test_flag_on_writes_output_file(tmp_path, flag_on):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    out_file = tmp_path / "arb.json"
    args = _make_args(
        input=str(p),
        crux_id="crux_aaa",
        side="reject",
        rationale="Evidence does not support the claim.",
        output=str(out_file),
    )
    cmd_crux_arbitrate(args)
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["side"] == "reject"


def test_flag_on_unknown_crux_id_exits_nonzero(tmp_path, flag_on):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(
        input=str(p),
        crux_id="crux_doesnt_exist",
        side="accept",
        rationale="Test",
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_crux_arbitrate(args)
    assert exc_info.value.code == 1


def test_flag_on_disqualified_crux_exits_nonzero(tmp_path, flag_on):
    p = _write_crux_file(tmp_path, [_NOT_QUALIFYING_CRUX])
    args = _make_args(
        input=str(p),
        crux_id="crux_bbb",
        side="accept",
        rationale="Test",
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_crux_arbitrate(args)
    assert exc_info.value.code == 1


def test_flag_on_missing_crux_id_exits_nonzero(tmp_path, flag_on):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(input=str(p), side="accept", rationale="Test")
    with pytest.raises(SystemExit) as exc_info:
        cmd_crux_arbitrate(args)
    assert exc_info.value.code == 1


def test_flag_on_missing_side_exits_nonzero(tmp_path, flag_on):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(input=str(p), crux_id="crux_aaa", rationale="Test")
    with pytest.raises(SystemExit) as exc_info:
        cmd_crux_arbitrate(args)
    assert exc_info.value.code == 1


def test_flag_on_missing_rationale_exits_nonzero(tmp_path, flag_on):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(input=str(p), crux_id="crux_aaa", side="accept")
    with pytest.raises(SystemExit) as exc_info:
        cmd_crux_arbitrate(args)
    assert exc_info.value.code == 1


def test_flag_on_evidence_citations_stored(tmp_path, capsys, flag_on):
    p = _write_crux_file(tmp_path, [_QUALIFYING_CRUX])
    args = _make_args(
        input=str(p),
        crux_id="crux_aaa",
        side="accept",
        rationale="Evidence attached.",
        evidence=["docs/soak_policy.md", "https://example.com/result"],
        json=True,
    )
    cmd_crux_arbitrate(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "docs/soak_policy.md" in data["evidence_citations"]
    assert "https://example.com/result" in data["evidence_citations"]


# ---------------------------------------------------------------------------
# _render_arbitration unit test
# ---------------------------------------------------------------------------


def test_render_arbitration_text_format():
    arb = build_arbitration(
        _QUALIFYING_CRUX,
        operator="bob",
        side="split",
        rationale="Both sides valid in different contexts.",
    )
    text = _render_arbitration(arb, json_output=False)
    assert "Arbitration created" in text
    assert "bob" in text
    assert "split" in text
    assert "checksum" in text


def test_render_arbitration_json_roundtrip():
    arb = build_arbitration(
        _QUALIFYING_CRUX,
        operator="carol",
        side="defer",
        rationale="Pending further evidence.",
    )
    text = _render_arbitration(arb, json_output=True)
    data = json.loads(text)
    assert data["operator"] == "carol"
    assert data["side"] == "defer"
    assert data["is_reversed"] is False
