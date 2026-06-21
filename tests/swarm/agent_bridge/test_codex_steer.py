"""Tests for the advisory, monotonic-restrictive Codex steer-back channel.

The central property under test: a steering directive can only ever ADD caution.
``effective_forbidden_actions`` is always a superset of the caller's baseline,
non-steerable tokens are rejected, and a malformed directive on disk is dropped
(which can never loosen the posture).
"""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pytest

from aragora.swarm.agent_bridge import codex_steer as st

NOW = datetime(2026, 6, 15, 21, 0, 0, tzinfo=UTC)


def _directive(**kwargs) -> st.SteeringDirective:
    base = {"issued_by": "claude", "issued_at": "2026-06-15T20:55:00Z"}
    base.update(kwargs)
    return st.SteeringDirective(**base)


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    mailbox = tmp_path / "mailbox.jsonl"
    st.write_directive(
        _directive(add_forbidden_actions=["merge"], note="hold off"), mailbox_path=mailbox
    )
    out = st.read_directives(mailbox_path=mailbox, hours=24.0, now=NOW)
    assert len(out) == 1
    assert out[0].add_forbidden_actions == ["merge"]
    assert out[0].note == "hold off"


def test_non_steerable_action_is_rejected() -> None:
    with pytest.raises(st.SteeringValidationError):
        _directive(add_forbidden_actions=["grant_admin"])  # not in the vocabulary


def test_effective_forbidden_actions_is_always_a_superset() -> None:
    base = ["merge"]
    directives = [_directive(add_forbidden_actions=["mark_ready", "rerun_required_ci"])]
    effective = st.effective_forbidden_actions(base, directives)
    assert set(base).issubset(set(effective))
    assert "mark_ready" in effective and "rerun_required_ci" in effective


def test_effective_forbidden_actions_cannot_remove_baseline() -> None:
    # Even with no directives, the baseline is preserved exactly (no shrink path).
    base = ["merge", "mutate_branch_protection"]
    assert set(base).issubset(set(st.effective_forbidden_actions(base, [])))


def test_target_pr_scopes_the_directive() -> None:
    directives = [_directive(add_forbidden_actions=["merge"], target_pr=8444)]
    # Applies to the matching PR...
    assert "merge" in st.effective_forbidden_actions([], directives, pr=8444)
    # ...but not to a different PR.
    assert "merge" not in st.effective_forbidden_actions([], directives, pr=8405)


def test_global_directive_applies_to_every_pr() -> None:
    directives = [_directive(add_forbidden_actions=["merge"], target_pr=None)]
    assert "merge" in st.effective_forbidden_actions([], directives, pr=999)
    assert "merge" in st.effective_forbidden_actions([], directives, pr=None)


def test_off_limits_pr_gets_full_steerable_set() -> None:
    directives = [_directive(off_limits_prs=[8446])]
    effective = st.effective_forbidden_actions([], directives, pr=8446)
    assert st.STEERABLE_FORBIDDEN_ACTIONS.issubset(set(effective))
    # A non-pinned PR is unaffected.
    assert st.effective_forbidden_actions([], directives, pr=1) == []


def test_malformed_directive_on_disk_is_dropped_not_loosening(tmp_path: Path) -> None:
    mailbox = tmp_path / "mailbox.jsonl"
    mailbox.write_text(
        "garbage line\n"
        + json.dumps(
            {
                "issued_by": "x",
                "issued_at": "2026-06-15T20:55:00Z",
                "add_forbidden_actions": ["unknown_token"],
            }
        )  # invalid -> dropped
        + "\n"
        + json.dumps(
            {
                "issued_by": "claude",
                "issued_at": "2026-06-15T20:56:00Z",
                "add_forbidden_actions": ["merge"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = st.read_directives(mailbox_path=mailbox, hours=24.0, now=NOW)
    assert len(out) == 1
    assert out[0].add_forbidden_actions == ["merge"]


def test_read_directives_window_filters_old(tmp_path: Path) -> None:
    mailbox = tmp_path / "mailbox.jsonl"
    st.write_directive(
        _directive(add_forbidden_actions=["merge"], issued_at="2026-06-10T00:00:00Z"),
        mailbox_path=mailbox,
    )
    assert st.read_directives(mailbox_path=mailbox, hours=12.0, now=NOW) == []


def test_render_phase0_block_empty_without_directives() -> None:
    assert st.render_phase0_block([]) == ""


def test_render_phase0_block_includes_pins_actions_and_note() -> None:
    directives = [
        _directive(add_forbidden_actions=["merge"], off_limits_prs=[8446], note="fusion lane"),
    ]
    block = st.render_phase0_block(directives, pr=8446)
    assert "OFF-LIMITS" in block and "8446" in block
    assert "fusion lane" in block
    assert "sole merge authority" in block


def test_off_limits_prs_rejects_nonpositive() -> None:
    with pytest.raises(st.SteeringValidationError):
        _directive(off_limits_prs=[0])
