"""Tests for the count-based open-PR WIP cap (Loop Control Plane v2 companion).

``wip_budget`` mirrors ``loop_budget`` (dollar ceilings) but governs a different
decision: whether a generation lane should create *another* PR given how many it
already has open. The fail-safe direction matches loop_budget's philosophy -- a
fabricated/unknown count must NEVER classify ``over_cap`` and block a legitimate
fleet; blocking on a number we cannot trust is exactly the fabricated-alarm class
the control plane exists to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aragora.swarm.wip_budget import (
    WIP_DEGRADED,
    WIP_OVER_CAP,
    WIP_UNAVAILABLE,
    WIP_WITHIN_CAP,
    WipPolicy,
    _as_count,
    classify_wip,
    resolve_wip_budget,
)


@pytest.fixture(autouse=True)
def _clear_wip_env(monkeypatch):
    # The env fleet-default would flip the no-ceiling / no-count assertions
    # (missing-file -> a cap; count-only -> ok). Mirror test_loop_budget's
    # delenv discipline so these tests are environment-independent.
    monkeypatch.delenv("ARAGORA_WIP_OPEN_PR_CAP", raising=False)


# --- policy load -----------------------------------------------------------


def _write_policy(root: Path, payload: dict) -> None:
    d = root / ".aragora"
    d.mkdir(parents=True, exist_ok=True)
    (d / "wip_budgets.json").write_text(json.dumps(payload), encoding="utf-8")


def test_policy_load_per_fleet_and_default(tmp_path):
    _write_policy(tmp_path, {"default_cap": 40, "fleets": {"an0mium": {"cap": 20}}})
    policy = WipPolicy.load(tmp_path)
    assert policy.cap_for("an0mium")[0] == 20
    assert policy.cap_for("other")[0] == 40  # falls to default


def test_policy_load_missing_file_is_empty_not_raising(tmp_path):
    policy = WipPolicy.load(tmp_path)  # no .aragora/wip_budgets.json
    assert policy.cap_for("any") == (None, "none")


def test_policy_load_unreadable_json_degrades_to_empty(tmp_path):
    d = tmp_path / ".aragora"
    d.mkdir()
    (d / "wip_budgets.json").write_text("{not json", encoding="utf-8")
    policy = WipPolicy.load(tmp_path)
    assert policy.cap_for("any")[0] is None


def test_policy_env_fleet_default_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_WIP_OPEN_PR_CAP", "15")
    policy = WipPolicy.load(tmp_path)  # no policy file -> env default
    cap, source = policy.cap_for("an0mium")
    assert cap == 15
    assert "ARAGORA_WIP_OPEN_PR_CAP" in source


def test_policy_rejects_negative_and_nonint_caps(tmp_path):
    _write_policy(tmp_path, {"default_cap": -5, "fleets": {"x": {"cap": 3.7}}})
    policy = WipPolicy.load(tmp_path)
    assert policy.cap_for("x")[0] is None  # 3.7 not an integer cap
    assert policy.cap_for("y")[0] is None  # -5 rejected


# --- resolve quadrants -----------------------------------------------------


def test_resolve_ceiling_plus_count_is_ok(tmp_path):
    _write_policy(tmp_path, {"fleets": {"an0mium": {"cap": 20}}})
    wip = resolve_wip_budget(12, WipPolicy.load(tmp_path), "an0mium")
    assert wip["source_status"] == "ok"
    assert wip["ceiling"] == 20
    assert wip["open_pr_count"] == 12
    assert wip["remaining"] == 8


def test_resolve_ceiling_unknown_count_is_degraded(tmp_path):
    _write_policy(tmp_path, {"fleets": {"an0mium": {"cap": 20}}})
    wip = resolve_wip_budget(None, WipPolicy.load(tmp_path), "an0mium")
    assert wip["source_status"] == "degraded"
    assert wip["ceiling"] == 20
    assert wip["open_pr_count"] is None
    assert wip["remaining"] is None


def test_resolve_count_no_ceiling_is_degraded(tmp_path):
    wip = resolve_wip_budget(50, WipPolicy.load(tmp_path), "an0mium")
    assert wip["source_status"] == "degraded"
    assert wip["ceiling"] is None
    assert wip["open_pr_count"] == 50


def test_resolve_neither_is_unavailable(tmp_path):
    wip = resolve_wip_budget(None, WipPolicy.load(tmp_path), "an0mium")
    assert wip["source_status"] == "unavailable"


# --- classify_wip ----------------------------------------------------------


def test_classify_within_cap_allows_generation(tmp_path):
    _write_policy(tmp_path, {"fleets": {"f": {"cap": 20}}})
    decision = classify_wip(resolve_wip_budget(12, WipPolicy.load(tmp_path), "f"))
    assert decision.verdict == WIP_WITHIN_CAP
    assert decision.allow_generation is True


def test_classify_over_cap_blocks_generation_at_boundary(tmp_path):
    _write_policy(tmp_path, {"fleets": {"f": {"cap": 20}}})
    pol = WipPolicy.load(tmp_path)
    at = classify_wip(resolve_wip_budget(20, pol, "f"))  # count == cap -> over
    assert at.verdict == WIP_OVER_CAP
    assert at.allow_generation is False
    under = classify_wip(resolve_wip_budget(19, pol, "f"))  # cap-1 -> within
    assert under.verdict == WIP_WITHIN_CAP
    assert under.allow_generation is True


def test_classify_unknown_count_fails_safe_never_over_cap(tmp_path):
    # The fail-safe contract: a ceiling with an unknown count must DEGRADE and
    # keep allowing generation, never fabricate an over_cap halt.
    _write_policy(tmp_path, {"fleets": {"f": {"cap": 20}}})
    decision = classify_wip(resolve_wip_budget(None, WipPolicy.load(tmp_path), "f"))
    assert decision.verdict == WIP_DEGRADED
    assert decision.allow_generation is True


def test_classify_unavailable_allows_generation(tmp_path):
    decision = classify_wip(resolve_wip_budget(None, WipPolicy.load(tmp_path), "f"))
    assert decision.verdict == WIP_UNAVAILABLE
    assert decision.allow_generation is True


def test_classify_self_derives_and_ignores_misleading_status():
    # A caller cannot bypass the fail-safe by mislabelling status="ok" with a
    # missing count: classify derives the verdict from the data, not the label.
    degraded = classify_wip({"source_status": "ok", "open_pr_count": None, "ceiling": 20})
    assert degraded.verdict == WIP_DEGRADED
    assert degraded.allow_generation is True
    # ...and an honest ok dict still gates correctly.
    over = classify_wip({"source_status": "ok", "open_pr_count": 25, "ceiling": 20})
    assert over.verdict == WIP_OVER_CAP


def test_cap_zero_freezes_generation(tmp_path):
    # cap=0 is the deliberate "freeze" knob: 0 open PRs already >= 0 -> over_cap.
    _write_policy(tmp_path, {"fleets": {"frozen": {"cap": 0}}})
    decision = classify_wip(resolve_wip_budget(0, WipPolicy.load(tmp_path), "frozen"))
    assert decision.verdict == WIP_OVER_CAP
    assert decision.allow_generation is False


def test_as_count_rejects_untrusted_values():
    # Parity with loop_budget._as_float regression guards: anything that isn't a
    # clean non-negative integer reads as "unknown" (None) and so fails safe.
    assert _as_count(True) is None  # bool is not a count
    assert _as_count(False) is None
    assert _as_count(-3) is None
    assert _as_count(3.7) is None  # non-integral float
    assert _as_count(float("nan")) is None
    assert _as_count(float("inf")) is None
    assert _as_count("not-an-int") is None
    assert _as_count(3.0) == 3  # integral float ok
    assert _as_count("12") == 12
    assert _as_count(0) == 0
