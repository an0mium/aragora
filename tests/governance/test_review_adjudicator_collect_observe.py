"""Governance pins for observe-only review adjudicator wiring (#8748)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aragora.swarm.quorum_evidence import ReviewerResult, collect_evidence

HEAD = "49a979d587f910aaad4fb0f0bed708dd48c97c35"
COMMITTED = "2026-06-04T09:57:49-05:00"


def _collect_with(
    monkeypatch: pytest.MonkeyPatch,
    *,
    flag: bool,
    reviewer_bodies: dict[str, str],
    apply: bool = True,
) -> dict[str, Any]:
    if flag:
        monkeypatch.setenv("ARAGORA_ENABLE_REVIEW_ADJUDICATOR", "1")
    else:
        monkeypatch.delenv("ARAGORA_ENABLE_REVIEW_ADJUDICATOR", raising=False)
    monkeypatch.setenv("ARAGORA_ENABLE_TIERED_MERGE_GATE", "1")

    def context_fetcher(_repo: str, _pr: int) -> dict[str, str]:
        return {"head_sha": HEAD, "head_committed_at": COMMITTED}

    def tier_fetcher(_repo: str, _pr: int) -> int:
        return 1

    def prompt_builder(_repo: str, _pr: int, _ctx: dict[str, Any]) -> str:
        return "review prompt"

    def reviewer_runner(family: str, _prompt: str) -> ReviewerResult:
        return ReviewerResult(family, reviewer_bodies[family], True)

    def linter(
        _pr: int,
        _head_sha: str,
        _head_committed_at: str,
        _author: str,
        body: str,
        _env: dict[str, str],
    ) -> dict[str, Any]:
        family = "claude" if "Model family: claude" in body else "openai"
        return {
            "would_count": "Verdict: PASS" in body,
            "counted_reviewer_ids": [family] if "Verdict: PASS" in body else [],
            "problems": [],
        }

    outcome = collect_evidence(
        repo="o/r",
        pr=1,
        families=["claude", "openai"],
        author="me",
        apply=apply,
        context_fetcher=context_fetcher,
        tier_fetcher=tier_fetcher,
        prompt_builder=prompt_builder,
        reviewer_runner=reviewer_runner,
        linter=linter,
        poster=lambda _repo, _pr, _body: None,
    )
    return outcome.to_dict()


def test_review_adjudicator_flag_off_keeps_collect_json_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _collect_with(
        monkeypatch,
        flag=False,
        reviewer_bodies={
            "claude": "Verdict: PASS\nVerified bounded reviewer timeout behavior.",
            "openai": "Verdict: CHANGES-REQUESTED\n- [P1] concrete blocker.",
        },
    )

    encoded = json.dumps(payload, sort_keys=True)

    assert '"adjudication"' not in encoded
    assert payload["action"] == "prepare"
    assert payload["supportive_families"] == ["claude"]
    assert payload["dissenting_families"] == ["openai"]


def test_review_adjudicator_flag_on_hard_bar_blocks_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _collect_with(
        monkeypatch,
        flag=True,
        reviewer_bodies={
            "claude": (
                "Verdict: PASS\n"
                "Verified aragora/swarm/quorum_evidence.py:2100 keeps posting prepare-only."
            ),
            "openai": (
                "Verdict: CHANGES-REQUESTED\n"
                "- [P1] aragora/swarm/quorum_evidence.py:2107 would post dissenting "
                "evidence without a prepare guard."
            ),
        },
    )

    assert payload["action"] == "prepare"
    assert payload["supportive_families"] == ["claude"]
    assert payload["dissenting_families"] == ["openai"]
    assert payload["adjudication"]["verdict"] == "adjudicated_block"
    assert payload["adjudication"]["verdict"] != "adjudicated_settle"
    assert payload["adjudication"]["blocking_findings"]


def test_review_adjudicator_flag_on_without_stall_records_no_adjudication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _collect_with(
        monkeypatch,
        flag=True,
        apply=False,
        reviewer_bodies={
            "claude": "Verdict: PASS\nVerified bounded reviewer timeout behavior.",
            "openai": "Verdict: PASS\nVerified transport retry behavior.",
        },
    )

    assert payload["action"] == "prepare"
    assert sorted(payload["supportive_families"]) == ["claude", "openai"]
    assert payload["dissenting_families"] == []
    assert "adjudication" not in payload


def test_review_adjudicator_exception_leaves_collect_outcome_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aragora.swarm import review_adjudicator

    def broken_adjudicate(_items: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(review_adjudicator, "adjudicate", broken_adjudicate)

    payload = _collect_with(
        monkeypatch,
        flag=True,
        reviewer_bodies={
            "claude": "Verdict: PASS\nVerified bounded reviewer timeout behavior.",
            "openai": "Verdict: CHANGES-REQUESTED\n- [P1] concrete blocker.",
        },
    )

    assert payload["action"] == "prepare"
    assert payload["supportive_families"] == ["claude"]
    assert payload["dissenting_families"] == ["openai"]
    assert "adjudication" not in payload
