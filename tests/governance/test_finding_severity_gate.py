"""Governance tests for the finding-severity dissent gate (Tier 4 pre-approval).

These tests are the pre-approval regression target for the design in
``docs/specs/FINDING_SEVERITY_GATE.md``, per
``docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`` (a change
to *what blocks a merge at a given Tier* is a Tier 4 merge-authority
self-modification).

They pin BOTH regimes of the opt-in ``ARAGORA_ENABLE_SEVERITY_GATED_DISSENT`` flag:

* **Flag OFF (default / strict)** — byte-identical to today: a
  ``Verdict: CHANGES-REQUESTED`` comment blocks regardless of finding severity, even
  one that lists only ``[P2]``; ``[P0]``/``[P1]`` blocks; a clean PASS counts.
* **Flag ON** — a CHANGES-REQUESTED comment with only ``[P2]``/``[P3]`` (or no
  finding) is *advisory*: non-blocking AND non-counting; a real ``[P1]`` finding or a
  populated Blocker label STILL blocks; a clean PASS still counts.

The two gate halves — ``review_queue._dissenting_views_from_comments`` and
``quorum_evidence.EvidenceItem.dissenting`` — are asserted to AGREE for the same body
under each flag state (lockstep invariant).
"""

from __future__ import annotations

import pytest

from aragora.cli.commands.review_queue import (
    _build_model_review_quorum,
    _dissenting_views_from_comments,
)
from aragora.cli.commands.review_queue_comment_verdicts import (
    has_blocking_finding_or_label,
    has_blocking_or_negative_verdict,
    highest_blocking_severity,
)
from aragora.swarm.quorum_evidence import EvidenceItem, severity_gated_dissent_enabled

_FLAG = "ARAGORA_ENABLE_SEVERITY_GATED_DISSENT"

# A stable head SHA cited in every grounded comment body so
# ``_is_comment_grounded_on_head`` accepts the comment under the SHA-prefix rule.
_HEAD = "cd87c5a1b2db34f04167906553502db3ede9525e"

_P2_ONLY_CHANGES_REQUESTED = (
    "## Claude independent model review\n"
    f"Current head: {_HEAD}\n"
    "Verdict: CHANGES-REQUESTED\n"
    "[P2] Prefer a constant over the magic number on line 40.\n"
    "[P3] Typo in the docstring."
)
_P1_CHANGES_REQUESTED = (
    "## Claude independent model review\n"
    f"Current head: {_HEAD}\n"
    "Verdict: CHANGES-REQUESTED\n"
    "[P1] Unvalidated input flows into the SQL query."
)
_BLOCKER_LABEL_CHANGES_REQUESTED = (
    "## Claude independent model review\n"
    f"Current head: {_HEAD}\n"
    "Verdict: CHANGES-REQUESTED\n"
    "Blocking finding: the auth check is bypassable."
)
_NO_FINDING_CHANGES_REQUESTED = (
    f"## Claude independent model review\nCurrent head: {_HEAD}\nVerdict: CHANGES-REQUESTED"
)
# A populated Blocker label whose finding text starts with bare "no" — common
# security phrasing ("no authentication", "no validation", "no authorization").
# This MUST block: a populated Blocker label always blocks (the stated invariant).
# Regression for the severity-gate bypass (openai #8574 P1).
_BLOCKER_LABEL_NO_AUTH_CHANGES_REQUESTED = (
    "## Claude independent model review\n"
    f"Current head: {_HEAD}\n"
    "Verdict: CHANGES-REQUESTED\n"
    "Blockers: no authentication on admin endpoint"
)
_BLOCKER_LABEL_NO_FINDING_SECURITY_PHRASINGS = (
    "Blockers: no authentication on admin endpoint",
    "Blockers: no validation",
    "Blockers: no authorization",
    "Blockers: no rate limiting on the login route",
)
# Legitimate no-finding Blocker values that must STAY advisory (non-blocking).
_BLOCKER_LABEL_GENUINE_NO_FINDING = (
    "Blockers: none",
    "Blockers: no issues",
    "Blockers: no blockers",
    "Blocking findings: no blocking findings",
    "Blockers: no concerns",
)


def _comment(body: str, *, login: str = "an0mium") -> dict:
    return {"author": {"login": login}, "body": body}


def _grounded_pr(comments: list[dict]) -> dict:
    """A minimal Tier-classifiable PR payload with a stable head and clean checks."""
    return {
        "number": 9999,
        "title": "test pr",
        "headRefOid": _HEAD,
        "files": [{"path": "aragora/cli/commands/swarm.py"}],
        "comments": comments,
        "reviews": [],
        "statusCheckRollup": [{"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"}],
    }


def _supporting_comments() -> list[dict]:
    """Two western-frontier PASS signals + adversarial dogfood so an otherwise-clean
    packet satisfies the quorum — the dissent comment is then the only variable."""
    return [
        {
            "author": {"login": "an0mium"},
            "body": (
                "## Codex review\n"
                f"Current head: {_HEAD}\n"
                "Verdict: approve.\nFocused adversarial dogfood passed."
            ),
        },
        {
            "author": {"login": "an0mium"},
            "body": (f"## Grok independent model review\nCurrent head: {_HEAD}\nVerdict: approve."),
        },
    ]


def _evidence(body: str, *, verdict: str = "changes_requested") -> EvidenceItem:
    return EvidenceItem(family="claude", body=body, would_count=False, verdict=verdict)


# --------------------------------------------------------------------------- #
# Flag default / helper sanity
# --------------------------------------------------------------------------- #


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert severity_gated_dissent_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_flag_truthy_values(monkeypatch, value):
    monkeypatch.setenv(_FLAG, value)
    assert severity_gated_dissent_enabled() is True


def test_highest_blocking_severity_pins_finding_recognition():
    assert highest_blocking_severity(_P1_CHANGES_REQUESTED) == "P1"
    assert highest_blocking_severity(_P2_ONLY_CHANGES_REQUESTED) is None
    assert highest_blocking_severity(_NO_FINDING_CHANGES_REQUESTED) is None
    # "[P1] None:" is an explicit no-finding head and must NOT register.
    assert highest_blocking_severity("[P1] None: no blocking issues found") is None


def test_has_blocking_finding_or_label_excludes_bare_negative_verdict():
    # The bare-negative-verdict trigger is the ONLY thing the severity gate drops.
    assert has_blocking_or_negative_verdict(_P2_ONLY_CHANGES_REQUESTED) is True
    assert has_blocking_finding_or_label(_P2_ONLY_CHANGES_REQUESTED) is False
    assert has_blocking_finding_or_label(_NO_FINDING_CHANGES_REQUESTED) is False
    # ...but a real [P1] finding and a populated Blocker label still trigger.
    assert has_blocking_finding_or_label(_P1_CHANGES_REQUESTED) is True
    assert has_blocking_finding_or_label(_BLOCKER_LABEL_CHANGES_REQUESTED) is True


# --------------------------------------------------------------------------- #
# Regression: populated Blocker label whose finding starts with bare "no"
# (security phrasing) must STILL block. openai #8574 [P1] merge-gate bypass.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("body", _BLOCKER_LABEL_NO_FINDING_SECURITY_PHRASINGS)
def test_blocker_label_security_phrasing_starting_with_no_still_blocks(body):
    # "no authentication" / "no validation" / "no authorization" / "no rate
    # limiting" are REAL findings, not no-finding markers. They must block.
    assert has_blocking_finding_or_label(body) is True
    # The full-body changes_requested form blocks too.
    assert has_blocking_finding_or_label(_BLOCKER_LABEL_NO_AUTH_CHANGES_REQUESTED) is True


@pytest.mark.parametrize("body", _BLOCKER_LABEL_GENUINE_NO_FINDING)
def test_blocker_label_genuine_no_finding_stays_advisory(body):
    # The legit no-finding case still works: "none"/"no issues"/"no blockers"/
    # "no blocking findings"/"no concerns" are non-blocking.
    assert has_blocking_finding_or_label(body) is False


def test_p1_none_head_still_no_finding_after_fix():
    # The pre-existing "[P1] None:" no-finding behavior is unchanged.
    assert has_blocking_finding_or_label("[P1] None: defense-in-depth is solid") is False
    assert has_blocking_finding_or_label("[P1] N/A") is False


class TestFlagOnBlockerLabelSecurityBypass:
    """Flag ON: a security-phrased populated Blocker label promotes a BLOCKING
    dissent (not advisory). Regression for openai #8574 [P1]."""

    def test_no_auth_blocker_label_still_blocks_dissent(self, monkeypatch):
        monkeypatch.setenv(_FLAG, "1")
        advisory: list[dict] = []
        dissent = _dissenting_views_from_comments(
            [_comment(_BLOCKER_LABEL_NO_AUTH_CHANGES_REQUESTED)],
            head_sha=_HEAD,
            advisory_views=advisory,
        )
        assert len(dissent) == 1
        assert dissent[0]["position"] == "changes_requested"
        assert advisory == []
        assert _evidence(_BLOCKER_LABEL_NO_AUTH_CHANGES_REQUESTED).dissenting is True

    @pytest.mark.parametrize("value", _BLOCKER_LABEL_NO_FINDING_SECURITY_PHRASINGS)
    def test_security_phrasings_are_blocking_dissent(self, monkeypatch, value):
        monkeypatch.setenv(_FLAG, "1")
        body = (
            "## Claude independent model review\n"
            f"Current head: {_HEAD}\n"
            f"Verdict: CHANGES-REQUESTED\n{value}"
        )
        assert _evidence(body).dissenting is True


# --------------------------------------------------------------------------- #
# Flag OFF (default / strict) — pins TODAY's behavior
# --------------------------------------------------------------------------- #


class TestFlagOffStrict:
    def test_p2_only_changes_requested_still_blocks(self, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        dissent = _dissenting_views_from_comments(
            [_comment(_P2_ONLY_CHANGES_REQUESTED)], head_sha=_HEAD
        )
        assert len(dissent) == 1
        assert dissent[0]["position"] == "changes_requested"
        # quorum_evidence half agrees: [P2]-only changes_requested is dissenting.
        assert _evidence(_P2_ONLY_CHANGES_REQUESTED).dissenting is True

    def test_p1_changes_requested_blocks(self, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        assert (
            len(_dissenting_views_from_comments([_comment(_P1_CHANGES_REQUESTED)], head_sha=_HEAD))
            == 1
        )
        assert _evidence(_P1_CHANGES_REQUESTED).dissenting is True

    def test_pass_counts(self, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        item = EvidenceItem(
            family="claude", body="Verdict: approve.", would_count=True, verdict="pass"
        )
        assert item.dissenting is False
        assert item.supportive is True

    def test_integration_p2_only_blocks_quorum(self, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        pr = _grounded_pr([*_supporting_comments(), _comment(_P2_ONLY_CHANGES_REQUESTED)])
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        # The [P2]-only dissent blocks (flag OFF): unresolved_dissent is set and no
        # advisory downgrade occurs. (``status`` may report a different incomplete
        # cause first when this synthetic packet's quorum is itself unsatisfied; the
        # gate-governing signal is ``unresolved_dissent``.)
        assert quorum["unresolved_dissent"] is True
        assert "unresolved model dissent is present" in quorum["reasons"]
        assert quorum["advisory_views"] == []


# --------------------------------------------------------------------------- #
# Flag ON — severity-gated
# --------------------------------------------------------------------------- #


class TestFlagOnSeverityGated:
    def test_p2_only_changes_requested_is_advisory_not_blocking(self, monkeypatch):
        monkeypatch.setenv(_FLAG, "1")
        advisory: list[dict] = []
        dissent = _dissenting_views_from_comments(
            [_comment(_P2_ONLY_CHANGES_REQUESTED)], head_sha=_HEAD, advisory_views=advisory
        )
        # Non-blocking: it does NOT promote a dissent...
        assert dissent == []
        # ...but it is recorded (still visible / audited), non-counting.
        assert len(advisory) == 1
        assert advisory[0]["position"] == "advisory_changes_requested"
        assert advisory[0]["blocking"] is False
        assert advisory[0]["highest_severity"] is None
        # quorum_evidence half agrees: [P2]-only changes_requested is NOT dissenting
        # and (supportive unchanged) NOT supportive -> non-blocking AND non-counting.
        item = _evidence(_P2_ONLY_CHANGES_REQUESTED)
        assert item.dissenting is False
        assert item.supportive is False

    def test_no_finding_changes_requested_is_advisory(self, monkeypatch):
        monkeypatch.setenv(_FLAG, "1")
        advisory: list[dict] = []
        dissent = _dissenting_views_from_comments(
            [_comment(_NO_FINDING_CHANGES_REQUESTED)], head_sha=_HEAD, advisory_views=advisory
        )
        assert dissent == []
        assert len(advisory) == 1
        assert _evidence(_NO_FINDING_CHANGES_REQUESTED).dissenting is False

    def test_p1_finding_still_blocks(self, monkeypatch):
        monkeypatch.setenv(_FLAG, "1")
        advisory: list[dict] = []
        dissent = _dissenting_views_from_comments(
            [_comment(_P1_CHANGES_REQUESTED)], head_sha=_HEAD, advisory_views=advisory
        )
        assert len(dissent) == 1
        assert advisory == []
        assert _evidence(_P1_CHANGES_REQUESTED).dissenting is True

    def test_blocker_label_still_blocks(self, monkeypatch):
        monkeypatch.setenv(_FLAG, "1")
        dissent = _dissenting_views_from_comments(
            [_comment(_BLOCKER_LABEL_CHANGES_REQUESTED)], head_sha=_HEAD
        )
        assert len(dissent) == 1
        assert _evidence(_BLOCKER_LABEL_CHANGES_REQUESTED).dissenting is True

    def test_pass_still_counts(self, monkeypatch):
        monkeypatch.setenv(_FLAG, "1")
        item = EvidenceItem(
            family="claude", body="Verdict: approve.", would_count=True, verdict="pass"
        )
        assert item.dissenting is False
        assert item.supportive is True

    def test_integration_p2_only_does_not_block_clean_packet(self, monkeypatch):
        monkeypatch.setenv(_FLAG, "1")
        pr = _grounded_pr([*_supporting_comments(), _comment(_P2_ONLY_CHANGES_REQUESTED)])
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        # The [P2]-only dissent no longer blocks an otherwise-clean packet...
        assert quorum["unresolved_dissent"] is False
        assert quorum["status"] != "unresolved_dissent"
        # ...and it remains visible as a non-blocking advisory note.
        assert len(quorum["advisory_views"]) == 1
        assert any(
            "advisory finding" in reason and "not blocking" in reason
            for reason in quorum["reasons"]
        )

    def test_integration_p1_still_blocks_quorum(self, monkeypatch):
        monkeypatch.setenv(_FLAG, "1")
        pr = _grounded_pr([*_supporting_comments(), _comment(_P1_CHANGES_REQUESTED)])
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["unresolved_dissent"] is True
        assert "unresolved model dissent is present" in quorum["reasons"]


# --------------------------------------------------------------------------- #
# Lockstep invariant — both halves agree for the same body under each flag state
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "body, off_blocks, on_blocks",
    [
        (_P2_ONLY_CHANGES_REQUESTED, True, False),
        (_NO_FINDING_CHANGES_REQUESTED, True, False),
        (_P1_CHANGES_REQUESTED, True, True),
        (_BLOCKER_LABEL_CHANGES_REQUESTED, True, True),
    ],
)
def test_both_gate_halves_agree(monkeypatch, body, off_blocks, on_blocks):
    # Flag OFF
    monkeypatch.delenv(_FLAG, raising=False)
    rq_off = bool(_dissenting_views_from_comments([_comment(body)], head_sha=_HEAD))
    qe_off = _evidence(body).dissenting
    assert rq_off == qe_off == off_blocks

    # Flag ON
    monkeypatch.setenv(_FLAG, "1")
    rq_on = bool(_dissenting_views_from_comments([_comment(body)], head_sha=_HEAD))
    qe_on = _evidence(body).dissenting
    assert rq_on == qe_on == on_blocks
