"""Tests for the operator-advisory-settlement relief valve.

Spec: docs/specs/OPERATOR_ADVISORY_SETTLEMENT.md (#8933 incident, PR #8939).

These pin the two new load-bearing helpers directly (the valve's family
accounting and its trusted-author marker check) plus the flag gate. The valve's
full assembly in ``_build_model_review_quorum`` is exercised through the public
``review-queue merge-packet`` path in the CLI integration suite; here we lock
the security-critical primitives so a reviewer can see the invariants hold.
"""

from __future__ import annotations

from typing import Any

from aragora.cli.commands import review_queue as rq

HEAD = "abfea376c6216173b6f4ed84306893acc9563545"
COMMITTED_AT = "2026-07-10T23:00:00Z"


def _review(family: str, *, verdict: str = "approve", body_extra: str = "") -> dict[str, Any]:
    """A grounded, non-bot, countable model-review comment for ``family``."""
    return {
        "author": {"login": "an0mium"},
        "createdAt": "2026-07-11T00:00:00Z",
        "body": (
            f"## {family} independent model review\n"
            f"**Model family:** {family}\n"
            f"Head: {HEAD}\n"
            f"Verdict: {verdict}.\n"
            f"{body_extra}"
        ),
    }


# --- validated-family accounting (condition 4) -----------------------------


class TestValidatedFamilies:
    def test_two_distinct_families_are_counted(self) -> None:
        comments = [_review("claude"), _review("openai")]
        _wf, _diss, blocking, families = rq._advisory_settle_review_signals(
            comments, head_sha=HEAD, head_committed_at=COMMITTED_AT
        )
        assert families == frozenset({"claude", "openai"})
        assert blocking is False

    def test_settlement_marker_and_plain_prose_do_not_inflate_family_count(self) -> None:
        # Family attribution uses the SAME validated pass advisory_settle uses:
        # a comment must be a RECOGNIZED model review (not raw text) to attribute
        # a family. The settlement marker comment and ordinary prose are not
        # recognized reviews, so a single genuine review cannot be padded to a
        # false quorum-of-two by non-review chatter. (Family attribution for a
        # recognized review follows advisory_settle's self-declared-header
        # standard by design; the valve's binding authorization is the
        # trusted-creator commit status, not this corroborating count.)
        prose = {
            "author": {"login": "an0mium"},
            "createdAt": "2026-07-11T00:00:00Z",
            "body": f"Looks good to me, thanks. (reviewed {HEAD})",
        }
        marker = _marker("scarmani")
        marker["createdAt"] = "2026-07-11T00:00:00Z"
        _wf, _diss, _block, families = rq._advisory_settle_review_signals(
            [_review("claude"), prose, marker], head_sha=HEAD, head_committed_at=COMMITTED_AT
        )
        assert families == frozenset({"claude"})

    def test_bot_authored_review_does_not_count_family(self) -> None:
        # Positive family attribution requires a non-bot author (the blocking
        # scan still consults bot reviews, but they never ADD a countable family).
        bot = _review("openai")
        bot["author"] = {"login": "github-actions[bot]"}
        _wf, _diss, _block, families = rq._advisory_settle_review_signals(
            [_review("claude"), bot], head_sha=HEAD, head_committed_at=COMMITTED_AT
        )
        assert families == frozenset({"claude"})

    def test_blocking_finding_is_reported(self) -> None:
        comments = [_review("claude", verdict="request changes", body_extra="- [P1] real bug")]
        _wf, _diss, blocking, _families = rq._advisory_settle_review_signals(
            comments, head_sha=HEAD, head_committed_at=COMMITTED_AT
        )
        assert blocking is True

    def test_stale_comment_excluded(self) -> None:
        stale = _review("claude")
        stale["createdAt"] = "2020-01-01T00:00:00Z"
        stale["body"] = stale["body"].replace(f"Head: {HEAD}\n", "")  # no SHA cite either
        _wf, _diss, _block, families = rq._advisory_settle_review_signals(
            [stale], head_sha=HEAD, head_committed_at=COMMITTED_AT
        )
        assert families == frozenset()


# --- trusted-author marker comment (condition 6) ---------------------------


def _marker(author: str, *, head: str = HEAD, token: str = "admin_squash_merge") -> dict[str, Any]:
    return {
        "author": {"login": author},
        "body": (
            "Tier-4 Human Settlement Authorization\n\n"
            f"PR: #8939\nExact head: {head}\n"
            f"Authorized action: {token}.\n"
            "Human-risk settlement: I accept the Tier 4 risk for this PR.\n"
        ),
    }


class TestOperatorSettlementComment:
    def _pr(self, *comments: dict[str, Any]) -> dict[str, Any]:
        return {"comments": list(comments)}

    def test_trusted_author_marker_accepted(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(rq, "_trusted_settlement_creator", lambda: "scarmani")
        pr = self._pr(_marker("scarmani"))
        assert rq._has_operator_settlement_comment(pr, head_sha=HEAD) is True

    def test_untrusted_author_marker_rejected(self, monkeypatch: Any) -> None:
        # Same marker text, non-operator author — must NOT authorize.
        monkeypatch.setattr(rq, "_trusted_settlement_creator", lambda: "scarmani")
        pr = self._pr(_marker("an0mium"))
        assert rq._has_operator_settlement_comment(pr, head_sha=HEAD) is False

    def test_trusted_author_case_insensitive(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(rq, "_trusted_settlement_creator", lambda: "scarmani")
        pr = self._pr(_marker("ScarMani"))
        assert rq._has_operator_settlement_comment(pr, head_sha=HEAD) is True

    def test_wrong_head_rejected(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(rq, "_trusted_settlement_creator", lambda: "scarmani")
        pr = self._pr(_marker("scarmani", head="0" * 40))
        assert rq._has_operator_settlement_comment(pr, head_sha=HEAD) is False

    def test_missing_merge_token_rejected(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(rq, "_trusted_settlement_creator", lambda: "scarmani")
        pr = self._pr(_marker("scarmani", token="please merge"))
        assert rq._has_operator_settlement_comment(pr, head_sha=HEAD) is False

    def test_empty_head_rejected(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(rq, "_trusted_settlement_creator", lambda: "scarmani")
        pr = self._pr(_marker("scarmani"))
        assert rq._has_operator_settlement_comment(pr, head_sha="") is False


# --- flag gate (condition 1) -----------------------------------------------


class TestFlagGate:
    def test_default_off(self) -> None:
        assert rq._operator_advisory_settlement_enabled(env={}) is False

    def test_on_values(self) -> None:
        for value in ("1", "true", "yes", "on", "ON", "True"):
            assert (
                rq._operator_advisory_settlement_enabled(
                    env={"ARAGORA_ENABLE_OPERATOR_ADVISORY_SETTLEMENT": value}
                )
                is True
            )

    def test_off_values(self) -> None:
        for value in ("0", "false", "no", "", "off"):
            assert (
                rq._operator_advisory_settlement_enabled(
                    env={"ARAGORA_ENABLE_OPERATOR_ADVISORY_SETTLEMENT": value}
                )
                is False
            )
