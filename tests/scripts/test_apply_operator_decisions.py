"""Tests for ``scripts/apply_operator_decisions.py``.

Fixture-driven; never invokes real ``gh``. The script under test
opens ``subprocess.run`` via the standard module reference, so
``monkeypatch.setattr(aod.subprocess, "run", ...)`` is sufficient
to intercept every call. ``shutil.which`` is patched separately so
the CLI does not bail on a missing ``gh`` binary in CI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    """Import the script as a module without polluting ``sys.path``."""

    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "apply_operator_decisions.py"
    spec = importlib.util.spec_from_file_location(
        "apply_operator_decisions_under_test", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aod = _load_module()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_HEAD_SHA_DEFAULT = "a" * 40  # canonical match for FakeGh's default
_DRIFTED_HEAD = "b" * 40
_RECEIPT_BODY = b'{"pr_url":"https://github.com/synaptent/aragora/pull/100"}\n'
_RECEIPT_SHA = hashlib.sha256(_RECEIPT_BODY).hexdigest()


def make_entry(
    pr_number: int,
    decision: str | None,
    *,
    head_sha: str = _HEAD_SHA_DEFAULT,
    comment: str = "",
) -> dict[str, Any]:
    return {
        "pr_number": pr_number,
        "head_sha": head_sha,
        "tier": "2",
        "decision": decision,
        "comment": comment,
        "first_focused_at_utc": "2026-05-17T17:00:00.000Z",
        "decided_at_utc": "2026-05-17T17:00:05.000Z",
        "decision_seconds": 5.0,
    }


def canonical_payload(decisions: list[Any], **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "aragora-operator-decisions/1.0",
        "generated_at_utc": "2026-05-17T17:00:00.000Z",
        "receipt_id_hint": "open-queue-settlement-test",
        "receipt_repo": "synaptent/aragora",
        "receipt_sha256": _RECEIPT_SHA,
        "receipt_sha256_verified": True,
        "decisions": decisions,
    }
    base.update(overrides)
    canonical = aod.canonical_json(base)
    base["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return base


def write_payload(
    tmp_path: Path,
    decisions: list[Any],
    **overrides: Any,
) -> Path:
    payload = canonical_payload(decisions, **overrides)
    p = tmp_path / "operator-decisions.json"
    p.write_text(json.dumps(payload))
    return p


def write_receipt(tmp_path: Path, body: bytes = _RECEIPT_BODY) -> Path:
    p = tmp_path / "settlement-receipt.json"
    p.write_bytes(body)
    return p


def apply_args(tmp_path: Path, decisions_path: Path, *extra: str) -> list[str]:
    return [
        str(decisions_path),
        "--apply",
        "--receipt-path",
        str(write_receipt(tmp_path)),
        *extra,
    ]


class FakeGh:
    """Records every subprocess.run call; returns canned results."""

    def __init__(
        self,
        *,
        head_oid: str = _HEAD_SHA_DEFAULT,
        apply_succeeds: bool = True,
    ) -> None:
        self.head_oid = head_oid
        self.apply_succeeds = apply_succeeds
        self.calls: list[list[str]] = []

    def run(self, cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"headRefOid": self.head_oid}),
                stderr="",
            )
        if not self.apply_succeeds:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, stderr="gh failed (simulated)"
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    def review_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[:3] == ["gh", "pr", "review"]]

    def close_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[:3] == ["gh", "pr", "close"]]


@pytest.fixture
def fake_gh(monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    fake = FakeGh()
    monkeypatch.setattr(aod.subprocess, "run", fake.run)
    monkeypatch.setattr(
        aod.shutil,
        "which",
        lambda exe: "/usr/local/bin/gh" if exe == "gh" else None,
    )
    return fake


# ---------------------------------------------------------------------------
# Signature, parsing, CLI plumbing
# ---------------------------------------------------------------------------


def test_sig_mismatch_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = canonical_payload([make_entry(100, "approve_tier")])
    payload["payload_sha256"] = "0" * 64
    p = tmp_path / "decisions.json"
    p.write_text(json.dumps(payload))

    rc = aod.main([str(p)])

    assert rc == 2
    assert fake_gh.calls == []
    assert "payload_sha256 mismatch" in capsys.readouterr().err


def test_missing_file_returns_2(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = aod.main([str(tmp_path / "does-not-exist.json")])
    assert rc == 2
    assert "file not found" in capsys.readouterr().err


def test_invalid_json_returns_2(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "garbage.json"
    p.write_text("not valid json {{{")
    rc = aod.main([str(p)])
    assert rc == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_no_gh_on_path_returns_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(aod.shutil, "which", lambda _exe: None)
    p = write_payload(tmp_path, [make_entry(100, "approve_tier")])
    rc = aod.main([str(p), "--apply"])
    assert rc == 2
    assert "gh` CLI not found" in capsys.readouterr().err


def test_unsupported_schema_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(
        tmp_path,
        [make_entry(100, "approve_tier")],
        schema_version="aragora-operator-decisions/0.9",
    )

    rc = aod.main([str(p)])

    assert rc == 2
    assert fake_gh.calls == []
    assert "unsupported schema_version" in capsys.readouterr().err


def test_unverified_receipt_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(
        tmp_path,
        [make_entry(100, "approve_tier")],
        receipt_sha256_verified=False,
    )

    rc = aod.main([str(p)])

    assert rc == 2
    assert fake_gh.calls == []
    assert "receipt_sha256_verified must be true" in capsys.readouterr().err


def test_string_receipt_verification_flag_fails_closed_before_any_gh_call(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(
        tmp_path,
        [make_entry(100, "approve_tier")],
        receipt_sha256_verified="true",
    )

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "receipt_sha256_verified must be a boolean" in capsys.readouterr().err


def test_invalid_receipt_sha_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(
        tmp_path,
        [make_entry(100, "approve_tier")],
        receipt_sha256="not-a-sha",
    )

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "receipt_sha256 must be a lowercase 64-character SHA-256" in capsys.readouterr().err


def test_invalid_receipt_repo_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(
        tmp_path,
        [make_entry(100, "approve_tier")],
        receipt_repo="synaptent/aragora --repo attacker/other",
    )

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "receipt_repo must be a GitHub repository" in capsys.readouterr().err


def test_malformed_decision_row_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier"), "not-an-object"])

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "decisions[1] must be a JSON object" in capsys.readouterr().err


def test_invalid_pr_number_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier"), {"pr_number": "nope"}])

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "pr_number must be a positive integer" in capsys.readouterr().err


def test_string_pr_number_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = make_entry(100, "approve_tier")
    entry["pr_number"] = "100"
    p = write_payload(tmp_path, [entry])

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "pr_number must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("head_sha", 123, "head_sha must be a non-empty string"),
        ("comment", {"not": "text"}, "comment must be a string"),
        ("tier", 2, "tier must be a string or null"),
        ("first_focused_at_utc", 7, "first_focused_at_utc must be a string or null"),
        ("decided_at_utc", [], "decided_at_utc must be a string or null"),
        ("decision_seconds", "5.0", "decision_seconds must be a number or null"),
    ],
)
def test_type_invalid_decision_row_returns_2_and_makes_no_gh_calls(
    tmp_path: Path,
    fake_gh: FakeGh,
    capsys: pytest.CaptureFixture[str],
    field: str,
    bad_value: Any,
    message: str,
) -> None:
    entry = make_entry(100, "approve_tier")
    entry[field] = bad_value
    p = write_payload(tmp_path, [entry])

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Default dry-run
# ---------------------------------------------------------------------------


def test_dry_run_default_does_not_call_gh(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", comment="lgtm")])
    rc = aod.main([str(p)])
    assert rc == 0
    assert fake_gh.calls == []
    out = capsys.readouterr().out
    assert "WOULD APPLY" in out
    assert "DRY RUN" in out


def test_apply_requires_receipt_path_before_any_gh_call(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", comment="lgtm")])

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "--apply requires --receipt-path" in capsys.readouterr().err


def test_apply_receipt_mismatch_returns_2_before_any_gh_call(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", comment="lgtm")])
    receipt_path = write_receipt(tmp_path, b'{"receipt":"different"}\n')

    rc = aod.main([str(p), "--apply", "--receipt-path", str(receipt_path)])

    assert rc == 2
    assert fake_gh.calls == []
    assert "receipt_sha256 mismatch" in capsys.readouterr().err


def test_apply_receipt_repo_mismatch_returns_2_before_any_gh_call(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(
        tmp_path,
        [make_entry(100, "approve_tier", comment="lgtm")],
        receipt_repo="attacker/repo",
    )

    rc = aod.main(apply_args(tmp_path, p))

    assert rc == 2
    assert fake_gh.calls == []
    assert "receipt_repo mismatch" in capsys.readouterr().err


def test_apply_receipt_without_pr_url_returns_2_before_any_gh_call(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt_body = b'{"not_pr_url":"https://github.com/synaptent/aragora/pull/100"}\n'
    p = write_payload(
        tmp_path,
        [make_entry(100, "approve_tier", comment="lgtm")],
        receipt_sha256=hashlib.sha256(receipt_body).hexdigest(),
    )
    receipt_path = write_receipt(tmp_path, receipt_body)

    rc = aod.main([str(p), "--apply", "--receipt-path", str(receipt_path)])

    assert rc == 2
    assert fake_gh.calls == []
    assert "does not contain a GitHub pr_url" in capsys.readouterr().err


def test_apply_and_dry_run_conflict_returns_2_and_makes_no_gh_calls(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", comment="lgtm")])

    rc = aod.main([str(p), "--apply", "--dry-run"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "--apply and --dry-run are mutually exclusive" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Per-decision apply paths
# ---------------------------------------------------------------------------


def test_apply_approve_tier_calls_review_approve(tmp_path: Path, fake_gh: FakeGh) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", comment="lgtm")])
    rc = aod.main(apply_args(tmp_path, p))
    assert rc == 0
    # First call is HEAD verify, second is the review.
    assert fake_gh.calls[0][:3] == ["gh", "pr", "view"]
    review = fake_gh.calls[1]
    assert review[:4] == ["gh", "pr", "review", "100"]
    assert review[4:6] == ["--repo", "synaptent/aragora"]
    assert "--approve" in review
    assert "--body" in review


def test_apply_approve_downgrade_prepends_downgraded_marker(
    tmp_path: Path, fake_gh: FakeGh
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_downgrade", comment="downgrade to T1")])
    aod.main(apply_args(tmp_path, p))
    review = fake_gh.review_calls()[0]
    body = review[review.index("--body") + 1]
    assert body.startswith("DOWNGRADED:")


def test_apply_request_changes_calls_request_changes(tmp_path: Path, fake_gh: FakeGh) -> None:
    p = write_payload(tmp_path, [make_entry(100, "request_changes", comment="please fix X")])
    aod.main(apply_args(tmp_path, p))
    review = fake_gh.review_calls()[0]
    assert review[:4] == ["gh", "pr", "review", "100"]
    assert review[4:6] == ["--repo", "synaptent/aragora"]
    assert "--request-changes" in review


def test_apply_reject_calls_close_with_comment(tmp_path: Path, fake_gh: FakeGh) -> None:
    p = write_payload(tmp_path, [make_entry(100, "reject", comment="duplicate")])
    aod.main(apply_args(tmp_path, p))
    close = fake_gh.close_calls()[0]
    assert close[:4] == ["gh", "pr", "close", "100"]
    assert close[4:6] == ["--repo", "synaptent/aragora"]
    assert "--comment" in close


def test_apply_uses_receipt_repo_for_every_gh_call(tmp_path: Path, fake_gh: FakeGh) -> None:
    receipt_body = b'{"pr_url":"https://github.com/alternate/repo/pull/100"}\n'
    p = write_payload(
        tmp_path,
        [make_entry(100, "approve_tier", comment="lgtm")],
        receipt_repo="alternate/repo",
        receipt_sha256=hashlib.sha256(receipt_body).hexdigest(),
    )
    receipt_path = write_receipt(tmp_path, receipt_body)

    rc = aod.main([str(p), "--apply", "--receipt-path", str(receipt_path)])

    assert rc == 0
    assert fake_gh.calls
    for call in fake_gh.calls:
        assert call[call.index("--repo") + 1] == "alternate/repo"


def test_apply_hold_operator_skips_without_gh_call(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "hold_operator")])
    rc = aod.main(apply_args(tmp_path, p))
    assert rc == 0
    assert fake_gh.calls == []
    assert "operator-only" in capsys.readouterr().out


def test_null_decision_skipped(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, None)])
    rc = aod.main(apply_args(tmp_path, p))
    assert rc == 0
    assert fake_gh.calls == []
    assert "no decision recorded" in capsys.readouterr().out


def test_unknown_decision_fails_closed_before_any_gh_call(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "weird_decision")])
    rc = aod.main([str(p), "--apply"])
    assert rc == 2
    assert fake_gh.calls == []
    assert "unsupported value" in capsys.readouterr().err


def test_unknown_decision_in_later_row_blocks_earlier_apply(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(
        tmp_path,
        [
            make_entry(100, "approve_tier"),
            make_entry(200, "future_decision"),
        ],
    )

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "decisions[1].decision" in capsys.readouterr().err


@pytest.mark.parametrize(
    "entry_patch",
    [
        {"decision": ""},
        {"decision": None, "_remove_decision_key": True},
    ],
)
def test_empty_or_missing_decision_fails_closed_before_any_gh_call(
    tmp_path: Path,
    fake_gh: FakeGh,
    capsys: pytest.CaptureFixture[str],
    entry_patch: dict[str, Any],
) -> None:
    first = make_entry(100, "approve_tier")
    malformed = make_entry(200, None)
    if entry_patch.pop("_remove_decision_key", False):
        malformed.pop("decision")
    else:
        malformed.update(entry_patch)
    p = write_payload(tmp_path, [first, malformed])

    rc = aod.main([str(p), "--apply"])

    assert rc == 2
    assert fake_gh.calls == []
    assert "decisions[1].decision must be a string or null" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Head-SHA drift safety
# ---------------------------------------------------------------------------


def test_head_drift_skips_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeGh(head_oid=_DRIFTED_HEAD)
    monkeypatch.setattr(aod.subprocess, "run", fake.run)
    monkeypatch.setattr(
        aod.shutil, "which", lambda exe: "/usr/local/bin/gh" if exe == "gh" else None
    )
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", head_sha=_HEAD_SHA_DEFAULT)])
    rc = aod.main(apply_args(tmp_path, p))
    assert rc == 0
    # Only the HEAD-view call; no review.
    assert len(fake.calls) == 1
    assert fake.calls[0][:3] == ["gh", "pr", "view"]
    assert fake.calls[0][4:6] == ["--repo", "synaptent/aragora"]
    out = capsys.readouterr().out
    assert "DRIFT" in out
    assert "HEAD DRIFT" in out


# ---------------------------------------------------------------------------
# Filtering / hold list
# ---------------------------------------------------------------------------


def test_only_pr_filter_touches_only_listed(tmp_path: Path, fake_gh: FakeGh) -> None:
    p = write_payload(
        tmp_path,
        [
            make_entry(100, "approve_tier"),
            make_entry(200, "approve_tier"),
            make_entry(300, "approve_tier"),
        ],
    )
    aod.main(apply_args(tmp_path, p, "--only-pr", "200"))
    reviews = fake_gh.review_calls()
    assert len(reviews) == 1
    assert reviews[0][3] == "200"
    # No HEAD-view for non-targets either.
    for n in ("100", "300"):
        assert not any(c[:4] == ["gh", "pr", "view", n] for c in fake_gh.calls)


def test_held_pr_hard_skip(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    # Pick a real held PR number from the script's hard-coded list.
    held = next(iter(aod.HELD_PR_NUMBERS))
    p = write_payload(tmp_path, [make_entry(held, "approve_tier")])
    rc = aod.main(apply_args(tmp_path, p))
    assert rc == 0
    assert fake_gh.calls == []
    out = capsys.readouterr().out
    assert "HELD" in out
    assert f"#{held}" in out


def test_held_pr_skipped_in_dry_run_too(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    held = next(iter(aod.HELD_PR_NUMBERS))
    p = write_payload(tmp_path, [make_entry(held, "reject")])
    rc = aod.main([str(p)])
    assert rc == 0
    assert fake_gh.calls == []
    assert "HELD" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# JSON output mode
# ---------------------------------------------------------------------------


def test_json_output_shape(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", comment="lgtm")])
    rc = aod.main([str(p), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["applied"] is False
    assert isinstance(data["payload_sha256"], str)
    assert isinstance(data["receipt_sha256"], str)
    assert isinstance(data["results"], list)
    first = data["results"][0]
    assert first["pr_number"] == 100
    assert first["status"] == "would-apply"
    assert first["gh_command"][:6] == [
        "gh",
        "pr",
        "review",
        "100",
        "--repo",
        "synaptent/aragora",
    ]
    assert "--approve" in first["gh_command"]


# ---------------------------------------------------------------------------
# Comment-body footer
# ---------------------------------------------------------------------------


def test_footer_carries_both_sha_prefixes(tmp_path: Path, fake_gh: FakeGh) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", comment="lgtm")])
    aod.main(apply_args(tmp_path, p))
    review = fake_gh.review_calls()[0]
    body = review[review.index("--body") + 1]
    assert "Applied from operator-decisions" in body
    assert "bound to packet" in body
    # The first ten chars of receipt_sha must appear verbatim in the footer.
    assert _RECEIPT_SHA[:10] in body


def test_footer_emits_even_when_comment_empty(tmp_path: Path, fake_gh: FakeGh) -> None:
    p = write_payload(tmp_path, [make_entry(100, "approve_tier", comment="")])
    aod.main(apply_args(tmp_path, p))
    review = fake_gh.review_calls()[0]
    body = review[review.index("--body") + 1]
    # Body still contains the binding line even with no operator comment.
    assert "Applied from operator-decisions" in body


# ---------------------------------------------------------------------------
# Failure surfacing
# ---------------------------------------------------------------------------


def test_gh_failure_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGh(apply_succeeds=False)
    monkeypatch.setattr(aod.subprocess, "run", fake.run)
    monkeypatch.setattr(
        aod.shutil, "which", lambda exe: "/usr/local/bin/gh" if exe == "gh" else None
    )
    p = write_payload(tmp_path, [make_entry(100, "approve_tier")])
    rc = aod.main(apply_args(tmp_path, p))
    # HEAD-view succeeds (its branch is independent of apply_succeeds);
    # the review call fails → status=failed → rc=1.
    assert rc == 1
