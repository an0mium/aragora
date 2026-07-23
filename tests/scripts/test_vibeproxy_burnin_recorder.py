from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aragora.agents.transports.claude_vibeproxy import ClaudeVibeProxyAttempt
from aragora.agents.transports.vibeproxy import ResolvedModelRoute, TransportMode
from scripts import vibeproxy_burnin_recorder as cli


class _Policy:
    mode = TransportMode.REQUIRED

    def resolve(
        self,
        provider: str,
        model: str,
        capabilities: tuple[str, ...],
    ) -> ResolvedModelRoute:
        assert provider == "anthropic"
        assert capabilities == ("chat",)
        return ResolvedModelRoute(
            provider="anthropic",
            requested_model=model,
            resolved_model="anthropic/claude-opus-4.8",
            transport="vibeproxy",
            base_url="http://127.0.0.1:8318/v1",
            capabilities=frozenset({"chat"}),
        )


def _pr(head: str = "a" * 40) -> dict[str, Any]:
    return {
        "number": 9483,
        "state": "OPEN",
        "isDraft": True,
        "title": "VibeProxy transport",
        "body": "Part of #9409",
        "headRefOid": head,
        "url": "https://github.com/synaptent/aragora/pull/9483",
        "files": [{"path": "aragora/agents/transports/claude_vibeproxy.py"}],
    }


def test_shadow_review_is_non_countable_and_never_posts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen_prompts: list[str] = []
    monkeypatch.setattr(cli, "_load_pr", lambda _repo, _pr_number: _pr())
    monkeypatch.setattr(cli, "_load_diff", lambda _repo, _pr_number: "diff --git a/a b/a")
    monkeypatch.setattr(cli, "_required_policy", lambda: _Policy())

    def fake_run(prompt: str, **_kwargs: object) -> ClaudeVibeProxyAttempt:
        seen_prompts.append(prompt)
        return ClaudeVibeProxyAttempt(
            attempted=True,
            required=True,
            ok=True,
            text="No blocking findings.\nVerdict: PASS",
            response_model="anthropic/claude-opus-4.8",
            harness="local VibeProxy",
            timeout_seconds=10,
        )

    monkeypatch.setattr(cli, "run_claude_vibeproxy", fake_run)
    records = tmp_path / "calls.jsonl"
    proof = tmp_path / "latest.json"

    code = cli.main(
        [
            "--json",
            "--records",
            str(records),
            "--proof",
            str(proof),
            "shadow-review",
            "--pr",
            "9483",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["ok"] is True
    assert output["comment_posted"] is False
    assert output["evidence_composed"] is False
    assert output["review_body_persisted"] is False
    assert output["record"]["countable"] is False
    assert output["record"]["shadow_review"]["verdict"] == "PASS"
    assert output["record"]["shadow_review"]["posted"] is False
    assert output["record"]["alias_disclosure"]["source"] == ("ARAGORA_VIBEPROXY_MODEL_MAP")
    assert "NON-COUNTABLE" in seen_prompts[0]
    assert "Exact head: " + "a" * 40 in seen_prompts[0]
    persisted = records.read_text(encoding="utf-8")
    assert "No blocking findings" not in persisted
    assert "gh pr comment" not in Path(cli.__file__).read_text(encoding="utf-8")


def test_shadow_head_drift_is_recorded_but_not_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter([_pr(), _pr(), _pr("c" * 40)])
    monkeypatch.setattr(cli, "_load_pr", lambda _repo, _pr_number: next(states))
    monkeypatch.setattr(cli, "_load_diff", lambda _repo, _pr_number: "diff --git a/a b/a")
    monkeypatch.setattr(cli, "_required_policy", lambda: _Policy())
    monkeypatch.setattr(
        cli,
        "run_claude_vibeproxy",
        lambda *_args, **_kwargs: ClaudeVibeProxyAttempt(
            attempted=True,
            required=True,
            ok=True,
            text="Verdict: PASS",
            response_model="anthropic/claude-opus-4.8",
        ),
    )

    code, result = cli.run_shadow_review(
        repo="synaptent/aragora",
        pr_number=9483,
        model="claude-opus-4-8",
        reviewer_timeout=10,
        max_diff_chars=10_000,
        records_path=tmp_path / "calls.jsonl",
        proof_path=tmp_path / "latest.json",
    )

    assert code == 1
    assert result["record"]["ok"] is True
    assert result["record"]["clean"] is False
    assert result["record"]["error_class"] == "head_drift"
    assert result["record"]["shadow_review"]["head_stable"] is False
    assert result["proof"]["gates"]["shadow_reviews"]["observed"] == 0


def test_shadow_review_without_observed_response_model_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_load_pr", lambda _repo, _pr_number: _pr())
    monkeypatch.setattr(cli, "_load_diff", lambda _repo, _pr_number: "diff --git a/a b/a")
    monkeypatch.setattr(cli, "_required_policy", lambda: _Policy())
    monkeypatch.setattr(
        cli,
        "run_claude_vibeproxy",
        lambda *_args, **_kwargs: ClaudeVibeProxyAttempt(
            attempted=True,
            required=True,
            ok=True,
            text="Verdict: PASS",
        ),
    )

    code, result = cli.run_shadow_review(
        repo="synaptent/aragora",
        pr_number=9483,
        model="claude-opus-4-8",
        reviewer_timeout=10,
        max_diff_chars=10_000,
        records_path=tmp_path / "calls.jsonl",
        proof_path=tmp_path / "latest.json",
    )

    assert code == 1
    assert result["record"]["response_model"] is None
    assert result["record"]["error_class"] == "missing_response_model"
    assert result["record"]["family_identity_ok"] is False
    assert "missing_response_model" in result["record"]["identity_errors"]
    assert result["proof"]["gates"]["shadow_reviews"]["observed"] == 0


def test_transport_failure_records_stable_error_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_load_pr", lambda _repo, _pr_number: _pr())
    monkeypatch.setattr(cli, "_load_diff", lambda _repo, _pr_number: "diff --git a/a b/a")
    monkeypatch.setattr(cli, "_required_policy", lambda: _Policy())
    monkeypatch.setattr(
        cli,
        "run_claude_vibeproxy",
        lambda *_args, **_kwargs: ClaudeVibeProxyAttempt(
            attempted=True,
            required=True,
            ok=False,
            error="VibeProxy request timed out",
        ),
    )

    code, result = cli.run_shadow_review(
        repo="synaptent/aragora",
        pr_number=9483,
        model="claude-opus-4-8",
        reviewer_timeout=10,
        max_diff_chars=10_000,
        records_path=tmp_path / "calls.jsonl",
        proof_path=tmp_path / "latest.json",
    )

    assert code == 1
    assert result["record"]["error_class"] == "timeout"
    assert result["record"]["response_model"] is None
    assert result["record"]["clean"] is False


def test_summarize_emits_empty_not_ready_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    proof = tmp_path / "latest.json"

    code = cli.main(
        [
            "--json",
            "--records",
            str(tmp_path / "missing.jsonl"),
            "--proof",
            str(proof),
            "summarize",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert code == 0
    assert result["schema_version"] == "aragora.vibeproxy-burnin-proof.v1"
    assert result["ready"] is False
    assert result["total_records"] == 0
    assert json.loads(proof.read_text(encoding="utf-8")) == result
