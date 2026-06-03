from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.errors import CLISubprocessError
from aragora.swarm.review_routing import (
    ReviewCandidate,
    ReviewRoutingError,
    generate_review_response,
    preflight_review_candidate,
    resolve_review_candidates,
)


def _make_repo_with_profile_script(root: Path) -> Path:
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "claude_profile.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return root


def _write_pool_health(
    root: Path, profiles: list[dict[str, str]], *, age_seconds: float = 0.0
) -> None:
    generated_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    payload = {
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "profiles": profiles,
    }
    health_path = root / ".aragora" / "claude_pool_health.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_review_candidates_skips_worker_family_and_expands_claude_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_REVIEW_PROVIDER_ORDER", "codex,claude,openrouter")
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROFILES", "max-01,max-02")

    candidates = resolve_review_candidates(
        worker_model="codex",
        preferred_review_model="claude",
    )

    assert [candidate.label for candidate in candidates] == [
        "claude:max-01",
        "claude:max-02",
        "openrouter",
    ]


@pytest.mark.parametrize(
    ("preferred_review_model", "expected_label"),
    [
        ("gemini", "gemini"),
        ("gemini-cli", "gemini"),
        ("grok", "grok"),
        ("grok-cli", "grok"),
    ],
)
def test_resolve_review_candidates_honors_requested_direct_provider_family(
    monkeypatch: pytest.MonkeyPatch,
    preferred_review_model: str,
    expected_label: str,
) -> None:
    monkeypatch.setenv("ARAGORA_REVIEW_PROVIDER_ORDER", "codex,claude,openrouter")
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROFILES", "max-01")

    candidates = resolve_review_candidates(
        worker_model="codex",
        preferred_review_model=preferred_review_model,
    )

    assert [candidate.label for candidate in candidates] == [
        expected_label,
        "claude:max-01",
        "openrouter",
    ]


def test_preflight_review_candidate_accepts_direct_provider_key() -> None:
    def fake_secret_presence(name: str):
        source = "env" if name == "GOOGLE_API_KEY" else "none"
        return SimpleNamespace(source=source)

    with patch(
        "aragora.swarm.review_routing.get_secret_presence", side_effect=fake_secret_presence
    ):
        result = preflight_review_candidate(
            ReviewCandidate(provider="gemini", label="gemini"),
            repo_root=Path("/tmp/repo"),
        )

    assert result == {"ok": True, "detail": "gemini API key is configured"}


def test_preflight_review_candidate_blocks_missing_direct_provider_key() -> None:
    with patch(
        "aragora.swarm.review_routing.get_secret_presence",
        return_value=SimpleNamespace(source="none"),
    ):
        result = preflight_review_candidate(
            ReviewCandidate(provider="grok", label="grok"),
            repo_root=Path("/tmp/repo"),
        )

    assert result == {
        "ok": False,
        "detail": "XAI_API_KEY or GROK_API_KEY is not configured",
    }


@pytest.mark.asyncio
async def test_generate_review_response_fails_over_to_next_candidate() -> None:
    with (
        patch(
            "aragora.swarm.review_routing.resolve_review_candidates",
            return_value=[
                ReviewCandidate(provider="codex", label="codex"),
                ReviewCandidate(provider="claude", label="claude:max-01", profile="max-01"),
            ],
        ),
        patch(
            "aragora.swarm.review_routing.preflight_review_candidate",
            side_effect=[
                {"ok": True, "detail": "codex available"},
                {"ok": True, "detail": "claude available"},
            ],
        ),
        patch(
            "aragora.swarm.review_routing._run_review_candidate",
            new=AsyncMock(
                side_effect=[
                    CLISubprocessError(
                        message="codex failed",
                        agent_name="codex",
                        returncode=1,
                        stderr="cli error",
                    ),
                    '{"status":"passed","findings":[]}',
                ]
            ),
        ),
    ):
        result = await generate_review_response(
            "review this",
            worker_model="gemini-cli",
            preferred_review_model="codex",
            repo_root=Path("/tmp/repo"),
        )

    assert result["candidate"]["label"] == "claude:max-01"
    assert result["attempts"][0]["candidate"] == "codex"
    assert result["attempts"][0]["kind"] == "cli_failure"
    assert result["attempts"][1]["candidate"] == "claude:max-01"


@pytest.mark.asyncio
async def test_generate_review_response_runs_requested_direct_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_REVIEW_PROVIDER_ORDER", "codex,claude,openrouter")

    agent = AsyncMock()
    agent.generate = AsyncMock(return_value='{"status":"passed","findings":[]}')

    with (
        patch(
            "aragora.swarm.review_routing.preflight_review_candidate",
            return_value={"ok": True, "detail": "gemini available"},
        ),
        patch("aragora.swarm.review_routing.create_agent", return_value=agent) as create_agent,
    ):
        result = await generate_review_response(
            "review this",
            worker_model="codex",
            preferred_review_model="gemini",
            repo_root=Path("/tmp/repo"),
        )

    assert result["candidate"]["label"] == "gemini"
    assert result["attempts"] == [
        {
            "candidate": "gemini",
            "stage": "generate",
            "detail": "ok",
        }
    ]
    create_agent.assert_called_once_with(
        "gemini",
        name="campaign-review",
        role="critic",
        enable_fallback=False,
    )
    agent.generate.assert_awaited_once_with("review this")


@pytest.mark.asyncio
async def test_generate_review_response_candidate_blocker_stops_before_generation() -> None:
    run_candidate = AsyncMock(side_effect=AssertionError("candidate generation should not run"))
    blocker = {
        "title": "Requested reviewer routed to Codex",
        "body": "requested non-Codex reviewer selected Codex",
        "priority": "P1",
    }

    with (
        patch(
            "aragora.swarm.review_routing.resolve_review_candidates",
            return_value=[ReviewCandidate(provider="codex", label="codex")],
        ),
        patch(
            "aragora.swarm.review_routing.preflight_review_candidate",
            return_value={"ok": True, "detail": "codex available"},
        ),
        patch("aragora.swarm.review_routing._run_review_candidate", new=run_candidate),
    ):
        result = await generate_review_response(
            "review this",
            worker_model="claude",
            preferred_review_model="grok",
            repo_root=Path("/tmp/repo"),
            candidate_blocker=lambda candidate: blocker if candidate.provider == "codex" else None,
        )

    assert result["candidate"]["label"] == "codex"
    assert result["response"] == ""
    assert result["blocked"] == blocker
    assert result["attempts"] == [
        {
            "candidate": "codex",
            "stage": "route_guard",
            "kind": "blocked_nonreviewable",
            "detail": "Requested reviewer routed to Codex",
        }
    ]
    run_candidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_review_response_records_unexpected_exception_detail() -> None:
    with (
        patch(
            "aragora.swarm.review_routing.resolve_review_candidates",
            return_value=[
                ReviewCandidate(provider="codex", label="codex"),
                ReviewCandidate(provider="openrouter", label="openrouter"),
            ],
        ),
        patch(
            "aragora.swarm.review_routing.preflight_review_candidate",
            side_effect=[
                {"ok": True, "detail": "codex available"},
                {"ok": True, "detail": "openrouter available"},
            ],
        ),
        patch(
            "aragora.swarm.review_routing._run_review_candidate",
            new=AsyncMock(
                side_effect=[
                    RuntimeError("backend misconfigured"),
                    '{"status":"passed","findings":[]}',
                ]
            ),
        ),
    ):
        result = await generate_review_response(
            "review this",
            worker_model="claude",
            preferred_review_model="codex",
            repo_root=Path("/tmp/repo"),
        )

    assert result["candidate"]["label"] == "openrouter"
    assert result["attempts"][0] == {
        "candidate": "codex",
        "stage": "generate",
        "kind": "RuntimeError",
        "detail": "RuntimeError: backend misconfigured",
    }


@pytest.mark.asyncio
async def test_generate_review_response_raises_with_attempt_history() -> None:
    with (
        patch(
            "aragora.swarm.review_routing.resolve_review_candidates",
            return_value=[
                ReviewCandidate(provider="codex", label="codex"),
                ReviewCandidate(provider="openrouter", label="openrouter"),
            ],
        ),
        patch(
            "aragora.swarm.review_routing.preflight_review_candidate",
            side_effect=[
                {"ok": False, "detail": "codex CLI not found"},
                {"ok": False, "detail": "OpenRouter TLS check failed"},
            ],
        ),
    ):
        with pytest.raises(ReviewRoutingError) as exc_info:
            await generate_review_response(
                "review this",
                worker_model="claude",
                preferred_review_model="codex",
                repo_root=Path("/tmp/repo"),
            )

    assert str(exc_info.value) == "No configured review candidate succeeded. Check logs for detail."
    assert exc_info.value.attempts == [
        {
            "candidate": "codex",
            "stage": "preflight",
            "detail": "codex CLI not found",
        },
        {
            "candidate": "openrouter",
            "stage": "preflight",
            "detail": "OpenRouter TLS check failed",
        },
    ]


def test_resolve_review_candidates_includes_max_13_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARAGORA_CLAUDE_REVIEW_PROFILES", raising=False)
    monkeypatch.setenv("ARAGORA_REVIEW_PROVIDER_ORDER", "codex,claude")

    candidates = resolve_review_candidates(
        worker_model="codex",
        preferred_review_model="claude",
    )
    claude_profiles = [c.profile for c in candidates if c.provider == "claude"]

    assert claude_profiles == [f"max-{i:02d}" for i in range(1, 14)]
    assert "max-13" in claude_profiles


def test_claude_preflight_skips_expired_profile_from_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROBE", "snapshot")
    repo = _make_repo_with_profile_script(tmp_path)
    _write_pool_health(repo, [{"name": "max-01", "email": "a@b.c", "state": "expired"}])

    with patch("aragora.swarm.review_routing.shutil.which", return_value="/usr/bin/claude"):
        result = preflight_review_candidate(
            ReviewCandidate(provider="claude", label="claude:max-01", profile="max-01"),
            repo_root=repo,
        )

    assert result["ok"] is False
    assert result["kind"] == "claude_unauthenticated"
    assert "login" in result["detail"]


def test_claude_preflight_trusts_ok_snapshot_without_status_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROBE", "snapshot")
    repo = _make_repo_with_profile_script(tmp_path)
    _write_pool_health(repo, [{"name": "max-02", "email": "a@b.c", "state": "ok"}])

    run_spy = MagicMock(
        side_effect=AssertionError("status subprocess must not run for ok snapshot")
    )
    with (
        patch("aragora.swarm.review_routing.shutil.which", return_value="/usr/bin/claude"),
        patch("aragora.swarm.review_routing.subprocess.run", run_spy),
    ):
        result = preflight_review_candidate(
            ReviewCandidate(provider="claude", label="claude:max-02", profile="max-02"),
            repo_root=repo,
        )

    assert result == {"ok": True, "detail": "claude:max-02 verified (snapshot)"}
    run_spy.assert_not_called()


def test_claude_preflight_falls_back_to_status_when_snapshot_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROBE", "snapshot")
    monkeypatch.setenv("ARAGORA_CLAUDE_POOL_HEALTH_TTL", "3600")
    repo = _make_repo_with_profile_script(tmp_path)
    _write_pool_health(
        repo,
        [{"name": "max-03", "email": "a@b.c", "state": "expired"}],
        age_seconds=7200,
    )

    with (
        patch("aragora.swarm.review_routing.shutil.which", return_value="/usr/bin/claude"),
        patch(
            "aragora.swarm.review_routing.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ),
    ):
        result = preflight_review_candidate(
            ReviewCandidate(provider="claude", label="claude:max-03", profile="max-03"),
            repo_root=repo,
        )

    assert result == {"ok": True, "detail": "claude:max-03 authenticated"}


@pytest.mark.asyncio
async def test_generate_review_response_reports_claude_pool_unauthenticated() -> None:
    with (
        patch(
            "aragora.swarm.review_routing.resolve_review_candidates",
            return_value=[
                ReviewCandidate(provider="claude", label="claude:max-01", profile="max-01"),
                ReviewCandidate(provider="claude", label="claude:max-02", profile="max-02"),
            ],
        ),
        patch(
            "aragora.swarm.review_routing.preflight_review_candidate",
            side_effect=[
                {
                    "ok": False,
                    "kind": "claude_unauthenticated",
                    "detail": "max-01 token is expired",
                },
                {
                    "ok": False,
                    "kind": "claude_unauthenticated",
                    "detail": "max-02 token is expired",
                },
            ],
        ),
    ):
        with pytest.raises(ReviewRoutingError) as exc_info:
            await generate_review_response(
                "review this",
                worker_model="codex",
                preferred_review_model="claude",
                repo_root=Path("/tmp/repo"),
            )

    assert exc_info.value.category == "claude_pool_unauthenticated"
    assert "claude_profiles_bootstrap.sh login" in str(exc_info.value)
    assert exc_info.value.attempts[0]["kind"] == "claude_unauthenticated"


@pytest.mark.asyncio
async def test_generate_review_response_marks_billing_exhaustion() -> None:
    with (
        patch(
            "aragora.swarm.review_routing.resolve_review_candidates",
            return_value=[
                ReviewCandidate(provider="claude", label="claude:max-01", profile="max-01")
            ],
        ),
        patch(
            "aragora.swarm.review_routing.preflight_review_candidate",
            return_value={"ok": True, "detail": "claude available"},
        ),
        patch(
            "aragora.swarm.review_routing._run_review_candidate",
            new=AsyncMock(
                side_effect=CLISubprocessError(
                    message="CLI command failed with return code 1",
                    agent_name="claude:max-01",
                    returncode=1,
                    stderr="Credit balance is too low",
                )
            ),
        ),
    ):
        with pytest.raises(ReviewRoutingError) as exc_info:
            await generate_review_response(
                "review this",
                worker_model="codex",
                preferred_review_model="claude",
                repo_root=Path("/tmp/repo"),
            )

    assert exc_info.value.category == "billing_exhausted"
    assert str(exc_info.value) == (
        "Reviewer capacity is exhausted. Check the active reviewer account and available credits."
    )
    assert exc_info.value.attempts == [
        {
            "candidate": "claude:max-01",
            "stage": "generate",
            "kind": "billing_exhausted",
            "detail": "Reviewer credits are exhausted.",
        }
    ]
