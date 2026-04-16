from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aragora.agents.errors import CLISubprocessError
from aragora.swarm.review_routing import (
    ReviewCandidate,
    ReviewRoutingError,
    generate_review_response,
    resolve_review_candidates,
)


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
