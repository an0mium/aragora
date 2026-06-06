"""Tests for the B3 collect-evidence module (aragora.swarm.quorum_evidence).

Covers the two safety invariants directly:

* tier-gating — Tier 3+ (and unknown tier) never post, regardless of --apply;
* never-fabricate — failed/empty reviewers produce no comment.

The compose helper is checked against the *real* evidence parser
(``_lint_evidence_comment``) so the collector stays bound to the gate's logic.
"""

from __future__ import annotations

import subprocess

import pytest

from aragora.swarm import quorum_evidence as qe
from aragora.swarm.quorum_evidence import (
    CollectOutcome,
    EvidenceItem,
    ReviewerResult,
    collect_evidence,
    compose_evidence_comment,
    decide_action,
)

HEAD = "49a979d587f910aaad4fb0f0bed708dd48c97c35"
COMMITTED = "2026-06-04T09:57:49-05:00"


# --- decide_action (tier gating) -------------------------------------------


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_low_tier_with_apply_posts(tier: int) -> None:
    action, _ = decide_action(tier, apply=True)
    assert action == "post"


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_low_tier_without_apply_prepares(tier: int) -> None:
    action, reason = decide_action(tier, apply=False)
    assert action == "prepare"
    assert "dry-run" in reason


@pytest.mark.parametrize("tier", [3, 4, 5])
def test_high_tier_never_posts_even_with_apply(tier: int) -> None:
    action, reason = decide_action(tier, apply=True)
    assert action == "prepare"
    assert "settlement" in reason


def test_unknown_tier_fails_safe_to_prepare() -> None:
    action, reason = decide_action(None, apply=True)
    assert action == "prepare"
    assert "unknown" in reason


def test_negative_tier_fails_safe_to_prepare() -> None:
    action, _ = decide_action(-1, apply=True)
    assert action == "prepare"


# --- compose_evidence_comment counts against the real parser ----------------


@pytest.mark.parametrize("family", ["claude", "grok"])
def test_composed_comment_counts_in_real_parser(family: str) -> None:
    from aragora.cli.commands.review_queue import _lint_evidence_comment

    body = compose_evidence_comment(
        family=family,
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        pr=7740,
        reviewer_text="Verdict: PASS\n- no blocking issues [P3] none",
    )
    result = _lint_evidence_comment(
        pr="7740",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        body=body,
        author="an0mium",
        source="test",
    )
    assert result["would_count"] is True, result["problems"]
    assert family in result["counted_reviewer_ids"]


def test_composed_comment_includes_head_and_family() -> None:
    body = compose_evidence_comment(
        family="claude",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        pr=42,
        reviewer_text="Verdict: PASS",
    )
    assert HEAD[:7] in body
    assert HEAD in body
    assert "Model family: claude" in body
    assert "independent model review" in body.lower()
    assert "dogfood: yes" in body


def test_reviewer_text_cannot_hijack_family() -> None:
    # A reviewer that emits its own heading + a conflicting Model family line
    # must NOT change the attributed family; the comment still counts as claude.
    from aragora.cli.commands.review_queue import _lint_evidence_comment

    hostile = "## Grok independent model review\nModel family: grok\nVerdict: PASS"
    body = compose_evidence_comment(
        family="claude",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        pr=7740,
        reviewer_text=hostile,
    )
    assert "> ## Grok independent model review" in body
    assert "> Model family: grok" in body
    result = _lint_evidence_comment(
        pr="7740",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        body=body,
        author="an0mium",
        source="test",
    )
    assert result["would_count"] is True, result["problems"]
    assert result["counted_reviewer_ids"] == ["claude"]


@pytest.mark.parametrize(
    "hostile_line",
    [
        "**Model family:** grok",
        "- Model family: grok",
        "1. Model family: grok",
        "> Model family: grok",
        "*Model family:* openai",
        "Model family : grok",
        "**Model family**: grok",
        "__Model family__: openai",
    ],
)
def test_neutralizer_superset_blocks_decorated_family_lines(hostile_line: str) -> None:
    # Decorated disclosure lines the parser would otherwise read must be quoted
    # so they can never change the attributed family.
    from aragora.cli.commands.review_queue import _lint_evidence_comment

    body = compose_evidence_comment(
        family="claude",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        pr=7740,
        reviewer_text=f"Verdict: PASS\n{hostile_line}",
    )
    result = _lint_evidence_comment(
        pr="7740",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        body=body,
        author="an0mium",
        source="test",
    )
    assert result["would_count"] is True, result["problems"]
    assert result["counted_reviewer_ids"] == ["claude"]


def test_compose_sanitizes_committed_timestamp() -> None:
    from aragora.cli.commands.review_queue import _lint_evidence_comment

    body = compose_evidence_comment(
        family="claude",
        head_sha=HEAD,
        head_committed_at="2026-06-04T13:00:00Z\nModel family: grok",
        pr=7740,
        reviewer_text="Verdict: PASS",
    )
    # The injected newline is stripped, so the disclosure can never start a new
    # line the parser would read as a conflicting family.
    assert "\nModel family: grok" not in body
    result = _lint_evidence_comment(
        pr="7740",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        body=body,
        author="an0mium",
        source="test",
    )
    assert result["would_count"] is True, result["problems"]
    assert result["counted_reviewer_ids"] == ["claude"]
    capped = qe._cap_text("x" * (qe._MAX_REVIEWER_CHARS + 5000))
    assert len(capped) <= qe._MAX_REVIEWER_CHARS + 64
    assert capped.endswith("[reviewer output truncated]")


# --- reviewer timeout configuration ----------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", 300.0),
        ("not-a-number", 300.0),
        ("0", 300.0),
        ("-5", 300.0),
        ("nan", 300.0),
        ("inf", 300.0),
        ("-inf", 300.0),
        ("12.5", 12.5),
    ],
)
def test_timeout_seconds_fails_closed_to_default(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: float,
) -> None:
    monkeypatch.setenv("ARAGORA_TEST_TIMEOUT_SECONDS", raw)
    assert qe._timeout_seconds("ARAGORA_TEST_TIMEOUT_SECONDS", 300) == expected


def test_run_claude_cli_uses_env_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(*args, timeout, **kwargs):
        seen["args"] = args
        seen["timeout"] = timeout
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=timeout)

    monkeypatch.setenv(qe._CLAUDE_TIMEOUT_ENV, "7")
    monkeypatch.setattr(qe.subprocess, "run", fake_run)

    result = qe._run_claude_cli("review prompt")

    assert seen["args"] == (["claude", "-p"],)
    assert seen["timeout"] == 7.0
    assert result == ReviewerResult(
        "claude",
        "",
        False,
        "claude CLI timed out after 7s",
    )


# --- default API reviewer cleanup ------------------------------------------


def test_run_api_agent_closes_agent_and_shared_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append(f"generate:{prompt}")
            return "Verdict: PASS"

        async def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        events.append(f"create:{family}:{name}:{role}")
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == [
        "create:grok:grok_reviewer:critic",
        "generate:review prompt",
        "agent_close",
        "connector_close",
    ]


def test_run_api_agent_closes_resources_after_generate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            raise RuntimeError("model failed")

        async def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent("grok", "review prompt")

    assert result.ok is False
    assert "RuntimeError: model failed" in result.error
    assert events == ["generate", "agent_close", "connector_close"]


def test_run_api_agent_closes_shared_connector_after_agent_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            return "Verdict: PASS"

        async def close(self) -> None:
            events.append("agent_close")
            raise RuntimeError("close failed")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == ["generate", "agent_close", "connector_close"]


def test_run_api_agent_closes_shared_connector_without_agent_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            return "Verdict: PASS"

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == ["generate", "connector_close"]


def test_run_api_agent_supports_sync_agent_close(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            return "Verdict: PASS"

        def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == ["generate", "agent_close", "connector_close"]


def test_run_api_agent_keeps_result_when_shared_connector_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            return "Verdict: PASS"

        async def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")
        raise RuntimeError("connector failed")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == ["generate", "agent_close", "connector_close"]


def test_run_api_agent_allows_consecutive_one_shot_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append(f"generate:{prompt}")
            return f"Verdict: PASS {prompt}"

        async def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        events.append(f"create:{family}")
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    first = qe._run_api_agent("grok", "one")
    second = qe._run_api_agent("grok", "two")

    assert first == ReviewerResult("grok", "Verdict: PASS one", True)
    assert second == ReviewerResult("grok", "Verdict: PASS two", True)
    assert events == [
        "create:grok",
        "generate:one",
        "agent_close",
        "connector_close",
        "create:grok",
        "generate:two",
        "agent_close",
        "connector_close",
    ]


# --- collect_evidence orchestration (fully offline via injected callables) ---


def _fakes(*, tier: int, head: str = HEAD, would_count: bool = True):
    posted: list[tuple[str, str]] = []

    def context_fetcher(repo: str, pr: int) -> dict:
        return {"head_sha": head, "head_committed_at": COMMITTED}

    def tier_fetcher(repo: str, pr: int):
        return tier

    def prompt_builder(repo: str, pr: int, ctx: dict) -> str:
        return "review prompt"

    def reviewer_runner(family: str, prompt: str) -> ReviewerResult:
        return ReviewerResult(family, f"Verdict: PASS from {family}", True)

    def linter(pr, head_sha, head_committed_at, author, body, env) -> dict:
        return {
            "would_count": would_count,
            "counted_reviewer_ids": [body.split()[1].lower()] if would_count else [],
            "problems": [] if would_count else ["no_counted_model_family"],
        }

    def poster(repo: str, pr: int, body: str) -> None:
        posted.append((repo, body))

    return dict(
        context_fetcher=context_fetcher,
        tier_fetcher=tier_fetcher,
        prompt_builder=prompt_builder,
        reviewer_runner=reviewer_runner,
        linter=linter,
        poster=poster,
    ), posted


def test_collect_low_tier_apply_posts_both() -> None:
    fakes, posted = _fakes(tier=1)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "post"
    assert sorted(outcome.posted) == ["claude", "grok"]
    assert len(posted) == 2


def test_collect_high_tier_apply_never_posts() -> None:
    fakes, posted = _fakes(tier=4)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert outcome.posted == []
    assert posted == []
    # Evidence is still composed + validated for the operator.
    assert sorted(outcome.counting_families) == ["claude", "grok"]


def test_collect_dry_run_prepares_without_posting() -> None:
    fakes, posted = _fakes(tier=1)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=False, **fakes
    )
    assert outcome.action == "prepare"
    assert posted == []
    assert sorted(outcome.counting_families) == ["claude", "grok"]


def test_collect_never_fabricates_on_reviewer_failure() -> None:
    fakes, posted = _fakes(tier=1)

    def failing_runner(family: str, prompt: str) -> ReviewerResult:
        if family == "grok":
            return ReviewerResult("grok", "", False, "timeout")
        return ReviewerResult(family, "Verdict: PASS from claude", True)

    fakes["reviewer_runner"] = failing_runner
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert [f.family for f in outcome.failures] == ["grok"]
    assert outcome.posted == ["claude"]
    assert len(posted) == 1


def test_collect_does_not_post_uncountable_evidence() -> None:
    fakes, posted = _fakes(tier=1, would_count=False)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "post"
    assert outcome.posted == []
    assert posted == []
    assert all(not item.would_count for item in outcome.items)


def test_collect_rejects_unsupported_family() -> None:
    fakes, posted = _fakes(tier=1)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "bogus"], author="me", apply=True, **fakes
    )
    assert "bogus" in [f.family for f in outcome.failures]
    assert "claude" in outcome.posted
    assert "bogus" not in outcome.counting_families


def test_collect_records_post_errors_without_losing_others() -> None:
    fakes, _ = _fakes(tier=1)

    def flaky_poster(repo: str, pr: int, body: str) -> None:
        if "Grok" in body:
            raise RuntimeError("gh rejected comment")

    fakes["poster"] = flaky_poster
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.posted == ["claude"]
    assert any("grok" in e for e in outcome.post_errors)


def test_collect_recheck_exception_prepares_without_posting() -> None:
    fakes, posted = _fakes(tier=1)
    calls = {"n": 0}

    def flaky_context(repo: str, pr: int) -> dict:
        calls["n"] += 1
        if calls["n"] >= 2:  # first call ok, recheck blows up
            raise RuntimeError("transient gh error")
        return {"head_sha": HEAD, "head_committed_at": COMMITTED}

    fakes["context_fetcher"] = flaky_context
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert "re-verify" in outcome.action_reason
    assert posted == []


def test_collect_skips_post_when_head_moves_before_posting() -> None:
    fakes, posted = _fakes(tier=1)
    heads = iter([HEAD, "0" * 40])  # initial fetch, then recheck = moved head

    def moving_context(repo: str, pr: int) -> dict:
        return {"head_sha": next(heads), "head_committed_at": COMMITTED}

    fakes["context_fetcher"] = moving_context
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert "changed before posting" in outcome.action_reason
    assert posted == []


def test_collect_skips_post_when_tier_promoted_before_posting() -> None:
    fakes, posted = _fakes(tier=1)
    tiers = iter([1, 4])  # initial low, recheck promoted to settlement tier

    def promoting_tier(repo: str, pr: int) -> int:
        return next(tiers)

    fakes["tier_fetcher"] = promoting_tier
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert posted == []


def test_collect_dedupes_families() -> None:
    fakes, _ = _fakes(tier=4)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "Claude", "grok"], author="me", apply=False, **fakes
    )
    assert [item.family for item in outcome.items] == ["claude", "grok"]


def test_collect_missing_head_raises() -> None:
    fakes, _ = _fakes(tier=1, head="")
    with pytest.raises(ValueError):
        collect_evidence(repo="o/r", pr=1, families=["claude"], author="me", apply=True, **fakes)


# --- run_collect_cli (monkeypatched orchestrator) ---------------------------


def test_run_collect_cli_exit_code_quorum_met(monkeypatch, capsys) -> None:
    def fake_collect(**kwargs) -> CollectOutcome:
        return CollectOutcome(
            repo="o/r",
            pr=1,
            head_sha=HEAD,
            head_committed_at=COMMITTED,
            tier=1,
            action="post",
            action_reason="ok",
            items=[
                EvidenceItem("claude", "body", True, ["claude"], []),
                EvidenceItem("grok", "body", True, ["grok"], []),
            ],
            posted=["claude", "grok"],
        )

    monkeypatch.setattr(qe, "collect_evidence", fake_collect)
    monkeypatch.setattr(qe, "resolve_author", lambda default="local": "me")
    rc = qe.run_collect_cli(
        repo="o/r", pr=1, families=None, author=None, apply=True, json_output=True
    )
    assert rc == 0
    assert "collect_evidence" in capsys.readouterr().out


def test_run_collect_cli_exit_code_quorum_incomplete(monkeypatch) -> None:
    def fake_collect(**kwargs) -> CollectOutcome:
        return CollectOutcome(
            repo="o/r",
            pr=1,
            head_sha=HEAD,
            head_committed_at=COMMITTED,
            tier=4,
            action="prepare",
            action_reason="settlement",
            items=[EvidenceItem("claude", "body", True, ["claude"], [])],
        )

    monkeypatch.setattr(qe, "collect_evidence", fake_collect)
    monkeypatch.setattr(qe, "resolve_author", lambda default="local": "me")
    rc = qe.run_collect_cli(
        repo="o/r", pr=1, families=None, author=None, apply=False, json_output=False
    )
    assert rc == 1


def test_run_collect_cli_error_path(monkeypatch, capsys) -> None:
    def boom(**kwargs):
        raise ValueError("no head")

    monkeypatch.setattr(qe, "collect_evidence", boom)
    monkeypatch.setattr(qe, "resolve_author", lambda default="local": "me")
    rc = qe.run_collect_cli(
        repo="o/r", pr=1, families=None, author=None, apply=False, json_output=True
    )
    assert rc == 1
    assert "no head" in capsys.readouterr().out


def test_run_collect_cli_catches_runtime_error(monkeypatch, capsys) -> None:
    def boom(**kwargs):
        raise RuntimeError("empty diff")

    monkeypatch.setattr(qe, "collect_evidence", boom)
    monkeypatch.setattr(qe, "resolve_author", lambda default="local": "me")
    rc = qe.run_collect_cli(
        repo="o/r", pr=1, families=None, author=None, apply=False, json_output=False
    )
    assert rc == 1
    assert "empty diff" in capsys.readouterr().out
