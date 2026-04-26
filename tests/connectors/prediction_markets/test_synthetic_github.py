"""Tests for the AGT-04 SyntheticGitHubAdapter connector (issue #6065)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.connectors.prediction_markets.synthetic_github import (
    DEFAULT_POSITION_CAP,
    SYNTHETIC_MARKETS_FLAG,
    SyntheticGitHubAdapter,
    SyntheticGitHubError,
    open_adapter,
    synthetic_markets_enabled,
)
from aragora.markets.store import MarketStore
from aragora.markets.types import Market


@pytest.fixture()
def store(tmp_path: Path) -> MarketStore:
    return MarketStore(tmp_path / "markets")


@pytest.fixture()
def adapter(store: MarketStore, monkeypatch: pytest.MonkeyPatch) -> SyntheticGitHubAdapter:
    monkeypatch.setenv(SYNTHETIC_MARKETS_FLAG, "1")
    return SyntheticGitHubAdapter(store=store, require_expiry=False)


def _stub(responses: dict[tuple, tuple[int, str]]):
    def runner(args, **_kw):
        key = tuple(str(a) for a in args)
        for pattern, (rc, stdout) in responses.items():
            if all(p in key for p in pattern):
                return subprocess.CompletedProcess(
                    args=list(args), returncode=rc, stdout=stdout, stderr=""
                )
        return subprocess.CompletedProcess(
            args=list(args), returncode=1, stdout="", stderr="no stub"
        )

    return runner


def _expired(store: MarketStore, kind: str, target: dict) -> Market:
    m = Market.create(
        question_kind=kind,
        target=target,
        description="t",
        resolution_window_days=1,
        created_at=datetime.now(tz=UTC) - timedelta(days=3),
    )
    return store.add_market(m)


# --- Flag guard ---


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SYNTHETIC_MARKETS_FLAG, raising=False)
    assert synthetic_markets_enabled() is False


@pytest.mark.parametrize("v", ["1", "true", "yes", "on"])
def test_enabled_truthy(monkeypatch: pytest.MonkeyPatch, v: str) -> None:
    monkeypatch.setenv(SYNTHETIC_MARKETS_FLAG, v)
    assert synthetic_markets_enabled() is True


def test_create_blocked_when_disabled(store: MarketStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SYNTHETIC_MARKETS_FLAG, raising=False)
    with pytest.raises(SyntheticGitHubError, match=SYNTHETIC_MARKETS_FLAG):
        SyntheticGitHubAdapter(store=store).create_pr_merge_market(repo="owner/repo", pr_number=1)


def test_place_blocked_when_disabled(store: MarketStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SYNTHETIC_MARKETS_FLAG, raising=False)
    with pytest.raises(SyntheticGitHubError, match=SYNTHETIC_MARKETS_FLAG):
        SyntheticGitHubAdapter(store=store).place_position(
            "x", agent_id="a", probability=0.5, stake=10
        )


# --- Market creation ---


def test_pr_merge_market(adapter: SyntheticGitHubAdapter) -> None:
    m = adapter.create_pr_merge_market(repo="owner/repo", pr_number=42)
    assert m.question_kind == "pr_merge"
    assert m.target == {"repo": "owner/repo", "number": 42}
    assert m.market_id.startswith("mkt_pr_merge_")


def test_issue_close_market(adapter: SyntheticGitHubAdapter) -> None:
    m = adapter.create_issue_close_market(repo="owner/repo", issue_number=99)
    assert m.question_kind == "issue_close"


def test_ci_pass_market(adapter: SyntheticGitHubAdapter) -> None:
    m = adapter.create_ci_pass_market(repo="owner/repo", ref="abc123")
    assert m.target["ref"] == "abc123"


def test_content_addressed_id_stable(adapter: SyntheticGitHubAdapter) -> None:
    """market_id is content-addressed from kind+target, so same inputs produce same id."""
    from datetime import UTC, datetime

    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    m1 = adapter.create_pr_merge_market(repo="owner/repo", pr_number=5, created_at=fixed)
    m2 = adapter.create_pr_merge_market(repo="owner/repo", pr_number=5, created_at=fixed)
    assert m1.market_id == m2.market_id


def test_invalid_repo_raises(adapter: SyntheticGitHubAdapter) -> None:
    with pytest.raises((ValueError, SyntheticGitHubError)):
        adapter.create_pr_merge_market(repo="not-valid", pr_number=1)


# --- Positions ---


def test_place_position_success(adapter: SyntheticGitHubAdapter) -> None:
    m = adapter.create_pr_merge_market(repo="owner/repo", pr_number=10)
    pos = adapter.place_position(m.market_id, agent_id="agent-007", probability=0.75, stake=50)
    assert pos.probability == 0.75 and pos.agent_id == "agent-007"


def test_stake_cap_enforced(adapter: SyntheticGitHubAdapter) -> None:
    m = adapter.create_pr_merge_market(repo="owner/repo", pr_number=11)
    with pytest.raises(SyntheticGitHubError):
        adapter.place_position(
            m.market_id, agent_id="a", probability=0.5, stake=DEFAULT_POSITION_CAP + 1
        )


def test_invalid_probability_raises(adapter: SyntheticGitHubAdapter) -> None:
    m = adapter.create_pr_merge_market(repo="owner/repo", pr_number=12)
    with pytest.raises(SyntheticGitHubError):
        adapter.place_position(m.market_id, agent_id="a", probability=1.5, stake=10)


# --- Resolution ---


def test_resolve_merged_pr(store: MarketStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SYNTHETIC_MARKETS_FLAG, "1")
    stub = _stub(
        {
            ("pr", "view"): (
                0,
                json.dumps(
                    {
                        "state": "MERGED",
                        "merged": True,
                        "mergedAt": "2026-04-25T10:00:00Z",
                        "closedAt": None,
                    }
                ),
            )
        }
    )
    a = SyntheticGitHubAdapter(store=store, gh_runner=stub, require_expiry=False)
    event = a.resolve_market(_expired(store, "pr_merge", {"repo": "owner/repo", "number": 42}))
    assert event.outcome == "yes"


def test_batch_skips_non_expired(store: MarketStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SYNTHETIC_MARKETS_FLAG, "1")
    store.add_market(
        Market.create(
            question_kind="issue_close",
            target={"repo": "owner/repo", "number": 1},
            description="f",
            resolution_window_days=30,
        )
    )
    assert SyntheticGitHubAdapter(store=store, require_expiry=True).resolve_expired_batch() == []


def test_batch_skips_transient_failure(store: MarketStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SYNTHETIC_MARKETS_FLAG, "1")
    a = SyntheticGitHubAdapter(store=store, gh_runner=_stub({}), require_expiry=False)
    _expired(store, "pr_merge", {"repo": "owner/repo", "number": 99})
    assert a.resolve_expired_batch() == []  # transient failure silently skipped


# --- open_adapter ---


def test_open_adapter(tmp_path: Path) -> None:
    a = open_adapter(tmp_path / "store")
    assert isinstance(a, SyntheticGitHubAdapter)
